from copy import deepcopy
import torch
from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.data import iterators, encoders
from fairseq.trainer import Trainer

extracted_grads = []
def extract_grad_hook(module, grad_in, grad_out):
    extracted_grads.append(grad_out[0])

# returns the wordpiece embedding weight matrix
def get_embedding_weight(model, bpe_vocab_size):
    for module in model.modules():
        if isinstance(module, torch.nn.Embedding):
            if module.weight.shape[0] == bpe_vocab_size:
                return module.weight.detach().cpu()
    exit("Embedding matrix not found")

# add hooks for embeddings, only add a hook to encoder wordpiece embeddings (not position)
def add_hooks(model, bpe_vocab_size):
    hook_registered = False
    for module in model.modules():
        if isinstance(module, torch.nn.Embedding):
            if module.weight.shape[0] == bpe_vocab_size:
                module.weight.requires_grad = True
                module.register_backward_hook(extract_grad_hook)
                hook_registered = True
    if not hook_registered:
        exit("Embedding matrix not found")


def get_user_input(trainer, bpe):
    user_input = input('Enter the input sentence that you to attack: ')
    if user_input.strip() == '':
        print("You entered a blank token, try again")
        return None

    # tokenize input and get lengths
    tokenized_bpe_input = trainer.task.source_dictionary.encode_line(bpe.encode(user_input)).long().unsqueeze(dim=0)

    # check if the user input a token with is an UNK
    bpe_vocab_size = trainer.get_model().encoder.embed_tokens.weight.shape[0]
    for token in tokenized_bpe_input[0]:
        if torch.eq(token, bpe_vocab_size) or torch.gt(token, bpe_vocab_size): # >= max vocab size
            print('You entered an UNK token for your model, please try again. This usually occurs when (1) you entered '
                ' unicode or other strange symbols, (2) your model uses a lowercased dataset but you entered uppercase, or '
                ' (3) your model is expecting apostrophies as &apos; and quotes as &quot;')
            return None

    length_user_input = torch.LongTensor([len(tokenized_bpe_input[0])])
    # build samples which is input to the model
    samples = {'net_input': {'src_tokens': tokenized_bpe_input, 'src_lengths': length_user_input}, 'ntokens': len(tokenized_bpe_input[0])}

    return samples


# runs the samples through the model and fills extracted_grads with the gradient w.r.t. the embedding
def get_loss_and_input_grad(trainer, samples, target_mask=None, no_backwards=False, reduce_loss=True):
    trainer._set_seed()
    trainer.get_model().eval() # we want grads from eval() to turn off dropout and stuff    
    trainer.zero_grad()

    # fills extracted_grads with the gradient w.r.t. the embedding
    sample = trainer._prepare_sample(samples)
    loss, _, _, = trainer.criterion(trainer.get_model(), sample, target_mask=target_mask, reduce=reduce_loss)    
    if not no_backwards:
        trainer.optimizer.backward(loss)
    return sample['net_input']['src_lengths'], loss.detach().cpu()


# take samples (which is batch size 1) and repeat it batch_size times to do batched inference / loss calculation
# for all of the possible attack candidates
def build_inference_samples(samples, batch_size, args, candidate_input_tokens, trainer, bpe, changed_positions=None, untouchable_token_blacklist=None, adversarial_token_blacklist=None, num_trigger_tokens=None):
    # copy and repeat the samples instead batch size elements
    samples_repeated_by_batch = deepcopy(samples)
    samples_repeated_by_batch['ntokens'] *= batch_size
    samples_repeated_by_batch['target'] = samples_repeated_by_batch['target'].repeat(batch_size, 1)
    samples_repeated_by_batch['net_input']['prev_output_tokens'] = samples_repeated_by_batch['net_input']['prev_output_tokens'].repeat(batch_size, 1)
    samples_repeated_by_batch['net_input']['src_tokens'] = samples_repeated_by_batch['net_input']['src_tokens'].repeat(batch_size, 1)
    samples_repeated_by_batch['net_input']['src_lengths'] = samples_repeated_by_batch['net_input']['src_lengths'].repeat(batch_size, 1)
    samples_repeated_by_batch['nsentences'] = batch_size

    all_inference_samples = [] # stores a list of batches of candidates
    all_changed_positions = [] # stores all the changed_positions for each batch element

    current_batch_size = 0
    current_batch_changed_position = []
    current_inference_samples = deepcopy(samples_repeated_by_batch) # stores one batch worth of candidates
    for index in range(len(candidate_input_tokens)): # for all the positions in the input
        for token_id in candidate_input_tokens[index]: # for all the candidates
            # for malicious nonsense
            if changed_positions is not None:
                # if we have already changed this position, skip
                if changed_positions[index]: 
                    continue
            # for universal triggers            
            if num_trigger_tokens is not None: 
                # want to change the last tokens, not the first, for triggers
                index_to_use = index - num_trigger_tokens - 1 # -1 to skip <eos>
            else:
                index_to_use = index

            # for targeted flips
            # don't touch the word if its in the blacklist
            if untouchable_token_blacklist is not None and current_inference_samples['net_input']['src_tokens'][current_batch_size][index_to_use] in untouchable_token_blacklist:
                continue
            # don't insert any blacklisted tokens into the source side
            if adversarial_token_blacklist is not None and any([token_id == blacklisted_token for blacklisted_token in adversarial_token_blacklist]): 
                continue

            original_token = deepcopy(current_inference_samples['net_input']['src_tokens'][current_batch_size][index_to_use]) # save the original token, might be used below if there is an error
            current_inference_samples['net_input']['src_tokens'][current_batch_size][index_to_use] = torch.LongTensor([token_id]).squeeze(0) # change one token

            # there are cases where making a BPE swap would cause the BPE segmentation to change.
            # in other words, the input we are using would be invalid because we are using an old segmentation
            # for these cases, we just skip those candidates            
            string_input_tokens = bpe.decode(trainer.task.source_dictionary.string(current_inference_samples['net_input']['src_tokens'][current_batch_size], None))
            retokenized_string_input_tokens = trainer.task.source_dictionary.encode_line(bpe.encode(string_input_tokens)).long().unsqueeze(dim=0)
            if torch.cuda.is_available() and not trainer.args.cpu:
                retokenized_string_input_tokens = retokenized_string_input_tokens.cuda()
            if len(retokenized_string_input_tokens[0]) != len(current_inference_samples['net_input']['src_tokens'][current_batch_size]) or \
                not torch.all(torch.eq(retokenized_string_input_tokens[0],current_inference_samples['net_input']['src_tokens'][current_batch_size])):
                # undo the token we replaced and move to the next candidate
                current_inference_samples['net_input']['src_tokens'][current_batch_size][index_to_use] = original_token
                continue
                                    
            current_batch_size += 1
            current_batch_changed_position.append(index_to_use) # save its changed position

            if current_batch_size == batch_size: # batch is full
                all_inference_samples.append(deepcopy(current_inference_samples))
                current_inference_samples = deepcopy(samples_repeated_by_batch)
                current_batch_size = 0
                all_changed_positions.append(current_batch_changed_position)
                current_batch_changed_position = []

    return all_inference_samples, all_changed_positions

def get_attack_candidates(trainer, samples, embedding_weight, num_gradient_candidates=500, target_mask=None, increase_loss=False):
    # clear grads, compute new grads, and get candidate tokens
    global extracted_grads; extracted_grads = [] # clear old extracted_grads
    src_lengths, _ = get_loss_and_input_grad(trainer, samples, target_mask=target_mask) # gradient is now filled
    
    # for models with shared embeddings, position 1 in extracted_grads will be the encoder grads, 0 is decoder
    if len(extracted_grads) > 1:
        gradient_position = 1
    else:
        gradient_position = 0
    assert len(extracted_grads) <= 2 and len(extracted_grads[gradient_position]) == 1 # make sure gradients are not accumulating
    # first [] gets decoder/encoder grads, then gets ride of batch (we have batch size 1)
    # then we index into before the padding (though there shouldn't be any padding because we do batch size 1).
    # then the -1 is to ignore the pad symbol.
    input_gradient = extracted_grads[gradient_position][0][0:src_lengths[0]-1].cpu() 
    input_gradient = input_gradient.unsqueeze(0)

    gradient_dot_embedding_matrix = torch.einsum("bij,kj->bik", (input_gradient, embedding_weight))
    if not increase_loss:
        gradient_dot_embedding_matrix *= -1    # lower versus increase the class probability.
    if num_gradient_candidates > 1: # get top k options
        _, best_k_ids = torch.topk(gradient_dot_embedding_matrix, num_gradient_candidates, dim=2)
        return best_k_ids.detach().cpu().numpy()[0]
    else:
        _, best_at_each_step = gradient_dot_embedding_matrix.max(2)
        return best_at_each_step[0].detach().cpu().numpy()

def setup():
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser)
    # make sure everything is reset before loading the model
    args.reset_optimizer = True
    args.reset_meters = True
    args.reset_dataloader = True
    args.reset_lr_scheduler = True
    args.path = args.restore_file
    args.max_sentences_valid = 1  # We attack batch size 1 at the moment
    args.beam = 1 # beam size 1 for inference on the model, could use higher
    utils.import_user_module(args)
    
    torch.manual_seed(args.seed)

    # setup task, model, loss function, and trainer
    task = tasks.setup_task(args)
    if not args.interactive_attacks:
        for valid_sub_split in args.valid_subset.split(','): # load validation data
            task.load_dataset(valid_sub_split, combine=False, epoch=0)
    models, _= checkpoint_utils.load_model_ensemble(args.path.split(':'), arg_overrides={}, task=task)
    assert len(models) == 1 # Make sure you didn't pass an ensemble of models in
    model = models[0]

    if torch.cuda.is_available() and not args.cpu:
        assert torch.cuda.device_count() == 1 # only works on 1 GPU for now
        torch.cuda.set_device(0)
        model.cuda()
    args.beam = 1 # beam size 1 for now
    model.make_generation_fast_(beamable_mm_beam_size=args.beam, need_attn=False)

    criterion = task.build_criterion(args)
    trainer = Trainer(args, task, model, criterion)
    generator = task.build_generator(args)

    bpe_vocab_size = trainer.get_model().encoder.embed_tokens.weight.shape[0]
    add_hooks(trainer.get_model(), bpe_vocab_size) # add gradient hooks to embeddings
    embedding_weight = get_embedding_weight(trainer.get_model(), bpe_vocab_size) # save the embedding matrix
    if not args.interactive_attacks:
        subset = args.valid_subset.split(',')[0] # only one validation subset handled
        itr = trainer.task.get_batch_iterator(dataset=trainer.task.dataset(subset),
                                      max_tokens=args.max_tokens_valid,
                                      max_sentences=args.max_sentences_valid,
                                      max_positions=utils.resolve_max_positions(
                                      trainer.task.max_positions(),
                                      trainer.get_model().max_positions(),),
                                      ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
                                      required_batch_size_multiple=args.required_batch_size_multiple,
                                      seed=args.seed,
                                      num_shards=args.distributed_world_size,
                                      shard_id=args.distributed_rank,
                                      num_workers=args.num_workers,).next_epoch_itr(shuffle=False)
    else:
        itr = [None] * 100000  # a fake dataset to go through, overwritten when doing interactive attacks

    # Handle BPE
    bpe = encoders.build_bpe(args)
    assert bpe is not None
    return args, trainer, generator, embedding_weight, itr, bpe
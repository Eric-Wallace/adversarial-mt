import numpy as np
from copy import deepcopy
import torch
from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.data import iterators, encoders
from fairseq.trainer import Trainer
import all_attack_utils

def main():
    args, trainer, generator, embedding_weight, itr, bpe = all_attack_utils.setup()
    for i, samples in enumerate(itr): # for the whole validation set (could be fake data if its interactive model)
        universal_attack(samples, args, trainer, generator, embedding_weight, itr, bpe)

def universal_attack(samples, args, trainer, generator, embedding_weight, itr, bpe):
    if args.interactive_attacks: # get user input and build samples
        samples = all_attack_utils.get_user_input(trainer, bpe)
        while samples is None:
            samples = all_attack_utils.get_user_input(trainer, bpe)
    
    if torch.cuda.is_available() and not trainer.args.cpu:
        samples['net_input']['src_tokens'] = samples['net_input']['src_tokens'].cuda()
        samples['net_input']['src_lengths'] = samples['net_input']['src_lengths'].cuda()

    translations = trainer.task.inference_step(generator, [trainer.get_model()], samples)
    original_prediction = translations[0][0]['tokens']    
    samples['target'] = original_prediction.unsqueeze(0)    
    samples['net_input']['prev_output_tokens'] = torch.cat((samples['target'][0][-1:],
                                                            samples['target'][0][:-1]), dim=0).unsqueeze(dim=0)
    if torch.cuda.is_available() and not args.cpu:
        samples['target'] = samples['target'].cuda()
        samples['net_input']['prev_output_tokens'] = samples['net_input']['prev_output_tokens'].cuda()
    
    print('Original Prediction ', bpe.decode(trainer.task.target_dictionary.string(original_prediction, None)))

    # add random trigger to the user input
    original_samples = deepcopy(samples)
    samples = deepcopy(original_samples)

    num_trigger_tokens = 5
    # if punctuation is already present at the end, we want to replace it with a comma. Else, add a comma at the end
    if samples['net_input']['src_tokens'][0][-2] in [5, 129,  88,   4,  89,  43]: # if the token is . ; ! , ? :
        num_tokens_to_add = num_trigger_tokens + 1 # the + 1 is to we can replace the last token with <eos>
    else:
        num_tokens_to_add = num_trigger_tokens + 2 # the extra + 1 is to we can add the comma
    trigger_concatenated_source_tokens = torch.cat((samples['net_input']['src_tokens'][0][0:-1], torch.randint(5, 9, (1, num_tokens_to_add)).cuda().squeeze(0)),dim=0) # add random tokens to initialize trigger
    trigger_concatenated_source_tokens[-num_trigger_tokens - 2] = torch.LongTensor([4]).squeeze(0).cuda() # add a ,
    trigger_concatenated_source_tokens[-1] = torch.LongTensor([2]).squeeze(0).cuda() # add <eos>
    samples['net_input']['src_tokens'] = trigger_concatenated_source_tokens.unsqueeze(0)
    samples['net_input']['src_lengths'] += num_tokens_to_add - 1 # not counting <eos>

    if samples['target'][0][-2] in [5, 129,  88,   4,  89,  43]: # if there is punctuation in the target then replace with ,
        samples['target'][0][-2] = torch.LongTensor([4]).squeeze(0).cuda()
        samples['net_input']['prev_output_tokens'][0][-1] = torch.LongTensor([4]).squeeze(0).cuda()

    original_target = deepcopy(samples['target'])
    best_found_loss = 999999999999999
    if not args.suffix_dropper:
        best_found_loss *= -1 # we want small losses for this
    while True: # gradient attack will early stop        
        assert samples['net_input']['src_tokens'].cpu().numpy()[0][-1] == 2 # make sure pad is always there
        translations = trainer.task.inference_step(generator, [trainer.get_model()], samples)
        predictions = translations[0][0]['tokens']            
        assert all(torch.eq(samples['target'].squeeze(0), original_target.squeeze(0))) # make sure target is never updated

        if args.suffix_dropper:
            increase_loss = False
        else:
            increase_loss = True
        candidate_input_tokens = all_attack_utils.get_attack_candidates(trainer, samples, embedding_weight, num_gradient_candidates=500, target_mask=None, increase_loss=increase_loss)
        candidate_input_tokens = candidate_input_tokens[-num_trigger_tokens:] # the trigger candidates are at the end
        batch_size = 64
        all_inference_samples, _ = all_attack_utils.build_inference_samples(samples, batch_size, args, candidate_input_tokens, trainer, bpe, num_trigger_tokens=num_trigger_tokens)

        current_best_found_loss = 9999999
        if not args.suffix_dropper:
            current_best_found_loss *= -1 # we want small losses for this
        current_best_found_tokens = None
        for inference_indx, inference_sample in enumerate(all_inference_samples):
            _, losses = all_attack_utils.get_loss_and_input_grad(trainer, inference_sample, target_mask=None, no_backwards=True, reduce_loss=False)            
            losses = losses.reshape(batch_size, samples['target'].shape[1]) # unflatten losses            
            losses = torch.sum(losses, dim=1)            

            for loss_indx, loss in enumerate(losses):
                if args.suffix_dropper:
                    if loss < current_best_found_loss:
                        current_best_found_loss = loss
                        current_best_found_tokens = inference_sample['net_input']['src_tokens'][loss_indx].unsqueeze(0)
                else:
                    if loss > current_best_found_loss:
                        current_best_found_loss = loss
                        current_best_found_tokens = inference_sample['net_input']['src_tokens'][loss_indx].unsqueeze(0)

        if args.suffix_dropper:
            if current_best_found_loss < best_found_loss: # update best tokens
                best_found_loss = current_best_found_loss
                samples['net_input']['src_tokens'] = current_best_found_tokens
            # gradient is deterministic, so if it didnt flip another then its never going to
            else:                
                break
        else:
            if current_best_found_loss > best_found_loss: # update best tokens
                best_found_loss = current_best_found_loss
                samples['net_input']['src_tokens'] = current_best_found_tokens
                # gradient is deterministic, so if it didnt flip another then its never going to
            else:            
                break
    print('Final Input ', bpe.decode(trainer.task.source_dictionary.string(samples['net_input']['src_tokens'].cpu()[0], None)))
    translations = trainer.task.inference_step(generator, [trainer.get_model()], samples)
    final_prediction = translations[0][0]['tokens']
    print('Final Prediction ', bpe.decode(trainer.task.target_dictionary.string(final_prediction, None)))
    print()

if __name__ == '__main__':
    main()
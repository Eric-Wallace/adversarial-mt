from copy import deepcopy
import torch
from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.data import iterators, encoders
from fairseq.trainer import Trainer
import all_attack_utils

def malicious_nonsense(samples, args, trainer, generator, embedding_weight, itr, bpe):
    if args.interactive_attacks: # get user input and build samples
        samples = all_attack_utils.get_user_input(trainer, bpe)
        while samples is None: # if the user entered an UNK, ask for another sample
            samples = all_attack_utils.get_user_input(trainer, bpe)

    if torch.cuda.is_available() and not trainer.args.cpu:
        samples['net_input']['src_tokens'] = samples['net_input']['src_tokens'].cuda()
        samples['net_input']['src_lengths'] = samples['net_input']['src_lengths'].cuda()

    original_samples = deepcopy(samples)
    translations = trainer.task.inference_step(generator, [trainer.get_model()], samples)
    original_prediction = translations[0][0]['tokens']
    # at this point, samples only contains the input to the model, but no target
    # We want to add the model's prediction as the target in samples
    # so that we can compute the cross-entropy loss w.r.t to the model's prediction
    samples['target'] = original_prediction.unsqueeze(0) # overwrite samples
    # prev_output_tokens is the right rotated version of the target (i.e., teacher forcing)
    samples['net_input']['prev_output_tokens'] = torch.cat((samples['target'][0][-1:],
                                                            samples['target'][0][:-1]), dim=0).unsqueeze(dim=0)
    if torch.cuda.is_available() and not args.cpu:
        samples['target'] = samples['target'].cuda()
        samples['net_input']['prev_output_tokens'] = samples['net_input']['prev_output_tokens'].cuda()

    # begin the attack
    # we want to change all positions at least once, so if a position is already changed, don't change it again.
    # [False] for the sequence length, but minus -1 to ignore </s>
    changed_positions = [False] * (samples['net_input']['src_tokens'].shape[1] - 1)
    num_gradient_candidates = 500 # increase the number to make the attack better, but also slower to run
    while True: # we break at the end when we can't find another good token
        assert samples['net_input']['src_tokens'].cpu().numpy()[0][-1] == 2 # make sure </s> it always there
        print('Current Tokens', bpe.decode(trainer.task.source_dictionary.string(samples['net_input']['src_tokens'].cpu()[0], None)))
        translations = trainer.task.inference_step(generator, [trainer.get_model()], samples)
        print('Current Prediction', bpe.decode(trainer.task.target_dictionary.string(torch.LongTensor(translations[0][0]['tokens'].cpu()), None)))

        # get candidate attack tokens
        candidate_input_tokens = all_attack_utils.get_attack_candidates(trainer, samples, embedding_weight, num_gradient_candidates=num_gradient_candidates)

        # batch up all the candidate input tokens to make evaluation faster
        new_found_input_tokens = None
        batch_size = 64
        all_inference_batches, all_changed_positions = all_attack_utils.build_inference_samples(samples, batch_size, args, candidate_input_tokens, trainer, bpe, changed_positions=changed_positions)

        # for all the possible new samples
        for inference_indx, inference_batch in enumerate(all_inference_batches):
            # run the model on a batch of samples
            predictions = trainer.task.inference_step(generator, [trainer.get_model()], inference_batch)
            for prediction_indx, prediction in enumerate(predictions): # for all predictions in that batch
                prediction = prediction[0]['tokens']
                # if prediction is the same, then save input. These are the possible malicious nonsense inputs
                if prediction.shape == original_prediction.shape and all(torch.eq(prediction, original_prediction)):
                    # if the "new" candidate is actually the same as the current tokens, skip it
                    if all(torch.eq(inference_batch['net_input']['src_tokens'][prediction_indx],samples['net_input']['src_tokens'].squeeze(0))):
                        continue
                    new_found_input_tokens = deepcopy(inference_batch['net_input']['src_tokens'][prediction_indx].unsqueeze(0))
                    changed_positions[all_changed_positions[inference_indx][prediction_indx]] = True
                    break # break twice
            if new_found_input_tokens is not None:
                break

        # Update current input if the new candidate flipped a position
        if new_found_input_tokens is not None:
            samples['net_input']['src_tokens'] = new_found_input_tokens

        # gradient is deterministic, so if it didnt flip another then its never going to
        else:
            break

    print("Can not flip any more\n")
    return changed_positions


def main():
    args, trainer, generator, embedding_weight, itr, bpe = all_attack_utils.setup()
    num_samples_changed = 0.0; num_total_samples = 0.0; num_tokens_changed = 0.0; total_num_tokens = 0.0
    for i, samples in enumerate(itr): # for the whole validation set (could be fake data if its interactive model)
        changed_positions = malicious_nonsense(samples, args, trainer, generator, embedding_weight, itr, bpe)

        num_total_samples += 1.0
        if any(changed_positions): # count up the number of positions that changed
            num_samples_changed += 1.0
            num_tokens_changed += sum(changed_positions)
            total_num_tokens += len(changed_positions)

    if num_total_samples > 0.0:
         print('Total Num Samples', num_total_samples)
         print('Percent Samples Changed', num_samples_changed / num_total_samples)
         print('Percent Tokens Changed', num_tokens_changed / total_num_tokens)

if __name__ == '__main__':
    main()
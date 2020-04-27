# Imitating and Attacking Production Machine Translation Systems

This is the official code for "Imitating and Attacking Production Machine Translation Systems". This repository contains the code for replicating our adversarial attack experiments on your own MT models.

Read our [blog](http://www.ericswallace.com/imitating) for more information on the method.

## Dependencies

This code is written using [Fairseq](https://github.com/facebookresearch/fairseq) and PyTorch. The code is based on an older version of Fairseq, from [this commit](https://github.com/pytorch/fairseq/tree/99fbd317f6b3256a39868d6568e70672f0f512b9). The code is made to run on one GPU or CPU. I used one GTX 1080 for all the experiments. Most experiments run in a few minutes.

## Installation

An easy way to install the code is to create a fresh anaconda environment:

```
conda create -n attacking python=3.6
source activate attacking
pip install -e . # install local version of fairseq
pip install -r requirements.txt
```
Now you should be ready to go!


## Code Structure 

The repository is broken down by attack type:
+ `malicious_nonsense.py` contains the malicious nonsense attack.
+ `targeted_flips.py` contains the targeted flips attack.
+ `universal.py` contains the two universal attacks (untargeted and suffix dropper).

The file `attack_utils.py` contains additional code for evaluating models, the first-order taylor expansion, computing embedding gradients, and evaluating the top candidates for the attack. Overall, the code in this repository is a stripped down and cleaned up version of the code used in the paper. The code is designed to be easy to understand and quick to get started with.


## Getting Started

First, you need to get a machine translation model. Fortunately, `fairseq` already has a number of pretrained models available. See [this repository](https://github.com/pytorch/fairseq/tree/master/examples/translation) for a complete list. Here we will download a transformer-based English-German model that is trained on the WMT16 dataset.

```bash
wget https://dl.fbaipublicfiles.com/fairseq/models/wmt16.en-de.joined-dict.transformer.tar.bz2
wget https://dl.fbaipublicfiles.com/fairseq/data/wmt16.en-de.joined-dict.newstest2014.tar.bz2

bunzip2 wmt16.en-de.joined-dict.transformer.tar.bz2
bunzip2 wmt16.en-de.joined-dict.newstest2014.tar.bz2
tar -xvf wmt16.en-de.joined-dict.transformer.tar
tar -xvf wmt16.en-de.joined-dict.newstest2014.tar
```

### Malicious Nonsense

Now we can run an interactive version of the malicious nonsense attack. 
```bash
export CUDA_VISIBLE_DEVICES=0
python malicious_nonsense.py wmt16.en-de.joined-dict.newstest2014/ --arch transformer_vaswani_wmt_en_de_big --restore-file wmt16.en-de.joined-dict.transformer/model.pt  --bpe subword_nmt --bpe-codes wmt16.en-de.joined-dict.transformer/bpecodes --interactive-attacks --source-lang en --target-lang de
```
The arguments we passed in are: the dataset we downloaded, the model architecture type (we downloaded a Transformer Big architecture), the model checkpoint path, the path to the BPE dictionary, and a flag to enable interactive attacks, respectively. The `--source-lang` and `--target-lang` flags are usually ok to omit because `fairseq` can automatically infer the language pair. If you want to run the attack on the WMT16 test set rather than interactively, you can omit the `--interactive-attacks` flag and pass in `--valid-subset test`. If you do not have a GPU, omit the `export CUDA_VISIBLE_DEVICES=0` command and also pass in the `--cpu` argument in the command.

Now you can enter a sentence that you want to turn into malicious nonsense. Let's try something benign like `I am a student at the University down the hill`. You can also try something more malicious like `Barack Obama was shot by a rebel group` or whatever your desired adversarial malicious input/output from the model is.

### Targeted Flips

The other attacks follow the same arguments as malicious nonsense.

```bash
python targeted_flips.py wmt16.en-de.joined-dict.newstest2014/ --arch transformer_vaswani_wmt_en_de_big --restore-file wmt16.en-de.joined-dict.transformer/model.pt  --bpe subword_nmt --bpe-codes wmt16.en-de.joined-dict.transformer/bpecodes --interactive-attacks --source-lang en --target-lang de
```

For targeted flips we currently assume that `--interactive-attacks` is set. 

First, enter the sentence that you want to attack, e.g., `I am sad` which translates to `Ich bin traurig` for the English-German model we downloaded above. Then, choose the word in the target side that you want to flip, e.g., `traurig` and what you want to flip it to, e.g., `froh` (which means happy/glad in English). Then, you can enter nothing for the optional lists. This should cause the attack to flip the input from `I am sad` to `I am glad`.

Of course, `I am glad` is not "adversarial" in the sense that the model is making a correct translation. We can restrict the attack from adding the word `glad` into the attack. The attack finds `I am lee` which the model translates as `Ich bin froh`.

### Universal Attacks

```bash
python universal.py wmt16.en-de.joined-dict.newstest2014/ --arch transformer_vaswani_wmt_en_de_big --restore-file wmt16.en-de.joined-dict.transformer/model.pt  --bpe subword_nmt --bpe-codes wmt16.en-de.joined-dict.transformer/bpecodes --interactive-attacks --source-lang en --target-lang de
```

This commands defaults to the untargeted attack. Passing `--suffix-dropper` will perform the suffix dropper attack.

## Contributions and Contact

This code was developed by Eric Wallace, contact available at ericwallace@berkeley.edu.

If you'd like to contribute code, feel free to open a [pull request](https://github.com/Eric-Wallace/adversarial-mt/pulls). If you find an issue with the code, please open an [issue](https://github.com/Eric-Wallace/adversarial-mt/issues).

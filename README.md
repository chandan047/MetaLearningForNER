# Meta-Learning for NER

This is built upon the base-code for the paper [Learning to Learn to Disambiguate: Meta-Learning for Few-Shot Word Sense Disambiguation](https://arxiv.org/abs/2004.14355). The code will be updated soon to contain more experiments.


## Getting started

- Clone the repository: `git clone git@github.com:Nithin-Holla/MetaWSD.git`.
- Create a virtual environment.
- Install the required packages: `pip install -r MetaWSD/requirements.txt`.
- Create a directory for storing the data: `mkdir data`.
- Navigate to the data directory: `cd data`.
- Copy the `ontonotes-bert` directory into the data folder.
- Navigate back: `cd ..`

## Preparing the data

- Make sure `train.txt`, `val.txt` and `test.txt` are in the `ontonotes-bert` folder
- The `labels-train.txt` and `labels-test.txt` indicate the entity classes for training episodes and test episodes respectively.


## Training the models

- The YAML configuration files for all the models are in `config/wsd`. To train a model, run `python MetaWSD/train_ner.py --config CONFIG_FILE`.
- Training on multiple GPUs is supported for the MAML variants only. In order to use multiple GPUs, specify the flag `--multi_gpu`.


## Troubleshooting

(Already done. No need to do it again.)

If you have a `RuntimeError` with Proto(FO)MAML and BERT, you can install the `higher` library from this fork: [https://github.com/Nithin-Holla/higher](https://github.com/Nithin-Holla/higher), which has a temporary fix for this. Also, replace `diffopt.step(loss)` with `diffopt.step(loss, retain_graph=True)` in `models/seq_meta.py`.


## Citation

If you use this code repository, please consider citing the paper:
```bib
@article{holla2020metawsd,
  title={Learning to Learn to Disambiguate: Meta-Learning for Few-Shot Word Sense Disambiguation.},
  author={Holla, Nithin and Mishra, Pushkar and Yannakoudakis, Helen and Shutova, Ekaterina},
  journal={arXiv preprint arXiv:2004.14355},
  year={2020}
}
```

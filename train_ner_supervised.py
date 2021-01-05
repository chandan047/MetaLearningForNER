import logging
import os
import random
import warnings
from argparse import ArgumentParser

import coloredlogs
import torch
import yaml

from datetime import datetime
from datasets import utils
from models.supervised import SupervisedNetwork


logger = logging.getLogger('MetaLearningLog')
coloredlogs.install(logger=logger, level='DEBUG', fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

warnings.filterwarnings("ignore", category=UserWarning)


def load_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    config['base_path'] = os.path.dirname(os.path.abspath(__file__))
    config['stamp'] = "stable" #str(datetime.now()).replace(':', '-').replace(' ', '_')
    return config


if __name__ == '__main__':

    # Parse arguments
    parser = ArgumentParser()
    parser.add_argument('--config', dest='config_file', type=str, help='Configuration file', required=True)
    parser.add_argument('--multi_gpu', action='store_true')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config_file)
    config['multi_gpu'] = args.multi_gpu
    logger.info('Using configuration: {}'.format(config))

    # Set seeds for reproducibility
    torch.manual_seed(42)
    random.seed(42)

    # Directory for saving models
    os.makedirs(os.path.join(config['base_path'], 'saved_models'), exist_ok=True)

    # Path for NER dataset
    ner_base_path = os.path.join(config['base_path'], '../data/ontonotes-bert/')

    ner_train_path = os.path.join(ner_base_path, 'train.txt')
    ner_val_path = os.path.join(ner_base_path, 'dev.txt')
    ner_test_path = os.path.join(ner_base_path, 'test.txt')

    # labels_train = os.path.join(ner_base_path, 'labels.txt')
    # labels_test = os.path.join(ner_base_path, 'labels.txt')
    labels_train = os.path.join(ner_base_path, 'labels-g1-train.txt')
    labels_test = os.path.join(ner_base_path, 'labels-g1-train.txt')
    
    # Generate episodes for NER
    logger.info('Generating batches for NER')
    train_dataloader, label_map = utils.generate_ner_batches(dir=ner_train_path,
                                                     labels_file=labels_train,
                                                     batch_size=config['batch_size'],
                                                     meta_train=True)
    val_dataloader, label_map = utils.generate_ner_batches(dir=ner_val_path,
                                                     labels_file=labels_test,
                                                     batch_size=config['eval_batch_size'],
                                                     meta_train=False)
    test_dataloader, label_map = utils.generate_ner_batches(dir=ner_test_path,
                                                     labels_file=labels_test,
                                                     batch_size=config['eval_batch_size'],
                                                     meta_train=False)
    
    logger.info('Finished generating batches for NER')

    learner = SupervisedNetwork(config)

    # Meta-training
    learner.training(train_dataloader, val_dataloader, label_map)
    logger.info('Supervised learning completed')

    # Meta-testing
    learner.testing(test_dataloader, label_map)
    logger.info('Supervised testing completed')

    

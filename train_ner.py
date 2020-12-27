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
from models.baseline import Baseline
from models.majority_classifier import MajorityClassifier
from models.maml import MAML
from models.nearest_neighbor import NearestNeighborClassifier
from models.proto_network import PrototypicalNetwork


logger = logging.getLogger('MetaLearningLog')
coloredlogs.install(logger=logger, level='DEBUG', fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

warnings.filterwarnings("ignore", category=UserWarning)


def load_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    config['base_path'] = os.path.dirname(os.path.abspath(__file__))
    config['stamp'] = str(datetime.now()).replace(':', '-').replace(' ', '_')
    return config


if __name__ == '__main__':

    # Parse arguments
    parser = ArgumentParser()
    parser.add_argument('--config', dest='config_file', type=str, help='Configuration file', required=True)
    parser.add_argument('--multi_gpu', action='store_true')
    parser.add_argument('--tune_layers',type=int,action='store',default=12,help="Set the number of layers to freeze while training BERT")
    args = parser.parse_args()

    config = load_config(args.config_file)
    config['multi_gpu'] = args.multi_gpu
    config['learner_params']['fine_tune_layers'] = args.tune_layers
    logger.info('Using configuration: {}'.format(config))

    # Set seeds for reproducibility
    torch.manual_seed(42)
    random.seed(42)

    # Episodes for meta-training, meta-validation and meta-testing
    train_episodes, val_episodes, test_episodes = [], [], []

    # Directory for saving models
    os.makedirs(os.path.join(config['base_path'], 'saved_models'), exist_ok=True)

    # Path for NER dataset
    ner_base_path = os.path.join(config['base_path'], '../data/ontonotes-bert/')
#     ner_train_path = os.path.join(ner_base_path, 'dev-g1-traincls-{}shot.txt'.format(str(config['num_test_samples']['ner'])))
#     ner_val_path = os.path.join(ner_base_path, 'test-g1-testcls-{}shot.txt'.format(str(config['num_test_samples']['ner'])))
#     ner_test_path = os.path.join(ner_base_path, 'test-g1-testcls-{}shot.txt'.format(str(config['num_test_samples']['ner'])))

    ner_train_path = os.path.join(ner_base_path, 'train.txt')
    ner_val_path = os.path.join(ner_base_path, 'dev.txt')
    ner_test_path = os.path.join(ner_base_path, 'test.txt')

    labels_train = os.path.join(ner_base_path, 'labels-g1-train.txt')
    labels_test = os.path.join(ner_base_path, 'labels-g1-test.txt')
    
    # Generate episodes for NER
    logger.info('Generating episodes for NER')
    ner_train_episodes, _ = utils.generate_ner_episodes(dir=ner_train_path,
                                                     labels_file=labels_train,
                                                     n_episodes=config['num_train_episodes']['ner'],
                                                     n_support_examples=config['num_shots']['ner'],
                                                     n_query_examples=config['num_test_samples']['ner'],
                                                     task='ner',
                                                     meta_train=True)
    ner_val_episodes, _ = utils.generate_ner_episodes(dir=ner_val_path,
                                                   labels_file=labels_test,
                                                   n_episodes=config['num_val_episodes']['ner'],
                                                   n_support_examples=config['num_shots']['ner'],
                                                   n_query_examples=config['num_test_samples']['ner'],
                                                   task='ner',
                                                   meta_train=True)
    ner_test_episodes, label_map = utils.generate_ner_episodes(dir=ner_test_path,
                                                    labels_file=labels_test,
                                                    n_episodes=config['num_test_episodes']['ner'],
                                                    n_support_examples=config['num_shots']['ner'],
                                                    n_query_examples=config['num_test_samples']['ner'],
                                                    task='ner',
                                                    meta_train=False)
    train_episodes.extend(ner_train_episodes)
    val_episodes.extend(ner_val_episodes)
    test_episodes.extend(ner_test_episodes)
    logger.info('Finished generating episodes for NER')

    # Initialize meta learner
    if config['meta_learner'] == 'maml':
        meta_learner = MAML(config)
    elif config['meta_learner'] == 'proto_net':
        meta_learner = PrototypicalNetwork(config)
    elif config['meta_learner'] == 'baseline':
        meta_learner = Baseline(config)
    elif config['meta_learner'] == 'majority':
        meta_learner = MajorityClassifier()
    elif config['meta_learner'] == 'nearest_neighbor':
        meta_learner = NearestNeighborClassifier(config)
    else:
        raise NotImplementedError

    # Meta-training
    meta_learner.training(train_episodes, val_episodes)
    logger.info('Meta-learning completed')

    # Meta-testing
    meta_learner.testing(test_episodes, label_map)
    logger.info('Meta-testing completed')

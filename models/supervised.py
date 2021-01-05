import coloredlogs
import logging
import os
import torch
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from models.seq_supervised import SeqSupervisedNetwork

logger = logging.getLogger('SupervisedLearningLog')
coloredlogs.install(logger=logger, level='DEBUG', fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class SupervisedNetwork:
    def __init__(self, config):
        now = datetime.now()
        date_time = now.strftime("%m-%d-%H-%M-%S")
        self.tensorboard_writer = SummaryWriter(log_dir='runs/ProtoNet-' + date_time)
        
        self.base_path = config['base_path']
        self.stamp = config['stamp']
        self.meta_epochs = config['num_meta_epochs']
        self.early_stopping = config['early_stopping']
        self.stopping_threshold = config.get('stopping_threshold', 1e-3)

        if 'seq' in config['meta_model']:
            self.model = SeqSupervisedNetwork(config)

        logger.info('Supervised network instantiated')

    def training(self, train_dataloader, val_dataloader, tags):
        best_loss = float('inf')
        best_f1 = 0
        patience = 0
        model_path = os.path.join(self.base_path, 'saved_models', 'Supervised-{}.h5'.format(self.stamp))
        classifier_path = os.path.join(self.base_path, 'saved_models', 'SupervisedClassifier-{}.h5'.format(self.stamp))
        logger.info('Model name: Supervised-{}.h5'.format(self.stamp))
        for epoch in range(self.meta_epochs):
            logger.info('Starting epoch {}/{}'.format(epoch + 1, self.meta_epochs))
            avg_loss, avg_accuracy, avg_precision, avg_recall, avg_f1 = self.model(train_dataloader, tags=tags)

            logger.info('Train epoch {}: Avg loss = {:.5f}, avg accuracy = {:.5f}, avg precision = {:.5f}, '
                        'avg recall = {:.5f}, avg F1 score = {:.5f}'.format(epoch + 1, avg_loss, avg_accuracy,
                                                                            avg_precision, avg_recall, avg_f1))

            self.tensorboard_writer.add_scalar('Loss/train', avg_loss, global_step=epoch + 1)
            self.tensorboard_writer.add_scalar('F1/train', avg_f1, global_step=epoch + 1)

            avg_loss, avg_accuracy, avg_precision, avg_recall, avg_f1 = self.model(val_dataloader, tags=tags, testing=True)

            logger.info('Val epoch {}: Avg loss = {:.5f}, avg accuracy = {:.5f}, avg precision = {:.5f}, '
                        'avg recall = {:.5f}, avg F1 score = {:.5f}'.format(epoch + 1, avg_loss, avg_accuracy,
                                                                            avg_precision, avg_recall, avg_f1))

            self.tensorboard_writer.add_scalar('Loss/val', avg_loss, global_step=epoch + 1)
            self.tensorboard_writer.add_scalar('F1/val', avg_f1, global_step=epoch + 1)

            if avg_f1 > best_f1 + self.stopping_threshold:
                patience = 0
                best_loss = avg_loss
                best_f1 = avg_f1
                logger.info('Saving the model since the F1 improved')
                torch.save(self.model.learner.state_dict(), model_path)
                torch.save(self.model.classifier.state_dict(), classifier_path)
                logger.info('')
            else:
                patience += 1
                logger.info('F1 did not improve')
                logger.info('')
                if patience == self.early_stopping:
                    break

            # Log params and grads into tensorboard
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    self.tensorboard_writer.add_histogram('Params/' + name, param.data.view(-1),
                                                     global_step=epoch + 1)
                    self.tensorboard_writer.add_histogram('Grads/' + name, param.grad.data.view(-1),
                                                     global_step=epoch + 1)

        # self.model.learner.load_state_dict(torch.load(model_path))
        # self.model.classifier.load_state_dict(torch.load(classifier_path))
        return best_f1

    def testing(self, test_dataloader, tags):
        logger.info('---------- Supervised testing starts here ----------')
        model_path = os.path.join(self.base_path, 'saved_models', 'Supervised-{}.h5'.format(self.stamp))
        classifier_path = os.path.join(self.base_path, 'saved_models', 'SupervisedClassifier-{}.h5'.format(self.stamp))
        
        self.model.learner.load_state_dict(torch.load(model_path))
        self.model.classifier.load_state_dict(torch.load(classifier_path))
        
        _, accuracy, precision, recall, f1_score = self.model(test_dataloader, tags=tags, testing=True)

        logger.info('Avg meta-testing metrics: Accuracy = {:.5f}, precision = {:.5f}, recall = {:.5f}, '
                    'F1 score = {:.5f}'.format(accuracy,
                                               precision,
                                               recall,
                                               f1_score))
        return f1_score

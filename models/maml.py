from torch import optim, nn

import coloredlogs
import logging
import os
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import models.utils
from models.seq_meta import SeqMetaModel

logger = logging.getLogger('MAML Log')
coloredlogs.install(logger=logger, level='DEBUG',
                    fmt='%(asctime)s - %(name)s - %(levelname)s'
                        ' - %(message)s')
tensorboard_writer = SummaryWriter(log_dir='runs/MAML')


class MAML:
    def __init__(self, config):
        self.base_path = config['base_path']
        self.stamp = config['stamp']
        self.updates = config['num_updates']
        self.meta_epochs = config['num_meta_epochs']
        self.early_stopping = config['early_stopping']
        self.meta_lr = config.get('meta_lr', 1e-3)
        self.meta_weight_decay = config.get('meta_weight_decay', 0.0)
        self.stopping_threshold = config.get('stopping_threshold', 1e-3)
        self.fomaml = config.get('fomaml', False)
        self.multi_gpu = torch.cuda.device_count() > 1 if 'cuda' in config.get('device', 'cpu') else False

        if self.multi_gpu:
            self.n_devices = torch.cuda.device_count()
            logger.info('Using {} GPUs'.format(self.n_devices))

        if 'seq_meta' in config['meta_model']:
            self.meta_model = SeqMetaModel(config)

        if self.fomaml:
            logger.info('FOMAML instantiated')
        else:
            logger.info('MAML instantiated')

    def _replicate_model(self):
        replica_meta_models = models.utils.replicate_model_to_gpus(self.meta_model, list(range(self.n_devices)))
        return replica_meta_models

    def _multi_gpu_training(self, train_episodes):

        chunked_train_episodes = []
        for i in range(self.n_devices):
            chunk = train_episodes[i::self.n_devices]
            chunked_train_episodes.append([chunk])
        kwargs_tup = ({'updates': self.updates}, ) * self.n_devices

        parallel_outputs = nn.parallel.parallel_apply(self.replica_meta_models, chunked_train_episodes,
                                                      kwargs_tup, list(range(self.n_devices)))

        losses, accuracies, precisions, recalls, f1s = [], [], [], [], []
        for i in range(self.n_devices):
            losses.extend(parallel_outputs[i][0])
            accuracies.extend(parallel_outputs[i][1])
            precisions.extend(parallel_outputs[i][2])
            recalls.extend(parallel_outputs[i][3])
            f1s.extend(parallel_outputs[i][4])

        target_device = next(self.meta_model.learner.parameters()).device
        for name, param in self.meta_model.learner.named_parameters():
            if param.requires_grad:
                grad_sum = 0
                for r in self.replica_meta_models:
                    for n, p in r.learner.named_parameters():
                        if n == name:
                            grad_sum += p.grad.to(target_device)
                param.grad = grad_sum / self.n_devices

        return losses, accuracies, precisions, recalls, f1s

    def _synchronize_weights(self):
        for rm in self.replica_meta_models[1:]:
            rm.load_state_dict(self.meta_model.state_dict())

    def training(self, train_episodes, val_episodes):
        learner_params = [p for p in self.meta_model.learner.parameters() if p.requires_grad]
        meta_optimizer = optim.Adam(learner_params, lr=self.meta_lr, weight_decay=self.meta_weight_decay)
        best_loss = float('inf')
        best_f1 = 0
        patience = 0
        model_path = os.path.join(self.base_path, 'saved_models', 'MetaModel-{}.h5'.format(self.stamp))
        logger.info('Model name: MetaModel-{}.h5'.format(self.stamp))

        if self.multi_gpu:
            self.replica_meta_models = self._replicate_model()

        for epoch in range(self.meta_epochs):
            logger.info('Starting epoch {}'.format(epoch+1))
            meta_optimizer.zero_grad()

            if not self.multi_gpu:
                losses, accuracies, precisions, recalls, f1s = self.meta_model(train_episodes, self.updates)
            else:
                losses, accuracies, precisions, recalls, f1s = self._multi_gpu_training(train_episodes)

            meta_optimizer.step()

            if self.multi_gpu:
                self._synchronize_weights()

            avg_loss = np.mean(losses)
            avg_accuracy = np.mean(accuracies)
            avg_precision = np.mean(precisions)
            avg_recall = np.mean(recalls)
            avg_f1 = np.mean(f1s)

            logger.info('Meta train epoch {}: Avg loss = {:.5f}, avg accuracy = {:.5f}, avg precision = {:.5f}, '
                        'avg recall = {:.5f}, avg F1 score = {:.5f}'.format(epoch + 1, avg_loss, avg_accuracy,
                                                                            avg_precision, avg_recall, avg_f1))

            tensorboard_writer.add_scalar('Loss/train', avg_loss, global_step=epoch+1)

            losses, accuracies, precisions, recalls, f1s = self.meta_model(val_episodes, self.updates, testing=True)

            avg_loss = np.mean(losses)
            avg_accuracy = np.mean(accuracies)
            avg_precision = np.mean(precisions)
            avg_recall = np.mean(recalls)
            avg_f1 = np.mean(f1s)

            logger.info('Meta val epoch {}: Avg loss = {:.5f}, avg accuracy = {:.5f}, avg precision = {:.5f}, '
                        'avg recall = {:.5f}, avg F1 score = {:.5f}'.format(epoch + 1, avg_loss, avg_accuracy,
                                                                            avg_precision, avg_recall, avg_f1))

            tensorboard_writer.add_scalar('Loss/val', avg_loss, global_step=epoch+1)

            if avg_loss < best_loss - self.stopping_threshold:
                patience = 0
                best_loss = avg_loss
                best_f1 = avg_f1
                torch.save(self.meta_model.learner.state_dict(), model_path)
                logger.info('Saving the model since the loss improved')
            else:
                patience += 1
                logger.info('Loss did not improve')
                if patience == self.early_stopping:
                    break

            # Log params and grads into tensorboard
            for name, param in self.meta_model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    tensorboard_writer.add_histogram('Params/' + name, param.data.view(-1), global_step=epoch+1)
                    tensorboard_writer.add_histogram('Grads/' + name, param.grad.data.view(-1), global_step=epoch+1)

        self.meta_model.learner.load_state_dict(torch.load(model_path))
        return best_f1

    def testing(self, test_episodes):
        logger.info('---------- Meta testing starts here ----------')
        episode_accuracies, episode_precisions, episode_recalls, episode_f1s = [], [], [], []
        for episode in test_episodes:
            loss, accuracy, precision, recall, f1_score = self.meta_model([episode], updates=self.updates, testing=True)
            loss = loss[0]
            accuracy, precision, recall, f1_score = accuracy[0], precision[0], recall[0], f1_score[0]
            episode_accuracies.append(accuracy)
            episode_precisions.append(precision)
            episode_recalls.append(recall)
            episode_f1s.append(f1_score)

        logger.info('Avg meta-testing metrics: Accuracy = {:.5f}, precision = {:.5f}, recall = {:.5f}, '
                    'F1 score = {:.5f}'.format(np.mean(episode_accuracies),
                                               np.mean(episode_precisions),
                                               np.mean(episode_recalls),
                                               np.mean(episode_f1s)))

        return np.mean(episode_f1s)

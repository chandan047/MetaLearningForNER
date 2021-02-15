import torchtext
# from allennlp.modules import Elmo
# from allennlp.modules.elmo import batch_to_ids

from torch import nn
from torch import optim

import pdb
import coloredlogs
import logging
import os
import pdb
import torch
from transformers import BertTokenizer, AdamW, get_constant_schedule_with_warmup

from models import utils
from models.base_models import RNNSequenceModel, MLPModel, BERTSequenceModel
from models.utils import make_prediction

logger = logging.getLogger('Log')
coloredlogs.install(logger=logger, level='DEBUG', fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class SeqSupervisedNetwork(nn.Module):
    def __init__(self, config):
        super(SeqSupervisedNetwork, self).__init__()
        self.base_path = config['base_path']
        self.early_stopping = config['early_stopping']
        self.lr = config.get('meta_lr', 1e-3)
        self.weight_decay = config.get('meta_weight_decay', 0.0)

        if 'seq' in config['learner_model']:
            self.learner = RNNSequenceModel(config['learner_params'])
        elif 'mlp' in config['learner_model']:
            self.learner = MLPModel(config['learner_params'])
        elif 'bert' in config['learner_model']:
            self.learner = BERTSequenceModel(config['learner_params'])
        
        self.dropout = nn.Dropout(config['learner_params']['dropout_ratio'])
        self.classifier = nn.Linear(config['learner_params']['embed_dim'], config['learner_params']['num_outputs']['ner'])

        self.num_outputs = config['learner_params']['num_outputs']
        self.vectors = config.get('vectors', 'glove')

        if self.vectors == 'elmo':
            self.elmo = Elmo(options_file="https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json",
                             weight_file="https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5",
                             num_output_representations=1,
                             dropout=0,
                             requires_grad=False)
        elif self.vectors == 'glove':
            self.glove = torchtext.vocab.GloVe(name='840B', dim=300)
        elif self.vectors == 'bert':
            self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

        self.loss_fn = {}
        for task in config['learner_params']['num_outputs']:
            self.loss_fn[task] = nn.CrossEntropyLoss(ignore_index=-1)

        if config.get('trained_learner', False):
            self.learner.load_state_dict(torch.load(
                os.path.join(self.base_path, 'saved_models', config['trained_learner'])
            ))
            self.classifier.load_state_dict(torch.load(
                os.path.join(self.base_path, 'saved_models', config['trained_classifier'])
            ))
            logger.info('Loaded trained learner model {}'.format(config['trained_learner']))

        self.device = torch.device(config.get('device', 'cpu'))
        self.to(self.device)

        if self.vectors == 'elmo':
            self.elmo.to(self.device)

        self.initialize_optimizer_scheduler()

    def initialize_optimizer_scheduler(self):
        learner_params = [p for p in self.learner.parameters() if p.requires_grad]
        learner_params += self.dropout.parameters()
        learner_params += self.classifier.parameters()
        
        if isinstance(self.learner, BERTSequenceModel):
            self.optimizer = AdamW(learner_params, lr=self.lr, weight_decay=self.weight_decay)
            self.lr_scheduler = get_constant_schedule_with_warmup(self.optimizer, num_warmup_steps=100)
        else:
            self.optimizer = optim.Adam(learner_params, lr=self.lr, weight_decay=self.weight_decay)
            self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=500, gamma=0.5)

    def vectorize(self, batch_x, batch_len, batch_y):
        with torch.no_grad():
            if self.vectors == 'elmo':
                char_ids = batch_to_ids(batch_x)
                char_ids = char_ids.to(self.device)
                batch_x = self.elmo(char_ids)['elmo_representations'][0]
            elif self.vectors == 'glove':
                max_batch_len = max(batch_len)
                vec_batch_x = torch.ones((len(batch_x), max_batch_len, 300))
                for i, sent in enumerate(batch_x):
                    sent_emb = self.glove.get_vecs_by_tokens(sent, lower_case_backup=True)
                    vec_batch_x[i, :len(sent_emb)] = sent_emb
                batch_x = vec_batch_x.to(self.device)
            elif self.vectors == 'bert':
                max_batch_len = max(batch_len) + 2
                input_ids = torch.zeros((len(batch_x), max_batch_len)).long()
                for i, sent in enumerate(batch_x):
                    sent_token_ids = self.bert_tokenizer.encode(sent, add_special_tokens=True)
                    input_ids[i, :len(sent_token_ids)] = torch.tensor(sent_token_ids)
                batch_x = input_ids.to(self.device)

        batch_len = torch.tensor(batch_len).to(self.device)
        batch_y = torch.tensor(batch_y).to(self.device)
        return batch_x, batch_len, batch_y

    def forward(self, dataloader, tags=None, testing=False, writer=None):
        if not testing:
            self.train()
        else:
            self.eval()
        
        avg_loss = 0
        all_predictions, all_labels = [], []
        
        for batch_id, batch in enumerate(dataloader):
            batch_x, batch_len, batch_y = next(iter(batch))
            batch_x, batch_len, batch_y = self.vectorize(batch_x, batch_len, batch_y)

            batch_x_repr = self.learner(batch_x, batch_len)
            output = self.dropout(batch_x_repr)
            output = self.classifier(output)

            batch_size, seq_len = output.shape[0], output.shape[1]
            
            output = output.view(batch_size * seq_len, -1)
            batch_y = batch_y.view(-1)
            loss = self.loss_fn['ner'](output, batch_y)
            avg_loss += loss.item()
            
            if not testing:
                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)
                self.optimizer.step()
                self.lr_scheduler.step()

            output = output.view(batch_size, seq_len, -1)
            batch_y = batch_y.view(batch_size, seq_len)
            
            predictions, labels = [], []
            
            for bid in range(batch_size):
                relevant_indices = torch.nonzero(batch_y[bid] != -1).view(-1).detach()
                predictions.append(list(make_prediction(output[bid][relevant_indices]).detach().cpu().numpy()))
                # pdb.set_trace()
                labels.append(list(batch_y[bid][relevant_indices].detach().cpu().numpy()))
            
            accuracy, precision, recall, f1_score = utils.calculate_seqeval_metrics(predictions,
                                                                            labels, tags, binary=False)

            if writer is not None:
                writer.add_scalar('Loss/iter', avg_loss / (batch_id+1), global_step=batch_id + 1)
                writer.add_scalar('F1/iter', f1_score, global_step=batch_id + 1)
            
            if (batch_id + 1) % 100 == 0:
                logger.info('Batch {}/{}, task {} [supervised]: Loss = {:.5f}, accuracy = {:.5f}, precision = {:.5f}, '
                        'recall = {:.5f}, F1 score = {:.5f}'.format(batch_id + 1, len(dataloader), 'ner',
                                                                    loss.item(), accuracy, precision, recall, f1_score))
            
            all_predictions.extend(predictions)
            all_labels.extend(labels)

        avg_loss /= len(dataloader)

        # Calculate metrics
        accuracy, precision, recall, f1_score = utils.calculate_seqeval_metrics(all_predictions,
                                                                        all_labels, binary=False)

        return avg_loss, accuracy, precision, recall, f1_score


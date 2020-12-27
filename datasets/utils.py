import glob
import json
import os
import random

from torch.utils import data
from torch.nn import CrossEntropyLoss
from torch.utils.data import Subset

from datasets.episode import Episode
from datasets.wsd_dataset import WordWSDDataset, MetaWSDDataset
from datasets.ner_dataset import NERSampler,SequentialSampler, read_examples_from_file, get_labels
from transformers import BertTokenizer

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

def write_json(json_dict, file_name):
    with open(file_name, 'w', encoding='utf8') as f:
        json.dump(json_dict, f, indent=4)


def read_json(file_name):
    with open(file_name, 'r', encoding='utf8') as f:
        json_dict = json.load(f)
    return json_dict


def get_max_batch_len(batch):
    return max([len(x[0]) for x in batch])


def prepare_batch(batch):
    max_len = get_max_batch_len(batch)
    x = []
    lengths = []
    y = []
    for inp_seq, target_seq in batch:
        lengths.append(len(inp_seq))
        target_seq = target_seq + [-1] * (max_len - len(target_seq))
        x.append(inp_seq)
        y.append(target_seq)
        
    # print (lengths[0], x[0], y[0])
    
    return x, lengths, y


def prepare_bert_batch(batch):
    x = []
    lengths = []
    y = []
    for sentences, labels in batch:
        tokens = []
        label_ids = []
        length = 0
        for word, label in zip(sentences, labels):
            word_tokens = bert_tokenizer.tokenize(word)
            tokens.extend(word_tokens)
            label_ids.extend([label] + [-1] * (len(word_tokens) - 1))
            length += len(word_tokens)
        
        # check
#         if all([lab == -1 for lab in label_ids]):
#             print (labels)
#         assert(all([lab == -1 for lab in label_ids]) == False)
               
        x.append(tokens)
        lengths.append(length)
        y.append(label_ids)
    
    max_len = max(lengths)
    for i in range(len(y)):
        y[i] = y[i] + [-1] * (max_len - len(y[i]))
    
#     print (x[-1])
#     print (batch[-1][1])
#     print (y[-1])
    
    return x, lengths, y


def prepare_task_batch(batch):
    return batch


def generate_semcor_wsd_episodes(wsd_dataset, n_episodes, n_support_examples, n_query_examples, task):
    word_splits = {k: v for (k, v) in wsd_dataset.word_splits.items() if len(v['sentences']) >
                   (n_support_examples + n_query_examples)}

    if n_episodes > len(word_splits):
        raise Exception('Not enough data available to generate {} episodes'.format(n_episodes))

    episodes = []
    for word in word_splits.keys():
        if len(episodes) == n_episodes:
            break
        indices = list(range(len(word_splits[word]['sentences'])))
        random.shuffle(indices)
        start_index = 0
        train_subset = WordWSDDataset(sentences=[word_splits[word]['sentences'][i] for i in indices[start_index: start_index + n_support_examples]],
                                      labels=[word_splits[word]['labels'][i] for i in indices[start_index: start_index + n_support_examples]],
                                      n_classes=len(wsd_dataset.sense_inventory[word]))
        support_loader = data.DataLoader(train_subset, batch_size=n_support_examples, collate_fn=prepare_batch)
        start_index += n_support_examples
        test_subset = WordWSDDataset(sentences=[word_splits[word]['sentences'][i] for i in indices[start_index: start_index + n_query_examples]],
                                     labels=[word_splits[word]['labels'][i] for i in indices[start_index: start_index + n_query_examples]],
                                     n_classes=len(wsd_dataset.sense_inventory[word]))
        query_loader = data.DataLoader(test_subset, batch_size=n_query_examples, collate_fn=prepare_batch)
        episode = Episode(support_loader=support_loader,
                          query_loader=query_loader,
                          base_task=task,
                          task_id=task + '-' + word,
                          n_classes=train_subset.n_classes)
        episodes.append(episode)
    return episodes


def generate_wsd_episodes(dir, n_episodes, n_support_examples, n_query_examples, task, meta_train=True):
    episodes = []
    for file_name in glob.glob(os.path.join(dir, '*.json')):
        if len(episodes) == n_episodes:
            break
        word = file_name.split(os.sep)[-1].split('.')[0]
        word_wsd_dataset = MetaWSDDataset(file_name)
        train_subset = Subset(word_wsd_dataset, range(0, n_support_examples))
        support_loader = data.DataLoader(train_subset, batch_size=n_support_examples, collate_fn=prepare_batch)
        if meta_train:
            test_subset = Subset(word_wsd_dataset, range(n_support_examples, n_support_examples + n_query_examples))
        else:
            test_subset = Subset(word_wsd_dataset, range(n_support_examples, len(word_wsd_dataset)))
        query_loader = data.DataLoader(test_subset, batch_size=n_query_examples, collate_fn=prepare_batch)
        episode = Episode(support_loader=support_loader,
                          query_loader=query_loader,
                          base_task=task,
                          task_id=task + '-' + word,
                          n_classes=word_wsd_dataset.n_classes)
        episodes.append(episode)
    return episodes


def generate_ner_episodes(dir, labels_file, n_episodes, n_support_examples, n_query_examples, task, 
                          meta_train=False, vectors='bert'):
    episodes = []
    labels = get_labels(labels_file)
    examples, label_map = read_examples_from_file(dir, labels)
    print ('label_map', label_map)
    if meta_train == True:
        ner_dataset = NERSampler(examples, labels, label_map, 6, n_support_examples, n_query_examples, n_episodes)
    else:
        ner_dataset = SequentialSampler(examples, labels, label_map, 6, n_support_examples, n_query_examples, n_episodes)
    for index, ner_data in enumerate(ner_dataset):
        tags, sup_sents, query_sents = ner_data
        
#         print (len(tags), len(sup_sents.labels), len(query_sents.labels))
        if vectors == 'bert':
            support_loader = data.DataLoader(sup_sents, batch_size=6*n_support_examples, 
                                             collate_fn=lambda pb: prepare_bert_batch(pb))
            query_loader = data.DataLoader(query_sents, batch_size=6*n_query_examples, 
                                             collate_fn=lambda pb: prepare_bert_batch(pb))
        else:
            support_loader = data.DataLoader(sup_sents, batch_size=6*n_support_examples, 
                                             collate_fn=prepare_batch)
            query_loader = data.DataLoader(query_sents, batch_size=6*n_query_examples, 
                                             collate_fn=prepare_batch)
        episode = Episode(support_loader=support_loader,
                          query_loader=query_loader,
                          base_task=task,
                          task_id=task + '-' + str(index),
                          n_classes=len(labels))
        
        episodes.append(episode)
    return episodes,label_map

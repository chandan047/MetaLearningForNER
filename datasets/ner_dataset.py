import itertools
import json
import os
import random
from collections import defaultdict, Counter
from tqdm.auto import tqdm, trange

from torch.nn import CrossEntropyLoss
from torch.utils import data
import numpy as np

from datasets import utils

class NERSampler:

    def __init__(self, dataset, labels, label_map, n_cls, n_shot, n_query=5, n_batch=100):
        print (f'Number of examples in NER dataset is {len(dataset)}')
        self.labels = labels
        self.classes = set()
        for lab in labels:
            if len(lab) > 2:
                self.classes.add(lab[2:])
        self.label_map = label_map
        self.n_cls = n_cls
        self.n_shot = n_shot
        self.n_query = n_query
        self.n_batch = n_batch
        self.dataset = dataset
        print ("{}-way {}-shot with {}-query and {} batchsize".format(self.n_cls, self.n_shot, self.n_query, self.n_batch))
        self.sent_class_map, self.class_sent_map = self._get_sent_class_maps(dataset)
        
        # stats on data
        print ('## STATISTICS ##')
        for cls in self.class_sent_map:
            print (cls, len(self.class_sent_map[cls]))
        
        self.data = self.make_episodes()

    def make_episodes(self):
        """
        Sample mini-batches for episode training
        """
        tags_epi, sup_epi, query_epi = [], [], []
        
        for _ in trange(self.n_batch):
            classes = self._sample_classes()
#             print ("sampled classes", classes)
            tags = defaultdict(lambda:-1)
            # tags['O'] = 0
            for cls in classes:
                if cls not in tags:
                    tags[cls] = len(tags)
            sup_sents, query_sents = self.sample_sentences(classes, tags)
#             print ('sampled support labels', sup_sents.labels)
#             print ('sampled query labels', query_sents.labels)
            
            tags_epi.append(tags)
            sup_epi.append(sup_sents)
            query_epi.append(query_sents)
        
        return tags_epi, sup_epi, query_epi

    def __getitem__(self, index):
        return self.data[0][index], self.data[1][index], self.data[2][index]
            
    def __len__(self):
        return self.n_batch

    @staticmethod
    def _get_sent_class_maps(dataset):
        # map from a sentence Id to a list of pairs with
        # B-Xs and the freqs of B-X in the sentence
        sent_class_map = defaultdict(list)
        # map from B-X to a list of pairs with
        # sentence ids and the freqs of B-X in the sentence
        class_sent_map = defaultdict(list)
        for i, sent in enumerate(dataset):
            _, tags = sent.words, sent.labels
            class_freqs = Counter()
            for tag in tags:
                if tag.startswith('B-'):
                    # we only store the `X` part of `B-X`
                    class_freqs[tag[2:]] += 1
            for cls, freq in class_freqs.items():
                sent_class_map[i].append((cls, freq))
                class_sent_map[cls].append((i, freq))
        return sent_class_map, class_sent_map

    def tagged_labels(self, labels, tags):
        return [
            tags[lab[2:]] if len(lab) > 2 else tags[lab] 
            for lab in labels
        ]
    
    def sample_sentences(self, classes, tags):
        """
        Sample support and query sentences. A greedy algorithm is implemented
        that always sample less freqent classes first.
        :param classes: the entity classes of interests
        :param n_shot: the number of support points
        :param n_query: the number of query points
        :return: two lists of sentence Ids for support and query sets
                 respectively
        """
        sup_sents, query_sents = [], []
        # sample support set
        sampled_cls_counters = {cls: 0 for cls in classes}
        for cls in classes:
            # not enough sentences for the class, so sample with replacement
            replacement = (len(self.class_sent_map[cls]) < self.n_shot)
            while sampled_cls_counters[cls] < self.n_shot:
                sent, _ = random.choice(self.class_sent_map[cls])
                if not replacement and sent in sup_sents:
                    continue
                for inn_cls, freq in self.sent_class_map[sent]:
                    if inn_cls in sampled_cls_counters:
                        sampled_cls_counters[inn_cls] += freq
                sup_sents.append(sent)
        # sample query set
        sampled_cls_counters = {cls: 0 for cls in classes}
        for cls in classes:
            # not enough sentences for the class, so sample with replacement
            replacement = (len(self.class_sent_map[cls]) < self.n_shot + self.n_query)
            while sampled_cls_counters[cls] < self.n_query:
                sent, _ = random.choice(self.class_sent_map[cls])
                if not replacement and (sent in sup_sents
                                        or sent in query_sents):
                    continue
                for inn_cls, freq in self.sent_class_map[sent]:
                    if inn_cls in sampled_cls_counters:
                        sampled_cls_counters[inn_cls] += freq
                query_sents.append(sent)
            
        return MetaNERDataset(
            [self.dataset[d].words for d in sup_sents],
            [self.tagged_labels(self.dataset[d].labels, tags) for d in sup_sents], 
            self.n_cls
        ), MetaNERDataset(
            [self.dataset[d].words for d in query_sents],
            [self.tagged_labels(self.dataset[d].labels, tags) for d in query_sents],
            self.n_cls
        )

    def _sample_classes(self):
        """
        Subsample entity classes, sorted by frequencies
        :param targets: target classes to sample from
        :param n_cls: num of entity classes to sample
        :return: a list of classes
        """
        sorted_list = []
        for cls, val in self.class_sent_map.items():
            if cls not in self.classes:
                continue
            sorted_list.append((cls, len(val)))
        assert len(sorted_list) >= self.n_cls
        random.shuffle(sorted_list)
        sorted_list = sorted_list[:self.n_cls]
        sorted_list = sorted(sorted_list, key=lambda p: p[1])
        return [cls for cls, _ in sorted_list]


class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid, words, labels):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.words = words
        self.labels = labels


def read_examples_from_file(data_dir, valid_labels):
    print (f'valid labels: {valid_labels}')
    file_path = data_dir
    guid_index = 1
    examples = []
    with open(file_path, encoding="utf-8") as f:
        words = []
        labels = []
        for line in f:
            if line.startswith("-DOCSTART-") or line.strip() == "":
                if words:
                    for i, label in enumerate(labels):
                        if label not in valid_labels:
                            labels[i] = 'O'
                    examples.append(InputExample(guid="{}".format(guid_index), words=words, labels=labels))
                    guid_index += 1
                    words = []
                    labels = []
            else:
                splits = line.split()
                words.append(splits[0])
                if len(splits) > 1:
                    labels.append(splits[-1].replace("\n", ""))
                else:
                    # Examples could have no label for mode = "test"
                    labels.append("O")
        if words:
            for i, label in enumerate(labels):
                if label not in valid_labels:
                    labels[i] = 'O'
            examples.append(InputExample(guid="{}".format(guid_index), words=words, labels=labels))
        
    label_map = defaultdict(int)
    for i, label in enumerate(valid_labels):   # assumption that valid_labels[0] == 'O'
        if label == 'O':
            label_map[label] = i
        else:
            if label[2:] not in label_map:
                label_map[label[2:]] = len(label_map)
    
    return examples, label_map


def get_labels(path):
    if path:
        with open(path, "r") as f:
            labels = f.read().splitlines()
        if "O" not in labels:
            labels = ["O"] + labels
        return labels
    else:
        return ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]



class MetaNERDataset(data.Dataset):

    def __init__(self, sentences, labels, n_classes):
        self.sentences = sentences
        self.labels = labels
        self.n_classes = n_classes

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        return self.sentences[index], self.labels[index]


# class MetaNERDataset(data.Dataset):

#     def __init__(self, file_name):
#         json_dict = utils.read_json(file_name)
#         self.sentences, self.labels = [],  []
#         for entry in json_dict:
#             self.sentences.append(entry['sentence'])
#             self.labels.append(entry['label'])
#         self.n_classes = np.max(list(itertools.chain(*self.labels))) + 1

#     def __len__(self):
#         return len(self.sentences)

#     def __getitem__(self, index):
#         return self.sentences[index], self.labels[index]


class SequentialSampler:
    def __init__(self, dataset, labels, label_map, n_cls, n_shot, n_query=5, n_batch=100):
        print (f'Number of examples in NER dataset is {len(dataset)}')
        self.labels = labels
        self.classes = set()
        for lab in labels:
            if len(lab) > 2:
                self.classes.add(lab[2:])
        self.label_map = label_map
        self.n_cls = n_cls
        self.n_shot = n_shot
        self.n_query = n_query
        self.n_batch = n_batch
        self.dataset = dataset
        print ("{}-way {}-shot with {}-query and {} batchsize".format(self.n_cls, self.n_shot, self.n_query, self.n_batch))
        self.sent_class_map, self.class_sent_map = self._get_sent_class_maps(dataset)
        
        # stats on data
        print ('## STATISTICS ##')
        for cls in self.class_sent_map:
            print (cls, len(self.class_sent_map[cls]))
        
        self.data = self.make_episodes()

    def make_episodes(self):
        """
        Sample mini-batches for episode training
        """
        tags_epi, sup_epi, query_epi = [], [], []
        
        for _ in trange(self.n_batch):
            classes = self._sample_classes()
#             print ("sampled classes", classes)
            tags = defaultdict(lambda:-1)
            tags['O'] = 0
            for cls in classes:
                if cls not in tags:
                    tags[cls] = len(tags)
            sup_sents = self.sample_support_sentences(classes,tags)
            for i in range(int(len(self.dataset)/(self.n_cls*self.n_shot))):
                query_sents = self.sample_query_sentences(classes, tags, i)

                tags_epi.append(tags)
                sup_epi.append(sup_sents)
                query_epi.append(query_sents)
        
        return tags_epi, sup_epi, query_epi

    def __getitem__(self, index):
        return self.data[0][index], self.data[1][index], self.data[2][index]
            
    def __len__(self):
        return self.n_batch

    @staticmethod
    def _get_sent_class_maps(dataset):
        # map from a sentence Id to a list of pairs with
        # B-Xs and the freqs of B-X in the sentence
        sent_class_map = defaultdict(list)
        # map from B-X to a list of pairs with
        # sentence ids and the freqs of B-X in the sentence
        class_sent_map = defaultdict(list)
        for i, sent in enumerate(dataset):
            _, tags = sent.words, sent.labels
            class_freqs = Counter()
            for tag in tags:
                if tag.startswith('B-'):
                    # we only store the `X` part of `B-X`
                    class_freqs[tag[2:]] += 1
            for cls, freq in class_freqs.items():
                sent_class_map[i].append((cls, freq))
                class_sent_map[cls].append((i, freq))
        return sent_class_map, class_sent_map

    def tagged_labels(self, labels, tags):
        return [
            tags[lab[2:]] if len(lab) > 2 else tags[lab] 
            for lab in labels
        ]
    def sample_support_sentences(self, classes, tags):
        """
        Sample support and query sentences. A greedy algorithm is implemented
        that always sample less freqent classes first.
        :param classes: the entity classes of interests
        :param n_shot: the number of support points
        :param n_query: the number of query points
        :return: two lists of sentence Ids for support and query sets
                 respectively
        """
        sup_sents = []
        # sample support set
        sampled_cls_counters = {cls: 0 for cls in classes}
        for cls in classes:
            # not enough sentences for the class, so sample with replacement
            replacement = (len(self.class_sent_map[cls]) < self.n_shot)
            while sampled_cls_counters[cls] < self.n_shot:
                sent, _ = random.choice(self.class_sent_map[cls])
                if not replacement and sent in sup_sents:
                    continue
                for inn_cls, freq in self.sent_class_map[sent]:
                    if inn_cls in sampled_cls_counters:
                        sampled_cls_counters[inn_cls] += freq
                sup_sents.append(sent)
        return MetaNERDataset(
            [self.dataset[d].words for d in sup_sents],
            [self.tagged_labels(self.dataset[d].labels, tags) for d in sup_sents], 
            self.n_cls+1
        )
    
    def sample_query_sentences(self, classes, tags, i):
        """
        Sample support and query sentences. A greedy algorithm is implemented
        that always sample less freqent classes first.
        :param classes: the entity classes of interests
        :param n_shot: the number of support points
        :param n_query: the number of query points
        :return: two lists of sentence Ids for support and query sets
                 respectively
        """
        query_sents = [d for d in range(i*self.n_cls*self.n_shot,(i+1)*self.n_cls*self.n_shot)]
           
        return MetaNERDataset(
            [self.dataset[d].words for d in query_sents],
            [self.tagged_labels(self.dataset[d].labels, tags) for d in query_sents],
            self.n_cls+1
        )

    def _sample_classes(self):
        """
        Subsample entity classes, sorted by frequencies
        :param targets: target classes to sample from
        :param n_cls: num of entity classes to sample
        :return: a list of classes
        """
        sorted_list = []
        for cls, val in self.class_sent_map.items():
            if cls not in self.classes:
                continue
            sorted_list.append((cls, len(val)))
        assert len(sorted_list) >= self.n_cls
        random.shuffle(sorted_list)
        sorted_list = sorted_list[:self.n_cls]
        sorted_list = sorted(sorted_list, key=lambda p: p[1])
        return [cls for cls, _ in sorted_list]

from collections import Counter
import json

import torch
from torch import cuda
from torch.utils.data.dataloader import DataLoader

UNKNOWN_WORD = 'UNK'

device = 'cuda' if cuda.is_available() else 'cpu'


class DataLoaderA(torch.utils.data.Dataset):
    variation = None
    w_2_idx = None
    tag_2_idx = None
    idx_2_tag = None
    voc = None

    def __init__(self, path, word_dict=None, target_dict=None, is_train=False, variation='a'):
        sentences, tags = self.data_loading(path)
        train = False
        if is_train and DataLoaderA.w_2_idx is None:
            DataLoaderA.variation = variation
            self.dicts_creation(sentences, tags)
            train = True
            self.save_dict(word_dict, target_dict)

        self.X = self.sent_2_idx(sentences, train)
        self.Y = self.tag_to_idx(tags)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x, y = self.X[idx], self.Y[idx]

        data = torch.tensor(x, dtype=torch.long)
        tag = torch.tensor(y, dtype=torch.long)
        return data, tag

    def sent_2_idx(self, sentences, train):
        id_sentences = []
        for sent in sentences:
            sent_temp = []
            for word in sent.split():
                if train and DataLoaderA.voc[word] < 5:
                    word = UNKNOWN_WORD
                elif not train and word not in DataLoaderA.w_2_idx.keys():
                    if word.lower() in DataLoaderA.w_2_idx.keys():
                        word = word.lower()
                    else:
                        word = UNKNOWN_WORD
                sent_temp.append(DataLoaderA.w_2_idx.get(word, DataLoaderA.w_2_idx[UNKNOWN_WORD]))
            id_sentences.append(sent_temp)
        return id_sentences

    def tag_to_idx(self, tags):
        id_tags = []
        for tag in tags:
            tag_temp = []
            for t in tag.split():
                tag_temp.append(DataLoaderA.tag_2_idx.get(t))
            id_tags.append(tag_temp)

        return id_tags

    def dicts_creation(self, sentences, tags):
        sent_temp = ' '.join(sentences).split()
        tag_temp = ' '.join(tags).split()

        DataLoaderA.voc = Counter(sent_temp)
        DataLoaderA.voc[UNKNOWN_WORD] = 999
        tags = set(tag_temp)

        DataLoaderA.w_2_idx = dict(zip(DataLoaderA.voc.keys(), range(1, len(DataLoaderA.voc) + 1)))
        DataLoaderA.w_2_idx['<PAD>'] = 0
        DataLoaderA.tag_2_idx = dict(zip(tags, range(1, len(tags) + 1)))
        DataLoaderA.tag_2_idx['<PAD>'] = 0
        DataLoaderA.idx_2_tag = {k: v for v, k in self.tag_2_idx.items()}

    def save_dict(self, word_dict, target_dic):
        with open(word_dict + '.json', 'w') as fp:
            json.dump(DataLoaderA.w_2_idx, fp)
        with open(target_dic + '.json', 'w') as fp:
            json.dump(DataLoaderA.tag_2_idx, fp)

    def data_loading(self, path):
        with open(path) as file:
            sent_temp = []
            tags_temp = []
            sentences = []
            tags = []

            for line in file:
                try:
                    word, label = line.split()
                    sent_temp.append(word), tags_temp.append(label)
                except ValueError:
                    sequence_sent = ' '.join(sent_temp)
                    sequence_tag = ' '.join(tags_temp)

                    sentences.append(sequence_sent)
                    tags.append(sequence_tag)

                    sent_temp = []
                    tags_temp = []

        return sentences, tags
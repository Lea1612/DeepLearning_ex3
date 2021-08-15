from collections import Counter
import json

from torch import cuda
from load_data_A import DataLoaderA

UNKNOWN_WORD = 'UNK'

device = 'cuda' if cuda.is_available() else 'cpu'


class DataLoaderD(DataLoaderA):
    w_to_nb = {}
    voc = {}

    def __init__(self, path, word_dict=None, target_dict=None, is_train=False, variation='a'):
        self.sent_var = []
        self.w_sents = []
        self.sent_size = []
        self.w_size = []
        sentences, tags = self.data_loading(path)
        train = False
        if is_train and DataLoaderA.w_2_idx is None:
            DataLoaderA.variation = variation
            self.create_dictionaries(sentences, tags)
            train = True
            self.save_dict(word_dict, target_dict)

        self.X = self.sent_w_idx(train)
        self.X2 = self.sent_ch_idx()
        self.Y = self.tag_to_idx(tags)

    def data_loading(self, path):
        with open(path) as file:
            sent_temp = []
            tags_temp = []
            sent_ch_temp = []
            sentences = []
            sents_ch = []
            tags = []

            for line in file:
                try:
                    word, label = line.split()
                    sent_ch_temp.append(list(word))
                    sent_temp.append(word)
                    tags_temp.append(label)
                except ValueError:
                    self.w_size.append([len(row) for row in sent_ch_temp])
                    self.sent_size.append(len(sent_temp))
                    self.sent_var.append(sent_ch_temp)
                    self.w_sents.append(sent_temp)
                    sequence_sent_ch = ' '.join(set([c for row in sent_ch_temp for c in row]))
                    sequence_sent = ' '.join(sent_temp)
                    sequence_tag = ' '.join(tags_temp)
                    sentences.append(sequence_sent)
                    sents_ch.append(sequence_sent_ch)
                    tags.append(sequence_tag)

                    sent_temp = []
                    tags_temp = []
                    sent_ch_temp = []

        return (sentences, sents_ch), tags

    def create_dictionaries(self, sentences, targets):
        sen_temp = ' '.join(sentences[0]).split()
        sen_temp_char = ' '.join(sentences[1]).split()
        tar_temp = ' '.join(targets).split()

        DataLoaderA.voc = Counter(sen_temp)
        DataLoaderA.voc[UNKNOWN_WORD] = 999
        targets = set(tar_temp)

        DataLoaderA.w_2_idx = dict(
            zip(DataLoaderA.voc.keys(), range(1, len(DataLoaderA.voc) + 1)))
        DataLoaderA.w_2_idx['<PAD>'] = 0
        DataLoaderA.tag_2_idx = dict(zip(targets, range(1, len(targets) + 1)))
        DataLoaderA.tag_2_idx['<PAD>'] = 0
        DataLoaderA.idx_2_tag = {k: v for v, k in self.tag_2_idx.items()}

        DataLoaderD.voc = Counter(sen_temp_char)
        DataLoaderD.voc[UNKNOWN_WORD] = 999

        DataLoaderD.w_to_nb = dict(
            zip(DataLoaderD.voc.keys(), range(1, len(DataLoaderD.voc) + 1)))
        DataLoaderD.w_to_nb['<PAD>'] = 0

    def save_dict(self, word_dict, target_dict):
        with open(word_dict + "_1.json", 'w') as fp:
            json.dump(DataLoaderA.w_2_idx, fp)
        with open(word_dict + "_2.json", 'w') as fp:
            json.dump(DataLoaderD.w_to_nb, fp)
        with open(target_dict + ".json", 'w') as fp:
            json.dump(DataLoaderA.tag_2_idx, fp)

    def sent_ch_idx(self):
        idx_sents_ch = []
        for sent in self.sent_var:
            sent_temp = []
            for ch_list in sent:
                sent_temp.append([DataLoaderD.w_to_nb[ch] for ch in ch_list])
            idx_sents_ch.append(sent_temp)
        return idx_sents_ch

    def sent_w_idx(self, train):
        idx_sents_w = []
        for sent in self.w_sents:
            sent_temp = []
            for w in sent:
                if train and DataLoaderA.voc[w] < 5:
                    w = UNKNOWN_WORD
                elif not train and w not in DataLoaderA.w_2_idx.keys():
                    if w.lower() in DataLoaderA.w_2_idx.keys():
                        w = w.lower()
                    else:
                        w = UNKNOWN_WORD
                sent_temp.append(DataLoaderA.w_2_idx.get(w, DataLoaderA.w_2_idx[UNKNOWN_WORD]))
            idx_sents_w.append(sent_temp)
        return idx_sents_w

    def tag_to_idx(self, tags):
        idx_tags = []
        for tag in tags:
            tag_temp = []
            for t in tag.split():
                tag_temp.append(DataLoaderA.tag_2_idx.get(t))
            idx_tags.append(tag_temp)

        return idx_tags

    def __getitem__(self, idx):
        return self.X[idx], self.X2[idx], self.Y[idx], self.sent_size[idx], self.w_size[idx]

from torch import cuda
from load_data_A import DataLoaderA

UNKNOWN_WORD = 'UNK'

device = 'cuda' if cuda.is_available() else 'cpu'


class DataLoaderB(DataLoaderA):

    def __init__(self, path, word_dict=None, target_dict=None, is_train=False, variation='a'):
        self.sents_var = []
        self.w_size = []
        self.sent_size = []
        super(DataLoaderB, self).__init__(path, word_dict, target_dict, is_train, variation)

    def data_loading(self, path):
        with open(path) as file:
            temp_sentences_char = []
            sentences_char = []
            temp_targets = []
            targets = []

            for line in file:
                try:
                    word, label = line.split()
                    temp_sentences_char.append(list(word))
                    temp_targets.append(label)
                except ValueError:
                    self.w_size.append([len(row) for row in temp_sentences_char])
                    self.sent_size.append(len(temp_sentences_char))
                    self.sents_var.append(temp_sentences_char)
                    sequence_sen_char = ' '.join(set([c for row in temp_sentences_char for c in row]))
                    sequence_target = ' '.join(temp_targets)
                    sentences_char.append(sequence_sen_char)
                    targets.append(sequence_target)

                    temp_targets = []
                    temp_sentences_char = []

        return sentences_char, targets

    def sent_2_idx(self, sentences, train):
        id_sentences = []
        for sent in self.sents_var:
            sent_temp = []
            for ch_list in sent:
                sent_temp.append([DataLoaderA.w_2_idx[ch] for ch in ch_list])
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

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.sent_size[idx], self.w_size[idx]

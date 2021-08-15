from torch import cuda
from load_data_A import DataLoaderA

UNKNOWN_WORD = 'UNK'

device = 'cuda' if cuda.is_available() else 'cpu'


class DataLoaderC(DataLoaderA):

    def __init__(self, path, word_dict=None, target_dict=None, is_train=False, variation='a'):
        self.sentences_var = []
        super(DataLoaderC, self).__init__(path, word_dict, target_dict, is_train=is_train, variation=variation)

    def data_loading(self, path):
        with open(path) as file:
            sent_temp = []
            tags_temp = []
            sentences = []
            tags = []

            for line in file:
                try:
                    word, label = line.split()
                    sent_temp.append([word, word[:3], word[-3:]]), tags_temp.append(label)
                except ValueError:
                    self.sentences_var.append(sent_temp)
                    sequence_sent = ' '.join(set([word for row in sent_temp for word in row]))
                    sequence_tag = ' '.join(tags_temp)
                    sentences.append(sequence_sent)
                    tags.append(sequence_tag)

                    sent_temp = []
                    tags_temp = []

        return sentences, tags

    def sent_2_idx(self, sentences, train):
        id_sentences = []
        for sent in self.sentences_var:
            sent_temp = []
            for word, prefix, suffix in sent:
                if train and DataLoaderA.voc[word] < 5:
                    word = UNKNOWN_WORD
                elif not train and word not in DataLoaderA.w_2_idx.keys():
                    if word.lower() in DataLoaderA.w_2_idx.keys():
                        word = word.lower()
                    else:
                        word = UNKNOWN_WORD
                sent_temp.append([DataLoaderA.w_2_idx[word],
                                  DataLoaderA.w_2_idx.get(prefix, DataLoaderA.w_2_idx[UNKNOWN_WORD]),
                                  DataLoaderA.w_2_idx.get(suffix, DataLoaderA.w_2_idx[UNKNOWN_WORD])])
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

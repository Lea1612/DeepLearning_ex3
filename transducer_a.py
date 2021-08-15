import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch import cuda

device = 'cuda' if cuda.is_available() else 'cpu'


class TransducerA(nn.Module):
    def __init__(self, voc_length, tagset_size, corpus, bidirectional=True, padding_idx=None,
                 init_hyper_parameters=True, hidden_size=0, emb_dim=0, batch_size=0, is_predict=False):
        super(TransducerA, self).__init__()
        if init_hyper_parameters:
            self.init_hyper_parameters(corpus, is_predict)
        else:
            self.init_hyper_parameters_from_values(hidden_size, emb_dim, batch_size, corpus)

        self.embedded = nn.Embedding(voc_length, self.embedding_dim, padding_idx=padding_idx)
        self.hidden = self.hidden_initialization(self.batch_size)

        self.nb_directions = 2 if bidirectional else 1
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, bidirectional=bidirectional, num_layers=2, dropout=0.3)

        self.hidden_out = nn.Linear(self.hidden_dim * 2, tagset_size)
        self.softmax = nn.Softmax(dim=2)

    def init_hyper_parameters(self, corpus, is_predict):
        if corpus == 'pos':
            self.hidden_dim = 256
            self.embedding_dim = 2000
            self.batch_size = 100 if not is_predict else 1
            self.learning_rate = 0.0003
        else:
            self.hidden_dim = 250
            self.embedding_dim = 2000
            self.batch_size = 128 if not is_predict else 1
            self.learning_rate = 0.0007

    def init_hyper_parameters_from_values(self, hidden_dim, embedding_dim, batch_size, corpus):
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.learning_rate = 0.0003 if corpus == 'pos' else 0.0005

    def hidden_initialization(self, batch_size):
        return (torch.zeros(2, batch_size, self.hidden_dim).to(device),
                torch.zeros(2, batch_size, self.hidden_dim).to(device))

    def forward(self, sentence, word_len, soft_max=True):
        embs = self.embedded(sentence)
        embs = pack_padded_sequence(embs, word_len, batch_first=True, enforce_sorted=False)
        rnn_out, self.hidden = self.lstm(embs)
        rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True)
        output = self.hidden_out(rnn_out)

        if soft_max:
            output = self.softmax(output)

        output = output.view(sentence.shape[0], sentence.shape[1], -1)
        return output

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch import cuda

device = 'cuda' if cuda.is_available() else 'cpu'


class TransducerB(nn.Module):
    def __init__(self, voc_length, tagset_size, corpus, bidirectional=True, padding_idx=None,
                 init_hyper_parameters=True,
                 hidden_size=0, emb_dim=0, batch_size=0, btw_rnn=0, is_predict=False):

        super(TransducerB, self).__init__()
        if init_hyper_parameters:
            self.init_hyper_parameters(corpus, is_predict)
        else:
            self.init_hyper_parameters_from_values(hidden_size, emb_dim, batch_size, btw_rnn)

        self.embedded = nn.Embedding(voc_length, self.embedding_dim, padding_idx=padding_idx)
        self.hidden = self.hidden_initialization(self.batch_size)

        self.nb_directions = 2 if bidirectional else 1
        self.lstm = nn.LSTM(self.embedding_dim, self.btw_rnns)
        self.lstm_2 = nn.LSTM(self.btw_rnns, self.hidden_dim, bidirectional=bidirectional, num_layers=2)

        self.hidden_out = nn.Linear(self.hidden_dim * 2, tagset_size)
        self.softmax = nn.Softmax(dim=2)

    def init_hyper_parameters(self, corpus, is_predict):
        if corpus == 'pos':
            self.hidden_dim = 120
            self.embedding_dim = 80
            self.batch_size = 8 if not is_predict else 1
            self.learning_rate = 0.003
            self.btw_rnns = 200
        else:
            self.hidden_dim = 250
            self.embedding_dim = 80
            self.batch_size = 80 if not is_predict else 1
            self.learning_rate = 0.003
            self.btw_rnns = 300

    def init_hyper_parameters_from_values(self, hidden_dim, embedding_dim, batch_size, btw_rnn):
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.btw_rnns = btw_rnn
        self.learning_rate = 0.003

    def hidden_initialization(self, batch_size):
        return (torch.zeros(4, batch_size, self.hidden_dim).to(device),
                torch.zeros(4, batch_size, self.hidden_dim).to(device))

    def hidden_char_initialization(self, batch_size):
        return (torch.zeros(1, batch_size, self.btw_rnns).to(device),
                torch.zeros(1, batch_size, self.btw_rnns).to(device))

    def input_words_preparation(self, sentence, w_lens):
        inputs = sentence.view(sentence.shape[0] * sentence.shape[1], -1)
        batch_word_len = sentence.shape[0] * sentence.shape[1]

        w_lens, idx = w_lens.sort(dim=0, descending=True)
        inputs = inputs[idx]

        inputs = inputs[w_lens > 0]
        w_lens = w_lens[w_lens > 0]

        return inputs, batch_word_len, w_lens, idx

    def last_cell_LL(self, rnn_out, lengths, w_lens, batch_word_len, idx, sentence):
        last_cell = torch.cat([rnn_out[i, j.data - 1]
                               for i, j in enumerate(lengths)]) \
            .view(len(w_lens), self.btw_rnns)

        last_cell = torch.cat((last_cell,
                               torch.zeros(batch_word_len - len(w_lens),
                                           self.btw_rnns,
                                           device=device)))
        _, revers = idx.sort(0)
        last_cell = last_cell[revers]
        last_cell = last_cell.view(sentence.shape[0], sentence.shape[1], -1)

        return last_cell

    def forward(self, sentence, sentence_lens, w_lens, soft_max=True):
        # input
        inputs, batch_word_len, w_lens, idx = self.input_words_preparation(sentence, w_lens)

        # embedding
        embeds = self.embedded(inputs)
        embeds = pack_padded_sequence(embeds, w_lens, batch_first=True)

        # RNN
        rnn_out, _ = self.lstm(embeds, self.hidden_char_initialization(inputs.shape[0]))

        # Linear layer
        rnn_out, lengths = pad_packed_sequence(rnn_out, batch_first=True)

        last_cell = self.last_cell_LL(rnn_out, lengths, w_lens, batch_word_len, idx, sentence)

        sentence_lens, idx = sentence_lens.sort(dim=0, descending=True)
        last_cell = last_cell[idx]

        packed = pack_padded_sequence(last_cell, sentence_lens, batch_first=True)
        rnn_out2, _ = self.lstm_2(packed, self.hidden_initialization(sentence.shape[0]))
        rnn_out2, _ = pad_packed_sequence(rnn_out2, batch_first=True)

        _, revers = idx.sort(0)
        rnn_out2 = rnn_out2[revers]

        output = self.hidden_out(rnn_out2)
        # Softmax
        if soft_max:
            output = self.softmax(output)

        output = output.view(sentence.shape[0], sentence.shape[1], -1)

        return output

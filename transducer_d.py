import torch
from torch import nn
from torch import cuda

from transducer_a import TransducerA
from transducer_b import TransducerB

device = 'cuda' if cuda.is_available() else 'cpu'


class TransducerD(nn.Module):
    def __init__(self, voc_length, tagset_size, corpus, padding_idx=None, is_predict=False):
        super(TransducerD, self).__init__()
        self.init_hyper_parameters(corpus, is_predict)
        self.modelA = TransducerA(voc_length=voc_length[0],
                                  emb_dim=self.embedding_dim[0],
                                  hidden_size=self.hidden_dim[0],
                                  tagset_size=tagset_size,
                                  batch_size=self.batch_size,
                                  padding_idx=padding_idx,
                                  corpus=corpus,
                                  init_hyper_parameters=False).to(device)

        self.modelB = TransducerB(voc_length=voc_length[1],
                                  emb_dim=self.embedding_dim[1],
                                  hidden_size=self.hidden_dim[1],
                                  tagset_size=tagset_size,
                                  batch_size=self.batch_size,
                                  padding_idx=padding_idx,
                                  btw_rnn=self.btw_rnns,
                                  corpus=corpus,
                                  init_hyper_parameters=False).to(device)
        self.hidden_out = nn.Linear(tagset_size * 2, tagset_size)
        self.softmax = nn.Softmax(dim=2)

    def init_hyper_parameters(self, corpus, is_predict):
        if corpus == 'pos':
            self.hidden_dim = (256, 120)
            self.embedding_dim = (2000, 80)
            self.batch_size = 64 if not is_predict else 1
            self.learning_rate = 0.0003
            self.btw_rnns = 200
        else:
            self.hidden_dim = (200, 250)
            self.embedding_dim = (1000, 80)
            self.batch_size = 64 if not is_predict else 1
            self.learning_rate = 0.0003
            self.btw_rnns = 500

    def forward(self, sentence, sen_len, w_lens, soft_max=True):
        first_out = self.modelA(sentence[0], sen_len, False)
        second_out = self.modelB(sentence[1], sen_len, w_lens, False)
        output = torch.cat((first_out, second_out), dim=2)
        output = self.hidden_out(output)
        if soft_max:
            output = self.softmax(output)
        return output

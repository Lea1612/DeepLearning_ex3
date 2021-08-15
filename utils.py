import torch
from torch import cuda
from torch.nn.utils.rnn import pad_sequence
from load_data_A import DataLoaderA
import matplotlib.pyplot as plt


device = 'cuda' if cuda.is_available() else 'cpu'


def pad_collate(batch):
    (xx, yy) = zip(*batch)
    x_lens = [len(x) for x in xx]
    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0).to(device)
    yy_pad = pad_sequence(yy, batch_first=True, padding_value=0).to(device)

    return xx_pad, yy_pad, x_lens, None


def sent_padding(x, longest_sent):
    diff = longest_sent - len(x)
    return x + [0] * diff


def longest_word_sent_zeros(w_size, sent_size, batch):
    longest_word = max([j for i in w_size for j in i])
    longest_sent = max(sent_size)
    word = torch.zeros((len(batch), longest_sent, longest_word), dtype=torch.long).to(device=device)

    return longest_word, longest_sent, word

def pad_part_2(batch):
    X, X2, Y, sent_size, w_size = zip(*batch)
    longest_word, longest_sent, word = longest_word_sent_zeros(w_size, sent_size, batch)

    sentence = torch.zeros((len(batch), longest_sent), dtype=torch.long).to(device=device)
    yy_pad = torch.zeros((len(batch), longest_sent), dtype=torch.long).to(device=device)
    w_size = sum([sent_padding(i, longest_sent) for i in w_size], [])
    for i, item in enumerate(zip(X, X2, Y)):
        sen, w, y = item
        w = [sent_padding(i, longest_word) for i in w]
        word[i, :sent_size[i], :longest_word] = torch.Tensor(w)
        sentence[i, :sent_size[i]] = torch.Tensor(sen)
        yy_pad[i, :sent_size[i]] = torch.Tensor(y)

    return (sentence, word), yy_pad, torch.tensor(sent_size), torch.tensor(w_size)


def pad_part_3(batch):
    X, Y, sent_size, w_size = zip(*batch)
    longest_word, longest_sent, word = longest_word_sent_zeros(w_size, sent_size, batch)

    yy_pad = torch.zeros((len(batch), longest_sent), dtype=torch.long).to(device=device)
    w_size = sum([sent_padding(i, longest_sent) for i in w_size], [])
    for i, item in enumerate(zip(X, Y)):
        w, y = item
        w = [sent_padding(i, longest_word) for i in w]
        word[i, :sent_size[i], :longest_word] = torch.Tensor(w)
        yy_pad[i, :sent_size[i]] = torch.Tensor(y)

    return word, yy_pad, torch.tensor(sent_size), torch.tensor(w_size)


def create_word_lens(xx_pad):
    word_lens = []
    for k in range(xx_pad.shape[0]):
        for i in range(xx_pad.shape[1]):
            for j in range(xx_pad.shape[2]):
                if xx_pad[k][i][j] == 0:
                    word_lens.append(j)
                    break
            else:
                word_lens.append(j + 1)
    return word_lens


def pad_collate_sorted(batch):
    if DataLoaderA.variation == 'd':
        return pad_part_2(batch)
    if DataLoaderA.variation == 'b':
        return pad_part_3(batch)
    (xx, yy) = zip(*batch)
    if xx[0].dim() == 1 or DataLoaderA.variation == 'c':
        return pad_collate(batch)

    x_lens = [len(x) for x in xx]
    shape_0 = max([item.shape[0] for item in xx])
    shape_1 = max([item.shape[1] for item in xx])
    xx_pad = torch.zeros(len(batch), shape_0, shape_1, dtype=torch.long).to(device)
    yy_pad = torch.zeros(len(batch), shape_0, dtype=torch.long).to(device)
    for i, item in enumerate(zip(xx, yy)):
        x, y = item
        xx_pad[i, :x.shape[0], :x.shape[1]] = x
        yy_pad[i, :x.shape[0]] = y

    word_lens = create_word_lens(xx_pad)

    return xx_pad, yy_pad, torch.tensor(x_lens).to(device), torch.tensor(word_lens).to(device)


def save_graph(train_graph, test_graph, y_axis):
    plt.suptitle(y_axis, fontsize=20)
    plt.figure()
    plt.plot(train_graph, color='r', label='train')
    plt.plot(test_graph, color='g', label='test')
    plt.xlabel('Epochs')
    plt.legend(loc="upper left")
    plt.ylabel(y_axis)
    plt.savefig(y_axis + '.png')
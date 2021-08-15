import json
from sys import argv

import torch
from torch import cuda
from transducer_a import TransducerA
from transducer_b import TransducerB
from transducer_c import TransducerC
from transducer_d import TransducerD

device = 'cuda' if cuda.is_available() else 'cpu'
UNKNOWN_WORD = 'UNK'


def json_load(item):
    return json.load(open(item))


def dict_loading(w_dict, target_dict):
    char_dict = None
    if isinstance(w_dict, tuple):
        w_dict = json_load(w_dict[0])
        char_dict = json_load(w_dict[1])
    else:
        w_dict = json_load(w_dict)

    target_dict = json_load(target_dict)

    items = {v: k for k, v in target_dict.items()}

    return w_dict, char_dict, target_dict, items


def add_padding_zero(sent, longest_sent):
    size_diff = longest_sent - len(sent)
    return sent + [0] * size_diff


def repr_b_prep(x_batch):
    len_x = torch.tensor([len(x_batch)])
    lane_size = torch.tensor([len(i) for i in x_batch])
    max_size = max(lane_size).item()
    x_batch = torch.tensor([add_padding_zero(x, max_size) for x in x_batch]).to(device)
    prediction = model(x_batch.unsqueeze(0), len_x, lane_size).view(-1, tag_size)

    return len_x, lane_size, max_size, x_batch, prediction


def repr_d_prep(x_batch):
    ch_sentence = x_batch[1]
    len_x = torch.tensor([len(ch_sentence)])
    lane_size = torch.tensor([len(i) for i in ch_sentence])
    max_size = max(lane_size).item()
    ch_sentence = torch.tensor([add_padding_zero(x, max_size) for x in ch_sentence]).to(device)
    x_batch = (x_batch[0].unsqueeze(0), ch_sentence.unsqueeze(0))
    probs = model(x_batch, len_x, lane_size).view(-1, tag_size)

    return ch_sentence, len_x, lane_size, max_size, x_batch, probs


def predict(model, test, path, corpus):
    model.eval()
    predicted = []
    if repr != 'd':
        test = test[0]
    else:
        test = zip(dataset[0], dataset[1])

    with torch.no_grad():
        for x_batch in test:
            if repr == 'b' and repr is not None:
                len_x, lane_size, max_size, x_batch, prediction = repr_b_prep(x_batch)

            elif repr == 'd' and repr is not None:
                ch_sentence, len_x, lane_size, max_size, x_batch, prediction = repr_d_prep(x_batch)

            else:
                prediction = model(x_batch.unsqueeze(0), torch.tensor([x_batch.shape[0]])).view(-1, tag_size)
            predicted.append([index_to_label[i.item()] for i in prediction.argmax(dim=1)] + [''])
    i = 0
    line = 0
    write_prediction_file(path, predicted, line, i, corpus)


def write_prediction_file(path, predicted, line, i, corpus):
    with open('test4.' + corpus, 'w') as file_pred:
        with open(path) as f:
            for word in f:
                if word == '\n':
                    file_pred.write(word)
                    i = 0
                    line += 1
                else:
                    word = word.strip()
                    label = predicted[line][i]
                    i += 1
                    file_pred.write(f"{word}\t{label}\n")


def load_data(path):
    with open(path) as file:
        temp_sentences = []
        sentences = []
        temp_characters = []
        characters = []
        unk = word_dict[UNKNOWN_WORD]
        unk_c = char_dict[UNKNOWN_WORD] if repr == 'd' else None
        if repr == 'a':
            load_data_transducer_a(file, unk, sentences, temp_sentences)
        elif repr == 'b':
            load_data_transducer_b(file, unk, sentences, temp_sentences)
        elif repr == 'c':
            load_data_transducer_c(file, unk, sentences, temp_sentences)
        else:
            load_data_transducer_d(file, unk, unk_c, sentences, characters, temp_sentences, temp_characters)

    return sentences, characters


def load_data_transducer_a(file, unk, sentences, temp_sentences):
    for word in file:
        if word != '\n':
            word = word.strip()
            if word not in word_dict.keys():
                word = word_dict.get(word.lower(), unk)
            else:
                word = word_dict[word]
            temp_sentences.append(word)
        else:
            sentences.append(torch.tensor(temp_sentences).to(device))
            temp_sentences = []


def load_data_transducer_b(file, unk, sentences, temp_sentences):
    for word in file:
        if word != '\n':
            word = word.strip()
            temp_sentences.append([word_dict.get(ch, unk) for ch in list(word.strip())])
        else:
            sentences.append(temp_sentences)
            temp_sentences = []


def load_data_transducer_c(file, unk, sentences, temp_sentences):
    for word in file:
        if word != '\n':
            word = word.strip()
            if word not in word_dict.keys():
                word_new = word_dict.get(word.lower(), unk)
            else:
                word_new = word_dict[word]
            res = [word_new,
                   word_dict.get(word[:3], unk),
                   word_dict.get(word[-3:], unk)]
            temp_sentences.append(res)
        else:
            sentences.append(torch.tensor(temp_sentences).to(device))
            temp_sentences = []


def load_data_transducer_d(file, unk, unk_c, sentences, characters, temp_sentences, temp_characters):
    for word in file:
        if word != '\n':
            word = word.strip()
            ch = list(word)
            if word not in word_dict.keys():
                word = word_dict.get(word.lower(), unk)
            else:
                word = word_dict[word]
            temp_sentences.append(word)
            temp_characters.append([char_dict.get(c, unk_c) for c in ch])
        else:
            sentences.append(torch.tensor(temp_sentences).to(device))
            characters.append(temp_characters)
            temp_sentences = []
            temp_characters = []


if __name__ == '__main__':
    repr = argv[1]
    model_file = argv[2]
    test_file = argv[3]
    corpus = argv[4]
    target_dict = argv[5]
    word_dict = argv[6]
    word_dict_2 = argv[7] if len(argv) == 8 else None

    if word_dict_2 is not None:
        word_dict = (word_dict, word_dict_2)

    word_dict, char_dict, label_to_index, index_to_label = dict_loading(word_dict, target_dict)
    dataset = load_data(test_file)
    tag_size = len(label_to_index)

    if repr == 'a':
        model = TransducerA(voc_length=len(word_dict),
                            tagset_size=tag_size,
                            corpus=corpus,
                            padding_idx=0,
                            is_predict=True).to(device)

    elif repr == 'b':
        model = TransducerB(voc_length=len(word_dict),
                            tagset_size=tag_size,
                            corpus=corpus,
                            padding_idx=0,
                            is_predict=True).to(device)

    elif repr == 'c':
        model = TransducerC(voc_length=len(word_dict),
                            tagset_size=tag_size,
                            corpus=corpus,
                            padding_idx=0,
                            is_predict=True).to(device)

    else:
        model = TransducerD(voc_length=(len(word_dict), len(char_dict)),
                            tagset_size=tag_size,
                            corpus=corpus,
                            padding_idx=0,
                            is_predict=True).to(device)

    model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
    predict(model, dataset, test_file, corpus)

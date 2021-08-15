from sys import argv
import torch
from torch import cuda, nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from utils import pad_collate_sorted
from load_data_A import DataLoaderA
from load_data_C import DataLoaderC
from load_data_B import DataLoaderB
from load_data_D import DataLoaderD
from transducer_a import TransducerA
from transducer_b import TransducerB
from transducer_c import TransducerC
from transducer_d import TransducerD

device = 'cuda' if cuda.is_available() else 'cpu'
SENTENCES_TO_PRINT = 500


def save_evaluation_results(result_type, results):
    with open(f"{repr}_{corpus}_{result_type}", 'w') as file:
        file.write('\n'.join(results))


def evaluate(model, test_loader, corpus, criterion):
    correct = 0
    total = 0
    loss = 0
    tag_pad_id = DataLoaderA.tag_2_idx['<PAD>']
    model.eval()
    with torch.no_grad():
        for x_test, y_test, len_x, lane_size in test_loader:
            labels = y_test.view(-1)
            if lane_size is not None:
                y_pred = model(x_test, len_x, lane_size).view(-1, tag_size)
            else:
                y_pred = model(x_test, len_x).view(-1, tag_size)
            _, predicted = torch.max(y_pred.data, 1)

            mask = get_mask(corpus, labels, predicted, tag_pad_id)
            predicted = predicted[mask]
            labels = labels[mask]
            y_pred = y_pred[mask]

            if labels.size(0) != 0:
                loss += criterion(y_pred, labels)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
    print(f'Acc:{100 * correct / total:.2f} Loss:{loss / total:}')
    return 100 * correct / total, loss / total


def get_mask(corpus, labels, predicted, tag_pad_token):
    if corpus == 'pos':
        mask = (labels > tag_pad_token)
    else:
        mask = (
                (labels > tag_pad_token) &
                (
                    ~((predicted == DataLoaderA.tag_2_idx['O']) & (labels == DataLoaderA.tag_2_idx['O']))
                )
        )
    return mask


def weight_loss_prep():
    weight = [0.3 if k == 'O' else 1 for k, v in
              {k: v for k, v in sorted(DataLoaderA.tag_2_idx.items(),
                                       key=lambda item: item[1])}.items()]

    criterion = nn.CrossEntropyLoss(torch.FloatTensor(weight).to(device),
                                    ignore_index=DataLoaderA.tag_2_idx['<PAD>'])

    return weight, criterion


def train(model, training, test, learning_rate, epoch, corpus):
    loss_dev = []
    accuracy_dev = []

    if corpus == 'ner':
        weight, criterion = weight_loss_prep()

    else:
        criterion = nn.CrossEntropyLoss(ignore_index=DataLoaderA.tag_2_idx['<PAD>'])

    optimizer = Adam(model.parameters(), lr=learning_rate)
    loss_list = []
    for i in range(epoch):
        trained_sentences = 0
        print(f"Epoch number: {i + 1}")
        for x_batch, y_batch, len_x, lane_size in training:
            trained_sentences += int(y_batch.shape[0])
            y_batch = y_batch.view(-1)

            model.train()
            optimizer.zero_grad()
            if lane_size is not None:
                y_predict = model(x_batch, len_x, w_lens=lane_size, soft_max=False).view(-1, tag_size)
            else:
                y_predict = model(x_batch, len_x, soft_max=False).view(-1, tag_size)
            loss = criterion(y_predict, y_batch)
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())

            if trained_sentences > SENTENCES_TO_PRINT:
                trained_sentences = trained_sentences % SENTENCES_TO_PRINT
                acc, loss = evaluate(model, test, corpus, criterion)
                loss_dev.append(loss)
                accuracy_dev.append(acc)

    save_evaluation_results('loss', [str(i.item()) for i in loss_dev])
    save_evaluation_results('acc', [str(i) for i in accuracy_dev])
    torch.save(model.state_dict(), model_file)


if __name__ == '__main__':
    repr = argv[1]
    train_file = argv[2]
    model_file = argv[3]
    corpus = argv[4]
    dev_file = argv[5]
    target_dict = argv[6]
    word_dict = argv[7]

    if repr == 'a':
        train_dataset = DataLoaderA(path=train_file,
                                    variation=repr,
                                    word_dict=word_dict,
                                    target_dict=target_dict,
                                    is_train=True)

        dev_dataset = DataLoaderA(dev_file, repr)
        tag_size = len(DataLoaderA.tag_2_idx)

        model = TransducerA(voc_length=len(DataLoaderA.w_2_idx),
                            tagset_size=tag_size,
                            corpus=corpus,
                            padding_idx=0).to(device)

    elif repr == 'b':
        train_dataset = DataLoaderB(path=train_file,
                                    variation=repr,
                                    word_dict=word_dict,
                                    target_dict=target_dict,
                                    is_train=True)

        dev_dataset = DataLoaderB(dev_file, repr)
        tag_size = len(DataLoaderB.tag_2_idx)

        model = TransducerB(voc_length=len(DataLoaderA.w_2_idx),
                            tagset_size=tag_size,
                            corpus=corpus,
                            padding_idx=0).to(device)

    elif repr == 'c':
        train_dataset = DataLoaderC(path=train_file,
                                    variation=repr,
                                    word_dict=word_dict,
                                    target_dict=target_dict,
                                    is_train=True)

        dev_dataset = DataLoaderC(dev_file, repr)
        tag_size = len(DataLoaderC.tag_2_idx)

        model = TransducerC(voc_length=len(DataLoaderA.w_2_idx),
                            tagset_size=tag_size,
                            corpus=corpus,
                            padding_idx=0).to(device)

    else:
        train_dataset = DataLoaderD(path=train_file,
                                    variation=repr,
                                    word_dict=word_dict,
                                    target_dict=target_dict,
                                    is_train=True)

        dev_dataset = DataLoaderD(dev_file, repr)
        tag_size = len(DataLoaderD.tag_2_idx)

        model = TransducerD(voc_length=(len(DataLoaderA.w_2_idx), len(DataLoaderD.w_to_nb)),
                            tagset_size=tag_size,
                            corpus=corpus,
                            padding_idx=0).to(device)

    train_set = DataLoader(train_dataset, batch_size=model.batch_size, shuffle=True, collate_fn=pad_collate_sorted)
    dev_set = DataLoader(dev_dataset, batch_size=5, shuffle=False, collate_fn=pad_collate_sorted)
    train(model, train_set, dev_set, learning_rate=model.learning_rate, epoch=5, corpus=corpus)

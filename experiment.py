import random
import sys
from datetime import datetime

import torch
from torch import nn
from torch import cuda
from torch.optim import Adam
from utils import save_graph

device = 'cuda' if cuda.is_available() else 'cpu'


class Acceptor(nn.Module):
    def __init__(self, output_size, emb_length, hidden_lstm, emb_dim, hidden_out=5):
        super().__init__()
        self.hidden_dim = hidden_lstm
        self.emb_size = emb_dim
        self.embedded = nn.Embedding(emb_length, emb_dim)
        self.input_hidden = self.hidden_initialization()
        self.lstm = nn.LSTM(emb_dim, hidden_lstm)
        self.hidden = nn.Linear(hidden_lstm, hidden_out)
        self.tanh = nn.Tanh()
        self.output = nn.Linear(hidden_out, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def hidden_initialization(self):
        return (torch.zeros(1, 1, self.hidden_dim).to(device),
                torch.zeros(1, 1, self.hidden_dim).to(device))

    def forward(self, x, soft_max=True):
        out = self.embedded(x).view(x.shape[1], -1, self.emb_size)
        out, self.input_hidden = self.lstm(out, self.input_hidden)
        out = self.hidden(out.view(x.shape[1], 1, -1)[-1])
        out = self.tanh(out)
        out = self.output(out)
        if soft_max:
            out = self.softmax(out)
        return out


word_id = {'a': 0, 'b': 1, 'c': 2, 'd': 3, '1': 4, '2': 5, '3': 6, '4': 7, '5': 8, '6': 9, '7': 10, '8': 11, '9': 12}


def word_2_id(line):
    return [word_id.get(word) for word in line]


def load_data(pos_file, neg_file):
    positive_samples = file_2_id(pos_file)
    negative_samples = file_2_id(neg_file)
    label = get_label(positive_samples, negative_samples)
    all_data = positive_samples + negative_samples

    data_label_zip = list(zip(all_data, label))
    random.shuffle(data_label_zip)
    split = int(len(all_data) * 0.8)
    return data_label_zip[:split], data_label_zip[split:]


def file_2_id(input_file):
    examples_2_id = []
    for line in open(input_file).read().split('\n'):
        examples_2_id.append(torch.LongTensor(word_2_id(line)).unsqueeze(0))
    return examples_2_id


def get_label(pos_examples, neg_examples):
    return torch.LongTensor([[1]] * len(pos_examples) + [[0]] * len(neg_examples))


def train_acceptor(model, train_data, dev_data, lr=0.01, epoch=30):
    train_accuracies = []
    train_losses = []
    dev_accuracies = []
    dev_losses = []

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    for t in range(epoch):
        model.train()
        random.shuffle(train_data)
        for x, y in train_data:
            model.input_hidden = (model.input_hidden[0].detach(), model.input_hidden[1].detach())
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x, soft_max=False)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        dev_accuracy, dev_loss = evaluate(model, dev_data, criterion)
        train_accuracy, train_loss = evaluate(model, train_data, criterion)

        train_accuracies.append(train_accuracy)
        train_losses.append(train_loss)
        dev_accuracies.append(dev_accuracy)
        dev_losses.append(dev_loss)
        print(
            f"Epoch:{t + 1}\n Train Accuracy:{train_accuracy:.2f} Loss:{train_loss:.4f}\n Dev Accuracy: {dev_accuracy:.2f} Loss:{dev_loss:.4f} ")
        if dev_accuracy == 100:
            print("Stop training reached the maximum accuracy on dev")
            break
    save_graph(train_accuracies, dev_accuracies, 'Accuracy')
    save_graph(train_losses, dev_losses, 'Loss')
    return acceptor


def evaluate(model, data, criterion):
    loss_list = []
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for x_batch, y_batch in data:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = model(x_batch)
            loss_list.append(criterion(outputs, y_batch).item())

            total += len(y_batch)
            correct += (outputs.argmax(axis=1) == y_batch).sum().item()

    return 100 * correct / total, sum(loss_list) / total


if __name__ == '__main__':
    pos = sys.argv[1]
    neg = sys.argv[2]
    acceptor = Acceptor(output_size=2,
                        emb_length=len(word_id),
                        emb_dim=len(word_id),
                        hidden_lstm=8,
                        hidden_out=2).to(device)

    train, test = load_data(pos, neg)

    start = datetime.now()
    acceptor = train_acceptor(acceptor, train, test, epoch=50, lr=0.001)

    print(f"End: {datetime.now() - start}")

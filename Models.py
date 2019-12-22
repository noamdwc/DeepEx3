import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader, random_split, TensorDataset
from utils import get_vocab, word2idx_list


def split_data(X, Y, sizes):
    X, Y = np.asarray(X), np.asarray(Y)
    X, Y = torch.from_numpy(X), torch.from_numpy(Y)
    X, Y = X.long(), Y.long()
    return random_split(TensorDataset(X, Y), sizes)


def get_defult_device():
    '''pick gpu if avilable, otherwise chose cpu'''
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def to_device(data, device):
    '''move tensor\s to device'''
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader():
    '''Wrap the dataloader to move data to device'''

    def __init__(self, dataLoader, device=None):
        self.dl = dataLoader
        if device == None:
            self.device = get_defult_device()
        else:
            self.device = device

    def __iter__(self):
        '''Yield the data after moving it to the device'''
        for d in self.dl:
            yield to_device(d, self.device)

    def __len__(self):
        '''len of the dataloader'''
        return len(self.dl)


class Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, in_dim, hid_dim, out_dim=2):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, in_dim)
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Sequential(nn.Linear(in_dim, hid_dim), nn.ReLU(), self.dropout, nn.Linear(hid_dim, out_dim))

    def forward(self, x):
        x = self.embedding(x)
        x, (final_hidden_state, final_cell_state) = self.lstm(x)
        x = x[:, -1, :]
        # x = self.dropout(x)
        x = self.linear(x)
        return F.log_softmax(x, dim=1)

    def train_model(self, train_data_loader, dev_data_loader, epoch_num, lr=0.001):
        self.train()
        train_data_loader = DeviceDataLoader(train_data_loader)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        loss_func = nn.CrossEntropyLoss()
        for epoch in range(epoch_num):
            succ = 0.0
            sum_loss = 0.0
            data_len = 0
            # print("EPOCH: ", epoch)
            for i, batch in enumerate(train_data_loader):
                if i % 100 == 0:
                    print(i)
                data, labels = batch
                if torch.cuda.is_available():
                    data, labels = data.cuda(), labels.cuda()
                pre = self(data)
                label = labels
                loss = loss_func(pre, label)
                opt.zero_grad()
                loss.backward()
                opt.step()
                sum_loss += loss
                data_len += len(label)
                succ += torch.sum(torch.argmax(pre, dim=1) == label)
            if epoch % 1 == 0:
                print("EVALUATING")
                self.train(False)
                self.evaluate_model(dev_data_loader)
                self.train()

            self.train(False)

    def evaluate_model(self, dev_data_loader):
        self.eval()
        succ = 0.0
        sum_loss = 0.0
        data_len = 0
        loss_func = nn.CrossEntropyLoss()
        for i, batch in enumerate(dev_data_loader):
            data, labels = batch
            if torch.cuda.is_available():
                data, labels = data.cuda(), labels.cuda()
            pre = self(data)
            label = labels
            loss = loss_func(pre, label)
            sum_loss += loss
            data_len += len(label)
            succ += torch.sum(torch.argmax(pre, dim=1) == label)
        print("evaluation average loss: {}, succeed rate: {}%".format(sum_loss,
                                                                      100 * succ / data_len))


if __name__ == '__main__':
    model = Model(len(get_vocab()), 50, 64, 32, 2)
    X = []
    with open("good.txt") as file:
        x = file.readline()
        x = x[:-1]
        x = word2idx_list(x)
        X.append(x)
    file.close()
    X = np.asarray(X)
    X = torch.from_numpy(X)
    X = X.long()
    Y = torch.ones(X.shape)
    Y = Y.long()
    d = DataLoader(TensorDataset(X, Y), 1, shuffle=True, num_workers=1)
    for batch in d:
        x, _ = batch
        pre = model(x)
        print(torch.argmax(pre))

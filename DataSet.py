import torch
from torch.utils.data import DataLoader, random_split, TensorDataset, Dataset
import numpy as np
from utils import word2idx_list


class sentsDataset(Dataset):
    def __init__(self, X, Y):
        super(sentsDataset, self).__init__()
        self.X = X
        self.Y = Y

    def __getitem__(self, index):
        x = self.X[index]
        y = self.Y[index]
        x, y = torch.tensor(x), torch.tensor(y)
        x, y = x.long(), y.long()
        return x, y

    def __len__(self):
        return len(self.X)


class wordsDataset(Dataset):
    def __init__(self, X, Y):
        super(wordsDataset, self).__init__()
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        x, y = self.X[item], self.Y[item]
        return torch.tensor(x).long(), torch.tensor(y).long()


class DataBuilder():
    def __init__(self, convertor):
        self.pos = []
        self.neg = []
        self.convertor = convertor

    def __len__(self):
        return len(self.pos) + len(self.neg)

    def add_pos(self, words):
        self.pos += words
        return self

    def add_neg(self, words):
        self.neg += words
        return self

    def _read(self, file_path):
        words = []
        with open(file_path) as file:
            line = file.readline()
            while line:
                line = line[:-1]
                words.append(line)
                line = file.readline()
        file.close()
        return words

    def read_pos(self, file_path):
        self.pos += self._read(file_path)
        return self

    def read_neg(self, file_path):
        self.neg += self._read(file_path)
        return self

    def _write(self, path_file, words):
        with open(path_file, "w") as file:
            for word in words:
                file.write(word + "\n")
        file.close()

    def save(self, pos_path, neg_path):
        self._write(pos_path, self.pos)
        self._write(neg_path, self.neg)

    def build_data(self, train_procent, batch_size=1):
        X = self.pos + self.neg
        Y = [np.asarray(1) for _ in range(len(self.pos))] \
            + [np.asarray(0) for _ in range(len(self.neg))]
        X = [np.asarray(self.convertor(x)) for x in X]
        dataset = wordsDataset(X, Y)
        train_size = int(len(dataset) * train_procent)
        train_set, dev_set = random_split(dataset, (train_size, len(dataset) - train_size))
        return DataLoader(train_set, batch_size=batch_size, shuffle=True), DataLoader(dev_set, batch_size=batch_size,
                                                                                      shuffle=True)


if __name__ == '__main__':
    m = DataBuilder(word2idx_list)
    m.read_neg("bad.txt")
    m.read_pos("good.txt")
    m.build_data(0.8)

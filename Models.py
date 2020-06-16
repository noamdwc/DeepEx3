import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader, random_split, TensorDataset
from utils import get_vocab, word2idx_list, ner_tags
import math

torch.manual_seed(1)


def create_dataLoader(parser, batch_size, num_workers=0, shuffle=True):
    dataset = parser.get_data()
    return DataLoader(dataset, shuffle=shuffle, num_workers=num_workers)


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


def print_records(record_train, record_dev):
    print("SUMMERY:")
    print("\tsuccess rate:")
    print("\t\ttrain:")
    for i, succ_rate in enumerate(record_dev):
        print("\t\t", i, succ_rate[0].item(), sep="\t")
    print("\t\tdev:")
    for i, succ_rate in enumerate(record_dev):
        print("\t\t", i, succ_rate[0].item(), sep="\t")


class Part1Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, in_dim, hid_dim, out_dim=2):
        super(Part1Model, self).__init__()
        self.embedding_dim = embedding_dim
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        # init it to one hot vectors:
        one_hots = torch.zeros((vocab_size, embedding_dim))
        counter = 0
        for i in range(vocab_size):
            one_hots[i][counter] = 1
            counter += 1
        self.embed.from_pretrained(one_hots, freeze=False)
        nn.init.uniform_(self.embed.weight, -1.0, 1.0)
        self.in_dim = in_dim
        self.lstm = nn.LSTM(embedding_dim, in_dim, batch_first=True)
        # self.dropout = nn.Dropout(0.2)
        self.in_linear = nn.Linear(in_dim, hid_dim)
        self.act_func = torch.tanh
        self.out_linear = nn.Linear(hid_dim, out_dim)
        self.softmax = nn.Softmax(dim=1)
        # self.hidden = self.init_hidden(1)

    def to_cuda(self):
        if torch.cuda.is_available():
            self.embed = self.embed.cuda()
            self.lstm = self.lstm.cuda()
            self.in_linear = self.in_linear.cuda()
            self.out_linear = self.out_linear.cuda()

    def forward(self, x):
        if x.device == torch.device('cpu'):
            if torch.cuda.is_available():
                x = x.cuda()
        x = self.embed(x)
        x, (__, _) = self.lstm(x)
        x, _ = torch.max(x, dim=1)
        x = self.in_linear(x)
        x = self.act_func(x)
        # x = self.dropout(x)
        x = self.out_linear(x)
        x = F.log_softmax(x, dim=1)
        return x

    def train_model(self, train_data_loader, dev_data_loader, epoch_num, lr=0.001, batch_size=1):
        self.train(True)
        record_dev = []
        record_train = []
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        loss_func = nn.CrossEntropyLoss()
        for epoch in range(epoch_num):
            print("EPOCH: ", epoch)
            for i, batch in enumerate(train_data_loader):
                opt.zero_grad()
                x, y = batch
                if torch.cuda.is_available():
                    x, y = x.cuda(), y.cuda()
                pre = self(x)
                pre = pre.view(1, -1)
                loss = loss_func(pre, y)
                loss.backward()
                opt.step()
            if epoch % 1 == 0:
                print("EVALUATING DEV:")
                record_dev.append(self.evaluate_model(dev_data_loader))
                self.train()

        self.train(False)
        print_records(record_train, record_dev)

    def evaluate_model(self, dev_data_loader):
        self.eval()
        succ = 0.0
        sum_loss = 0.0
        loss_func = nn.CrossEntropyLoss()
        data_len = 0
        for i, batch in enumerate(dev_data_loader):
            data, label = batch
            if torch.cuda.is_available():
                data, label = data.cuda(), label.cuda()
            pre = self(data)
            pre = pre.view(1, -1)
            loss = loss_func(pre, label)
            sum_loss += loss
            data_len += len(label)
            succ += torch.sum(torch.argmax(pre, dim=1) == label)
        print("evaluation  succeed rate: {}%".format(100 * succ / data_len))
        return 100 * succ / data_len, sum_loss / len(dev_data_loader)


if __name__ == '__main__':
    model = Part1Model(len(get_vocab()), 50, 64, 32, 2)
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


class LSTMTagger(nn.Module):
    def __init__(self, embedding_dim, in_dim, hidden_dim, vocab_size, tagset_size, c=None, b=None,
                 d=False):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.d = d
        if self.d:
            self.lstm1 = nn.LSTM(embedding_dim * 2, hidden_dim // 2, bidirectional=True, num_layers=2)
        else:
            self.lstm1 = nn.LSTM(embedding_dim, hidden_dim // 2, bidirectional=True, num_layers=2)

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.sub_word = c != None
        if self.sub_word:
            w2starts, starts_size, w2ends, ends_size = c
            self.w2starts = w2starts
            self.w2ends = w2ends
            self.starts_embedding = nn.Embedding(starts_size, embedding_dim)
            self.ends_embedding = nn.Embedding(ends_size, embedding_dim)
        self.char_em = b != None or d
        if self.char_em:
            alphabet_size, car_embedding_dim, I2F, C2I = b
            self.car_embedding = nn.Embedding(alphabet_size, car_embedding_dim)
            self.lstm_car = nn.LSTM(car_embedding_dim, embedding_dim)
            self.C2I = C2I
            self.I2F = I2F
            self.hidden = self.init_hidden(hidden_dim)
            self.hidden_car = self.init_hidden(car_embedding_dim)
            self.hidden_dim = hidden_dim
            self.car_embedding_dim = car_embedding_dim
        self.softmax = nn.LogSoftmax(dim=1)
        self.to_cuda()


    def to_cuda(self):
        if torch.cuda.is_available():
            self.softmax = self.softmax.cuda()
            self.word_embeddings = self.word_embeddings.cuda()
            self.lstm1 = self.lstm1.cuda()
            if self.sub_word:
                self.starts_embedding = self.starts_embedding.cuda()
                self.ends_embedding = self.ends_embedding.cuda()
            if self.char_em:
                self.car_embedding = self.car_embedding.cuda()
                self.lstm_car = self.lstm_car.cuda()

    def init_hidden(self, dim):
        return (autograd.Variable(torch.zeros(1, 1, dim)),
                autograd.Variable(torch.zeros(1, 1, dim)))

    def forward(self, sentence):
        if sentence.device == torch.device('cpu'):
            if torch.cuda.is_available():
                sentence = sentence.cuda()
        if self.sub_word:
            x = sentence
            x_copy = x.clone().detach()
            if torch.cuda.is_available():
                x_copy = x_copy.cpu()
            x_copy = x_copy.numpy().tolist()
            x_start = torch.tensor([self.w2starts[idx] for idx in x_copy])
            x_end = torch.tensor([self.w2ends[idx] for idx in x_copy])
            if torch.cuda.is_available():
                x_start, x_end = x_start.cuda(), x_end.cuda()
            embeds = self.word_embeddings(x) + self.starts_embedding(x_start) + self.ends_embedding(x_end)
        else:
            embeds = self.word_embeddings(sentence)
        if self.char_em:
            lstm_car_result = []
            for word in sentence:
                self.hidden_car = self.init_hidden(self.car_embedding_dim)
                word = self.I2F[word.item()]
                word = torch.tensor([self.C2I[car] for car in word]).long()
                if torch.cuda.is_available():
                    word = word.cuda()
                car_embeds = self.car_embedding(word)
                lstm_car_out, self.hidden_car = self.lstm_car(car_embeds.view(len(word), 1, self.car_embedding_dim))
                lstm_car_result.append(lstm_car_out[-1])
            if self.d:
                lstm_car_result = torch.stack(lstm_car_result)
                embeds = torch.cat((lstm_car_result.view(1, len(sentence), -1)[0], embeds), dim=1)
            else:
                embeds = torch.stack(lstm_car_result)

        lstm_out, _ = self.lstm1(embeds.view(len(sentence), 1, -1))
        # lstm_out, _ = self.lstm2(lstm_out)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = self.softmax(tag_space)
        return tag_scores


def evaluate_model(model, dev_data_loader):
    model.eval()
    succ = 0.0
    sum_loss = 0.0
    loss_func = nn.CrossEntropyLoss()
    data_len = 0
    for i, batch in enumerate(dev_data_loader):
        data, label = batch
        if torch.cuda.is_available():
            data, label = data.cuda(), label.cuda()
        data = data[0]
        label = label[0]
        pre = model(data)
        loss = loss_func(pre, label)
        sum_loss += loss
        ner_evaluation = True
        if ner_evaluation:
            # print("NER EVAL")
            O_tag_index = ner_tags["O"]
            O_tagged_predictions = torch.argmax(pre, dim=1) != O_tag_index
            O_tagged_labels = label != O_tag_index
            prediction_flags = torch.argmax(pre, dim=1) == label
            predictions_without_O_tag = O_tagged_predictions & prediction_flags & O_tagged_labels
            data_len += len(label) - torch.sum(~O_tagged_predictions | ~O_tagged_labels)
            succ += torch.sum(predictions_without_O_tag)
        else:
            data_len += len(label)
            succ += torch.sum(torch.argmax(pre, dim=1) == label)
    succ_rate = 100 * succ / data_len
    if math.isnan(succ_rate):
        succ_rate = 0
    print("evaluation  succeed rate: {}%".format(succ_rate))
    return succ_rate, sum_loss / len(dev_data_loader)


def train_model(model, train_data_loader, dev_data_loader, epoch_num, lr=0.1, batch_size=1, eval_num=-1):
    model.train()
    record_dev = []
    record_train = []
    train_data_loader = DeviceDataLoader(train_data_loader)
    dev_data_loader = DeviceDataLoader(dev_data_loader)
    opt = optim.SGD(model.parameters(), lr=lr)
    loss_func = nn.NLLLoss()
    for epoch in range(epoch_num):
        print("EPOCH: ", epoch)
        for i, batch in enumerate(train_data_loader):
            model.train(True)
            if i % 1000 == 0:
                print(i)
            opt.zero_grad()
            x, y = batch
            x = x[0]
            y = y[0]
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            pre = model(x)
            loss = loss_func(pre, y)
            loss.backward()
            opt.step()
            if eval_num != -1 and i % eval_num == 0:
                print("EVALUATING DEV:")
                record_dev.append(evaluate_model(model, dev_data_loader))
        if epoch % 1 == 0 and eval_num == -1:
            # print("EVALUATING TRAIN:")
            # record_train.append(evaluate_model(model, train_data_loader))
            print("EVALUATING DEV:")
            record_dev.append(evaluate_model(model, dev_data_loader))

        model.train(False)
    print_records(record_train, record_dev)


def save_model(model, file_path):
    torch.save(model, file_path)


def load_model(file_path):
    return torch.load(file_path)


def predict_loader(model, input_data_loader):
    record = []
    input_data_loader = DeviceDataLoader(input_data_loader)
    for i, batch in enumerate(input_data_loader):
        x, y = batch
        x = x[0]
        y = y[0]
        if torch.cuda.is_available():
            x, y = x.cuda(), y.cuda()
        pre = model(x)
        for prediction in pre:
            pre_argmax = torch.argmax(prediction)
            pre_int = pre_argmax.item()
            record.append(pre_int)
        record.append(-1)
    return record

from utils import UNIQUE
from DataSet import sentsDataset
import numpy as np
import torch

PAD = " "
sub_word_size = 3
UNIQUE_SIZE = len(UNIQUE)


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


def check_d(l):
    a = []
    for i, _ in enumerate(l.keys()):
        if i not in l.keys():
            a.append(i)
    if len(a) > 0: return a
    return True


def get_sub_word(word):
    size = len(word)
    if size == 1:
        return PAD + PAD + word, word + PAD + PAD
    elif size == 2:
        return PAD + word, word + PAD
    else:
        return word[0: sub_word_size], word[size - sub_word_size:size + 1]


class dictsList():
    def __init__(self, dicts, main_dict):
        self.dicts = dicts
        self.main = main_dict

    def __getitem__(self, item):
        return sum(d[item] for d in self.dicts) + self.main[item]

    def __len__(self):
        return len(self.main)

    def keys(self):
        return self.main.keys()

    def values(self):
        return self.main.values()


class dictUniqueDecorator():
    def __init__(self, w2idx, unitque=UNIQUE):
        self.words2index = w2idx
        self.unitque = unitque

    def __getitem__(self, item):
        if item in self.words2index.keys():
            return self.words2index[item]
        return self.words2index[self.unitque]

    def __len__(self):
        return len(self.words2index)

    def keys(self):
        return self.words2index.keys()

    def values(self):
        return self.words2index.values()


class dictListDecorator():
    def __init__(self, d):
        self.d = d

    def __getitem__(self, item):
        if type(item) == list:
            output = []
            for element in item:
                output.append(self.d[element])
            return output
        return self.d[item]

    def __len__(self):
        return len(self.d)

    def keys(self):
        return self.d.keys()

    def values(self):
        return self.d.values()


class Parser():
    def __init__(self, file_path, tags=None, w2idx=None, sep=" ", sub_word=False, is_char_em=False, is_test_data=False):
        self.sub_word = sub_word
        self.is_char_em = is_char_em
        self.tags = tags  # tag: index
        self.labels = []
        if w2idx == None:
            self.word2idx = {}
            self.words = []
        else:
            self.word2idx = w2idx
            self.words = w2idx.keys()
        self.windows = []
        self.win_tag = []
        # used only if sub_word
        starts = {}
        ends = {}
        # used only if is_char_em
        chars = set()

        sents = []
        with open(file_path) as f:
            line = f.readline()
            sent = ""
            sent_tags = []
            while line:
                if line[:-1] != "":
                    if is_test_data:
                        word = line[:-1]
                        if w2idx == None:
                            self.words.append(word)
                        if sub_word:
                            start, end = get_sub_word(word)
                            starts[word] = start
                            ends[word] = end
                        sent += word + " "
                        if is_char_em:
                            [chars.add(car) for car in word.split()]
                        self.labels.append(-1)
                        sent_tags.append(-1)
                    else:
                        if self.tags != None:
                            word, tag = line.split(sep)
                            word = word.lower()
                            tag = tag[:-1]
                        else:
                            word = line[:-1].lower()
                            tag = "-1"

                        if w2idx == None:
                            self.words.append(word)
                        if sub_word:
                            start, end = get_sub_word(word)
                            starts[word] = start
                            ends[word] = end
                        sent += word + " "
                        if self.tags != None:
                            self.labels.append(self.tags[tag])
                            sent_tags.append(self.tags[tag])
                        else:
                            self.labels.append(-1)
                            sent_tags.append(-1)
                else:  # end of a sentence
                    if len(sent) > 0:
                        sents.append((sent, sent_tags))
                        sent = ""
                        sent_tags = []

                line = f.readline()
        f.close()
        if w2idx == None:
            self.words.append(UNIQUE)
            self.words = set(self.words)
        self.word2idx = {word: idx for idx, word in enumerate(self.words)}
        self.word2idx = dictUniqueDecorator(self.word2idx)
        for sent, sent_tags in sents:
            sent = sent[:-1].split(" ")
            sent_embedded = [self.word2idx[word] for word in sent]
            self.windows.append(sent_embedded)
            self.win_tag.append(sent_tags)

        if sub_word:
            starts[UNIQUE] = UNIQUE[0:sub_word_size]
            ends[UNIQUE] = UNIQUE[UNIQUE_SIZE - sub_word_size:UNIQUE_SIZE + 1]
            starts_list = list(set(starts.values()))
            ends_list = list(set(ends.values()))
            self.starts_size = len(starts_list)
            self.ends_size = len(ends_list)
            start2idx = {st: idx for idx, st in enumerate(starts_list)}
            end2idx = {end: idx for idx, end in enumerate(ends_list)}
            w2idx_st = {self.word2idx[word]: start2idx[starts[word]] for word in starts.keys()}
            w2idx_end = {self.word2idx[word]: end2idx[ends[word]] for word in ends.keys()}
            self.w2starts = dictListDecorator(dictUniqueDecorator(w2idx_st, self.word2idx[UNIQUE]))
            self.w2ends = dictListDecorator(dictUniqueDecorator(w2idx_end, self.word2idx[UNIQUE]))
        if is_char_em:
            chars = list(chars)
            self.alphabet_size = len(chars) + 1
            self.I2F = {self.word2idx[word]: word for word in self.word2idx.keys()}
            self.C2I = {idx: car for idx, car in enumerate(chars)}
            self.C2I[UNIQUE] = len(chars)
            self.C2I = dictUniqueDecorator(self.C2I, UNIQUE)

    def get_data(self):
        return sentsDataset(self.windows, self.win_tag)

    def get_sub_word(self):
        if self.sub_word:
            return [self.w2starts, self.starts_size, self.w2ends, self.ends_size]
        return None

    def get_char_em(self, car_embedding_dim):
        if self.is_char_em:
            return self.alphabet_size, car_embedding_dim, self.I2F, self.C2I
        return None

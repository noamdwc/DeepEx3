import numpy as np
import re

# from ass2
UNIQUE = 'UUUNKKK'
pos_letters = ["dg", "a", "dg", "b", "dg", "c", "dg", "d", "dg"]
neg_letters = ["dg", "a", "dg", "c", "dg", "b", "dg", "d", "dg"]

pos_regex = "[1-9]+a+[1-9]+b+[1-9]+c+[1-9]+d+[1-9]+"
neg_regex = "[1-9]+a+[1-9]+c+[1-9]+b+[1-9]+d+[1-9]+"
_vocab = [str(i) for i in range(1, 10)] + ["a", "b", "c", "d"] + [UNIQUE]
F2I = {letter: idx + 1 for idx, letter in enumerate(_vocab)}


def is_pos(word):
    return re.match(pos_regex, word)


def is_neg(word):
    return re.match(neg_regex, word)


def get_vocab():
    return _vocab


def word2idx_list(word):
    return np.asarray([F2I[word[i]] for i in range(len(word))])


class calc_tags():
    def __init__(self,file_path):
        self.tags = {}
        with open(file_path) as f:
            line = f.readline()
            tag_count = 0
            while line:
                if line[:-1] != "":
                    word, tag = line.split()
                    if tag not in self.tags:
                        self.tags[tag] = tag_count
                        tag_count += 1
                line = f.readline()
        f.close()
        self.idx2tags = {v: k for k, v in self.tags.items()}

    def __getitem__(self, tag):
        if type(tag) == str:
            return self.tags[tag]
        else:
            return self.idx2tags[tag]

    def __len__(self):
        return len(self.tags)


##################################################################
#
#                        FROM ASS2
#
##################################################################

class calc_pos_tags():
    def __init__(self):
        self.tags = {}
        with open("pos/train") as f:
            line = f.readline()
            tag_count = 0
            while line:
                if line[:-1] != "":
                    word, tag = line.split()
                    if tag not in self.tags:
                        self.tags[tag] = tag_count
                        tag_count += 1
                line = f.readline()
        f.close()
        with open("pos/dev") as f:
            line = f.readline()
            tag_count = 0
            while line:
                if line[:-1] != "":
                    word, tag = line.split()
                    if tag not in self.tags:
                        self.tags[tag] = tag_count
                        tag_count += 1
                line = f.readline()
        f.close()
        self.idx2tags = {v: k for k, v in self.tags.items()}

    def __getitem__(self, tag):
        if type(tag) == str:
            return self.tags[tag]
        else:
            return self.idx2tags[tag]

    def __len__(self):
        return len(self.tags)


class calc_ner_tags():
    def __init__(self):
        self.tags = {}
        with open("ner/train") as f:
            line = f.readline()
            tag_count = 0
            while line:
                if line[:-1] != "":
                    word, tag = line.split()
                    if tag not in self.tags:
                        self.tags[tag] = tag_count
                        tag_count += 1
                line = f.readline()
        f.close()
        with open("ner/dev") as f:
            line = f.readline()
            tag_count = 0
            while line:
                if line[:-1] != "":
                    word, tag = line.split()
                    if tag not in self.tags:
                        self.tags[tag] = tag_count
                        tag_count += 1
                line = f.readline()
        f.close()
        self.idx2tags = {v: k for k, v in self.tags.items()}

    def __getitem__(self, tag):
        if type(tag) == str:
            return self.tags[tag]
        else:
            return self.idx2tags[tag]

    def __len__(self):
        return len(self.tags)


pos_tags = calc_pos_tags()
ner_tags = calc_ner_tags()

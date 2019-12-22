import numpy as np
import re
pos_regex = "[1-9]+a+[1-9]+b+[1-9]+c+[1-9]+d+[1-9]+"
neg_regex = "[1-9]+a+[1-9]+c+[1-9]+b+[1-9]+d+[1-9]+"
_vocab = [str(i) for i in range(1, 10)] + ["a", "b", "c", "d"]
F2I = {letter: idx for idx, letter in enumerate(_vocab)}

def is_pos(word):
    return re.match(pos_regex, word)
def is_neg(word):
    return re.match(neg_regex, word)

def get_vocab():
    return _vocab


def word2idx_list(word):
    return np.asarray([F2I[word[i]] for i in range(len(word))])

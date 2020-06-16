from utils import pos_regex, neg_regex, word2idx_list, pos_letters, neg_letters
# import rstr
from xeger import Xeger
from torch.nn import Softmax
import torch
from random import randint

softmax = Softmax(dim=1)


def to_file(file_path, words):
    with open(file_path, "w") as file:
        for word in words:
            file.write(word + '\n')


#
# def my_gen(letters, size=100):
#     out = ""
#     sizes = torch.tensor([randint(1, 10) for _ in range(len(letters))])
#     sizes = softmax(sizes)
#     sizes = sizes * size
#     for i in range(len(letters)):
#         out += letters[i] * int(sizes[i].item())
#     return out
#     #
#     # for let in letters:
#     #     for i in range(randint(1, 10)):
#     #         add = let
#     #         if let == "dg":
#     #             add = str(randint(1, 9))
#     #         out += add
#     # return out

#
# def my_gen_k(k, letters):
#     words = []
#     while len(words) < k:
#         words.append(my_gen(letters))
#     return words

#
# def my_gen2file(k, file_path, letters):
#     to_file(file_path, my_gen_k(k, letters))
#
def gen_example(regex, limit):
    example = Xeger(limit=limit)
    example = example.xeger(regex)
    return example


starting_limit = 90


def get_one(regex):
    limit = starting_limit
    example = gen_example(regex, limit)
    while len(example) > 100:
        limit -= 2
        example = gen_example(regex, limit)
    return example


def get_one_by_letters(letters):
    out = ""
    for let in letters:
        if let == "dg":
            for _ in range(randint(10, 15)):
                out += str(randint(1, 9))
        else:
            out += let * randint(10, 15)
    return out


def gen_k(k, regex):
    examples = set()
    while len(examples) < k:
        examples.add(get_one(regex))
    return list(examples)


def gen_k_by_letters(k, letters):
    examples = set()
    while len(examples) < k:
        examples.add(get_one_by_letters(letters))
    return list(examples)


def gen2file(k, regex, file_path):
    to_file(file_path, gen_k(k, regex))


def gen2file_by_letters(k, letters, file_path):
    to_file(file_path, gen_k_by_letters(k, letters))


def get_samples(k):
    k = int(k / 2)
    pos_labels = [1 for _ in range(k)]
    neg_labels = [0 for _ in range(k)]
    pos_samples = gen_k(k, pos_regex, word2idx_list)
    neg_samples = gen_k(k, neg_regex, word2idx_list)
    X = pos_samples + neg_samples
    Y = pos_labels + neg_labels
    return X, Y


if __name__ == '__main__':
    # gen2file(1000, pos_regex, "pos_test.txt")
    # gen2file(1000, neg_regex, "neg_test.txt")
    #
    gen2file_by_letters(500, pos_letters, "pos_test.txt")
    gen2file_by_letters(500, neg_letters, "neg_test.txt")

#############################
#                           #
#        MAKE IT FAIL       #
#                           #
#############################

fail_abc = ["1", "0"]


# palindrome:

def get_one_fail1(pos, size):
    if pos:
        mid = ""
        if size % 2 == 1:
            mid += str(randint(0, 1))
            size -= 1
        side = ""
        for _ in range(size / 2):
            side += str(randint(0, 1))
        return side + mid + "".join(reversed(side))
    # if false:
    out = ""
    for _ in range(size):
        out += str(randint(0, 1))
    return out


# mid zeros:

def get_one_fail2(pos, size):
    if pos:
        mid = "0"
        side1 = ""
        for _ in range(size / 2):
            side1 += str(randint(0, 1))
        side2 = ""
        for _ in range(size / 2):
            side2 += str(randint(0, 1))
        return side1 + mid + side2
    # if false:
    out = ""
    for _ in range(size):
        out += str(randint(0, 1))
    # make sure the mid letter is 1
    index = len(out) / 2
    out = out[:index] + "1" + out[index + 1:]
    return out


# first and last letter are the same:

def get_one_fail3(pos, size):
    if pos:
        if size == 1:
            return str(randint(0, 1))
        out = ""
        for _ in range(size - 2):
            out += str(randint(0, 1))
        edge = str(randint(0, 1))
        out = edge + out + edge
        return out
    # if false:
    out = ""
    for _ in range(size - 1):
        out += str(randint(0, 1))
    # make sure the mid letter is 1
    if out[0] == "0":
        out += "1"
    else:
        out += "0"
    return out


def get_fail(k, pos, size, kind):
    out = set()
    if kind == 1:
        create = get_one_fail1
    elif kind == 2:
        create = get_one_fail2
    elif kind == 3:
        create = get_one_fail3
    else:
        raise Exception("error at get fail, kind should be 1, 2 or 3")
    while len(out) < k:
        out.add(create(pos, size))
    return out

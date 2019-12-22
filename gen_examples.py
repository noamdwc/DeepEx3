from utils import pos_regex, neg_regex, word2idx_list
import rstr


def gen_k(k, regex, convertor=None):
    examples = []
    while len(examples) < k:
        examples.append(rstr.xeger(regex))
    if convertor != None:
        examples = [convertor(example) for example in examples]
    else:
        examples = list(examples)
    return examples


def gen2file(k, regex, file_path):
    examples = gen_k(k, regex)
    with open(file_path, "w") as file:
        for ex in examples:
            file.write(ex + '\n')


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
    gen2file(500, neg_regex, "bad.txt")

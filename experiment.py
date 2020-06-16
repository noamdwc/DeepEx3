from utils import get_vocab, word2idx_list
from gen_examples import gen_k
from Models import Part1Model, train_model
from DataSet import DataBuilder

if __name__ == '__main__':
    import sys

    # HYPER PARAMETERS
    lr = 0.15
    embedding_dim = 50
    in_dim = 64
    hid_dim = 32
    out_dim = 2
    k = 500
    epoch_num = 30
    train_porction = 0.5
    batch_size = 1
    if len(sys.argv) > 1:
        epoch_num = int(sys.argv[1])

    vocab = get_vocab()
    model = Part1Model(len(vocab), embedding_dim, in_dim, hid_dim, out_dim)
    # train_dataloader, dev_dataloader = Maker(word2idx_list).add_pos(gen_k(5000, pos_regex)).add_neg(gen_k(5000, neg_regex)).get_data(train_porction)
    train_dataloader, dev_dataloader = DataBuilder(word2idx_list) \
        .read_pos("pos_test.txt") \
        .read_neg("neg_test.txt") \
        .build_data(train_porction, batch_size)
    # X, Y  = get_samples(k)
    import torch

    if torch.cuda.is_available():
        model = model.cuda()
        model.to_cuda()
    model.train_model(train_dataloader, dev_dataloader, epoch_num, lr, batch_size)
    # train_model(model, train_dataloader, dev_dataloader, epoch_num, lr, batch_size)

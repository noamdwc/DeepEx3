from utils import pos_regex, neg_regex, get_vocab, F2I, word2idx_list
from gen_examples import gen_k
from Models import Model, split_data
from DataSet import Maker

if __name__ == '__main__':
    # HYPER PARAMETERS
    lr = 15
    embedding_dim = 100
    in_dim = 128
    hid_dim = 62
    out_dim = 2
    k = 500
    epoch_num = 10
    train_porction = 0.8

    vocab = get_vocab()
    model = Model(len(vocab), embedding_dim, in_dim, hid_dim, out_dim)
    train_dataloader, dev_dataloader = Maker(word2idx_list).add_pos(gen_k(5000, pos_regex)).add_neg(gen_k(5000, neg_regex)).get_data(train_porction)
    # train_dataloader, dev_dataloader = Maker(word2idx_list).read_pos("good.txt").read_neg("bad.txt").get_data(
    #     train_porction)
    # X, Y = get_samples(k)

    model.train_model(train_dataloader, dev_dataloader, epoch_num, lr)

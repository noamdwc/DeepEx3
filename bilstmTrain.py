from utils import pos_tags, ner_tags
from Models import create_dataLoader, LSTMTagger, train_model, save_model
from Parser import Parser

SUB_WORD = False
CHAR_EM = True
D = True
SRC = "pos"

if __name__ == '__main__':
    import sys

    if len(sys.argv) < 4:
        print("error at console, not enough parameter")
    repr_symbol = sys.argv[1]
    train_file = sys.argv[2]
    model_path = sys.argv[3]
    if repr_symbol == "a":
        SUB_WORD = False
        CHAR_EM = False
        D = False
    elif repr_symbol == "b":
        SUB_WORD = False
        CHAR_EM = True
        D = False
    elif repr_symbol == "c":
        SUB_WORD = True
        CHAR_EM = False
        D = False
    elif repr_symbol == "d":
        SUB_WORD = False
        CHAR_EM = True
        D = True
    else:
        print("error at repr, not supported symbol. please enter one of a,b,c,d")

    eval_num = -1
    if len(sys.argv) > 4:
        eval_num = int(sys.argv[4])
    SRC = train_file
    if SRC == "ner":
        Tags = ner_tags
        SEP = '\t'
    if SRC == "pos":
        Tags = pos_tags
        SEP = ' '

    # HYPER PARAMETER
    embedding_dim = 50
    in_dim = 64
    hid_dim = 32
    out_dim = len(Tags)
    epoch_num = 5
    car_embedding_dim = 30
    batch_size = 1
    lr = 0.1
    num_workers = 0

    # for ner:
    if SRC == "ner":
        lr = 0.04

    if len(sys.argv) > 5:
        lr = float(sys.argv[5])

    # train_parser = Parser(train_file,)
    train_parser = Parser(SRC + "/train", Tags, sep=SEP, sub_word=SUB_WORD, is_char_em=CHAR_EM)
    dev_parser = Parser(SRC + "/dev", Tags, train_parser.word2idx, sep=SEP, sub_word=SUB_WORD)
    train_loader = create_dataLoader(train_parser, batch_size, num_workers)
    dev_loader = create_dataLoader(dev_parser, batch_size, num_workers)
    if SUB_WORD:
        model = LSTMTagger(embedding_dim, in_dim, hid_dim, len(train_parser.word2idx), tagset_size=out_dim,
                           c=train_parser.get_sub_word(), b=train_parser.get_char_em(car_embedding_dim), d=D)
    else:
        model = LSTMTagger(embedding_dim, in_dim, hid_dim, len(train_parser.word2idx), tagset_size=out_dim,
                           b=train_parser.get_char_em(car_embedding_dim), d=D)
    import torch

    if torch.cuda.is_available():
        model = model.cuda()
    train_model(model, train_loader, dev_loader, epoch_num, eval_num=eval_num)
    print(lr)
    save_model(model, model_path)

from Models import create_dataLoader, load_model, predict_loader
from Parser import Parser
import torch

SUB_WORD = False
CHAR_EM = True
D = True
SRC = "pos"

if __name__ == '__main__':
    import sys

    if len(sys.argv) < 4:
        print("error at console, not enough parameter")
    repr_symbol = sys.argv[1]
    model_path = sys.argv[2]
    SRC = sys.argv[3]

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
    SEP = ""
    eval_num = 500
    # HYPER PARAMETER
    embedding_dim = 50
    in_dim = 64
    hid_dim = 32
    epoch_num = 5
    car_embedding_dim = 30
    batch_size = 1
    lr = 0.1
    num_workers = 0

    input_parser = Parser(SRC, sep=SEP, sub_word=SUB_WORD, is_char_em=CHAR_EM)
    input_loader = create_dataLoader(input_parser, batch_size, num_workers, shuffle=False)

    if torch.cuda.is_available():
        model = torch.load(model_path)
        model = model.cuda()
    else:
        model = torch.load(model_path, map_location='cpu')
    results = predict_loader(model, input_loader)
    print(results)

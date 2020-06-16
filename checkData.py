import re

pos_regex = "[1-9]+a+[1-9]+b+[1-9]+c+[1-9]+d+[1-9]+"
neg_regex = "[1-9]+a+[1-9]+c+[1-9]+b+[1-9]+d+[1-9]+"


def is_pos(word):
    return re.match(pos_regex, word)


def is_neg(word):
    return re.match(neg_regex, word)


def check_file(file_path, checker):
    counter = 0
    flag = False
    with open(file_path) as file:
        line = file.readline()
        while line:
            counter += 1
            if not checker(line):
                flag = True
                print("not match with word: ", line, " at line: ", counter)
            line = file.readline()
    file.close()
    if not flag:
        print("\t" + file_path + " is checked out")


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 3:
        print("wrong format")
        print("the program needs two arguments"
              "first is the path for the pos.txt examples, and the second is for neg.txt examples")
        exit(1)
    pos_path = sys.argv[1]
    neg_path = sys.argv[2]
    print("check good file:")
    check_file(pos_path, is_pos)
    print("check bad file:")
    check_file(neg_path, is_neg)

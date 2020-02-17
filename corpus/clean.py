import os
import re
from nltk.tokenize import sent_tokenize
from argparse import ArgumentParser


def is_ascii(s):
    return all(ord(c) < 128 for c in s)


def hasSpecialCharacter(inputString):
    specials = '[@_#$%^&*<>}(){~:]'
    return any(char in specials for char in inputString)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", "--input_file", required=True,
                        help="input file to clean")
    args = vars(parser.parse_args())

    fi = open(os.path.join(os.path.abspath(""), "corpus",
                           args["input_file"]), "r", encoding="utf-8")
    fo = open(os.path.join(os.path.abspath(""), "corpus",
                           args["input_file"][:-4] + ".tsv"), "w", encoding="utf-8")

    texts = []

    for line in fi.readlines():
        abstracts = re.findall("<AbstractText>(.*?)</AbstractText>", line)
        if abstracts != []:
            for abstract in abstracts:
                sentences = sent_tokenize(abstract)
                for sentence in sentences:
                    if is_ascii(sentence) and hasSpecialCharacter(sentence) == False and len(sentence) > 10:
                        fo.write(sentence + "\n")
                        fo.flush()

    fi.close()
    fo.close()

    os.remove(os.path.join(os.path.abspath(""), "corpus", args["input_file"]))

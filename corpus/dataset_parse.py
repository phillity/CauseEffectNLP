import os
import spacy
import pickle
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from corpus_parse_util import parse_sp


np.random.seed(42)
nlp = spacy.load("en_core_web_sm")
nlp.disable_pipes("ner")


def seed_patterns():
    patterns_pos, patterns_neg = [], []
    seed_patterns_pos = pd.read_csv(
        os.path.join(os.path.abspath(""), "data", "seed_patterns_pos.tsv"), delimiter="\t", header=None).values
    seed_patterns_neg = pd.read_csv(
        os.path.join(os.path.abspath(""), "data", "seed_patterns_neg.tsv"), delimiter="\t", header=None).values

    for pattern, seed_patterns in [(patterns_pos, seed_patterns_pos), (patterns_neg, seed_patterns_neg)]:
        for seed_pattern in seed_patterns:
            x, y, zs, sentence = seed_pattern

            for z in zs.split("|"):
                try:
                    tmp_sentence = sentence.replace(
                        "X", x).replace("Y", y).replace("Z", z)
                    tmp_sentence.replace(tmp_sentence[0], tmp_sentence[0].upper())

                    doc = nlp(tmp_sentence)
                    edges = parse_sp(x, y, doc, nlp)
                    edges = [",".join(edge) for edge in edges]

                    if edges not in pattern and len(edges) > 1:
                        pattern.append(edges)

                except Exception as e:
                    pass

    return patterns_pos, patterns_neg


def pattern_intersect(edges, patterns, threshold=1.0):
    for i, pattern in enumerate(patterns):
        if len(list(set(edges).intersection(pattern))) / len(pattern) >= threshold:
            return pattern
    return None


def parse(patterns_pos, patterns_neg, raw_sentences):
    nlp.add_pipe(nlp.create_pipe("merge_noun_chunks"))
    pos_dataset = open(os.path.join(os.path.abspath(""),
                                    "corpus", args["input_file"][:-4] + "_pos.tsv"), "w")
    neg_dataset = open(os.path.join(os.path.abspath(""),
                                    "corpus", args["input_file"][:-4] + "_neg.tsv"), "w")
    pos_neg_dataset = open(os.path.join(os.path.abspath(""),
                                        "corpus", args["input_file"][:-4] + "_pos_neg.tsv"), "w")

    for raw_sentence in raw_sentences.readlines():
        try:
            doc = nlp(raw_sentence.decode(errors="ignore"))
            sentences = [sent.string.strip() for sent in doc.sents]

            for sentence in sentences:
                doc = nlp(sentence)
                noun_chunks = list(doc.noun_chunks)
                flag = False

                for i, x in enumerate(noun_chunks):
                    for j, y in enumerate(noun_chunks):

                        if i == j:
                            continue

                        if flag:
                            continue

                        edges = parse_sp(
                            x.lower_, y.lower_, doc, nlp)
                        edges = [",".join(edge) for edge in edges]

                        ppos = pattern_intersect(edges, patterns_pos)
                        pneg = pattern_intersect(edges, patterns_neg)
                        if ppos is not None:
                            if "not" not in str(ppos):
                                pos_dataset.write(
                                    "\t".join([x.lower_, y.lower_, sentence, str(ppos), "1"]) + "\n")
                                pos_dataset.flush()

                            else:
                                pos_neg_dataset.write(
                                    "\t".join([x.lower_, y.lower_, sentence, str(ppos), "0"]) + "\n")
                                pos_neg_dataset.flush()

                            flag = True

                        elif pneg is not None:
                            neg_dataset.write(
                                "\t".join([x.lower_, y.lower_, sentence, str(pneg), "0"]) + "\n")
                            neg_dataset.flush()
                            flag = True

                        else:
                            continue

        except Exception as e:
            pass

    pos_dataset.close()
    neg_dataset.close()
    pos_neg_dataset.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", "--input_file", required=True,
                        help="input file to parse")
    args = vars(parser.parse_args())

    patterns_pos, patterns_neg = seed_patterns()

    raw_sentences = open(os.path.join(os.path.abspath(""),
                                      "corpus", args["input_file"]), "rb")
    parse(patterns_pos, patterns_neg, raw_sentences)
    raw_sentences.close()

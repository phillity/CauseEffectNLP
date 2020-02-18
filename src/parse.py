import os
import spacy
import pickle
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from parse_utils import parse_sp


np.random.seed(42)
nlp = spacy.load("en_core_web_sm")


def seed_patterns():
    patterns, dataset = [], []
    seed_patterns = pd.read_csv(
        os.path.join(os.path.abspath(""), "data", "seed_patterns.tsv"), delimiter="\t", header=None).values
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

                dataset.append([x, y, tmp_sentence, "1"])
                if edges not in patterns and len(edges) > 1:
                    patterns.append(edges)

            except Exception as e:
                pass

    return patterns


def pattern_intersect(edges, patterns, threshold=0.5):
    for i, pattern in enumerate(patterns):
        if len(list(set(edges).intersection(pattern))) / len(pattern) > threshold:
            return True
    return False


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", "--input_file", required=True,
                        help="input file to parse")
    args = vars(parser.parse_args())

    patterns = seed_patterns()

    raw_sentences = open(os.path.join(os.path.abspath(""),
                                      "corpus", args["input_file"]), "r")

    new_dataset = open(os.path.join(os.path.abspath(""),
                                    "data", args["input_file"]), "w")

    for raw_sentence in raw_sentences.readlines():
        try:
            doc = nlp(raw_sentence)
            sentences = [sent.string.strip() for sent in doc.sents]

            for sentence in sentences:
                doc = nlp(sentence)
                noun_chunks = list(doc.noun_chunks)

                for i, x in enumerate(noun_chunks):
                    for j, y in enumerate(noun_chunks):
                        if i == j:
                            continue

                        edges = parse_sp(
                            x.root.lower_, y.root.lower_, doc, nlp)
                        edges = [",".join(edge) for edge in edges]

                        if pattern_intersect(edges, patterns):
                            new_dataset.write(
                                "\t".join([x.root.lower_, y.root.lower_, sentence, "1"]) + "\n")
                            new_dataset.flush()

        except Exception as e:
            pass

    new_dataset.close()

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
    patterns = []
    seed_patterns = pd.read_csv(
        os.path.join(os.path.abspath(""), "data", "seed_patterns.tsv"), delimiter="\t", header=None).values
    for seed_pattern in seed_patterns:
        x, y, zs, sentence = seed_pattern
        for z in zs.split("|"):
            tmp_sentence = sentence.replace(
                "X", x).replace("Y", y).replace("Z", z)
            tmp_sentence.replace(tmp_sentence[0], tmp_sentence[0].upper())
            patterns.append([x, y, tmp_sentence, "1"])
        patterns.append([x, y, sentence, "1"])

    pickle.dump(patterns, open(os.path.join(
        os.path.abspath(""), "data", "seed_patterns.pkl"), "wb"))
    return patterns


def pattern_intersect(edges, patterns, threshold=0.75):
    for i, pattern in enumerate(patterns):
        pattern = [",".join(p) for p in pattern]
        if len(list(set(edges).intersection(pattern))) / len(pattern) > threshold:
            return True
    return False


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", "--input_file", required=True,
                        help="input file to parse")
    args = vars(parser.parse_args())

    patterns = pickle.load(
        open(os.path.join(os.path.abspath(""), "data", "seed_patterns.pkl"), "r"))

    raw_sentences = pd.read_csv(os.path.join(os.path.abspath(""), "corpus", args["input_file"]),
                                delimiter="\n", header=None).values.ravel().tolist()

    new_dataset = open(os.path.join(os.path.abspath(""),
                                    "data", args["input_file"]), "w")
    new_cnt = 0
    for cnt, raw_sentence in enumerate(raw_sentences):
        try:
            doc = nlp(raw_sentence)
            sentences = [sent.string.strip() for sent in doc.sents]

            for sentence in sentences:
                if sentence[-1] != "." or sentence[-1] != "!" or sentence[-1] != "?":
                    sentence += "."
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
                            new_cnt += 1
                            new_dataset.write(
                                "\t".join([x.root.lower_, y.root.lower_, sentence, "1"]) + "\n")
                            new_dataset.flush()

        except Exception as e:
            pass

    new_dataset.close()

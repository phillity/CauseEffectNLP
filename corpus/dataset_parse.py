import os
import spacy
import pickle
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from dataset_parse_util import parse_sp


np.random.seed(42)
nlp = spacy.load("en_core_web_sm")
nlp.disable_pipes("ner")
nlp.add_pipe(nlp.create_pipe("merge_noun_chunks"))


def generate_patterns():
    patterns = {}
    for fi in ["ce", "hyp", "me"]:
        patterns[fi] = []
        seed_patterns = pd.read_csv(
            os.path.join(os.path.abspath(""), "data", "seed_patterns_{}.tsv".format(fi)), delimiter="\t", header=None).values

        for seed_pattern in seed_patterns:
            x, y, zs, sentence = seed_pattern

            for z in zs.split("|"):
                try:
                    tmp_sentence = sentence.replace(
                        "X", x).replace("Y", y).replace("Z", z)
                    tmp_sentence.replace(
                        tmp_sentence[0], tmp_sentence[0].upper())

                    doc = nlp(tmp_sentence)
                    edges = parse_sp(x, y, doc, nlp)
                    edges = [",".join(edge) for edge in edges]

                    if edges not in patterns[fi] and len(edges) > 1:
                        patterns[fi].append(edges)

                except Exception as e:
                    pass

    return patterns


def pattern_intersect(edges, patterns, threshold=1.0):
    for i, pattern in enumerate(patterns):
        if len(list(set(edges).intersection(pattern))) / len(pattern) >= threshold:
            return pattern
    return None


def parse(patterns_all, raw_sentences):
    rel_pos, rel_neg = {}, {}
    for fi in ["ce", "hyp", "me"]:
        rel_pos[fi] = open(os.path.join(os.path.abspath(""),
                                         "corpus", args["input_file"][:-4] + "_{}_pos.tsv".format(fi)), "w")
        rel_neg[fi] = open(os.path.join(os.path.abspath(""),
                                         "corpus", args["input_file"][:-4] + "_{}_neg.tsv".format(fi)), "w")

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

                        edges = parse_sp(
                            x.lower_, y.lower_, doc, nlp)
                        edges = [",".join(edge) for edge in edges]

                        patt_ce = pattern_intersect(edges, patterns["ce"])
                        patt_hyp = pattern_intersect(edges, patterns["hyp"])
                        patt_me = pattern_intersect(edges, patterns["me"])

                        if patt_ce is not None:
                            if "not" not in str(sentence):
                                rel_pos["ce"].write(
                                    "\t".join([x.lower_, y.lower_, sentence, str(patt_ce), "1"]) + "\n")
                                rel_pos["ce"].flush()

                                rel_neg["hyp"].write(
                                    "\t".join([x.lower_, y.lower_, sentence, str(patt_ce), "0"]) + "\n")
                                rel_neg["hyp"].flush()
                                rel_neg["me"].write(
                                    "\t".join([x.lower_, y.lower_, sentence, str(patt_ce), "0"]) + "\n")
                                rel_neg["me"].flush()

                        if patt_hyp is not None:
                            if "not" not in str(sentence):
                                rel_pos["hyp"].write(
                                    "\t".join([x.lower_, y.lower_, sentence, str(patt_hyp), "1"]) + "\n")
                                rel_pos["hyp"].flush()

                                rel_neg["ce"].write(
                                    "\t".join([x.lower_, y.lower_, sentence, str(patt_hyp), "0"]) + "\n")
                                rel_neg["ce"].flush()
                                rel_neg["me"].write(
                                    "\t".join([x.lower_, y.lower_, sentence, str(patt_hyp), "0"]) + "\n")
                                rel_neg["me"].flush()

                        if patt_me is not None:
                            if "not" not in str(sentence):
                                rel_pos["me"].write(
                                    "\t".join([x.lower_, y.lower_, sentence, str(patt_me), "1"]) + "\n")
                                rel_pos["me"].flush()

                                rel_neg["ce"].write(
                                    "\t".join([x.lower_, y.lower_, sentence, str(patt_me), "0"]) + "\n")
                                rel_neg["ce"].flush()
                                rel_neg["hyp"].write(
                                    "\t".join([x.lower_, y.lower_, sentence, str(patt_me), "0"]) + "\n")
                                rel_neg["hyp"].flush()

        except Exception as e:
            pass

    for fi in ["ce", "hyp", "me"]:
        rel_pos[fi].close(), rel_neg[fi].close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", "--input_file", required=True,
                        help="input file to parse")
    args = vars(parser.parse_args())

    patterns = generate_patterns()

    raw_sentences = open(os.path.join(os.path.abspath(""),
                                      "corpus", args["input_file"]), "rb")
    parse(patterns, raw_sentences)
    raw_sentences.close()

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser


np.random.seed(42)


def pattern_split(dataset):
    fi_pos = open(os.path.join(os.path.abspath(""), "corpus",
                               "{}_{}.tsv".format(dataset, "pos")), "rb")
    fi_neg = open(os.path.join(os.path.abspath(""), "corpus",
                               "{}_{}.tsv".format(dataset, "neg")), "rb")

    dataset_pos, dataset_neg = [], []
    patterns_pos, patterns_neg = [], []
    for fi_in, dataset, patterns in [(fi_pos, dataset_pos, patterns_pos), (fi_neg, dataset_neg, patterns_neg)]:
        for line in fi_in.readlines():
            if len(line.decode(errors="ignore").split("\t")) == 5:
                x, y, sentence, pattern, label = line.decode(
                    errors="ignore").split("\t")
                patterns.append(pattern)
                dataset.append([x, y, sentence])
    patterns_pos, patterns_neg = np.array(patterns_pos), np.array(patterns_neg)

    unique_pos, counts_pos = np.unique(
        patterns_pos, return_counts=True)
    unique_pos = unique_pos[counts_pos.argsort()]
    counts_pos = counts_pos[counts_pos.argsort()]

    most_freq_pos = unique_pos[-1]
    unique_pos = np.delete(unique_pos, -1)
    train_pos, test_pos = train_test_split(
        unique_pos, test_size=0.1, shuffle=True, random_state=42)
    test_pos = np.append(test_pos, most_freq_pos)
    train_pos, val_pos = train_test_split(
        train_pos, test_size=0.1, shuffle=True, random_state=42)

    unique_neg, counts_neg = np.unique(
        patterns_neg, return_counts=True)
    train_neg, test_neg = train_test_split(
        unique_neg, test_size=0.1, shuffle=True, random_state=42)
    train_neg, val_neg = train_test_split(
        train_neg, test_size=0.1, shuffle=True, random_state=42)

    train, val, test = [[] for i in range(3)]
    for patterns in [patterns_pos, patterns_neg]:
        for i, pattern in enumerate(patterns):
            if pattern in train_pos:
                dataset_pos[i].append("1")
                train.append(dataset_pos[i])
            elif pattern in val_pos:
                dataset_pos[i].append("1")
                val.append(dataset_pos[i])
            elif pattern in test_pos:
                dataset_pos[i].append("1")
                test.append(dataset_pos[i])
            elif pattern in train_neg:
                dataset_neg[i].append("0")
                train.append(dataset_neg[i])
            elif pattern in val_neg:
                dataset_neg[i].append("0")
                val.append(dataset_neg[i])
            else:
                dataset_neg[i].append("0")
                test.append(dataset_neg[i])

    print("Number of total sentences: {}".format(
        len(dataset_pos) + len(dataset_neg)))
    print("Number of positive patterns: {}".format(unique_pos.shape[0]))
    print("Number of positive sentences: {}".format(patterns_pos.shape[0]))
    print("Number of negative patterns: {}".format(unique_neg.shape[0]))
    print("Number of negative sentences: {}\n".format(patterns_neg.shape[0]))

    print("Number of total training sentences: {}".format(len(train)))
    print("Number of training positive patterns: {}".format(len(train_pos)))
    print("Number of training positive sentences: {}".format(
        len([1 for train_i in train if train_i[-1] == "1"])))
    print("Number of training negative patterns: {}".format(len(train_neg)))
    print("Number of training negative sentences: {}\n".format(
        len([0 for train_i in train if train_i[-1] == "0"])))

    print("Number of total validation sentences: {}".format(len(val)))
    print("Number of validation positive patterns: {}".format(len(val_pos)))
    print("Number of validation positive sentences: {}".format(
        len([1 for val_i in val if val_i[-1] == "1"])))
    print("Number of validation negative patterns: {}".format(len(val_neg)))
    print("Number of validation negative sentences: {}\n".format(
        len([0 for val_i in val if val_i[-1] == "0"])))

    print("Number of total testing sentences: {}".format(len(test)))
    print("Number of testing positive patterns: {}".format(len(test_pos)))
    print("Number of testing positive sentences: {}".format(
        len([1 for test_i in test if test_i[-1] == "1"])))
    print("Number of testing negative patterns: {}".format(len(test_neg)))
    print("Number of testing negative sentences: {}".format(
        len([0 for test_i in test if test_i[-1] == "0"])))
    
    return train, val, test


def dataset_split(dataset):
    fi_in = open(os.path.join(os.path.abspath(
        ""), "corpus", dataset + ".tsv"), "r")
    
    data, y = [], []
    for line in fi_in.read().split("\n"):
        if len(line.split("\t")) == 4:
            data.append(line.split("\t"))
            y.append(int(line[-1]))

    train, test, train_y, test_y = train_test_split(
        data, y, test_size=0.1, shuffle=True, random_state=42)
    train, val, train_y, val_y = train_test_split(
        train, train_y, test_size=0.1, shuffle=True, random_state=42)

    print("Number of sentences: {}".format(len(y)))
    print("Number of positive sentences: {}".format(np.sum(y)))
    print("Number of negative sentences: {}\n".format(len(y) - np.sum(y)))

    print("Number of training sentences: {}".format(len(train_y)))
    print("Number of training positive sentences: {}".format(np.sum(train_y)))
    print("Number of training negative sentences: {}\n".format(
        len(train_y) - np.sum(train_y)))

    print("Number of validation sentences: {}".format(len(val_y)))
    print("Number of validation positive sentences: {}".format(np.sum(val_y)))
    print("Number of validation negative sentences: {}\n".format(
        len(val_y) - np.sum(val_y)))

    print("Number of testing sentences: {}".format(len(test_y)))
    print("Number of testing positive sentences: {}".format(np.sum(test_y)))
    print("Number of testing negative sentences: {}".format(
        len(test_y) - np.sum(test_y)))

    return train, val, test


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-d", "--dataset", required=True,
                        help="dataset to split")
    args = vars(parser.parse_args())

    if args["dataset"] == "pubmed" or args["dataset"] == "yahoo_qa" or args["dataset"] == "cdr":
        train, val, test = pattern_split(args["dataset"])
    else:
        train, val, test = dataset_split(args["dataset"])

    for part, data in [("train", train), ("val", val), ("test", test)]:
        fi = open(os.path.join(os.path.abspath(
            ""), "corpus", "{}_{}.tsv".format(args["dataset"], part)), "w", newline="\n")
        for row in data:
            fi.write("\t".join(row) + "\n")
        fi.close()

import os
import h5py
import spacy
import numpy as np
import tensorflow as tf
from argparse import ArgumentParser
from embed_utils import parse_sentence, path_embedding


np.random.seed(42)


def embed(dataset, maxlen):
    nlp = spacy.load("en_core_web_sm")
    nlp.disable_pipes("ner")
    nlp.add_pipe(nlp.create_pipe("merge_noun_chunks"))
    use_map = {}
    drop_sen_cnt = 0

    
    for dataset_part in ["_train.tsv", "_val.tsv", "_test.tsv"]:
        fi_in = open(os.path.join(os.path.abspath(""), "corpus",
                                  dataset + dataset_part), "r")

        fi_out = h5py.File(os.path.join(os.path.abspath(""),
                                        "data", dataset + dataset_part[:-4] + ".hdf5"), "w")
        grp = fi_out.create_group(dataset_part[1:-4])
        X = grp.create_dataset("X", (0, maxlen, 1100),
                               maxshape=(None, maxlen, 1100), dtype="f", compression="gzip", compression_opts=9)
        y = grp.create_dataset("y", (0,), maxshape=(
            None,), dtype="i", compression="gzip")

        num_lines = 0
        with open(os.path.join(os.path.abspath(""), "corpus", dataset + dataset_part)) as fi_tmp:
            for i, _ in enumerate(fi_tmp):
                pass
            num_lines = i + 1

        for i, line in enumerate(fi_in.read().split("\n")):
            if len(line.split("\t")) == 4:
                print(
                    "{} -- {}/{}    total dropped -- {}".format(dataset_part[1:-4], i + 1, num_lines, drop_sen_cnt), end="\r")
        
                try:
                    line = line.split("\t")
                    edges = parse_sentence(line[0], line[1], nlp(line[2]), nlp)
                    path_emb, use_map = path_embedding(edges, use_map)

                    if path_emb.shape[0] < 10:
                        path_emb = np.vstack(
                            [path_emb, np.zeros((10 - path_emb.shape[0], 1100))])
                    if path_emb.shape[0] > 10:
                        path_emb = path_emb[:10]

                    X.resize((X.shape[0] + 1), axis=0)
                    X[-1] = path_emb
                    y.resize((y.shape[0] + 1), axis=0)
                    y[-1] = int(line[-1])

                except Exception as e:
                    drop_sen_cnt += 1
                    pass

            if i % 1000 == 0:
                fi_out.flush()
        
        print("\t\t\t\t\t\t\t\t\t", end="\r")
        fi_in.close()
        fi_out.close()

    print("Dropped sentence count: {}".format(drop_sen_cnt))
    tf.keras.backend.clear_session()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-d", "--dataset", required=True, choices=["yahoo_qa", "pubmed", "cdr", "ade"],
                        help="dataset to embed")
    parser.add_argument("-m", "--maxval", required=False, type=int, default=10,
                        help="maxval dimension hyperparameter")
    args = vars(parser.parse_args())

    embed(args["dataset"], args["maxval"])

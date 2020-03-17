import os
import spacy
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from networkx import Graph, DiGraph, descendants, shortest_path


np.random.seed(42)
tf.random.set_seed(42)


# part of speech dict
# https://universaldependencies.org/u/pos/
POS = {'adj': 0, 'adp': 1, 'adv': 2, 'aux': 3, 'cconj': 4, 'conj': 5, 'det': 6, 'intj': 7, 'noun': 8,
       'num': 9, 'part': 10, 'pron': 11, 'propn': 12, 'punct': 13, 'sconj': 14, 'sym': 15, 'verb': 16, 'x': 17}
# dependency dict
# https://universaldependencies.org/u/dep/
DEP = {'acl': 0, 'acomp': 1, 'advcl': 2, 'advmod': 3, 'agent': 4, 'amod': 5, 'appos': 6, 'attr': 7, 'aux': 8, 'auxpass': 9, 'case': 10, 'cc': 11, 'ccomp': 12, 'complm': 13, 'compound': 14, 'conj': 15, 'csubj': 16, 'csubjpass': 17, 'dative': 18, 'dep': 19, 'det': 20, 'dobj': 21, 'expl': 22, 'hmod': 23, 'hyph': 24, 'infmod': 25, 'intj': 26, 'iobj': 27, 'mark': 28, 'meta': 29,
       'neg': 30, 'nmod': 31, 'nn': 32, 'nounmod': 33, 'npadvmod': 34, 'npmod': 35, 'nsubj': 36, 'nsubjpass': 37, 'num': 38, 'number': 39, 'nummod': 40, 'oprd': 41, 'parataxis': 42, 'partmod': 43, 'pcomp': 44, 'pobj': 45, 'poss': 46, 'possessive': 47, 'preconj': 48, 'predet': 49, 'prep': 50, 'prt': 51, 'punct': 52, 'quantmod': 53, 'rcmod': 54, 'relcl': 55, 'root': 56, 'xcomp': 57}
USE = hub.load(os.path.join(os.path.abspath(""), "model", "5"))


def path_embedding(edges, use_map):
    path_emb = np.zeros((edges.shape[0], 1100))
    mask = np.zeros((edges.shape[0],), dtype=bool)

    words = np.unique(edges[:, [0, -1]])
    words_missing = list(set(words) - set(use_map.keys()))
    if len(words_missing) > 0:
        words_emb = USE(words_missing).numpy()
        for i, word in enumerate(words_missing):
            use_map[word] = words_emb[i]

    for i, edge in enumerate(edges):
        try:
            pos = np.zeros((len(POS),))
            pos[POS[edge[1]]] = 1.0

            dep = np.zeros((len(DEP),))
            dep[DEP[edge[2]]] = 1.0

            path_emb[i] = np.hstack(
                [use_map[edge[0]], pos, dep, use_map[edge[3]]])
            mask[i] = True

        except Exception as e:
            pass

    path_emb = path_emb[mask]
    return path_emb, use_map


def parse_sentence(x, y, doc, nlp):
    # Get directed and undirected graphs
    graph_edges = []
    for token in doc:
        if x.lower() in token.lower_:
            x = token.lower_ + str(token.i)
        if y.lower() in token.lower_:
            y = token.lower_ + str(token.i)
        for child in token.children:
            graph_edges.append((token.lower_ + str(token.i),
                                child.lower_ + str(child.i)))
    directed_graph = DiGraph(graph_edges)
    undirected_graph = Graph(graph_edges)

    # Shortest path between x and y
    p = []
    sp = shortest_path(undirected_graph, source=x, target=y)
    for token in doc:
        for child in token.children:
            if token.lower_ + str(token.i) in sp and child.lower_ + str(child.i) in sp:
                p.append((token.lemma_.lower(),
                          token.pos_.lower(),
                          child.dep_,
                          child.lemma_.lower()))

    # Descendants of x and y
    xd = sorted(descendants(directed_graph, x), key=lambda z: int(z[-1]))
    yd = sorted(descendants(directed_graph, y), key=lambda z: int(z[-1]))
    for v, d in [(x, xd), (y, yd)]:
        for desc in d:
            vp = shortest_path(directed_graph, source=v, target=desc)
            for t, c in zip(vp, vp[1:]):
                for token in doc:
                    for child in token.children:
                        if token.lower_ + str(token.i) == t and child.lower_ + str(child.i) == c:
                            edge = (token.lemma_.lower(),
                                    token.pos_.lower(),
                                    child.dep_,
                                    child.lemma_.lower())
                            if edge not in p:
                                p.append(edge)
    return np.array(p)

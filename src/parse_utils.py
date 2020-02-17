import os
import spacy
from spacy.matcher import PhraseMatcher
import numpy as np
from networkx import Graph, DiGraph, descendants, shortest_path


np.random.seed(42)


# part of speech dict
# https://universaldependencies.org/u/pos/
POS = {'adj': 0, 'adp': 1, 'adv': 2, 'aux': 3, 'cconj': 4, 'conj': 5, 'det': 6, 'intj': 7, 'noun': 8,
       'num': 9, 'part': 10, 'pron': 11, 'propn': 12, 'punct': 13, 'sconj': 14, 'sym': 15, 'verb': 16, 'x': 17}
# dependency dict
# https://universaldependencies.org/u/dep/
DEP = {'acl': 0, 'acomp': 1, 'advcl': 2, 'advmod': 3, 'agent': 4, 'amod': 5, 'appos': 6, 'attr': 7, 'aux': 8, 'auxpass': 9, 'case': 10, 'cc': 11, 'ccomp': 12, 'complm': 13, 'compound': 14, 'conj': 15, 'csubj': 16, 'csubjpass': 17, 'dative': 18, 'dep': 19, 'det': 20, 'dobj': 21, 'expl': 22, 'hmod': 23, 'hyph': 24, 'infmod': 25, 'intj': 26, 'iobj': 27, 'mark': 28, 'meta': 29,
       'neg': 30, 'nmod': 31, 'nn': 32, 'nounmod': 33, 'npadvmod': 34, 'npmod': 35, 'nsubj': 36, 'nsubjpass': 37, 'num': 38, 'number': 39, 'nummod': 40, 'oprd': 41, 'parataxis': 42, 'partmod': 43, 'pcomp': 44, 'pobj': 45, 'poss': 46, 'possessive': 47, 'preconj': 48, 'predet': 49, 'prep': 50, 'prt': 51, 'punct': 52, 'quantmod': 53, 'rcmod': 54, 'relcl': 55, 'root': 56, 'xcomp': 57}


def parse_sp(x, y, doc, nlp):
    # Find x and y phrase chunks in sentence
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    matcher.add("X", None, nlp(x), nlp(x + "s"))
    matcher.add("Y", None, nlp(y), nlp(y + "s"))
    matches = matcher(doc)

    # Delete overlapping matches
    del_idx = []
    for i in range(len(matches)):
        (match_id_i, start_i, end_i) = matches[i]
        for j in range(i + 1, len(matches)):
            (match_id_j, start_j, end_j) = matches[j]
            if end_i >= start_j and end_i >= end_j:
                del_idx.append(j)
            elif end_i >= start_j and end_i <= end_j:
                if end_i - start_i >= end_j - start_j:
                    del_idx.append(j)
                else:
                    del_idx.append(i)
            else:
                pass
    matches = [match for idx, match in enumerate(
        matches) if idx not in del_idx]
    matches = sorted(matches, key=lambda z: z[1], reverse=True)

    # Choose one chunk for x and one chunk for y
    seen = set()
    matches = [(a, b, c)
               for a, b, c in matches if not (a in seen or seen.add(a))]
    if len(matches) != 2:
        return []

    # Merge x and y chunks
    for (match_id, start, end) in matches:
        string_id = nlp.vocab.strings[match_id]
        if string_id == "X":
            x_span = doc[start:end]
            x_span.merge(x_span.root.tag_, x_span.root.lemma_,
                         x_span.root.ent_type_)
        else:
            y_span = doc[start:end]
            y_span.merge(y_span.root.tag_, y_span.root.lemma_,
                         y_span.root.ent_type_)

    # Track x and y chunks
    x = x_span.lower_ + str(x_span.start)
    y = y_span.lower_ + str(y_span.start)

    #  Get directed and undirected graphs
    graph_edges = []
    for token in doc:
        for child in token.children:
            graph_edges.append((token.lower_ + str(token.i),
                                child.lower_ + str(child.i)))
    undirected_graph = Graph(graph_edges)

    # Shortest path between x and y
    p = []
    sp = shortest_path(undirected_graph, source=x, target=y)
    for token in doc:
        for child in token.children:
            if token.lower_ + str(token.i) in sp and child.lower_ + str(child.i) in sp:
                if token.lower_ == x_span.lower_ and child.lower_ == y_span.lower_:
                    p.append(("x",
                              token.pos_.lower(),
                              child.dep_,
                              "y"))
                elif token.lower_ == y_span.lower_ and child.lower_ == x_span.lower_:
                    p.append(("y",
                              token.pos_.lower(),
                              child.dep_,
                              "x"))
                elif token.lower_ == x_span.lower_:
                    p.append(("x",
                              token.pos_.lower(),
                              child.dep_,
                              child.lemma_.lower()))
                elif child.lower_ == x_span.lower_:
                    p.append((token.lemma_.lower(),
                              token.pos_.lower(),
                              child.dep_,
                              "x"))
                elif token.lower_ == y_span.lower_:
                    p.append(("y",
                              token.pos_.lower(),
                              child.dep_,
                              child.lemma_.lower()))
                elif child.lower_ == y_span.lower_:
                    p.append((token.lemma_.lower(),
                              token.pos_.lower(),
                              child.dep_,
                              "y"))
                else:
                    p.append((token.lemma_.lower(),
                              token.pos_.lower(),
                              child.dep_,
                              child.lemma_.lower()))
    return np.array(p).tolist()

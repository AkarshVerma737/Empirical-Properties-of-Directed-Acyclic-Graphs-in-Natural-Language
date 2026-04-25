"""
depgraph.py
Parses CoNLL-U format treebank files into networkx directed graphs.
"""

import re
import networkx as nx
from dataclasses import dataclass, field
from typing import List, Optional, Dict


@dataclass
class Token:
    id: int
    form: str
    lemma: str
    upos: str
    xpos: str
    feats: str
    head: int
    deprel: str

    @staticmethod
    def from_conllu_line(line: str) -> Optional["Token"]:
        fields = line.strip().split("\t")
        if len(fields) < 8:
            return None
        tok_id = fields[0]
        # Skip multiword tokens (e.g. "1-2") and empty nodes (e.g. "1.1")
        if "-" in tok_id or "." in tok_id:
            return None
        try:
            return Token(
                id=int(tok_id),
                form=fields[1],
                lemma=fields[2],
                upos=fields[3],
                xpos=fields[4],
                feats=fields[5],
                head=int(fields[6]),
                deprel=fields[7],
            )
        except ValueError:
            return None


def parse_conllu_file(filepath: str):
    """
    Generator that yields (sentence_id, list_of_tokens) for each sentence.
    Skips malformed sentences.
    """
    sentence_tokens = []
    sent_id = 0

    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#"):
                continue
            if line.strip() == "":
                if sentence_tokens:
                    yield sent_id, sentence_tokens
                    sent_id += 1
                sentence_tokens = []
            else:
                tok = Token.from_conllu_line(line)
                if tok is not None:
                    sentence_tokens.append(tok)
        if sentence_tokens:
            yield sent_id, sentence_tokens


def tokens_to_digraph(tokens: List[Token]) -> Optional[nx.DiGraph]:
    """
    Converts a list of tokens to a rooted directed graph.
    Edges go from head -> dependent.
    Returns None if the sentence is malformed (no root, multiple roots, etc.)
    """
    if len(tokens) < 3:
        return None

    G = nx.DiGraph()
    root = None

    for tok in tokens:
        G.add_node(tok.id, form=tok.form, upos=tok.upos, deprel=tok.deprel)

    for tok in tokens:
        if tok.head == 0:
            if root is not None:
                return None  # multiple roots — skip
            root = tok.id
            G.graph["root"] = tok.id
        else:
            if tok.head not in G.nodes:
                return None  # head refers to nonexistent node
            G.add_edge(tok.head, tok.id, deprel=tok.deprel)

    if root is None:
        return None
    if not nx.is_weakly_connected(G):
        return None

    return G


def load_treebank(filepath: str, max_sentences: int = None):
    """
    Loads a treebank file and returns a list of valid DiGraphs.
    """
    graphs = []
    for sent_id, tokens in parse_conllu_file(filepath):
        G = tokens_to_digraph(tokens)
        if G is not None:
            graphs.append(G)
        if max_sentences and len(graphs) >= max_sentences:
            break
    return graphs

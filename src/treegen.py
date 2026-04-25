"""
treegen.py
Generates random labeled rooted trees with the same number of nodes as a given tree.
Uses Prüfer sequences for uniform random labeled tree generation.
"""

import numpy as np
import networkx as nx
from typing import Optional


def prufer_to_tree(prufer_seq: list) -> nx.Graph:
    """
    Converts a Prüfer sequence to an undirected labeled tree.
    Nodes are labeled 0..n-1 where n = len(prufer_seq) + 2.
    """
    n = len(prufer_seq) + 2
    degree = [1] * n
    for node in prufer_seq:
        degree[node] += 1

    edges = []
    for node in prufer_seq:
        for leaf in range(n):
            if degree[leaf] == 1:
                edges.append((node, leaf))
                degree[node] -= 1
                degree[leaf] -= 1
                break

    # Last edge
    last_edge = [i for i in range(n) if degree[i] == 1]
    edges.append((last_edge[0], last_edge[1]))

    T = nx.Graph()
    T.add_nodes_from(range(n))
    T.add_edges_from(edges)
    return T


def random_rooted_tree(n: int, root: int = 0) -> nx.DiGraph:
    """
    Generates a uniformly random rooted directed tree with n nodes.
    Root is always node 0. Edges go from parent -> child.
    """
    if n == 1:
        G = nx.DiGraph()
        G.add_node(0)
        G.graph["root"] = 0
        return G

    if n == 2:
        G = nx.DiGraph()
        G.add_edge(0, 1)
        G.graph["root"] = 0
        return G

    prufer_seq = np.random.randint(0, n, size=n - 2).tolist()
    T_undirected = prufer_to_tree(prufer_seq)

    # Root the tree at node 0 using BFS
    T_directed = nx.bfs_tree(T_undirected, source=root)
    T_directed.graph["root"] = root
    return T_directed


def random_tree_matching(real_graph: nx.DiGraph) -> nx.DiGraph:
    """
    Generates a random tree with the same number of nodes as the input graph.
    """
    n = real_graph.number_of_nodes()
    return random_rooted_tree(n)

"""
compute_metrics.py
Computes structural metrics for dependency trees:
  - Arity: number of children per node (branching factor)
  - Depth: longest path from root to leaf
  - Density: edge count / (n * (n-1)) — standard graph density
"""

import networkx as nx
from typing import Dict, List, Optional, Tuple
import numpy as np


def get_root(G: nx.DiGraph) -> Optional[int]:
    """Returns the root node (node with in-degree 0)."""
    if "root" in G.graph:
        return G.graph["root"]
    roots = [n for n, d in G.in_degree() if d == 0]
    return roots[0] if len(roots) == 1 else None


def compute_arity(G: nx.DiGraph) -> List[int]:
    """
    Returns a list of out-degrees (children count) for every node.
    This is the arity / branching factor distribution.
    """
    return [d for _, d in G.out_degree()]


def compute_depth(G: nx.DiGraph) -> int:
    """
    Returns the depth of the tree = longest path from root to any leaf.
    Returns -1 if root cannot be found.
    """
    root = get_root(G)
    if root is None:
        return -1

    # BFS from root, track depth of each node
    max_depth = 0
    queue = [(root, 0)]
    visited = set()

    while queue:
        node, depth = queue.pop(0)
        if node in visited:
            continue
        visited.add(node)
        max_depth = max(max_depth, depth)
        for child in G.successors(node):
            queue.append((child, depth + 1))

    return max_depth


def compute_density(G: nx.DiGraph) -> float:
    """
    Returns the graph density = |E| / (|V| * (|V| - 1)).
    For a tree with n nodes: |E| = n-1, so density = 1/(n).
    Computed generically to allow comparison with random structures.
    """
    n = G.number_of_nodes()
    if n <= 1:
        return 0.0
    return G.number_of_edges() / (n * (n - 1))


def compute_all_metrics(G: nx.DiGraph) -> Dict:
    """
    Computes all metrics for a single graph.
    Returns a dict with arity_list, mean_arity, max_arity, depth, density.
    """
    arity_list = compute_arity(G)
    depth = compute_depth(G)
    density = compute_density(G)

    return {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "arity_list": arity_list,
        "mean_arity": float(np.mean(arity_list)) if arity_list else 0.0,
        "max_arity": int(max(arity_list)) if arity_list else 0,
        "depth": depth,
        "density": density,
    }


def aggregate_metrics(metrics_list: List[Dict]) -> Dict:
    """
    Aggregates metrics across many sentences into flat lists for plotting.
    """
    all_arities = []
    mean_arities = []
    max_arities = []
    depths = []
    densities = []
    n_nodes = []

    for m in metrics_list:
        all_arities.extend(m["arity_list"])
        mean_arities.append(m["mean_arity"])
        max_arities.append(m["max_arity"])
        depths.append(m["depth"])
        densities.append(m["density"])
        n_nodes.append(m["n_nodes"])

    return {
        "all_arities": all_arities,
        "mean_arities": mean_arities,
        "max_arities": max_arities,
        "depths": depths,
        "densities": densities,
        "n_nodes": n_nodes,
    }

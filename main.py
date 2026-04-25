"""
main.py
Full pipeline for LE2: Empirical Properties of Dependency DAGs vs Random Trees.

Usage:
    python main.py --data_dir ./data --results_dir ./results
"""

import os
import sys
import argparse
import json
from typing import Optional
import pandas as pd
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from depgraph import load_treebank
from treegen import random_tree_matching
from compute_metrics import compute_all_metrics, aggregate_metrics
from analysis import run_analysis, build_summary_table
from visualize import (
    plot_language_distributions,
    plot_cross_language_heatmap,
    plot_arity_histogram_grid,
    plot_depth_density_scatter,
)

# ─────────────────────────────────────────────────────────────
# Treebank configuration
# Keys: display name  →  Values: filename inside data_dir
# Download from: https://github.com/surfacesyntacticud/SUD_English-EWT etc.
# ─────────────────────────────────────────────────────────────
TREEBANKS = {
    "English":  "en_ewt-sud-train.conllu",
    "Hindi":    "hi_hdtb-sud-train.conllu",
    "French":   "fr_gsd-sud-train.conllu",
    "German":   "de_gsd-sud-train.conllu",
    "Spanish":  "es_gsd-sud-train.conllu",
    "Chinese":  "zh_gsd-sud-train.conllu",
    "Arabic":   "ar_padt-sud-train.conllu",
    "Russian":  "ru_syntagrus-sud-train.conllu",
    "Japanese": "ja_gsd-sud-train.conllu",
    "Turkish":  "tr_imst-sud-train.conllu",
    "Basque":   "eu_bdt-sud-train.conllu",
}


def process_language(lang: str, filepath: str, max_sentences: Optional[int], results_dir: str):
    print(f"\n{'='*50}")
    print(f"  Processing: {lang}")
    print(f"{'='*50}")

    # 1. Load real trees
    print(f"  Loading treebank from {filepath} ...")
    real_graphs = load_treebank(filepath, max_sentences=max_sentences)
    print(f"  Loaded {len(real_graphs)} valid sentences.")

    if len(real_graphs) < 10:
        print(f"  WARNING: Too few sentences for {lang}, skipping.")
        return None, None

    # 2. Compute metrics on real trees
    print(f"  Computing metrics on real trees ...")
    real_metrics = [compute_all_metrics(G) for G in tqdm(real_graphs, desc="  Real")]

    # 3. Generate matched random trees and compute metrics
    print(f"  Generating random trees and computing metrics ...")
    rand_metrics = []
    for G in tqdm(real_graphs, desc="  Rand"):
        R = random_tree_matching(G)
        rand_metrics.append(compute_all_metrics(R))

    # 4. Aggregate
    real_agg = aggregate_metrics(real_metrics)
    rand_agg = aggregate_metrics(rand_metrics)

    # 5. Per-language violin plot
    plot_language_distributions(real_agg, rand_agg, lang, results_dir)

    return real_agg, rand_agg


def main():
    parser = argparse.ArgumentParser(description="LE2: Dependency DAG vs Random Tree Analysis")
    parser.add_argument("--data_dir",       default="./data",    help="Directory containing .conllu files")
    parser.add_argument("--results_dir",    default="./results", help="Directory to save results and figures")
    parser.add_argument(
        "--max_sentences",
        type=int,
        default=None,
        help="Max sentences per language (default: use all sentences in each corpus)",
    )
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    fig_dir = os.path.join(args.results_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    all_real = {}
    all_rand = {}
    all_analysis = []

    for lang, filename in TREEBANKS.items():
        filepath = os.path.join(args.data_dir, filename)
        if not os.path.exists(filepath):
            print(f"  SKIP {lang}: file not found at {filepath}")
            continue

        real_agg, rand_agg = process_language(lang, filepath, args.max_sentences, fig_dir)
        if real_agg is None:
            continue

        all_real[lang] = real_agg
        all_rand[lang] = rand_agg
        all_analysis.append(run_analysis(real_agg, rand_agg, lang))

    if not all_analysis:
        print("\nNo languages processed. Check that .conllu files exist in --data_dir.")
        return

    # Cross-language plots
    print("\nGenerating cross-language plots ...")
    summary_df = build_summary_table(all_analysis)
    plot_cross_language_heatmap(summary_df, fig_dir)
    plot_arity_histogram_grid(all_real, all_rand, fig_dir)
    plot_depth_density_scatter(all_real, all_rand, fig_dir)

    # Save summary table
    summary_path = os.path.join(args.results_dir, "summary_table.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary table saved: {summary_path}")
    print("\n" + summary_df.to_string(index=False))

    # Save raw JSON results
    raw_path = os.path.join(args.results_dir, "raw_results.json")
    with open(raw_path, "w") as f:
        # Convert numpy types for JSON serialization
        def convert(o):
            if isinstance(o, (int, float, str, bool)) or o is None:
                return o
            if isinstance(o, list):
                return [convert(i) for i in o]
            if isinstance(o, dict):
                return {k: convert(v) for k, v in o.items()}
            return float(o)
        json.dump([convert(r) for r in all_analysis], f, indent=2)
    print(f"Raw results saved: {raw_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()

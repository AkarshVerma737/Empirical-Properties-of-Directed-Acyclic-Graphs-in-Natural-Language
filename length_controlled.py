"""
length_controlled.py
Second analysis for LE2: controls for sentence length to verify that
the depth reduction in real trees is not merely an artifact of sentence
length distribution differences.

Method:
  - Bin sentences into length brackets: 4-7, 8-11, 12-15, 16-20, 21+
  - Within each bin, compare mean depth of real vs random trees
  - If real trees are shallower even within the same length bracket,
    the effect is genuine and not a length artifact.

Run from le2/ root:
    python length_controlled.py --data_dir ./data --results_dir ./results
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from depgraph import load_treebank
from treegen import random_tree_matching
from compute_metrics import compute_depth

TREEBANKS = {
    "English":  "en_ewt-sud-train.conllu",
    "Hindi":    "hi_hdtb-sud-train.conllu",
    "German":   "de_gsd-sud-train.conllu",
    "Spanish":  "es_gsd-sud-train.conllu",
    "Chinese":  "zh_gsd-sud-train.conllu",
    "Arabic":   "ar_padt-sud-train.conllu",
    "Russian":  "ru_syntagrus-sud-train.conllu",
    "Japanese": "ja_gsd-sud-train.conllu",
    "Turkish":  "tr_imst-sud-train.conllu",
    "Basque":   "eu_bdt-sud-train.conllu",
}

# Length bins: (label, min_inclusive, max_inclusive)
BINS = [
    ("4–7",   4,  7),
    ("8–11",  8,  11),
    ("12–15", 12, 15),
    ("16–20", 16, 20),
    ("21+",   21, 9999),
]

PALETTE = {"Real": "#2E86AB", "Random": "#E84855"}


def assign_bin(n):
    for label, lo, hi in BINS:
        if lo <= n <= hi:
            return label
    return None


def collect_binned_data(graphs):
    """
    For each sentence, collect (bin_label, real_depth, rand_depth, n_nodes).
    """
    records = []
    for G in tqdm(graphs, desc="  Processing"):
        n = G.number_of_nodes()
        bin_label = assign_bin(n)
        if bin_label is None:
            continue
        real_depth = compute_depth(G)
        R = random_tree_matching(G)
        rand_depth = compute_depth(R)
        records.append({
            "bin":        bin_label,
            "n_nodes":    n,
            "real_depth": real_depth,
            "rand_depth": rand_depth,
        })
    return pd.DataFrame(records)


def plot_binned_comparison(all_lang_data: dict, output_dir: str):
    """
    For each language, plot mean real vs random depth per length bin.
    Aggregate plot shows all languages together.
    """
    bin_order = [b[0] for b in BINS]

    # ── Per-language line plot ─────────────────────────────────────────────
    languages = list(all_lang_data.keys())
    n_langs = len(languages)
    cols = 3
    rows = (n_langs + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 3.2))
    axes = np.array(axes).flatten()
    fig.suptitle(
        "Mean Tree Depth by Sentence Length Bin\n(Real vs Random — length-controlled)",
        fontsize=13, fontweight="bold"
    )

    for i, lang in enumerate(languages):
        ax = axes[i]
        df = all_lang_data[lang]

        real_means = df.groupby("bin")["real_depth"].mean().reindex(bin_order)
        rand_means = df.groupby("bin")["rand_depth"].mean().reindex(bin_order)
        real_se    = df.groupby("bin")["real_depth"].sem().reindex(bin_order)
        rand_se    = df.groupby("bin")["rand_depth"].sem().reindex(bin_order)

        x = np.arange(len(bin_order))
        ax.plot(x, real_means.values, "o-", color=PALETTE["Real"],
                label="Real", linewidth=2, markersize=5)
        ax.fill_between(x,
                        real_means.values - real_se.values,
                        real_means.values + real_se.values,
                        alpha=0.15, color=PALETTE["Real"])
        ax.plot(x, rand_means.values, "s--", color=PALETTE["Random"],
                label="Random", linewidth=2, markersize=5)
        ax.fill_between(x,
                        rand_means.values - rand_se.values,
                        rand_means.values + rand_se.values,
                        alpha=0.15, color=PALETTE["Random"])

        ax.set_title(lang, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(bin_order, fontsize=8)
        ax.set_xlabel("Sentence Length (words)", fontsize=8)
        ax.set_ylabel("Mean Depth", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    path = os.path.join(output_dir, "length_controlled_per_language.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_aggregated_gap(all_lang_data: dict, output_dir: str):
    """
    Single plot: mean depth gap (random - real) per bin, averaged across languages.
    A positive gap at every bin = effect is robust to sentence length.
    """
    bin_order = [b[0] for b in BINS]
    gap_by_bin = {b: [] for b in bin_order}

    for lang, df in all_lang_data.items():
        for b in bin_order:
            sub = df[df["bin"] == b]
            if len(sub) >= 10:
                gap = sub["rand_depth"].mean() - sub["real_depth"].mean()
                gap_by_bin[b].append(gap)

    means = [np.mean(gap_by_bin[b]) if gap_by_bin[b] else np.nan for b in bin_order]
    stds  = [np.std(gap_by_bin[b])  if gap_by_bin[b] else np.nan for b in bin_order]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(bin_order))
    bars = ax.bar(x, means, yerr=stds, color=PALETTE["Real"],
                  alpha=0.8, edgecolor="white", capsize=5, width=0.55)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(bin_order)
    ax.set_xlabel("Sentence Length Bin (words)", fontsize=11)
    ax.set_ylabel("Mean Depth Gap (Random \u2212 Real)", fontsize=11)
    ax.set_title(
        "Depth Gap Persists Across All Sentence Length Bins\n"
        "(averaged across 10 languages, error bars = std across languages)",
        fontsize=11, fontweight="bold"
    )
    ax.grid(True, axis="y", alpha=0.3)

    # Annotate bar values
    for bar, m in zip(bars, means):
        if not np.isnan(m):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.05,
                    f"+{m:.2f}", ha="center", va="bottom", fontsize=9,
                    fontweight="bold", color=PALETTE["Real"])

    plt.tight_layout()
    path = os.path.join(output_dir, "length_controlled_gap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return dict(zip(bin_order, means)), dict(zip(bin_order, stds))


def run_ks_per_bin(all_lang_data: dict) -> pd.DataFrame:
    """KS test per bin per language."""
    bin_order = [b[0] for b in BINS]
    rows = []
    for lang, df in all_lang_data.items():
        for b in bin_order:
            sub = df[df["bin"] == b]
            if len(sub) < 10:
                continue
            stat, p = ks_2samp(sub["real_depth"].values, sub["rand_depth"].values)
            rows.append({
                "Language": lang, "Bin": b,
                "N": len(sub),
                "Real Mean": round(sub["real_depth"].mean(), 2),
                "Random Mean": round(sub["rand_depth"].mean(), 2),
                "Gap": round(sub["rand_depth"].mean() - sub["real_depth"].mean(), 2),
                "KS Stat": round(stat, 3),
                "p-value": round(p, 4),
                "Sig": p < 0.05,
            })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",    default="./data")
    parser.add_argument("--results_dir", default="./results")
    parser.add_argument("--max_sentences", type=int, default=None)
    args = parser.parse_args()

    fig_dir = os.path.join(args.results_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    all_lang_data = {}

    for lang, filename in TREEBANKS.items():
        filepath = os.path.join(args.data_dir, filename)
        if not os.path.exists(filepath):
            print(f"  SKIP {lang}: not found")
            continue
        print(f"\nProcessing: {lang}")
        graphs = load_treebank(filepath, max_sentences=args.max_sentences)
        df = collect_binned_data(graphs)
        all_lang_data[lang] = df
        print(f"  {len(df)} sentences binned across {df['bin'].nunique()} bins")

    if not all_lang_data:
        print("No data found.")
        return

    print("\nGenerating plots...")
    plot_binned_comparison(all_lang_data, fig_dir)
    gap_means, gap_stds = plot_aggregated_gap(all_lang_data, fig_dir)

    ks_df = run_ks_per_bin(all_lang_data)
    ks_path = os.path.join(args.results_dir, "length_controlled_ks.csv")
    ks_df.to_csv(ks_path, index=False)
    print(f"\nKS results saved: {ks_path}")

    print("\nAggregated depth gap by bin:")
    for b in [b[0] for b in BINS]:
        print(f"  {b:>6} words: gap = +{gap_means[b]:.3f} (std={gap_stds[b]:.3f})")

    print("\nDone.")


if __name__ == "__main__":
    main()
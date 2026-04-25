"""
visualize.py
Generates all plots for the LE2 analysis:
 - Per-language violin plots for arity, depth, density (real vs random)
 - Cross-language heatmap of mean values
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List
import os

PALETTE = {"Real": "#2E86AB", "Random": "#E84855"}
sns.set_theme(style="whitegrid", font_scale=1.1)


def plot_language_distributions(
    real_agg: Dict,
    rand_agg: Dict,
    language: str,
    output_dir: str,
):
    """
    For one language, plots violin/box distributions of arity, depth, density.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"{language} — Real vs Random Trees", fontsize=14, fontweight="bold")

    metrics = [
        ("all_arities", "Arity (children per node)", axes[0]),
        ("depths",      "Tree Depth",                axes[1]),
        ("densities",   "Graph Density",             axes[2]),
    ]

    for key, label, ax in metrics:
        real_vals = real_agg[key]
        rand_vals = rand_agg[key]

        # Cap arity at 95th percentile for readability
        if key == "all_arities":
            cap = max(np.percentile(real_vals, 95), np.percentile(rand_vals, 95))
            real_vals = [v for v in real_vals if v <= cap]
            rand_vals = [v for v in rand_vals if v <= cap]

        df_plot = pd.DataFrame({
            "Value": real_vals + rand_vals,
            "Type":  ["Real"] * len(real_vals) + ["Random"] * len(rand_vals),
        })

        sns.violinplot(
            data=df_plot, x="Type", y="Value", hue="Type",
            palette=PALETTE, inner="box", ax=ax, cut=0, legend=False,
        )
        ax.set_title(label)
        ax.set_xlabel("")
        ax.set_ylabel(label)

    plt.tight_layout()
    path = os.path.join(output_dir, f"{language}_distributions.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_cross_language_heatmap(summary_df: pd.DataFrame, output_dir: str):
    """
    Heatmap: rows = languages, cols = metrics, cells = Real Mean / Random Mean ratio.
    Values > 1 mean real > random; < 1 mean real < random.
    """
    pivot_real = summary_df.pivot(index="Language", columns="Metric", values="Real Mean")
    pivot_rand = summary_df.pivot(index="Language", columns="Metric", values="Random Mean")
    ratio = pivot_real / pivot_rand

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Cross-Language: Real / Random Mean Ratios\n(>1 = Real larger, <1 = Real smaller)",
                 fontsize=13, fontweight="bold")

    for ax, metric in zip(axes, ["arity", "depth", "density"]):
        data = summary_df[summary_df["Metric"] == metric].copy()
        data = data.set_index("Language")[["Real Mean", "Random Mean"]]
        data.plot(kind="barh", ax=ax, color=[PALETTE["Real"], PALETTE["Random"]],
                  edgecolor="white", width=0.7)
        ax.set_title(f"{metric.capitalize()}", fontweight="bold")
        ax.set_xlabel("Mean Value")
        ax.set_ylabel("")
        ax.legend(loc="lower right", fontsize=9)

    plt.tight_layout()
    path = os.path.join(output_dir, "cross_language_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_arity_histogram_grid(
    all_real: Dict[str, Dict],
    all_rand: Dict[str, Dict],
    output_dir: str,
):
    """
    Grid plot: one row per language, showing arity histogram (real vs random).
    """
    languages = list(all_real.keys())
    n = len(languages)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3.5))
    axes = np.array(axes).flatten()
    fig.suptitle("Arity Distribution: Real (blue) vs Random (red)", fontsize=14, fontweight="bold")

    for i, lang in enumerate(languages):
        ax = axes[i]
        real_vals = all_real[lang]["all_arities"]
        rand_vals = all_rand[lang]["all_arities"]

        cap = int(max(np.percentile(real_vals, 97), np.percentile(rand_vals, 97))) + 1
        bins = range(0, cap + 1)

        ax.hist(real_vals,   bins=bins, alpha=0.6, color=PALETTE["Real"],   label="Real",   density=True)
        ax.hist(rand_vals,   bins=bins, alpha=0.6, color=PALETTE["Random"], label="Random", density=True)
        ax.set_title(lang, fontweight="bold")
        ax.set_xlabel("# Children")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    path = os.path.join(output_dir, "arity_histogram_grid.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_depth_density_scatter(
    all_real: Dict[str, Dict],
    all_rand: Dict[str, Dict],
    output_dir: str,
):
    """
    Scatter of mean depth vs mean density per language, real and random.
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    for lang in all_real:
        r_depth   = np.mean(all_real[lang]["depths"])
        r_density = np.mean(all_real[lang]["densities"])
        n_depth   = np.mean(all_rand[lang]["depths"])
        n_density = np.mean(all_rand[lang]["densities"])

        ax.scatter(r_depth, r_density, color=PALETTE["Real"],   s=80, zorder=3)
        ax.scatter(n_depth, n_density, color=PALETTE["Random"], s=80, zorder=3)
        ax.annotate(lang, (r_depth, r_density), fontsize=7, ha="right")
        ax.plot([r_depth, n_depth], [r_density, n_density], "gray", lw=0.8, alpha=0.5)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=PALETTE["Real"],   markersize=10, label="Real"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=PALETTE["Random"], markersize=10, label="Random"),
    ]
    ax.legend(handles=legend_elements)
    ax.set_xlabel("Mean Tree Depth")
    ax.set_ylabel("Mean Graph Density")
    ax.set_title("Depth vs Density per Language (Real vs Random)", fontweight="bold")

    plt.tight_layout()
    path = os.path.join(output_dir, "depth_density_scatter.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

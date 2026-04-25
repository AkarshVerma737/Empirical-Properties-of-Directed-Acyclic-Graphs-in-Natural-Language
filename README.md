# LE2 — Empirical Properties of DAGs in Natural Language

**Authors:** Akarsh Verma (220094), Amit Kumar (228070125), Ayush Meena (220268)

Empirical analysis comparing structural properties of natural language dependency trees
against randomly generated trees across 10 typologically diverse languages.

---

## Project Structure

```
Empirical-Properties-of-Directed-Acyclic-Graphs-in-Natural-Language/
├── main.py                   # Main pipeline: runs full analysis for all languages
├── length_controlled.py      # Second analysis: sentence-length-controlled depth comparison
├── download_data.sh          # Downloads all SUD treebanks from grew.fr
├── requirements.txt          # Python dependencies
│
├── src/
│   ├── depgraph.py           # Parses CoNLL-U files into NetworkX DiGraphs
│   ├── treegen.py            # Generates random trees via Prüfer sequences
│   ├── compute_metrics.py    # Computes arity, depth, and density per tree
│   ├── analysis.py           # KS tests and summary statistics
│   └── visualize.py          # All plots: violin plots, histograms, scatter, heatmap
│
├── data/                     # Downloaded .conllu treebank files (created by download_data.sh)
│
└── results/
    ├── summary_table.csv         # Main results: mean/std/KS per language per metric
    ├── raw_results.json          # Full aggregated stats per language
    ├── length_controlled_ks.csv  # KS tests per language per length bin
    └── figures/
        ├── arity_histogram_grid.png
        ├── cross_language_comparison.png
        ├── depth_density_scatter.png
        ├── {Language}_distributions.png     # One per language (10 files)
        ├── length_controlled_per_language.png
        └── length_controlled_gap.png
```

---

## Setup

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Download treebanks**
```bash
bash download_data.sh
```
Downloads SUD v2.17 treebanks for 11 languages from `grew.fr` into `./data/`.

---

## Running the Analysis

### Main analysis (arity, depth, density)
```bash
python main.py
```
Runs the full pipeline on all languages. Saves all figures and tables to `./results/`.

Optional flags:
```bash
python main.py --data_dir ./data --results_dir ./results --max_sentences X
```
| Flag | Default | Description |
|---|---|---|
| `--data_dir` | `./data` | Directory containing `.conllu` files |
| `--results_dir` | `./results` | Output directory for figures and CSVs |
| `--max_sentences` | `None` (all) | Cap sentences per language (use for fast testing) |


### Length-controlled analysis
```bash
python length_controlled.py
```
Bins sentences by length (4–7, 8–11, 12–15, 16–20, 21+ words) and compares real vs.
random depth within each bin to rule out sentence-length confounds.

Produces:
- `results/figures/length_controlled_per_language.png`
- `results/figures/length_controlled_gap.png`
- `results/length_controlled_ks.csv`

Optional flags: same `--data_dir`, `--results_dir`, `--max_sentences` as `main.py`.

---

## Module Descriptions

### `src/depgraph.py`
Parses CoNLL-U format treebank files into NetworkX `DiGraph` objects.

- `parse_conllu_file(filepath)` — generator yielding `(sent_id, tokens)` per sentence
- `tokens_to_digraph(tokens)` — converts token list to directed graph; returns `None` for malformed sentences (no root, multiple roots, disconnected)
- `load_treebank(filepath, max_sentences)` — loads a full treebank file, returns list of valid `DiGraph` objects

Filters applied: multiword tokens (`1-2` style IDs), empty nodes (`.` style IDs), missing/multiple roots, disconnected graphs, sentences with fewer than 3 tokens.

### `src/treegen.py`
Generates uniformly random labeled trees using Prüfer sequences.

- `prufer_to_tree(prufer_seq)` — converts a Prüfer sequence to an undirected `nx.Graph`
- `random_rooted_tree(n, root=0)` — generates a random rooted directed tree with `n` nodes
- `random_tree_matching(real_graph)` — generates a random tree with the same node count as the input graph

A Prüfer sequence of length `n-2` drawn uniformly from `{0,...,n-1}` encodes a uniformly
random labeled tree on `n` nodes. The undirected tree is then rooted at node 0 via BFS.

### `src/compute_metrics.py`
Computes three structural metrics for any directed graph.

- `compute_arity(G)` — returns list of out-degrees for all nodes
- `compute_depth(G)` — BFS from root; returns length of longest root-to-leaf path
- `compute_density(G)` — returns `|E| / (|V| * (|V|-1))`
- `compute_all_metrics(G)` — returns dict with all metrics for one graph
- `aggregate_metrics(metrics_list)` — flattens per-sentence metrics into lists for plotting

### `src/analysis.py`
Statistical testing and summary table generation.

- `ks_test(real, random, metric_name)` — two-sample KS test via `scipy.stats.ks_2samp`
- `run_analysis(real_agg, rand_agg, language)` — runs KS tests for all three metrics
- `build_summary_table(all_results)` — builds cross-language `pandas.DataFrame`

### `src/visualize.py`
All plotting functions.

- `plot_language_distributions(real_agg, rand_agg, language, output_dir)` — violin plots per language
- `plot_cross_language_heatmap(summary_df, output_dir)` — bar chart comparing real/random means
- `plot_arity_histogram_grid(all_real, all_rand, output_dir)` — grid of arity histograms
- `plot_depth_density_scatter(all_real, all_rand, output_dir)` — depth vs density scatter

### `main.py`
Orchestrates the full pipeline:
1. Loads each treebank via `depgraph.load_treebank`
2. Computes metrics on real trees
3. Generates matched random trees and computes metrics
4. Runs KS tests via `analysis.run_analysis`
5. Generates all plots via `visualize`
6. Saves `summary_table.csv` and `raw_results.json`

### `length_controlled.py`
Length-controlled second analysis:
1. Loads treebanks (same as `main.py`)
2. Assigns each sentence to a length bin
3. Computes real and random depth per sentence
4. Computes mean depth gap (random − real) per bin per language
5. Generates per-language line plots and aggregated bar chart
6. Saves KS tests per bin to `length_controlled_ks.csv`

---

## Languages Analysed

| Language | Treebank | Typology |
|---|---|---|
| English | SUD_English-EWT | SVO, analytic |
| Hindi | SUD_Hindi-HDTB | SOV, moderately inflected |
| French | SUD_French-GSD | SVO, analytic |
| German | SUD_German-GSD | SOV/V2, inflected |
| Spanish | SUD_Spanish-GSD | SVO, inflected |
| Chinese | SUD_Chinese-GSD | SVO, isolating |
| Arabic | SUD_Arabic-PADT | VSO, heavily inflected |
| Russian | SUD_Russian-SynTagRus | free order, highly inflected |
| Japanese | SUD_Japanese-GSD | SOV, head-final, agglutinative |
| Turkish | SUD_Turkish-IMST | SOV, agglutinative |
| Basque | SUD_Basque-BDT | SOV, ergative, agglutinative |

---

## Expected Output

After running both scripts, `results/figures/` will contain 15 figures:

| File | Description |
|---|---|
| `arity_histogram_grid.png` | Arity distributions for all 10 languages |
| `cross_language_comparison.png` | Bar chart of real vs random means across languages |
| `depth_density_scatter.png` | Depth vs density scatter per language |
| `{Lang}_distributions.png` (×10) | Per-language violin plots |
| `length_controlled_per_language.png` | Depth by length bin per language |
| `length_controlled_gap.png` | Aggregated depth gap per length bin |
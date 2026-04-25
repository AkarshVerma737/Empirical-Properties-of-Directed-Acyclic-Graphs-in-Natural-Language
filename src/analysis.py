"""
analysis.py
Statistical comparison between real and random tree metrics using KS tests.
"""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from typing import Dict, List


def summary_stats(values: List[float], label: str) -> Dict:
    arr = np.array(values)
    return {
        "label": label,
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def ks_test(real: List[float], random: List[float], metric_name: str) -> Dict:
    stat, p = ks_2samp(real, random)
    return {
        "metric": metric_name,
        "ks_stat": round(stat, 4),
        "p_value": round(p, 6),
        "significant": p < 0.05,
    }


def run_analysis(real_agg: Dict, rand_agg: Dict, language: str) -> Dict:
    """
    Runs full statistical analysis for one language.
    Returns summary stats and KS test results for all 3 metrics.
    """
    results = {"language": language, "stats": {}, "ks_tests": []}

    for metric, real_key, rand_key in [
        ("arity",    "all_arities", "all_arities"),
        ("depth",    "depths",      "depths"),
        ("density",  "densities",   "densities"),
    ]:
        real_vals = real_agg[real_key]
        rand_vals = rand_agg[rand_key]

        results["stats"][f"real_{metric}"] = summary_stats(real_vals, f"real_{metric}")
        results["stats"][f"rand_{metric}"] = summary_stats(rand_vals, f"rand_{metric}")
        results["ks_tests"].append(ks_test(real_vals, rand_vals, metric))

    return results


def build_summary_table(all_results: List[Dict]) -> pd.DataFrame:
    """
    Builds a cross-language summary DataFrame.
    """
    rows = []
    for r in all_results:
        lang = r["language"]
        for metric in ["arity", "depth", "density"]:
            real_s = r["stats"][f"real_{metric}"]
            rand_s = r["stats"][f"rand_{metric}"]
            ks = next(x for x in r["ks_tests"] if x["metric"] == metric)
            rows.append({
                "Language": lang,
                "Metric": metric,
                "Real Mean": round(real_s["mean"], 3),
                "Real Std": round(real_s["std"], 3),
                "Random Mean": round(rand_s["mean"], 3),
                "Random Std": round(rand_s["std"], 3),
                "KS Stat": ks["ks_stat"],
                "p-value": ks["p_value"],
                "Significant": ks["significant"],
            })
    return pd.DataFrame(rows)

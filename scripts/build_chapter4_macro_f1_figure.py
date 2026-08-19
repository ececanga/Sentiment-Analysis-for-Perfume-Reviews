#!/usr/bin/env python3
"""Build Figure 4.1 from the frozen five-model thesis result table."""

from __future__ import annotations

from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.apply_verified_results_v1_to_v2_thesis import verify_results_package


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "outputs/thesis_consolidated_results_v1"
SOURCE = RESULTS / "synthetic_five_model_pooled.csv"
OUTPUT = PROJECT_ROOT / "docs/thesis/figures/figure4.1_five_model_macro_f1.png"


def main() -> None:
    verify_results_package()
    frame = pd.read_csv(SOURCE)
    frame = frame.loc[frame["metric"].eq("macro_f1")].sort_values("model_order")
    if frame.shape[0] != 5 or frame["model_id"].nunique() != 5:
        raise ValueError("Expected one macro-F1 row for each of five models")

    labels = [
        "Logistic\nRegression",
        "XGBoost",
        "Random\nForest",
        "BERT",
        "BiLSTM with\nAttention",
    ]
    youtube_only = frame["youtube_only"].to_numpy(dtype=float)
    augmented = frame["youtube_plus_weighted_synthetic_784x3"].to_numpy(dtype=float)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 10,
            "axes.labelsize": 11,
            "legend.fontsize": 9.5,
        }
    )
    figure, axis = plt.subplots(figsize=(8.0, 4.75), constrained_layout=True)
    x = np.arange(len(labels))
    width = 0.36

    bars_youtube = axis.bar(
        x - width / 2,
        youtube_only,
        width,
        label="YouTube-only training",
        color="#4C78A8",
        edgecolor="#1F2937",
        linewidth=0.7,
        hatch="//",
    )
    bars_augmented = axis.bar(
        x + width / 2,
        augmented,
        width,
        label="YouTube + weighted synthetic training",
        color="#F2A65A",
        edgecolor="#1F2937",
        linewidth=0.7,
        hatch="..",
    )

    axis.set_ylabel("Macro-F1 on held-out YouTube perfumes")
    axis.set_xticks(x, labels)
    axis.set_ylim(0.0, 0.46)
    axis.set_yticks(np.arange(0.0, 0.46, 0.05))
    axis.grid(axis="y", color="#D1D5DB", linewidth=0.6, alpha=0.8)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=2, frameon=False)

    for bars in (bars_youtube, bars_augmented):
        for bar in bars:
            value = bar.get_height()
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.006,
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=8.4,
            )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(figure)
    print(f"Created {OUTPUT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build an SVG table for randomized synthetic-only review-count sweep metrics."""

from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = (
    ROOT
    / "outputs/synthetic_only_random_review_count_sweep_gemini_browser"
    / "synthetic_only_random_review_count_sweep_summary.csv"
)
DEFAULT_OUT_DIR = ROOT / "outputs/synthetic_only_random_review_count_sweep_gemini_browser/figures"

COLUMNS = [
    ("model_family", "Model", 210),
    ("synthetic_reviews", "Synthetic reviews", 150),
    ("synthetic_perfumes", "Perfumes", 105),
    ("runs", "Random orders", 120),
    ("macro_f1", "F1 mean +/- SD", 150),
    ("macro_roc_auc", "ROC-AUC mean +/- SD", 170),
    ("macro_pr_auc", "PR-AUC mean +/- SD", 160),
]

MODEL_COLORS = {
    "Logistic Regression": "#2563eb",
    "XGBoost": "#dc2626",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build randomized synthetic-only review-count sweep table as SVG.")
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return sorted(rows, key=lambda row: (row["model_family"], int(row["synthetic_reviews"])))


def fmt(row: dict[str, str], key: str) -> str:
    if key in {"model_family", "synthetic_reviews", "synthetic_perfumes", "runs"}:
        return str(row[key])
    mean_value = float(row[f"{key}_mean"])
    std_value = float(row[f"{key}_std"])
    return f"{mean_value:.4f} +/- {std_value:.4f}"


def text(
    *,
    x: float,
    y: float,
    value: str,
    size: int = 13,
    weight: str = "400",
    color: str = "#111827",
    anchor: str = "start",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{color}" text-anchor="{anchor}">'
        f"{html.escape(value)}</text>"
    )


def build_svg(rows: list[dict[str, str]]) -> str:
    margin_x = 34
    top = 32
    title_h = 74
    row_h = 34
    header_h = 40
    table_w = sum(width for _, _, width in COLUMNS)
    width = table_w + margin_x * 2
    height = top + title_h + header_h + len(rows) * row_h + 54

    x_positions = [margin_x]
    for _, _, col_w in COLUMNS[:-1]:
        x_positions.append(x_positions[-1] + col_w)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        text(x=margin_x, y=top + 20, value="Randomized Synthetic-Only Review Count Sweep Metrics", size=24, weight="700"),
        text(
            x=margin_x,
            y=top + 48,
            value="Mean +/- SD across random perfume orders; folds are grouped by perfume_id.",
            size=13,
            color="#374151",
        ),
    ]

    table_y = top + title_h
    parts.append(f'<rect x="{margin_x}" y="{table_y}" width="{table_w}" height="{header_h}" fill="#e8eef8"/>')
    for col_index, ((_, label, col_w), col_x) in enumerate(zip(COLUMNS, x_positions)):
        parts.append(
            f'<rect x="{col_x}" y="{table_y}" width="{col_w}" height="{header_h}" '
            'fill="none" stroke="#cbd5e1" stroke-width="1"/>'
        )
        anchor = "start" if col_index == 0 else "middle"
        text_x = col_x + 10 if col_index == 0 else col_x + col_w / 2
        parts.append(text(x=text_x, y=table_y + 25, value=label, size=12, weight="700", anchor=anchor))

    best_by_model_metric: set[tuple[str, str, str]] = set()
    for model in {row["model_family"] for row in rows}:
        model_rows = [row for row in rows if row["model_family"] == model]
        for metric in ["macro_f1", "macro_roc_auc", "macro_pr_auc"]:
            best = max(model_rows, key=lambda row: float(row[f"{metric}_mean"]))
            best_by_model_metric.add((model, best["synthetic_reviews"], metric))

    y = table_y + header_h
    previous_model = ""
    for row_index, row in enumerate(rows):
        model = row["model_family"]
        base_fill = "#ffffff" if row_index % 2 == 0 else "#f8fafc"
        if model != previous_model and previous_model:
            parts.append(
                f'<line x1="{margin_x}" y1="{y}" x2="{margin_x + table_w}" y2="{y}" '
                'stroke="#64748b" stroke-width="1.5"/>'
            )
        parts.append(f'<rect x="{margin_x}" y="{y}" width="{table_w}" height="{row_h}" fill="{base_fill}"/>')

        for col_index, ((key, _, col_w), col_x) in enumerate(zip(COLUMNS, x_positions)):
            value = fmt(row, key)
            metric_key = key if key.startswith("macro_") else ""
            is_best_metric = (model, row["synthetic_reviews"], metric_key) in best_by_model_metric
            color = MODEL_COLORS.get(model, "#111827") if col_index == 0 else "#111827"
            weight = "700" if is_best_metric or col_index == 0 else "400"
            if col_index == 0:
                anchor = "start"
                text_x = col_x + 10
            else:
                anchor = "middle"
                text_x = col_x + col_w / 2
            parts.append(
                f'<rect x="{col_x}" y="{y}" width="{col_w}" height="{row_h}" '
                'fill="none" stroke="#e2e8f0" stroke-width="1"/>'
            )
            parts.append(text(x=text_x, y=y + 22, value=value, size=12, weight=weight, color=color, anchor=anchor))
        previous_model = model
        y += row_h

    parts.append(
        text(
            x=margin_x,
            y=height - 20,
            value="Bold metric values mark the best mean value for that model within each metric.",
            size=12,
            color="#374151",
        )
    )
    parts.append("</svg>\n")
    return "\n".join(parts)


def main() -> None:
    args = parse_args()
    rows = load_rows(args.summary_csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "synthetic_only_random_review_count_sweep_table.svg"
    out_path.write_text(build_svg(rows), encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()

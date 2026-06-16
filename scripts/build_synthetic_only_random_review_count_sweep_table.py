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

MODEL_COLORS = {
    "Logistic Regression": "#2563eb",
    "XGBoost": "#dc2626",
}

METRICS = [
    ("macro_f1", "F1"),
    ("macro_roc_auc", "ROC-AUC"),
    ("macro_pr_auc", "PR-AUC"),
]

TABLE_COLUMNS = [
    ("synthetic_reviews", "Reviews", 86),
    ("synthetic_perfumes", "Perfumes", 82),
    ("macro_f1", "F1", 112),
    ("macro_roc_auc", "ROC-AUC", 128),
    ("macro_pr_auc", "PR-AUC", 128),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build randomized synthetic-only review-count sweep table as SVG.")
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return sorted(rows, key=lambda row: (row["model_family"], int(row["synthetic_reviews"])))


def fmt_mean(row: dict[str, str], key: str) -> str:
    return f"{float(row[f'{key}_mean']):.4f}"


def fmt_sd(row: dict[str, str], key: str) -> str:
    return f"SD {float(row[f'{key}_std']):.4f}"


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


def rows_by_model(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["model_family"], []).append(row)
    for model_rows in grouped.values():
        model_rows.sort(key=lambda row: int(row["synthetic_reviews"]))
    return grouped


def column_positions(start_x: float) -> list[float]:
    positions = [start_x]
    for _, _, col_w in TABLE_COLUMNS[:-1]:
        positions.append(positions[-1] + col_w)
    return positions


def render_model_panel(
    *,
    model: str,
    model_rows: list[dict[str, str]],
    x: float,
    y: float,
    width: float,
    row_h: float,
    header_h: float,
) -> list[str]:
    panel_h = 48 + header_h + len(model_rows) * row_h + 40
    table_w = sum(width for _, _, width in TABLE_COLUMNS)
    col_xs = column_positions(x)
    color = MODEL_COLORS.get(model, "#111827")

    best_metric_counts = {
        metric: max(model_rows, key=lambda row: float(row[f"{metric}_mean"]))["synthetic_reviews"]
        for metric, _ in METRICS
    }

    parts = [
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{panel_h:.1f}" fill="#ffffff" stroke="#cbd5e1"/>',
        text(x=x + 14, y=y + 28, value=model, size=18, weight="700", color=color),
        text(x=x + width - 14, y=y + 27, value=f'{model_rows[0]["runs"]} random orders', size=12, color="#4b5563", anchor="end"),
    ]

    header_y = y + 48
    parts.append(f'<rect x="{x}" y="{header_y:.1f}" width="{table_w}" height="{header_h}" fill="#e8eef8"/>')
    for col_index, ((_, label, col_w), col_x) in enumerate(zip(TABLE_COLUMNS, col_xs)):
        parts.append(
            f'<rect x="{col_x:.1f}" y="{header_y:.1f}" width="{col_w}" height="{header_h}" '
            'fill="none" stroke="#cbd5e1" stroke-width="1"/>'
        )
        anchor = "start" if col_index == 0 else "middle"
        text_x = col_x + 10 if col_index == 0 else col_x + col_w / 2
        parts.append(text(x=text_x, y=header_y + 24, value=label, size=12, weight="700", anchor=anchor))

    row_y = header_y + header_h
    for row_index, row in enumerate(model_rows):
        base_fill = "#ffffff" if row_index % 2 == 0 else "#f8fafc"
        parts.append(f'<rect x="{x:.1f}" y="{row_y:.1f}" width="{table_w}" height="{row_h}" fill="{base_fill}"/>')

        for col_index, ((key, _, col_w), col_x) in enumerate(zip(TABLE_COLUMNS, col_xs)):
            parts.append(
                f'<rect x="{col_x:.1f}" y="{row_y:.1f}" width="{col_w}" height="{row_h}" '
                'fill="none" stroke="#e2e8f0" stroke-width="1"/>'
            )

            if key in {"synthetic_reviews", "synthetic_perfumes"}:
                anchor = "start" if col_index == 0 else "middle"
                text_x = col_x + 10 if col_index == 0 else col_x + col_w / 2
                parts.append(text(x=text_x, y=row_y + 31, value=row[key], size=13, color="#111827", anchor=anchor))
                continue

            is_best = row["synthetic_reviews"] == best_metric_counts[key]
            text_x = col_x + col_w / 2
            parts.append(
                text(
                    x=text_x,
                    y=row_y + 23,
                    value=fmt_mean(row, key),
                    size=13,
                    weight="700" if is_best else "400",
                    color="#111827",
                    anchor="middle",
                )
            )
            parts.append(text(x=text_x, y=row_y + 42, value=fmt_sd(row, key), size=10, color="#64748b", anchor="middle"))
        row_y += row_h

    parts.append(
        text(
            x=x + 14,
            y=y + panel_h - 16,
            value="Bold = best mean for this model.",
            size=11,
            color="#374151",
        )
    )
    return parts


def build_svg(rows: list[dict[str, str]]) -> str:
    margin_x = 34
    top = 32
    title_h = 82
    row_h = 54
    header_h = 38
    panel_gap = 28
    table_w = sum(width for _, _, width in TABLE_COLUMNS)
    panel_w = table_w
    width = margin_x * 2 + panel_w * 2 + panel_gap
    panel_h = 48 + header_h + 7 * row_h + 40
    height = top + title_h + panel_h + 44
    grouped_rows = rows_by_model(rows)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        text(x=margin_x, y=top + 20, value="Randomized Synthetic-Only Review Count Sweep Metrics", size=24, weight="700"),
        text(
            x=margin_x,
            y=top + 48,
            value="Top number is the mean across random perfume orders; the smaller gray value is SD. Folds are grouped by perfume_id.",
            size=13,
            color="#374151",
        ),
    ]

    panel_y = top + title_h
    parts.extend(
        render_model_panel(
            model="Logistic Regression",
            model_rows=grouped_rows["Logistic Regression"],
            x=margin_x,
            y=panel_y,
            width=panel_w,
            row_h=row_h,
            header_h=header_h,
        )
    )
    parts.extend(
        render_model_panel(
            model="XGBoost",
            model_rows=grouped_rows["XGBoost"],
            x=margin_x + panel_w + panel_gap,
            y=panel_y,
            width=panel_w,
            row_h=row_h,
            header_h=header_h,
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

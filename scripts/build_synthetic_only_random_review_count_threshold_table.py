#!/usr/bin/env python3
"""Build an SVG F1 table for randomized synthetic-only threshold sweeps."""

from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = (
    ROOT
    / "outputs/synthetic_only_random_review_count_sweep_gemini_browser_6reviews_thresholds"
    / "synthetic_only_random_review_count_sweep_summary.csv"
)
DEFAULT_OUT_DIR = ROOT / "outputs/synthetic_only_random_review_count_sweep_gemini_browser_6reviews_thresholds/figures"

MODEL_COLORS = {
    "Logistic Regression": "#2563eb",
    "XGBoost": "#dc2626",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build threshold sweep Macro F1 table as SVG.")
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def threshold_key(value: str) -> str:
    return f"{float(value):g}"


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
    return grouped


def render_panel(
    *,
    model: str,
    model_rows: list[dict[str, str]],
    thresholds: list[str],
    x: float,
    y: float,
    width: float,
    row_h: float,
    header_h: float,
) -> list[str]:
    counts = sorted({int(row["synthetic_reviews"]) for row in model_rows})
    row_lookup = {
        (int(row["synthetic_reviews"]), threshold_key(row["threshold"])): row
        for row in model_rows
    }
    review_w = 86
    perfume_w = 82
    metric_w = (width - review_w - perfume_w) / len(thresholds)
    panel_h = 48 + header_h + len(counts) * row_h + 40
    color = MODEL_COLORS.get(model, "#111827")

    best_by_count = {
        count: max(
            thresholds,
            key=lambda threshold: float(row_lookup[(count, threshold)]["macro_f1_mean"]),
        )
        for count in counts
    }

    parts = [
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{panel_h:.1f}" fill="#ffffff" stroke="#cbd5e1"/>',
        text(x=x + 14, y=y + 28, value=model, size=18, weight="700", color=color),
        text(x=x + width - 14, y=y + 27, value=f'{model_rows[0]["runs"]} random orders', size=12, color="#4b5563", anchor="end"),
    ]

    header_y = y + 48
    col_xs = [x, x + review_w]
    for _ in thresholds:
        col_xs.append(col_xs[-1] + perfume_w if len(col_xs) == 2 else col_xs[-1] + metric_w)
    widths = [review_w, perfume_w, *([metric_w] * len(thresholds))]
    labels = ["Reviews", "Perfumes", *[f"F1 @ {threshold}" for threshold in thresholds]]
    parts.append(f'<rect x="{x:.1f}" y="{header_y:.1f}" width="{width:.1f}" height="{header_h:.1f}" fill="#e8eef8"/>')
    for index, (col_x, col_w, label) in enumerate(zip(col_xs, widths, labels)):
        parts.append(
            f'<rect x="{col_x:.1f}" y="{header_y:.1f}" width="{col_w:.1f}" height="{header_h:.1f}" '
            'fill="none" stroke="#cbd5e1" stroke-width="1"/>'
        )
        anchor = "start" if index == 0 else "middle"
        text_x = col_x + 10 if index == 0 else col_x + col_w / 2
        parts.append(text(x=text_x, y=header_y + 24, value=label, size=12, weight="700", anchor=anchor))

    row_y = header_y + header_h
    for row_index, count in enumerate(counts):
        fill = "#ffffff" if row_index % 2 == 0 else "#f8fafc"
        parts.append(f'<rect x="{x:.1f}" y="{row_y:.1f}" width="{width:.1f}" height="{row_h:.1f}" fill="{fill}"/>')
        perfume_count = row_lookup[(count, thresholds[0])]["synthetic_perfumes"]
        cell_values = [str(count), perfume_count]
        for col_index, threshold in enumerate(thresholds):
            row = row_lookup[(count, threshold)]
            cell_values.append((f"{float(row['macro_f1_mean']):.4f}", f"SD {float(row['macro_f1_std']):.4f}"))

        for col_index, (col_x, col_w) in enumerate(zip(col_xs, widths)):
            parts.append(
                f'<rect x="{col_x:.1f}" y="{row_y:.1f}" width="{col_w:.1f}" height="{row_h:.1f}" '
                'fill="none" stroke="#e2e8f0" stroke-width="1"/>'
            )
            if col_index < 2:
                anchor = "start" if col_index == 0 else "middle"
                text_x = col_x + 10 if col_index == 0 else col_x + col_w / 2
                parts.append(text(x=text_x, y=row_y + 31, value=cell_values[col_index], size=13, anchor=anchor))
                continue

            threshold = thresholds[col_index - 2]
            mean, sd = cell_values[col_index]
            weight = "700" if threshold == best_by_count[count] else "400"
            text_x = col_x + col_w / 2
            parts.append(text(x=text_x, y=row_y + 23, value=mean, size=13, weight=weight, anchor="middle"))
            parts.append(text(x=text_x, y=row_y + 42, value=sd, size=10, color="#64748b", anchor="middle"))
        row_y += row_h

    parts.append(text(x=x + 14, y=y + panel_h - 16, value="Bold = best threshold for this row.", size=11, color="#374151"))
    return parts


def build_svg(rows: list[dict[str, str]]) -> str:
    thresholds = sorted({threshold_key(row["threshold"]) for row in rows}, key=float)
    grouped = rows_by_model(rows)
    margin_x = 34
    top = 32
    title_h = 82
    row_h = 54
    header_h = 38
    panel_gap = 28
    panel_w = 662
    width = margin_x * 2 + panel_w * 2 + panel_gap
    max_rows = max(len({int(row["synthetic_reviews"]) for row in model_rows}) for model_rows in grouped.values())
    panel_h = 48 + header_h + max_rows * row_h + 40
    height = top + title_h + panel_h + 44

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        text(x=margin_x, y=top + 20, value="Synthetic-Only Threshold Sensitivity: Macro F1", size=24, weight="700"),
        text(
            x=margin_x,
            y=top + 48,
            value="Top number is mean Macro F1 across random perfume orders; smaller gray value is SD.",
            size=13,
            color="#374151",
        ),
    ]

    panel_y = top + title_h
    parts.extend(
        render_panel(
            model="Logistic Regression",
            model_rows=grouped["Logistic Regression"],
            thresholds=thresholds,
            x=margin_x,
            y=panel_y,
            width=panel_w,
            row_h=row_h,
            header_h=header_h,
        )
    )
    parts.extend(
        render_panel(
            model="XGBoost",
            model_rows=grouped["XGBoost"],
            thresholds=thresholds,
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
    out_path = args.out_dir / "synthetic_only_random_review_count_threshold_sweep_table.svg"
    out_path.write_text(build_svg(rows), encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()

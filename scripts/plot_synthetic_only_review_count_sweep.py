#!/usr/bin/env python3
"""Build an SVG plot for synthetic-only review-count sweep metrics."""

from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = ROOT / "outputs/synthetic_only_review_count_sweep_gemini_browser/synthetic_only_review_count_sweep_summary.csv"
DEFAULT_OUT_DIR = ROOT / "outputs/synthetic_only_review_count_sweep_gemini_browser/figures"

METRICS = [
    ("macro_f1", "Macro F1"),
    ("macro_roc_auc", "Macro ROC-AUC"),
    ("macro_pr_auc", "Macro PR-AUC"),
]

MODEL_STYLE = {
    "Logistic Regression": {"color": "#2563eb", "marker": "circle"},
    "XGBoost": {"color": "#dc2626", "marker": "square"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot synthetic-only review-count sweep metrics as SVG.")
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def text(
    *,
    x: float,
    y: float,
    value: str,
    size: int = 12,
    weight: str = "400",
    color: str = "#111827",
    anchor: str = "start",
    extra: str = "",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{color}" text-anchor="{anchor}" {extra}>'
        f"{html.escape(value)}</text>"
    )


def line(x1: float, y1: float, x2: float, y2: float, color: str = "#cbd5e1", width: float = 1) -> str:
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{width}"/>'


def polyline(points: Iterable[tuple[float, float]], color: str) -> str:
    point_text = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f'<polyline points="{point_text}" fill="none" stroke="{color}" stroke-width="2.4" stroke-linejoin="round" stroke-linecap="round"/>'


def marker(x: float, y: float, color: str, marker_name: str, hollow: bool = False) -> str:
    fill = "#ffffff" if hollow else color
    stroke_width = 2 if hollow else 1
    if marker_name == "square":
        return (
            f'<rect x="{x - 4:.1f}" y="{y - 4:.1f}" width="8" height="8" '
            f'fill="{fill}" stroke="{color}" stroke-width="{stroke_width}"/>'
        )
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{fill}" stroke="{color}" stroke-width="{stroke_width}"/>'


def grouped(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        out.setdefault(row["model_family"], []).append(row)
    for model_rows in out.values():
        model_rows.sort(key=lambda item: int(item["synthetic_reviews"]))
    return out


def render_panel(
    *,
    rows_by_model: dict[str, list[dict[str, str]]],
    metric: str,
    title: str,
    x: float,
    y: float,
    width: float,
    height: float,
) -> list[str]:
    all_rows = [row for model_rows in rows_by_model.values() for row in model_rows]
    counts = [int(row["synthetic_reviews"]) for row in all_rows]
    values = [float(row[metric]) for row in all_rows]
    x_min, x_max = min(counts), max(counts)
    y_min, y_max = min(values), max(values)
    y_pad = max((y_max - y_min) * 0.18, 0.01)
    y_min = max(0.0, y_min - y_pad)
    y_max = min(1.0, y_max + y_pad)

    def sx(count: int) -> float:
        return x + (count - x_min) / (x_max - x_min) * width if x_max > x_min else x + width / 2

    def sy(value: float) -> float:
        return y + height - (value - y_min) / (y_max - y_min) * height if y_max > y_min else y + height / 2

    parts = [
        text(x=x, y=y - 18, value=title, size=16, weight="700"),
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" fill="#ffffff" stroke="#cbd5e1"/>',
    ]

    for idx in range(5):
        ratio = idx / 4
        grid_y = y + height - ratio * height
        grid_value = y_min + ratio * (y_max - y_min)
        parts.append(line(x, grid_y, x + width, grid_y, color="#e5e7eb"))
        parts.append(text(x=x - 8, y=grid_y + 4, value=f"{grid_value:.3f}", size=10, color="#4b5563", anchor="end"))

    x_ticks = [99, 201, 300, 501, 1002, 2001, 2352]
    for tick in x_ticks:
        tick_x = sx(tick)
        parts.append(line(tick_x, y + height, tick_x, y + height + 5, color="#9ca3af"))
        parts.append(
            text(
                x=tick_x - 2,
                y=y + height + 22,
                value=str(tick),
                size=9,
                color="#4b5563",
                anchor="end",
                extra='transform="rotate(-35 {:.1f} {:.1f})"'.format(tick_x - 2, y + height + 22),
            )
        )

    for model_family, model_rows in rows_by_model.items():
        style = MODEL_STYLE.get(model_family, {"color": "#111827", "marker": "circle"})
        points = [(sx(int(row["synthetic_reviews"])), sy(float(row[metric]))) for row in model_rows]
        parts.append(polyline(points, style["color"]))
        best_row = max(model_rows, key=lambda row: float(row[metric]))
        for row, (point_x, point_y) in zip(model_rows, points):
            parts.append(marker(point_x, point_y, style["color"], style["marker"], hollow=row is best_row))
        best_x = sx(int(best_row["synthetic_reviews"]))
        best_y = sy(float(best_row[metric]))
        label_y = best_y - 10 if best_y > y + 24 else best_y + 20
        parts.append(
            text(
                x=best_x,
                y=label_y,
                value=f"{int(best_row['synthetic_reviews'])}: {float(best_row[metric]):.4f}",
                size=10,
                color=style["color"],
                anchor="middle",
            )
        )

    parts.append(text(x=x + width / 2, y=y + height + 58, value="Synthetic reviews", size=11, anchor="middle", color="#374151"))
    return parts


def build_svg(rows: list[dict[str, str]]) -> str:
    rows_by_model = grouped(rows)
    width = 1380
    height = 640
    panel_w = 390
    panel_h = 350
    panel_y = 150
    panel_xs = [80, 515, 950]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        text(x=width / 2, y=44, value="Synthetic-Only Review Count Sweep", size=24, weight="700", anchor="middle"),
        text(
            x=width / 2,
            y=70,
            value="Grouped cross-validation by perfume_id; hollow markers show the best value for each model within each metric.",
            size=13,
            color="#374151",
            anchor="middle",
        ),
    ]

    legend_x = width / 2 - 170
    for index, model_family in enumerate(["Logistic Regression", "XGBoost"]):
        style = MODEL_STYLE[model_family]
        x = legend_x + index * 220
        parts.append(marker(x, 105, style["color"], style["marker"]))
        parts.append(text(x=x + 14, y=109, value=model_family, size=13, color="#111827"))

    for panel_x, (metric, title) in zip(panel_xs, METRICS):
        parts.extend(render_panel(rows_by_model=rows_by_model, metric=metric, title=title, x=panel_x, y=panel_y, width=panel_w, height=panel_h))

    parts.append("</svg>\n")
    return "\n".join(parts)


def main() -> None:
    args = parse_args()
    rows = load_rows(args.summary_csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "synthetic_only_review_count_sweep_metrics.svg"
    out_path.write_text(build_svg(rows), encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()

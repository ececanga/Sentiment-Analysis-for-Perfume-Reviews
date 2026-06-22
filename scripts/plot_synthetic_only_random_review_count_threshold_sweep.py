#!/usr/bin/env python3
"""Build an SVG F1 plot for randomized synthetic-only threshold sweeps."""

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

THRESHOLD_COLORS = {
    "0.25": "#2563eb",
    "0.3": "#059669",
    "0.5": "#dc2626",
    "0.7": "#7c3aed",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot threshold sweep Macro F1 as SVG.")
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


def polyline(points: list[tuple[float, float]], color: str) -> str:
    point_text = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return (
        f'<polyline points="{point_text}" fill="none" stroke="{color}" '
        'stroke-width="2.4" stroke-linejoin="round" stroke-linecap="round"/>'
    )


def marker(x: float, y: float, color: str, hollow: bool = False) -> str:
    fill = "#ffffff" if hollow else color
    stroke_width = 2 if hollow else 1
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.2" fill="{fill}" stroke="{color}" stroke-width="{stroke_width}"/>'


def render_error_bar(x: float, y_mean: float, y_low: float, y_high: float, color: str) -> list[str]:
    if abs(y_high - y_low) < 0.5:
        return []
    return [
        line(x, y_low, x, y_high, color=color, width=1.0),
        line(x - 3.5, y_low, x + 3.5, y_low, color=color, width=1.0),
        line(x - 3.5, y_high, x + 3.5, y_high, color=color, width=1.0),
    ]


def render_panel(
    *,
    rows: list[dict[str, str]],
    model_family: str,
    x: float,
    y: float,
    width: float,
    height: float,
) -> list[str]:
    model_rows = [row for row in rows if row["model_family"] == model_family]
    counts = sorted({int(row["synthetic_reviews"]) for row in model_rows})
    thresholds = sorted({threshold_key(row["threshold"]) for row in model_rows}, key=float)
    intervals = []
    for row in model_rows:
        mean_value = float(row["macro_f1_mean"])
        std_value = float(row["macro_f1_std"])
        intervals.extend([mean_value - std_value, mean_value + std_value])
    x_min, x_max = min(counts), max(counts)
    y_min, y_max = min(intervals), max(intervals)
    y_pad = max((y_max - y_min) * 0.18, 0.01)
    y_min = max(0.0, y_min - y_pad)
    y_max = min(1.0, y_max + y_pad)

    def sx(count: int) -> float:
        return x + (count - x_min) / (x_max - x_min) * width if x_max > x_min else x + width / 2

    def sy(value: float) -> float:
        clipped = min(max(value, y_min), y_max)
        return y + height - (clipped - y_min) / (y_max - y_min) * height if y_max > y_min else y + height / 2

    parts = [
        text(x=x, y=y - 18, value=model_family, size=18, weight="700"),
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" fill="#ffffff" stroke="#cbd5e1"/>',
    ]
    for idx in range(5):
        ratio = idx / 4
        grid_y = y + height - ratio * height
        grid_value = y_min + ratio * (y_max - y_min)
        parts.append(line(x, grid_y, x + width, grid_y, color="#e5e7eb"))
        parts.append(text(x=x - 8, y=grid_y + 4, value=f"{grid_value:.3f}", size=10, color="#4b5563", anchor="end"))

    for count in counts:
        tick_x = sx(count)
        parts.append(line(tick_x, y + height, tick_x, y + height + 5, color="#9ca3af"))
        parts.append(
            text(
                x=tick_x - 2,
                y=y + height + 22,
                value=str(count),
                size=9,
                color="#4b5563",
                anchor="end",
                extra='transform="rotate(-35 {:.1f} {:.1f})"'.format(tick_x - 2, y + height + 22),
            )
        )

    for threshold in thresholds:
        threshold_rows = sorted(
            [row for row in model_rows if threshold_key(row["threshold"]) == threshold],
            key=lambda row: int(row["synthetic_reviews"]),
        )
        color = THRESHOLD_COLORS.get(threshold, "#111827")
        points = [
            (sx(int(row["synthetic_reviews"])), sy(float(row["macro_f1_mean"])))
            for row in threshold_rows
        ]
        parts.append(polyline(points, color))
        best_row = max(threshold_rows, key=lambda row: float(row["macro_f1_mean"]))
        for row, (point_x, point_y) in zip(threshold_rows, points):
            mean_value = float(row["macro_f1_mean"])
            std_value = float(row["macro_f1_std"])
            parts.extend(render_error_bar(point_x, point_y, sy(mean_value - std_value), sy(mean_value + std_value), color))
            parts.append(marker(point_x, point_y, color, hollow=row is best_row))

    parts.append(text(x=x + width / 2, y=y + height + 58, value="Synthetic reviews", size=11, anchor="middle", color="#374151"))
    return parts


def build_svg(rows: list[dict[str, str]]) -> str:
    width = 1360
    height = 650
    panel_y = 165
    panel_w = 540
    panel_h = 350
    thresholds = sorted({threshold_key(row["threshold"]) for row in rows}, key=float)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        text(x=width / 2, y=44, value="Synthetic-Only Threshold Sensitivity", size=24, weight="700", anchor="middle"),
        text(
            x=width / 2,
            y=72,
            value="Macro F1 by review count and threshold; points show mean across 5 random perfume orders, whiskers show one SD.",
            size=13,
            color="#374151",
            anchor="middle",
        ),
    ]
    legend_x = width / 2 - 250
    for index, threshold in enumerate(thresholds):
        color = THRESHOLD_COLORS.get(threshold, "#111827")
        x = legend_x + index * 140
        parts.append(marker(x, 112, color))
        parts.append(text(x=x + 14, y=116, value=f"threshold {threshold}", size=12, color="#111827"))

    parts.extend(render_panel(rows=rows, model_family="Logistic Regression", x=90, y=panel_y, width=panel_w, height=panel_h))
    parts.extend(render_panel(rows=rows, model_family="XGBoost", x=730, y=panel_y, width=panel_w, height=panel_h))
    parts.append("</svg>\n")
    return "\n".join(parts)


def main() -> None:
    args = parse_args()
    rows = load_rows(args.summary_csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "synthetic_only_random_review_count_threshold_sweep_f1.svg"
    out_path.write_text(build_svg(rows), encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()

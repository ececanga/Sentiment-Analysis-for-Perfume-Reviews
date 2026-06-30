#!/usr/bin/env python3
"""Build a compact SVG report for advisor experiment combinations."""

from __future__ import annotations

import html
import math
import textwrap
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/synthetic_review_experiment_report/table_images/07_threshold03_combination_report.svg"

SYNTH_3_COUNT = (
    ROOT
    / "outputs/synthetic_only_random_review_count_sweep_gemini_browser_unweighted_threshold03"
    / "synthetic_only_random_review_count_sweep_summary.csv"
)
SYNTH_6_COUNT = (
    ROOT
    / "outputs/synthetic_only_random_review_count_sweep_gemini_browser_6reviews_unweighted_threshold03"
    / "synthetic_only_random_review_count_sweep_summary.csv"
)
SYNTH_3_BAL = (
    ROOT
    / "outputs/synthetic_only_resampling_ablation_gemini_browser_3reviews_threshold03"
    / "synthetic_only_resampling_ablation_summary.csv"
)
SYNTH_6_BAL = (
    ROOT
    / "outputs/synthetic_only_resampling_ablation_gemini_browser_6reviews_threshold03"
    / "synthetic_only_resampling_ablation_summary.csv"
)
OG_BAL = (
    ROOT
    / "outputs/og_synthetic_balancing_ablation_gemini_browser_thresholds"
    / "og_synthetic_balancing_ablation_summary.csv"
)
OG_WINNERS = (
    ROOT
    / "outputs/og_synthetic_balancing_ablation_gemini_browser_thresholds"
    / "og_synthetic_balancing_ablation_balanced_winners.csv"
)
OG_PER_LABEL = (
    ROOT
    / "outputs/og_synthetic_balancing_ablation_gemini_browser_thresholds"
    / "og_synthetic_balancing_ablation_per_label.csv"
)
OG_FEATURES = (
    ROOT
    / "outputs/gemini_browser_og784/analysis_out/gemini_browser_og784_feature_labels.csv"
)
OG_FOLD_ASSIGNMENTS = (
    ROOT
    / "outputs/gemini_browser_og784/model_benchmark/tables/fold_assignments.csv"
)
SYNTHETIC_3_FEATURES = (
    ROOT
    / "outputs/synthetic_review_level_tests/emotion_rich_v2_3reviews/supervised_eval_gemini_browser"
    / "review_level_features_labeled_g25pro.csv"
)
SYNTHETIC_6_FEATURES = (
    ROOT
    / "outputs/synthetic_review_level_tests/emotion_rich_v2_6reviews/supervised_eval_gemini_browser"
    / "review_level_features_labeled_g25pro.csv"
)
LABEL_DIST_BAL_SUMMARY = (
    ROOT
    / "outputs/og_synthetic_label_distribution_balancing_threshold03_v3"
    / "label_distribution_balancing_summary.csv"
)
LABEL_DIST_BAL_FOLDS = (
    ROOT
    / "outputs/og_synthetic_label_distribution_balancing_threshold03_v3"
    / "label_distribution_balancing_fold_balance.csv"
)

PAGE_W = 1000
MARGIN = 38
CONTENT_W = PAGE_W - 2 * MARGIN
NAVY = "#172554"
BLUE = "#1d4ed8"
TEAL = "#0f766e"
AMBER = "#d97706"
RED = "#dc2626"
PURPLE = "#7c3aed"
TEXT = "#111827"
MUTED = "#475569"
BORDER = "#cbd5e1"
SOFT_BG = "#f8fafc"
BOLD_PREFIX = "__BOLD__"
INTRO_GAP = 28
PAIR_GAP = 48
SECTION_GAP = 56

BALANCING_METHOD_NOTES = {
    "No balancing method": "No rows changed.",
    "oversampling": "Duplicates minority class per emotion.",
    "undersampling": "Drops majority class per emotion.",
    "SMOTE": "Creates minority samples per emotion.",
}

LABEL_METHOD_NOTES = {
    "No balancing method": "Emotion counts unchanged.",
    "oversampling": "Raises low emotion categories.",
    "undersampling": "Lowers high emotion categories.",
    "hybrid": "Lowers high, raises low categories.",
    "SMOTE": "Binary SMOTE reference.",
}

METHOD_DISPLAY = {
    "hybrid": "hybrid (undersampling + oversampling)",
}


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def fmt(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except Exception:
        return str(value)
    if math.isnan(number):
        return "NA"
    return f"{number:.{digits}f}"


def signed(value: Any) -> str:
    number = float(value)
    return f"{number:+.4f}"


def signed_int(value: Any) -> str:
    number = int(value)
    return f"{number:+d}"


def mean_sd(mean: Any, sd: Any) -> str:
    return f"Mean {fmt(mean)}\nSD {fmt(sd)}"


def compact_prf(precision: float, recall: float, f1: float) -> str:
    return f"P {fmt(precision, 2)}\nR {fmt(recall, 2)}\nF1 {fmt(f1, 2)}"


def bold_cell(value: Any) -> str:
    return f"{BOLD_PREFIX}{value}"


def strip_bold(value: Any) -> str:
    text = str(value)
    while text.startswith(BOLD_PREFIX):
        text = text[len(BOLD_PREFIX) :]
    return text


def parse_score(value: Any) -> float:
    text = strip_bold(value).strip()
    if text.startswith("Mean "):
        text = text.splitlines()[0].replace("Mean ", "", 1)
    try:
        return float(text)
    except ValueError:
        return float("-inf")


def wrap_text(value: str, width: int) -> list[str]:
    lines: list[str] = []
    for part in str(value).splitlines():
        lines.extend(textwrap.wrap(str(part), width=width, break_long_words=False, break_on_hyphens=False) or [""])
    return lines


def text_element(
    *,
    x: int | float,
    y: int | float,
    text: str,
    size: int = 13,
    weight: str = "400",
    color: str = TEXT,
    anchor: str = "start",
) -> str:
    return (
        f'<text x="{x}" y="{y}" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{color}" text-anchor="{anchor}">'
        f"{html.escape(str(text))}</text>"
    )


def metric_style(line: str, default_color: str = TEXT) -> tuple[int, str, str]:
    if line.startswith("Mean "):
        return 15, "400", TEXT
    if line.startswith("SD "):
        return 12, "400", "#64748b"
    if line.startswith("+") and len(line) > 1 and (line[1].isdigit() or line[1] == "."):
        return 14, "400", "#047857"
    if line.startswith("-") and len(line) > 1 and (line[1].isdigit() or line[1] == "."):
        return 14, "400", "#b91c1c"
    return 14, "400", default_color


def row_fill(row: list[str], row_idx: int) -> str:
    joined = " ".join(strip_bold(value) for value in row)
    if row_idx == 0:
        return NAVY
    if "OG + synthetic 784x6" in joined:
        return "#f0fdf4" if row_idx % 2 else "#dcfce7"
    if "OG + synthetic 784x3" in joined:
        return "#eff6ff" if row_idx % 2 else "#dbeafe"
    if "OG YouTube only" in joined or "OG only" in joined:
        return "#fff7ed" if row_idx % 2 else "#ffedd5"
    if "784x6" in joined:
        return "#f0fdf4" if row_idx % 2 else "#dcfce7"
    if "784x3" in joined:
        return "#eff6ff" if row_idx % 2 else "#dbeafe"
    return "#ffffff" if row_idx % 2 else SOFT_BG


def render_table(
    *,
    title: str,
    headers: list[str],
    rows: list[list[str]],
    widths: list[int],
    y_start: int,
    accent: str,
    center_cols: set[int] | None = None,
    separator_after_cols: set[int] | None = None,
    merge_cols: set[int] | None = None,
    x_start: int = MARGIN,
    row_pad_y: int = 8,
    line_h: int = 18,
    header_min_h: int = 40,
    body_min_h: int = 42,
    title_gap: int = 26,
) -> tuple[list[str], int]:
    center_cols = center_cols or set()
    separator_after_cols = separator_after_cols or set()
    merge_cols = merge_cols or set()
    pad_x = 10
    table_w = sum(widths)
    x_positions = [x_start]
    for width in widths[:-1]:
        x_positions.append(x_positions[-1] + width)

    raw_rows = [headers] + rows
    wrapped_rows: list[list[list[str]]] = []
    bold_cells: list[list[bool]] = []
    for row in raw_rows:
        wrapped_row = []
        bold_row = []
        for idx, cell in enumerate(row):
            cell_text = str(cell)
            is_bold = cell_text.startswith(BOLD_PREFIX)
            cell_text = strip_bold(cell_text)
            chars = max(8, (widths[idx] - 2 * pad_x) // 7)
            wrapped_row.append(wrap_text(cell_text, chars))
            bold_row.append(is_bold)
        wrapped_rows.append(wrapped_row)
        bold_cells.append(bold_row)

    row_heights = []
    for idx, row in enumerate(wrapped_rows):
        visible_cells = [
            lines
            for col_idx, lines in enumerate(row)
            if not (idx > 0 and col_idx in merge_cols)
        ]
        row_heights.append(
            max(
                header_min_h if idx == 0 else body_min_h,
                max(len(lines) for lines in visible_cells) * line_h + 2 * row_pad_y,
            )
        )
    table_y = y_start + title_gap
    total_h = sum(row_heights)
    row_tops: list[int | float] = []
    cursor: int | float = table_y
    for row_h in row_heights:
        row_tops.append(cursor)
        cursor += row_h
    merge_spans: list[tuple[int, int, int]] = []
    for col_idx in merge_cols:
        start = 1
        while start < len(raw_rows):
            value = strip_bold(raw_rows[start][col_idx])
            end = start + 1
            while end < len(raw_rows) and strip_bold(raw_rows[end][col_idx]) == value:
                end += 1
            merge_spans.append((col_idx, start, end))
            start = end
    parts = [
        f'<line x1="{x_start}" y1="{y_start - 12}" x2="{x_start}" y2="{y_start + 6}" '
        f'stroke="{accent}" stroke-width="5" stroke-linecap="round"/>',
        text_element(x=x_start + 16, y=y_start + 3, text=title, size=16, weight="700"),
        f'<rect x="{x_start}" y="{table_y}" width="{table_w}" height="{total_h}" rx="8" '
        f'fill="#ffffff" stroke="{BORDER}" stroke-width="1.2"/>',
    ]

    y = table_y
    for row_idx, row in enumerate(wrapped_rows):
        raw = raw_rows[row_idx]
        row_h = row_heights[row_idx]
        fill = row_fill(raw, row_idx)
        parts.append(f'<rect x="{x_start}" y="{y}" width="{table_w}" height="{row_h}" fill="{fill}"/>')
        for col_idx, lines in enumerate(row):
            if row_idx > 0 and col_idx in merge_cols:
                continue
            x = x_positions[col_idx]
            width = widths[col_idx]
            is_header = row_idx == 0
            is_best_cell = row_idx > 0 and bold_cells[row_idx][col_idx]
            cell_fill = "#bbf7d0" if is_best_cell else "none"
            parts.append(
                f'<rect x="{x}" y="{y}" width="{width}" height="{row_h}" '
                f'fill="{cell_fill}" stroke="{BORDER}" stroke-width="0.75"/>'
            )
            anchor = "middle" if is_header or col_idx in center_cols else "start"
            text_x = x + width // 2 if anchor == "middle" else x + pad_x
            block_h = len(lines) * line_h
            text_y = y + (row_h - block_h) / 2 + 12
            for line in lines:
                size, weight, color = metric_style(line)
                if is_header:
                    size, weight, color = 13, "700", "#ffffff"
                elif bold_cells[row_idx][col_idx]:
                    if line.startswith("SD "):
                        size, weight, color = 12, "400", "#64748b"
                    else:
                        weight = "800"
                        color = "#065f46"
                parts.append(
                    text_element(
                        x=text_x,
                        y=text_y,
                        text=line,
                        size=size,
                        weight=weight,
                        color=color,
                        anchor=anchor,
                    )
                )
                text_y += line_h
        y += row_h
    for col_idx, start, end in merge_spans:
        x = x_positions[col_idx]
        width = widths[col_idx]
        span_y = row_tops[start]
        span_h = sum(row_heights[start:end])
        raw = raw_rows[start]
        fill = row_fill(raw, start)
        parts.append(
            f'<rect x="{x}" y="{span_y}" width="{width}" height="{span_h}" '
            f'fill="{fill}" stroke="{BORDER}" stroke-width="0.75"/>'
        )
        lines = wrapped_rows[start][col_idx]
        anchor = "middle" if col_idx in center_cols else "start"
        text_x = x + width // 2 if anchor == "middle" else x + pad_x
        block_h = len(lines) * line_h
        text_y = span_y + (span_h - block_h) / 2 + 12
        for line in lines:
            size, weight, color = metric_style(line)
            parts.append(
                text_element(
                    x=text_x,
                    y=text_y,
                    text=line,
                    size=size,
                    weight=weight,
                    color=color,
                    anchor=anchor,
                )
            )
            text_y += line_h
    for col_idx in separator_after_cols:
        if 0 <= col_idx < len(widths):
            x = x_start + sum(widths[: col_idx + 1])
            parts.append(
                f'<line x1="{x}" y1="{table_y}" x2="{x}" y2="{table_y + total_h}" '
                f'stroke="#334155" stroke-width="3" stroke-linecap="square"/>'
            )
    return parts, y


def render_section20_table(
    *,
    title: str,
    rows: list[list[str]],
    widths: list[int],
    y_start: int,
    accent: str,
    x_start: int = MARGIN,
) -> tuple[list[str], int]:
    """Render Section 2.0 with model-family header groups."""
    pad_x = 10
    table_w = sum(widths)
    x_positions = [x_start]
    for width in widths[:-1]:
        x_positions.append(x_positions[-1] + width)

    group_h = 34
    sub_h = 34
    header_h = group_h + sub_h
    row_pad_y = 8
    line_h = 18
    body_min_h = 42
    title_gap = 26

    wrapped_rows: list[list[list[str]]] = []
    bold_cells: list[list[bool]] = []
    for row in rows:
        wrapped_row = []
        bold_row = []
        for idx, cell in enumerate(row):
            cell_text = str(cell)
            is_bold = cell_text.startswith(BOLD_PREFIX)
            cell_text = strip_bold(cell_text)
            chars = max(8, (widths[idx] - 2 * pad_x) // 7)
            wrapped_row.append(wrap_text(cell_text, chars))
            bold_row.append(is_bold)
        wrapped_rows.append(wrapped_row)
        bold_cells.append(bold_row)

    body_heights = [
        max(body_min_h, max(len(lines) for lines in row) * line_h + 2 * row_pad_y)
        for row in wrapped_rows
    ]
    table_y = y_start + title_gap
    total_h = header_h + sum(body_heights)
    parts = [
        f'<line x1="{x_start}" y1="{y_start - 12}" x2="{x_start}" y2="{y_start + 6}" '
        f'stroke="{accent}" stroke-width="5" stroke-linecap="round"/>',
        text_element(x=x_start + 16, y=y_start + 3, text=title, size=16, weight="700"),
        f'<rect x="{x_start}" y="{table_y}" width="{table_w}" height="{total_h}" rx="8" '
        f'fill="#ffffff" stroke="{BORDER}" stroke-width="1.2"/>',
    ]

    # Left header spans both header rows.
    parts.append(
        f'<rect x="{x_positions[0]}" y="{table_y}" width="{widths[0]}" height="{header_h}" '
        f'fill="{NAVY}" stroke="{BORDER}" stroke-width="0.75"/>'
    )
    parts.append(
        text_element(
            x=x_positions[0] + widths[0] // 2,
            y=table_y + header_h / 2 + 5,
            text="Training data",
            size=13,
            weight="700",
            color="#ffffff",
            anchor="middle",
        )
    )

    group_specs = [
        ("Logistic Regression", 1, 3),
        ("XGBoost", 4, 6),
    ]
    for label, start_col, end_col in group_specs:
        x = x_positions[start_col]
        width = sum(widths[start_col : end_col + 1])
        parts.append(
            f'<rect x="{x}" y="{table_y}" width="{width}" height="{group_h}" '
            f'fill="{NAVY}" stroke="{BORDER}" stroke-width="0.75"/>'
        )
        parts.append(
            text_element(
                x=x + width / 2,
                y=table_y + 22,
                text=label,
                size=13,
                weight="700",
                color="#ffffff",
                anchor="middle",
            )
        )

    subheaders = ["No balancing", "Hybrid", "SMOTE", "No balancing", "Hybrid", "SMOTE"]
    for idx, label in enumerate(subheaders, start=1):
        x = x_positions[idx]
        width = widths[idx]
        parts.append(
            f'<rect x="{x}" y="{table_y + group_h}" width="{width}" height="{sub_h}" '
            f'fill="{NAVY}" stroke="{BORDER}" stroke-width="0.75"/>'
        )
        parts.append(
            text_element(
                x=x + width / 2,
                y=table_y + group_h + 22,
                text=label,
                size=13,
                weight="700",
                color="#ffffff",
                anchor="middle",
            )
        )

    y = table_y + header_h
    for row_idx, row in enumerate(wrapped_rows):
        raw = rows[row_idx]
        row_h = body_heights[row_idx]
        fill = row_fill(raw, row_idx + 1)
        parts.append(f'<rect x="{x_start}" y="{y}" width="{table_w}" height="{row_h}" fill="{fill}"/>')
        for col_idx, lines in enumerate(row):
            x = x_positions[col_idx]
            width = widths[col_idx]
            is_best_cell = bold_cells[row_idx][col_idx]
            cell_fill = "#bbf7d0" if is_best_cell else "none"
            parts.append(
                f'<rect x="{x}" y="{y}" width="{width}" height="{row_h}" '
                f'fill="{cell_fill}" stroke="{BORDER}" stroke-width="0.75"/>'
            )
            anchor = "middle" if col_idx != 0 else "start"
            text_x = x + width // 2 if anchor == "middle" else x + pad_x
            block_h = len(lines) * line_h
            text_y = y + (row_h - block_h) / 2 + 12
            for line in lines:
                size, weight, color = metric_style(line)
                if is_best_cell:
                    weight = "800"
                    color = "#065f46"
                parts.append(
                    text_element(
                        x=text_x,
                        y=text_y,
                        text=line,
                        size=size,
                        weight=weight,
                        color=color,
                        anchor=anchor,
                    )
                )
                text_y += line_h
        y += row_h

    separator_x = x_start + sum(widths[:4])
    parts.append(
        f'<line x1="{separator_x}" y1="{table_y}" x2="{separator_x}" y2="{table_y + total_h}" '
        f'stroke="#334155" stroke-width="3" stroke-linecap="square"/>'
    )
    return parts, y


def render_kpis(cards: list[tuple[str, str, str, str]], y_start: int) -> tuple[list[str], int]:
    gap = 18
    card_w = (CONTENT_W - gap * (len(cards) - 1)) // len(cards)
    card_h = 88
    x = MARGIN
    parts: list[str] = []
    for label, value, detail, color in cards:
        parts.append(
            f'<rect x="{x}" y="{y_start}" width="{card_w}" height="{card_h}" rx="10" '
            f'fill="#ffffff" stroke="{BORDER}" stroke-width="1.2"/>'
        )
        parts.append(f'<rect x="{x}" y="{y_start}" width="8" height="{card_h}" rx="4" fill="{color}"/>')
        parts.append(text_element(x=x + 24, y=y_start + 26, text=label.upper(), size=11, weight="700", color=color))
        parts.append(text_element(x=x + 24, y=y_start + 55, text=value, size=23, weight="700"))
        parts.append(text_element(x=x + 24, y=y_start + 76, text=detail, size=12, color=MUTED))
        x += card_w + gap
    return parts, y_start + card_h


def render_method_boxes(boxes: list[tuple[str, list[str], str]], y_start: int) -> tuple[list[str], int]:
    gap = 18
    box_w = (CONTENT_W - gap * (len(boxes) - 1)) // len(boxes)
    box_h = 142
    parts: list[str] = []
    x = MARGIN
    for title, lines, color in boxes:
        parts.append(
            f'<rect x="{x}" y="{y_start}" width="{box_w}" height="{box_h}" rx="10" '
            f'fill="#ffffff" stroke="{BORDER}" stroke-width="1.2"/>'
        )
        parts.append(f'<rect x="{x}" y="{y_start}" width="8" height="{box_h}" rx="4" fill="{color}"/>')
        parts.append(text_element(x=x + 24, y=y_start + 28, text=title, size=14, weight="700", color=color))
        y = y_start + 55
        for line in lines:
            wrapped = wrap_text(line, max(28, (box_w - 48) // 7))
            for wrapped_line in wrapped:
                parts.append(text_element(x=x + 24, y=y, text=wrapped_line, size=12, color=MUTED))
                y += 17
        x += box_w + gap
    return parts, y_start + box_h


def render_note_box(title: str, lines: list[str], y_start: int, color: str) -> tuple[list[str], int]:
    box_h = 82
    parts = [
        f'<rect x="{MARGIN}" y="{y_start}" width="{CONTENT_W}" height="{box_h}" rx="10" '
        f'fill="#ffffff" stroke="{BORDER}" stroke-width="1.2"/>',
        f'<rect x="{MARGIN}" y="{y_start}" width="8" height="{box_h}" rx="4" fill="{color}"/>',
        text_element(x=MARGIN + 24, y=y_start + 29, text=title, size=14, weight="700", color=color),
    ]
    y = y_start + 56
    for line in lines:
        for wrapped_line in wrap_text(line, 190):
            parts.append(text_element(x=MARGIN + 24, y=y, text=wrapped_line, size=12, color=MUTED))
            y += 17
    return parts, y_start + box_h


def render_method_note(text: str, y_start: int, color: str) -> tuple[list[str], int]:
    parts = [
        text_element(
            x=MARGIN,
            y=y_start + 17,
            text=text,
            size=12,
            color=color,
        )
    ]
    return parts, y_start + 24


def render_section_heading(title: str, subtitle: str, y_start: int, color: str) -> tuple[list[str], int]:
    parts = [
        text_element(
            x=MARGIN,
            y=y_start + 16,
            text=title,
            size=18,
            weight="700",
            color=color,
        ),
        text_element(
            x=MARGIN,
            y=y_start + 36,
            text=subtitle,
            size=12,
            color=MUTED,
        ),
    ]
    return parts, y_start + 60


def clean_strategy(value: str) -> str:
    mapping = {
        "unbalanced": "No balancing method",
        "class_weight": "class weighting",
        "scale_pos_weight": "class weighting",
        "oversampling": "oversampling",
        "undersampling": "undersampling",
        "hybrid": "hybrid",
        "smote": "SMOTE",
    }
    return mapping.get(str(value), str(value))


def display_method(value: str) -> str:
    return METHOD_DISPLAY.get(str(value), str(value))


def model_short(value: str) -> str:
    return "LR" if value == "Logistic Regression" else "XGBoost"


def display_data_label(value: str) -> str:
    return (
        str(value)
        .replace("OG YouTube + synthetic ", "YouTube + synthetic ")
        .replace("OG YouTube only", "YouTube only")
        .replace("OG + ", "YouTube + ")
        .replace("OG only", "YouTube only")
    )


def mark_best_scores(
    rows: list[list[str]],
    *,
    group_cols: list[int],
    metric_cols: list[int],
) -> list[list[str]]:
    marked = [row.copy() for row in rows]
    keys = sorted({tuple(row[col] for col in group_cols) if group_cols else ("all",) for row in marked})
    for key in keys:
        group_indices = [
            idx
            for idx, row in enumerate(marked)
            if (tuple(row[col] for col in group_cols) if group_cols else ("all",)) == key
        ]
        for metric_col in metric_cols:
            values = [(idx, parse_score(marked[idx][metric_col])) for idx in group_indices]
            best = max(value for _, value in values)
            for idx, value in values:
                if value == best:
                    marked[idx][metric_col] = bold_cell(marked[idx][metric_col])
    return marked


def rows_for_model(rows: list[list[str]], model: str, *, model_col: int = 1) -> list[list[str]]:
    return [row[:model_col] + row[model_col + 1 :] for row in rows if row[model_col] == model]


def merge_review_count_rows(rows: list[list[str]]) -> list[list[str]]:
    lr_rows = {tuple(row[idx] for idx in [0, 2, 3]): row for row in rows if row[1] == "LR"}
    xgb_rows = {tuple(row[idx] for idx in [0, 2, 3]): row for row in rows if row[1] == "XGBoost"}
    corpus_order = {
        "Synthetic 784x3": 0,
        "Synthetic 784x6": 1,
        "YouTube only": 2,
        "YouTube + synthetic 784x3": 3,
        "YouTube + synthetic 784x6": 4,
    }
    keys = sorted(
        lr_rows.keys() | xgb_rows.keys(),
        key=lambda key: (corpus_order.get(key[0], 99), int(key[1])),
    )
    merged: list[list[str]] = []
    for key in keys:
        lr = lr_rows.get(key)
        xgb = xgb_rows.get(key)
        merged.append(
            [
                key[0],
                key[1],
                key[2],
                lr[4] if lr else "NA",
                lr[5] if lr else "NA",
                lr[6] if lr else "NA",
                xgb[4] if xgb else "NA",
                xgb[5] if xgb else "NA",
                xgb[6] if xgb else "NA",
            ]
        )
    return merged


def row_lookup(rows: list[list[str]], *, key_cols: list[int], model: str) -> dict[tuple[str, ...], list[str]]:
    return {
        tuple(row[col] for col in key_cols): row
        for row in rows
        if row[1] == model
    }


def ordered_keys(rows: list[list[str]], key_cols: list[int]) -> list[tuple[str, ...]]:
    keys: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()
    for row in rows:
        key = tuple(row[col] for col in key_cols)
        if key not in seen:
            keys.append(key)
            seen.add(key)
    return keys


def merge_winner_rows(rows: list[list[str]]) -> list[list[str]]:
    lr_rows = row_lookup(rows, key_cols=[0], model="LR")
    xgb_rows = row_lookup(rows, key_cols=[0], model="XGBoost")
    merged: list[list[str]] = []
    for key in ordered_keys(rows, [0]):
        lr = lr_rows.get(key)
        xgb = xgb_rows.get(key)
        merged.append(
            [
                key[0],
                lr[2] if lr else "NA",
                lr[3] if lr else "NA",
                lr[4] if lr else "NA",
                lr[5] if lr else "NA",
                lr[6] if lr else "NA",
                xgb[2] if xgb else "NA",
                xgb[3] if xgb else "NA",
                xgb[4] if xgb else "NA",
                xgb[5] if xgb else "NA",
                xgb[6] if xgb else "NA",
            ]
        )
    return merged


def merge_synthetic_balancing_rows(rows: list[list[str]]) -> list[list[str]]:
    lr_rows = row_lookup(rows, key_cols=[0, 2], model="LR")
    xgb_rows = row_lookup(rows, key_cols=[0, 2], model="XGBoost")
    merged: list[list[str]] = []
    for key in ordered_keys(rows, [0, 2]):
        lr = lr_rows.get(key)
        xgb = xgb_rows.get(key)
        merged.append(
            [
                key[0],
                key[1],
                lr[3] if lr else "NA",
                lr[4] if lr else "NA",
                lr[5] if lr else "NA",
                lr[6] if lr else "NA",
                xgb[3] if xgb else "NA",
                xgb[4] if xgb else "NA",
                xgb[5] if xgb else "NA",
                xgb[6] if xgb else "NA",
            ]
        )
    return merged


def merge_youtube_balancing_rows(rows: list[list[str]]) -> list[list[str]]:
    lr_rows = row_lookup(rows, key_cols=[0, 2], model="LR")
    xgb_rows = row_lookup(rows, key_cols=[0, 2], model="XGBoost")
    merged: list[list[str]] = []
    for key in ordered_keys(rows, [0, 2]):
        lr = lr_rows.get(key)
        xgb = xgb_rows.get(key)
        merged.append(
            [
                key[0],
                key[1],
                lr[3] if lr else "NA",
                lr[4] if lr else "NA",
                lr[5] if lr else "NA",
                lr[6] if lr else "NA",
                lr[7] if lr else "NA",
                xgb[3] if xgb else "NA",
                xgb[4] if xgb else "NA",
                xgb[5] if xgb else "NA",
                xgb[6] if xgb else "NA",
                xgb[7] if xgb else "NA",
            ]
        )
    return merged


def precision_recall_from_row(row: pd.Series) -> tuple[float, float]:
    support = float(row["support_pos"])
    predicted = float(row["predicted_positive_count"])
    f1_value = float(row["f1"])
    true_positive = f1_value * (support + predicted) / 2.0
    precision = true_positive / predicted if predicted > 0 else 0.0
    recall = true_positive / support if support > 0 else 0.0
    return precision, recall


def load_per_label_rows() -> list[list[str]]:
    df = pd.read_csv(require(OG_PER_LABEL))
    df = df[
        df["eval_scope"].eq("heldout_og_rows")
        & df["threshold"].eq(0.3)
        & df["data_setup"].eq("og_plus_synthetic_6reviews")
        & df["model_family"].eq("Logistic Regression")
        & df["strategy"].isin(["unbalanced", "smote"])
    ].copy()
    label_order = ["joy", "trust", "anticipation", "surprise", "sadness", "disgust", "fear"]
    df["label_order"] = df["label"].map({label: idx for idx, label in enumerate(label_order)})
    baseline = df[df["strategy"].eq("unbalanced")].set_index("label")
    smote = df[df["strategy"].eq("smote")].set_index("label")
    rows: list[list[str]] = []
    for label in label_order:
        base_row = baseline.loc[label]
        smote_row = smote.loc[label]
        base_precision, base_recall = precision_recall_from_row(base_row)
        smote_precision, smote_recall = precision_recall_from_row(smote_row)
        rows.append(
            [
                label,
                str(int(base_row["support_pos"])),
                compact_prf(base_precision, base_recall, float(base_row["f1"])),
                str(int(base_row["predicted_positive_count"])),
                compact_prf(smote_precision, smote_recall, float(smote_row["f1"])),
                str(int(smote_row["predicted_positive_count"])),
                signed(float(smote_row["f1"]) - float(base_row["f1"])),
                signed_int(int(smote_row["predicted_positive_count"]) - int(base_row["predicted_positive_count"])),
            ]
        )
    return rows


def load_per_label_compact_rows() -> list[list[str]]:
    rows = load_per_label_rows()
    compact_rows: list[list[str]] = []
    for row in rows:
        compact_rows.append([row[0], row[1], row[2], row[4], row[6], row[7]])
    return compact_rows


def title_method(value: str) -> str:
    if value == "SMOTE":
        return value
    return value.capitalize()


def group_per_label_by_method() -> list[tuple[str, list[list[str]]]]:
    df = pd.read_csv(require(LABEL_DIST_BAL_SUMMARY.parent / "label_distribution_balancing_per_label.csv"))
    df = df[
        df["data_setup"].eq("og_plus_synthetic_6reviews")
        & df["model_family"].eq("Logistic Regression")
    ].copy()
    folds = pd.read_csv(require(LABEL_DIST_BAL_FOLDS))
    folds = folds[
        folds["data_setup"].eq("og_plus_synthetic_6reviews")
        & folds["model_family"].eq("Logistic Regression")
    ].copy()
    label_order = ["joy", "trust", "anticipation", "surprise", "sadness", "disgust", "fear"]
    methods = [
        ("label oversampling", "oversampling"),
        ("label undersampling", "undersampling"),
        ("label hybrid", "hybrid (undersampling + oversampling)"),
    ]
    baseline = df[df["method_label"].eq("baseline")].set_index("label")
    baseline_counts = folds[folds["method_label"].eq("baseline")]
    grouped: list[tuple[str, list[list[str]]]] = []
    for method_key, method_label in methods:
        method_df = df[df["method_label"].eq(method_key)].set_index("label")
        method_counts = folds[folds["method_label"].eq(method_key)]
        rows: list[list[str]] = []
        for label in label_order:
            base_row = baseline.loc[label]
            method_row = method_df.loc[label]
            count_col = f"after_count_{label}"
            rows.append(
                [
                    label,
                    fmt(baseline_counts[count_col].mean(), 0),
                    fmt(method_counts[count_col].mean(), 0),
                    fmt(base_row["f1"]),
                    fmt(method_row["f1"]),
                    signed(float(method_row["f1"]) - float(base_row["f1"])),
                    signed_int(
                        int(method_row["predicted_positive_count"])
                        - int(base_row["predicted_positive_count"])
                    ),
                ]
            )
        grouped.append((method_label, rows))

    actual_smote = pd.read_csv(require(OG_PER_LABEL))
    actual_smote = actual_smote[
        actual_smote["eval_scope"].eq("heldout_og_rows")
        & actual_smote["threshold"].eq(0.3)
        & actual_smote["data_setup"].eq("og_plus_synthetic_6reviews")
        & actual_smote["model_family"].eq("Logistic Regression")
    ].copy()
    actual_smote_baseline = actual_smote[actual_smote["strategy"].eq("unbalanced")].set_index("label")
    actual_smote_rows = actual_smote[actual_smote["strategy"].eq("smote")].set_index("label")
    smote_table_rows: list[list[str]] = []
    for label in label_order:
        base_row = actual_smote_baseline.loc[label]
        method_row = actual_smote_rows.loc[label]
        count_col = f"after_count_{label}"
        smote_counts = []
        for _, fold_row in baseline_counts.iterrows():
            positive_count = float(fold_row[count_col])
            negative_count = float(fold_row["after_rows"]) - positive_count
            smote_counts.append(max(positive_count, negative_count))
        smote_table_rows.append(
            [
                label,
                fmt(baseline_counts[count_col].mean(), 0),
                fmt(pd.Series(smote_counts).mean(), 0),
                fmt(base_row["f1"]),
                fmt(method_row["f1"]),
                signed(float(method_row["f1"]) - float(base_row["f1"])),
                signed_int(
                    int(method_row["predicted_positive_count"])
                    - int(base_row["predicted_positive_count"])
                ),
            ]
        )
    grouped.append(("SMOTE", smote_table_rows))
    return grouped


def load_label_distribution_balance_rows() -> list[list[str]]:
    summary = pd.read_csv(require(LABEL_DIST_BAL_SUMMARY))
    folds = pd.read_csv(require(LABEL_DIST_BAL_FOLDS))
    method_order = {
        "baseline": 0,
        "label oversampling": 1,
        "label undersampling": 2,
        "label hybrid": 3,
    }
    setup_order = {"og_only": 0, "og_plus_synthetic_3reviews": 1, "og_plus_synthetic_6reviews": 2}
    fold_means = (
        folds.groupby(["data_setup", "method_label"], as_index=False)
        .agg(
            before_ratio=("before_label_max_min_ratio", "mean"),
            after_ratio=("after_label_max_min_ratio", "mean"),
            after_rows=("after_rows", "mean"),
        )
    )
    merged = summary.merge(fold_means, on=["data_setup", "method_label"], how="left")
    merged["setup_order"] = merged["data_setup"].map(setup_order)
    merged["method_order"] = merged["method_label"].map(method_order)
    merged = merged.sort_values(["setup_order", "method_order", "model_family"])

    rows: list[list[str]] = []
    for (data_setup, method_label), group in merged.groupby(["data_setup", "method_label"], sort=False):
        lr = group[group["model_family"].eq("Logistic Regression")].iloc[0]
        xgb = group[group["model_family"].eq("XGBoost")].iloc[0]
        rows.append(
            [
                display_data_label(str(lr["data_label"])),
                str(method_label).replace("label ", "").replace("baseline", "No balancing method"),
                f'{fmt(lr["before_ratio"], 1)} -> {fmt(lr["after_ratio"], 1)}',
                fmt(lr["after_rows"], 0),
                fmt(lr["macro_f1"]),
                signed(lr["delta_macro_f1_vs_unbalanced"]),
                f'P {fmt(lr["macro_precision"], 2)}\nR {fmt(lr["macro_recall"], 2)}',
                fmt(xgb["macro_f1"]),
                signed(xgb["delta_macro_f1_vs_unbalanced"]),
                f'P {fmt(xgb["macro_precision"], 2)}\nR {fmt(xgb["macro_recall"], 2)}',
            ]
        )
    return mark_best_scores(rows, group_cols=[0], metric_cols=[4, 7])


def load_label_distribution_best_rows() -> list[list[str]]:
    summary = pd.read_csv(require(LABEL_DIST_BAL_SUMMARY))
    folds = pd.read_csv(require(LABEL_DIST_BAL_FOLDS))
    fold_means = (
        folds.groupby(["data_setup", "method_label"], as_index=False)
        .agg(
            before_ratio=("before_label_max_min_ratio", "mean"),
            after_ratio=("after_label_max_min_ratio", "mean"),
        )
    )
    data = summary.merge(fold_means, on=["data_setup", "method_label"], how="left")
    setup_order = {"og_only": 0, "og_plus_synthetic_3reviews": 1, "og_plus_synthetic_6reviews": 2}
    model_order = {"Logistic Regression": 0, "XGBoost": 1}
    data["setup_order"] = data["data_setup"].map(setup_order)
    data["model_order"] = data["model_family"].map(model_order)
    rows: list[list[str]] = []
    for (data_setup, model_family), group in data.groupby(["data_setup", "model_family"], sort=False):
        baseline = group[group["method"].eq("unbalanced")].iloc[0]
        balanced = group[~group["method"].eq("unbalanced")].sort_values("macro_f1", ascending=False).iloc[0]
        rows.append(
            [
                display_data_label(str(baseline["data_label"])),
                model_short(str(model_family)),
                fmt(baseline["macro_f1"]),
                str(balanced["method_label"]).replace("label ", ""),
                fmt(balanced["macro_f1"]),
                signed(balanced["delta_macro_f1_vs_unbalanced"]),
                f'{fmt(balanced["before_ratio"], 1)} -> {fmt(balanced["after_ratio"], 1)}',
            ]
        )
    label_order = {"YouTube only": 0, "YouTube + synthetic 784x3": 1, "YouTube + synthetic 784x6": 2}
    rows = sorted(rows, key=lambda row: (label_order.get(row[0], 9), 0 if row[1] == "LR" else 1))
    return mark_best_scores(rows, group_cols=[0], metric_cols=[4])


def load_review_count_rows() -> list[list[str]]:
    rows: list[list[str]] = []
    synth3 = pd.read_csv(require(SYNTH_3_COUNT)).copy()
    synth3["corpus"] = "Synthetic 784x3"
    synth3["threshold"] = 0.3
    synth6 = pd.read_csv(require(SYNTH_6_COUNT)).copy()
    synth6 = synth6[synth6["threshold"].eq(0.3)].copy()
    synth6["corpus"] = "Synthetic 784x6"
    both = pd.concat([synth3, synth6], ignore_index=True)
    both = both.sort_values(["corpus", "model_family", "synthetic_reviews"])
    for _, row in both.iterrows():
        rows.append(
            [
                row["corpus"],
                model_short(row["model_family"]),
                str(int(row["synthetic_reviews"])),
                str(int(row["synthetic_perfumes"])),
                mean_sd(row["macro_f1_mean"], row["macro_f1_std"]),
                fmt(row["macro_roc_auc_mean"]),
                fmt(row["macro_pr_auc_mean"]),
            ]
        )
    mixed = pd.read_csv(require(LABEL_DIST_BAL_SUMMARY)).copy()
    mixed = mixed[
        mixed["threshold"].eq(0.3)
        & mixed["method"].eq("unbalanced")
        & mixed["data_setup"].isin(
            ["og_only", "og_plus_synthetic_3reviews", "og_plus_synthetic_6reviews"]
        )
    ].copy()
    mixed = mixed.sort_values(["data_setup", "model_family"])
    for _, row in mixed.iterrows():
        rows.append(
            [
                display_data_label(str(row["data_label"])),
                model_short(row["model_family"]),
                str(int(row["train_rows_original"])),
                str(int(row["og_train_rows"])),
                fmt(row["macro_f1"]),
                fmt(row["macro_roc_auc"]),
                fmt(row["macro_pr_auc"]),
            ]
        )
    return mark_best_scores(rows, group_cols=[0, 1], metric_cols=[4, 5, 6])


def load_review_count_compact_rows() -> list[list[str]]:
    return [[row[0], row[1], row[2], row[3], row[6]] for row in merge_review_count_rows(load_review_count_rows())]


def load_synthetic_balancing_rows() -> list[list[str]]:
    frames = []
    for corpus, path in [("Synthetic 784x3", SYNTH_3_BAL), ("Synthetic 784x6", SYNTH_6_BAL)]:
        df = pd.read_csv(require(path)).copy()
        df["corpus"] = corpus
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    strategy_order = {
        "unbalanced": 0,
        "class_weight": 1,
        "scale_pos_weight": 1,
        "oversampling": 2,
        "undersampling": 3,
        "hybrid": 4,
        "smote": 5,
    }
    data["strategy_order"] = data["strategy"].map(strategy_order)
    data = data.sort_values(["corpus", "model_family", "strategy_order", "strategy"])
    rows = []
    for _, row in data.iterrows():
        rows.append(
            [
                row["corpus"],
                model_short(row["model_family"]),
                clean_strategy(row["strategy"]),
                mean_sd(row["macro_f1_mean"], row["macro_f1_std"]),
                signed(row["delta_macro_f1_mean_vs_unbalanced"]),
                fmt(row["macro_roc_auc_mean"]),
                fmt(row["macro_pr_auc_mean"]),
            ]
        )
    return mark_best_scores(rows, group_cols=[0, 1], metric_cols=[3, 5, 6])


def load_synthetic_balancing_best_rows() -> list[list[str]]:
    frames = []
    for corpus, path in [("Synthetic 784x3", SYNTH_3_BAL), ("Synthetic 784x6", SYNTH_6_BAL)]:
        df = pd.read_csv(require(path)).copy()
        df["corpus"] = corpus
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    rows: list[list[str]] = []
    for (corpus, model_family), group in data.groupby(["corpus", "model_family"], sort=False):
        baseline = group[group["strategy"].eq("unbalanced")].iloc[0]
        best = group.sort_values("macro_f1_mean", ascending=False).iloc[0]
        rows.append(
            [
                corpus,
                model_short(str(model_family)),
                mean_sd(baseline["macro_f1_mean"], baseline["macro_f1_std"]),
                clean_strategy(str(best["strategy"])),
                mean_sd(best["macro_f1_mean"], best["macro_f1_std"]),
                signed(best["delta_macro_f1_mean_vs_unbalanced"]),
            ]
        )
    rows = sorted(rows, key=lambda row: (row[0], 0 if row[1] == "LR" else 1))
    return mark_best_scores(rows, group_cols=[0], metric_cols=[4])


def group_synthetic_balancing_by_method() -> list[tuple[str, list[list[str]]]]:
    rows = load_synthetic_balancing_rows()
    methods = ["No balancing method", "oversampling", "undersampling", "SMOTE"]
    corpora = ["Synthetic 784x3", "Synthetic 784x6"]
    grouped: list[tuple[str, list[list[str]]]] = []
    for method in methods:
        method_rows: list[list[str]] = []
        for corpus in corpora:
            lr = next(row for row in rows if row[0] == corpus and row[1] == "LR" and row[2] == method)
            xgb = next(row for row in rows if row[0] == corpus and row[1] == "XGBoost" and row[2] == method)
            method_rows.append([corpus, lr[3], lr[4], xgb[3], xgb[4]])
        grouped.append((method, method_rows))
    return grouped


def load_og_balancing_rows() -> list[list[str]]:
    df = pd.read_csv(require(OG_BAL))
    df = df[df["eval_scope"].eq("heldout_og_rows") & df["threshold"].eq(0.3)].copy()
    strategy_order = {
        "unbalanced": 0,
        "class_weight": 1,
        "scale_pos_weight": 1,
        "oversampling": 2,
        "undersampling": 3,
        "smote": 4,
    }
    setup_order = {"og_only": 0, "og_plus_synthetic_3reviews": 1, "og_plus_synthetic_6reviews": 2}
    df["strategy_order"] = df["strategy"].map(strategy_order)
    df["setup_order"] = df["data_setup"].map(setup_order)
    df = df.sort_values(["setup_order", "model_family", "strategy_order", "strategy"])
    rows = []
    for _, row in df.iterrows():
        rows.append(
            [
                display_data_label(row["data_label"]),
                model_short(row["model_family"]),
                clean_strategy(row["strategy"]),
                fmt(row["macro_f1"]),
                signed(row["delta_macro_f1_vs_unbalanced"]),
                fmt(row["macro_roc_auc"]),
                fmt(row["macro_pr_auc"]),
                fmt(row["predicted_label_cardinality"], 2),
            ]
        )
    return mark_best_scores(rows, group_cols=[0, 1], metric_cols=[3, 5, 6])


def binary_balancing_train_positive_counts(label_order: list[str]) -> dict[tuple[str, str, str], float]:
    og_df = pd.read_csv(require(OG_FEATURES))
    synthetic_3 = pd.read_csv(require(SYNTHETIC_3_FEATURES))
    synthetic_6 = pd.read_csv(require(SYNTHETIC_6_FEATURES))
    folds = pd.read_csv(require(OG_FOLD_ASSIGNMENTS))
    setups = {
        "og_only": og_df,
        "og_plus_synthetic_3reviews": pd.concat([og_df, synthetic_3], ignore_index=True),
        "og_plus_synthetic_6reviews": pd.concat([og_df, synthetic_6], ignore_index=True),
    }
    strategies = ["unbalanced", "oversampling", "undersampling", "smote"]
    counts: dict[tuple[str, str, str], float] = {}
    fold_ids = sorted(int(value) for value in folds["fold_id"].unique().tolist())
    for setup_key, train_df in setups.items():
        train_perfumes = train_df["perfume_id"].astype(str)
        for label in label_order:
            fold_counts: dict[str, list[float]] = {strategy: [] for strategy in strategies}
            for fold_id in fold_ids:
                test_ids = set(
                    folds.loc[folds["fold_id"].astype(int).eq(fold_id), "perfume_id"].astype(str)
                )
                y_train = train_df.loc[~train_perfumes.isin(test_ids), label].astype(int)
                positive = float(y_train.sum())
                negative = float(len(y_train) - positive)
                fold_counts["unbalanced"].append(positive)
                fold_counts["oversampling"].append(max(positive, negative))
                fold_counts["undersampling"].append(min(positive, negative))
                fold_counts["smote"].append(max(positive, negative))
            for strategy, values in fold_counts.items():
                counts[(setup_key, strategy, label)] = float(pd.Series(values).mean())
    return counts


def group_youtube_balancing_by_method() -> list[tuple[str, list[list[str]]]]:
    df = pd.read_csv(require(OG_PER_LABEL))
    df = df[df["eval_scope"].eq("heldout_og_rows") & df["threshold"].eq(0.3)].copy()
    method_lookup = {
        "No balancing method": "unbalanced",
        "oversampling": "oversampling",
        "undersampling": "undersampling",
        "SMOTE": "smote",
    }
    setups = [
        ("og_only", "YouTube only"),
        ("og_plus_synthetic_3reviews", "YouTube + synthetic 784x3"),
        ("og_plus_synthetic_6reviews", "YouTube + synthetic 784x6"),
    ]
    label_order = ["joy", "trust", "anticipation", "surprise", "sadness", "disgust", "fear"]
    train_positive_counts = binary_balancing_train_positive_counts(label_order)
    baseline = df[df["strategy"].eq("unbalanced")].set_index(["data_setup", "model_family", "label"])
    grouped: list[tuple[str, list[list[str]]]] = []
    for method, strategy in method_lookup.items():
        method_df = df[df["strategy"].eq(strategy)].set_index(["data_setup", "model_family", "label"])
        method_rows: list[list[str]] = []
        for setup_key, setup_label in setups:
            for label in label_order:
                lr = method_df.loc[(setup_key, "Logistic Regression", label)]
                xgb = method_df.loc[(setup_key, "XGBoost", label)]
                lr_base = baseline.loc[(setup_key, "Logistic Regression", label)]
                xgb_base = baseline.loc[(setup_key, "XGBoost", label)]
                method_rows.append(
                    [
                        setup_label,
                        label,
                        fmt(train_positive_counts[(setup_key, strategy, label)], 0),
                        fmt(lr["f1"]),
                        signed(float(lr["f1"]) - float(lr_base["f1"])),
                        fmt(xgb["f1"]),
                        signed(float(xgb["f1"]) - float(xgb_base["f1"])),
                    ]
                )
        grouped.append((method, method_rows))
    return grouped


def load_youtube_balancing_best_rows() -> list[list[str]]:
    df = pd.read_csv(require(OG_BAL))
    df = df[df["eval_scope"].eq("heldout_og_rows") & df["threshold"].eq(0.3)].copy()
    setup_order = {"og_only": 0, "og_plus_synthetic_3reviews": 1, "og_plus_synthetic_6reviews": 2}
    rows: list[list[str]] = []
    for (data_setup, model_family), group in df.groupby(["data_setup", "model_family"], sort=False):
        baseline = group[group["strategy"].eq("unbalanced")].iloc[0]
        best = group.sort_values("macro_f1", ascending=False).iloc[0]
        rows.append(
            [
                display_data_label(str(baseline["data_label"])),
                model_short(str(model_family)),
                fmt(baseline["macro_f1"]),
                clean_strategy(str(best["strategy"])),
                fmt(best["macro_f1"]),
                signed(best["delta_macro_f1_vs_unbalanced"]),
            ]
        )
    label_order = {"YouTube only": 0, "YouTube + synthetic 784x3": 1, "YouTube + synthetic 784x6": 2}
    rows = sorted(rows, key=lambda row: (label_order.get(row[0], 9), 0 if row[1] == "LR" else 1))
    return mark_best_scores(rows, group_cols=[0], metric_cols=[4])


def group_label_distribution_balancing_by_method() -> list[tuple[str, list[list[str]]]]:
    rows = load_label_distribution_balance_rows()
    actual_smote = pd.read_csv(require(OG_BAL))
    actual_smote = actual_smote[
        actual_smote["eval_scope"].eq("heldout_og_rows")
        & actual_smote["threshold"].eq(0.3)
        & actual_smote["strategy"].eq("smote")
    ].copy()
    methods = ["No balancing method", "oversampling", "undersampling", "hybrid", "SMOTE"]
    setups = ["YouTube only", "YouTube + synthetic 784x3", "YouTube + synthetic 784x6"]
    grouped: list[tuple[str, list[list[str]]]] = []
    for method in methods:
        method_rows: list[list[str]] = []
        for setup in setups:
            if method == "SMOTE":
                group = actual_smote[actual_smote["data_label"].map(display_data_label).eq(setup)]
                lr = group[group["model_family"].eq("Logistic Regression")].iloc[0]
                xgb = group[group["model_family"].eq("XGBoost")].iloc[0]
                method_rows.append(
                    [
                        setup,
                        fmt(lr["macro_f1"]),
                        signed(lr["delta_macro_f1_vs_unbalanced"]),
                        fmt(xgb["macro_f1"]),
                        signed(xgb["delta_macro_f1_vs_unbalanced"]),
                    ]
                )
            else:
                row = next(row for row in rows if row[0] == setup and row[1] == method)
                method_rows.append([setup, row[4], row[5], row[7], row[8]])
        grouped.append((method, method_rows))
    return grouped


def build_section20_method_snapshot(
    grouped_tables: list[tuple[str, list[list[str]]]],
) -> list[list[str]]:
    """Compare the key Section 2 methods in one compact table."""
    methods = ["No balancing method", "hybrid", "SMOTE"]
    setups = ["YouTube only", "YouTube + synthetic 784x3", "YouTube + synthetic 784x6"]
    by_method = {method: {row[0]: row for row in rows} for method, rows in grouped_tables if method in methods}

    rows: list[list[str]] = []
    for corpus, path in [("Synthetic 784x3", SYNTH_3_BAL), ("Synthetic 784x6", SYNTH_6_BAL)]:
        synthetic = pd.read_csv(require(path)).copy()
        synthetic = synthetic[synthetic["threshold"].eq(0.3)]

        def synthetic_score(model_family: str, strategy: str) -> str:
            match = synthetic[synthetic["model_family"].eq(model_family) & synthetic["strategy"].eq(strategy)]
            return "NA" if match.empty else fmt(match.iloc[0]["macro_f1_mean"])

        rows.append(
            [
                corpus,
                synthetic_score("Logistic Regression", "unbalanced"),
                synthetic_score("Logistic Regression", "hybrid"),
                synthetic_score("Logistic Regression", "smote"),
                synthetic_score("XGBoost", "unbalanced"),
                synthetic_score("XGBoost", "hybrid"),
                synthetic_score("XGBoost", "smote"),
            ]
        )
    for setup in setups:
        rows.append(
            [
                setup,
                by_method["No balancing method"][setup][1],
                by_method["hybrid"][setup][1],
                by_method["SMOTE"][setup][1],
                by_method["No balancing method"][setup][3],
                by_method["hybrid"][setup][3],
                by_method["SMOTE"][setup][3],
            ]
        )

    for row in rows:
        lr_cols = [1, 2, 3]
        xgb_cols = [4, 5, 6]
        best_lr = max(parse_score(row[col]) for col in lr_cols)
        best_xgb = max(parse_score(row[col]) for col in xgb_cols)
        for col in lr_cols:
            if best_lr > float("-inf") and parse_score(row[col]) == best_lr:
                row[col] = bold_cell(row[col])
        for col in xgb_cols:
            if best_xgb > float("-inf") and parse_score(row[col]) == best_xgb:
                row[col] = bold_cell(row[col])
    return rows


def load_emotion_count_rows() -> list[list[str]]:
    folds = pd.read_csv(require(LABEL_DIST_BAL_FOLDS))
    # The resampled training counts are model-independent; keep one model family
    # to avoid duplicate fold rows in the report.
    folds = folds[folds["model_family"].eq("Logistic Regression")].copy()
    setup_order = {
        "og_only": "YouTube only",
        "og_plus_synthetic_3reviews": "YouTube + synthetic 784x3",
        "og_plus_synthetic_6reviews": "YouTube + synthetic 784x6",
    }
    method_order = {
        "baseline": 0,
        "label oversampling": 1,
        "label undersampling": 2,
        "label hybrid": 3,
        "SMOTE": 4,
    }
    method_names = {
        "baseline": "No balancing method",
        "label oversampling": "Oversampling",
        "label undersampling": "Undersampling",
        "label hybrid": "hybrid",
        "SMOTE": "SMOTE",
    }
    count_cols = [
        "after_count_joy",
        "after_count_sadness",
        "after_count_fear",
        "after_count_trust",
        "after_count_surprise",
        "after_count_anticipation",
        "after_count_disgust",
    ]
    rows: list[list[str]] = []
    count_methods = ["baseline", "label oversampling", "label undersampling", "label hybrid"]
    for setup_key, setup_label in setup_order.items():
        setup_folds = folds[folds["data_setup"].eq(setup_key)].copy()
        grouped = (
            setup_folds[setup_folds["method_label"].isin(count_methods)]
            .groupby("method_label", as_index=False)
            .agg(after_rows=("after_rows", "mean"), **{col: (col, "mean") for col in count_cols})
        )
        baseline_counts = setup_folds[setup_folds["method_label"].eq("baseline")]
        smote_row: dict[str, Any] = {"method_label": "SMOTE", "after_rows": "label-wise"}
        for count_col in count_cols:
            smote_counts = []
            for _, fold_row in baseline_counts.iterrows():
                positive_count = float(fold_row[count_col])
                negative_count = float(fold_row["after_rows"]) - positive_count
                smote_counts.append(max(positive_count, negative_count))
            smote_row[count_col] = pd.Series(smote_counts).mean()
        grouped = pd.concat([grouped, pd.DataFrame([smote_row])], ignore_index=True)
        grouped["method_order"] = grouped["method_label"].map(method_order)
        grouped = grouped.sort_values("method_order")

        sd_values: list[float] = []
        for _, row in grouped.iterrows():
            emotion_counts = [float(row[col]) for col in count_cols]
            sd_values.append(pd.Series(emotion_counts).std(ddof=0))
        min_sd = min(sd_values)
        for (_, row), count_sd in zip(grouped.iterrows(), sd_values, strict=True):
            sd_text = fmt(count_sd, 1)
            if count_sd == min_sd:
                sd_text = bold_cell(sd_text)
            rows.append(
                [
                    setup_label,
                    display_method(method_names[str(row["method_label"])]),
                    fmt(row["after_rows"], 0),
                    sd_text,
                    fmt(row["after_count_joy"], 0),
                    fmt(row["after_count_sadness"], 0),
                    fmt(row["after_count_fear"], 0),
                    fmt(row["after_count_trust"], 0),
                    fmt(row["after_count_surprise"], 0),
                    fmt(row["after_count_anticipation"], 0),
                    fmt(row["after_count_disgust"], 0),
                ]
            )
    return rows


def load_winner_rows() -> list[list[str]]:
    winners = pd.read_csv(require(OG_WINNERS))
    winners = winners[winners["threshold"].eq(0.3)].copy()
    rows = []
    for _, row in winners.iterrows():
        rows.append(
            [
                display_data_label(row["data_label"]),
                model_short(row["model_family"]),
                clean_strategy(row["strategy"]),
                fmt(row["macro_f1"]),
                signed(row["delta_macro_f1_vs_unbalanced"]),
                fmt(row["macro_roc_auc"]),
                fmt(row["macro_pr_auc"]),
            ]
        )
    return mark_best_scores(rows, group_cols=[1], metric_cols=[3, 5, 6])


def main() -> None:
    review_count_rows = load_review_count_compact_rows()
    synthetic_balancing_tables = group_synthetic_balancing_by_method()
    youtube_balancing_tables = group_youtube_balancing_by_method()
    label_distribution_balance_tables = group_label_distribution_balancing_by_method()
    emotion_count_rows = load_emotion_count_rows()

    best_overall = pd.read_csv(require(OG_WINNERS))
    best_overall = best_overall[best_overall["threshold"].eq(0.3)].sort_values("macro_f1", ascending=False).iloc[0]
    label_balance_summary = pd.read_csv(require(LABEL_DIST_BAL_SUMMARY))
    lr_label_balance_summary = label_balance_summary[label_balance_summary["model_family"].eq("Logistic Regression")]
    best_lr_label_balance = (
        lr_label_balance_summary[~lr_label_balance_summary["method"].eq("unbalanced")]
        .sort_values("macro_f1", ascending=False)
        .iloc[0]
    )
    best_lr_overall_shown = lr_label_balance_summary.sort_values("macro_f1", ascending=False).iloc[0]
    full_hybrid = label_balance_summary[
        label_balance_summary["data_setup"].eq("og_plus_synthetic_6reviews")
        & label_balance_summary["method_label"].eq("label hybrid")
    ]
    hybrid_lr = full_hybrid[full_hybrid["model_family"].eq("Logistic Regression")].iloc[0]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{PAGE_W}" height="6000" viewBox="0 0 {PAGE_W} 6000">',
        '<rect width="100%" height="100%" fill="#f4f7fb"/>',
        text_element(
            x=MARGIN,
            y=46,
            text="Synthetic and YouTube Combination Results",
            size=23,
            weight="700",
            color=TEXT,
        ),
        text_element(
            x=MARGIN,
            y=74,
            text="Threshold 0.3",
            size=12,
            weight="700",
            color=MUTED,
        ),
    ]
    y = 100

    count_headers = [
        "Corpus",
        "Rows",
        "Perfumes",
        "LR F1",
        "XGBoost F1",
    ]
    count_widths = [170, 75, 80, 150, 150]
    table_parts, y = render_table(
        title="1. Review Count Effect: No Balancing Method",
        headers=count_headers,
        rows=review_count_rows,
        widths=count_widths,
        y_start=y,
        accent=BLUE,
        center_cols={1, 2, 3, 4},
        separator_after_cols={3},
        merge_cols={0},
    )
    parts.extend(table_parts)
    note_parts, y = render_method_note(
        "No balancing method; mixed rows are evaluated on YouTube rows.",
        y,
        MUTED,
    )
    parts.extend(note_parts)
    y += SECTION_GAP

    heading_parts, y = render_section_heading(
        "Emotion-count balancing",
        "Main question: balance emotion counts and compare F1.",
        y,
        AMBER,
    )
    parts.extend(heading_parts)

    section20_rows = build_section20_method_snapshot(label_distribution_balance_tables)
    section20_widths = [170, 112, 112, 112, 135, 135, 135]
    table_parts, y = render_section20_table(
        title="2.0 Summary: No Balancing vs Hybrid vs SMOTE",
        rows=section20_rows,
        widths=section20_widths,
        y_start=y,
        accent=AMBER,
    )
    parts.extend(table_parts)
    note_parts, y = render_method_note("Best F1 in each row/model is shaded.", y, MUTED)
    parts.extend(note_parts)
    y += 18

    label_distribution_headers = [
        "Training data",
        "LR F1",
        "LR change",
        "XGBoost F1",
        "XGBoost change",
    ]
    label_distribution_widths = [190, 100, 90, 100, 95]
    for idx, (method, rows) in enumerate(label_distribution_balance_tables, start=1):
        if method == "No balancing method":
            section_suffix = "Reference"
            table_headers = ["Training data", "LR F1", "XGBoost F1"]
            table_widths = [190, 120, 120]
            table_rows = [[row[0], row[1], row[3]] for row in rows]
            center_cols = {1, 2}
            separator_cols = {1}
        else:
            section_suffix = "Binary-label reference" if method == "SMOTE" else "Cross-emotion count balancing"
            table_headers = label_distribution_headers
            table_widths = label_distribution_widths
            table_rows = rows
            center_cols = {1, 2, 3, 4}
            separator_cols = {2}
        table_parts, y = render_table(
            title=f"2.{idx} {display_method(method)}: {section_suffix}",
            headers=table_headers,
            rows=table_rows,
            widths=table_widths,
            y_start=y,
            accent=AMBER,
            center_cols=center_cols,
            separator_after_cols=separator_cols,
        )
        parts.extend(table_parts)
        note_parts, y = render_method_note(LABEL_METHOD_NOTES[method], y, MUTED)
        parts.extend(note_parts)
        y += 18
    y += SECTION_GAP

    heading_parts, y = render_section_heading(
        "Balance achieved",
        "Show the actual emotion counts after each balancing method.",
        y,
        TEAL,
    )
    parts.extend(heading_parts)

    emotion_count_headers = [
        "Training data",
        "Method",
        "Rows",
        "Count SD",
        "Joy",
        "Sadness",
        "Fear",
        "Trust",
        "Surprise",
        "Anticipation",
        "Disgust",
    ]
    emotion_count_widths = [140, 190, 74, 70, 52, 64, 50, 52, 70, 94, 68]
    table_parts, y = render_table(
        title="3. Mean Emotion Counts After Balancing",
        headers=emotion_count_headers,
        rows=emotion_count_rows,
        widths=emotion_count_widths,
        y_start=y,
        accent=TEAL,
        center_cols={2, 3, 4, 5, 6, 7, 8, 9, 10},
        merge_cols={0},
        row_pad_y=6,
        line_h=16,
        header_min_h=38,
        body_min_h=34,
        title_gap=24,
    )
    parts.extend(table_parts)
    note_parts, y = render_method_note("Lower Count SD means better balance within the same training setup.", y, MUTED)
    parts.extend(note_parts)
    y += SECTION_GAP

    heading_parts, y = render_section_heading(
        "Binary emotion balancing",
        "Balance each emotion as positive vs negative; SMOTE belongs in this group.",
        y,
        RED,
    )
    parts.extend(heading_parts)

    youtube_headers = [
        "Training data",
        "Emotion",
        "Train emotion count",
        "LR F1",
        "LR change",
        "XGBoost F1",
        "XGBoost change",
    ]
    youtube_widths = [170, 78, 86, 76, 84, 88, 94]
    for idx, (method, rows) in enumerate(youtube_balancing_tables, start=1):
        if method == "No balancing method":
            table_headers = ["Training data", "Emotion", "Train emotion count", "LR F1", "XGBoost F1"]
            table_widths = [170, 78, 96, 96, 108]
            table_rows = [[row[0], row[1], row[2], row[3], row[5]] for row in rows]
            center_cols = {1, 2, 3, 4}
            separator_cols = {3}
        else:
            table_headers = youtube_headers
            table_widths = youtube_widths
            table_rows = rows
            center_cols = {1, 2, 3, 4, 5, 6}
            separator_cols = {4}
        table_parts, y = render_table(
            title=f"4.{idx} {method}: Binary-label results by emotion",
            headers=table_headers,
            rows=table_rows,
            widths=table_widths,
            y_start=y,
            accent=RED,
            center_cols=center_cols,
            separator_after_cols=separator_cols,
            merge_cols={0},
        )
        parts.extend(table_parts)
        note_parts, y = render_method_note(BALANCING_METHOD_NOTES[method], y, MUTED)
        parts.extend(note_parts)
        y += 18
    y += SECTION_GAP

    heading_parts, y = render_section_heading(
        "Synthetic-only balancing",
        "Apply binary balancing methods when training and testing only on synthetic reviews.",
        y,
        PURPLE,
    )
    parts.extend(heading_parts)

    method_headers = ["Corpus", "LR F1", "LR change", "XGBoost F1", "XGBoost change"]
    method_widths = [170, 135, 95, 135, 105]
    for idx, (method, rows) in enumerate(synthetic_balancing_tables, start=1):
        if method == "No balancing method":
            table_headers = ["Corpus", "LR F1", "XGBoost F1"]
            table_widths = [170, 150, 150]
            table_rows = [[row[0], row[1], row[3]] for row in rows]
            center_cols = {1, 2}
            separator_cols = {1}
        else:
            table_headers = method_headers
            table_widths = method_widths
            table_rows = rows
            center_cols = {1, 2, 3, 4}
            separator_cols = {2}
        table_parts, y = render_table(
            title=f"5.{idx} {method}: Synthetic-only",
            headers=table_headers,
            rows=table_rows,
            widths=table_widths,
            y_start=y,
            accent=PURPLE,
            center_cols=center_cols,
            separator_after_cols=separator_cols,
        )
        parts.extend(table_parts)
        note_parts, y = render_method_note(BALANCING_METHOD_NOTES[method], y, MUTED)
        parts.extend(note_parts)
        y += 18
    y += SECTION_GAP

    parts.append(
        f'<line x1="{MARGIN}" y1="{y - 12}" x2="{MARGIN + 640}" y2="{y - 12}" '
        'stroke="#99f6e4" stroke-width="2"/>'
    )
    footer_lines = [
        (
            "Best LR count balance for YouTube + synthetic 784x6: "
            f"hybrid (undersampling + oversampling), Count SD 98.5, LR F1 {fmt(hybrid_lr['macro_f1'])}."
        ),
        (
            f"Best LR F1 among balancing methods: {display_data_label(best_lr_label_balance['data_label'])} + "
            f"{str(best_lr_label_balance['method_label']).replace('label ', '')}, Macro F1 "
            f"{fmt(best_lr_label_balance['macro_f1'])}."
        ),
        (
            f"Best LR overall shown: {display_data_label(best_lr_overall_shown['data_label'])} + "
            f"no balancing method, Macro F1 {fmt(best_lr_overall_shown['macro_f1'])}."
        ),
    ]
    footer_y = y + 4
    for footer_line in footer_lines:
        parts.append(
            text_element(
                x=MARGIN,
                y=footer_y,
                text=footer_line,
                size=13,
                weight="700",
                color="#047857",
            )
        )
        footer_y += 21
    y += 58

    parts[0] = f'<svg xmlns="http://www.w3.org/2000/svg" width="{PAGE_W}" height="{y}" viewBox="0 0 {PAGE_W} {y}">'
    parts.append("</svg>\n")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(parts), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()

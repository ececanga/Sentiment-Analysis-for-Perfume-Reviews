#!/usr/bin/env python3
"""Build a compact advisor report for the 784x6 synthetic-only threshold-0.5 sweep."""

from __future__ import annotations

import csv
import html
import json
import math
import textwrap
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/synthetic_review_experiment_report/table_images/05_synthetic_only_784x6_threshold05_compact_report.svg"

COMBINED_FEATURES = (
    ROOT
    / "outputs/synthetic_review_level_tests/emotion_rich_v2_6reviews/supervised_eval_gemini_browser"
    / "review_level_features_labeled_g25pro.csv"
)
COMBINED_REPORT = (
    ROOT
    / "outputs/synthetic_review_level_tests/emotion_rich_v2_6reviews/supervised_eval_gemini_browser"
    / "review_level_features_labeled_g25pro_report.json"
)
THRESHOLD05_SUMMARY = (
    ROOT
    / "outputs/synthetic_only_random_review_count_sweep_gemini_browser_6reviews_threshold05"
    / "synthetic_only_random_review_count_sweep_summary.csv"
)
THRESHOLD05_REPORT = (
    ROOT
    / "outputs/synthetic_only_random_review_count_sweep_gemini_browser_6reviews_threshold05"
    / "synthetic_only_random_review_count_sweep_report.json"
)
THRESHOLD_SWEEP_SUMMARY = (
    ROOT
    / "outputs/synthetic_only_random_review_count_sweep_gemini_browser_6reviews_thresholds"
    / "synthetic_only_random_review_count_sweep_summary.csv"
)
OLD_3REVIEW_RANDOM_SUMMARY = (
    ROOT
    / "outputs/synthetic_only_random_review_count_sweep_gemini_browser"
    / "synthetic_only_random_review_count_sweep_summary.csv"
)

OFFICIAL_TARGETS = ["joy", "sadness", "fear", "trust", "surprise", "anticipation", "disgust"]
ALL_EMOTIONS = ["joy", "sadness", "anger", "fear", "trust", "surprise", "anticipation", "disgust"]
VAD_DIMS = ["valence", "arousal", "dominance"]

PAGE_W = 1500
MARGIN = 34
CONTENT_W = 1432
BLUE = "#1d4ed8"
NAVY = "#172554"
TEAL = "#0f766e"
AMBER = "#d97706"
RED = "#dc2626"
TEXT = "#111827"
MUTED = "#475569"
BORDER = "#cbd5e1"
SOFT_BG = "#f8fafc"


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_json(path: Path) -> dict[str, Any]:
    with require(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_csv(path: Path) -> list[dict[str, str]]:
    with require(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def fmt(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except Exception:
        return str(value)
    if math.isnan(number):
        return "NA"
    return f"{number:.{digits}f}"


def pct(numerator: int, denominator: int) -> str:
    return f"{100.0 * numerator / denominator:.1f}%"


def mean_sd(mean: Any, sd: Any) -> str:
    return f"Mean {fmt(mean)}\nSD {fmt(sd)}"


def wrap_text(value: str, width: int) -> list[str]:
    lines: list[str] = []
    for part in str(value).splitlines():
        lines.extend(textwrap.wrap(part, width=width, break_long_words=False, break_on_hyphens=False) or [""])
    return lines


def text_element(
    *,
    x: int,
    y: int,
    text: str,
    size: int = 13,
    weight: str = "400",
    color: str = "#111827",
    anchor: str = "start",
) -> str:
    return (
        f'<text x="{x}" y="{y}" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{color}" text-anchor="{anchor}">'
        f"{html.escape(text)}</text>"
    )


def metric_style(line: str, default_color: str = TEXT) -> tuple[int, str, str]:
    if line.startswith("Mean "):
        return 13, "700", TEXT
    if line.startswith("SD "):
        return 11, "400", "#64748b"
    if line == "Yes":
        return 13, "700", "#047857"
    if line.startswith("No"):
        return 13, "700", "#b45309"
    return 13, "400", default_color


def model_color(value: str) -> str:
    if value == "Logistic Regression":
        return BLUE
    if value == "XGBoost":
        return RED
    return TEXT


def render_kpi_cards(cards: list[tuple[str, str, str]], y_start: int) -> tuple[list[str], int]:
    gap = 18
    card_w = (CONTENT_W - gap * (len(cards) - 1)) // len(cards)
    card_h = 88
    colors = [BLUE, TEAL, AMBER, RED]
    parts: list[str] = []
    x = MARGIN
    for index, (label, value, detail) in enumerate(cards):
        color = colors[index % len(colors)]
        parts.append(
            f'<rect x="{x}" y="{y_start}" width="{card_w}" height="{card_h}" rx="10" '
            f'fill="#ffffff" stroke="{BORDER}" stroke-width="1.2"/>'
        )
        parts.append(f'<rect x="{x}" y="{y_start}" width="8" height="{card_h}" rx="4" fill="{color}"/>')
        parts.append(text_element(x=x + 24, y=y_start + 26, text=label.upper(), size=11, weight="700", color=color))
        parts.append(text_element(x=x + 24, y=y_start + 55, text=value, size=24, weight="700", color=TEXT))
        parts.append(text_element(x=x + 24, y=y_start + 76, text=detail, size=12, color=MUTED))
        x += card_w + gap
    return parts, y_start + card_h


def render_table(
    *,
    title: str,
    headers: list[str],
    rows: list[list[str]],
    widths: list[int],
    numeric_cols: set[int],
    y_start: int,
    x_start: int = 34,
    center_cols: set[int] | None = None,
    accent: str = BLUE,
) -> tuple[list[str], int]:
    center_cols = center_cols or set()
    pad_x = 10
    pad_y = 9
    line_h = 16
    table_w = sum(widths)
    x_positions = [x_start]
    for width in widths[:-1]:
        x_positions.append(x_positions[-1] + width)

    wrapped_rows: list[list[list[str]]] = []
    for row in [headers] + rows:
        wrapped_row: list[list[str]] = []
        for idx, cell in enumerate(row):
            chars = max(8, (widths[idx] - 2 * pad_x) // 7)
            wrapped_row.append(wrap_text(str(cell), chars))
        wrapped_rows.append(wrapped_row)

    row_heights = []
    for row_idx, row in enumerate(wrapped_rows):
        min_h = 38 if row_idx == 0 else 42
        row_heights.append(max(min_h, max(len(lines) for lines in row) * line_h + 2 * pad_y))

    table_y = y_start + 32
    total_h = sum(row_heights)
    parts = [
        f'<rect x="{x_start}" y="{y_start - 21}" width="{table_w}" height="36" rx="7" fill="#ffffff" '
        f'stroke="#d7e1ef" stroke-width="1"/>',
        f'<rect x="{x_start}" y="{y_start - 21}" width="8" height="36" rx="4" fill="{accent}"/>',
        text_element(x=x_start + 20, y=y_start + 2, text=title, size=18, weight="700", color=TEXT),
        f'<rect x="{x_start}" y="{table_y}" width="{table_w}" height="{total_h}" rx="8" '
        f'fill="#ffffff" stroke="{BORDER}" stroke-width="1.2"/>',
    ]
    y = table_y
    for row_idx, row in enumerate(wrapped_rows):
        row_h = row_heights[row_idx]
        raw_row = ([headers] + rows)[row_idx]
        first_value = raw_row[0] if raw_row else ""
        if row_idx == 0:
            fill = NAVY
        elif first_value == "Logistic Regression":
            fill = "#eff6ff" if row_idx % 2 else "#dbeafe"
        elif first_value == "XGBoost":
            fill = "#fff7ed" if row_idx % 2 else "#ffedd5"
        else:
            fill = "#ffffff" if row_idx % 2 else SOFT_BG
        parts.append(f'<rect x="{x_start}" y="{y}" width="{table_w}" height="{row_h}" fill="{fill}"/>')
        for col_idx, lines in enumerate(row):
            x = x_positions[col_idx]
            width = widths[col_idx]
            parts.append(
                f'<rect x="{x}" y="{y}" width="{width}" height="{row_h}" '
                f'fill="none" stroke="{BORDER}" stroke-width="0.8"/>'
            )
            is_header = row_idx == 0
            is_numeric = col_idx in numeric_cols
            is_centered = col_idx in center_cols
            if is_header or is_centered or is_numeric:
                anchor = "middle"
                text_x = x + width // 2
            else:
                anchor = "start"
                text_x = x + pad_x
            block_h = len(lines) * line_h
            text_y = y + (row_h - block_h) / 2 + 12
            base_color = "#ffffff" if is_header else (model_color(str(first_value)) if col_idx == 0 else TEXT)
            for line in lines:
                size, weight, line_color = metric_style(line, base_color)
                if is_header:
                    size, weight, line_color = 12, "700", "#ffffff"
                elif col_idx == 0 and first_value in {"Logistic Regression", "XGBoost"}:
                    weight = "700"
                parts.append(
                    text_element(
                        x=text_x,
                        y=text_y,
                        text=line,
                        size=size,
                        weight=weight,
                        color=line_color,
                        anchor=anchor,
                    )
                )
                text_y += line_h
        y += row_h

    return parts, y


def render_note(
    *,
    title: str,
    lines: list[str],
    y_start: int,
    x_start: int = 34,
    width: int = 1432,
    accent: str = BLUE,
) -> tuple[list[str], int]:
    pad = 16
    line_h = 17
    wrapped: list[str] = []
    for line in lines:
        wrapped.extend(wrap_text(line, max(40, (width - 2 * pad) // 7)))
    box_h = (len(wrapped) + 1) * line_h + 2 * pad
    parts = [
        f'<rect x="{x_start}" y="{y_start}" width="{width}" height="{box_h}" '
        f'rx="9" fill="#ffffff" stroke="#d6dee9" stroke-width="1"/>',
        f'<rect x="{x_start}" y="{y_start}" width="7" height="{box_h}" rx="4" fill="{accent}"/>',
    ]
    y = y_start + pad + 13
    parts.append(text_element(x=x_start + pad + 6, y=y, text=title, size=13, weight="700", color=accent))
    y += line_h
    for line in wrapped:
        parts.append(text_element(x=x_start + pad + 6, y=y, text=line, size=12, color=MUTED))
        y += line_h
    return parts, y_start + box_h


def label_distribution_rows(df: pd.DataFrame) -> list[list[str]]:
    rows: list[list[str]] = []
    total = len(df)
    for label in ALL_EMOTIONS:
        count = int(df[label].sum())
        used = "Yes" if label in OFFICIAL_TARGETS else "No; excluded"
        rows.append([label, str(count), pct(count, total), used])
    return rows


def vad_rows(df: pd.DataFrame) -> list[list[str]]:
    rows = []
    for dim in VAD_DIMS:
        rows.append(
            [
                dim,
                fmt(df[dim].mean(), 4),
                str(int(df[dim].min())),
                str(int(df[dim].max())),
            ]
        )
    return rows


def count_rows() -> list[list[str]]:
    pairs = [(198, 33), (402, 67), (600, 100), (1002, 167), (2004, 334), (4002, 667), (4704, 784)]
    return [[str(reviews), str(perfumes), f"{perfumes} perfumes x 6 reviews"] for reviews, perfumes in pairs]


def threshold05_metric_rows(summary_rows: list[dict[str, str]]) -> list[list[str]]:
    rows: list[list[str]] = []
    for row in summary_rows:
        rows.append(
            [
                row["model_family"],
                row["synthetic_reviews"],
                row["synthetic_perfumes"],
                mean_sd(row["macro_f1_mean"], row["macro_f1_std"]),
                mean_sd(row["macro_roc_auc_mean"], row["macro_roc_auc_std"]),
                mean_sd(row["macro_pr_auc_mean"], row["macro_pr_auc_std"]),
            ]
        )
    return rows


def threshold_sensitivity_rows(threshold_rows: list[dict[str, str]]) -> list[list[str]]:
    rows: list[list[str]] = []
    for row in threshold_rows:
        rows.append(
            [
                row["model_family"],
                fmt(row["threshold"], 2),
                mean_sd(row["macro_f1_mean"], row["macro_f1_std"]),
                fmt(row["macro_roc_auc_mean"]),
                fmt(row["macro_pr_auc_mean"]),
            ]
        )
    return rows


def artifact_rows() -> list[list[str]]:
    return [
        ["Focused data", "outputs/synthetic_only_random_review_count_sweep_gemini_browser_6reviews_threshold05/"],
        ["Threshold support", "outputs/synthetic_only_random_review_count_sweep_gemini_browser_6reviews_thresholds/"],
        ["Advisor page", "docs/synthetic_review_experiment_report/advisor_synthetic_only_784x6_threshold05.md"],
        ["This report", rel(OUT)],
    ]


def historical_context_rows(
    old_rows: list[dict[str, str]],
    threshold_sweep_rows: list[dict[str, str]],
    threshold05_rows: list[dict[str, str]],
) -> list[list[str]]:
    rows: list[list[str]] = []
    for model in ["Logistic Regression", "XGBoost"]:
        old = next(row for row in old_rows if row["model_family"] == model and int(row["synthetic_reviews"]) == 2352)
        current_03 = next(
            row
            for row in threshold_sweep_rows
            if row["model_family"] == model and int(row["synthetic_reviews"]) == 4704 and abs(float(row["threshold"]) - 0.3) < 1e-9
        )
        current_05 = next(
            row
            for row in threshold05_rows
            if row["model_family"] == model and int(row["synthetic_reviews"]) == 4704
        )
        rows.extend(
            [
                [
                    model,
                    "784x3 randomized pilot",
                    "0.3",
                    "2352",
                    fmt(old["macro_f1_mean"]),
                    fmt(old["macro_roc_auc_mean"]),
                    fmt(old["macro_pr_auc_mean"]),
                ],
                [
                    model,
                    "784x6 sensitivity",
                    "0.3",
                    "4704",
                    fmt(current_03["macro_f1_mean"]),
                    fmt(current_03["macro_roc_auc_mean"]),
                    fmt(current_03["macro_pr_auc_mean"]),
                ],
                [
                    model,
                    "784x6 reporting",
                    "0.5",
                    "4704",
                    fmt(current_05["macro_f1_mean"]),
                    fmt(current_05["macro_roc_auc_mean"]),
                    fmt(current_05["macro_pr_auc_mean"]),
                ],
            ]
        )
    return rows


def main() -> None:
    features = pd.read_csv(require(COMBINED_FEATURES))
    combined_report = load_json(COMBINED_REPORT)
    threshold05_report = load_json(THRESHOLD05_REPORT)
    threshold05_rows = load_csv(THRESHOLD05_SUMMARY)
    all_threshold_sweep_rows = load_csv(THRESHOLD_SWEEP_SUMMARY)
    threshold_sweep_rows = [
        row
        for row in all_threshold_sweep_rows
        if int(row["synthetic_reviews"]) == 4704
    ]
    threshold_sweep_rows.sort(key=lambda row: (row["model_family"], float(row["threshold"])))
    old_3review_rows = load_csv(OLD_3REVIEW_RANDOM_SUMMARY)

    validation = combined_report["validation"]
    sweep_validation = threshold05_report["validation"]

    data_rows = [
        ["Corpus", "Emotion-rich synthetic v2, 784 perfumes x 6 review-level rows"],
        ["Reviews 1-3", "Existing Gemini 3.1 Pro browser-labeled synthetic v2 corpus"],
        ["Reviews 4-6", "Separate add-on generated with the same v2 method; existing rows were not re-labeled"],
        ["Generation", "Same source identity/context keyword files, same seed 20260612, no emotion/VAD label conditioning"],
        ["Labeler", "Gemini 3.1 Pro browser labels; model-generated reference labels, not human annotation"],
        ["Boundary", "Synthetic-only analysis; no YouTube rows are mixed into training or evaluation"],
    ]
    validation_rows = [
        ["Combined rows", str(validation["rows"])],
        ["Unique perfumes", str(validation["unique_perfumes"])],
        ["Review indices", ", ".join(str(value) for value in validation["review_indices"])],
        ["Rows per perfume", ", ".join(str(value) for value in validation["per_perfume_counts"])],
        ["Unique review IDs", str(validation["unique_synthetic_review_ids"])],
        ["Missing label/VAD values", str(validation["missing_label_values"])],
        ["Threshold-0.5 run rows", str(sweep_validation["runs_rows"])],
        ["Threshold-0.5 summary rows", str(sweep_validation["summary_rows"])],
        ["Threshold-0.5 per-label rows", str(sweep_validation["per_label_rows"])],
    ]
    method_rows = [
        ["Models", "Logistic Regression; XGBoost"],
        ["Features", "NRC, VAD, Hartmann, Savani, GoEmotions, Cardiff"],
        ["Split", "GroupKFold, 5 folds, grouped by perfume_id"],
        ["Randomness", "Perfume-order seeds 101, 202, 303, 404, 505; model seed 42"],
        ["Threshold", "0.5 for final reporting; sensitivity checked at 0.25, 0.3, 0.5, 0.7"],
        ["Metrics", "Macro F1, ROC-AUC, PR-AUC; official targets exclude anger"],
    ]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{PAGE_W}" height="2400" viewBox="0 0 {PAGE_W} 2400">',
        '<rect width="100%" height="100%" fill="#f4f7fb"/>',
        f'<rect x="{MARGIN}" y="24" width="{CONTENT_W}" height="126" rx="14" fill="{NAVY}"/>',
        f'<rect x="{MARGIN}" y="128" width="{CONTENT_W}" height="28" rx="12" fill="#dbeafe"/>',
        text_element(
            x=PAGE_W // 2,
            y=66,
            text="Advisor Compact Report: Synthetic-Only 784x6 Review Count Sweep",
            size=27,
            weight="700",
            color="#ffffff",
            anchor="middle",
        ),
        text_element(
            x=PAGE_W // 2,
            y=96,
            text="Final reporting threshold: 0.5 | Synthetic-only, grouped by perfume_id | Advisor-response / supplementary analysis",
            size=14,
            color="#bfdbfe",
            anchor="middle",
        ),
        text_element(
            x=PAGE_W // 2,
            y=146,
            text="784 perfumes | 4,704 labeled synthetic reviews | complete six-review perfume blocks",
            size=12,
            weight="700",
            color=NAVY,
            anchor="middle",
        ),
    ]
    y = 182

    kpi_parts, y = render_kpi_cards(
        [
            ("Rows", "4,704", "labeled synthetic reviews"),
            ("Perfumes", "784", "unique perfume identities"),
            ("Reviews", "6 each", "indices 1 through 6"),
            ("Threshold", "0.5", "final reporting setting"),
        ],
        y_start=y,
    )
    parts.extend(kpi_parts)
    y += 26

    note_parts, y = render_note(
        title="Executive summary",
        lines=[
            "This report gathers the completed 784x6 synthetic-only experiment. It answers whether increasing the number of synthetic reviews changes model performance when using only synthetic rows. It should not be presented as the main thesis benchmark because there are no YouTube rows in this sweep.",
            "Perfumes were shuffled first, then complete six-review perfume blocks were selected for each count. Threshold sensitivity was checked; threshold 0.5 is used here as the single reporting threshold.",
        ],
        y_start=y,
        accent=TEAL,
    )
    parts.extend(note_parts)
    y += 24

    table_parts, y = render_table(
        title="1. Data Provenance",
        headers=["Item", "Detail"],
        rows=data_rows,
        widths=[210, 1222],
        numeric_cols=set(),
        y_start=y,
        center_cols={0},
        accent=BLUE,
    )
    parts.extend(table_parts)
    y += 24

    left_parts, left_y = render_table(
        title="2. Validation",
        headers=["Check", "Value"],
        rows=validation_rows,
        widths=[280, 380],
        numeric_cols={1},
        y_start=y,
        x_start=34,
        center_cols={0, 1},
        accent=TEAL,
    )
    right_parts, right_y = render_table(
        title="3. Evaluation Method",
        headers=["Item", "Detail"],
        rows=method_rows,
        widths=[210, 540],
        numeric_cols=set(),
        y_start=y,
        x_start=716,
        center_cols={0},
        accent=AMBER,
    )
    parts.extend(left_parts)
    parts.extend(right_parts)
    y = max(left_y, right_y) + 24

    table_parts, y = render_table(
        title="4. Review Counts Tested",
        headers=["Synthetic reviews", "Perfumes", "Block meaning"],
        rows=count_rows(),
        widths=[240, 180, 1012],
        numeric_cols={0, 1},
        y_start=y,
        center_cols={0, 1, 2},
        accent=BLUE,
    )
    parts.extend(table_parts)
    y += 24

    left_parts, left_y = render_table(
        title="5. Combined 784x6 Emotion Labels",
        headers=["Label", "Positives", "Rate", "Metric target"],
        rows=label_distribution_rows(features),
        widths=[160, 130, 110, 260],
        numeric_cols={1, 2},
        y_start=y,
        x_start=34,
        center_cols={0, 1, 2, 3},
        accent=TEAL,
    )
    right_parts, right_y = render_table(
        title="6. Combined VAD Labels",
        headers=["Dimension", "Mean", "Min", "Max"],
        rows=vad_rows(features),
        widths=[210, 170, 170, 170],
        numeric_cols={1, 2, 3},
        y_start=y,
        x_start=746,
        center_cols={0, 1, 2, 3},
        accent=AMBER,
    )
    parts.extend(left_parts)
    parts.extend(right_parts)
    y = max(left_y, right_y) + 24

    table_parts, y = render_table(
        title="7. Main Threshold-0.5 Metrics, Mean and SD Across 5 Random Perfume Orders",
        headers=["Model", "Reviews", "Perfumes", "Macro F1", "ROC-AUC", "PR-AUC"],
        rows=threshold05_metric_rows(threshold05_rows),
        widths=[210, 130, 130, 260, 260, 260],
        numeric_cols={1, 2, 3, 4, 5},
        y_start=y,
        center_cols={0, 1, 2, 3, 4, 5},
        accent=RED,
    )
    parts.extend(table_parts)
    y += 24

    table_parts, y = render_table(
        title="8. Full-Corpus Threshold Sensitivity At 4,704 Reviews",
        headers=["Model", "Threshold", "Macro F1", "ROC-AUC", "PR-AUC"],
        rows=threshold_sensitivity_rows(threshold_sweep_rows),
        widths=[260, 170, 310, 240, 240],
        numeric_cols={1, 2, 3, 4},
        y_start=y,
        center_cols={0, 1, 2, 3, 4},
        accent=BLUE,
    )
    parts.extend(table_parts)
    y += 24

    note_parts, y = render_note(
        title="Interpretation",
        lines=[
            "At threshold 0.5, Logistic Regression has its best mean Macro F1 at 2,004 reviews (0.5795), while 4,704 reviews is essentially tied (0.5787) and has the best ROC-AUC and PR-AUC. XGBoost has its best mean Macro F1 at the full 4,704-review condition (0.5953).",
            "Higher review counts have very small SD, so they are more stable across random perfume-order seeds. The curve is not perfectly monotonic, so the careful claim is stability and generally stronger ranking metrics, not automatic F1 improvement at every added count.",
            "Threshold 0.5 was selected after sensitivity checks. A finer LR-only check found 0.49 was marginally highest on the full corpus, but it was effectively tied with 0.5; 0.5 is cleaner for reporting. ROC-AUC and PR-AUC do not change with threshold because they use ranking scores.",
            "Recommended advisor wording: This is synthetic-only advisor-response evidence about the synthetic corpus, not the main thesis benchmark.",
        ],
        y_start=y,
        accent=TEAL,
    )
    parts.extend(note_parts)
    y += 24

    table_parts, y = render_table(
        title="9. Historical Context",
        headers=["Model", "Run", "Threshold", "Reviews", "Macro F1", "ROC-AUC", "PR-AUC"],
        rows=historical_context_rows(old_3review_rows, all_threshold_sweep_rows, threshold05_rows),
        widths=[185, 250, 120, 130, 180, 180, 180],
        numeric_cols={2, 3, 4, 5, 6},
        y_start=y,
        center_cols={0, 1, 2, 3, 4, 5, 6},
        accent=AMBER,
    )
    parts.extend(table_parts)
    y += 24

    note_parts, y = render_note(
        title="Context note",
        lines=[
            "The 784x3 pilot and 784x6 reporting run are not a direct replacement comparison. The 784x6 corpus adds review indices 4-6 with additional labeled text diversity, and the final reporting threshold is 0.5 rather than 0.3. Lower F1 in 784x6 should be read as a harder and broader synthetic corpus, not as failed generation.",
        ],
        y_start=y,
        accent=AMBER,
    )
    parts.extend(note_parts)
    y += 24

    table_parts, y = render_table(
        title="10. Main Artifacts",
        headers=["Artifact", "Path"],
        rows=artifact_rows(),
        widths=[220, 1212],
        numeric_cols=set(),
        y_start=y,
        center_cols={0},
        accent=BLUE,
    )
    parts.extend(table_parts)
    y += 36

    parts.append(
        f'<rect x="{MARGIN}" y="{y - 18}" width="{CONTENT_W}" height="38" rx="9" fill="#ecfdf5" '
        'stroke="#bbf7d0" stroke-width="1"/>'
    )
    parts.append(
        text_element(
            x=PAGE_W // 2,
            y=y + 6,
            text="Status: committed outputs exist for labels, 784x6 combined features, threshold sensitivity, focused threshold-0.5 sweep, and advisor-facing report images.",
            size=12,
            weight="700",
            color="#047857",
            anchor="middle",
        )
    )
    y += 30

    parts[0] = f'<svg xmlns="http://www.w3.org/2000/svg" width="1500" height="{y}" viewBox="0 0 1500 {y}">'
    parts.append("</svg>\n")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(parts), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()

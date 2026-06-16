#!/usr/bin/env python3
"""Build a compact SVG report for recent YouTube/synthetic augmentation results."""

from __future__ import annotations

import csv
import html
import json
import math
import textwrap
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/synthetic_review_experiment_report/table_images/01_recent_synthetic_og_summary.svg"

OG_FEATURES = ROOT / "outputs/gemini_browser_og784/analysis_out/gemini_browser_og784_feature_labels.csv"
SYNTH_FEATURES = (
    ROOT
    / "outputs/synthetic_review_level_tests/emotion_rich_v2_3reviews/supervised_eval_gemini_browser"
    / "review_level_features_labeled_g25pro.csv"
)
LR_COMPARISON = ROOT / "outputs/review_level_synthetic_augmentation_gemini_browser_og784/metric_comparison.csv"
XGB_COMPARISON = ROOT / "outputs/review_level_synthetic_augmentation_gemini_browser_og784_xgboost/metric_comparison.csv"
LABEL_DISTRIBUTION = ROOT / "outputs/review_level_synthetic_augmentation_gemini_browser_og784/label_distribution.csv"
G25_API_COMPARISON = (
    ROOT
    / "outputs/gemini_judge/judge_g25pro_v1/model_benchmark"
    / "tables/final_model_comparison_for_thesis.csv"
)

TARGETS = ["joy", "sadness", "fear", "trust", "surprise", "anticipation", "disgust"]
TITLE_TO_CONTENT_GAP = 28
RELATED_GAP = 18
SECTION_GAP = 34


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_csv(path: Path) -> list[dict[str, str]]:
    with require(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_json(path: Path) -> dict[str, Any]:
    with require(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def fmt(value: Any, digits: int = 4) -> str:
    try:
        value_float = float(value)
    except Exception:
        return "NA"
    if math.isnan(value_float):
        return "NA"
    return f"{value_float:.{digits}f}"


def pct(value: Any) -> str:
    try:
        value_float = float(value)
    except Exception:
        return "NA"
    if math.isnan(value_float):
        return "NA"
    return f"{100.0 * value_float:.1f}%"


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def unique_count(rows: list[dict[str, str]], column: str) -> int:
    return len({row[column] for row in rows if row.get(column)})


def row_by_experiment(rows: list[dict[str, str]], experiment: str) -> dict[str, str]:
    for row in rows:
        if row["experiment"] == experiment:
            return row
    raise KeyError(experiment)


def row_by_model(rows: list[dict[str, str]], model_id: str) -> dict[str, str]:
    for row in rows:
        if row["model_id"] == model_id:
            return row
    raise KeyError(model_id)


def label_distribution_lookup(rows: list[dict[str, str]], source: str, label: str) -> dict[str, str]:
    for row in rows:
        if row["source"] == source and row["label"] == label:
            return row
    raise KeyError(f"{source}:{label}")


def wrap_text(text: str, width: int) -> list[str]:
    lines: list[str] = []
    for part in str(text).splitlines():
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
) -> tuple[list[str], int]:
    center_cols = center_cols or set()
    pad_x = 9
    pad_y = 7
    line_h = 16
    table_w = sum(widths)
    x_positions = [x_start]
    for width in widths[:-1]:
        x_positions.append(x_positions[-1] + width)

    wrapped_rows: list[list[list[str]]] = []
    for row in [headers] + rows:
        wrapped_row: list[list[str]] = []
        for idx, cell in enumerate(row):
            chars = max(10, (widths[idx] - 2 * pad_x) // 7)
            wrapped_row.append(wrap_text(cell, chars))
        wrapped_rows.append(wrapped_row)

    row_heights = [
        max(34, max(len(lines) for lines in row) * line_h + 2 * pad_y)
        for row in wrapped_rows
    ]

    parts = [text_element(x=x_start, y=y_start, text=title, size=18, weight="700")]
    y = y_start + TITLE_TO_CONTENT_GAP
    for row_idx, row in enumerate(wrapped_rows):
        row_h = row_heights[row_idx]
        fill = "#e8eef8" if row_idx == 0 else ("#ffffff" if row_idx % 2 else "#f8fafc")
        parts.append(f'<rect x="{x_start}" y="{y}" width="{table_w}" height="{row_h}" fill="{fill}"/>')
        for col_idx, lines in enumerate(row):
            x = x_positions[col_idx]
            width = widths[col_idx]
            parts.append(
                f'<rect x="{x}" y="{y}" width="{width}" height="{row_h}" '
                'fill="none" stroke="#cbd5e1" stroke-width="1"/>'
            )
            is_header = row_idx == 0
            is_numeric = col_idx in numeric_cols
            is_centered = col_idx in center_cols
            if is_centered:
                anchor = "middle"
                text_x = x + width // 2
            elif is_numeric:
                anchor = "end"
                text_x = x + width - pad_x
            else:
                anchor = "start"
                text_x = x + pad_x
            text_y = y + pad_y + 13
            for line in lines:
                parts.append(
                    text_element(
                        x=text_x,
                        y=text_y,
                        text=line,
                        size=13,
                        weight="700" if is_header else "400",
                        anchor=anchor,
                    )
                )
                text_y += line_h
        y += row_h

    return parts, y


def render_note(
    *,
    title: str | None,
    lines: list[str],
    y_start: int,
    x_start: int = 34,
    width: int = 1200,
) -> tuple[list[str], int]:
    pad = 12
    line_h = 16
    wrapped: list[str] = []
    for line in lines:
        wrapped.extend(wrap_text(line, max(40, (width - 2 * pad) // 7)))
    box_h = (len(wrapped) + (1 if title else 0)) * line_h + 2 * pad
    parts = [
        f'<rect x="{x_start}" y="{y_start}" width="{width}" height="{box_h}" '
        'rx="4" fill="#f8fafc" stroke="#d6dee9" stroke-width="1"/>'
    ]
    y = y_start + pad + 13
    if title:
        parts.append(text_element(x=x_start + pad, y=y, text=title, size=12, weight="700", color="#374151"))
        y += line_h
    for line in wrapped:
        parts.append(text_element(x=x_start + pad, y=y, text=line, size=12, color="#374151"))
        y += line_h
    return parts, y_start + box_h


def metric_rows(rows: list[dict[str, str]]) -> list[list[str]]:
    labels = {
        "og_only_reference": "YouTube Reviews only",
        "synthetic_only_review_level": "Synthetic-only sanity check",
        "og_plus_synthetic_all_rows_oof": "YouTube+synthetic all rows cross-validation (out-of-fold predictions)",
        "og_plus_synthetic_eval_og_rows": "YouTube-eval augmentation check",
    }
    notes = {
        "og_only_reference": "784 perfume-level YouTube documents.",
        "synthetic_only_review_level": "2352 review-level rows; useful but not thesis-final.",
        "og_plus_synthetic_all_rows_oof": "3136 mixed rows; synthetic rows dominate 3:1.",
        "og_plus_synthetic_eval_og_rows": "Train with mixed rows, evaluate only held-out YouTube perfume rows.",
    }
    row_counts = {
        "og_only_reference": "784 YouTube rows",
        "synthetic_only_review_level": "2352 synthetic rows",
        "og_plus_synthetic_all_rows_oof": "3136 mixed rows",
        "og_plus_synthetic_eval_og_rows": "3136 train; 784 test",
    }
    output: list[list[str]] = []
    for experiment in labels:
        row = row_by_experiment(rows, experiment)
        output.append(
            [
                labels[experiment],
                row_counts[experiment],
                fmt(row["macro_f1"]),
                fmt(row["macro_roc_auc"]),
                fmt(row["macro_pr_auc"]),
                notes[experiment],
            ]
        )
    return output


def main() -> None:
    og_rows = load_csv(OG_FEATURES)
    synth_rows = load_csv(SYNTH_FEATURES)
    lr_rows = load_csv(LR_COMPARISON)
    xgb_rows = load_csv(XGB_COMPARISON)
    label_rows = load_csv(LABEL_DISTRIBUTION)
    g25_rows = load_csv(G25_API_COMPARISON)

    og_perfumes = unique_count(og_rows, "perfume_id")
    synth_perfumes = unique_count(synth_rows, "perfume_id")
    mixed_rows = len(og_rows) + len(synth_rows)
    mixed_perfumes = len({row["perfume_id"] for row in og_rows} | {row["perfume_id"] for row in synth_rows})

    lr_og = row_by_experiment(lr_rows, "og_only_reference")
    lr_aug = row_by_experiment(lr_rows, "og_plus_synthetic_eval_og_rows")
    xgb_og = row_by_experiment(xgb_rows, "og_only_reference")
    xgb_aug = row_by_experiment(xgb_rows, "og_plus_synthetic_eval_og_rows")
    g25_lr = row_by_model(g25_rows, "logistic_regression")
    g25_xgb = row_by_model(g25_rows, "xgboost")
    lr_delta = float(lr_aug["macro_f1"]) - float(lr_og["macro_f1"])
    xgb_delta = float(xgb_aug["macro_f1"]) - float(xgb_og["macro_f1"])

    label_cells: list[list[str]] = []
    for label in TARGETS:
        og_label = label_distribution_lookup(label_rows, "og_784_perfume_level", label)
        synth_label = label_distribution_lookup(label_rows, "synthetic_v2_review_level", label)
        label_cells.append(
            [
                label,
                f"{og_label['positive_count']} / {og_label['rows']} ({pct(og_label['positive_rate'])})",
                f"{synth_label['positive_count']} / {synth_label['rows']} ({pct(synth_label['positive_rate'])})",
            ]
        )
    label_table: list[list[str]] = []
    for idx in range(0, len(label_cells), 2):
        left = label_cells[idx]
        right = label_cells[idx + 1] if idx + 1 < len(label_cells) else ["", "", ""]
        label_table.append([*left, *right])

    api_comparison_rows = [
        [
            "Logistic Regression",
            fmt(g25_lr["macro_f1"]),
            fmt(lr_og["macro_f1"]),
            f"{float(lr_og['macro_f1']) - float(g25_lr['macro_f1']):+.4f}",
            fmt(g25_lr["macro_pr_auc"]),
            fmt(lr_og["macro_pr_auc"]),
            "Same 784 perfume-level YouTube Reviews benchmark; labels differ by labeler.",
        ],
        [
            "XGBoost",
            fmt(g25_xgb["macro_f1"]),
            fmt(xgb_og["macro_f1"]),
            f"{float(xgb_og['macro_f1']) - float(g25_xgb['macro_f1']):+.4f}",
            fmt(g25_xgb["macro_pr_auc"]),
            fmt(xgb_og["macro_pr_auc"]),
            "XGBoost remains the stronger classical benchmark on 3.1 browser labels.",
        ],
    ]

    takeaway_rows = [
        [
            "Logistic Regression",
            "Main thesis model",
            fmt(lr_og["macro_f1"]),
            fmt(lr_aug["macro_f1"]),
            f"{lr_delta:+.4f}",
            "Small positive signal from synthetic augmentation.",
        ],
        [
            "XGBoost",
            "Secondary benchmark",
            fmt(xgb_og["macro_f1"]),
            fmt(xgb_aug["macro_f1"]),
            f"{xgb_delta:+.4f}",
            "Macro-F1 drops slightly after augmentation.",
        ],
    ]

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="900" viewBox="0 0 1280 900">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]
    y = 42
    parts.append(text_element(x=34, y=y, text="Recent Synthetic + YouTube Reviews Gemini 3.1 Pro (Browser) Results", size=24, weight="700"))
    y += 24
    parts.append(
        text_element(
            x=34,
            y=y,
            text="Gemini 3.1 Pro (browser) labels, official 7 emotion targets, NRC + VAD + 4 transformer feature groups; Gemini 2.5 Pro (API) shown where comparable.",
            size=13,
            color="#374151",
        )
    )
    y += 24

    dataset_table = [
        [
            "YouTube Reviews",
            str(og_perfumes),
            str(len(og_rows)),
            "Perfume-level aggregated documents",
            "One row per perfume after aggregating YouTube review evidence.",
        ],
        [
            "Mixed experiment",
            str(mixed_perfumes),
            str(mixed_rows),
            "Mixed perfume/review rows",
            "Folds stay grouped by perfume_id to avoid train/test leakage.",
        ],
        [
            "Synthetic v2",
            str(synth_perfumes),
            str(len(synth_rows)),
            "Review-level synthetic documents",
            "784 perfumes x 3 generated reviews = 2352 rows.",
        ],
    ]
    table_parts, y = render_table(
        title="1. Dataset Structure",
        headers=["Source", "Unique perfumes", "Rows / documents", "Unit in model", "Meaning"],
        rows=dataset_table,
        widths=[170, 125, 135, 260, 500],
        numeric_cols={1, 2},
        y_start=y,
    )
    parts.extend(table_parts)

    y += SECTION_GAP
    table_parts, y = render_table(
        title="2. Label Imbalance, YouTube Reviews vs Synthetic",
        headers=["Label", "YouTube positives", "Synthetic positives", "Label", "YouTube positives", "Synthetic positives"],
        rows=label_table,
        widths=[120, 220, 250, 120, 220, 250],
        numeric_cols=set(),
        y_start=y,
    )
    parts.extend(table_parts)

    y += SECTION_GAP
    table_parts, y = render_table(
        title="3. YouTube Reviews 784 Labeler Comparison, Where Applicable",
        headers=["Model", "2.5 Pro API F1", "3.1 Pro browser F1", "F1 change", "2.5 API PR", "3.1 browser PR", "Scope"],
        rows=api_comparison_rows,
        widths=[185, 120, 145, 100, 105, 125, 410],
        numeric_cols={1, 2, 3, 4, 5},
        y_start=y,
    )
    parts.extend(table_parts)

    y += RELATED_GAP
    note_parts, y = render_note(
        title=None,
        lines=[
            "The Gemini 2.5 Pro (API) comparison is YouTube Reviews only. It is not shown for the 2352 synthetic review rows because the full synthetic v2 set was labeled through Gemini 3.1 Pro (browser), not the 2.5 API."
        ],
        y_start=y,
    )
    parts.extend(note_parts)

    y += SECTION_GAP
    table_parts, y = render_table(
        title="4. Logistic Regression: Main Thesis Pipeline",
        headers=["Experiment", "Rows used", "Macro-F1", "ROC-AUC", "PR-AUC", "Interpretation"],
        rows=metric_rows(lr_rows),
        widths=[270, 170, 95, 95, 95, 465],
        numeric_cols={2, 3, 4},
        y_start=y,
    )
    parts.extend(table_parts)

    y += SECTION_GAP
    table_parts, y = render_table(
        title="5. XGBoost: Secondary Classical Benchmark",
        headers=["Experiment", "Rows used", "Macro-F1", "ROC-AUC", "PR-AUC", "Interpretation"],
        rows=metric_rows(xgb_rows),
        widths=[270, 170, 95, 95, 95, 465],
        numeric_cols={2, 3, 4},
        y_start=y,
    )
    parts.extend(table_parts)

    y += SECTION_GAP
    table_parts, y = render_table(
        title="6. Thesis Takeaway Metrics",
        headers=["Model", "Role", "YouTube-only F1", "YouTube+synthetic F1", "Change", "Reading"],
        rows=takeaway_rows,
        widths=[190, 165, 130, 155, 100, 450],
        numeric_cols={2, 3, 4},
        y_start=y,
    )
    parts.extend(table_parts)

    y += SECTION_GAP
    table_parts, y = render_table(
        title="7. Synthetic Generation Method",
        headers=["Part", "v1 synthetic", "v2 synthetic"],
        rows=[
            [
                "Main aim",
                "Describe the perfume safely.",
                "Write a human reaction to the perfume.",
            ],
            [
                "Inputs",
                "Perfume identity, persona, tone.",
                "Perfume identity, note-inspired keywords, human scenarios, Fragrantica phrasing.",
            ],
            [
                "Writing style",
                "Balanced and generic.",
                "Memories, places, body reactions, mixed opinions, sensory images.",
            ],
            [
                "Emotion labels",
                "Not used during generation.",
                "Not used during generation.",
            ],
            [
                "Why labels changed",
                "Too little emotional evidence, so labels looked neutral.",
                "More subjective evidence, so labels became more varied.",
            ],
        ],
        widths=[170, 430, 590],
        numeric_cols=set(),
        y_start=y,
    )
    parts.extend(table_parts)

    img_h = y + SECTION_GAP
    parts[0] = f'<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="{img_h}" viewBox="0 0 1280 {img_h}">'
    parts.append("</svg>\n")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(parts), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()

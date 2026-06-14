#!/usr/bin/env python3
"""Build the compact Flash-vs-Pro and benchmark SVG result page."""

from __future__ import annotations

import csv
import html
import json
import textwrap
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/flash_vs_pro_result_tables/table_images/20_compact_results_summary.svg"


def load_json(path: str) -> dict[str, Any]:
    with (ROOT / path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_csv(path: str) -> list[dict[str, str]]:
    with (ROOT / path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def fmt(value: Any, digits: int = 6) -> str:
    return f"{float(value):.{digits}f}"


def wrap_text(text: str, width: int) -> list[str]:
    lines: list[str] = []
    for part in str(text).splitlines():
        lines.extend(textwrap.wrap(part, width=width, break_long_words=False) or [""])
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
) -> tuple[list[str], int]:
    pad_x = 10
    pad_y = 9
    line_h = 17
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
        max(38, max(len(lines) for lines in row) * line_h + 2 * pad_y)
        for row in wrapped_rows
    ]

    parts = [text_element(x=x_start, y=y_start, text=title, size=18, weight="700")]
    y = y_start + 16
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
            is_numeric = col_idx in numeric_cols and not is_header
            anchor = "end" if is_numeric else "start"
            text_x = x + width - pad_x if is_numeric else x + pad_x
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

    return parts, y + 14


def render_note(
    *,
    title: str | None,
    lines: list[str],
    y_start: int,
    x_start: int = 34,
    width: int = 1200,
) -> tuple[list[str], int]:
    pad = 12
    line_h = 17
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
    return parts, y_start + box_h + 18


def main() -> None:
    decision = load_json("outputs/gemini_judge/judge_g25pro_v1/decision/branch_decision_summary.json")
    dataset_summary = load_json(
        "outputs/gemini_judge/judge_g25pro_v1/model_benchmark/datasets/perfume_text_dataset_v1_summary.json"
    )
    benchmark_rows = load_csv(
        "outputs/gemini_judge/judge_g25pro_v1/model_benchmark/tables/final_model_comparison_for_thesis.csv"
    )
    benchmark_comparison_rows = load_csv(
        "outputs/gemini_judge/judge_g25pro_v1/model_benchmark/tables/comparison_table.csv"
    )
    branch_comparison_rows = load_csv(
        "outputs/gemini_judge/judge_g25pro_v1/decision/benchmark_side_by_side.csv"
    )
    flash_threshold_sweep_rows = load_csv(
        "outputs/model_benchmark_threshold_sweep/tables/final_model_threshold_comparison_metric_grouped_03_05_07.csv"
    )

    benchmark_n = int(dataset_summary["n_perfumes"])
    chunk_n = int(dataset_summary["n_total_chunks"])
    flash_bench = float(decision["baseline_benchmark_mean_macro_f1"])
    pro_bench = float(decision["judge_benchmark_mean_macro_f1"])
    pro_thresholds = sorted({row["threshold"] for row in benchmark_comparison_rows})
    pro_threshold_text = ", ".join(pro_thresholds)
    lr_branch_row = next(row for row in branch_comparison_rows if row["model_id"] == "logistic_regression")
    lr_flash_threshold_row = next(row for row in flash_threshold_sweep_rows if row["Model"] == "Logistic Regression")
    lr_flash_fixed = float(lr_branch_row["baseline_macro_f1"])
    lr_pro_fixed = float(lr_branch_row["judge_macro_f1"])
    lr_flash_best = float(lr_flash_threshold_row["0.5 Macro-F1"])

    model_explanations = {
        "xgboost": "Best macro-F1.",
        "logistic_regression": "Almost tied with XGBoost in macro-F1; best PR-AUC.",
        "bilstm_attention": "Lower macro-F1 than the top classical models.",
        "bert": "Lower macro-F1 and PR-AUC than the top classical models.",
        "random_forest": "Lower macro-F1, but ROC-AUC and PR-AUC are still competitive.",
    }
    model_order = ["xgboost", "logistic_regression", "bilstm_attention", "bert", "random_forest"]
    rows_by_id = {row["model_id"]: row for row in benchmark_rows}

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="900" viewBox="0 0 1280 900">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]
    y = 42

    first_rows = [
        [
            "Mean Macro-F1 across 5 model rows",
            f"n = {benchmark_n}; avg over 5 models",
            fmt(flash_bench),
            fmt(pro_bench),
            f"+{fmt(pro_bench - flash_bench)}",
            "Each value is the average of the five model Macro-F1 rows at threshold 0.3.",
        ],
    ]
    table_parts, y = render_table(
        title="1. Gemini Flash vs Gemini 2.5 Pro, fixed threshold 0.3",
        headers=["Metric", "Dataset / n", "Gemini Flash", "Gemini 2.5 Pro", "Change", "Explanation"],
        rows=first_rows,
        widths=[190, 245, 125, 135, 110, 390],
        numeric_cols={2, 3, 4},
        y_start=y,
    )
    parts.extend(table_parts)

    y += 14
    lr_rows = [
        [
            "Logistic Regression Macro-F1, fixed threshold 0.3",
            fmt(lr_flash_fixed),
            fmt(lr_pro_fixed),
            f"+{fmt(lr_pro_fixed - lr_flash_fixed)}",
            "Direct same-threshold comparison for the main classical pipeline.",
        ],
    ]
    table_parts, y = render_table(
        title="2. Logistic Regression branch comparison, main classical pipeline",
        headers=["Metric", "Gemini Flash", "Gemini 2.5 Pro", "Change", "Note"],
        rows=lr_rows,
        widths=[380, 135, 150, 115, 410],
        numeric_cols={1, 2, 3},
        y_start=y,
    )
    parts.extend(table_parts)
    lr_note_parts, y = render_note(
        title=None,
        lines=[
            f"Flash's best Logistic Regression threshold in the sweep was 0.5, with Macro-F1 = {fmt(lr_flash_best)}. Compared with Flash's best Logistic Regression sweep value, Pro is still higher by +{fmt(lr_pro_fixed - lr_flash_best)}."
        ],
        y_start=y,
    )
    parts.extend(lr_note_parts)

    y += 14
    parts.append(text_element(x=34, y=y, text="3. Gemini 2.5 Pro final benchmark, fixed threshold 0.3", size=18, weight="700"))
    y += 18
    threshold_parts, y = render_note(
        title=None,
        lines=[
            f"Dataset: Gemini 2.5 Pro-labeled frozen set, n = {benchmark_n} perfumes. Threshold: fixed {pro_threshold_text}. The mean in Section 1 is the average of these five Macro-F1 rows."
        ],
        y_start=y,
    )
    parts.extend(threshold_parts)

    model_rows = []
    for model_id in model_order:
        row = rows_by_id[model_id]
        model_rows.append(
            [
                row["model_name"],
                fmt(row["macro_f1"]),
                fmt(row["macro_roc_auc"]),
                fmt(row["macro_pr_auc"]),
                model_explanations[model_id],
            ]
        )
    table_parts, y = render_table(
        title="",
        headers=["Model", "Macro-F1", "Macro ROC-AUC", "Macro PR-AUC", "Explanation"],
        rows=model_rows,
        widths=[250, 120, 150, 150, 525],
        numeric_cols={1, 2, 3},
        y_start=y - 4,
    )
    # Drop empty section-title text from the anonymous table.
    parts.extend(table_parts[1:])
    y -= 2

    flash_threshold_rows = [
        ["Logistic Regression", "0.269933", "0.271200", "0.206337"],
        ["Random Forest", "0.219452", "0.127249", "0.059406"],
        ["XGBoost", "0.254726", "0.230065", "0.147491"],
        ["BERT", "0.218461", "0.167550", "0.003123"],
        ["BiLSTM + Attention", "0.232358", "0.216145", "0.000000"],
    ]
    y += 14
    table_parts, y = render_table(
        title="4. Gemini Flash threshold sweep, same 5 models",
        headers=["Model", "0.3 Macro-F1", "0.5 Macro-F1", "0.7 Macro-F1"],
        rows=flash_threshold_rows,
        widths=[360, 170, 170, 170],
        numeric_cols={1, 2, 3},
        y_start=y,
    )
    parts.extend(table_parts)
    interpretation_parts, y = render_note(
        title=None,
        lines=[
            "The Section 1 Flash value uses the 0.3 column averaged over these five rows. Logistic Regression performs best for Gemini Flash at threshold 0.5. XGBoost performs best at threshold 0.3 among the tree models. Threshold 0.7 strongly reduces macro-F1, especially for BERT and BiLSTM."
        ],
        y_start=y,
    )
    parts.extend(interpretation_parts)

    img_h = y + 6
    parts[0] = f'<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="{img_h}" viewBox="0 0 1280 {img_h}">'
    parts.append("</svg>\n")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(parts), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()

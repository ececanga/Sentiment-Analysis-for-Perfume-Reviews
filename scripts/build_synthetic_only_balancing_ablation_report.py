#!/usr/bin/env python3
"""Build a compact report image for the synthetic-only balancing ablation."""

from __future__ import annotations

import html
import math
import textwrap
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/synthetic_review_experiment_report/table_images/06_synthetic_only_balancing_ablation_report.svg"
ABLATION_ROOT = ROOT / "outputs/synthetic_only_balancing_ablation_gemini_browser_6reviews_threshold05"
SUMMARY = ABLATION_ROOT / "synthetic_only_balancing_ablation_summary.csv"
PER_LABEL = ABLATION_ROOT / "synthetic_only_balancing_ablation_per_label.csv"
LABEL_DISTRIBUTION = ABLATION_ROOT / "synthetic_only_balancing_ablation_label_distribution.csv"

PAGE_W = 1500
MARGIN = 34
CONTENT_W = 1432
NAVY = "#172554"
BLUE = "#2563eb"
TEAL = "#0f766e"
AMBER = "#d97706"
RED = "#dc2626"
TEXT = "#111827"
MUTED = "#475569"
BORDER = "#cbd5e1"


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


def pct(value: Any) -> str:
    return f"{100.0 * float(value):.1f}%"


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
        f"{html.escape(text)}</text>"
    )


def wrap_text(value: str, width: int) -> list[str]:
    lines: list[str] = []
    for part in str(value).splitlines():
        lines.extend(textwrap.wrap(part, width=width, break_long_words=False, break_on_hyphens=False) or [""])
    return lines


def model_color(value: str) -> str:
    if value == "Logistic Regression":
        return BLUE
    if value == "XGBoost":
        return RED
    return TEXT


def value_style(line: str, default_color: str = TEXT) -> tuple[int, str, str]:
    if line.startswith("Mean "):
        return 13, "700", TEXT
    if line.startswith("SD "):
        return 11, "400", "#64748b"
    if line.startswith("+"):
        return 13, "700", "#047857"
    if line.startswith("-") and line != "-":
        return 13, "700", "#b91c1c"
    return 13, "400", default_color


def mean_sd(mean: Any, sd: Any) -> str:
    return f"Mean {fmt(mean)}\nSD {fmt(sd)}"


def signed(value: Any) -> str:
    number = float(value)
    return f"{number:+.4f}"


def render_table(
    *,
    title: str,
    headers: list[str],
    rows: list[list[str]],
    widths: list[int],
    y_start: int,
    x_start: int = MARGIN,
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

    raw_rows = [headers] + rows
    wrapped_rows: list[list[list[str]]] = []
    for row in raw_rows:
        wrapped_row = []
        for idx, cell in enumerate(row):
            chars = max(8, (widths[idx] - 2 * pad_x) // 7)
            wrapped_row.append(wrap_text(str(cell), chars))
        wrapped_rows.append(wrapped_row)

    row_heights = [
        max(38 if idx == 0 else 42, max(len(lines) for lines in row) * line_h + 2 * pad_y)
        for idx, row in enumerate(wrapped_rows)
    ]
    table_y = y_start + 32
    total_h = sum(row_heights)
    parts = [
        f'<rect x="{x_start}" y="{y_start - 21}" width="{table_w}" height="36" rx="7" fill="#ffffff" '
        f'stroke="#d7e1ef" stroke-width="1"/>',
        f'<rect x="{x_start}" y="{y_start - 21}" width="8" height="36" rx="4" fill="{accent}"/>',
        text_element(x=x_start + 20, y=y_start + 2, text=title, size=18, weight="700"),
        f'<rect x="{x_start}" y="{table_y}" width="{table_w}" height="{total_h}" rx="8" '
        f'fill="#ffffff" stroke="{BORDER}" stroke-width="1.2"/>',
    ]

    y = table_y
    for row_idx, row in enumerate(wrapped_rows):
        row_h = row_heights[row_idx]
        first_value = raw_rows[row_idx][0] if raw_rows[row_idx] else ""
        second_value = raw_rows[row_idx][1] if len(raw_rows[row_idx]) > 1 else ""
        if row_idx == 0:
            fill = NAVY
        elif first_value == "Logistic Regression":
            fill = "#eff6ff" if second_value == "unbalanced" else "#dbeafe"
        elif first_value == "XGBoost":
            fill = "#fff7ed" if second_value == "unbalanced" else "#ffedd5"
        else:
            fill = "#ffffff" if row_idx % 2 else "#f8fafc"
        parts.append(f'<rect x="{x_start}" y="{y}" width="{table_w}" height="{row_h}" fill="{fill}"/>')
        for col_idx, lines in enumerate(row):
            x = x_positions[col_idx]
            width = widths[col_idx]
            parts.append(
                f'<rect x="{x}" y="{y}" width="{width}" height="{row_h}" '
                f'fill="none" stroke="{BORDER}" stroke-width="0.8"/>'
            )
            is_header = row_idx == 0
            anchor = "middle" if is_header or col_idx in center_cols else "start"
            text_x = x + width // 2 if anchor == "middle" else x + pad_x
            block_h = len(lines) * line_h
            text_y = y + (row_h - block_h) / 2 + 12
            base_color = "#ffffff" if is_header else (model_color(str(first_value)) if col_idx == 0 else TEXT)
            for line in lines:
                size, weight, color = value_style(line, base_color)
                if is_header:
                    size, weight, color = 12, "700", "#ffffff"
                elif col_idx == 0 and first_value in {"Logistic Regression", "XGBoost"}:
                    weight = "700"
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
    return parts, y


def render_note(title: str, lines: list[str], y_start: int, accent: str = TEAL) -> tuple[list[str], int]:
    pad = 16
    line_h = 17
    wrapped: list[str] = []
    for line in lines:
        wrapped.extend(wrap_text(line, max(40, (CONTENT_W - 2 * pad) // 7)))
    box_h = (len(wrapped) + 1) * line_h + 2 * pad
    parts = [
        f'<rect x="{MARGIN}" y="{y_start}" width="{CONTENT_W}" height="{box_h}" rx="9" fill="#ffffff" '
        'stroke="#d6dee9" stroke-width="1"/>',
        f'<rect x="{MARGIN}" y="{y_start}" width="7" height="{box_h}" rx="4" fill="{accent}"/>',
    ]
    y = y_start + pad + 13
    parts.append(text_element(x=MARGIN + pad + 6, y=y, text=title, size=13, weight="700", color=accent))
    y += line_h
    for line in wrapped:
        parts.append(text_element(x=MARGIN + pad + 6, y=y, text=line, size=12, color=MUTED))
        y += line_h
    return parts, y_start + box_h


def render_kpi_cards(cards: list[tuple[str, str, str, str]], y_start: int) -> tuple[list[str], int]:
    gap = 18
    card_w = (CONTENT_W - gap * (len(cards) - 1)) // len(cards)
    card_h = 88
    parts: list[str] = []
    x = MARGIN
    for label, value, detail, color in cards:
        parts.append(
            f'<rect x="{x}" y="{y_start}" width="{card_w}" height="{card_h}" rx="10" fill="#ffffff" '
            f'stroke="{BORDER}" stroke-width="1.2"/>'
        )
        parts.append(f'<rect x="{x}" y="{y_start}" width="8" height="{card_h}" rx="4" fill="{color}"/>')
        parts.append(text_element(x=x + 24, y=y_start + 26, text=label.upper(), size=11, weight="700", color=color))
        parts.append(text_element(x=x + 24, y=y_start + 55, text=value, size=24, weight="700"))
        parts.append(text_element(x=x + 24, y=y_start + 76, text=detail, size=12, color=MUTED))
        x += card_w + gap
    return parts, y_start + card_h


def render_bar_chart(summary: pd.DataFrame, y_start: int) -> tuple[list[str], int]:
    width = CONTENT_W
    height = 270
    x0 = MARGIN
    y0 = y_start + 52
    plot_x = x0 + 250
    plot_y = y0 + 22
    plot_w = width - 330
    plot_h = 155
    parts = [
        f'<rect x="{x0}" y="{y_start}" width="{width}" height="{height}" rx="9" fill="#ffffff" '
        f'stroke="{BORDER}" stroke-width="1"/>',
        f'<rect x="{x0}" y="{y_start}" width="8" height="{height}" rx="4" fill="{RED}"/>',
        text_element(x=x0 + 20, y=y_start + 28, text="Macro F1: Unbalanced vs Balanced", size=18, weight="700"),
    ]
    for tick in [0.0, 0.2, 0.4, 0.6, 0.8]:
        x = plot_x + tick / 0.8 * plot_w
        parts.append(f'<line x1="{x}" y1="{plot_y}" x2="{x}" y2="{plot_y + plot_h}" stroke="#e2e8f0" stroke-width="1"/>')
        parts.append(text_element(x=x, y=plot_y + plot_h + 20, text=f"{tick:.1f}", size=11, color=MUTED, anchor="middle"))

    ordered = summary.sort_values(["model_family", "balance_setting"], ascending=[True, False]).reset_index(drop=True)
    bar_h = 24
    gap = 15
    for idx, row in ordered.iterrows():
        y = plot_y + idx * (bar_h + gap)
        model = row["model_family"]
        setting = row["balance_setting"]
        value = float(row["macro_f1_mean"])
        color = model_color(model) if setting == "balanced" else "#94a3b8"
        bar_w = value / 0.8 * plot_w
        model_label = "LR" if model == "Logistic Regression" else "XGB"
        label = f"{model_label} {setting}"
        parts.append(text_element(x=plot_x - 16, y=y + 17, text=label, size=12, weight="700", color=model_color(model), anchor="end"))
        parts.append(f'<rect x="{plot_x}" y="{y}" width="{bar_w}" height="{bar_h}" rx="4" fill="{color}"/>')
        parts.append(text_element(x=plot_x + bar_w + 8, y=y + 17, text=fmt(value), size=12, weight="700"))
    parts.append(text_element(x=plot_x + plot_w / 2, y=plot_y + plot_h + 42, text="Macro F1 mean across five perfume-order seeds", size=12, color=MUTED, anchor="middle"))
    return parts, y_start + height


def summary_rows(summary: pd.DataFrame) -> list[list[str]]:
    rows = []
    ordered = summary.sort_values(["model_family", "balance_setting"], ascending=[True, False])
    for _, row in ordered.iterrows():
        rows.append(
            [
                row["model_family"],
                row["balance_setting"],
                row["balance_detail"],
                str(int(row["runs"])),
                mean_sd(row["macro_f1_mean"], row["macro_f1_std"]),
                signed(row["delta_macro_f1_mean_vs_unbalanced"]),
                mean_sd(row["macro_roc_auc_mean"], row["macro_roc_auc_std"]),
                mean_sd(row["macro_pr_auc_mean"], row["macro_pr_auc_std"]),
            ]
        )
    return rows


def per_label_delta_rows(per_label: pd.DataFrame) -> list[list[str]]:
    rows = []
    balanced = per_label[per_label["balance_setting"].eq("balanced")].copy()
    unbalanced = per_label[per_label["balance_setting"].eq("unbalanced")][
        ["model_family", "label", "f1_mean", "pr_auc_mean"]
    ].rename(columns={"f1_mean": "unbalanced_f1", "pr_auc_mean": "unbalanced_pr_auc"})
    merged = balanced.merge(unbalanced, on=["model_family", "label"], how="left")
    merged["abs_delta"] = merged["delta_f1_mean_vs_unbalanced"].abs()
    merged = merged.sort_values(["model_family", "abs_delta"], ascending=[True, False])
    for _, row in merged.iterrows():
        rows.append(
            [
                row["model_family"],
                row["label"],
                pct(row["positive_rate"]),
                fmt(row["unbalanced_f1"]),
                fmt(row["f1_mean"]),
                signed(row["delta_f1_mean_vs_unbalanced"]),
                signed(row["delta_pr_auc_mean_vs_unbalanced"]),
            ]
        )
    return rows


def distribution_rows(labels: pd.DataFrame) -> list[list[str]]:
    return [
        [
            row["label"],
            str(int(row["positive_count"])),
            pct(row["positive_rate"]),
            fmt(row["neg_to_pos_ratio"], 2),
        ]
        for _, row in labels.iterrows()
    ]


def main() -> None:
    summary = pd.read_csv(require(SUMMARY))
    per_label = pd.read_csv(require(PER_LABEL))
    labels = pd.read_csv(require(LABEL_DISTRIBUTION))

    lr_delta = summary.loc[
        summary["model_id"].eq("logistic_regression_balanced"),
        "delta_macro_f1_mean_vs_unbalanced",
    ].iloc[0]
    xgb_delta = summary.loc[summary["model_id"].eq("xgboost_balanced"), "delta_macro_f1_mean_vs_unbalanced"].iloc[0]
    lr_f1 = summary.loc[summary["model_id"].eq("logistic_regression_balanced"), "macro_f1_mean"].iloc[0]
    xgb_f1 = summary.loc[summary["model_id"].eq("xgboost_balanced"), "macro_f1_mean"].iloc[0]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{PAGE_W}" height="2200" viewBox="0 0 {PAGE_W} 2200">',
        '<rect width="100%" height="100%" fill="#f4f7fb"/>',
        f'<rect x="{MARGIN}" y="24" width="{CONTENT_W}" height="118" rx="14" fill="{NAVY}"/>',
        text_element(
            x=PAGE_W // 2,
            y=66,
            text="Synthetic-Only Balancing Ablation: 784x6 Corpus",
            size=28,
            weight="700",
            color="#ffffff",
            anchor="middle",
        ),
        text_element(
            x=PAGE_W // 2,
            y=98,
            text="Threshold 0.5 | GroupKFold by perfume_id | five random perfume-order seeds",
            size=14,
            color="#bfdbfe",
            anchor="middle",
        ),
        f'<rect x="{MARGIN}" y="122" width="{CONTENT_W}" height="26" rx="12" fill="#dbeafe"/>',
        text_element(
            x=PAGE_W // 2,
            y=140,
            text="The previous LR/XGBoost sweep already used balanced model settings; this isolates their effect.",
            size=12,
            weight="700",
            color=NAVY,
            anchor="middle",
        ),
    ]
    y = 176
    kpi_parts, y = render_kpi_cards(
        [
            ("LR F1 gain", signed(lr_delta), f"balanced mean F1 {fmt(lr_f1)}", BLUE),
            ("XGB F1 gain", signed(xgb_delta), f"balanced mean F1 {fmt(xgb_f1)}", RED),
            ("Rows", "4,704", "784 perfumes x 6 reviews", TEAL),
            ("Labels", "7", "anger excluded; zero positives", AMBER),
        ],
        y,
    )
    parts.extend(kpi_parts)
    y += 24

    note_parts, y = render_note(
        "Main takeaway",
        [
            "Balancing improves macro F1 for both models because it makes the models predict rare labels more often, especially sadness and disgust. It is not a free improvement on every metric: common-label F1 can drop, and PR-AUC changes are small or mixed because ranking quality is mostly unchanged.",
            "The balanced rows are therefore the right advisor-facing default for macro-F1 reporting, while the ablation explains why macro F1 is sensitive to label imbalance.",
        ],
        y,
        accent=TEAL,
    )
    parts.extend(note_parts)
    y += 24

    chart_parts, y = render_bar_chart(summary, y)
    parts.extend(chart_parts)
    y += 24

    table_parts, y = render_table(
        title="1. Full-Corpus Metric Summary",
        headers=["Model", "Variant", "Setting", "Runs", "Macro F1", "F1 Delta", "ROC-AUC", "PR-AUC"],
        rows=summary_rows(summary),
        widths=[180, 130, 265, 80, 190, 120, 190, 190],
        y_start=y,
        center_cols={0, 1, 3, 4, 5, 6, 7},
        accent=RED,
    )
    parts.extend(table_parts)
    y += 24

    table_parts, y = render_table(
        title="2. Per-Label F1 Changes From Balancing",
        headers=["Model", "Label", "Positive rate", "Unbal F1", "Bal F1", "F1 Delta", "PR-AUC Delta"],
        rows=per_label_delta_rows(per_label),
        widths=[185, 135, 140, 145, 145, 145, 150],
        y_start=y,
        center_cols={0, 1, 2, 3, 4, 5, 6},
        accent=BLUE,
    )
    parts.extend(table_parts)
    y += 24

    table_parts, y = render_table(
        title="3. Label Distribution",
        headers=["Label", "Positives", "Rate", "Neg/pos"],
        rows=distribution_rows(labels),
        widths=[360, 360, 360, 352],
        y_start=y,
        x_start=MARGIN,
        center_cols={0, 1, 2, 3},
        accent=TEAL,
    )
    parts.extend(table_parts)
    y += 24

    note_parts, y = render_note(
        "How to read this",
        [
            "Positive F1 Delta means the balanced model improved that label's F1 over its unbalanced counterpart. Large gains on sadness and disgust show the imbalance correction is doing what it is supposed to do.",
            "Negative deltas for anticipation and some LR joy results mean balancing can reduce performance on labels that were already easier or more common. The macro score still improves because rare labels matter equally in macro averaging.",
        ],
        y,
        accent=AMBER,
    )
    parts.extend(note_parts)
    y += 28

    parts.append(
        f'<rect x="{MARGIN}" y="{y - 18}" width="{CONTENT_W}" height="38" rx="9" fill="#ecfdf5" '
        'stroke="#bbf7d0" stroke-width="1"/>'
    )
    parts.append(
        text_element(
            x=PAGE_W // 2,
            y=y + 6,
            text="Status: balancing ablation complete for the full synthetic-only 784x6 corpus at threshold 0.5.",
            size=12,
            weight="700",
            color="#047857",
            anchor="middle",
        )
    )
    y += 30

    parts[0] = f'<svg xmlns="http://www.w3.org/2000/svg" width="{PAGE_W}" height="{y}" viewBox="0 0 {PAGE_W} {y}">'
    parts.append("</svg>\n")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(parts), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build a summarized SVG report for recent YouTube/synthetic results."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_SCRIPT = ROOT / "scripts/build_recent_synthetic_og_summary_image.py"
OUT = ROOT / "docs/synthetic_review_experiment_report/table_images/02_recent_synthetic_og_summarized.svg"
TABLE_W = 1190

spec = importlib.util.spec_from_file_location("recent_report_base", BASE_SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load base report helpers from {BASE_SCRIPT}")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def main() -> None:
    youtube_rows = base.load_csv(base.OG_FEATURES)
    synth_rows = base.load_csv(base.SYNTH_FEATURES)
    lr_rows = base.load_csv(base.LR_COMPARISON)
    xgb_rows = base.load_csv(base.XGB_COMPARISON)
    label_rows = base.load_csv(base.LABEL_DISTRIBUTION)
    g25_rows = base.load_csv(base.G25_API_COMPARISON)

    youtube_perfumes = base.unique_count(youtube_rows, "perfume_id")
    synth_perfumes = base.unique_count(synth_rows, "perfume_id")
    mixed_rows = len(youtube_rows) + len(synth_rows)
    mixed_perfumes = len({row["perfume_id"] for row in youtube_rows} | {row["perfume_id"] for row in synth_rows})

    lr_youtube = base.row_by_experiment(lr_rows, "og_only_reference")
    lr_aug = base.row_by_experiment(lr_rows, "og_plus_synthetic_eval_og_rows")
    lr_synth = base.row_by_experiment(lr_rows, "synthetic_only_review_level")
    xgb_youtube = base.row_by_experiment(xgb_rows, "og_only_reference")
    xgb_aug = base.row_by_experiment(xgb_rows, "og_plus_synthetic_eval_og_rows")
    xgb_synth = base.row_by_experiment(xgb_rows, "synthetic_only_review_level")
    g25_lr = base.row_by_model(g25_rows, "logistic_regression")
    g25_xgb = base.row_by_model(g25_rows, "xgboost")

    dataset_rows = [
        ["YouTube Reviews", str(youtube_perfumes), str(len(youtube_rows)), "Perfume-level aggregated documents"],
        ["Mixed experiment", str(mixed_perfumes), str(mixed_rows), "YouTube + synthetic rows, grouped by perfume"],
        ["Synthetic v2", str(synth_perfumes), str(len(synth_rows)), "784 perfumes x 3 review-level documents"],
    ]

    generation_rows = [
        ["v1 synthetic", "Perfume identity, persona, tone", "Safer descriptions", "Labels looked mostly neutral"],
        ["v2 synthetic", "Notes, scenarios, Fragrantica phrasing", "Human reactions", "Labels became more varied"],
    ]

    label_rows_table = []
    for label in base.TARGETS:
        youtube_label = base.label_distribution_lookup(label_rows, "og_784_perfume_level", label)
        synth_label = base.label_distribution_lookup(label_rows, "synthetic_v2_review_level", label)
        label_rows_table.append(
            [
                label,
                f"{youtube_label['positive_count']} ({base.pct(youtube_label['positive_rate'])})",
                f"{synth_label['positive_count']} ({base.pct(synth_label['positive_rate'])})",
            ]
        )

    labeler_rows = [
        [
            "Logistic Regression",
            base.fmt(g25_lr["macro_f1"]),
            base.fmt(lr_youtube["macro_f1"]),
            f"{float(lr_youtube['macro_f1']) - float(g25_lr['macro_f1']):+.4f}",
        ],
        [
            "XGBoost",
            base.fmt(g25_xgb["macro_f1"]),
            base.fmt(xgb_youtube["macro_f1"]),
            f"{float(xgb_youtube['macro_f1']) - float(g25_xgb['macro_f1']):+.4f}",
        ],
    ]

    results_rows = [
        [
            "Logistic Regression",
            "Main thesis model",
            base.fmt(lr_youtube["macro_f1"]),
            base.fmt(lr_synth["macro_f1"]),
            base.fmt(lr_aug["macro_f1"]),
            f"{float(lr_aug['macro_f1']) - float(lr_youtube['macro_f1']):+.4f}",
            "Small improvement",
        ],
        [
            "XGBoost",
            "Secondary benchmark",
            base.fmt(xgb_youtube["macro_f1"]),
            base.fmt(xgb_synth["macro_f1"]),
            base.fmt(xgb_aug["macro_f1"]),
            f"{float(xgb_aug['macro_f1']) - float(xgb_youtube['macro_f1']):+.4f}",
            "Slight drop",
        ],
    ]
    ranking_rows = [
        [
            "Logistic Regression",
            base.fmt(lr_youtube["macro_roc_auc"]),
            base.fmt(lr_aug["macro_roc_auc"]),
            f"{float(lr_aug['macro_roc_auc']) - float(lr_youtube['macro_roc_auc']):+.4f}",
            base.fmt(lr_youtube["macro_pr_auc"]),
            base.fmt(lr_aug["macro_pr_auc"]),
            f"{float(lr_aug['macro_pr_auc']) - float(lr_youtube['macro_pr_auc']):+.4f}",
        ],
        [
            "XGBoost",
            base.fmt(xgb_youtube["macro_roc_auc"]),
            base.fmt(xgb_aug["macro_roc_auc"]),
            f"{float(xgb_aug['macro_roc_auc']) - float(xgb_youtube['macro_roc_auc']):+.4f}",
            base.fmt(xgb_youtube["macro_pr_auc"]),
            base.fmt(xgb_aug["macro_pr_auc"]),
            f"{float(xgb_aug['macro_pr_auc']) - float(xgb_youtube['macro_pr_auc']):+.4f}",
        ],
    ]

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="900" viewBox="0 0 1280 900">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]
    y = 42
    parts.append(
        base.text_element(
            x=34,
            y=y,
            text="Summarized Synthetic + YouTube Reviews Results",
            size=24,
            weight="700",
        )
    )
    y += 24
    parts.append(
        base.text_element(
            x=34,
            y=y,
            text="Gemini 3.1 Pro (browser) labels; Gemini 2.5 Pro (API) shown only for comparable YouTube Review results.",
            size=13,
            color="#374151",
        )
    )
    y += 24

    table_parts, y = base.render_table(
        title="1. Data Used",
        headers=["Source", "Perfumes", "Rows", "Meaning"],
        rows=dataset_rows,
        widths=[220, 130, 130, TABLE_W - 480],
        numeric_cols={1, 2},
        y_start=y,
        center_cols={0, 1, 2, 3},
    )
    parts.extend(table_parts)

    y += base.SECTION_GAP
    table_parts, y = base.render_table(
        title="2. Label Distribution",
        headers=["Label", "YouTube positives", "Synthetic positives"],
        rows=label_rows_table,
        widths=[250, 470, 470],
        numeric_cols=set(),
        y_start=y,
        center_cols={0, 1, 2},
    )
    parts.extend(table_parts)

    y += base.SECTION_GAP
    table_parts, y = base.render_table(
        title="3. Labeler Comparison on YouTube Reviews",
        headers=["Model", "2.5 Pro API F1", "3.1 Pro browser F1", "Change"],
        rows=labeler_rows,
        widths=[360, 260, 310, 260],
        numeric_cols={1, 2, 3},
        y_start=y,
        center_cols={0, 1, 2, 3},
    )
    parts.extend(table_parts)

    y += base.SECTION_GAP
    table_parts, y = base.render_table(
        title="4. Main Result",
        headers=["Model", "Role", "YouTube-only F1", "Synthetic-only F1", "Augmented YouTube F1", "Change", "Reading"],
        rows=results_rows,
        widths=[185, 165, 145, 145, 170, 105, 275],
        numeric_cols={2, 3, 4, 5},
        y_start=y,
        center_cols={0, 1, 2, 3, 4, 5, 6},
    )
    parts.extend(table_parts)

    y += base.SECTION_GAP
    table_parts, y = base.render_table(
        title="5. ROC-AUC and PR-AUC Changes",
        headers=["Model", "YouTube ROC", "Augmented ROC", "ROC change", "YouTube PR", "Augmented PR", "PR change"],
        rows=ranking_rows,
        widths=[260, 150, 170, 135, 150, 170, 155],
        numeric_cols={1, 2, 3, 4, 5, 6},
        y_start=y,
        center_cols={0, 1, 2, 3, 4, 5, 6},
    )
    parts.extend(table_parts)

    y += base.SECTION_GAP
    table_parts, y = base.render_table(
        title="6. How v2 Synthetic Reviews Were Made",
        headers=["Version", "Main inputs", "Review style", "Result"],
        rows=generation_rows,
        widths=[180, 350, 300, TABLE_W - 830],
        numeric_cols=set(),
        y_start=y,
        center_cols={0, 1, 2, 3},
    )
    parts.extend(table_parts)

    img_h = y + base.SECTION_GAP
    parts[0] = f'<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="{img_h}" viewBox="0 0 1280 {img_h}">'
    parts.append("</svg>\n")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(parts), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()

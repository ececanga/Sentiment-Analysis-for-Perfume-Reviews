#!/usr/bin/env python3
"""Build meeting-facing result tables from frozen artifacts.

This script reads existing summary artifacts only. It does not run collection,
Gemini labeling, training, or evaluation.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "docs" / "flash_vs_pro_result_tables"


PATHS = {
    "decision": REPO_ROOT / "outputs/gemini_judge/judge_g25pro_v1/decision/branch_decision_summary.json",
    "manifest": REPO_ROOT / "outputs/gemini_judge/judge_g25pro_v1/phase1_freeze_manifest.json",
    "phase2a_pause": REPO_ROOT / "outputs/gemini_judge/judge_g25pro_2k_v1/planning/phase2a_collection_pause_summary.json",
    "thesis": REPO_ROOT / "outputs/gemini_judge/judge_g25pro_v1/analysis_out/thesis_results_summary.json",
    "benchmark": REPO_ROOT / "outputs/gemini_judge/judge_g25pro_v1/model_benchmark/tables/final_model_comparison_for_thesis.csv",
    "stage7_labels": REPO_ROOT / "outputs/gemini_judge/judge_g25pro_v1/training/stage7_gemini784_label_diagnostics.csv",
    "stage7_vad": REPO_ROOT / "outputs/gemini_judge/judge_g25pro_v1/training/stage7_gemini784_vad_metrics.csv",
    "stage8": REPO_ROOT / "outputs/gemini_judge/judge_g25pro_v1/analysis_out/stage8_importance_summary_report.json",
}


def load_json(key: str) -> dict[str, Any]:
    path = PATHS[key]
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_csv(key: str) -> list[dict[str, str]]:
    path = PATHS[key]
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    text = str(value)
    try:
        number = float(text)
    except ValueError:
        return text
    return f"{number:.{digits}f}"


def fmt_pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "n/a"


def md_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def write_table(
    slug: str,
    title: str,
    headers: list[str],
    rows: list[list[Any]],
    notes: list[str] | None = None,
) -> dict[str, str]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    md_path = OUT_DIR / f"{slug}.md"
    csv_path = OUT_DIR / f"{slug}.csv"

    with md_path.open("w", encoding="utf-8") as handle:
        handle.write(f"# {title}\n\n")
        handle.write("| " + " | ".join(md_escape(h) for h in headers) + " |\n")
        handle.write("| " + " | ".join("---" for _ in headers) + " |\n")
        for row in rows:
            handle.write("| " + " | ".join(md_escape(cell) for cell in row) + " |\n")
        if notes:
            handle.write("\nNotes:\n")
            for note in notes:
                handle.write(f"- {note}\n")

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)

    return {"title": title, "md": md_path.name, "csv": csv_path.name}


def add_index(entries: list[dict[str, str]], missing: list[str]) -> None:
    readme = OUT_DIR / "README.md"
    with readme.open("w", encoding="utf-8") as handle:
        handle.write("# Flash vs Pro and Results Tables\n\n")
        handle.write("Use this folder for meeting-readable tables. Each table has a Markdown version and a CSV version.\n\n")
        handle.write("Generated from frozen artifacts only. No new experiments were run.\n\n")
        handle.write("## Tables\n\n")
        handle.write("| # | Table | Markdown | CSV |\n")
        handle.write("|---:|---|---|---|\n")
        for index, entry in enumerate(entries, start=1):
            handle.write(
                f"| {index} | {entry['title']} | [{entry['md']}](./{entry['md']}) | "
                f"[{entry['csv']}](./{entry['csv']}) |\n"
            )
        handle.write("\n## Missing / Flagged\n\n")
        if missing:
            for item in missing:
                handle.write(f"- {item}\n")
        else:
            handle.write("- No critical requested result artifact was missing for these tables.\n")


def main() -> None:
    missing = [f"Missing artifact: {path}" for path in PATHS.values() if not path.exists()]
    if missing:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "README.md").write_text("\n".join(missing) + "\n", encoding="utf-8")
        raise SystemExit("Missing required artifacts; wrote README.md with details.")

    decision = load_json("decision")
    manifest = load_json("manifest")
    phase2a = load_json("phase2a_pause")
    thesis = load_json("thesis")
    benchmark_rows = load_csv("benchmark")
    stage7_label_rows = load_csv("stage7_labels")
    stage7_vad_rows = load_csv("stage7_vad")
    stage8 = load_json("stage8")

    entries: list[dict[str, str]] = []

    entries.append(
        write_table(
            "01_decision_snapshot",
            "Decision Snapshot",
            ["Topic", "Status", "Result", "Explanation"],
            [
                ["Gemini Flash baseline", "complete/frozen", "784 labels", "Original reference branch."],
                ["Gemini 2.5 Pro branch", "complete/frozen", "784 labels", "Main thesis reference-label branch."],
                ["Phase 1 decision", decision["decision"].upper(), "GO", "Pro beat Flash under predefined checks."],
                ["2K expansion", phase2a["status"], "35 new usable perfumes", "Paused, not merged."],
                ["Main thesis dataset", "complete", "784 perfumes", "Use frozen 784 for main results."],
                ["Phase2A leakage", "excluded", "0 overlap", "Main benchmark stays clean."],
            ],
        )
    )

    baseline_gold = decision["baseline_gold_macro_f1"]
    judge_gold = decision["judge_gold_macro_f1"]
    baseline_bench = decision["baseline_benchmark_mean_macro_f1"]
    judge_bench = decision["judge_benchmark_mean_macro_f1"]
    entries.append(
        write_table(
            "02_flash_vs_pro_branch_metrics",
            "Gemini Flash vs Gemini 2.5 Pro Branch Metrics",
            ["Metric", "Flash baseline", "Gemini 2.5 Pro", "Change", "Explanation"],
            [
                ["Gold macro-F1", fmt(baseline_gold), fmt(judge_gold), fmt(judge_gold - baseline_gold), "Pro aligned better with the gold check."],
                ["Benchmark mean macro-F1", fmt(baseline_bench), fmt(judge_bench), fmt(judge_bench - baseline_bench), "Downstream benchmark improved with Pro labels."],
                ["Label CSV rows", fmt(manifest["row_counts"]["baseline_label_csv_rows"], 0), fmt(manifest["row_counts"]["judge_label_csv_rows"], 0), "0", "Same 784-perfume branch size."],
                ["Label JSONL rows", "n/a", fmt(manifest["row_counts"]["judge_label_jsonl_rows"], 0), "n/a", "Pro JSONL output is complete."],
                ["Labeling failures", "n/a", fmt(manifest["integrity_summary"]["judge_labeling_failed_count"], 0), "n/a", "No failed Pro labels."],
                ["Missing IDs", "n/a", fmt(manifest["integrity_summary"]["judge_labeling_missing_ids_count"], 0), "n/a", "No missing Pro perfumes."],
            ],
            [
                "Macro-F1 averages F1 across emotions equally, so rare emotions count as much as common ones.",
                "Gemini Pro labels are model-based reference labels, not perfect human ground truth.",
            ],
        )
    )

    inventory_counts = phase2a["counts"]
    entries.append(
        write_table(
            "03_phase2a_inventory_and_pause_counts",
            "Phase2A Inventory and Pause Counts",
            ["Item", "Count", "Explanation"],
            [
                ["Frozen usable transcript perfumes", fmt(inventory_counts["frozen_phase1_perfumes"], 0), "Original transcript dataset was complete."],
                ["Initial usable outside frozen 784", "0", "Expansion required new collection."],
                ["Phase2A raw rows", fmt(inventory_counts["phase2a_raw_rows"], 0), "Partial collection output."],
                ["Phase2A formatted rows", fmt(inventory_counts["phase2a_formatted_rows"], 0), "Formatted partial collection output."],
                ["Unique usable Phase2A perfumes", fmt(inventory_counts["phase2a_unique_perfumes"], 0), "Only 35 new usable perfumes."],
                ["Overlap with frozen 784", fmt(inventory_counts["phase2a_overlap_with_frozen_784"], 0), "No overlap/leakage."],
                ["Total usable union", fmt(inventory_counts["total_usable_including_frozen_784"], 0), "819 total is still far from 2K."],
                ["Additional usable perfumes needed for 2K", "1,216", "The target gap was too large for the timeline."],
            ],
        )
    )

    status_counts = inventory_counts["phase2a_processed_status_counts"]
    status_order = ["done", "transcript_fetch_error", "transcript_unavailable", "api_error", "quota_error", "search_no_videos"]
    status_meaning = {
        "done": "Usable new transcript perfumes collected.",
        "transcript_fetch_error": "Main evidence of transcript/IP blocking.",
        "transcript_unavailable": "Video/candidate found, but usable captions unavailable.",
        "api_error": "Search/API failure.",
        "quota_error": "Search quota issue.",
        "search_no_videos": "No useful video found.",
    }
    entries.append(
        write_table(
            "04_phase2a_status_counts",
            "Phase2A Status Counts",
            ["Status", "Count", "Explanation"],
            [[status, fmt(status_counts.get(status, 0), 0), status_meaning[status]] for status in status_order],
        )
    )

    integrity = thesis["dataset_integrity"]
    entries.append(
        write_table(
            "05_frozen784_integrity",
            "Frozen 784 Dataset Integrity",
            ["Check", "Result", "Explanation"],
            [
                ["Unified feature rows", fmt(integrity["unified_total"], 0), "Feature table covers the frozen dataset."],
                ["Gemini Pro label rows", fmt(integrity["gemini_total"], 0), "Label table covers the frozen dataset."],
                ["Matched rows", fmt(integrity["matched"], 0), "Strict 784/784 join succeeded."],
                ["Unified-only rows", fmt(integrity["unified_only"], 0), "No feature-only perfumes."],
                ["Gemini-only rows", fmt(integrity["gemini_only"], 0), "No label-only perfumes."],
                ["Duplicate unified IDs", fmt(integrity["duplicate_ids_unified_count"], 0), "No duplicate feature IDs."],
                ["Duplicate Gemini IDs", fmt(integrity["duplicate_ids_gemini_count"], 0), "No duplicate label IDs."],
                ["Hash mismatches", fmt(integrity["hash_mismatch_count"], 0), "IDs/names aligned cleanly."],
                ["Stage8 matched rows", fmt(integrity["stage8_matched_rows"], 0), "Feature importance aligned with labels."],
                ["Stage8 target truth mismatches", fmt(integrity["stage8_target_truth_mismatch_count"], 0), "OOF targets match labels."],
                ["Phase2A overlap with branch table", fmt(integrity["phase2a_overlap_with_branch_merged_table"], 0), "No Phase2A leakage."],
            ],
        )
    )

    label_counts = thesis["gemini_label_distribution"]["emotion_positive_counts"]
    label_rates = thesis["gemini_label_distribution"]["emotion_positive_rates"]
    label_order = ["joy", "anticipation", "trust", "sadness", "surprise", "disgust", "fear", "anger"]
    label_meaning = {
        "joy": "Strong/common label.",
        "anticipation": "Strong/common label.",
        "trust": "Moderate/common label.",
        "sadness": "Rare; cautious claims.",
        "surprise": "Rare; cautious claims.",
        "disgust": "Exploratory rare label.",
        "fear": "Exploratory rare label.",
        "anger": "Not estimable.",
    }
    entries.append(
        write_table(
            "06_gemini_label_distribution",
            "Gemini Pro Label Distribution",
            ["Emotion", "Positive count", "Positive rate", "Explanation"],
            [[label, fmt(label_counts[label], 0), fmt_pct(label_rates[label]), label_meaning[label]] for label in label_order],
            ["Common emotions are learnable; rare emotions are not stable enough for strong claims."],
        )
    )

    model_meaning = {
        "xgboost": "Best macro-F1.",
        "logistic_regression": "Best macro PR-AUC; nearly tied on F1.",
        "bilstm_attention": "Weaker than classical models.",
        "bert": "Weaker than classical models.",
        "random_forest": "Lower F1, decent ranking metrics.",
    }
    ordered_benchmark = sorted(benchmark_rows, key=lambda row: float(row["macro_f1"]), reverse=True)
    entries.append(
        write_table(
            "07_model_benchmark",
            "Model Benchmark Results",
            ["Model", "Macro-F1", "Macro ROC-AUC", "Macro PR-AUC", "Explanation"],
            [
                [
                    row["model_name"],
                    fmt(row["macro_f1"]),
                    fmt(row["macro_roc_auc"]),
                    fmt(row["macro_pr_auc"]),
                    model_meaning.get(row["model_id"], ""),
                ]
                for row in ordered_benchmark
            ],
            ["Best macro-F1: XGBoost = 0.318425.", "Best macro PR-AUC: Logistic Regression = 0.326201."],
        )
    )

    stage7_global = thesis["stage7_training"]["global_emotion_metrics"]
    entries.append(
        write_table(
            "08_stage7_global_metrics",
            "Stage7 Global Emotion Metrics",
            ["Metric", "Value", "Explanation"],
            [
                ["Macro-F1 over nondegenerate labels", fmt(stage7_global["macro_f1_nondegenerate"]), "Modest balanced multi-label performance."],
                ["Micro-F1 over nondegenerate labels", fmt(stage7_global["micro_f1_nondegenerate"]), "Better when common labels dominate."],
                ["Subset accuracy", fmt(stage7_global["subset_accuracy"]), "Exact full-label matches are hard."],
                ["Hamming loss", fmt(stage7_global["hamming_loss"]), "Average per-label error is moderate."],
                ["Label cardinality", fmt(stage7_global["label_cardinality"]), "About 1.38 positive emotions per perfume."],
            ],
        )
    )

    stage7_order = ["joy", "anticipation", "trust", "fear", "disgust", "sadness", "surprise", "anger"]
    stage7_by_label = {row["emotion"]: row for row in stage7_label_rows}
    stage7_meaning = {
        "joy": "Strongest practical label.",
        "anticipation": "Useful signal.",
        "trust": "Weaker but present.",
        "fear": "Very rare; ROC-AUC can mislead.",
        "disgust": "Exploratory only.",
        "sadness": "Rare; cautious interpretation.",
        "surprise": "Rare; weak F1.",
        "anger": "Not estimable.",
    }
    entries.append(
        write_table(
            "09_stage7_per_label_diagnostics",
            "Stage7 Per-Label Emotion Diagnostics",
            ["Emotion", "Positives", "Status", "F1", "ROC-AUC", "Avg precision", "Explanation"],
            [
                [
                    label,
                    fmt(stage7_by_label[label]["n_pos"], 0),
                    stage7_by_label[label]["status"],
                    fmt(stage7_by_label[label]["f1"], 3),
                    fmt(stage7_by_label[label]["roc_auc"], 3),
                    fmt(stage7_by_label[label]["average_precision"], 3),
                    stage7_meaning[label],
                ]
                for label in stage7_order
            ],
            ["For rare labels, support count and F1 are more important than ROC-AUC."],
        )
    )

    vad_stats = thesis["gemini_label_distribution"]["gemini_vad_stats"]
    entries.append(
        write_table(
            "10_gemini_vad_target_distribution",
            "Gemini VAD Target Distribution",
            ["Dimension", "Mean", "Std. dev.", "Explanation"],
            [
                ["valence", fmt(vad_stats["valence"]["mean"], 3), fmt(vad_stats["valence"]["std"], 3), "Reviews lean positive."],
                ["arousal", fmt(vad_stats["arousal"]["mean"], 3), fmt(vad_stats["arousal"]["std"], 3), "Moderate activation/excitement."],
                ["dominance", fmt(vad_stats["dominance"]["mean"], 3), fmt(vad_stats["dominance"]["std"], 3), "Moderate confidence/control tone."],
            ],
        )
    )

    vad_meaning = {
        "valence": "Easiest VAD dimension.",
        "arousal": "Some signal, still modest.",
        "dominance": "Weakest by explained variance/correlation.",
        "average": "Useful but not highly predictive.",
    }
    entries.append(
        write_table(
            "11_stage7_vad_prediction",
            "Stage7 VAD Prediction Results",
            ["Dimension", "MAE", "RMSE", "R2", "Pearson r", "Spearman r", "Explanation"],
            [
                [
                    row["dimension"],
                    fmt(row["mae"], 3),
                    fmt(row["rmse"], 3),
                    fmt(row["r2"], 3),
                    fmt(row["pearson_r"], 3),
                    fmt(row["spearman_r"], 3),
                    vad_meaning[row["dimension"]],
                ]
                for row in stage7_vad_rows
            ],
        )
    )

    agreement = thesis["agreement_analysis"]
    nrc_acc = agreement["nrc_vs_gemini"]["per_emotion_accuracy"]
    entries.append(
        write_table(
            "12_nrc_vs_gemini_agreement",
            "NRC Binary Agreement with Gemini",
            ["Emotion/measure", "Accuracy", "Explanation"],
            [["Mean binary accuracy", fmt(agreement["nrc_vs_gemini"]["mean_binary_accuracy"], 3), "Weak overall agreement."]]
            + [[label, fmt(nrc_acc[label], 3), "Broad lexicon matching is not calibrated to Gemini labels."] for label in label_order],
        )
    )

    top_agreement = agreement["top_label_agreement"]["match_rates"]
    top_order = ["Cardiff_Label", "Savani_Label", "GoEmotions_Label", "JHartmann_Label"]
    entries.append(
        write_table(
            "13_transformer_top_label_agreement",
            "Transformer Top-Label Agreement with Gemini",
            ["Source", "Match rate", "Explanation"],
            [
                [source.replace("_Label", "").replace("JHartmann", "Hartmann"), fmt(top_agreement[source], 3), "Contextual emotion agreement is stronger than NRC matching."]
                for source in top_order
            ],
        )
    )

    vad_agreement = agreement["vad_continuous_agreement"]
    entries.append(
        write_table(
            "14_lexicon_vad_vs_gemini_vad",
            "Lexicon VAD vs Gemini VAD Agreement",
            ["Dimension", "Pearson r", "Spearman r", "MAE", "Explanation"],
            [
                [dimension, fmt(vad_agreement[dimension]["pearson_r"], 3), fmt(vad_agreement[dimension]["spearman_r"], 3), fmt(vad_agreement[dimension]["mae"], 3), "Modest directional alignment; scales differ."]
                for dimension in ["valence", "arousal", "dominance"]
            ],
        )
    )

    headline = stage8["headline"]
    entries.append(
        write_table(
            "15_stage8_discrete_emotion_family_importance",
            "Stage8 Discrete Emotion Family Importance",
            ["Feature family", "Mean positive importance", "Explanation"],
            [
                [row["family"], fmt(row["mean_positive_importance"], 4), "Transformer-derived features dominate." if row["family"] in {"Hartmann", "GoEmotions", "Cardiff"} else "Lower direct signal for discrete Gemini labels."]
                for row in headline["emotion_family_top_nonrare"]
            ],
            ["Permutation importance is interpretive, not causal."],
        )
    )

    entries.append(
        write_table(
            "16_stage8_top_discrete_emotion_features",
            "Stage8 Top Discrete Emotion Features",
            ["Feature", "Mean positive importance"],
            [[row["feature"], fmt(row["mean_positive_importance"], 4)] for row in headline["emotion_feature_top_nonrare"]],
        )
    )

    entries.append(
        write_table(
            "17_stage8_vad_family_importance",
            "Stage8 VAD Family Importance",
            ["Feature family", "Mean positive importance", "Explanation"],
            [
                [row["family"], fmt(row["mean_positive_importance"], 4), "Strongest grouped VAD signal." if row["family"] == "GoEmotions" else "Secondary VAD signal."]
                for row in headline["vad_family_top"]
            ],
        )
    )

    entries.append(
        write_table(
            "18_stage8_top_vad_features",
            "Stage8 Top VAD Features",
            ["Feature", "Mean positive importance"],
            [[row["feature"], fmt(row["mean_positive_importance"], 4)] for row in headline["vad_feature_top"]],
            ["NRC affect-count features are useful for VAD even though NRC is weak for direct Gemini label matching."],
        )
    )

    missing_notes = [
        "No critical requested result artifact was missing for these tables.",
        "No Phase2A modeling table is included because Phase2A was not merged or labeled for main quantitative results.",
        "No human-ground-truth claim is included because Gemini labels are model-based references.",
    ]
    entries.append(
        write_table(
            "19_missing_or_not_included",
            "Missing or Not Included",
            ["Item", "Status"],
            [[note, "not missing / intentionally excluded"] for note in missing_notes],
        )
    )

    add_index(entries, [])
    print(f"Wrote {len(entries)} tables to {OUT_DIR}")


if __name__ == "__main__":
    main()

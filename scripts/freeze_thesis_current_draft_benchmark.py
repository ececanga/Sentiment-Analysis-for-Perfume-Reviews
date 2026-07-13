#!/usr/bin/env python3
"""Create and verify the non-final Phase 1 thesis benchmark snapshot.

The snapshot is deliberately non-overwriting. Its large ``payload`` directory
is kept out of Git, while the root README, experiment manifest, and checksum
files can be versioned. Every payload file is copied byte-for-byte and recorded
with its source path, size, modification time, and SHA-256 digest.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SNAPSHOT = PROJECT_ROOT / "archive/thesis_current_draft_benchmark_2026-07-13"
SNAPSHOT_ID = "thesis_current_draft_benchmark_2026-07-13"
MODELS = (
    "logistic_regression",
    "xgboost",
    "random_forest",
    "bert",
    "bilstm_attention",
)
TARGETS = ("joy", "sadness", "fear", "trust", "surprise", "anticipation", "disgust")

PINNED_CONFIGS = (
    (
        "a182f2e0273dff33b875f27e5cb1ea9b6af87333",
        "logistic_regression_xgboost",
    ),
    (
        "fb63b85b26e6bd3599fdb15cb44bd4401a0bb521",
        "random_forest_bert_bilstm_attention",
    ),
)

# Each selected directory is copied recursively. The categories are retained in
# the experiment manifest so the archive can be audited without path guessing.
ARCHIVE_ENTRIES = (
    ("project_source", "README.md"),
    ("project_source", "config/evaluation_config.yaml"),
    ("project_source", "config/gemini_judge/judge_g25pro_v1.yaml"),
    (
        "project_source",
        "config/evaluation_config_gemini_browser_og784_current_draft_20260713.yaml",
    ),
    ("project_source", "notes/fullCode.py"),
    ("project_source", "src"),
    ("project_source", "scripts"),
    ("thesis_draft", "docs/thesis/ECE_CANGA_THESIS_DRAFT.docx"),
    ("thesis_support", "docs/synthetic_review_experiment_report"),
    ("canonical_raw_input", "data/raw/YouTube_Perfume_Reviews_FullyFormatted.csv"),
    ("browser_annotation", "data/annotation_packets/og784_gemini_browser_pro_review_level"),
    ("browser_annotation", "labels/gemini_browser_og784"),
    ("current_draft_benchmark", "outputs/gemini_browser_og784"),
    (
        "engineered_feature_base",
        "outputs/gemini_judge/judge_g25pro_v1/analysis_out/gemini_unified_merged.csv",
    ),
    ("synthetic_input", "data/synthetic"),
    (
        "synthetic_annotation",
        "data/annotation_packets/emotion_rich_v2_3reviews_gemini_browser_review_level",
    ),
    (
        "synthetic_annotation",
        "data/annotation_packets/emotion_rich_v2_reviews4x6_gemini_browser_review_level",
    ),
    ("synthetic_annotation", "labels/gemini_browser_review_level"),
    ("synthetic_annotation", "labels/gemini_browser_review_level_reviews4x6"),
    (
        "synthetic_features",
        "outputs/synthetic_review_level_tests/emotion_rich_v2_3reviews/supervised_eval_gemini_browser",
    ),
    (
        "synthetic_features",
        "outputs/synthetic_review_level_tests/emotion_rich_v2_6reviews/supervised_eval_gemini_browser",
    ),
    (
        "thesis_table_4_7",
        "outputs/synthetic_only_random_review_count_sweep_gemini_browser_unweighted_threshold03",
    ),
    (
        "thesis_table_4_7",
        "outputs/synthetic_only_random_review_count_sweep_gemini_browser_6reviews_unweighted_threshold03",
    ),
    (
        "synthetic_context",
        "outputs/synthetic_only_random_review_count_sweep_gemini_browser",
    ),
    (
        "synthetic_context",
        "outputs/synthetic_only_random_review_count_sweep_gemini_browser_6reviews",
    ),
    (
        "thesis_table_4_8",
        "outputs/synthetic_only_resampling_ablation_gemini_browser_3reviews_threshold03",
    ),
    (
        "thesis_table_4_8",
        "outputs/synthetic_only_resampling_ablation_gemini_browser_6reviews_threshold03",
    ),
    (
        "thesis_table_4_9",
        "outputs/review_level_synthetic_augmentation_gemini_browser_og784",
    ),
    (
        "thesis_table_4_9",
        "outputs/review_level_synthetic_augmentation_gemini_browser_og784_xgboost",
    ),
    (
        "thesis_table_4_10",
        "outputs/og_synthetic_label_distribution_balancing_threshold03_v3",
    ),
    ("historical_non_final", "outputs/gemini_judge/judge_g25pro_v1/model_benchmark"),
    ("historical_non_final", "outputs/model_benchmark"),
    ("historical_non_final", "outputs/model_benchmark_t02"),
    ("historical_non_final", "outputs/model_benchmark_t03"),
    ("historical_non_final", "outputs/model_benchmark_t07"),
    ("historical_non_final", "outputs/model_benchmark_threshold_sweep"),
)

SKIP_DIR_NAMES = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
SKIP_FILE_SUFFIXES = {".pyc", ".pyo"}
SKIP_FILE_NAMES = {".DS_Store"}

THESIS_TABLES: list[dict[str, Any]] = [
    {
        "table": "4.1",
        "status": "current_draft_non_final",
        "description": "Five-model browser-label benchmark summary.",
        "artifacts": [
            "outputs/gemini_browser_og784/model_benchmark/tables/comparison_table.csv"
        ],
    },
    {
        "table": "4.2",
        "status": "historical_non_final",
        "description": "Earlier mixed-branch threshold-0.3 comparison; not final thesis evidence.",
        "artifacts": [
            "outputs/gemini_judge/judge_g25pro_v1/model_benchmark/tables/comparison_table.csv",
            "outputs/model_benchmark/tables/comparison_table.csv",
        ],
    },
    {
        "table": "4.3",
        "status": "historical_non_final",
        "description": "Mixed-source threshold comparison; must not be reported as a controlled sweep.",
        "artifacts": [
            "outputs/model_benchmark_threshold_sweep/tables/final_model_threshold_comparison_03_05_07.csv"
        ],
    },
    {
        "table": "4.4",
        "status": "historical_non_final",
        "description": "Historical best fixed thresholds; not final evidence.",
        "artifacts": [
            "outputs/model_benchmark_threshold_sweep/tables/final_model_threshold_comparison_for_thesis.csv",
            "outputs/model_benchmark_threshold_sweep/tables/best_fixed_threshold_per_model.csv"
        ],
    },
    {
        "table": "4.5",
        "status": "current_draft_non_final",
        "description": "Per-label metrics for the five browser-label benchmark models.",
        "artifacts": [
            f"outputs/gemini_browser_og784/model_benchmark/metrics/{model}_per_label.csv"
            for model in MODELS
        ],
    },
    {
        "table": "4.6",
        "status": "historical_non_final",
        "description": "Earlier judge-branch per-label results; not final thesis evidence.",
        "artifacts": [
            f"outputs/gemini_judge/judge_g25pro_v1/model_benchmark/metrics/{model}_per_label.csv"
            for model in MODELS
        ],
    },
    {
        "table": "4.7",
        "status": "current_draft_exploratory",
        "description": "Synthetic-only random review-count sweep without class weighting at threshold 0.3.",
        "artifacts": [
            "outputs/synthetic_only_random_review_count_sweep_gemini_browser_unweighted_threshold03/synthetic_only_random_review_count_sweep_summary.csv",
            "outputs/synthetic_only_random_review_count_sweep_gemini_browser_6reviews_unweighted_threshold03/synthetic_only_random_review_count_sweep_summary.csv",
        ],
    },
    {
        "table": "4.8",
        "status": "current_draft_exploratory",
        "description": "Synthetic-only resampling ablation at threshold 0.3.",
        "artifacts": [
            "outputs/synthetic_only_resampling_ablation_gemini_browser_3reviews_threshold03/synthetic_only_resampling_ablation_summary.csv",
            "outputs/synthetic_only_resampling_ablation_gemini_browser_6reviews_threshold03/synthetic_only_resampling_ablation_summary.csv",
        ],
    },
    {
        "table": "4.9",
        "status": "current_draft_exploratory",
        "description": "Review-level synthetic augmentation comparisons for LR and XGBoost.",
        "artifacts": [
            "outputs/review_level_synthetic_augmentation_gemini_browser_og784/metric_comparison.csv",
            "outputs/review_level_synthetic_augmentation_gemini_browser_og784_xgboost/metric_comparison.csv",
            "outputs/review_level_synthetic_augmentation_gemini_browser_og784/comparison_report.json",
            "outputs/review_level_synthetic_augmentation_gemini_browser_og784_xgboost/comparison_report.json",
        ],
    },
    {
        "table": "4.10",
        "status": "current_draft_exploratory",
        "description": "Original-plus-synthetic label-distribution balancing experiment.",
        "artifacts": [
            "outputs/og_synthetic_label_distribution_balancing_threshold03_v3/label_distribution_balancing_summary.csv",
            "outputs/og_synthetic_label_distribution_balancing_threshold03_v3/label_distribution_balancing_report.json",
        ],
    },
    {
        "table": "5.1",
        "status": "narrative_synthesis",
        "description": "Chapter 5 synthesis of Chapter 4; no independent result artifact.",
        "artifacts": [],
    },
]

KNOWN_LIMITATIONS = [
    "This is a current-draft benchmark snapshot, not the final thesis benchmark.",
    "The perfume-level KFold split does not group shared source videos or normalized transcripts.",
    "A local audit found 341 of 392 shared video IDs crossing folds, affecting 462 of 784 perfumes.",
    "A local audit found 340 normalized transcript hashes crossing folds, affecting 459 of 784 perfumes.",
    "The exact browser model version is asserted in the project log but is not machine-verifiable from the preserved browser responses.",
    "Per-batch browser labeling times, browser session identifiers, and exact ingestion commands were not preserved.",
    "For batches 6-8, raw part files are repeated copies of their complete multi-section manual-paste source, not independent browser exports.",
    "The five model runs span two Git commits and two dates; no single clean checkout produced the whole table.",
    "The artifact field named macro_pr_auc contains average precision and should be described as macro average precision.",
    "The historical threshold tables combine incompatible artifact branches and are non-final.",
    "Uncertainty intervals, paired significance tests, dummy/TF-IDF baselines, and grouped-CV results are absent from this snapshot.",
]


class SnapshotError(RuntimeError):
    """Raised when creation or verification cannot preserve the contract."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot-root",
        type=Path,
        default=DEFAULT_SNAPSHOT,
        help=f"Snapshot directory (default: {DEFAULT_SNAPSHOT.relative_to(PROJECT_ROOT)}).",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify an existing payload and metadata without reading current source artifacts.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def should_skip(path: Path) -> bool:
    return (
        any(part in SKIP_DIR_NAMES for part in path.parts)
        or path.name in SKIP_FILE_NAMES
        or path.suffix in SKIP_FILE_SUFFIXES
        or path.name.startswith("~$")
    )


def iter_selected_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        if not should_skip(path):
            yield path
        return
    if not path.is_dir():
        raise SnapshotError(f"Required archive entry is missing: {path.relative_to(PROJECT_ROOT)}")
    for candidate in sorted(path.rglob("*")):
        if candidate.is_file() and not should_skip(candidate.relative_to(path)):
            yield candidate


def git_text(*args: str, check: bool = True) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if check and proc.returncode != 0:
        raise SnapshotError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout


def collect_git_state() -> dict[str, Any]:
    status = git_text("status", "--porcelain=v1", "--untracked-files=all")
    return {
        "head": git_text("rev-parse", "HEAD").strip(),
        "branch": git_text("branch", "--show-current").strip() or "detached",
        "dirty": bool(status.strip()),
        "status_porcelain_path": "provenance/git_status_porcelain.txt",
        "diff_stat_path": "provenance/git_diff_stat.txt",
        "note": "The snapshot records and hashes selected working-tree files, including selected modified and untracked artifacts.",
    }


def csv_records(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def require_unique_ids(rows: list[dict[str, str]], column: str, label: str) -> set[str]:
    if not rows or column not in rows[0]:
        raise SnapshotError(f"{label} is missing required column {column!r}")
    values = [row[column] for row in rows]
    if len(values) != len(set(values)):
        raise SnapshotError(f"{label} contains duplicate {column} values")
    return set(values)


def validate_benchmark(root: Path) -> dict[str, Any]:
    labels_path = root / "labels/gemini_browser_og784/gemini_browser_labels_og784_v1.csv"
    feature_path = root / "outputs/gemini_browser_og784/analysis_out/gemini_browser_og784_feature_labels.csv"
    folds_path = root / "outputs/gemini_browser_og784/model_benchmark/tables/fold_assignments.csv"
    comparison_path = root / "outputs/gemini_browser_og784/model_benchmark/tables/comparison_table.csv"
    dataset_path = root / "outputs/gemini_browser_og784/model_benchmark/datasets/perfume_text_dataset_v1.jsonl"

    labels = csv_records(labels_path)
    features = csv_records(feature_path)
    folds = csv_records(folds_path)
    comparison = csv_records(comparison_path)
    if len(labels) != 784 or len(features) != 784 or len(folds) != 784:
        raise SnapshotError(
            f"Expected 784 labels/features/folds; found {len(labels)}/{len(features)}/{len(folds)}"
        )

    label_ids = require_unique_ids(labels, "perfume_id", "label table")
    feature_ids = require_unique_ids(features, "perfume_id", "feature-label table")
    fold_ids = require_unique_ids(folds, "perfume_id", "fold manifest")
    if label_ids != feature_ids or label_ids != fold_ids:
        raise SnapshotError("Label, feature, and fold perfume ID sets do not match")

    fold_by_id = {row["perfume_id"]: row["fold_id"] for row in folds}
    feature_by_id = {row["perfume_id"]: row for row in features}
    fold_sizes: dict[str, int] = {}
    for fold in fold_by_id.values():
        fold_sizes[fold] = fold_sizes.get(fold, 0) + 1
    if sorted(fold_sizes.values()) != [156, 157, 157, 157, 157]:
        raise SnapshotError(f"Unexpected fold sizes: {fold_sizes}")

    comparison_models = [row.get("model_id", "") for row in comparison]
    if set(comparison_models) != set(MODELS) or len(comparison_models) != len(MODELS):
        raise SnapshotError(f"Comparison table model set is invalid: {comparison_models}")

    model_checks: dict[str, Any] = {}
    run_commits: set[str] = set()
    for model in MODELS:
        oof_path = root / f"outputs/gemini_browser_og784/model_benchmark/predictions/{model}_oof.csv"
        oof = csv_records(oof_path)
        if len(oof) != 784:
            raise SnapshotError(f"{model} OOF has {len(oof)} rows instead of 784")
        oof_ids = require_unique_ids(oof, "perfume_id", f"{model} OOF")
        if oof_ids != fold_ids:
            raise SnapshotError(f"{model} OOF perfume IDs differ from the fold manifest")
        for row in oof:
            perfume_id = row["perfume_id"]
            if row["fold_id"] != fold_by_id[perfume_id]:
                raise SnapshotError(f"{model} OOF fold mismatch for {perfume_id}")
            for target in TARGETS:
                truth_column = f"true_{target}"
                if truth_column not in row:
                    raise SnapshotError(f"{model} OOF is missing {truth_column}")
                if row[truth_column] != feature_by_id[perfume_id][target]:
                    raise SnapshotError(f"{model} OOF target mismatch for {perfume_id}/{target}")
                probability = row.get(f"prob_{target}", "")
                if probability == "":
                    raise SnapshotError(f"{model} OOF has a missing probability for {perfume_id}/{target}")

        metrics_path = root / f"outputs/gemini_browser_og784/model_benchmark/metrics/{model}.json"
        per_label_path = root / f"outputs/gemini_browser_og784/model_benchmark/metrics/{model}_per_label.csv"
        run_path = root / f"outputs/gemini_browser_og784/model_benchmark/logs/{model}_run.json"
        for required in (metrics_path, per_label_path, run_path):
            if not required.is_file():
                raise SnapshotError(f"Missing model artifact: {required.relative_to(root)}")
        run = json.loads(run_path.read_text(encoding="utf-8"))
        if run.get("model_id") != model:
            raise SnapshotError(f"Run manifest/model mismatch for {model}")
        run_commits.add(str(run.get("git_commit", "")))
        model_checks[model] = {
            "oof_rows": len(oof),
            "unique_perfumes": len(oof_ids),
            "fold_mismatches": 0,
            "target_mismatches": 0,
            "missing_probabilities": 0,
            "run_timestamp_utc": run.get("timestamp_utc"),
            "git_commit": run.get("git_commit"),
            "command": run.get("command"),
        }

    with dataset_path.open("r", encoding="utf-8") as handle:
        dataset_rows = [json.loads(line) for line in handle if line.strip()]
    if len(dataset_rows) != 784:
        raise SnapshotError(f"Deep-text dataset has {len(dataset_rows)} rows instead of 784")

    report_path = root / "labels/gemini_browser_og784/gemini_browser_labels_og784_v1_report.json"
    label_report = json.loads(report_path.read_text(encoding="utf-8"))
    if label_report.get("completed_rows") != 784 or label_report.get("remaining_rows") != 0:
        raise SnapshotError("Browser label completion report is not complete")

    support = {
        target: sum(int(row[target]) for row in labels)
        for target in ("joy", "sadness", "anger", "fear", "trust", "surprise", "anticipation", "disgust")
    }
    figures = list(
        (root / "outputs/gemini_browser_og784/model_benchmark/figures").glob("*.png")
    )
    return {
        "status": "passed",
        "labels": {"rows": len(labels), "unique_perfumes": len(label_ids), "positive_support": support},
        "feature_label_rows": len(features),
        "folds": {"rows": len(folds), "sizes": dict(sorted(fold_sizes.items()))},
        "deep_text_dataset_rows": len(dataset_rows),
        "comparison_models": comparison_models,
        "benchmark_png_figures": len(figures),
        "run_commits": sorted(run_commits),
        "models": model_checks,
    }


def copy_selected_files(staging: Path) -> list[dict[str, Any]]:
    payload = staging / "payload"
    payload.mkdir(parents=True)
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for category, relative in ARCHIVE_ENTRIES:
        source_entry = PROJECT_ROOT / relative
        if not source_entry.exists():
            raise SnapshotError(f"Required archive entry is missing: {relative}")
        for source in iter_selected_files(source_entry):
            source_relative = source.relative_to(PROJECT_ROOT).as_posix()
            if source_relative in seen:
                continue
            seen.add(source_relative)
            destination = payload / source_relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            source_digest = sha256_file(source)
            shutil.copy2(source, destination)
            destination_digest = sha256_file(destination)
            if destination_digest != source_digest:
                raise SnapshotError(f"Checksum mismatch while copying {source_relative}")
            stat = source.stat()
            records.append(
                {
                    "category": category,
                    "source_kind": "working_tree",
                    "source_path": source_relative,
                    "payload_path": source_relative,
                    "bytes": stat.st_size,
                    "source_mtime_ns": stat.st_mtime_ns,
                    "sha256": source_digest,
                }
            )
    return records


def add_generated_payload(
    staging: Path,
    records: list[dict[str, Any]],
    *,
    relative: str,
    content: bytes,
    category: str,
    source_path: str,
) -> None:
    destination = staging / "payload" / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)
    records.append(
        {
            "category": category,
            "source_kind": "generated_provenance" if not source_path.startswith("git:") else "git_object",
            "source_path": source_path,
            "payload_path": relative,
            "bytes": len(content),
            "source_mtime_ns": None,
            "sha256": sha256_bytes(content),
        }
    )


def add_provenance_files(
    staging: Path, records: list[dict[str, Any]], git_state: dict[str, Any]
) -> None:
    for commit, model_tag in PINNED_CONFIGS:
        content = subprocess.run(
            ["git", "show", f"{commit}:config/evaluation_config.yaml"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
        ).stdout
        add_generated_payload(
            staging,
            records,
            relative=f"provenance/config_at_{commit}_{model_tag}.yaml",
            content=content,
            category="pinned_run_configuration",
            source_path=f"git:{commit}:config/evaluation_config.yaml",
        )

    status = git_text("status", "--porcelain=v1", "--untracked-files=all")
    diff_stat = "WORKTREE DIFF STAT\n" + git_text("diff", "--stat", check=False)
    diff_stat += "\nSTAGED DIFF STAT\n" + git_text("diff", "--cached", "--stat", check=False)
    add_generated_payload(
        staging,
        records,
        relative="provenance/git_status_porcelain.txt",
        content=status.encode("utf-8"),
        category="git_provenance",
        source_path="generated:git status --porcelain=v1 --untracked-files=all",
    )
    add_generated_payload(
        staging,
        records,
        relative="provenance/git_diff_stat.txt",
        content=diff_stat.encode("utf-8"),
        category="git_provenance",
        source_path="generated:git diff --stat and git diff --cached --stat",
    )
    add_generated_payload(
        staging,
        records,
        relative="provenance/git_state.json",
        content=(json.dumps(git_state, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        category="git_provenance",
        source_path="generated:git state summary",
    )
    add_generated_payload(
        staging,
        records,
        relative="provenance/thesis_table_artifact_map.json",
        content=(json.dumps(THESIS_TABLES, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        category="experiment_manifest_support",
        source_path="generated:Phase 1 thesis table audit",
    )

    non_final = """# Non-final and historical artifacts

This payload intentionally preserves artifacts that must not be presented as
final thesis evidence. Preservation does not endorse their methodology.

- `outputs/gemini_browser_og784/model_benchmark/` is the current-draft browser
  benchmark. It uses perfume-level KFold assignments with known video/transcript
  source overlap and must be superseded by grouped cross-validation.
- `src/training/train_eval_gemini784.py` is the legacy eight-label trainer and
  is retained only for history.
- `outputs/model_benchmark*` and
  `outputs/gemini_judge/judge_g25pro_v1/model_benchmark/` are historical
  branches, not substitutes for the browser-label benchmark.
- `outputs/model_benchmark_threshold_sweep/` combines values from different
  branches/Git states and is not a controlled final threshold experiment.
- Thesis Tables 4.2, 4.3, 4.4, and 4.6 are historical/non-final. Tables 4.1 and
  4.5 are current-draft/non-final until the grouped-CV rerun. Tables 4.7-4.10
  are exploratory synthetic experiments.
"""
    add_generated_payload(
        staging,
        records,
        relative="provenance/NON_FINAL_ARTIFACTS.md",
        content=non_final.encode("utf-8"),
        category="experiment_manifest_support",
        source_path="generated:Phase 1 status classification",
    )


def archive_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_category: dict[str, dict[str, int]] = {}
    for record in records:
        bucket = by_category.setdefault(record["category"], {"files": 0, "bytes": 0})
        bucket["files"] += 1
        bucket["bytes"] += int(record["bytes"])
    return {
        "file_count": len(records),
        "total_bytes": sum(int(record["bytes"]) for record in records),
        "by_category": dict(sorted(by_category.items())),
    }


def run_records(root: Path) -> list[dict[str, Any]]:
    records = []
    for model in MODELS:
        path = root / f"outputs/gemini_browser_og784/model_benchmark/logs/{model}_run.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        records.append(
            {
                "model_id": model,
                "timestamp_utc": data.get("timestamp_utc"),
                "git_commit": data.get("git_commit"),
                "dataset_path": data.get("dataset_path"),
                "split_manifest_path": data.get("split_manifest_path"),
                "threshold": data.get("threshold"),
                "command": data.get("command"),
                "run_manifest": path.relative_to(root).as_posix(),
            }
        )
    return records


def build_readme(summary: dict[str, Any]) -> str:
    mib = summary["total_bytes"] / (1024 * 1024)
    return f"""# Thesis current-draft benchmark snapshot (2026-07-13)

Status: **archived current-draft benchmark; non-final**.

This Phase 1 snapshot preserves the exact selected working artifacts before the
source-group leakage repair and benchmark rerun. It is non-overwriting and does
not convert the archived results into final thesis evidence.

## Archive contract

- `{summary['file_count']:,}` payload files ({mib:.1f} MiB) are inventoried in
  `experiment_manifest.json` and hashed in `SHA256SUMS`.
- Canonical labels, feature inputs, raw transcripts, fold assignments, five OOF
  prediction sets, metrics, run manifests, tables, figures, annotation packets,
  pasted browser outputs, synthetic experiments, source code, and the thesis
  draft are retained under `payload/` using their repository-relative paths.
- The payload is intentionally Git-ignored because of its size. The metadata
  files at this directory root are suitable for version control.
- Two commit-pinned copies of `config/evaluation_config.yaml` and a clearly
  marked reconstructed browser-specific configuration are included.

## Critical interpretation boundary

The existing fold manifest groups by perfume only. A source audit found 341 of
392 shared video IDs crossing folds (462 perfumes) and 340 normalized transcript
hashes crossing folds (459 perfumes). These values require a grouped-CV rerun
before the benchmark is used as final evidence.

The project log records the browser labeler as Gemini 3.1 Pro. The repository
preserves the prompt, packets, pasted CSV responses, validation outputs, and
final label tables, but the browser responses do not contain machine-verifiable
model-version or session metadata.

The field historically called `macro_pr_auc` was computed as average precision;
future thesis text should call it **macro average precision**.

## Non-final material

`payload/provenance/NON_FINAL_ARTIFACTS.md` identifies the legacy eight-label
trainer, historical result branches, invalid mixed-source threshold tables, and
the thesis-table status classification. `experiment_manifest.json` maps Tables
4.1-4.10 and 5.1 to their preserved artifacts.

## Verify

From the repository root:

```bash
.venv/bin/python scripts/freeze_thesis_current_draft_benchmark.py --verify-only
```

Verification checks every payload SHA-256 digest, file size, metadata checksum,
and rejects unrecorded payload files.
"""


def write_metadata(
    staging: Path,
    *,
    records: list[dict[str, Any]],
    validation: dict[str, Any],
    git_state: dict[str, Any],
) -> None:
    records.sort(key=lambda item: item["payload_path"])
    summary = archive_summary(records)
    manifest = {
        "schema_version": 1,
        "snapshot_id": SNAPSHOT_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "archived_current_draft_non_final_pending_grouped_cv_rerun",
        "immutability_policy": "non_overwriting; create fails if snapshot root exists",
        "payload_git_policy": "payload is local and Git-ignored; root metadata is versionable",
        "git_state": git_state,
        "archive_summary": summary,
        "validation": validation,
        "annotation_provenance": {
            "recorded_labeler": "Gemini 3.1 Pro in the browser",
            "exact_model_identity": "not_machine_verifiable_from_preserved_responses",
            "packet_generated_at_utc": "2026-06-15T08:10:54.077823+00:00",
            "final_report_generated_at_utc": "2026-06-15T10:00:04.763160+00:00",
            "prompt": "data/annotation_packets/og784_gemini_browser_pro_review_level/prompt.md",
            "packet_manifest": "data/annotation_packets/og784_gemini_browser_pro_review_level/manifest.csv",
            "raw_outputs": "labels/gemini_browser_og784/raw_batches",
            "manual_pastes": "labels/gemini_browser_og784/manual_pastes",
            "final_labels": "labels/gemini_browser_og784/gemini_browser_labels_og784_v1.csv",
        },
        "known_source_overlap_audit": {
            "shared_video_ids": 392,
            "shared_video_ids_crossing_folds": 341,
            "perfumes_affected_by_cross_fold_video_ids": 462,
            "normalized_transcript_hashes_crossing_folds": 340,
            "perfumes_affected_by_cross_fold_transcript_hashes": 459,
            "status": "known_limitation_to_be_repaired_in_next_phase",
        },
        "known_limitations": KNOWN_LIMITATIONS,
        "run_records": run_records(staging / "payload"),
        "thesis_table_artifact_map": THESIS_TABLES,
        "non_final_artifacts": {
            "legacy_eight_label_trainer": "src/training/train_eval_gemini784.py",
            "historical_result_roots": [
                "outputs/gemini_judge/judge_g25pro_v1/model_benchmark",
                "outputs/model_benchmark",
                "outputs/model_benchmark_t02",
                "outputs/model_benchmark_t03",
                "outputs/model_benchmark_t07",
                "outputs/model_benchmark_threshold_sweep",
            ],
            "historical_thesis_tables": ["4.2", "4.3", "4.4", "4.6"],
        },
        "files": records,
    }
    readme = build_readme(summary)
    manifest_text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    checksums = "".join(
        f"{record['sha256']}  payload/{record['payload_path']}\n" for record in records
    )
    (staging / "README.md").write_text(readme, encoding="utf-8")
    (staging / "experiment_manifest.json").write_text(manifest_text, encoding="utf-8")
    (staging / "SHA256SUMS").write_text(checksums, encoding="utf-8")
    metadata_checksums = (
        f"{sha256_file(staging / 'README.md')}  README.md\n"
        f"{sha256_file(staging / 'experiment_manifest.json')}  experiment_manifest.json\n"
        f"{sha256_file(staging / 'SHA256SUMS')}  SHA256SUMS\n"
    )
    (staging / "METADATA_SHA256SUMS").write_text(metadata_checksums, encoding="utf-8")


def verify_snapshot(snapshot: Path) -> dict[str, Any]:
    manifest_path = snapshot / "experiment_manifest.json"
    if not manifest_path.is_file():
        raise SnapshotError(f"Missing experiment manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest.get("files", [])
    expected_paths: set[str] = set()
    total_bytes = 0
    for record in records:
        relative = str(record["payload_path"])
        if relative in expected_paths:
            raise SnapshotError(f"Duplicate payload record: {relative}")
        expected_paths.add(relative)
        path = snapshot / "payload" / relative
        if not path.is_file():
            raise SnapshotError(f"Missing payload file: {relative}")
        size = path.stat().st_size
        if size != int(record["bytes"]):
            raise SnapshotError(f"Size mismatch for {relative}: {size} != {record['bytes']}")
        digest = sha256_file(path)
        if digest != record["sha256"]:
            raise SnapshotError(f"SHA-256 mismatch for {relative}")
        total_bytes += size

    actual_paths = {
        path.relative_to(snapshot / "payload").as_posix()
        for path in (snapshot / "payload").rglob("*")
        if path.is_file()
    }
    extras = sorted(actual_paths - expected_paths)
    if extras:
        raise SnapshotError(f"Unrecorded payload files: {extras[:10]}")
    missing = sorted(expected_paths - actual_paths)
    if missing:
        raise SnapshotError(f"Manifest-only payload files: {missing[:10]}")

    metadata_file = snapshot / "METADATA_SHA256SUMS"
    for line in metadata_file.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        if sha256_file(snapshot / relative) != digest:
            raise SnapshotError(f"Metadata SHA-256 mismatch for {relative}")

    summary = manifest["archive_summary"]
    if len(records) != int(summary["file_count"]) or total_bytes != int(summary["total_bytes"]):
        raise SnapshotError("Manifest archive summary does not match verified payload")
    if manifest.get("validation", {}).get("status") != "passed":
        raise SnapshotError("Creation-time benchmark validation was not recorded as passed")
    return {
        "status": "passed",
        "snapshot": str(snapshot),
        "verified_files": len(records),
        "verified_bytes": total_bytes,
        "snapshot_status": manifest.get("status"),
    }


def create_snapshot(snapshot: Path) -> dict[str, Any]:
    snapshot = snapshot.resolve()
    if snapshot.exists():
        raise SnapshotError(
            f"Snapshot already exists and will not be overwritten: {snapshot}. Use --verify-only."
        )
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{snapshot.name}.staging-", dir=snapshot.parent))
    try:
        # Validate before copy, then validate the copied payload independently.
        source_validation = validate_benchmark(PROJECT_ROOT)
        git_state = collect_git_state()
        records = copy_selected_files(staging)
        add_provenance_files(staging, records, git_state)
        payload_validation = validate_benchmark(staging / "payload")
        if source_validation != payload_validation:
            raise SnapshotError("Source and copied-payload benchmark validations differ")
        write_metadata(
            staging,
            records=records,
            validation=payload_validation,
            git_state=git_state,
        )
        if snapshot.exists():
            raise SnapshotError(f"Snapshot appeared during creation and will not be replaced: {snapshot}")
        os.rename(staging, snapshot)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return verify_snapshot(snapshot)


def main() -> int:
    args = parse_args()
    snapshot = args.snapshot_root
    if not snapshot.is_absolute():
        snapshot = PROJECT_ROOT / snapshot
    try:
        result = verify_snapshot(snapshot) if args.verify_only else create_snapshot(snapshot)
    except (SnapshotError, OSError, ValueError, json.JSONDecodeError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Date Created: 2026-07-13
# Date Last Modified: 2026-07-13

"""Run resumable source-grouped BERT augmentation and balancing experiments.

The accepted BERT benchmark remains untouched. This runner reuses its frozen
outer/inner manifests and model configuration while adding synthetic review
texts only inside the corresponding outer-training/inner-training partition.
Validation and test predictions always use original YouTube transcript chunks.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.synthetic.evaluate_synthetic_only_resampling_ablation import balance_training_frame
from src.synthetic.evaluate_synthetic_review_count_sweep import select_complete_perfume_blocks
from src.training.model_benchmark_runner import get_git_commit, load_config, resolve_project_path, stringify_path
from src.training.run_bert_benchmark import (
    BertRunSettings,
    aggregate_chunk_probabilities,
    compute_probability_metrics,
    detect_device,
    load_bert_settings,
    load_deep_text_dataset,
    merge_perfume_probabilities,
    run_finetuned_bert_outer_fold,
    set_global_seed,
    tokenize_chunk_texts,
)
from src.training.source_grouped_folds import SOURCE_GROUP_COLUMN, sha256_file, validate_grouped_inner_manifest
from src.utils.paths import PROJECT_ROOT


TARGETS = ["joy", "sadness", "fear", "trust", "surprise", "anticipation", "disgust"]
REVIEW_COUNTS = [0, 99, 201, 300, 501, 1002, 2001, 2352]
SMOTE_EXCLUSION = (
    "Not applicable to raw-text BERT: SMOTE interpolates continuous feature vectors, "
    "whereas token-ID sequences are discrete and interpolated embeddings would not "
    "correspond to observed review texts."
)


@dataclass(frozen=True)
class ExperimentCondition:
    stage: str
    condition_id: str
    data_setup: str
    strategy: str
    synthetic_source: str
    synthetic_review_count: int
    positive_weighting: bool


def experiment_plan() -> list[ExperimentCondition]:
    review_conditions = [
        ExperimentCondition(
            stage="review_count",
            condition_id=f"weighted_synthetic_reviews_{count}",
            data_setup="youtube_only" if count == 0 else "youtube_plus_selected_784x3",
            strategy="positive_weighted_bce",
            synthetic_source="none" if count == 0 else "synthetic_784x3_subset",
            synthetic_review_count=count,
            positive_weighting=True,
        )
        for count in REVIEW_COUNTS
    ]
    balancing_conditions: list[ExperimentCondition] = []
    for setup, source, count in [
        ("youtube_only", "none", 0),
        ("youtube_plus_synthetic_784x3", "synthetic_784x3", 2352),
        ("youtube_plus_synthetic_784x6", "synthetic_784x6", 4704),
    ]:
        for strategy in ("unbalanced", "hybrid"):
            balancing_conditions.append(
                ExperimentCondition(
                    stage="balancing",
                    condition_id=f"{setup}_{strategy}",
                    data_setup=setup,
                    strategy=strategy,
                    synthetic_source=source,
                    synthetic_review_count=count,
                    positive_weighting=False,
                )
            )
    return review_conditions + balancing_conditions


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_text(value: datetime | None = None) -> str:
    return (value or utc_now()).replace(microsecond=0).isoformat()


def format_duration(seconds: float | int | None) -> str:
    if seconds is None or not np.isfinite(float(seconds)):
        return "unknown"
    seconds = max(0, int(round(float(seconds))))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def stable_json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def normalized_text(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def resolve_immutable_local_checkpoint(configured_checkpoint: str) -> tuple[str, dict[str, Any]]:
    configured_path = Path(configured_checkpoint).expanduser()
    if configured_path.is_dir():
        return str(configured_path.resolve()), {
            "configured_checkpoint": configured_checkpoint,
            "resolved_snapshot": str(configured_path.resolve()),
            "revision": None,
        }

    cache_root = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / ("models--" + configured_checkpoint.replace("/", "--"))
    )
    ref_path = cache_root / "refs" / "main"
    if not ref_path.is_file():
        return configured_checkpoint, {
            "configured_checkpoint": configured_checkpoint,
            "resolved_snapshot": None,
            "revision": None,
        }
    revision = ref_path.read_text(encoding="utf-8").strip()
    snapshot = cache_root / "snapshots" / revision
    required = ["config.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt", "pytorch_model.bin"]
    missing = [name for name in required if not (snapshot / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Cached BERT snapshot {snapshot} lacks files: {missing}")
    return str(snapshot.resolve()), {
        "configured_checkpoint": configured_checkpoint,
        "resolved_snapshot": str(snapshot.resolve()),
        "revision": revision,
        "files": {
            name: {"sha256": sha256_file(snapshot / name), "bytes": (snapshot / name).stat().st_size}
            for name in required
        },
    }


def validate_synthetic_frame(
    path: Path,
    *,
    expected_reviews_per_perfume: int,
    perfume_ids: set[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = pd.read_csv(path, keep_default_na=False)
    required = {
        "synthetic_review_id",
        "perfume_id",
        "Perfume",
        "review_index",
        "Final Transcript",
        *TARGETS,
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Synthetic input {path} lacks columns: {sorted(missing)}")

    frame = frame.copy()
    frame["synthetic_review_id"] = frame["synthetic_review_id"].astype(str).str.strip()
    frame["perfume_id"] = frame["perfume_id"].astype(str).str.strip()
    frame["Final Transcript"] = frame["Final Transcript"].astype(str).str.strip()
    frame["review_index"] = pd.to_numeric(frame["review_index"], errors="raise").astype(int)
    for label in TARGETS:
        frame[label] = pd.to_numeric(frame[label], errors="raise").astype(int)
        if not frame[label].isin([0, 1]).all():
            raise ValueError(f"Synthetic label {label} is not binary in {path}")

    expected_rows = len(perfume_ids) * expected_reviews_per_perfume
    per_perfume = frame.groupby("perfume_id").size()
    review_indices = sorted(frame["review_index"].unique().tolist())
    if len(frame) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows in {path}, found {len(frame)}")
    if set(frame["perfume_id"]) != perfume_ids:
        raise ValueError(f"Synthetic perfume universe does not match the frozen 784 IDs: {path}")
    if per_perfume.nunique() != 1 or int(per_perfume.iloc[0]) != expected_reviews_per_perfume:
        raise ValueError(f"Synthetic rows per perfume are inconsistent in {path}")
    if review_indices != list(range(1, expected_reviews_per_perfume + 1)):
        raise ValueError(f"Unexpected review indices in {path}: {review_indices}")
    if frame["synthetic_review_id"].duplicated().any() or (frame["synthetic_review_id"] == "").any():
        raise ValueError(f"Synthetic review IDs must be unique and non-empty: {path}")
    if (frame["Final Transcript"] == "").any():
        raise ValueError(f"Synthetic transcripts must be non-empty: {path}")

    normalized = frame["Final Transcript"].map(normalized_text)
    duplicate_rows = frame.loc[normalized.duplicated(keep=False), [
        "synthetic_review_id", "perfume_id", "review_index", *TARGETS
    ]].copy()
    cross_perfume_duplicate_groups = 0
    if not duplicate_rows.empty:
        duplicate_rows["normalized_text"] = normalized.loc[duplicate_rows.index]
        cross_perfume_duplicate_groups = int(
            (duplicate_rows.groupby("normalized_text")["perfume_id"].nunique() > 1).sum()
        )
    if cross_perfume_duplicate_groups:
        raise ValueError(f"Synthetic input contains cross-perfume duplicate transcripts: {path}")

    audit = {
        "path": stringify_path(path),
        "sha256": sha256_file(path),
        "rows": int(len(frame)),
        "perfumes": int(frame["perfume_id"].nunique()),
        "reviews_per_perfume": expected_reviews_per_perfume,
        "review_indices": review_indices,
        "unique_normalized_texts": int(normalized.nunique()),
        "within_perfume_duplicate_rows": int(len(duplicate_rows)),
        "cross_perfume_duplicate_groups": cross_perfume_duplicate_groups,
        "positive_support": {label: int(frame[label].sum()) for label in TARGETS},
    }
    return frame.reset_index(drop=True), audit


def validate_three_is_subset_of_six(synthetic_3: pd.DataFrame, synthetic_6: pd.DataFrame) -> None:
    columns = ["synthetic_review_id", "perfume_id", "review_index", "Final Transcript", *TARGETS]
    left = synthetic_3[columns].sort_values("synthetic_review_id").reset_index(drop=True)
    right = synthetic_6.loc[synthetic_6["review_index"].le(3), columns].sort_values(
        "synthetic_review_id"
    ).reset_index(drop=True)
    if not left.equals(right):
        raise ValueError("The 784x3 labeled corpus is not an exact review-1..3 subset of 784x6.")


def build_master_frames(
    perfume_df: pd.DataFrame,
    base_chunk_df: pd.DataFrame,
    synthetic_6: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    base = base_chunk_df.copy().reset_index(drop=True)
    base["unit_id"] = "og::" + base["perfume_id"].astype(str)
    base["unit_type"] = "og_perfume"
    base["source_type"] = "og_youtube"
    base["synthetic_review_id"] = ""
    base["review_index"] = 0
    base["unit_chunk_count"] = base["chunk_count"].astype(int)

    perfume_lookup = perfume_df.set_index("perfume_id")
    synthetic_rows: list[dict[str, Any]] = []
    for row in synthetic_6.itertuples(index=False):
        perfume_id = str(row.perfume_id)
        canonical = perfume_lookup.loc[perfume_id]
        synthetic_review_id = str(row.synthetic_review_id)
        record: dict[str, Any] = {
            "perfume_id": perfume_id,
            "Perfume": str(row.Perfume),
            "fold_id": int(canonical["fold_id"]),
            "chunk_count": 1,
            "chunk_index": int(row.review_index),
            "chunk_text": f"REVIEW: {str(getattr(row, '_18', '') or getattr(row, 'Final_Transcript', '')).strip()}",
            SOURCE_GROUP_COLUMN: str(canonical[SOURCE_GROUP_COLUMN]),
            "source_group_size": int(canonical["source_group_size"]),
            "video_id": synthetic_review_id,
            "transcript_sha256": hashlib.sha256(str(getattr(row, '_18', '') or getattr(row, 'Final_Transcript', '')).encode("utf-8")).hexdigest(),
            "unit_id": f"synthetic::{synthetic_review_id}",
            "unit_type": "synthetic_review",
            "source_type": "synthetic",
            "synthetic_review_id": synthetic_review_id,
            "review_index": int(row.review_index),
            "unit_chunk_count": 1,
        }
        # itertuples sanitizes the spaced column name; use the source frame below
        synthetic_rows.append(record)

    synthetic_chunk_df = pd.DataFrame(synthetic_rows)
    synthetic_chunk_df["chunk_text"] = "REVIEW: " + synthetic_6["Final Transcript"].astype(str).str.strip().to_numpy()
    synthetic_chunk_df["transcript_sha256"] = synthetic_6["Final Transcript"].map(
        lambda value: hashlib.sha256(str(value).encode("utf-8")).hexdigest()
    ).to_numpy()
    for label in TARGETS:
        synthetic_chunk_df[label] = synthetic_6[label].astype(int).to_numpy()

    master = pd.concat([base, synthetic_chunk_df], ignore_index=True, sort=False)
    master["master_row_index"] = np.arange(len(master), dtype=int)
    master["sample_weight"] = 1.0 / master["unit_chunk_count"].astype(float)
    if master["chunk_text"].astype(str).str.strip().eq("").any():
        raise ValueError("Master BERT chunk frame contains empty text.")

    unit_columns = [
        "unit_id", "unit_type", "source_type", "synthetic_review_id", "perfume_id", "Perfume",
        "fold_id", SOURCE_GROUP_COLUMN, "review_index", *TARGETS,
    ]
    unit_df = master[unit_columns].drop_duplicates("unit_id", keep="first").reset_index(drop=True)
    unit_to_indices = {
        str(unit_id): group["master_row_index"].to_numpy(dtype=np.int64)
        for unit_id, group in master.groupby("unit_id", sort=False)
    }
    return master, unit_df, unit_to_indices


def token_cache_fingerprint(master: pd.DataFrame, settings: BertRunSettings) -> str:
    digest = hashlib.sha256()
    for unit_id, text_value in master[["unit_id", "chunk_text"]].itertuples(index=False, name=None):
        digest.update(str(unit_id).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(text_value).encode("utf-8"))
        digest.update(b"\n")
    digest.update(settings.checkpoint.encode("utf-8"))
    digest.update(str(settings.max_length).encode("ascii"))
    return digest.hexdigest()


def load_or_build_token_cache(
    master: pd.DataFrame,
    settings: BertRunSettings,
    *,
    cache_path: Path,
) -> tuple[dict[str, np.ndarray], str]:
    fingerprint = token_cache_fingerprint(master, settings)
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cached:
            actual = str(cached["fingerprint"].item())
            if actual != fingerprint:
                raise ValueError(f"Existing token cache has the wrong fingerprint: {cache_path}")
            arrays = {
                "input_ids": cached["input_ids"],
                "attention_mask": cached["attention_mask"],
                "token_type_ids": cached["token_type_ids"],
            }
        return arrays, fingerprint

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = tokenize_chunk_texts(
        master,
        model_name=settings.checkpoint,
        max_length=settings.max_length,
        local_files_only=settings.local_files_only,
    )
    temporary = cache_path.with_suffix(".tmp.npz")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, fingerprint=np.asarray(fingerprint), **arrays)
    temporary.replace(cache_path)
    return arrays, fingerprint


def logical_pos_weight(units: pd.DataFrame) -> np.ndarray:
    n_rows = float(len(units))
    weights: list[float] = []
    for label in TARGETS:
        positive = float(units[label].sum())
        negative = n_rows - positive
        weights.append(negative / positive if positive > 0 else 1.0)
    return np.asarray(weights, dtype=np.float32)


def expand_units_to_chunk_indices(
    selected_units: pd.DataFrame,
    unit_to_indices: Mapping[str, np.ndarray],
) -> np.ndarray:
    expanded: list[int] = []
    for unit_id in selected_units["unit_id"].astype(str):
        if unit_id not in unit_to_indices:
            raise KeyError(f"No master chunks found for logical unit {unit_id}")
        expanded.extend(int(value) for value in unit_to_indices[unit_id])
    return np.asarray(expanded, dtype=np.int64)


def synthetic_unit_ids_for_condition(
    condition: ExperimentCondition,
    synthetic_3: pd.DataFrame,
    synthetic_6: pd.DataFrame,
) -> set[str]:
    if condition.synthetic_source == "none":
        return set()
    if condition.synthetic_source == "synthetic_784x3_subset":
        selected = select_complete_perfume_blocks(synthetic_3, condition.synthetic_review_count)
    elif condition.synthetic_source == "synthetic_784x3":
        selected = synthetic_3
    elif condition.synthetic_source == "synthetic_784x6":
        selected = synthetic_6
    else:
        raise ValueError(f"Unsupported synthetic source: {condition.synthetic_source}")
    return {f"synthetic::{value}" for value in selected["synthetic_review_id"].astype(str)}


def condition_initial_seconds(condition: ExperimentCondition, device_type: str) -> float:
    if condition.stage == "review_count":
        mps = {
            0: 2400, 99: 2460, 201: 2520, 300: 2580,
            501: 2700, 1002: 3000, 2001: 3600, 2352: 3900,
        }[condition.synthetic_review_count]
    else:
        mps = {
            ("youtube_only", "unbalanced"): 2400,
            ("youtube_only", "hybrid"): 1800,
            ("youtube_plus_synthetic_784x3", "unbalanced"): 3600,
            ("youtube_plus_synthetic_784x3", "hybrid"): 3000,
            ("youtube_plus_synthetic_784x6", "unbalanced"): 4800,
            ("youtube_plus_synthetic_784x6", "hybrid"): 4200,
        }[(condition.data_setup, condition.strategy)]
    return float(mps if device_type == "mps" else mps * 2.2)


def completed_condition_durations(out_root: Path, plan: Sequence[ExperimentCondition]) -> dict[str, float]:
    durations: dict[str, float] = {}
    for condition in plan:
        report_path = out_root / condition.stage / condition.condition_id / "condition_report.json"
        if report_path.is_file():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if report.get("status") == "complete":
                durations[condition.condition_id] = float(report["wall_time_seconds"])
    return durations


def calibrated_estimates(
    out_root: Path,
    plan: Sequence[ExperimentCondition],
    *,
    device_type: str,
) -> tuple[dict[str, float], float]:
    initial = {condition.condition_id: condition_initial_seconds(condition, device_type) for condition in plan}
    completed = completed_condition_durations(out_root, plan)
    ratios = [completed[key] / initial[key] for key in completed if initial[key] > 0]
    calibration = float(np.median(ratios)) if ratios else 1.0
    calibration = float(np.clip(calibration, 0.4, 2.5))
    estimates = {
        condition.condition_id: completed.get(condition.condition_id, initial[condition.condition_id] * calibration)
        for condition in plan
    }
    return estimates, calibration


def progress_payload(
    *,
    out_root: Path,
    plan: Sequence[ExperimentCondition],
    active: ExperimentCondition,
    condition_index: int,
    fold_id: int,
    epoch: int | None,
    condition_started: float,
    device_type: str,
    status: str,
) -> dict[str, Any]:
    estimates, calibration = calibrated_estimates(out_root, plan, device_type=device_type)
    completed = completed_condition_durations(out_root, plan)
    elapsed = time.monotonic() - condition_started
    active_remaining = max(0.0, estimates[active.condition_id] - elapsed)
    remaining_conditions = [item for item in plan[condition_index + 1 :] if item.condition_id not in completed]
    total_remaining = active_remaining + sum(estimates[item.condition_id] for item in remaining_conditions)
    stage_remaining = active_remaining + sum(
        estimates[item.condition_id]
        for item in remaining_conditions
        if item.stage == active.stage
    )
    now = utc_now()
    return {
        "status": status,
        "updated_utc": utc_text(now),
        "device": device_type,
        "stage": active.stage,
        "condition_id": active.condition_id,
        "condition_index": condition_index + 1,
        "condition_total": len(plan),
        "fold": fold_id,
        "fold_total": 5,
        "epoch": epoch,
        "stage_eta_seconds": stage_remaining,
        "stage_eta": format_duration(stage_remaining),
        "stage_estimated_finish_utc": utc_text(now + timedelta(seconds=stage_remaining)),
        "experiment_eta_seconds": total_remaining,
        "experiment_eta": format_duration(total_remaining),
        "experiment_estimated_finish_utc": utc_text(now + timedelta(seconds=total_remaining)),
        "eta_calibration": calibration,
    }


def write_progress(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_write_json(path, payload)
    print(
        "[bert-experiment] "
        f"stage={payload['stage']} condition={payload['condition_index']}/{payload['condition_total']} "
        f"id={payload['condition_id']} fold={payload['fold']}/{payload['fold_total']} "
        f"epoch={payload.get('epoch') or '-'} stage_eta={payload['stage_eta']} "
        f"experiment_eta={payload['experiment_eta']}",
        flush=True,
    )


def fold_fingerprint(
    *,
    condition: ExperimentCondition,
    fold_id: int,
    provenance: Mapping[str, Any],
    settings: BertRunSettings,
    device_type: str,
) -> str:
    return stable_json_sha256(
        {
            "condition": asdict(condition),
            "fold_id": fold_id,
            "provenance": dict(provenance),
            "settings": asdict(settings),
            "device_type": device_type,
        }
    )


def save_fold_artifact(
    path: Path,
    *,
    fingerprint: str,
    perfume_ids: Sequence[str],
    probabilities: np.ndarray,
    summary: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            fingerprint=np.asarray(fingerprint),
            perfume_id=np.asarray(list(perfume_ids), dtype=str),
            probabilities=np.asarray(probabilities, dtype=np.float32),
            summary_json=np.asarray(json.dumps(summary, allow_nan=False)),
        )
    temporary.replace(path)


def load_fold_artifact(path: Path, *, expected_fingerprint: str) -> tuple[list[str], np.ndarray, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as artifact:
        actual = str(artifact["fingerprint"].item())
        if actual != expected_fingerprint:
            raise ValueError(f"Stale/incompatible fold checkpoint: {path}")
        perfume_ids = artifact["perfume_id"].astype(str).tolist()
        probabilities = artifact["probabilities"].astype(np.float32)
        summary = json.loads(str(artifact["summary_json"].item()))
    return perfume_ids, probabilities, summary


def condition_report_paths(out_root: Path, condition: ExperimentCondition) -> dict[str, Path]:
    root = out_root / condition.stage / condition.condition_id
    return {
        "root": root,
        "report": root / "condition_report.json",
        "predictions": root / "oof_predictions.csv",
        "per_label": root / "per_label_metrics.csv",
    }


def run_condition(
    *,
    condition: ExperimentCondition,
    condition_index: int,
    plan: Sequence[ExperimentCondition],
    out_root: Path,
    perfume_df: pd.DataFrame,
    master_chunk_df: pd.DataFrame,
    unit_df: pd.DataFrame,
    unit_to_indices: Mapping[str, np.ndarray],
    tokenized_inputs: Mapping[str, np.ndarray],
    synthetic_3: pd.DataFrame,
    synthetic_6: pd.DataFrame,
    inner_manifest: pd.DataFrame,
    settings: BertRunSettings,
    threshold: float,
    seed: int,
    device: torch.device,
    provenance: Mapping[str, Any],
    resume: bool,
    only_fold: int | None = None,
    smoke_max_train_units: int | None = None,
) -> dict[str, Any]:
    paths = condition_report_paths(out_root, condition)
    if paths["report"].is_file() and resume and only_fold is None:
        existing = json.loads(paths["report"].read_text(encoding="utf-8"))
        if existing.get("status") == "complete" and existing.get("condition_fingerprint") == stable_json_sha256(
            {"condition": asdict(condition), "provenance": dict(provenance), "settings": asdict(settings), "device": device.type}
        ):
            print(f"[bert-experiment] skip complete condition={condition.condition_id}", flush=True)
            return existing

    condition_started_wall = utc_now()
    condition_started = time.monotonic()
    allowed_synthetic = synthetic_unit_ids_for_condition(condition, synthetic_3, synthetic_6)
    oof_prob = np.full((len(perfume_df), len(TARGETS)), np.nan, dtype=np.float32)
    fold_summaries: list[dict[str, Any]] = []
    perfume_index = {value: idx for idx, value in enumerate(perfume_df["perfume_id"].astype(str))}
    fold_ids = [only_fold] if only_fold is not None else [1, 2, 3, 4, 5]
    progress_path = out_root / "progress.json"

    for fold_id in fold_ids:
        expected_fingerprint = fold_fingerprint(
            condition=condition,
            fold_id=fold_id,
            provenance=provenance,
            settings=settings,
            device_type=device.type,
        )
        fold_path = paths["root"] / "folds" / f"fold_{fold_id}.npz"
        test_perfume_df = perfume_df.loc[perfume_df["fold_id"].eq(fold_id)].sort_values("perfume_id").reset_index(drop=True)
        inner = inner_manifest.loc[inner_manifest["outer_fold_id"].eq(fold_id)]
        train_ids = set(inner.loc[inner["inner_role"].eq("train"), "perfume_id"].astype(str))
        val_ids = set(inner.loc[inner["inner_role"].eq("val"), "perfume_id"].astype(str))
        test_ids = set(test_perfume_df["perfume_id"].astype(str))
        if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
            raise ValueError(f"Perfume leakage in fold {fold_id}")
        train_groups = set(perfume_df.loc[perfume_df["perfume_id"].isin(train_ids), SOURCE_GROUP_COLUMN].astype(str))
        val_groups = set(perfume_df.loc[perfume_df["perfume_id"].isin(val_ids), SOURCE_GROUP_COLUMN].astype(str))
        test_groups = set(test_perfume_df[SOURCE_GROUP_COLUMN].astype(str))
        if train_groups & val_groups or train_groups & test_groups or val_groups & test_groups:
            raise ValueError(f"Source-group leakage in fold {fold_id}")

        if resume and fold_path.is_file():
            saved_ids, saved_prob, fold_summary = load_fold_artifact(
                fold_path, expected_fingerprint=expected_fingerprint
            )
            if saved_ids != test_perfume_df["perfume_id"].astype(str).tolist():
                raise ValueError(f"Saved fold perfume order is invalid: {fold_path}")
            for row_idx, perfume_id in enumerate(saved_ids):
                oof_prob[perfume_index[perfume_id], :] = saved_prob[row_idx, :]
            fold_summaries.append(fold_summary)
            print(f"[bert-experiment] resume condition={condition.condition_id} fold={fold_id}/5", flush=True)
            continue

        candidate = unit_df.loc[
            unit_df["perfume_id"].astype(str).isin(train_ids)
            & (
                unit_df["unit_type"].eq("og_perfume")
                | unit_df["unit_id"].astype(str).isin(allowed_synthetic)
            )
        ].copy().reset_index(drop=True)
        if candidate.empty:
            raise ValueError(f"No training units for {condition.condition_id} fold {fold_id}")
        if candidate.loc[candidate["unit_type"].eq("synthetic_review"), "perfume_id"].isin(val_ids | test_ids).any():
            raise ValueError(f"Synthetic validation/test perfume entered training in fold {fold_id}")

        before_counts = {label: int(candidate[label].sum()) for label in TARGETS}
        balance_report: dict[str, Any] | None = None
        selected = candidate
        if condition.strategy == "hybrid":
            selected, balance_report = balance_training_frame(
                candidate,
                TARGETS,
                seed=seed + fold_id * 1009,
            )
        if smoke_max_train_units is not None:
            selected = selected.head(smoke_max_train_units).copy().reset_index(drop=True)
        train_indices = expand_units_to_chunk_indices(selected, unit_to_indices)
        pos_weight = logical_pos_weight(candidate) if condition.positive_weighting else np.ones(len(TARGETS), dtype=np.float32)

        val_perfume_df = perfume_df.loc[perfume_df["perfume_id"].isin(val_ids)].sort_values("perfume_id").reset_index(drop=True)
        train_perfume_df = perfume_df.loc[perfume_df["perfume_id"].isin(train_ids)].sort_values("perfume_id").reset_index(drop=True)
        val_mask = (
            master_chunk_df["unit_type"].eq("og_perfume")
            & master_chunk_df["perfume_id"].astype(str).isin(val_ids)
        ).to_numpy()
        test_mask = (
            master_chunk_df["unit_type"].eq("og_perfume")
            & master_chunk_df["perfume_id"].astype(str).isin(test_ids)
        ).to_numpy()
        train_mask = np.zeros(len(master_chunk_df), dtype=bool)

        initial_payload = progress_payload(
            out_root=out_root,
            plan=plan,
            active=condition,
            condition_index=condition_index,
            fold_id=fold_id,
            epoch=None,
            condition_started=condition_started,
            device_type=device.type,
            status="running",
        )
        write_progress(progress_path, initial_payload)

        def on_epoch(epoch_record: dict[str, Any]) -> None:
            payload = progress_payload(
                out_root=out_root,
                plan=plan,
                active=condition,
                condition_index=condition_index,
                fold_id=fold_id,
                epoch=int(epoch_record["epoch"]),
                condition_started=condition_started,
                device_type=device.type,
                status="running",
            )
            write_progress(progress_path, payload)

        fold_started = time.monotonic()
        test_chunk_probs, _, fold_summary = run_finetuned_bert_outer_fold(
            tokenized_inputs=tokenized_inputs,
            chunk_df=master_chunk_df,
            train_chunk_mask=train_mask,
            val_chunk_mask=val_mask,
            test_chunk_mask=test_mask,
            train_perfume_df=train_perfume_df,
            val_perfume_df=val_perfume_df,
            test_perfume_df=test_perfume_df,
            target_cols=TARGETS,
            settings=settings,
            threshold=threshold,
            seed=seed,
            device=device,
            outer_fold_id=fold_id,
            train_chunk_indices=train_indices,
            sample_weights_override=master_chunk_df["sample_weight"].to_numpy(dtype=np.float32),
            pos_weight_override=pos_weight,
            progress_callback=on_epoch,
        )
        test_chunk_df = master_chunk_df.loc[test_mask].copy().reset_index(drop=True)
        aggregated = aggregate_chunk_probabilities(test_chunk_df, test_chunk_probs, target_cols=TARGETS)
        merged = merge_perfume_probabilities(test_perfume_df, aggregated, target_cols=TARGETS)
        fold_prob = merged[[f"prob_{label}" for label in TARGETS]].to_numpy(dtype=np.float32)
        fold_summary.update(
            {
                "condition_id": condition.condition_id,
                "fold_wall_time_seconds": time.monotonic() - fold_started,
                "logical_train_units_before_balance": int(len(candidate)),
                "logical_train_units_after_balance": int(len(selected)),
                "logical_train_unique_units_after_balance": int(selected["unit_id"].nunique()),
                "synthetic_train_units_before_balance": int(candidate["unit_type"].eq("synthetic_review").sum()),
                "synthetic_train_units_after_balance": int(selected["unit_type"].eq("synthetic_review").sum()),
                "logical_positive_counts_before_balance": before_counts,
                "logical_positive_counts_after_balance": {label: int(selected[label].sum()) for label in TARGETS},
                "balance_report": balance_report,
            }
        )
        save_fold_artifact(
            fold_path,
            fingerprint=expected_fingerprint,
            perfume_ids=test_perfume_df["perfume_id"].astype(str).tolist(),
            probabilities=fold_prob,
            summary=fold_summary,
        )
        for row_idx, perfume_id in enumerate(test_perfume_df["perfume_id"].astype(str)):
            oof_prob[perfume_index[perfume_id], :] = fold_prob[row_idx, :]
        fold_summaries.append(fold_summary)
        gc.collect()
        if device.type == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()

    if only_fold is not None:
        return {
            "status": "smoke_complete",
            "condition_id": condition.condition_id,
            "folds": fold_summaries,
            "wall_time_seconds": time.monotonic() - condition_started,
        }
    if not np.isfinite(oof_prob).all():
        raise ValueError(f"Condition {condition.condition_id} has incomplete OOF probabilities")

    y_true = perfume_df[TARGETS].to_numpy(dtype=int)
    per_label_rows, metrics = compute_probability_metrics(
        y_true=y_true,
        y_prob=oof_prob,
        target_cols=TARGETS,
        threshold=threshold,
    )
    prediction = (oof_prob >= threshold).astype(int)
    prediction_columns = ["perfume_id", "fold_id", "Perfume", SOURCE_GROUP_COLUMN, "source_group_size"]
    predictions = perfume_df[prediction_columns].copy()
    for label_idx, label in enumerate(TARGETS):
        predictions[f"true_{label}"] = y_true[:, label_idx]
        predictions[f"prob_{label}"] = oof_prob[:, label_idx]
        predictions[f"pred_{label}"] = prediction[:, label_idx]
    paths["root"].mkdir(parents=True, exist_ok=True)
    predictions.to_csv(paths["predictions"], index=False)
    pd.DataFrame(per_label_rows).to_csv(paths["per_label"], index=False)

    condition_fingerprint = stable_json_sha256(
        {"condition": asdict(condition), "provenance": dict(provenance), "settings": asdict(settings), "device": device.type}
    )
    report = {
        "status": "complete",
        "condition": asdict(condition),
        "condition_fingerprint": condition_fingerprint,
        "started_utc": utc_text(condition_started_wall),
        "completed_utc": utc_text(),
        "wall_time_seconds": time.monotonic() - condition_started,
        "device": device.type,
        "threshold": threshold,
        "seed": seed,
        "n_eval_perfumes": int(len(perfume_df)),
        "n_source_groups": int(perfume_df[SOURCE_GROUP_COLUMN].nunique()),
        "macro_f1": float(metrics["macro_f1"]),
        "macro_roc_auc": float(metrics["macro_roc_auc"]),
        "macro_pr_auc": float(metrics["macro_pr_auc"]),
        "secondary_metrics": {
            key: float(metrics[key])
            for key in ("micro_f1", "subset_accuracy", "hamming_loss", "label_cardinality")
        },
        "folds": fold_summaries,
        "provenance": dict(provenance),
        "artifacts": {
            "predictions": stringify_path(paths["predictions"]),
            "per_label": stringify_path(paths["per_label"]),
        },
    }
    atomic_write_json(paths["report"], report)
    print(
        f"[bert-experiment] complete condition={condition.condition_id} "
        f"macro_f1={metrics['macro_f1']:.4f} roc_auc={metrics['macro_roc_auc']:.4f} "
        f"wall={format_duration(report['wall_time_seconds'])}",
        flush=True,
    )
    return report


def write_summaries(out_root: Path, plan: Sequence[ExperimentCondition], provenance: Mapping[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    for condition in plan:
        path = condition_report_paths(out_root, condition)["report"]
        if not path.is_file():
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("status") != "complete":
            continue
        rows.append(
            {
                "stage": condition.stage,
                "condition_id": condition.condition_id,
                "data_setup": condition.data_setup,
                "strategy": condition.strategy,
                "synthetic_reviews": condition.synthetic_review_count,
                "positive_weighting": condition.positive_weighting,
                "macro_f1": report["macro_f1"],
                "macro_roc_auc": report["macro_roc_auc"],
                "macro_pr_auc": report["macro_pr_auc"],
                "wall_time_seconds": report["wall_time_seconds"],
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(out_root / "experiment_summary.csv", index=False)
    payload = {
        "status": "complete" if len(rows) == len(plan) else "partial",
        "completed_conditions": len(rows),
        "planned_conditions": len(plan),
        "threshold": 0.3,
        "smote": {"status": "not_applicable", "reason": SMOTE_EXCLUSION},
        "provenance": dict(provenance),
        "conditions": rows,
    }
    atomic_write_json(out_root / "experiment_report.json", payload)

    lines = [
        "# Source-Grouped BERT Experiments",
        "",
        f"Status: **{payload['status']}** ({len(rows)}/{len(plan)} conditions complete).",
        "",
        "All held-out evaluation uses the 784 original YouTube perfumes and the repaired source-component folds.",
        "Synthetic reviews for outer-test and inner-validation perfume IDs are excluded from training.",
        "",
        "| Stage | Condition | Synthetic reviews | Macro-F1 | Macro ROC-AUC | Macro PR-AUC | Wall time |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['stage']} | {row['condition_id']} | {row['synthetic_reviews']} | "
            f"{row['macro_f1']:.4f} | {row['macro_roc_auc']:.4f} | {row['macro_pr_auc']:.4f} | "
            f"{format_duration(row['wall_time_seconds'])} |"
        )
    lines.extend(["", f"SMOTE: **not applicable**. {SMOTE_EXCLUSION}", ""])
    (out_root / "EXPERIMENT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def validate_weighted_control_against_accepted_baseline(
    *,
    out_root: Path,
    accepted_predictions_path: Path,
    probability_atol: float = 1e-6,
) -> dict[str, Any]:
    control_path = (
        out_root
        / "review_count"
        / "weighted_synthetic_reviews_0"
        / "oof_predictions.csv"
    )
    if not control_path.is_file() or not accepted_predictions_path.is_file():
        raise FileNotFoundError("Control-gate prediction input is missing.")
    control = pd.read_csv(control_path)
    accepted = pd.read_csv(accepted_predictions_path)
    identity_columns = ["perfume_id", "fold_id", *[f"true_{label}" for label in TARGETS]]
    if not control[identity_columns].equals(accepted[identity_columns]):
        raise ValueError("The CPU control and accepted baseline do not have identical rows/folds/labels.")
    probability_columns = [f"prob_{label}" for label in TARGETS]
    absolute = np.abs(
        control[probability_columns].to_numpy(dtype=float)
        - accepted[probability_columns].to_numpy(dtype=float)
    )
    max_absolute_difference = float(absolute.max())
    gate = {
        "status": "pass" if max_absolute_difference <= probability_atol else "fail",
        "accepted_predictions": stringify_path(accepted_predictions_path),
        "control_predictions": stringify_path(control_path),
        "probability_atol": probability_atol,
        "max_absolute_probability_difference": max_absolute_difference,
        "rows": int(len(control)),
    }
    atomic_write_json(out_root / "weighted_control_validation.json", gate)
    if gate["status"] != "pass":
        raise RuntimeError(
            "The zero-synthetic CPU control did not reproduce the accepted BERT baseline: "
            f"max probability difference={max_absolute_difference:.8g}."
        )
    print(
        "[bert-experiment] accepted-baseline control gate passed "
        f"max_prob_diff={max_absolute_difference:.3g}",
        flush=True,
    )
    return gate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "evaluation_config_gemini_browser_og784_source_grouped_v1.yaml",
    )
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "gemini_browser_og784_source_grouped_v1" / "model_benchmark" / "datasets" / "perfume_text_dataset_v1.jsonl",
    )
    parser.add_argument(
        "--synthetic-3-csv",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "synthetic_review_level_tests" / "emotion_rich_v2_3reviews" / "supervised_eval_gemini_browser" / "review_level_features_labeled_g25pro.csv",
    )
    parser.add_argument(
        "--synthetic-6-csv",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "synthetic_review_level_tests" / "emotion_rich_v2_6reviews" / "supervised_eval_gemini_browser" / "review_level_features_labeled_g25pro.csv",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "bert_source_grouped_experiments_v1",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stages", nargs="+", choices=["review_count", "balancing"])
    parser.add_argument("--conditions", nargs="+")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-fold", type=int, default=5, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--smoke-max-train-units", type=int, default=64)
    parser.add_argument(
        "--accepted-baseline-predictions",
        type=Path,
        default=(
            PROJECT_ROOT
            / "outputs"
            / "gemini_browser_og784_source_grouped_v1"
            / "model_benchmark"
            / "predictions"
            / "bert_oof.csv"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    configured_targets = list(config["targets"]["official_emotion_columns"])
    if configured_targets != TARGETS:
        raise ValueError(f"Unexpected configured target order: {configured_targets}")
    settings = load_bert_settings(config)
    resolved_checkpoint, checkpoint_provenance = resolve_immutable_local_checkpoint(settings.checkpoint)
    settings = replace(settings, checkpoint=resolved_checkpoint)
    if args.smoke_test:
        settings = replace(settings, max_epochs=1, patience=1)
    threshold = float(config["evaluation"]["probability_threshold"])
    seed = int(config["seeds"]["data_seed"])
    set_global_seed(seed)
    device = detect_device(args.device)

    outer_path = resolve_project_path(config["splits"]["outer_cv"]["manifest_path"])
    inner_path = resolve_project_path(config["splits"]["inner_validation"]["manifest_path"])
    perfume_df, base_chunk_df = load_deep_text_dataset(
        input_jsonl=args.input_jsonl,
        fold_manifest_path=outer_path,
        target_cols=TARGETS,
        excluded_target_cols=dict(config["targets"]["excluded_emotion_columns"]),
        canonical_labels_path=resolve_project_path(config["data"]["canonical_label_csv"]),
    )
    inner_manifest = pd.read_csv(inner_path)
    validate_grouped_inner_manifest(
        inner_manifest,
        perfume_df,
        unit_id_col="perfume_id",
        group_col=SOURCE_GROUP_COLUMN,
        fold_id_col="fold_id",
    )
    perfume_ids = set(perfume_df["perfume_id"].astype(str))
    synthetic_3, synthetic_3_audit = validate_synthetic_frame(
        args.synthetic_3_csv,
        expected_reviews_per_perfume=3,
        perfume_ids=perfume_ids,
    )
    synthetic_6, synthetic_6_audit = validate_synthetic_frame(
        args.synthetic_6_csv,
        expected_reviews_per_perfume=6,
        perfume_ids=perfume_ids,
    )
    validate_three_is_subset_of_six(synthetic_3, synthetic_6)
    master_chunk_df, unit_df, unit_to_indices = build_master_frames(
        perfume_df, base_chunk_df, synthetic_6
    )

    args.out_root.mkdir(parents=True, exist_ok=True)
    tokenized_inputs, cache_fingerprint = load_or_build_token_cache(
        master_chunk_df,
        settings,
        cache_path=args.out_root / "cache" / "master_tokens.npz",
    )
    script_path = Path(__file__).resolve()
    base_runner_path = PROJECT_ROOT / "src" / "training" / "run_bert_benchmark.py"
    provenance = {
        "git_commit_at_launch": get_git_commit(),
        "config": {"path": stringify_path(args.config), "sha256": sha256_file(args.config)},
        "base_jsonl": {"path": stringify_path(args.input_jsonl), "sha256": sha256_file(args.input_jsonl)},
        "synthetic_3": synthetic_3_audit,
        "synthetic_6": synthetic_6_audit,
        "outer_manifest": {"path": stringify_path(outer_path), "sha256": sha256_file(outer_path)},
        "inner_manifest": {"path": stringify_path(inner_path), "sha256": sha256_file(inner_path)},
        "runner_sha256": sha256_file(script_path),
        "base_bert_runner_sha256": sha256_file(base_runner_path),
        "checkpoint": checkpoint_provenance,
        "token_cache_fingerprint": cache_fingerprint,
        "master_chunks": int(len(master_chunk_df)),
        "logical_units": int(len(unit_df)),
        "smote_exclusion": SMOTE_EXCLUSION,
    }
    atomic_write_json(args.out_root / "experiment_plan.json", {
        "created_utc": utc_text(),
        "device": device.type,
        "settings": asdict(settings),
        "threshold": threshold,
        "seed": seed,
        "conditions": [asdict(item) for item in experiment_plan()],
        "provenance": provenance,
    })

    full_plan = experiment_plan()
    selected_plan = [
        condition
        for condition in full_plan
        if (not args.stages or condition.stage in args.stages)
        and (not args.conditions or condition.condition_id in args.conditions)
    ]
    if not selected_plan:
        raise ValueError("No experiment conditions were selected.")

    print(
        f"[bert-experiment] device={device.type} conditions={len(selected_plan)} "
        f"master_chunks={len(master_chunk_df)} logical_units={len(unit_df)}",
        flush=True,
    )
    if args.smoke_test:
        smoke_condition = selected_plan[0]
        smoke_root = args.out_root / "smoke"
        result = run_condition(
            condition=smoke_condition,
            condition_index=0,
            plan=[smoke_condition],
            out_root=smoke_root,
            perfume_df=perfume_df,
            master_chunk_df=master_chunk_df,
            unit_df=unit_df,
            unit_to_indices=unit_to_indices,
            tokenized_inputs=tokenized_inputs,
            synthetic_3=synthetic_3,
            synthetic_6=synthetic_6,
            inner_manifest=inner_manifest,
            settings=settings,
            threshold=threshold,
            seed=seed,
            device=device,
            provenance=provenance,
            resume=False,
            only_fold=args.smoke_fold,
            smoke_max_train_units=args.smoke_max_train_units,
        )
        atomic_write_json(smoke_root / "smoke_report.json", result)
        print(f"[bert-experiment] smoke complete wall={format_duration(result['wall_time_seconds'])}", flush=True)
        return

    for condition_index, condition in enumerate(selected_plan):
        run_condition(
            condition=condition,
            condition_index=condition_index,
            plan=selected_plan,
            out_root=args.out_root,
            perfume_df=perfume_df,
            master_chunk_df=master_chunk_df,
            unit_df=unit_df,
            unit_to_indices=unit_to_indices,
            tokenized_inputs=tokenized_inputs,
            synthetic_3=synthetic_3,
            synthetic_6=synthetic_6,
            inner_manifest=inner_manifest,
            settings=settings,
            threshold=threshold,
            seed=seed,
            device=device,
            provenance=provenance,
            resume=args.resume,
        )
        if condition.condition_id == "weighted_synthetic_reviews_0":
            validate_weighted_control_against_accepted_baseline(
                out_root=args.out_root,
                accepted_predictions_path=args.accepted_baseline_predictions,
            )
        write_summaries(args.out_root, selected_plan, provenance)

    write_summaries(args.out_root, selected_plan, provenance)
    atomic_write_json(args.out_root / "progress.json", {
        "status": "complete",
        "updated_utc": utc_text(),
        "device": device.type,
        "completed_conditions": len(selected_plan),
        "condition_total": len(selected_plan),
        "stage_eta": "0s",
        "experiment_eta": "0s",
    })
    print("[bert-experiment] all selected conditions complete", flush=True)


if __name__ == "__main__":
    main()

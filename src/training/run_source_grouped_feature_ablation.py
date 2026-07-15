# Date Created: 2026-07-15

"""Run the Phase 5 supervised feature-family ablation.

This is a supplemental analysis.  It never writes into the accepted five-model
benchmark root.  Fifteen balanced Logistic Regression variants reuse the
frozen seven-label dataset and exact-source-grouped outer folds.  The runner
also produces paired source-component bootstrap intervals, held-out grouped
permutation importance, and standardized coefficient-stability diagnostics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/codex-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.training.model_benchmark_runner import (
    collect_package_versions,
    get_git_commit,
    load_config,
    select_feature_columns,
    stringify_path,
    validate_dataset,
    validate_frozen_input_hash,
    validate_source_group_audit,
)
from src.training.run_logistic_regression_benchmark import MODEL_SPEC
from src.training.source_grouped_folds import (
    OFFICIAL_LABELS,
    SOURCE_GROUP_COLUMN,
    load_and_validate_outer_fold_manifest,
    sha256_file,
)
from src.utils.paths import PROJECT_ROOT


THRESHOLD = 0.3
SEED = 42
BOOTSTRAP_REPS = 2000
PERMUTATION_REPEATS = 30
CONFIDENCE_LEVEL = 0.95
FULL_VARIANT = "full_hybrid"
TFIDF_REFERENCE = "tfidf_logistic_regression_reference"

FAMILY_PREFIXES: tuple[tuple[str, str], ...] = (
    ("NRC", "NRC_"),
    ("VAD", "VAD_"),
    ("Hartmann", "Hartmann_"),
    ("Savani", "Savani_"),
    ("GoEmotions", "GoEmotions_"),
    ("Cardiff", "Cardiff_"),
)
EXPECTED_FAMILY_COUNTS = {
    "NRC": 10,
    "VAD": 3,
    "Hartmann": 7,
    "Savani": 6,
    "GoEmotions": 28,
    "Cardiff": 4,
}
TRANSFORMER_FAMILIES = ("Hartmann", "Savani", "GoEmotions", "Cardiff")

VARIANT_LABELS = {
    "full_hybrid": "Full hybrid",
    "nrc_only": "NRC only",
    "vad_only": "VAD only",
    "lexicon_only": "NRC + VAD",
    "hartmann_only": "Hartmann only",
    "savani_only": "Savani only",
    "goemotions_only": "GoEmotions only",
    "cardiff_only": "Cardiff only",
    "transformer_only": "All transformer features",
    "no_nrc": "Full minus NRC",
    "no_vad": "Full minus VAD",
    "no_hartmann": "Full minus Hartmann",
    "no_savani": "Full minus Savani",
    "no_goemotions": "Full minus GoEmotions",
    "no_cardiff": "Full minus Cardiff",
}

DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "config/evaluation_config_gemini_browser_og784_source_grouped_v1.yaml"
)
DEFAULT_FEATURE_CSV = (
    PROJECT_ROOT
    / "outputs/gemini_browser_og784/analysis_out/gemini_browser_og784_feature_labels.csv"
)
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "outputs/gemini_browser_og784_source_grouped_v1/model_benchmark/tables/fold_assignments.csv"
)
DEFAULT_ACCEPTED_OOF = (
    PROJECT_ROOT
    / "outputs/gemini_browser_og784_source_grouped_v1/model_benchmark/predictions/logistic_regression_oof.csv"
)
DEFAULT_ACCEPTED_METRICS = (
    PROJECT_ROOT
    / "outputs/gemini_browser_og784_source_grouped_v1/model_benchmark/metrics/logistic_regression.json"
)
DEFAULT_TFIDF_OOF = (
    PROJECT_ROOT
    / "outputs/tfidf_logistic_regression_source_grouped_v1/tfidf_logistic_regression_oof.csv"
)
DEFAULT_TFIDF_METRICS = (
    PROJECT_ROOT / "outputs/tfidf_logistic_regression_source_grouped_v1/metrics.json"
)
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs/source_grouped_feature_ablation_v1"
DEFAULT_ACCEPTED_EXPERIMENT_ROOT = (
    PROJECT_ROOT / "outputs/gemini_browser_og784_source_grouped_v1"
)
DEFAULT_ACCEPTED_BENCHMARK_ROOT = DEFAULT_ACCEPTED_EXPERIMENT_ROOT / "model_benchmark"
DEFAULT_FEATURE_BRANCH_ROOT = PROJECT_ROOT / "outputs/gemini_browser_og784"
DEFAULT_TFIDF_ROOT = PROJECT_ROOT / "outputs/tfidf_logistic_regression_source_grouped_v1"
DEFAULT_ACCEPTED_CHECKSUMS = DEFAULT_ACCEPTED_EXPERIMENT_ROOT / "SHA256SUMS"

CODE_PROVENANCE_PATHS = OrderedDict(
    [
        ("phase5_runner", Path(__file__).resolve()),
        (
            "logistic_regression_runner",
            PROJECT_ROOT / "src/training/run_logistic_regression_benchmark.py",
        ),
        (
            "benchmark_runner",
            PROJECT_ROOT / "src/training/model_benchmark_runner.py",
        ),
        (
            "source_grouped_folds",
            PROJECT_ROOT / "src/training/source_grouped_folds.py",
        ),
    ]
)


def _paths_collide(candidate: Path, protected: Path) -> bool:
    candidate = candidate.resolve()
    protected = protected.resolve()
    return (
        candidate == protected
        or protected in candidate.parents
        or candidate in protected.parents
    )


def prepare_output_root(out_dir: Path, *, overwrite: bool) -> Path:
    """Create a clean supplemental root without risking accepted inputs."""

    out_dir = Path(out_dir).resolve()
    for protected in (
        DEFAULT_ACCEPTED_EXPERIMENT_ROOT,
        DEFAULT_FEATURE_BRANCH_ROOT,
        DEFAULT_TFIDF_ROOT,
    ):
        if _paths_collide(out_dir, protected):
            raise ValueError(
                f"Phase 5 output root collides with a protected input root: "
                f"{out_dir} versus {protected.resolve()}"
            )
    if out_dir.exists() and any(out_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Phase 5 output root is non-empty; pass --overwrite to replace "
                f"only its checksum-owned files: {out_dir}"
            )
        checksum_path = out_dir / "SHA256SUMS"
        if not checksum_path.is_file():
            raise ValueError(
                f"Refusing to overwrite a non-empty root without SHA256SUMS: {out_dir}"
            )
        owned = {checksum_path.resolve()}
        for line in checksum_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            digest, relative = line.split("  ", 1)
            path = (out_dir / relative).resolve()
            if out_dir not in path.parents:
                raise ValueError(f"Unsafe path in existing SHA256SUMS: {relative}")
            if not path.is_file() or sha256_file(path) != digest:
                raise ValueError(
                    "Refusing to overwrite a root whose current files do not "
                    f"match SHA256SUMS: {relative}"
                )
            owned.add(path)
        existing = {path.resolve() for path in out_dir.rglob("*") if path.is_file()}
        extras = sorted(str(path) for path in existing - owned)
        missing = sorted(str(path) for path in owned - existing)
        if extras or missing:
            raise ValueError(
                "Refusing to overwrite a root whose files differ from its checksum "
                f"ownership set: extras={extras}, missing={missing}"
            )
        for path in sorted(existing, key=lambda value: len(value.parts), reverse=True):
            path.unlink()
        for directory in sorted(
            (path for path in out_dir.rglob("*") if path.is_dir()),
            key=lambda value: len(value.parts),
            reverse=True,
        ):
            directory.rmdir()
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def validate_canonical_run_configuration(
    *,
    out_dir: Path,
    config_path: Path,
    feature_csv: Path,
    manifest_csv: Path,
    accepted_oof_csv: Path,
    accepted_metrics_json: Path,
    tfidf_oof_csv: Path,
    tfidf_metrics_json: Path,
    bootstrap_reps: int,
    permutation_repeats: int,
    seed: int,
    confidence_level: float,
    enforce_expected_schema: bool,
) -> None:
    """Keep the versioned canonical root tied to one immutable design."""

    if Path(out_dir).resolve() != DEFAULT_OUT_DIR.resolve():
        return
    expected_paths = {
        "config_path": DEFAULT_CONFIG,
        "feature_csv": DEFAULT_FEATURE_CSV,
        "manifest_csv": DEFAULT_MANIFEST,
        "accepted_oof_csv": DEFAULT_ACCEPTED_OOF,
        "accepted_metrics_json": DEFAULT_ACCEPTED_METRICS,
        "tfidf_oof_csv": DEFAULT_TFIDF_OOF,
        "tfidf_metrics_json": DEFAULT_TFIDF_METRICS,
    }
    supplied_paths = {
        "config_path": config_path,
        "feature_csv": feature_csv,
        "manifest_csv": manifest_csv,
        "accepted_oof_csv": accepted_oof_csv,
        "accepted_metrics_json": accepted_metrics_json,
        "tfidf_oof_csv": tfidf_oof_csv,
        "tfidf_metrics_json": tfidf_metrics_json,
    }
    mismatches = [
        name
        for name, expected in expected_paths.items()
        if Path(supplied_paths[name]).resolve() != expected.resolve()
    ]
    if mismatches:
        raise ValueError(
            "The canonical Phase 5 root requires canonical input paths; changed: "
            + ", ".join(mismatches)
        )
    expected_parameters = {
        "bootstrap_reps": BOOTSTRAP_REPS,
        "permutation_repeats": PERMUTATION_REPEATS,
        "seed": SEED,
        "confidence_level": CONFIDENCE_LEVEL,
        "enforce_expected_schema": True,
    }
    supplied_parameters = {
        "bootstrap_reps": bootstrap_reps,
        "permutation_repeats": permutation_repeats,
        "seed": seed,
        "confidence_level": confidence_level,
        "enforce_expected_schema": enforce_expected_schema,
    }
    changed = [
        name
        for name, expected in expected_parameters.items()
        if supplied_parameters[name] != expected
    ]
    if changed:
        raise ValueError(
            "The canonical Phase 5 root requires frozen analysis parameters; "
            "changed: " + ", ".join(changed)
        )


@dataclass
class PreparedInputs:
    data: pd.DataFrame
    target_columns: list[str]
    feature_columns: list[str]
    families: OrderedDict[str, list[str]]
    variants: OrderedDict[str, list[str]]
    accepted_oof: pd.DataFrame
    accepted_metrics: dict[str, Any]
    tfidf_oof: pd.DataFrame
    tfidf_metrics: dict[str, Any]
    hashes: dict[str, str]
    config: dict[str, Any]


@dataclass
class VariantRun:
    variant: str
    feature_columns: list[str]
    oof: pd.DataFrame
    summary: dict[str, Any]
    per_label: pd.DataFrame
    fold_diagnostics: pd.DataFrame
    fitted_models: dict[tuple[int, str], Any]
    constant_probabilities: dict[tuple[int, str], float]
    coefficient_rows: list[dict[str, Any]]


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, Path):
        return stringify_path(value)
    return value


def _safe_roc_auc(
    truth: np.ndarray, probability: np.ndarray, sample_weight: np.ndarray | None = None
) -> float | None:
    if sample_weight is None:
        positive = int((truth == 1).sum())
        negative = int((truth == 0).sum())
    else:
        positive = float(sample_weight[truth == 1].sum())
        negative = float(sample_weight[truth == 0].sum())
    if positive <= 0 or negative <= 0:
        return None
    return float(roc_auc_score(truth, probability, sample_weight=sample_weight))


def _safe_average_precision(
    truth: np.ndarray, probability: np.ndarray, sample_weight: np.ndarray | None = None
) -> float | None:
    if sample_weight is None:
        positive = int((truth == 1).sum())
    else:
        positive = float(sample_weight[truth == 1].sum())
    if positive <= 0:
        return None
    return float(
        average_precision_score(truth, probability, sample_weight=sample_weight)
    )


def _mean_defined(values: Sequence[float | None]) -> float | None:
    defined = [float(value) for value in values if value is not None]
    return float(np.mean(defined)) if defined else None


def build_feature_families(
    feature_columns: Sequence[str], *, enforce_expected_counts: bool = False
) -> OrderedDict[str, list[str]]:
    columns = list(feature_columns)
    families: OrderedDict[str, list[str]] = OrderedDict()
    assigned: set[str] = set()
    for family, prefix in FAMILY_PREFIXES:
        family_columns = [column for column in columns if column.startswith(prefix)]
        if not family_columns:
            raise ValueError(f"Feature family is empty: {family}")
        families[family] = family_columns
        overlap = assigned.intersection(family_columns)
        if overlap:
            raise ValueError(f"Feature families overlap: {sorted(overlap)}")
        assigned.update(family_columns)
    if assigned != set(columns):
        raise ValueError(
            "Feature families do not partition the canonical feature columns: "
            f"unassigned={sorted(set(columns) - assigned)}"
        )
    if enforce_expected_counts:
        observed = {family: len(values) for family, values in families.items()}
        if observed != EXPECTED_FAMILY_COUNTS:
            raise ValueError(
                f"Canonical feature-family counts changed: expected "
                f"{EXPECTED_FAMILY_COUNTS}, got {observed}"
            )
    return families


def build_feature_variants(
    feature_columns: Sequence[str], families: Mapping[str, Sequence[str]]
) -> OrderedDict[str, list[str]]:
    full = list(feature_columns)

    def keeping(selected_families: Sequence[str]) -> list[str]:
        allowed = {
            column
            for family in selected_families
            for column in families[family]
        }
        return [column for column in full if column in allowed]

    def excluding(family: str) -> list[str]:
        removed = set(families[family])
        return [column for column in full if column not in removed]

    variants: OrderedDict[str, list[str]] = OrderedDict(
        [
            ("full_hybrid", full),
            ("nrc_only", keeping(["NRC"])),
            ("vad_only", keeping(["VAD"])),
            ("lexicon_only", keeping(["NRC", "VAD"])),
            ("hartmann_only", keeping(["Hartmann"])),
            ("savani_only", keeping(["Savani"])),
            ("goemotions_only", keeping(["GoEmotions"])),
            ("cardiff_only", keeping(["Cardiff"])),
            ("transformer_only", keeping(list(TRANSFORMER_FAMILIES))),
            ("no_nrc", excluding("NRC")),
            ("no_vad", excluding("VAD")),
            ("no_hartmann", excluding("Hartmann")),
            ("no_savani", excluding("Savani")),
            ("no_goemotions", excluding("GoEmotions")),
            ("no_cardiff", excluding("Cardiff")),
        ]
    )
    if set(variants) != set(VARIANT_LABELS):
        raise RuntimeError("Feature-variant definitions and labels differ")
    if any(not columns for columns in variants.values()):
        raise ValueError("A feature ablation variant has no columns")
    return variants


def _align_oof(
    path: Path,
    *,
    data: pd.DataFrame,
    target_columns: Sequence[str],
    context: str,
) -> pd.DataFrame:
    frame = pd.read_csv(path, keep_default_na=False)
    required = {
        "perfume_id",
        "fold_id",
        SOURCE_GROUP_COLUMN,
        *[f"true_{label}" for label in target_columns],
        *[f"prob_{label}" for label in target_columns],
        *[f"pred_{label}" for label in target_columns],
    }
    if missing := required - set(frame.columns):
        raise ValueError(f"{context} lacks columns: {sorted(missing)}")
    frame["perfume_id"] = frame["perfume_id"].astype(str).str.strip()
    if frame["perfume_id"].duplicated().any() or (frame["perfume_id"] == "").any():
        raise ValueError(f"{context} has duplicate or empty perfume IDs")
    expected_ids = data["perfume_id"].astype(str).tolist()
    if set(frame["perfume_id"]) != set(expected_ids):
        raise ValueError(f"{context} perfume universe differs from the frozen data")
    frame = frame.set_index("perfume_id").loc[expected_ids].reset_index()
    for column in ["fold_id", *[f"true_{label}" for label in target_columns], *[f"pred_{label}" for label in target_columns]]:
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(int)
    for label in target_columns:
        probability = pd.to_numeric(frame[f"prob_{label}"], errors="raise").to_numpy(float)
        if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
            raise ValueError(f"{context} has an invalid probability for {label}")
        frame[f"prob_{label}"] = probability
        expected_prediction = (probability >= THRESHOLD).astype(int)
        if not np.array_equal(frame[f"pred_{label}"].to_numpy(int), expected_prediction):
            raise ValueError(f"{context} stored predictions violate threshold {THRESHOLD}")
        if not np.array_equal(
            frame[f"true_{label}"].to_numpy(int), data[label].to_numpy(int)
        ):
            raise ValueError(f"{context} truth differs from the frozen target: {label}")
    if not np.array_equal(frame["fold_id"].to_numpy(int), data["fold_id"].to_numpy(int)):
        raise ValueError(f"{context} fold IDs differ from the frozen manifest")
    if not np.array_equal(
        frame[SOURCE_GROUP_COLUMN].astype(str).to_numpy(),
        data[SOURCE_GROUP_COLUMN].astype(str).to_numpy(),
    ):
        raise ValueError(f"{context} source-group IDs differ from the frozen manifest")
    return frame


def prepare_inputs(
    *,
    config_path: Path,
    feature_csv: Path,
    manifest_csv: Path,
    accepted_oof_csv: Path,
    accepted_metrics_json: Path,
    tfidf_oof_csv: Path,
    tfidf_metrics_json: Path,
    enforce_expected_schema: bool = True,
) -> PreparedInputs:
    config = load_config(config_path)
    target_columns = list(config["targets"]["official_emotion_columns"])
    if target_columns != list(OFFICIAL_LABELS):
        raise ValueError("The Phase 5 target order must match the seven official labels")
    if float(config["evaluation"]["probability_threshold"]) != THRESHOLD:
        raise ValueError(f"Phase 5 requires the frozen threshold {THRESHOLD}")
    input_hash = validate_frozen_input_hash(
        config, key="feature_label_csv", path=feature_csv, required=True
    )
    source_audit_path, source_audit_hash = validate_source_group_audit(config)
    raw = pd.read_csv(feature_csv)
    data = validate_dataset(
        raw,
        unit_id_col="perfume_id",
        target_cols=target_columns,
        excluded_target_cols=dict(config["targets"]["excluded_emotion_columns"]),
    )
    feature_columns = select_feature_columns(data, list(config["data"]["feature_prefixes"]))
    if not feature_columns:
        raise ValueError("No canonical numeric feature columns were found")
    families = build_feature_families(
        feature_columns, enforce_expected_counts=enforce_expected_schema
    )
    variants = build_feature_variants(feature_columns, families)

    manifest = load_and_validate_outer_fold_manifest(
        manifest_csv,
        expected_ids=data["perfume_id"].astype(str).tolist(),
        n_splits=int(config["splits"]["outer_cv"]["n_splits"]),
    )
    manifest = (
        manifest.set_index("perfume_id")
        .loc[data["perfume_id"].astype(str).tolist()]
        .reset_index()
    )
    metadata_columns = ["perfume_id", SOURCE_GROUP_COLUMN, "source_group_size", "fold_id"]
    if "Perfume" in manifest.columns:
        metadata_columns.insert(1, "Perfume")
    data = data.merge(
        manifest[metadata_columns],
        on="perfume_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_manifest"),
    )
    if "Perfume_manifest" in data.columns:
        if "Perfume" not in data.columns:
            data = data.rename(columns={"Perfume_manifest": "Perfume"})
        else:
            data = data.drop(columns=["Perfume_manifest"])
    if data[[SOURCE_GROUP_COLUMN, "source_group_size", "fold_id"]].isna().any().any():
        raise ValueError("The outer manifest did not align to every frozen row")
    data["fold_id"] = pd.to_numeric(data["fold_id"], errors="raise").astype(int)
    data["source_group_size"] = pd.to_numeric(
        data["source_group_size"], errors="raise"
    ).astype(int)

    accepted_oof = _align_oof(
        accepted_oof_csv,
        data=data,
        target_columns=target_columns,
        context="Accepted Logistic Regression OOF",
    )
    tfidf_oof = _align_oof(
        tfidf_oof_csv,
        data=data,
        target_columns=target_columns,
        context="TF-IDF Logistic Regression OOF",
    )
    accepted_metrics = json.loads(accepted_metrics_json.read_text(encoding="utf-8"))
    if accepted_metrics.get("feature_columns") != feature_columns:
        raise ValueError("Canonical feature order differs from the accepted LR metrics")
    tfidf_metrics = json.loads(tfidf_metrics_json.read_text(encoding="utf-8"))
    hashes = {
        "config": sha256_file(config_path),
        "feature_csv": input_hash,
        "manifest_csv": sha256_file(manifest_csv),
        "source_audit": str(source_audit_hash),
        "accepted_lr_oof": sha256_file(accepted_oof_csv),
        "accepted_lr_metrics": sha256_file(accepted_metrics_json),
        "tfidf_oof": sha256_file(tfidf_oof_csv),
        "tfidf_metrics": sha256_file(tfidf_metrics_json),
    }
    if accepted_metrics.get("config_sha256") != hashes["config"]:
        raise ValueError("Config hash differs from the accepted LR provenance")
    if accepted_metrics.get("dataset_sha256") != hashes["feature_csv"]:
        raise ValueError("Feature CSV hash differs from the accepted LR provenance")
    accepted_split = accepted_metrics.get("split_provenance", {})
    if accepted_split.get("outer_manifest_sha256") != hashes["manifest_csv"]:
        raise ValueError("Outer manifest hash differs from the accepted LR provenance")
    if accepted_split.get("source_audit_sha256") != hashes["source_audit"]:
        raise ValueError("Source-audit hash differs from the accepted LR provenance")
    if accepted_metrics.get("model_parameters") != MODEL_SPEC.model_params:
        raise ValueError("Accepted LR parameters differ from the Phase 5 model")
    tfidf_inputs = tfidf_metrics.get("input_sha256", {})
    if tfidf_inputs.get("manifest_csv") != hashes["manifest_csv"]:
        raise ValueError("TF-IDF outer manifest hash differs from Phase 5")
    if tfidf_inputs.get("label_csv") != hashes["feature_csv"]:
        raise ValueError("TF-IDF label-universe hash differs from Phase 5")
    tfidf_classifier = tfidf_metrics.get("classifier", {})
    if tfidf_classifier.get("class_weight", "missing") is not None:
        raise ValueError("The contextual TF-IDF reference must use unweighted LR")
    return PreparedInputs(
        data=data,
        target_columns=target_columns,
        feature_columns=feature_columns,
        families=families,
        variants=variants,
        accepted_oof=accepted_oof,
        accepted_metrics=accepted_metrics,
        tfidf_oof=tfidf_oof,
        tfidf_metrics=tfidf_metrics,
        hashes=hashes,
        config=config,
    )


def _label_metrics(
    truth: np.ndarray,
    probability: np.ndarray,
    prediction: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
) -> dict[str, Any]:
    tp_mask = (truth == 1) & (prediction == 1)
    fp_mask = (truth == 0) & (prediction == 1)
    tn_mask = (truth == 0) & (prediction == 0)
    fn_mask = (truth == 1) & (prediction == 0)
    if sample_weight is None:
        tp = float(tp_mask.sum())
        fp = float(fp_mask.sum())
        tn = float(tn_mask.sum())
        fn = float(fn_mask.sum())
    else:
        tp = float(sample_weight[tp_mask].sum())
        fp = float(sample_weight[fp_mask].sum())
        tn = float(sample_weight[tn_mask].sum())
        fn = float(sample_weight[fn_mask].sum())
    return {
        "positive_support": int(truth.sum()) if sample_weight is None else float(sample_weight[truth == 1].sum()),
        "negative_support": int((truth == 0).sum()) if sample_weight is None else float(sample_weight[truth == 0].sum()),
        "predicted_positive_count": int(prediction.sum()) if sample_weight is None else float(sample_weight[prediction == 1].sum()),
        "tp": int(tp) if sample_weight is None else tp,
        "fp": int(fp) if sample_weight is None else fp,
        "tn": int(tn) if sample_weight is None else tn,
        "fn": int(fn) if sample_weight is None else fn,
        "precision": float(
            precision_score(
                truth, prediction, sample_weight=sample_weight, zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                truth, prediction, sample_weight=sample_weight, zero_division=0
            )
        ),
        "f1": float(
            f1_score(truth, prediction, sample_weight=sample_weight, zero_division=0)
        ),
        "roc_auc": _safe_roc_auc(truth, probability, sample_weight),
        "average_precision": _safe_average_precision(
            truth, probability, sample_weight
        ),
    }


def evaluate_oof(
    oof: pd.DataFrame,
    target_columns: Sequence[str],
    *,
    sample_weight: np.ndarray | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    label_rows: list[dict[str, Any]] = []
    truth_matrix: list[np.ndarray] = []
    prediction_matrix: list[np.ndarray] = []
    for label in target_columns:
        truth = oof[f"true_{label}"].to_numpy(int)
        probability = oof[f"prob_{label}"].to_numpy(float)
        prediction = oof[f"pred_{label}"].to_numpy(int)
        label_rows.append(
            {
                "label": label,
                **_label_metrics(
                    truth,
                    probability,
                    prediction,
                    sample_weight=sample_weight,
                ),
            }
        )
        truth_matrix.append(truth)
        prediction_matrix.append(prediction)
    truth_2d = np.column_stack(truth_matrix)
    prediction_2d = np.column_stack(prediction_matrix)
    summary = {
        "macro_f1": float(np.mean([row["f1"] for row in label_rows])),
        "macro_average_precision": _mean_defined(
            [row["average_precision"] for row in label_rows]
        ),
        "macro_roc_auc": _mean_defined([row["roc_auc"] for row in label_rows]),
        "micro_f1": float(
            f1_score(
                truth_2d,
                prediction_2d,
                average="micro",
                sample_weight=sample_weight,
                zero_division=0,
            )
        ),
        "hamming_loss": float(
            hamming_loss(truth_2d, prediction_2d, sample_weight=sample_weight)
        ),
        "mean_predicted_labels": float(
            np.average(prediction_2d.sum(axis=1), weights=sample_weight)
        ),
    }
    return summary, pd.DataFrame(label_rows)


def train_variant(
    prepared: PreparedInputs,
    *,
    variant: str,
    feature_columns: Sequence[str],
    seed: int = SEED,
    threshold: float = THRESHOLD,
    retain_models: bool = False,
) -> VariantRun:
    data = prepared.data
    target_columns = prepared.target_columns
    X = data[list(feature_columns)].to_numpy(float)
    y = data[target_columns].to_numpy(int)
    fold_ids = data["fold_id"].to_numpy(int)
    probabilities = np.zeros_like(y, dtype=float)
    predictions = np.zeros_like(y, dtype=int)
    fitted_models: dict[tuple[int, str], Any] = {}
    constant_probabilities: dict[tuple[int, str], float] = {}
    fold_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    family_by_feature = {
        column: family
        for family, columns in prepared.families.items()
        for column in columns
    }

    for fold_id in sorted(np.unique(fold_ids).tolist()):
        train_idx = np.where(fold_ids != fold_id)[0]
        test_idx = np.where(fold_ids == fold_id)[0]
        for label_index, label in enumerate(target_columns):
            y_train = y[train_idx, label_index]
            convergence_warning_count = 0
            n_iter: int | None = None
            if np.unique(y_train).size < 2:
                constant = float(y_train[0]) if len(y_train) else 0.0
                probability = np.full(len(test_idx), constant, dtype=float)
                status = "constant_training_class"
                constant_probabilities[(int(fold_id), label)] = constant
            else:
                model = MODEL_SPEC.build_model(seed, y_train)
                with warnings.catch_warnings(record=True) as captured:
                    warnings.simplefilter("always", ConvergenceWarning)
                    model.fit(X[train_idx], y_train)
                convergence_warning_count = sum(
                    issubclass(item.category, ConvergenceWarning) for item in captured
                )
                probability = model.predict_proba(X[test_idx])[:, 1]
                status = "trained"
                n_iter = int(np.max(model.named_steps["clf"].n_iter_))
                if retain_models:
                    fitted_models[(int(fold_id), label)] = model
                if variant == FULL_VARIANT:
                    coefficients = model.named_steps["clf"].coef_[0]
                    for feature, coefficient in zip(feature_columns, coefficients):
                        coefficient_rows.append(
                            {
                                "fold_id": int(fold_id),
                                "label": label,
                                "feature": feature,
                                "family": family_by_feature[feature],
                                "standardized_coefficient": float(coefficient),
                                "training_status": status,
                            }
                        )
            prediction = (probability >= threshold).astype(int)
            probabilities[test_idx, label_index] = probability
            predictions[test_idx, label_index] = prediction
            fold_rows.append(
                {
                    "variant": variant,
                    "fold_id": int(fold_id),
                    "label": label,
                    "train_rows": int(len(train_idx)),
                    "test_rows": int(len(test_idx)),
                    "train_positive_support": int(y_train.sum()),
                    "test_positive_support": int(y[test_idx, label_index].sum()),
                    "training_status": status,
                    "convergence_warning_count": int(convergence_warning_count),
                    "n_iter": n_iter,
                }
            )

    metadata_columns = ["perfume_id"]
    if "Perfume" in data.columns:
        metadata_columns.append("Perfume")
    metadata_columns.extend([SOURCE_GROUP_COLUMN, "source_group_size", "fold_id"])
    oof = data[metadata_columns].copy()
    for label_index, label in enumerate(target_columns):
        oof[f"true_{label}"] = y[:, label_index]
        oof[f"prob_{label}"] = probabilities[:, label_index]
        oof[f"pred_{label}"] = predictions[:, label_index]
    summary, per_label = evaluate_oof(oof, target_columns)
    component_weights = 1.0 / data["source_group_size"].to_numpy(float)
    component_summary, _ = evaluate_oof(
        oof, target_columns, sample_weight=component_weights
    )
    summary.update(
        {
            "variant": variant,
            "variant_label": VARIANT_LABELS[variant],
            "feature_count": int(len(feature_columns)),
            "feature_families": ";".join(
                family
                for family, columns in prepared.families.items()
                if set(columns).intersection(feature_columns)
            ),
            "threshold": float(threshold),
            "seed": int(seed),
            "class_weight": "balanced",
            "component_equal_macro_f1": component_summary["macro_f1"],
            "component_equal_macro_average_precision": component_summary[
                "macro_average_precision"
            ],
            "component_equal_macro_roc_auc": component_summary["macro_roc_auc"],
        }
    )
    per_label.insert(0, "variant", variant)
    return VariantRun(
        variant=variant,
        feature_columns=list(feature_columns),
        oof=oof,
        summary=summary,
        per_label=per_label,
        fold_diagnostics=pd.DataFrame(fold_rows),
        fitted_models=fitted_models,
        constant_probabilities=constant_probabilities,
        coefficient_rows=coefficient_rows,
    )


def validate_full_hybrid_reproduction(
    generated: pd.DataFrame,
    accepted: pd.DataFrame,
    target_columns: Sequence[str],
    *,
    probability_atol: float = 1e-12,
) -> dict[str, Any]:
    for column in ["perfume_id", "fold_id", SOURCE_GROUP_COLUMN]:
        if not np.array_equal(
            generated[column].astype(str).to_numpy(),
            accepted[column].astype(str).to_numpy(),
        ):
            raise ValueError(f"Full-hybrid reproduction differs in {column}")
    max_difference = 0.0
    prediction_mismatches = 0
    truth_mismatches = 0
    for label in target_columns:
        truth_mismatches += int(
            np.sum(
                generated[f"true_{label}"].to_numpy(int)
                != accepted[f"true_{label}"].to_numpy(int)
            )
        )
        prediction_mismatches += int(
            np.sum(
                generated[f"pred_{label}"].to_numpy(int)
                != accepted[f"pred_{label}"].to_numpy(int)
            )
        )
        max_difference = max(
            max_difference,
            float(
                np.max(
                    np.abs(
                        generated[f"prob_{label}"].to_numpy(float)
                        - accepted[f"prob_{label}"].to_numpy(float)
                    )
                )
            ),
        )
    if truth_mismatches or prediction_mismatches or max_difference > probability_atol:
        raise ValueError(
            "Full-hybrid replay did not reproduce the accepted Logistic Regression: "
            f"truth_mismatches={truth_mismatches}, "
            f"prediction_mismatches={prediction_mismatches}, "
            f"max_probability_difference={max_difference}"
        )
    return {
        "status": "passed",
        "probability_atol": float(probability_atol),
        "max_probability_difference": float(max_difference),
        "truth_mismatches": int(truth_mismatches),
        "prediction_mismatches": int(prediction_mismatches),
    }


def _draw_component_weights(
    data: pd.DataFrame, *, reps: int, seed: int
) -> tuple[np.ndarray, int]:
    group_ids = sorted(data[SOURCE_GROUP_COLUMN].astype(str).unique().tolist())
    group_index = {group_id: index for index, group_id in enumerate(group_ids)}
    row_group_index = data[SOURCE_GROUP_COLUMN].astype(str).map(group_index).to_numpy(int)
    inverse_size = 1.0 / data["source_group_size"].to_numpy(float)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(group_ids), size=(reps, len(group_ids)))
    counts = np.zeros((reps, len(group_ids)), dtype=np.int32)
    np.add.at(
        counts,
        (np.repeat(np.arange(reps), len(group_ids)), sampled.ravel()),
        1,
    )
    weights = counts[:, row_group_index].astype(float) * inverse_size
    return weights, len(group_ids)


def weighted_binary_ranking_metrics(
    truth: np.ndarray, probability: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized weighted ROC-AUC and average precision for many weight rows."""

    truth = np.asarray(truth, dtype=int)
    probability = np.asarray(probability, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if weights.ndim == 1:
        weights = weights[None, :]
    if weights.shape[1] != len(truth):
        raise ValueError("Weight matrix and truth length differ")

    ascending = np.argsort(probability, kind="mergesort")
    asc_scores = probability[ascending]
    asc_truth = truth[ascending]
    asc_weights = weights[:, ascending]
    starts = np.r_[0, np.flatnonzero(np.diff(asc_scores) != 0) + 1]
    positive_groups = np.add.reduceat(asc_weights * asc_truth, starts, axis=1)
    negative_groups = np.add.reduceat(asc_weights * (1 - asc_truth), starts, axis=1)
    negative_before = np.cumsum(negative_groups, axis=1) - negative_groups
    roc_numerator = np.sum(
        positive_groups * (negative_before + 0.5 * negative_groups), axis=1
    )
    total_positive = positive_groups.sum(axis=1)
    total_negative = negative_groups.sum(axis=1)
    roc = np.full(weights.shape[0], np.nan, dtype=float)
    roc_defined = (total_positive > 0) & (total_negative > 0)
    roc[roc_defined] = roc_numerator[roc_defined] / (
        total_positive[roc_defined] * total_negative[roc_defined]
    )

    descending = ascending[::-1]
    desc_scores = probability[descending]
    desc_truth = truth[descending]
    desc_weights = weights[:, descending]
    desc_starts = np.r_[0, np.flatnonzero(np.diff(desc_scores) != 0) + 1]
    desc_positive = np.add.reduceat(
        desc_weights * desc_truth, desc_starts, axis=1
    )
    desc_total = np.add.reduceat(desc_weights, desc_starts, axis=1)
    cumulative_positive = np.cumsum(desc_positive, axis=1)
    cumulative_total = np.cumsum(desc_total, axis=1)
    precision = np.divide(
        cumulative_positive,
        cumulative_total,
        out=np.zeros_like(cumulative_positive),
        where=cumulative_total > 0,
    )
    ap = np.full(weights.shape[0], np.nan, dtype=float)
    ap_defined = total_positive > 0
    ap[ap_defined] = np.sum(
        desc_positive[ap_defined] * precision[ap_defined], axis=1
    ) / total_positive[ap_defined]
    return roc, ap


def bootstrap_metric_matrices(
    oof_by_variant: Mapping[str, pd.DataFrame],
    target_columns: Sequence[str],
    weights: np.ndarray,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, np.ndarray]]:
    metric_by_variant: dict[str, dict[str, np.ndarray]] = {}
    defined_counts = {
        "component_equal_macro_f1": np.full(weights.shape[0], len(target_columns), dtype=int),
        "component_equal_macro_average_precision": np.zeros(weights.shape[0], dtype=int),
        "component_equal_macro_roc_auc": np.zeros(weights.shape[0], dtype=int),
    }
    reference = next(iter(oof_by_variant.values()))
    reference_truth = {
        label: reference[f"true_{label}"].to_numpy(int) for label in target_columns
    }
    truth_defined_ap: list[np.ndarray] = []
    truth_defined_roc: list[np.ndarray] = []
    for label in target_columns:
        truth = reference_truth[label]
        positive_weight = weights[:, truth == 1].sum(axis=1)
        negative_weight = weights[:, truth == 0].sum(axis=1)
        truth_defined_ap.append(positive_weight > 0)
        truth_defined_roc.append((positive_weight > 0) & (negative_weight > 0))
    defined_counts["component_equal_macro_average_precision"] = np.sum(
        np.column_stack(truth_defined_ap), axis=1
    )
    defined_counts["component_equal_macro_roc_auc"] = np.sum(
        np.column_stack(truth_defined_roc), axis=1
    )

    for variant, oof in oof_by_variant.items():
        f1_values: list[np.ndarray] = []
        ap_values: list[np.ndarray] = []
        roc_values: list[np.ndarray] = []
        for label in target_columns:
            truth = reference_truth[label]
            if not np.array_equal(truth, oof[f"true_{label}"].to_numpy(int)):
                raise ValueError(f"OOF truth mismatch for {variant}/{label}")
            prediction = oof[f"pred_{label}"].to_numpy(int)
            probability = oof[f"prob_{label}"].to_numpy(float)
            tp = weights @ ((truth == 1) & (prediction == 1)).astype(float)
            fp = weights @ ((truth == 0) & (prediction == 1)).astype(float)
            fn = weights @ ((truth == 1) & (prediction == 0)).astype(float)
            denominator = (2.0 * tp) + fp + fn
            f1_values.append(
                np.divide(
                    2.0 * tp,
                    denominator,
                    out=np.zeros_like(tp),
                    where=denominator > 0,
                )
            )
            roc, ap = weighted_binary_ranking_metrics(truth, probability, weights)
            roc_values.append(roc)
            ap_values.append(ap)
        metric_by_variant[variant] = {
            "component_equal_macro_f1": np.mean(np.vstack(f1_values), axis=0),
            "component_equal_macro_average_precision": np.nanmean(
                np.vstack(ap_values), axis=0
            ),
            "component_equal_macro_roc_auc": np.nanmean(
                np.vstack(roc_values), axis=0
            ),
        }
    return metric_by_variant, defined_counts


def build_bootstrap_tables(
    oof_by_variant: Mapping[str, pd.DataFrame],
    *,
    data: pd.DataFrame,
    target_columns: Sequence[str],
    reps: int,
    seed: int,
    confidence_level: float,
    comparison_variants: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    weights, n_components = _draw_component_weights(data, reps=reps, seed=seed)
    point_weights = (1.0 / data["source_group_size"].to_numpy(float))[None, :]
    bootstrap_values, defined_counts = bootstrap_metric_matrices(
        oof_by_variant, target_columns, weights
    )
    point_values, point_defined = bootstrap_metric_matrices(
        oof_by_variant, target_columns, point_weights
    )
    alpha = 1.0 - confidence_level
    interval_rows: list[dict[str, Any]] = []
    difference_rows: list[dict[str, Any]] = []
    metric_order = (
        "component_equal_macro_average_precision",
        "component_equal_macro_f1",
        "component_equal_macro_roc_auc",
    )
    for variant, metrics in bootstrap_values.items():
        for metric in metric_order:
            values = metrics[metric]
            lower, upper = np.quantile(
                values, [alpha / 2.0, 1.0 - (alpha / 2.0)], method="linear"
            )
            interval_rows.append(
                {
                    "variant": variant,
                    "metric": metric,
                    "point_estimate": float(point_values[variant][metric][0]),
                    "bootstrap_mean": float(values.mean()),
                    "bootstrap_standard_error": float(values.std(ddof=1)),
                    "confidence_level": float(confidence_level),
                    "ci_lower": float(lower),
                    "ci_upper": float(upper),
                    "bootstrap_reps": int(reps),
                    "seed": int(seed),
                    "n_source_components": int(n_components),
                    "defined_label_count_point": int(point_defined[metric][0]),
                    "defined_label_count_min": int(defined_counts[metric].min()),
                    "defined_label_count_max": int(defined_counts[metric].max()),
                }
            )
    for variant in comparison_variants:
        if variant == FULL_VARIANT:
            continue
        for metric in metric_order:
            differences = (
                bootstrap_values[FULL_VARIANT][metric]
                - bootstrap_values[variant][metric]
            )
            point_difference = float(
                point_values[FULL_VARIANT][metric][0]
                - point_values[variant][metric][0]
            )
            lower, upper = np.quantile(
                differences,
                [alpha / 2.0, 1.0 - (alpha / 2.0)],
                method="linear",
            )
            difference_rows.append(
                {
                    "reference_variant": FULL_VARIANT,
                    "comparison_variant": variant,
                    "metric": metric,
                    "difference_definition": "full_hybrid_minus_comparison",
                    "point_estimate_difference": point_difference,
                    "bootstrap_mean_difference": float(differences.mean()),
                    "bootstrap_standard_error": float(differences.std(ddof=1)),
                    "confidence_level": float(confidence_level),
                    "ci_lower": float(lower),
                    "ci_upper": float(upper),
                    "ci_excludes_zero": bool(lower > 0 or upper < 0),
                    "bootstrap_reps": int(reps),
                    "seed": int(seed),
                    "n_source_components": int(n_components),
                    "resampling_unit": SOURCE_GROUP_COLUMN,
                    "defined_label_count_min": int(defined_counts[metric].min()),
                    "defined_label_count_max": int(defined_counts[metric].max()),
                }
            )
    method = {
        "reps": int(reps),
        "seed": int(seed),
        "confidence_level": float(confidence_level),
        "interval_method": "percentile interval with linear quantile interpolation",
        "n_source_components": int(n_components),
        "source_component_ids_sha256": _sha256_text(
            "\n".join(
                sorted(data[SOURCE_GROUP_COLUMN].astype(str).unique().tolist())
            )
        ),
        "resampling_unit": SOURCE_GROUP_COLUMN,
        "component_weight": "each sampled source component has total row weight one",
        "difference_definition": "full_hybrid_minus_comparison",
        "scope": "evaluation-sample uncertainty conditional on the fitted outer-fold models",
    }
    return pd.DataFrame(interval_rows), pd.DataFrame(difference_rows), method


def _matrix_metrics(
    truth: np.ndarray,
    probability: np.ndarray,
    *,
    threshold: float,
    sample_weight: np.ndarray | None = None,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    prediction = (probability >= threshold).astype(int)
    rows: list[dict[str, float]] = []
    for index, label in enumerate(OFFICIAL_LABELS):
        metrics = _label_metrics(
            truth[:, index],
            probability[:, index],
            prediction[:, index],
            sample_weight=sample_weight,
        )
        rows.append(
            {
                "label": label,
                "f1": float(metrics["f1"]),
                "average_precision": float(metrics["average_precision"]),
                "roc_auc": float(metrics["roc_auc"]),
            }
        )
    return (
        {
            "macro_f1": float(np.mean([row["f1"] for row in rows])),
            "macro_average_precision": float(
                np.mean([row["average_precision"] for row in rows])
            ),
            "macro_roc_auc": float(np.mean([row["roc_auc"] for row in rows])),
        },
        rows,
    )


def build_grouped_permutation_importance(
    prepared: PreparedInputs,
    full_run: VariantRun,
    *,
    repeats: int,
    seed: int,
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = prepared.data
    feature_columns = prepared.feature_columns
    feature_index = {feature: index for index, feature in enumerate(feature_columns)}
    X = data[feature_columns].to_numpy(float)
    truth = data[prepared.target_columns].to_numpy(int)
    folds = data["fold_id"].to_numpy(int)
    component_weights = 1.0 / data["source_group_size"].to_numpy(float)
    baseline_probability = np.column_stack(
        [full_run.oof[f"prob_{label}"].to_numpy(float) for label in prepared.target_columns]
    )
    baseline_pooled, baseline_pooled_label = _matrix_metrics(
        truth, baseline_probability, threshold=threshold
    )
    baseline_component, baseline_component_label = _matrix_metrics(
        truth,
        baseline_probability,
        threshold=threshold,
        sample_weight=component_weights,
    )
    baseline_by_label = {
        "pooled": {row["label"]: row for row in baseline_pooled_label},
        "component_equal": {row["label"]: row for row in baseline_component_label},
    }
    repeat_rows: list[dict[str, Any]] = []
    per_label_rows: list[dict[str, Any]] = []
    family_names = list(prepared.families)
    for repeat in range(repeats):
        for family in family_names:
            indices = [feature_index[column] for column in prepared.families[family]]
            permuted_probability = np.zeros_like(baseline_probability)
            for fold_id in sorted(np.unique(folds).tolist()):
                test_idx = np.where(folds == fold_id)[0]
                X_test = X[test_idx].copy()
                rng = np.random.default_rng(
                    # Reuse the same row permutation for every family within a
                    # repeat/fold so family-importance comparisons are paired.
                    np.random.SeedSequence([seed, repeat, int(fold_id)])
                )
                permutation = rng.permutation(len(test_idx))
                X_test[:, indices] = X_test[permutation][:, indices]
                for label_index, label in enumerate(prepared.target_columns):
                    key = (int(fold_id), label)
                    if key in full_run.fitted_models:
                        probability = full_run.fitted_models[key].predict_proba(X_test)[:, 1]
                    else:
                        probability = np.full(
                            len(test_idx), full_run.constant_probabilities[key], dtype=float
                        )
                    permuted_probability[test_idx, label_index] = probability
            pooled, pooled_labels = _matrix_metrics(
                truth, permuted_probability, threshold=threshold
            )
            component, component_labels = _matrix_metrics(
                truth,
                permuted_probability,
                threshold=threshold,
                sample_weight=component_weights,
            )
            for estimator, baseline, permuted in (
                ("pooled_oof", baseline_pooled, pooled),
                ("component_equal", baseline_component, component),
            ):
                repeat_rows.append(
                    {
                        "family": family,
                        "repeat": int(repeat + 1),
                        "estimator": estimator,
                        "macro_average_precision_drop": float(
                            baseline["macro_average_precision"]
                            - permuted["macro_average_precision"]
                        ),
                        "macro_f1_drop": float(
                            baseline["macro_f1"] - permuted["macro_f1"]
                        ),
                        "macro_roc_auc_drop": float(
                            baseline["macro_roc_auc"] - permuted["macro_roc_auc"]
                        ),
                    }
                )
            for estimator, rows in (
                ("pooled_oof", pooled_labels),
                ("component_equal", component_labels),
            ):
                lookup = baseline_by_label[
                    "pooled" if estimator == "pooled_oof" else "component_equal"
                ]
                for row in rows:
                    baseline_row = lookup[row["label"]]
                    per_label_rows.append(
                        {
                            "family": family,
                            "repeat": int(repeat + 1),
                            "estimator": estimator,
                            "label": row["label"],
                            "average_precision_drop": float(
                                baseline_row["average_precision"]
                                - row["average_precision"]
                            ),
                            "f1_drop": float(baseline_row["f1"] - row["f1"]),
                            "roc_auc_drop": float(
                                baseline_row["roc_auc"] - row["roc_auc"]
                            ),
                        }
                    )
    repeat_df = pd.DataFrame(repeat_rows)
    summary_rows: list[dict[str, Any]] = []
    for (family, estimator), group in repeat_df.groupby(
        ["family", "estimator"], sort=False
    ):
        row: dict[str, Any] = {
            "family": family,
            "estimator": estimator,
            "repeats": int(len(group)),
            "importance_definition": "unpermuted_metric_minus_permuted_metric",
        }
        for column in (
            "macro_average_precision_drop",
            "macro_f1_drop",
            "macro_roc_auc_drop",
        ):
            values = group[column].to_numpy(float)
            row[f"{column}_mean"] = float(values.mean())
            row[f"{column}_std"] = float(values.std(ddof=1))
            row[f"{column}_q025"] = float(np.quantile(values, 0.025))
            row[f"{column}_q975"] = float(np.quantile(values, 0.975))
        summary_rows.append(row)
    per_label_repeats = pd.DataFrame(per_label_rows)
    per_label_summary_rows: list[dict[str, Any]] = []
    for (family, estimator, label), group in per_label_repeats.groupby(
        ["family", "estimator", "label"], sort=False
    ):
        row = {
            "family": family,
            "estimator": estimator,
            "label": label,
            "repeats": int(len(group)),
            "importance_definition": "unpermuted_metric_minus_permuted_metric",
        }
        for column in (
            "average_precision_drop",
            "f1_drop",
            "roc_auc_drop",
        ):
            values = group[column].to_numpy(float)
            row[f"{column}_mean"] = float(values.mean())
            row[f"{column}_std"] = float(values.std(ddof=1))
            row[f"{column}_q025"] = float(np.quantile(values, 0.025))
            row[f"{column}_q975"] = float(np.quantile(values, 0.975))
        per_label_summary_rows.append(row)
    return (
        pd.DataFrame(summary_rows),
        repeat_df,
        pd.DataFrame(per_label_summary_rows),
        per_label_repeats,
    )


def build_excluded_label_permutation_sensitivity(
    per_label_repeats: pd.DataFrame,
    *,
    target_columns: Sequence[str],
    excluded_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reaggregate repeat-level permutation drops without one rare label."""

    if excluded_label not in target_columns:
        raise ValueError(f"Unknown excluded label: {excluded_label}")
    included_labels = [label for label in target_columns if label != excluded_label]
    filtered = per_label_repeats.loc[
        per_label_repeats["label"].isin(included_labels)
    ].copy()
    repeat_rows: list[dict[str, Any]] = []
    for (family, repeat, estimator), group in filtered.groupby(
        ["family", "repeat", "estimator"], sort=False
    ):
        if set(group["label"]) != set(included_labels):
            raise ValueError(
                f"Incomplete excluded-label permutation group: "
                f"{family}/{repeat}/{estimator}"
            )
        repeat_rows.append(
            {
                "family": family,
                "repeat": int(repeat),
                "estimator": estimator,
                "excluded_label": excluded_label,
                "included_label_count": int(len(included_labels)),
                "macro_average_precision_drop": float(
                    group["average_precision_drop"].mean()
                ),
                "macro_f1_drop": float(group["f1_drop"].mean()),
                "macro_roc_auc_drop": float(group["roc_auc_drop"].mean()),
            }
        )
    repeat_df = pd.DataFrame(repeat_rows)
    summary_rows: list[dict[str, Any]] = []
    for (family, estimator), group in repeat_df.groupby(
        ["family", "estimator"], sort=False
    ):
        row: dict[str, Any] = {
            "family": family,
            "estimator": estimator,
            "excluded_label": excluded_label,
            "included_label_count": int(len(included_labels)),
            "repeats": int(len(group)),
            "importance_definition": "unpermuted_metric_minus_permuted_metric",
        }
        for column in (
            "macro_average_precision_drop",
            "macro_f1_drop",
            "macro_roc_auc_drop",
        ):
            values = group[column].to_numpy(float)
            row[f"{column}_mean"] = float(values.mean())
            row[f"{column}_std"] = float(values.std(ddof=1))
            row[f"{column}_q025"] = float(np.quantile(values, 0.025))
            row[f"{column}_q975"] = float(np.quantile(values, 0.975))
        summary_rows.append(row)
    return pd.DataFrame(summary_rows), repeat_df


def build_coefficient_stability(
    coefficient_rows: Sequence[Mapping[str, Any]], *, top_k: int = 10
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    coefficients = pd.DataFrame(coefficient_rows)
    if coefficients.empty:
        raise ValueError("No full-hybrid coefficients were retained")
    coefficients["absolute_coefficient"] = coefficients[
        "standardized_coefficient"
    ].abs()
    coefficients["absolute_rank_within_fold_label"] = coefficients.groupby(
        ["fold_id", "label"]
    )["absolute_coefficient"].rank(method="min", ascending=False)
    coefficients["in_top_k_absolute"] = (
        coefficients["absolute_rank_within_fold_label"] <= top_k
    )
    tolerance = 1e-10
    stability_rows: list[dict[str, Any]] = []
    for (label, feature, family), group in coefficients.groupby(
        ["label", "feature", "family"], sort=False
    ):
        values = group["standardized_coefficient"].to_numpy(float)
        positive = int((values > tolerance).sum())
        negative = int((values < -tolerance).sum())
        zero = int(len(values) - positive - negative)
        stability_rows.append(
            {
                "label": label,
                "feature": feature,
                "family": family,
                "n_folds": int(len(values)),
                "coefficient_mean": float(values.mean()),
                "coefficient_median": float(np.median(values)),
                "coefficient_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "coefficient_q25": float(np.quantile(values, 0.25)),
                "coefficient_q75": float(np.quantile(values, 0.75)),
                "coefficient_min": float(values.min()),
                "coefficient_max": float(values.max()),
                "mean_absolute_coefficient": float(np.mean(np.abs(values))),
                "positive_fold_count": positive,
                "negative_fold_count": negative,
                "zero_fold_count": zero,
                "dominant_sign_fraction": float(max(positive, negative, zero) / len(values)),
                "top_k_absolute_appearance_rate": float(
                    group["in_top_k_absolute"].mean()
                ),
            }
        )
    stability = pd.DataFrame(stability_rows)
    family_rows: list[dict[str, Any]] = []
    for (label, family), group in stability.groupby(["label", "family"], sort=False):
        family_rows.append(
            {
                "label": label,
                "family": family,
                "feature_count": int(len(group)),
                "mean_absolute_coefficient": float(
                    group["mean_absolute_coefficient"].mean()
                ),
                "median_dominant_sign_fraction": float(
                    group["dominant_sign_fraction"].median()
                ),
                "mean_top_k_absolute_appearance_rate": float(
                    group["top_k_absolute_appearance_rate"].mean()
                ),
            }
        )
    vector_rows: list[dict[str, Any]] = []
    for label, label_frame in coefficients.groupby("label", sort=False):
        pivot = label_frame.pivot(
            index="feature", columns="fold_id", values="standardized_coefficient"
        )
        spearman_values: list[float] = []
        jaccard_values: list[float] = []
        for left, right in combinations(pivot.columns.tolist(), 2):
            left_values = pivot[left]
            right_values = pivot[right]
            left_ranks = left_values.rank(method="average").to_numpy(float)
            right_ranks = right_values.rank(method="average").to_numpy(float)
            correlation = float(np.corrcoef(left_ranks, right_ranks)[0, 1])
            spearman_values.append(correlation)
            left_top = set(
                left_values.abs().nlargest(min(top_k, len(left_values))).index.tolist()
            )
            right_top = set(
                right_values.abs().nlargest(min(top_k, len(right_values))).index.tolist()
            )
            jaccard_values.append(
                float(len(left_top & right_top) / len(left_top | right_top))
            )
        vector_rows.append(
            {
                "label": label,
                "fold_pair_count": int(len(spearman_values)),
                "median_signed_coefficient_spearman": float(
                    np.median(spearman_values)
                ),
                "mean_signed_coefficient_spearman": float(np.mean(spearman_values)),
                "median_top_k_jaccard": float(np.median(jaccard_values)),
                "mean_top_k_jaccard": float(np.mean(jaccard_values)),
                "top_k": int(top_k),
            }
        )
    return coefficients, stability, pd.DataFrame(family_rows), pd.DataFrame(vector_rows)


def _save_metric_figure(summary: pd.DataFrame, path: Path) -> None:
    ordered = summary.sort_values("macro_average_precision", ascending=True)
    labels = ordered["variant_label"].tolist()
    metrics = [
        ("macro_average_precision", "Macro average precision"),
        ("macro_f1", "Macro-F1 at 0.3"),
        ("macro_roc_auc", "Macro ROC-AUC"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 8), sharey=True)
    colors = ["#3155C6" if value == FULL_VARIANT else "#9EB3E8" for value in ordered["variant"]]
    for axis, (column, title) in zip(axes, metrics):
        axis.barh(labels, ordered[column], color=colors)
        axis.set_title(title)
        axis.set_xlim(0, max(0.65, float(ordered[column].max()) * 1.08))
        axis.grid(axis="x", alpha=0.2)
    fig.suptitle("Source-grouped Logistic Regression feature-family ablation")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_permutation_figure(
    summary: pd.DataFrame, path: Path, *, repeats: int
) -> None:
    pooled = summary.loc[summary["estimator"] == "pooled_oof"].sort_values(
        "macro_average_precision_drop_mean"
    )
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(
        pooled["family"],
        pooled["macro_average_precision_drop_mean"],
        xerr=pooled["macro_average_precision_drop_std"],
        color="#3155C6",
        alpha=0.85,
        capsize=3,
    )
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set_xlabel("Macro average-precision drop after held-out family permutation")
    ax.set_title(f"Grouped permutation importance ({repeats} deterministic repeats)")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_checksums(root: Path, checksum_path: Path) -> None:
    files = sorted(
        path for path in root.rglob("*") if path.is_file() and path != checksum_path
    )
    lines = [f"{sha256_file(path)}  {path.relative_to(root)}" for path in files]
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def verify_checksum_manifest(root: Path, checksum_path: Path) -> str:
    """Verify every file named by a checksum manifest and return its own hash."""

    root = root.resolve()
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split("  ", 1)
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file():
            raise ValueError(f"Invalid checksum entry: {relative}")
        observed = sha256_file(path)
        if observed != digest:
            raise ValueError(
                f"Checksum mismatch in protected accepted benchmark: {relative}"
            )
    return sha256_file(checksum_path)


def build_excluded_label_sensitivity(
    runs: Mapping[str, VariantRun],
    *,
    data: pd.DataFrame,
    target_columns: Sequence[str],
    excluded_label: str,
    bootstrap_reps: int,
    seed: int,
    confidence_level: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Recalculate variant evidence after excluding one prespecified rare label."""

    if excluded_label not in target_columns:
        raise ValueError(f"Unknown excluded label: {excluded_label}")
    included_labels = [label for label in target_columns if label != excluded_label]
    component_weights = 1.0 / data["source_group_size"].to_numpy(float)
    rows: list[dict[str, Any]] = []
    for variant, run in runs.items():
        pooled, _ = evaluate_oof(run.oof, included_labels)
        component_equal, _ = evaluate_oof(
            run.oof, included_labels, sample_weight=component_weights
        )
        rows.append(
            {
                "variant": variant,
                "variant_label": VARIANT_LABELS[variant],
                "excluded_label": excluded_label,
                "included_label_count": int(len(included_labels)),
                "feature_count": int(len(run.feature_columns)),
                **pooled,
                "component_equal_macro_f1": component_equal["macro_f1"],
                "component_equal_macro_average_precision": component_equal[
                    "macro_average_precision"
                ],
                "component_equal_macro_roc_auc": component_equal["macro_roc_auc"],
            }
        )
    summary = pd.DataFrame(rows)
    full = summary.loc[summary["variant"] == FULL_VARIANT].iloc[0]
    for metric in ("macro_average_precision", "macro_f1", "macro_roc_auc"):
        summary[f"variant_minus_full_{metric}"] = (
            summary[metric] - float(full[metric])
        )
    intervals, differences, method = build_bootstrap_tables(
        OrderedDict((variant, run.oof) for variant, run in runs.items()),
        data=data,
        target_columns=included_labels,
        reps=bootstrap_reps,
        seed=seed,
        confidence_level=confidence_level,
        comparison_variants=list(runs),
    )
    method = {
        **method,
        "excluded_label": excluded_label,
        "included_labels": included_labels,
        "role": "rare-label sensitivity analysis; seven-label results remain primary",
    }
    return summary, intervals, differences, method


def run_feature_ablation(
    *,
    config_path: Path = DEFAULT_CONFIG,
    feature_csv: Path = DEFAULT_FEATURE_CSV,
    manifest_csv: Path = DEFAULT_MANIFEST,
    accepted_oof_csv: Path = DEFAULT_ACCEPTED_OOF,
    accepted_metrics_json: Path = DEFAULT_ACCEPTED_METRICS,
    tfidf_oof_csv: Path = DEFAULT_TFIDF_OOF,
    tfidf_metrics_json: Path = DEFAULT_TFIDF_METRICS,
    out_dir: Path = DEFAULT_OUT_DIR,
    bootstrap_reps: int = BOOTSTRAP_REPS,
    permutation_repeats: int = PERMUTATION_REPEATS,
    seed: int = SEED,
    confidence_level: float = CONFIDENCE_LEVEL,
    enforce_expected_schema: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    if bootstrap_reps < 2:
        raise ValueError("bootstrap_reps must be at least 2")
    if permutation_repeats < 2:
        raise ValueError("permutation_repeats must be at least 2")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be in (0, 1)")
    validate_canonical_run_configuration(
        out_dir=out_dir,
        config_path=config_path,
        feature_csv=feature_csv,
        manifest_csv=manifest_csv,
        accepted_oof_csv=accepted_oof_csv,
        accepted_metrics_json=accepted_metrics_json,
        tfidf_oof_csv=tfidf_oof_csv,
        tfidf_metrics_json=tfidf_metrics_json,
        bootstrap_reps=bootstrap_reps,
        permutation_repeats=permutation_repeats,
        seed=seed,
        confidence_level=confidence_level,
        enforce_expected_schema=enforce_expected_schema,
    )
    if not DEFAULT_ACCEPTED_CHECKSUMS.is_file():
        raise FileNotFoundError(
            f"Accepted benchmark checksum manifest is missing: "
            f"{DEFAULT_ACCEPTED_CHECKSUMS}"
        )
    accepted_checksums_before = verify_checksum_manifest(
        DEFAULT_ACCEPTED_EXPERIMENT_ROOT, DEFAULT_ACCEPTED_CHECKSUMS
    )
    out_dir = prepare_output_root(Path(out_dir), overwrite=overwrite)
    predictions_dir = out_dir / "oof_predictions"
    figures_dir = out_dir / "figures"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    prepared = prepare_inputs(
        config_path=config_path,
        feature_csv=feature_csv,
        manifest_csv=manifest_csv,
        accepted_oof_csv=accepted_oof_csv,
        accepted_metrics_json=accepted_metrics_json,
        tfidf_oof_csv=tfidf_oof_csv,
        tfidf_metrics_json=tfidf_metrics_json,
        enforce_expected_schema=enforce_expected_schema,
    )
    runs: OrderedDict[str, VariantRun] = OrderedDict()
    for variant, feature_columns in prepared.variants.items():
        runs[variant] = train_variant(
            prepared,
            variant=variant,
            feature_columns=feature_columns,
            seed=seed,
            retain_models=variant == FULL_VARIANT,
        )

    reproduction = validate_full_hybrid_reproduction(
        runs[FULL_VARIANT].oof,
        prepared.accepted_oof,
        prepared.target_columns,
    )
    for variant, run in runs.items():
        run.oof.to_csv(predictions_dir / f"{variant}_oof.csv", index=False)
    summary = pd.DataFrame([run.summary for run in runs.values()])
    full_summary = summary.loc[summary["variant"] == FULL_VARIANT].iloc[0]
    for metric in ("macro_average_precision", "macro_f1", "macro_roc_auc"):
        summary[f"variant_minus_full_{metric}"] = summary[metric] - float(
            full_summary[metric]
        )
    per_label = pd.concat([run.per_label for run in runs.values()], ignore_index=True)
    fold_diagnostics = pd.concat(
        [run.fold_diagnostics for run in runs.values()], ignore_index=True
    )

    variant_rows: list[dict[str, Any]] = []
    for variant, feature_columns in prepared.variants.items():
        feature_set = set(feature_columns)
        included_families = [
            family
            for family, columns in prepared.families.items()
            if feature_set.intersection(columns)
        ]
        variant_rows.append(
            {
                "variant": variant,
                "variant_label": VARIANT_LABELS[variant],
                "feature_count": int(len(feature_columns)),
                "included_families": ";".join(included_families),
                "feature_columns": ";".join(feature_columns),
                "feature_columns_sha256": _sha256_text("\n".join(feature_columns)),
            }
        )
    variant_manifest = pd.DataFrame(variant_rows)

    oof_for_bootstrap = OrderedDict((variant, run.oof) for variant, run in runs.items())
    intervals, differences, bootstrap_method = build_bootstrap_tables(
        oof_for_bootstrap,
        data=prepared.data,
        target_columns=prepared.target_columns,
        reps=bootstrap_reps,
        seed=seed,
        confidence_level=confidence_level,
        comparison_variants=list(prepared.variants),
    )
    (
        fear_excluded_summary,
        fear_excluded_intervals,
        fear_excluded_differences,
        fear_excluded_method,
    ) = build_excluded_label_sensitivity(
        runs,
        data=prepared.data,
        target_columns=prepared.target_columns,
        excluded_label="fear",
        bootstrap_reps=bootstrap_reps,
        seed=seed,
        confidence_level=confidence_level,
    )
    tfidf_intervals, tfidf_difference, _ = build_bootstrap_tables(
        OrderedDict(
            [
                (FULL_VARIANT, runs[FULL_VARIANT].oof),
                (TFIDF_REFERENCE, prepared.tfidf_oof),
            ]
        ),
        data=prepared.data,
        target_columns=prepared.target_columns,
        reps=bootstrap_reps,
        seed=seed,
        confidence_level=confidence_level,
        comparison_variants=[FULL_VARIANT, TFIDF_REFERENCE],
    )
    tfidf_pooled, _ = evaluate_oof(prepared.tfidf_oof, prepared.target_columns)
    tfidf_component, _ = evaluate_oof(
        prepared.tfidf_oof,
        prepared.target_columns,
        sample_weight=1.0 / prepared.data["source_group_size"].to_numpy(float),
    )
    contextual_tfidf = pd.DataFrame(
        [
            {
                "comparison_role": "external_text_baseline_not_feature_ablation",
                "model": "Full hybrid balanced LR",
                **{
                    key: full_summary[key]
                    for key in ("macro_average_precision", "macro_f1", "macro_roc_auc")
                },
                "component_equal_macro_average_precision": full_summary[
                    "component_equal_macro_average_precision"
                ],
                "component_equal_macro_f1": full_summary[
                    "component_equal_macro_f1"
                ],
                "component_equal_macro_roc_auc": full_summary[
                    "component_equal_macro_roc_auc"
                ],
                "class_weight": "balanced",
            },
            {
                "comparison_role": "external_text_baseline_not_feature_ablation",
                "model": "TF-IDF + unweighted LR",
                **{
                    key: tfidf_pooled[key]
                    for key in ("macro_average_precision", "macro_f1", "macro_roc_auc")
                },
                "component_equal_macro_average_precision": tfidf_component[
                    "macro_average_precision"
                ],
                "component_equal_macro_f1": tfidf_component["macro_f1"],
                "component_equal_macro_roc_auc": tfidf_component["macro_roc_auc"],
                "class_weight": "none",
            },
        ]
    )

    (
        permutation_summary,
        permutation_repeats_df,
        permutation_per_label,
        permutation_per_label_repeats,
    ) = (
        build_grouped_permutation_importance(
            prepared,
            runs[FULL_VARIANT],
            repeats=permutation_repeats,
            seed=seed,
            threshold=THRESHOLD,
        )
    )
    (
        fear_excluded_permutation_summary,
        fear_excluded_permutation_repeats,
    ) = build_excluded_label_permutation_sensitivity(
        permutation_per_label_repeats,
        target_columns=prepared.target_columns,
        excluded_label="fear",
    )
    coefficient_raw, coefficient_stability, coefficient_family, coefficient_vectors = (
        build_coefficient_stability(runs[FULL_VARIANT].coefficient_rows)
    )

    paths = {
        "variant_manifest": out_dir / "feature_columns_by_variant.csv",
        "summary": out_dir / "ablation_summary.csv",
        "per_label": out_dir / "ablation_per_label.csv",
        "fold_diagnostics": out_dir / "ablation_fold_diagnostics.csv",
        "bootstrap_intervals": out_dir / "model_metric_bootstrap_intervals.csv",
        "paired_bootstrap": out_dir / "paired_bootstrap_vs_full.csv",
        "fear_excluded_summary": out_dir / "fear_excluded_ablation_summary.csv",
        "fear_excluded_intervals": out_dir
        / "fear_excluded_model_metric_bootstrap_intervals.csv",
        "fear_excluded_paired_bootstrap": out_dir
        / "fear_excluded_paired_bootstrap_vs_full.csv",
        "contextual_tfidf": out_dir / "contextual_tfidf_comparison.csv",
        "contextual_tfidf_intervals": out_dir / "contextual_tfidf_bootstrap_intervals.csv",
        "contextual_tfidf_difference": out_dir / "contextual_tfidf_paired_difference.csv",
        "permutation_summary": out_dir / "family_permutation_importance.csv",
        "permutation_repeats": out_dir / "family_permutation_importance_repeats.csv",
        "permutation_per_label": out_dir / "family_permutation_importance_per_label.csv",
        "permutation_per_label_repeats": out_dir
        / "family_permutation_importance_per_label_repeats.csv",
        "fear_excluded_permutation_summary": out_dir
        / "fear_excluded_family_permutation_importance.csv",
        "fear_excluded_permutation_repeats": out_dir
        / "fear_excluded_family_permutation_importance_repeats.csv",
        "coefficients": out_dir / "standardized_lr_coefficients.csv",
        "coefficient_stability": out_dir / "coefficient_stability.csv",
        "coefficient_family": out_dir / "coefficient_family_stability.csv",
        "coefficient_vectors": out_dir / "coefficient_vector_stability.csv",
        "metric_figure": figures_dir / "feature_ablation_metric_comparison.png",
        "permutation_figure": figures_dir / "grouped_permutation_importance.png",
        "report": out_dir / "REPORT.json",
        "checksums": out_dir / "SHA256SUMS",
    }
    variant_manifest.to_csv(paths["variant_manifest"], index=False)
    summary.to_csv(paths["summary"], index=False)
    per_label.to_csv(paths["per_label"], index=False)
    fold_diagnostics.to_csv(paths["fold_diagnostics"], index=False)
    intervals.to_csv(paths["bootstrap_intervals"], index=False)
    differences.to_csv(paths["paired_bootstrap"], index=False)
    fear_excluded_summary.to_csv(paths["fear_excluded_summary"], index=False)
    fear_excluded_intervals.to_csv(paths["fear_excluded_intervals"], index=False)
    fear_excluded_differences.to_csv(
        paths["fear_excluded_paired_bootstrap"], index=False
    )
    contextual_tfidf.to_csv(paths["contextual_tfidf"], index=False)
    tfidf_intervals.to_csv(paths["contextual_tfidf_intervals"], index=False)
    tfidf_difference.to_csv(paths["contextual_tfidf_difference"], index=False)
    permutation_summary.to_csv(paths["permutation_summary"], index=False)
    permutation_repeats_df.to_csv(paths["permutation_repeats"], index=False)
    permutation_per_label.to_csv(paths["permutation_per_label"], index=False)
    permutation_per_label_repeats.to_csv(
        paths["permutation_per_label_repeats"], index=False
    )
    fear_excluded_permutation_summary.to_csv(
        paths["fear_excluded_permutation_summary"], index=False
    )
    fear_excluded_permutation_repeats.to_csv(
        paths["fear_excluded_permutation_repeats"], index=False
    )
    coefficient_raw.to_csv(paths["coefficients"], index=False)
    coefficient_stability.to_csv(paths["coefficient_stability"], index=False)
    coefficient_family.to_csv(paths["coefficient_family"], index=False)
    coefficient_vectors.to_csv(paths["coefficient_vectors"], index=False)
    _save_metric_figure(summary, paths["metric_figure"])
    _save_permutation_figure(
        permutation_summary,
        paths["permutation_figure"],
        repeats=permutation_repeats,
    )

    primary_differences = differences.loc[
        differences["metric"] == "component_equal_macro_average_precision"
    ]
    leave_one_out_variants = [value for value in prepared.variants if value.startswith("no_")]
    fear_excluded_full = fear_excluded_summary.loc[
        fear_excluded_summary["variant"] == FULL_VARIANT
    ].iloc[0]
    accepted_checksums_after = verify_checksum_manifest(
        DEFAULT_ACCEPTED_EXPERIMENT_ROOT, DEFAULT_ACCEPTED_CHECKSUMS
    )
    if accepted_checksums_after != accepted_checksums_before:
        raise RuntimeError("The accepted benchmark checksum manifest changed during Phase 5")
    code_sha256 = {
        name: sha256_file(path) for name, path in CODE_PROVENANCE_PATHS.items()
    }
    report = {
        "analysis_id": "source_grouped_feature_ablation_v1",
        "status": "completed",
        "analysis_role": "post_acceptance_supervised_feature_family_supplement",
        "accepted_benchmark_root_modified": (
            accepted_checksums_after != accepted_checksums_before
        ),
        "output_safety": {
            "supplemental_output_root": stringify_path(out_dir),
            "protected_accepted_root": stringify_path(
                DEFAULT_ACCEPTED_EXPERIMENT_ROOT
            ),
            "protected_checksum_manifest": stringify_path(
                DEFAULT_ACCEPTED_CHECKSUMS
            ),
            "protected_checksum_manifest_sha256_before": accepted_checksums_before,
            "protected_checksum_manifest_sha256_after": accepted_checksums_after,
        },
        "inputs": {
            "config": stringify_path(config_path),
            "feature_csv": stringify_path(feature_csv),
            "manifest_csv": stringify_path(manifest_csv),
            "accepted_lr_oof": stringify_path(accepted_oof_csv),
            "accepted_lr_metrics": stringify_path(accepted_metrics_json),
            "tfidf_oof": stringify_path(tfidf_oof_csv),
            "tfidf_metrics": stringify_path(tfidf_metrics_json),
            "sha256": prepared.hashes,
        },
        "design": {
            "n_rows": int(len(prepared.data)),
            "n_source_components": int(prepared.data[SOURCE_GROUP_COLUMN].nunique()),
            "target_columns": prepared.target_columns,
            "feature_count": int(len(prepared.feature_columns)),
            "family_counts": {
                family: len(columns) for family, columns in prepared.families.items()
            },
            "variant_count": int(len(prepared.variants)),
            "variants": variant_rows,
            "model": MODEL_SPEC.model_params,
            "seed": int(seed),
            "threshold": THRESHOLD,
            "primary_ranking_metric": "pooled_oof_macro_average_precision",
            "primary_thresholded_metric": "pooled_oof_macro_f1_at_0.3",
            "secondary_ranking_metric": "pooled_oof_macro_roc_auc",
            "preprocessing_scope": "outer_training_fold_only",
            "summary_delta_definition": "variant_minus_full_hybrid",
        },
        "full_hybrid_reproduction_gate": reproduction,
        "bootstrap": bootstrap_method,
        "fear_excluded_sensitivity": {
            **fear_excluded_method,
            "fear_positive_support": int(prepared.data["fear"].sum()),
            "point_estimate_leader": fear_excluded_summary.sort_values(
                "macro_average_precision", ascending=False
            ).iloc[0]["variant"],
            "full_hybrid_pooled_metrics": {
                key: fear_excluded_full[key]
                for key in (
                    "macro_average_precision",
                    "macro_f1",
                    "macro_roc_auc",
                )
            },
            "permutation_summary": fear_excluded_permutation_summary.to_dict(
                orient="records"
            ),
        },
        "permutation": {
            "repeats": int(permutation_repeats),
            "seed": int(seed),
            "scope": (
                "joint family-column permutation across perfume rows within each "
                "held-out outer fold; source-component membership is not preserved "
                "by the permutation"
            ),
            "importance_definition": "unpermuted_metric_minus_permuted_metric",
            "interval_label": "Monte Carlo permutation quantiles, not confidence intervals",
        },
        "coefficients": {
            "scale": "one outer-training-fold standard deviation after median imputation",
            "raw_rows": int(len(coefficient_raw)),
            "top_k": 10,
            "interpretation": "associational; correlated features make signs and magnitudes non-causal",
        },
        "results_overview": {
            "pooled_primary_metric_point_estimate_leader": summary.sort_values(
                "macro_average_precision", ascending=False
            ).iloc[0]["variant"],
            "full_hybrid_pooled_metrics": {
                key: full_summary[key]
                for key in ("macro_average_precision", "macro_f1", "macro_roc_auc")
            },
            "leave_one_out_primary_differences": primary_differences.loc[
                primary_differences["comparison_variant"].isin(leave_one_out_variants)
            ].to_dict(orient="records"),
            "primary_pairwise_intervals_excluding_zero": primary_differences.loc[
                primary_differences["ci_excludes_zero"]
            ].to_dict(orient="records"),
        },
        "caveats": [
            "The 15 feature variants are post-hoc supplemental analyses rather than prespecified primary model rows.",
            "Fourteen full-versus-variant contrasts create multiplicity; raw 95% intervals are exploratory.",
            "The component bootstrap resamples fixed OOF predictions and does not include training-set refitting variability.",
            "One 334-perfume source component and ultra-rare labels limit uncertainty precision; fear has only three positive cases.",
            "The six-label fear-excluded calculation is a sensitivity analysis; the prespecified seven-label results remain primary.",
            "TF-IDF is contextual only because it uses unweighted Logistic Regression and cannot isolate feature-family causality.",
            "Ablation, grouped permutation, and coefficient stability are interpreted jointly; none is a causal effect.",
        ],
        "software": {
            "git_commit": get_git_commit(),
            "code_sha256": code_sha256,
            "package_versions": collect_package_versions(),
        },
        "artifacts": {key: stringify_path(path) for key, path in paths.items()},
    }
    paths["report"].write_text(
        json.dumps(_json_safe(report), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_checksums(out_dir, paths["checksums"])
    return {
        "report": report,
        "paths": paths,
        "summary": summary,
        "paired_bootstrap": differences,
        "permutation_summary": permutation_summary,
        "coefficient_stability": coefficient_stability,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the final source-grouped supervised feature-family ablation."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--feature-csv", type=Path, default=DEFAULT_FEATURE_CSV)
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--accepted-oof", type=Path, default=DEFAULT_ACCEPTED_OOF)
    parser.add_argument(
        "--accepted-metrics", type=Path, default=DEFAULT_ACCEPTED_METRICS
    )
    parser.add_argument("--tfidf-oof", type=Path, default=DEFAULT_TFIDF_OOF)
    parser.add_argument("--tfidf-metrics", type=Path, default=DEFAULT_TFIDF_METRICS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_REPS)
    parser.add_argument(
        "--permutation-repeats", type=int, default=PERMUTATION_REPEATS
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--confidence-level", type=float, default=CONFIDENCE_LEVEL)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace only files already owned by a matching Phase 5 SHA256SUMS "
            "manifest."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_feature_ablation(
        config_path=args.config,
        feature_csv=args.feature_csv,
        manifest_csv=args.manifest_csv,
        accepted_oof_csv=args.accepted_oof,
        accepted_metrics_json=args.accepted_metrics,
        tfidf_oof_csv=args.tfidf_oof,
        tfidf_metrics_json=args.tfidf_metrics,
        out_dir=args.out_dir,
        bootstrap_reps=args.bootstrap_reps,
        permutation_repeats=args.permutation_repeats,
        seed=args.seed,
        confidence_level=args.confidence_level,
        overwrite=args.overwrite,
    )
    print("Source-grouped feature ablation complete")
    print(f"- Summary: {result['paths']['summary']}")
    print(f"- Paired bootstrap: {result['paths']['paired_bootstrap']}")
    print(f"- Permutation importance: {result['paths']['permutation_summary']}")
    print(f"- Coefficient stability: {result['paths']['coefficient_stability']}")
    print(f"- Report: {result['paths']['report']}")


if __name__ == "__main__":
    main()

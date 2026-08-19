#!/usr/bin/env python3
"""Consolidate the frozen thesis results into one provenance-checked package.

This script is deliberately read-only with respect to experiment outputs.  It
does not train a model or recompute a metric.  It validates the owning checksum
manifest of every source experiment, reads exact values from the frozen CSV or
JSON artifacts, normalizes names and ordering, and writes a deterministic
versioned package for thesis tables and discussion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs/thesis_consolidated_results_v1"

EXPECTED_ROWS = 784
EXPECTED_SOURCE_COMPONENTS = 359
EXPECTED_THRESHOLD = 0.3
EXPECTED_BOOTSTRAP_REPS = 2000
FLOAT_TOLERANCE = 1e-12
REPORTED_DIFFERENCE_TOLERANCE = 1e-9

MODEL_SPECS = (
    (1, "logistic_regression", "Logistic Regression", "lr"),
    (2, "xgboost", "XGBoost", "xgboost"),
    (3, "random_forest", "Random Forest", "random_forest"),
    (4, "bert", "BERT", "bert"),
    (5, "bilstm_attention", "BiLSTM with Attention", "bilstm_attention"),
)
STRUCTURED_MODEL_IDS = ("logistic_regression", "xgboost", "random_forest")
METRICS = ("macro_f1", "macro_average_precision", "macro_roc_auc")
COMPONENT_METRICS = (
    "component_equal_macro_f1",
    "component_equal_macro_average_precision",
    "component_equal_macro_roc_auc",
)
LABEL_SCOPE_ORDER = {"seven_labels": 1, "fear_excluded_six_labels": 2}
METRIC_ORDER = {
    "macro_f1": 1,
    "macro_average_precision": 2,
    "macro_roc_auc": 3,
    "component_equal_macro_f1": 1,
    "component_equal_macro_average_precision": 2,
    "component_equal_macro_roc_auc": 3,
}

VARIANT_SPECS = (
    (1, "full_hybrid", "Full representation", 58, "NRC;VAD;Transformer-derived"),
    (2, "nrc_only", "NRC only", 10, "NRC"),
    (3, "vad_only", "VAD only", 3, "VAD"),
    (4, "transformer_only", "Transformer-derived only", 45, "Transformer-derived"),
    (5, "nrc_plus_vad", "NRC + VAD", 13, "NRC;VAD"),
    (
        6,
        "nrc_plus_transformer",
        "NRC + Transformer-derived",
        55,
        "NRC;Transformer-derived",
    ),
    (
        7,
        "vad_plus_transformer",
        "VAD + Transformer-derived",
        48,
        "VAD;Transformer-derived",
    ),
)
VARIANT_LOOKUP = {
    variant: {
        "variant_order": order,
        "variant_label": label,
        "feature_count": feature_count,
        "feature_groups": feature_groups,
    }
    for order, variant, label, feature_count, feature_groups in VARIANT_SPECS
}

# The older LR analysis already contains the advisor-requested combinations,
# but three of them were stored under earlier implementation-oriented names.
LR_SOURCE_TO_CANONICAL = {
    "full_hybrid": "full_hybrid",
    "nrc_only": "nrc_only",
    "vad_only": "vad_only",
    "transformer_only": "transformer_only",
    "lexicon_only": "nrc_plus_vad",
    "no_vad": "nrc_plus_transformer",
    "no_nrc": "vad_plus_transformer",
}
IDENTITY_SOURCE_TO_CANONICAL = {variant: variant for variant in VARIANT_LOOKUP}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_close(
    observed: float,
    expected: float,
    context: str,
    *,
    atol: float = FLOAT_TOLERANCE,
) -> None:
    if not np.isclose(
        float(observed), float(expected), rtol=0.0, atol=atol, equal_nan=True
    ):
        raise ValueError(f"{context}: observed {observed!r}, expected {expected!r}")


def as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    if value in {0, 1}:
        return bool(value)
    raise ValueError(f"Cannot interpret as boolean: {value!r}")


def interval_direction(lower: float, upper: float, *, positive: str, negative: str) -> str:
    if float(lower) > 0.0:
        return positive
    if float(upper) < 0.0:
        return negative
    return "not_established"


def parse_checksum_manifest(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise ValueError(f"Malformed checksum line {line_number} in {path}")
        relative = parts[1].strip()
        if relative.startswith("*"):
            relative = relative[1:]
        if relative.startswith("./"):
            relative = relative[2:]
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError(f"Unsafe checksum path in {path}: {relative}")
        if relative in entries:
            raise ValueError(f"Duplicate checksum entry in {path}: {relative}")
        entries[relative] = parts[0]
    if not entries:
        raise ValueError(f"Empty checksum manifest: {path}")
    return entries


def verify_checksum_root(
    root: Path, *, expected_artifact_count: int
) -> dict[str, Any]:
    manifest_path = root / "SHA256SUMS"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    entries = parse_checksum_manifest(manifest_path)
    if len(entries) != expected_artifact_count:
        raise ValueError(
            f"{root}: checksum manifest has {len(entries)} entries; "
            f"expected {expected_artifact_count}"
        )
    for relative, expected_hash in entries.items():
        artifact = root / relative
        if not artifact.is_file():
            raise FileNotFoundError(f"Manifest-listed artifact is missing: {artifact}")
        observed_hash = sha256_file(artifact)
        if observed_hash != expected_hash:
            raise ValueError(
                f"Checksum mismatch for {artifact}: {observed_hash} != {expected_hash}"
            )
    return {
        "root": root,
        "manifest_path": manifest_path,
        "manifest_sha256": sha256_file(manifest_path),
        "entries": entries,
        "artifact_count": len(entries),
        "status": "passed",
    }


def experiment_roots(project_root: Path) -> dict[str, tuple[Path, int]]:
    return {
        "accepted_five_model_baseline": (
            project_root / "outputs/gemini_browser_og784_source_grouped_v1",
            95,
        ),
        "weighted_synthetic_lr": (
            project_root / "outputs/source_grouped_weighted_synthetic_lr_v1",
            12,
        ),
        "weighted_synthetic_xgboost": (
            project_root / "outputs/source_grouped_weighted_synthetic_xgboost_v1",
            12,
        ),
        "weighted_synthetic_random_forest": (
            project_root / "outputs/source_grouped_weighted_synthetic_random_forest_v1",
            12,
        ),
        "weighted_synthetic_bert": (
            project_root / "outputs/source_grouped_weighted_synthetic_bert_v1",
            12,
        ),
        "weighted_synthetic_bilstm_attention": (
            project_root
            / "outputs/source_grouped_weighted_synthetic_bilstm_attention_v1",
            12,
        ),
        "feature_combination_logistic_regression": (
            project_root / "outputs/source_grouped_feature_ablation_v1",
            40,
        ),
        "feature_combination_xgboost": (
            project_root
            / "outputs/source_grouped_xgboost_feature_combination_ablation_v1",
            20,
        ),
        "feature_combination_random_forest": (
            project_root
            / "outputs/source_grouped_random_forest_feature_combination_ablation_v1",
            20,
        ),
    }


class FrozenSources:
    """Read registered artifacts only after validating their owning manifests."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root.resolve()
        self.roots: dict[str, dict[str, Any]] = {}
        for source_group, (root, expected_count) in experiment_roots(self.project_root).items():
            self.roots[source_group] = verify_checksum_root(
                root.resolve(), expected_artifact_count=expected_count
            )
        self.consumed: list[dict[str, Any]] = []

    def path(self, source_group: str, role: str, relative: str) -> Path:
        info = self.roots[source_group]
        if relative not in info["entries"]:
            raise ValueError(
                f"{source_group}/{relative} is not owned by its frozen SHA256SUMS"
            )
        path = info["root"] / relative
        observed_hash = sha256_file(path)
        expected_hash = info["entries"][relative]
        if observed_hash != expected_hash:
            raise ValueError(f"Source changed after root verification: {path}")
        self.consumed.append(
            {
                "source_group": source_group,
                "source_role": role,
                "source_path": path.relative_to(self.project_root).as_posix(),
                "sha256": observed_hash,
                "bytes": path.stat().st_size,
                "owning_manifest_path": info["manifest_path"]
                .relative_to(self.project_root)
                .as_posix(),
                "owning_manifest_sha256": info["manifest_sha256"],
                "manifest_entry_sha256": expected_hash,
                "verified_against_owning_manifest": True,
            }
        )
        return path

    def csv(self, source_group: str, role: str, relative: str) -> pd.DataFrame:
        return pd.read_csv(self.path(source_group, role, relative))

    def json(self, source_group: str, role: str, relative: str) -> dict[str, Any]:
        return json.loads(
            self.path(source_group, role, relative).read_text(encoding="utf-8")
        )

    def manifest_frame(self) -> pd.DataFrame:
        frame = pd.DataFrame(self.consumed)
        if frame.empty:
            raise ValueError("No source artifacts were consumed")
        if frame["source_path"].duplicated().any():
            duplicates = frame.loc[frame["source_path"].duplicated(), "source_path"].tolist()
            raise ValueError(f"A source artifact was registered twice: {duplicates}")
        return frame.sort_values(
            ["source_group", "source_role", "source_path"], kind="stable"
        ).reset_index(drop=True)


def model_metadata(model_id: str) -> dict[str, Any]:
    for order, candidate, name, short_id in MODEL_SPECS:
        if candidate == model_id:
            return {
                "model_order": order,
                "model_id": candidate,
                "model_name": name,
                "synthetic_root_id": short_id,
            }
    raise KeyError(model_id)


def read_baseline_and_synthetic(
    sources: FrozenSources,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
]:
    accepted_group = "accepted_five_model_baseline"
    fold_manifest = sources.csv(
        accepted_group,
        "accepted_source_grouped_fold_manifest",
        "model_benchmark/tables/fold_assignments.csv",
    )
    if len(fold_manifest) != EXPECTED_ROWS or fold_manifest["perfume_id"].nunique() != EXPECTED_ROWS:
        raise ValueError("Accepted fold manifest must contain 784 unique perfumes")
    if fold_manifest["source_group_id"].nunique() != EXPECTED_SOURCE_COMPONENTS:
        raise ValueError("Accepted fold manifest must contain 359 source components")
    if fold_manifest.groupby("source_group_id")["fold_id"].nunique().max() != 1:
        raise ValueError("A source component crosses accepted outer folds")
    source_audit = sources.json(
        accepted_group,
        "accepted_source_group_audit",
        "source_audit/source_group_audit.json",
    )
    if int(source_audit.get("n_source_components", -1)) != EXPECTED_SOURCE_COMPONENTS:
        raise ValueError("Accepted source audit must report 359 source components")
    if int(source_audit.get("cross_fold_source_group_count", -1)) != 0:
        raise ValueError("Accepted source audit reports cross-fold source components")
    accepted_component_metrics = sources.csv(
        accepted_group,
        "accepted_component_weighted_model_metrics",
        "model_benchmark/tables/component_weighted_model_metrics.csv",
    )
    if len(accepted_component_metrics) != 5 or set(
        accepted_component_metrics["model_id"]
    ) != {spec[1] for spec in MODEL_SPECS}:
        raise ValueError("Accepted component-weighted metric table must contain five models")

    youtube_rows: list[dict[str, Any]] = []
    youtube_component_rows: list[dict[str, Any]] = []
    synthetic_pooled_rows: list[dict[str, Any]] = []
    synthetic_interval_rows: list[dict[str, Any]] = []
    synthetic_difference_rows: list[dict[str, Any]] = []
    validations: dict[str, Any] = {}

    for _, model_id, model_name, synthetic_root_id in MODEL_SPECS:
        metadata = model_metadata(model_id)
        accepted = sources.json(
            accepted_group,
            f"accepted_{model_id}_metrics",
            f"model_benchmark/metrics/{model_id}.json",
        )
        if accepted["model_id"] != model_id or int(accepted["n_rows"]) != EXPECTED_ROWS:
            raise ValueError(f"Invalid accepted metrics identity/row count for {model_id}")
        assert_close(accepted["threshold"], EXPECTED_THRESHOLD, f"{model_id} threshold")
        accepted_oof = sources.csv(
            accepted_group,
            f"accepted_{model_id}_oof",
            f"model_benchmark/predictions/{model_id}_oof.csv",
        )
        if len(accepted_oof) != EXPECTED_ROWS or accepted_oof["perfume_id"].nunique() != EXPECTED_ROWS:
            raise ValueError(f"{model_id}: accepted OOF must contain 784 unique perfumes")
        if set(accepted_oof["perfume_id"]) != set(fold_manifest["perfume_id"]):
            raise ValueError(f"{model_id}: accepted OOF perfumes differ from fold manifest")

        synthetic_group = f"weighted_synthetic_{synthetic_root_id}"
        summary = sources.csv(
            synthetic_group, "condition_summary", "condition_summary.csv"
        )
        intervals = sources.csv(
            synthetic_group,
            "component_bootstrap_condition_intervals",
            "component_bootstrap_condition_intervals.csv",
        )
        differences = sources.csv(
            synthetic_group,
            "paired_component_bootstrap_differences",
            "paired_component_bootstrap_differences.csv",
        )
        validation = sources.json(
            synthetic_group, "validation_report", "validation_report.json"
        )
        protocol_path = sources.path(
            synthetic_group, "frozen_protocol", "PROTOCOL.yaml"
        )
        protocol_text = protocol_path.read_text(encoding="utf-8")
        if "repetitions: 2000" not in protocol_text or "threshold: 0.3" not in protocol_text:
            raise ValueError(f"{model_id}: frozen synthetic protocol contract is incomplete")
        validations[model_id] = validation

        expected_conditions = {
            "youtube_only",
            "youtube_plus_weighted_synthetic_784x3",
        }
        if len(summary) != 2 or set(summary["condition"]) != expected_conditions:
            raise ValueError(f"{model_id}: expected exactly the two frozen conditions")
        if set(summary["evaluation_rows"].astype(int)) != {EXPECTED_ROWS}:
            raise ValueError(f"{model_id}: condition evaluation rows must be 784")
        if not np.allclose(
            summary["threshold"].astype(float), EXPECTED_THRESHOLD, rtol=0.0, atol=0.0
        ):
            raise ValueError(f"{model_id}: condition thresholds must be 0.3")
        control = summary.set_index("condition").loc["youtube_only"]
        augmented = summary.set_index("condition").loc[
            "youtube_plus_weighted_synthetic_784x3"
        ]

        accepted_values = {
            "macro_f1": accepted["macro_f1"],
            "macro_average_precision": accepted["macro_pr_auc"],
            "macro_roc_auc": accepted["macro_roc_auc"],
        }
        for metric, exact_value in accepted_values.items():
            assert_close(control[metric], exact_value, f"{model_id} accepted {metric}")
        component_reference = accepted_component_metrics.set_index("model_id").loc[model_id]
        for metric in COMPONENT_METRICS:
            accepted_metric = metric.removeprefix("component_equal_")
            assert_close(
                control[metric],
                component_reference[accepted_metric],
                f"{model_id} accepted component-equal {metric}",
            )

        youtube_rows.append(
            {
                **{key: metadata[key] for key in ("model_order", "model_id", "model_name")},
                "input_branch": (
                    "58 structured features"
                    if model_id in STRUCTURED_MODEL_IDS
                    else "review text"
                ),
                "evaluation_rows": EXPECTED_ROWS,
                "threshold": EXPECTED_THRESHOLD,
                **accepted_values,
                "source_condition": "youtube_only",
            }
        )
        for metric in METRICS:
            synthetic_pooled_rows.append(
                {
                    **{
                        key: metadata[key]
                        for key in ("model_order", "model_id", "model_name")
                    },
                    "metric_order": METRIC_ORDER[metric],
                    "metric": metric,
                    "youtube_only": float(control[metric]),
                    "youtube_plus_weighted_synthetic_784x3": float(augmented[metric]),
                    "augmented_minus_youtube_only": float(augmented[metric] - control[metric]),
                    "evaluation_rows": EXPECTED_ROWS,
                    "threshold": EXPECTED_THRESHOLD,
                }
            )

        expected_interval_keys = {
            (scope, condition, metric)
            for scope in LABEL_SCOPE_ORDER
            for condition in expected_conditions
            for metric in COMPONENT_METRICS
        }
        observed_interval_keys = set(
            intervals[["label_scope", "condition", "metric"]].itertuples(
                index=False, name=None
            )
        )
        if observed_interval_keys != expected_interval_keys or len(intervals) != 12:
            raise ValueError(f"{model_id}: incomplete component condition intervals")
        for row in intervals.itertuples(index=False):
            if int(row.bootstrap_reps) != EXPECTED_BOOTSTRAP_REPS:
                raise ValueError(f"{model_id}: bootstrap repetitions must be 2000")
            if int(row.n_source_components) != EXPECTED_SOURCE_COMPONENTS:
                raise ValueError(f"{model_id}: component count must be 359")
            synthetic_interval_rows.append(
                {
                    **{
                        key: metadata[key]
                        for key in ("model_order", "model_id", "model_name")
                    },
                    "label_scope_order": LABEL_SCOPE_ORDER[row.label_scope],
                    "label_scope": row.label_scope,
                    "condition_order": 1 if row.condition == "youtube_only" else 2,
                    "condition": row.condition,
                    "metric_order": METRIC_ORDER[row.metric],
                    "metric": row.metric,
                    "point_estimate": row.point_estimate,
                    "bootstrap_mean": row.bootstrap_mean,
                    "bootstrap_standard_error": row.bootstrap_standard_error,
                    "ci_lower": row.ci_lower,
                    "ci_upper": row.ci_upper,
                    "confidence_level": row.confidence_level,
                    "bootstrap_reps": int(row.bootstrap_reps),
                    "seed": int(row.seed),
                    "n_source_components": int(row.n_source_components),
                    "resampling_unit": "source_group_id",
                    "interval_method": "percentile_linear",
                    "interval_scope": "conditional_on_fitted_outer_fold_models",
                    "multiplicity_adjustment": "none",
                }
            )
            if row.label_scope == "seven_labels" and row.condition == "youtube_only":
                youtube_component_rows.append(synthetic_interval_rows[-1].copy())

        expected_difference_keys = {
            (scope, metric)
            for scope in LABEL_SCOPE_ORDER
            for metric in COMPONENT_METRICS
        }
        observed_difference_keys = set(
            differences[["label_scope", "metric"]].itertuples(index=False, name=None)
        )
        if observed_difference_keys != expected_difference_keys or len(differences) != 6:
            raise ValueError(f"{model_id}: incomplete paired synthetic differences")
        condition_points = {
            (row["label_scope"], row["condition"], row["metric"]): float(
                row["point_estimate"]
            )
            for _, row in intervals.iterrows()
        }
        for row in differences.itertuples(index=False):
            expected_difference = condition_points[
                (row.label_scope, "youtube_plus_weighted_synthetic_784x3", row.metric)
            ] - condition_points[(row.label_scope, "youtube_only", row.metric)]
            assert_close(
                row.point_estimate_difference,
                expected_difference,
                f"{model_id}/{row.label_scope}/{row.metric} synthetic difference sign",
                atol=REPORTED_DIFFERENCE_TOLERANCE,
            )
            if int(row.bootstrap_reps) != EXPECTED_BOOTSTRAP_REPS:
                raise ValueError(f"{model_id}: bootstrap repetitions must be 2000")
            if int(row.n_source_components) != EXPECTED_SOURCE_COMPONENTS:
                raise ValueError(f"{model_id}: component count must be 359")
            computed_excludes_zero = float(row.ci_lower) > 0.0 or float(row.ci_upper) < 0.0
            if as_bool(row.ci_excludes_zero) != computed_excludes_zero:
                raise ValueError(f"{model_id}/{row.label_scope}/{row.metric}: invalid CI flag")
            if as_bool(row.positive_improvement_established) != (float(row.ci_lower) > 0.0):
                raise ValueError(
                    f"{model_id}/{row.label_scope}/{row.metric}: invalid positive-effect rule"
                )
            synthetic_difference_rows.append(
                {
                    **{
                        key: metadata[key]
                        for key in ("model_order", "model_id", "model_name")
                    },
                    "label_scope_order": LABEL_SCOPE_ORDER[row.label_scope],
                    "label_scope": row.label_scope,
                    "metric_order": METRIC_ORDER[row.metric],
                    "metric": row.metric,
                    "difference_definition": row.difference_definition,
                    "point_estimate_difference": row.point_estimate_difference,
                    "bootstrap_mean_difference": row.bootstrap_mean_difference,
                    "bootstrap_standard_error": row.bootstrap_standard_error,
                    "ci_lower": row.ci_lower,
                    "ci_upper": row.ci_upper,
                    "ci_excludes_zero": computed_excludes_zero,
                    "positive_improvement_established": float(row.ci_lower) > 0.0,
                    "established_direction": interval_direction(
                        row.ci_lower,
                        row.ci_upper,
                        positive="augmentation_better",
                        negative="augmentation_worse",
                    ),
                    "confidence_level": row.confidence_level,
                    "bootstrap_reps": int(row.bootstrap_reps),
                    "seed": int(row.seed),
                    "n_source_components": int(row.n_source_components),
                    "defined_label_count_point": int(row.defined_label_count_point),
                    "defined_label_count_min": int(row.defined_label_count_min),
                    "defined_label_count_max": int(row.defined_label_count_max),
                    "resampling_unit": "source_group_id",
                    "interval_method": "percentile_linear",
                    "interval_scope": "conditional_on_fitted_outer_fold_models",
                    "multiplicity_adjustment": "none",
                }
            )

        if validation.get("status") != "passed":
            raise ValueError(f"{model_id}: synthetic experiment validation did not pass")
        bootstrap_validation = validation.get("bootstrap", {})
        if int(bootstrap_validation.get("n_source_components", -1)) != EXPECTED_SOURCE_COMPONENTS:
            raise ValueError(f"{model_id}: validation does not report 359 components")

    youtube = pd.DataFrame(youtube_rows).sort_values("model_order", kind="stable")
    youtube_component = pd.DataFrame(youtube_component_rows).sort_values(
        ["model_order", "metric_order"], kind="stable"
    )
    synthetic_pooled = pd.DataFrame(synthetic_pooled_rows).sort_values(
        ["model_order", "metric_order"], kind="stable"
    )
    synthetic_intervals = pd.DataFrame(synthetic_interval_rows).sort_values(
        ["model_order", "label_scope_order", "condition_order", "metric_order"],
        kind="stable",
    )
    synthetic_differences = pd.DataFrame(synthetic_difference_rows).sort_values(
        ["model_order", "label_scope_order", "metric_order"], kind="stable"
    )
    return (
        youtube.reset_index(drop=True),
        youtube_component.reset_index(drop=True),
        synthetic_pooled.reset_index(drop=True),
        synthetic_intervals.reset_index(drop=True),
        synthetic_differences.reset_index(drop=True),
        validations,
    )


def normalize_ablation_summary(
    frame: pd.DataFrame,
    *,
    model_id: str,
    source_map: dict[str, str],
    feature_catalog: pd.DataFrame,
) -> pd.DataFrame:
    selected = frame.loc[frame["variant"].isin(source_map)].copy()
    selected["source_variant"] = selected["variant"]
    catalog = feature_catalog.loc[feature_catalog["variant"].isin(source_map)].copy()
    if len(catalog) != 7 or catalog["variant"].duplicated().any():
        raise ValueError(f"{model_id}: frozen feature-column catalog is incomplete")
    catalog = catalog.rename(
        columns={
            "variant": "source_variant",
            "variant_label": "source_variant_label",
            "feature_count": "catalog_feature_count",
            "included_families": "source_feature_families",
            "feature_columns_sha256": "source_feature_columns_sha256",
        }
    )
    selected = selected.merge(
        catalog[
            [
                "source_variant",
                "source_variant_label",
                "catalog_feature_count",
                "source_feature_families",
                "feature_columns",
                "source_feature_columns_sha256",
            ]
        ],
        on="source_variant",
        how="left",
        validate="one_to_one",
    )
    if not (
        selected["feature_count"].astype(int)
        == selected["catalog_feature_count"].astype(int)
    ).all():
        raise ValueError(f"{model_id}: summary and feature-column catalog counts differ")
    observed_column_counts = selected["feature_columns"].map(
        lambda value: len(str(value).split(";")) if str(value) else 0
    )
    if not (observed_column_counts == selected["feature_count"].astype(int)).all():
        raise ValueError(f"{model_id}: listed feature columns do not match feature_count")
    selected["variant"] = selected["source_variant"].map(source_map)
    if len(selected) != 7 or set(selected["variant"]) != set(VARIANT_LOOKUP):
        raise ValueError(f"{model_id}: expected all seven canonical feature combinations")
    metadata = model_metadata(model_id)
    selected.insert(0, "model_order", metadata["model_order"])
    selected.insert(1, "model_id", model_id)
    selected.insert(2, "model_name", metadata["model_name"])
    selected["variant_order"] = selected["variant"].map(
        lambda value: VARIANT_LOOKUP[value]["variant_order"]
    )
    selected["variant_label"] = selected["variant"].map(
        lambda value: VARIANT_LOOKUP[value]["variant_label"]
    )
    selected["feature_groups"] = selected["variant"].map(
        lambda value: VARIANT_LOOKUP[value]["feature_groups"]
    )
    selected["expected_feature_count"] = selected["variant"].map(
        lambda value: VARIANT_LOOKUP[value]["feature_count"]
    )
    if not (
        selected["feature_count"].astype(int)
        == selected["expected_feature_count"].astype(int)
    ).all():
        raise ValueError(f"{model_id}: feature counts differ from the frozen canonical design")
    if "evaluation_rows" not in selected:
        selected["evaluation_rows"] = EXPECTED_ROWS
    if set(selected["evaluation_rows"].astype(int)) != {EXPECTED_ROWS}:
        raise ValueError(f"{model_id}: every feature combination must evaluate 784 rows")
    if not np.allclose(
        selected["threshold"].astype(float), EXPECTED_THRESHOLD, rtol=0.0, atol=0.0
    ):
        raise ValueError(f"{model_id}: every feature combination must use threshold 0.3")
    columns = [
        "model_order",
        "model_id",
        "model_name",
        "variant_order",
        "variant",
        "variant_label",
        "source_variant",
        "source_variant_label",
        "feature_count",
        "feature_groups",
        "source_feature_families",
        "source_feature_columns_sha256",
        "evaluation_rows",
        "threshold",
        *METRICS,
        "micro_f1",
        "hamming_loss",
        "component_equal_macro_f1",
        "component_equal_macro_average_precision",
        "component_equal_macro_roc_auc",
    ]
    if "predicted_label_cardinality" in selected:
        columns.insert(columns.index("component_equal_macro_f1"), "predicted_label_cardinality")
    elif "mean_predicted_labels" in selected:
        selected = selected.rename(columns={"mean_predicted_labels": "predicted_label_cardinality"})
        columns.insert(columns.index("component_equal_macro_f1"), "predicted_label_cardinality")
    return selected[columns].sort_values("variant_order", kind="stable").reset_index(drop=True)


def normalize_ablation_intervals(
    frame: pd.DataFrame,
    *,
    model_id: str,
    source_map: dict[str, str],
    label_scope: str | None = None,
) -> pd.DataFrame:
    normalized = frame.loc[frame["variant"].isin(source_map)].copy()
    normalized["source_variant"] = normalized["variant"]
    normalized["variant"] = normalized["source_variant"].map(source_map)
    if label_scope is not None:
        normalized.insert(0, "label_scope", label_scope)
    expected_keys = {
        (scope, variant, metric)
        for scope in LABEL_SCOPE_ORDER
        for variant in VARIANT_LOOKUP
        for metric in COMPONENT_METRICS
    }
    metadata = model_metadata(model_id)
    normalized.insert(0, "model_order", metadata["model_order"])
    normalized.insert(1, "model_id", model_id)
    normalized.insert(2, "model_name", metadata["model_name"])
    normalized["label_scope_order"] = normalized["label_scope"].map(LABEL_SCOPE_ORDER)
    normalized["variant_order"] = normalized["variant"].map(
        lambda value: VARIANT_LOOKUP[value]["variant_order"]
    )
    normalized["metric_order"] = normalized["metric"].map(METRIC_ORDER)
    observed_keys = set(
        normalized[["label_scope", "variant", "metric"]].itertuples(
            index=False, name=None
        )
    )
    if observed_keys != expected_keys or len(normalized) != 42:
        raise ValueError(f"{model_id}: feature-combination condition intervals are incomplete")
    if set(normalized["bootstrap_reps"].astype(int)) != {EXPECTED_BOOTSTRAP_REPS}:
        raise ValueError(f"{model_id}: ablation intervals must use 2000 bootstrap reps")
    if set(normalized["n_source_components"].astype(int)) != {EXPECTED_SOURCE_COMPONENTS}:
        raise ValueError(f"{model_id}: ablation intervals must use 359 components")
    normalized["resampling_unit"] = "source_group_id"
    normalized["interval_method"] = "percentile_linear"
    normalized["interval_scope"] = "conditional_on_fitted_outer_fold_models"
    normalized["multiplicity_adjustment"] = "none"
    columns = [
        "model_order",
        "model_id",
        "model_name",
        "label_scope_order",
        "label_scope",
        "variant_order",
        "variant",
        "source_variant",
        "metric_order",
        "metric",
        "point_estimate",
        "bootstrap_mean",
        "bootstrap_standard_error",
        "confidence_level",
        "ci_lower",
        "ci_upper",
        "bootstrap_reps",
        "seed",
        "n_source_components",
        "defined_label_count_point",
        "defined_label_count_min",
        "defined_label_count_max",
        "resampling_unit",
        "interval_method",
        "interval_scope",
        "multiplicity_adjustment",
    ]
    return normalized[columns]


def normalize_ablation_differences(
    frame: pd.DataFrame,
    *,
    model_id: str,
    source_map: dict[str, str],
    label_scope: str | None = None,
) -> pd.DataFrame:
    comparison_map = {key: value for key, value in source_map.items() if value != "full_hybrid"}
    normalized = frame.loc[frame["comparison_variant"].isin(comparison_map)].copy()
    normalized["source_variant"] = normalized["comparison_variant"]
    normalized["comparison_variant"] = normalized["source_variant"].map(comparison_map)
    if label_scope is not None:
        normalized.insert(0, "label_scope", label_scope)
    metadata = model_metadata(model_id)
    normalized.insert(0, "model_order", metadata["model_order"])
    normalized.insert(1, "model_id", model_id)
    normalized.insert(2, "model_name", metadata["model_name"])
    normalized["label_scope_order"] = normalized["label_scope"].map(LABEL_SCOPE_ORDER)
    normalized["variant_order"] = normalized["comparison_variant"].map(
        lambda value: VARIANT_LOOKUP[value]["variant_order"]
    )
    normalized["metric_order"] = normalized["metric"].map(METRIC_ORDER)
    expected_keys = {
        (scope, variant, metric)
        for scope in LABEL_SCOPE_ORDER
        for variant in VARIANT_LOOKUP
        if variant != "full_hybrid"
        for metric in COMPONENT_METRICS
    }
    observed_keys = set(
        normalized[["label_scope", "comparison_variant", "metric"]].itertuples(
            index=False, name=None
        )
    )
    if observed_keys != expected_keys or len(normalized) != 36:
        raise ValueError(f"{model_id}: feature-combination paired effects are incomplete")
    if set(normalized["difference_definition"]) != {"full_hybrid_minus_comparison"}:
        raise ValueError(f"{model_id}: unexpected feature-combination difference direction")
    if set(normalized["bootstrap_reps"].astype(int)) != {EXPECTED_BOOTSTRAP_REPS}:
        raise ValueError(f"{model_id}: ablation differences must use 2000 bootstrap reps")
    if set(normalized["n_source_components"].astype(int)) != {EXPECTED_SOURCE_COMPONENTS}:
        raise ValueError(f"{model_id}: ablation differences must use 359 components")
    computed_excludes = normalized.apply(
        lambda row: float(row["ci_lower"]) > 0.0 or float(row["ci_upper"]) < 0.0,
        axis=1,
    )
    recorded_excludes = normalized["ci_excludes_zero"].map(as_bool)
    if not (computed_excludes == recorded_excludes).all():
        raise ValueError(f"{model_id}: ablation CI exclusion flags are inconsistent")
    normalized["ci_excludes_zero"] = computed_excludes
    normalized["established_direction"] = normalized.apply(
        lambda row: interval_direction(
            row["ci_lower"],
            row["ci_upper"],
            positive="full_better",
            negative="comparison_better",
        ),
        axis=1,
    )
    normalized["interval_method"] = "percentile_linear"
    normalized["interval_scope"] = "conditional_on_fitted_outer_fold_models"
    normalized["multiplicity_adjustment"] = "none"
    columns = [
        "model_order",
        "model_id",
        "model_name",
        "label_scope_order",
        "label_scope",
        "variant_order",
        "reference_variant",
        "comparison_variant",
        "source_variant",
        "metric_order",
        "metric",
        "difference_definition",
        "point_estimate_difference",
        "bootstrap_mean_difference",
        "bootstrap_standard_error",
        "confidence_level",
        "ci_lower",
        "ci_upper",
        "ci_excludes_zero",
        "established_direction",
        "bootstrap_reps",
        "seed",
        "n_source_components",
        "resampling_unit",
        "defined_label_count_min",
        "defined_label_count_max",
        "interval_method",
        "interval_scope",
        "multiplicity_adjustment",
    ]
    return normalized[columns]


def read_ablation(
    sources: FrozenSources,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    summaries: list[pd.DataFrame] = []
    intervals: list[pd.DataFrame] = []
    differences: list[pd.DataFrame] = []
    validations: dict[str, Any] = {}

    lr_group = "feature_combination_logistic_regression"
    lr_summary = sources.csv(lr_group, "ablation_summary", "ablation_summary.csv")
    lr_feature_catalog = sources.csv(
        lr_group, "feature_columns_by_variant", "feature_columns_by_variant.csv"
    )
    lr_intervals_seven = sources.csv(
        lr_group, "seven_label_component_intervals", "model_metric_bootstrap_intervals.csv"
    )
    lr_intervals_six = sources.csv(
        lr_group,
        "fear_excluded_component_intervals",
        "fear_excluded_model_metric_bootstrap_intervals.csv",
    )
    lr_differences_seven = sources.csv(
        lr_group, "seven_label_paired_differences", "paired_bootstrap_vs_full.csv"
    )
    lr_differences_six = sources.csv(
        lr_group,
        "fear_excluded_paired_differences",
        "fear_excluded_paired_bootstrap_vs_full.csv",
    )
    lr_report = sources.json(lr_group, "validation_report", "REPORT.json")
    if lr_report.get("status") != "completed":
        raise ValueError("Logistic Regression feature-combination analysis is not completed")
    if int(lr_report["design"]["n_rows"]) != EXPECTED_ROWS:
        raise ValueError("LR ablation must use 784 rows")
    if int(lr_report["design"]["n_source_components"]) != EXPECTED_SOURCE_COMPONENTS:
        raise ValueError("LR ablation must use 359 source components")
    validations["logistic_regression"] = lr_report
    summaries.append(
        normalize_ablation_summary(
            lr_summary,
            model_id="logistic_regression",
            source_map=LR_SOURCE_TO_CANONICAL,
            feature_catalog=lr_feature_catalog,
        )
    )
    intervals.append(
        normalize_ablation_intervals(
            pd.concat(
                [
                    lr_intervals_seven.assign(label_scope="seven_labels"),
                    lr_intervals_six.assign(label_scope="fear_excluded_six_labels"),
                ],
                ignore_index=True,
            ),
            model_id="logistic_regression",
            source_map=LR_SOURCE_TO_CANONICAL,
        )
    )
    differences.append(
        normalize_ablation_differences(
            pd.concat(
                [
                    lr_differences_seven.assign(label_scope="seven_labels"),
                    lr_differences_six.assign(label_scope="fear_excluded_six_labels"),
                ],
                ignore_index=True,
            ),
            model_id="logistic_regression",
            source_map=LR_SOURCE_TO_CANONICAL,
        )
    )

    for model_id in ("xgboost", "random_forest"):
        group = f"feature_combination_{model_id}"
        summary = sources.csv(group, "variant_summary", "variant_summary.csv")
        feature_catalog = sources.csv(
            group, "feature_columns_by_variant", "feature_columns_by_variant.csv"
        )
        interval = sources.csv(
            group,
            "component_bootstrap_variant_intervals",
            "component_bootstrap_variant_intervals.csv",
        )
        difference = sources.csv(
            group,
            "paired_component_bootstrap_vs_full",
            "paired_component_bootstrap_vs_full.csv",
        )
        validation = sources.json(group, "validation_report", "validation_report.json")
        formal_report = sources.json(group, "formal_report", "REPORT.json")
        protocol_path = sources.path(group, "frozen_protocol", "PROTOCOL.yaml")
        protocol_text = protocol_path.read_text(encoding="utf-8")
        if "repetitions: 2000" not in protocol_text or "threshold: 0.3" not in protocol_text:
            raise ValueError(f"{model_id}: frozen protocol contract is incomplete")
        if formal_report.get("status") not in {"completed", "passed"}:
            raise ValueError(f"{model_id}: formal ablation report is not complete/passed")
        validations[model_id] = validation
        if validation.get("status") != "passed":
            raise ValueError(f"{model_id}: feature-combination validation did not pass")
        if int(validation.get("rows_per_variant", -1)) != EXPECTED_ROWS:
            raise ValueError(f"{model_id}: ablation must evaluate 784 rows per variant")
        if int(validation.get("source_components", -1)) != EXPECTED_SOURCE_COMPONENTS:
            raise ValueError(f"{model_id}: ablation must use 359 source components")
        assert_close(
            validation.get("threshold"), EXPECTED_THRESHOLD, f"{model_id} ablation threshold"
        )
        summaries.append(
            normalize_ablation_summary(
                summary, model_id=model_id, source_map=IDENTITY_SOURCE_TO_CANONICAL
                , feature_catalog=feature_catalog
            )
        )
        intervals.append(
            normalize_ablation_intervals(
                interval, model_id=model_id, source_map=IDENTITY_SOURCE_TO_CANONICAL
            )
        )
        differences.append(
            normalize_ablation_differences(
                difference, model_id=model_id, source_map=IDENTITY_SOURCE_TO_CANONICAL
            )
        )

    pooled = pd.concat(summaries, ignore_index=True).sort_values(
        ["model_order", "variant_order"], kind="stable"
    )
    cross_model_hash_counts = pooled.groupby("variant")[
        "source_feature_columns_sha256"
    ].nunique()
    if not (cross_model_hash_counts == 1).all():
        mismatches = cross_model_hash_counts.loc[cross_model_hash_counts.ne(1)].to_dict()
        raise ValueError(
            "Canonical feature columns differ across LR/XGBoost/Random Forest: "
            f"{mismatches}"
        )
    component_intervals = pd.concat(intervals, ignore_index=True).sort_values(
        ["model_order", "label_scope_order", "variant_order", "metric_order"],
        kind="stable",
    )
    component_differences = pd.concat(differences, ignore_index=True).sort_values(
        ["model_order", "label_scope_order", "variant_order", "metric_order"],
        kind="stable",
    )

    interval_points = {
        (row.model_id, row.label_scope, row.variant, row.metric): row.point_estimate
        for row in component_intervals.itertuples(index=False)
    }
    for row in component_differences.itertuples(index=False):
        expected = interval_points[
            (row.model_id, row.label_scope, "full_hybrid", row.metric)
        ] - interval_points[
            (row.model_id, row.label_scope, row.comparison_variant, row.metric)
        ]
        assert_close(
            row.point_estimate_difference,
            expected,
            f"{row.model_id}/{row.label_scope}/{row.comparison_variant}/{row.metric} sign",
            atol=REPORTED_DIFFERENCE_TOLERANCE,
        )

    if len(pooled) != 21 or pooled[["model_id", "variant"]].duplicated().any():
        raise ValueError("Expected 21 unique model-feature pooled rows")
    if len(component_intervals) != 126 or component_intervals[
        ["model_id", "label_scope", "variant", "metric"]
    ].duplicated().any():
        raise ValueError("Expected 126 unique model-feature component intervals")
    if len(component_differences) != 108 or component_differences[
        ["model_id", "label_scope", "comparison_variant", "metric"]
    ].duplicated().any():
        raise ValueError("Expected 108 unique model-feature paired effects")
    return (
        pooled.reset_index(drop=True),
        component_intervals.reset_index(drop=True),
        component_differences.reset_index(drop=True),
        validations,
    )


def records_for_json(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records"))


def dataframe_to_markdown(frame: pd.DataFrame, *, digits: int = 4) -> str:
    def render(value: Any) -> str:
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.{digits}f}"
        return str(value)

    lines = [
        "| " + " | ".join(str(column) for column in frame.columns) + " |",
        "| " + " | ".join("---" for _ in frame.columns) + " |",
    ]
    lines.extend(
        "| " + " | ".join(render(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def write_readme(
    path: Path,
    *,
    youtube: pd.DataFrame,
    synthetic_pooled: pd.DataFrame,
    synthetic_differences: pd.DataFrame,
    ablation_pooled: pd.DataFrame,
) -> None:
    pooled_wide = synthetic_pooled.pivot(
        index=["model_order", "model_name"],
        columns="metric",
        values=["youtube_only", "youtube_plus_weighted_synthetic_784x3"],
    ).reset_index()
    pooled_wide.columns = [
        "model_order",
        "Model",
        "YouTube AP",
        "YouTube F1",
        "YouTube ROC-AUC",
        "Augmented AP",
        "Augmented F1",
        "Augmented ROC-AUC",
    ]
    pooled_wide = pooled_wide.sort_values("model_order").drop(columns="model_order")

    primary_effects = synthetic_differences.loc[
        synthetic_differences["label_scope"].eq("seven_labels"),
        [
            "model_name",
            "metric",
            "point_estimate_difference",
            "ci_lower",
            "ci_upper",
            "ci_excludes_zero",
        ],
    ].rename(
        columns={
            "model_name": "Model",
            "metric": "Metric",
            "point_estimate_difference": "Augmented - YouTube",
            "ci_lower": "95% CI lower",
            "ci_upper": "95% CI upper",
            "ci_excludes_zero": "CI excludes zero",
        }
    )
    ablation_view = ablation_pooled[
        [
            "model_name",
            "variant_label",
            "macro_f1",
            "macro_average_precision",
            "macro_roc_auc",
        ]
    ].rename(
        columns={
            "model_name": "Model",
            "variant_label": "Feature combination",
            "macro_f1": "Macro F1",
            "macro_average_precision": "Macro AP",
            "macro_roc_auc": "Macro ROC-AUC",
        }
    )

    lines = [
        "# Thesis Consolidated Results v1",
        "",
        "This package consolidates exact values from the frozen source-grouped experiments. It does not retrain models or recompute metrics. All test rows are the same 784 YouTube perfumes; source-grouped uncertainty uses 359 source components; binary predictions use threshold 0.3; confidence intervals use 2,000 component bootstrap repetitions.",
        "",
        "## Files",
        "",
        "- `youtube_five_model_pooled.csv`: exact accepted pooled metrics for the five YouTube-only models and their input branch.",
        "- `youtube_five_model_component_equal.csv`: seven-label component-equal metrics and 95% intervals for the five YouTube-only models.",
        "- `synthetic_five_model_augmented_pooled.csv`: exact pooled metrics for the five models trained with three weighted synthetic reviews per training perfume.",
        "- `synthetic_five_model_pooled.csv`: pooled YouTube-only, augmented, and augmented-minus-control values for three metrics.",
        "- `synthetic_five_model_component_intervals.csv`: both conditions, both label scopes, three component-equal metrics, and their intervals.",
        "- `synthetic_five_model_component_differences.csv`: paired augmented-minus-control effects and intervals for both label scopes.",
        "- `feature_combination_three_model_pooled.csv`: seven canonical feature combinations for LR, XGBoost, and Random Forest.",
        "- `feature_combination_three_model_component_intervals.csv`: component-equal interval estimates for all structured-model combinations and both scopes.",
        "- `feature_combination_three_model_component_differences.csv`: paired full-minus-comparison effects for six advisor-requested combinations.",
        "- `SOURCE_MANIFEST.csv`, `VALIDATION.json`, `REPORT.json`, and `SHA256SUMS`: provenance and integrity records.",
        "",
        "## Five-model pooled results",
        "",
        dataframe_to_markdown(pooled_wide),
        "",
        "## Paired synthetic effects (primary seven-label scope)",
        "",
        dataframe_to_markdown(primary_effects),
        "",
        "A positive difference favors weighted synthetic augmentation. An interval that includes zero does not establish a directional difference at the stated 95% level.",
        "",
        "## Structured-model feature-combination results",
        "",
        dataframe_to_markdown(ablation_view),
        "",
        "This feature-combination analysis applies to Logistic Regression, XGBoost, and Random Forest because those models consume the 58-column structured representation. The current BERT and BiLSTM-with-Attention models consume review text directly; applying these NRC/VAD/transformer-derived combinations to them would require new fusion architectures rather than an ablation of the fitted text-only models.",
        "",
        "## Interpretation boundary",
        "",
        "Gemini labels are model-based reference targets, not human ground truth. Bootstrap intervals quantify source-component sampling uncertainty conditional on the fitted outer-fold models; they do not include retraining uncertainty or multiplicity adjustment.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    # Pandas/Python use the shortest decimal string that round-trips to the
    # same binary float. This preserves exact stored values without rendering
    # protocol constants such as 0.3 as 0.29999999999999999.
    frame.to_csv(path, index=False, lineterminator="\n")


def write_checksums(output_root: Path) -> dict[str, str]:
    checksum_path = output_root / "SHA256SUMS"
    files = sorted(
        path
        for path in output_root.iterdir()
        if path.is_file() and path.name != checksum_path.name
    )
    entries = {path.name: sha256_file(path) for path in files}
    checksum_path.write_text(
        "".join(f"{digest}  ./{name}\n" for name, digest in entries.items()),
        encoding="utf-8",
    )
    return entries


def verify_output_checksums(output_root: Path) -> dict[str, Any]:
    manifest_path = output_root / "SHA256SUMS"
    entries = parse_checksum_manifest(manifest_path)
    actual_files = {
        path.name
        for path in output_root.iterdir()
        if path.is_file() and path.name != "SHA256SUMS"
    }
    if set(entries) != actual_files:
        raise ValueError("Consolidated output file set differs from SHA256SUMS")
    for relative, expected in entries.items():
        if sha256_file(output_root / relative) != expected:
            raise ValueError(f"Consolidated checksum mismatch: {relative}")
    return {
        "status": "passed",
        "artifact_count_excluding_checksum_manifest": len(entries),
        "checksum_manifest_sha256": sha256_file(manifest_path),
    }


def build_package(
    *,
    project_root: Path = PROJECT_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    output_root = output_root.resolve()
    protected_roots = [root.resolve() for root, _ in experiment_roots(project_root).values()]
    if output_root in protected_roots or any(root in output_root.parents for root in protected_roots):
        raise ValueError("Output root collides with a protected source experiment")
    output_root.mkdir(parents=True, exist_ok=True)
    unexpected = [path for path in output_root.iterdir() if not path.is_file()]
    if unexpected:
        raise ValueError(f"Output root contains unexpected directories: {unexpected}")

    sources = FrozenSources(project_root)
    (
        youtube,
        youtube_component,
        synthetic_pooled,
        synthetic_intervals,
        synthetic_differences,
        synthetic_validations,
    ) = read_baseline_and_synthetic(sources)
    (
        ablation_pooled,
        ablation_intervals,
        ablation_differences,
        ablation_validations,
    ) = read_ablation(sources)

    augmented = (
        synthetic_pooled.pivot(
            index=["model_order", "model_id", "model_name"],
            columns="metric",
            values="youtube_plus_weighted_synthetic_784x3",
        )
        .reset_index()
        .merge(
            youtube[["model_id", "input_branch"]],
            on="model_id",
            how="left",
            validate="one_to_one",
        )
        .sort_values("model_order", kind="stable")
    )
    augmented["condition"] = "youtube_plus_weighted_synthetic_784x3"
    augmented["synthetic_training_reviews_per_perfume"] = 3
    augmented["evaluation_rows"] = EXPECTED_ROWS
    augmented["threshold"] = EXPECTED_THRESHOLD
    augmented = augmented[
        [
            "model_order",
            "model_id",
            "model_name",
            "input_branch",
            "condition",
            "synthetic_training_reviews_per_perfume",
            "evaluation_rows",
            "threshold",
            *METRICS,
        ]
    ].reset_index(drop=True)

    expected_counts = {
        "youtube_five_model_pooled.csv": 5,
        "youtube_five_model_component_equal.csv": 15,
        "synthetic_five_model_augmented_pooled.csv": 5,
        "synthetic_five_model_pooled.csv": 15,
        "synthetic_five_model_component_intervals.csv": 60,
        "synthetic_five_model_component_differences.csv": 30,
        "feature_combination_three_model_pooled.csv": 21,
        "feature_combination_three_model_component_intervals.csv": 126,
        "feature_combination_three_model_component_differences.csv": 108,
    }
    frames = {
        "youtube_five_model_pooled.csv": youtube,
        "youtube_five_model_component_equal.csv": youtube_component,
        "synthetic_five_model_augmented_pooled.csv": augmented,
        "synthetic_five_model_pooled.csv": synthetic_pooled,
        "synthetic_five_model_component_intervals.csv": synthetic_intervals,
        "synthetic_five_model_component_differences.csv": synthetic_differences,
        "feature_combination_three_model_pooled.csv": ablation_pooled,
        "feature_combination_three_model_component_intervals.csv": ablation_intervals,
        "feature_combination_three_model_component_differences.csv": ablation_differences,
    }
    for filename, frame in frames.items():
        if len(frame) != expected_counts[filename]:
            raise ValueError(f"{filename}: unexpected row count {len(frame)}")
        write_csv(frame, output_root / filename)

    source_manifest = sources.manifest_frame()
    write_csv(source_manifest, output_root / "SOURCE_MANIFEST.csv")

    root_verification = {
        group: {
            "root": info["root"].relative_to(project_root).as_posix(),
            "checksum_manifest": info["manifest_path"]
            .relative_to(project_root)
            .as_posix(),
            "checksum_manifest_sha256": info["manifest_sha256"],
            "artifact_count": info["artifact_count"],
            "status": info["status"],
        }
        for group, info in sources.roots.items()
    }
    validation = {
        "package_id": "thesis_consolidated_results_v1",
        "status": "passed",
        "operation": "read_only_consolidation_of_frozen_metrics",
        "builder_script": Path(__file__).resolve().relative_to(project_root).as_posix(),
        "builder_script_sha256": sha256_file(Path(__file__).resolve()),
        "fixed_contract": {
            "evaluation_rows": EXPECTED_ROWS,
            "source_components": EXPECTED_SOURCE_COMPONENTS,
            "threshold": EXPECTED_THRESHOLD,
            "bootstrap_reps": EXPECTED_BOOTSTRAP_REPS,
            "confidence_level": 0.95,
            "synthetic_difference_definition": (
                "youtube_plus_weighted_synthetic_784x3_minus_youtube_only"
            ),
            "ablation_difference_definition": "full_hybrid_minus_comparison",
        },
        "table_row_counts": expected_counts,
        "unique_key_checks": {
            "youtube_models": int(youtube["model_id"].nunique()) == 5,
            "youtube_component_model_metric": not youtube_component[
                ["model_id", "metric"]
            ].duplicated().any(),
            "synthetic_pooled_model_metric": not synthetic_pooled[
                ["model_id", "metric"]
            ].duplicated().any(),
            "synthetic_effect_model_scope_metric": not synthetic_differences[
                ["model_id", "label_scope", "metric"]
            ].duplicated().any(),
            "ablation_model_variant": not ablation_pooled[
                ["model_id", "variant"]
            ].duplicated().any(),
            "ablation_interval_model_scope_variant_metric": not ablation_intervals[
                ["model_id", "label_scope", "variant", "metric"]
            ].duplicated().any(),
            "ablation_effect_model_scope_variant_metric": not ablation_differences[
                ["model_id", "label_scope", "comparison_variant", "metric"]
            ].duplicated().any(),
        },
        "sign_checks": {
            "synthetic_effects_equal_augmented_minus_control": True,
            "ablation_effects_equal_full_minus_comparison": True,
        },
        "accepted_baseline_control_checks": {
            "five_weighted_experiment_controls_match_accepted_metrics": True,
            "metric_absolute_tolerance": FLOAT_TOLERANCE,
        },
        "feature_combination_checks": {
            "cross_model_feature_columns_sha256_identical_by_variant": bool(
                (
                    ablation_pooled.groupby("variant")[
                        "source_feature_columns_sha256"
                    ].nunique()
                    == 1
                ).all()
            ),
            "canonical_variant_count": int(ablation_pooled["variant"].nunique()),
            "structured_model_count": int(ablation_pooled["model_id"].nunique()),
        },
        "source_manifest": {
            "consumed_artifact_count": int(len(source_manifest)),
            "all_consumed_artifacts_match_owning_manifest": bool(
                source_manifest["verified_against_owning_manifest"].all()
            ),
            "source_root_verification": root_verification,
        },
        "synthetic_experiment_status": {
            model_id: report.get("status")
            for model_id, report in synthetic_validations.items()
        },
        "feature_combination_experiment_status": {
            model_id: report.get("status")
            for model_id, report in ablation_validations.items()
        },
    }
    if not all(validation["unique_key_checks"].values()):
        raise ValueError("A consolidated table failed its unique-key check")
    (output_root / "VALIDATION.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    primary_synthetic_effects = synthetic_differences.loc[
        synthetic_differences["label_scope"].eq("seven_labels")
    ]
    report = {
        "package_id": "thesis_consolidated_results_v1",
        "status": "completed",
        "purpose": "single verified source for thesis result tables and discussion",
        "experiments_rerun": False,
        "docx_modified": False,
        "contents": {
            filename: {"rows": int(len(frame)), "columns": list(frame.columns)}
            for filename, frame in frames.items()
        },
        "primary_scope_summary": {
            "synthetic_effect_intervals_excluding_zero": records_for_json(
                primary_synthetic_effects.loc[
                    primary_synthetic_effects["ci_excludes_zero"],
                    [
                        "model_id",
                        "metric",
                        "point_estimate_difference",
                        "ci_lower",
                        "ci_upper",
                    ],
                ]
            ),
            "ablation_effect_interval_count_excluding_zero_unadjusted": int(
                ablation_differences.loc[
                    ablation_differences["label_scope"].eq("seven_labels"),
                    "ci_excludes_zero",
                ].sum()
            ),
        },
        "scope_notes": [
            "Synthetic effects are augmented minus YouTube-only; positive values favor augmentation.",
            "Feature-combination effects are full representation minus comparison; positive values favor the full representation.",
            "BERT and BiLSTM with Attention are text-only and are not part of the structured-feature ablation.",
            "Intervals are unadjusted 95% component-bootstrap intervals conditional on fitted outer-fold models.",
        ],
    }
    (output_root / "REPORT.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_readme(
        output_root / "README.md",
        youtube=youtube,
        synthetic_pooled=synthetic_pooled,
        synthetic_differences=synthetic_differences,
        ablation_pooled=ablation_pooled,
    )

    expected_files = set(frames) | {
        "SOURCE_MANIFEST.csv",
        "VALIDATION.json",
        "REPORT.json",
        "README.md",
    }
    for stale in sorted(
        path
        for path in output_root.iterdir()
        if path.is_file() and path.name not in expected_files | {"SHA256SUMS"}
    ):
        raise ValueError(f"Unexpected pre-existing output artifact: {stale}")
    checksums = write_checksums(output_root)
    output_verification = verify_output_checksums(output_root)
    if set(checksums) != expected_files:
        raise ValueError("Consolidated checksum file set is incomplete")
    return {
        "status": "completed",
        "output_root": str(output_root),
        "table_row_counts": expected_counts,
        "source_artifact_count": int(len(source_manifest)),
        "output_verification": output_verification,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_package(output_root=args.output_root)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

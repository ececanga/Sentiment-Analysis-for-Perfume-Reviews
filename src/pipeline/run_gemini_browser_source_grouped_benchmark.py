# Date Created: 2026-07-13
# Date Last Modified: 2026-07-13

"""Run the leakage-safe Gemini-browser source-grouped benchmark.

This workflow deliberately writes to a new experiment root.  It prepares
connected source groups, verifies the frozen outer/inner manifests, uses the
Logistic Regression run as a cheap end-to-end smoke test, and only then allows
the remaining classical and deep models to run.

The source audit stages are hard gates.  ``--only`` and ``--from-stage`` do not
bypass them: every downstream stage revalidates its required upstream
artifacts before starting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils.paths import PROJECT_ROOT
from src.training.source_grouped_folds import validate_configured_frozen_inputs
from src.training.run_dummy_prevalence_baselines import (
    BASELINE_ORDER,
    FIXED_THRESHOLD,
    compute_dummy_prevalence_diagnostics,
)


WORKFLOW_ID = "gemini_browser_og784_source_grouped_v1"
DEFAULT_OUTPUT_ROOT = Path("outputs") / WORKFLOW_ID
DEFAULT_CONFIG_PATH = Path("config") / "evaluation_config_gemini_browser_og784_source_grouped_v1.yaml"
DEFAULT_RAW_CSV = Path("data/raw/YouTube_Perfume_Reviews_FullyFormatted.csv")
DEFAULT_LABELS_CSV = Path("labels/gemini_browser_og784/gemini_browser_labels_og784_v1.csv")
DEFAULT_FEATURE_LABEL_CSV = Path(
    "outputs/gemini_browser_og784/analysis_out/gemini_browser_og784_feature_labels.csv"
)

MODEL_IDS = (
    "logistic_regression",
    "random_forest",
    "xgboost",
    "bert",
    "bilstm_attention",
)
EXPECTED_SPLIT_TYPE = "source_component_groupkfold"
EXPECTED_GROUPING_VERSION = "source_identity_v1"
EXPECTED_GROUP_COLUMN = "source_group_id"
OFFICIAL_LABELS = (
    "joy",
    "sadness",
    "fear",
    "trust",
    "surprise",
    "anticipation",
    "disgust",
)
SOURCE_GROUP_COLUMN_CANDIDATES = (
    "source_group_id",
    "source_component_id",
    "component_id",
)
VIDEO_COLUMN_CANDIDATES = ("video_id", "Video ID", "normalized_video_id")
TRANSCRIPT_HASH_COLUMN_CANDIDATES = (
    "transcript_sha256",
    "transcript_hash",
    "normalized_transcript_hash",
    "final_transcript_sha256",
)
CODE_PROVENANCE_PATHS = (
    "src/training/source_grouped_folds.py",
    "src/training/model_benchmark_runner.py",
    "src/training/threshold_tuning.py",
    "src/training/build_deep_model_text_dataset.py",
    "src/training/run_logistic_regression_benchmark.py",
    "src/training/run_random_forest_benchmark.py",
    "src/training/run_xgboost_benchmark.py",
    "src/training/run_bert_benchmark.py",
    "src/training/run_bilstm_attention_benchmark.py",
    "src/training/run_dummy_prevalence_baselines.py",
    "src/training/generate_source_grouped_diagnostics.py",
    "src/training/generate_final_benchmark_figures.py",
    "src/pipeline/run_gemini_browser_source_grouped_benchmark.py",
)


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def resolve_project_path(value: str | Path, *, project_root: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def stringify_path(path: Path, *, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return str(path.resolve())


def quote_command(command: Sequence[str]) -> str:
    return shlex.join([str(part) for part in command])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head(project_root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return completed.stdout.strip() or "unknown"


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically write a small workflow JSON artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    temp_path.write_text(json.dumps(dict(payload), indent=2) + "\n", encoding="utf-8")
    temp_path.replace(path)


def read_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def require_paths(paths: Iterable[Path], *, context: str) -> None:
    missing = [path for path in paths if not path.is_file()]
    if missing:
        joined = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"{context} is missing required artifact(s): {joined}")


def require_nonempty_paths(paths: Iterable[Path], *, context: str) -> None:
    require_paths(paths, context=context)
    empty = [path for path in paths if path.stat().st_size == 0]
    if empty:
        joined = ", ".join(str(path) for path in empty)
        raise ValueError(f"{context} contains empty artifact(s): {joined}")


def load_yaml_config(path: Path) -> Dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return payload


def _nested_mapping(payload: Mapping[str, Any], key: str, *, context: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} requires mapping {key}")
    return value


def _assert_close(*, actual: Any, expected: float, context: str, atol: float = 1e-9) -> None:
    try:
        numeric = float(actual)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} is not numeric") from exc
    if not np.isfinite(numeric) or not np.isclose(numeric, expected, rtol=0.0, atol=atol):
        raise ValueError(f"{context} mismatch: saved={numeric!r}, recomputed={expected!r}")


def _find_column(df: pd.DataFrame, candidates: Sequence[str], *, context: str) -> str:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise ValueError(f"{context} requires one of these columns: {', '.join(candidates)}")


def _normalize_id_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


def _normalize_fold_series(series: pd.Series, *, context: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="raise")
    if (numeric % 1 != 0).any():
        raise ValueError(f"{context} contains non-integer fold identifiers")
    return numeric.astype(int)


@dataclass(frozen=True)
class WorkflowPaths:
    output_root: Path
    source_audit_dir: Path
    benchmark_root: Path
    tables_dir: Path
    metrics_dir: Path
    predictions_dir: Path
    logs_dir: Path
    datasets_dir: Path
    workflow_dir: Path
    source_memberships_csv: Path
    source_components_csv: Path
    source_group_audit_json: Path
    fold_assignments_csv: Path
    inner_validation_folds_csv: Path
    fold_balance_csv: Path
    deep_jsonl: Path
    deep_chunks_csv: Path
    deep_summary_json: Path
    command_manifest_json: Path
    run_status_json: Path

    def model_metrics(self, model_id: str) -> Path:
        return self.metrics_dir / f"{model_id}.json"

    def model_per_label(self, model_id: str) -> Path:
        return self.metrics_dir / f"{model_id}_per_label.csv"

    def model_predictions(self, model_id: str) -> Path:
        return self.predictions_dir / f"{model_id}_oof.csv"

    def model_run_manifest(self, model_id: str) -> Path:
        return self.logs_dir / f"{model_id}_run.json"

    def model_log(self, model_id: str) -> Path:
        return self.logs_dir / f"{model_id}.log"

    def model_per_label_figure(self, model_id: str) -> Path:
        return self.benchmark_root / "figures" / f"{model_id}_per_label_metrics.png"

    def model_curve_figure(self, model_id: str) -> Path:
        return self.benchmark_root / "figures" / f"{model_id}_roc_pr_curves.png"

    def model_note(self, model_id: str) -> Path:
        return self.tables_dir / f"{model_id}_note.md"

    @property
    def baseline_dir(self) -> Path:
        return self.benchmark_root / "dummy_prevalence_baselines"

    @property
    def baseline_outputs(self) -> Tuple[Path, ...]:
        return tuple(
            self.baseline_dir / name
            for name in (
                "dummy_prevalence_oof_predictions.csv",
                "dummy_prevalence_summary.csv",
                "dummy_prevalence_per_label.csv",
                "dummy_prevalence_fold_diagnostics.csv",
                "dummy_prevalence_report.json",
            )
        )

    @property
    def diagnostic_outputs(self) -> Tuple[Path, ...]:
        return tuple(
            self.tables_dir / name
            for name in (
                "inner_fold_balance.csv",
                "outer_fold_per_label_metrics.csv",
                "component_weighted_model_metrics.csv",
                "paired_component_bootstrap_macro_f1.csv",
                "source_grouped_diagnostics_report.json",
            )
        )

    @property
    def final_export_outputs(self) -> Tuple[Path, ...]:
        return (
            self.tables_dir / "final_model_comparison_for_thesis.csv",
            self.benchmark_root / "figures" / "all_models_macro_f1_comparison.png",
            self.benchmark_root / "figures" / "all_models_macro_roc_auc_comparison.png",
            self.benchmark_root / "figures" / "all_models_macro_pr_auc_comparison.png",
            self.benchmark_root / "figures" / "all_models_grouped_metric_comparison.png",
        )

    def stage_receipt(self, stage_name: str) -> Path:
        return self.workflow_dir / "stages" / f"{stage_name}.json"

    def stage_log(self, stage_name: str) -> Path:
        return self.workflow_dir / "logs" / f"{stage_name}.log"


def build_workflow_paths(output_root: Path) -> WorkflowPaths:
    output_root = output_root.resolve()
    source_audit_dir = output_root / "source_audit"
    benchmark_root = output_root / "model_benchmark"
    tables_dir = benchmark_root / "tables"
    workflow_dir = output_root / "workflow"
    datasets_dir = benchmark_root / "datasets"
    return WorkflowPaths(
        output_root=output_root,
        source_audit_dir=source_audit_dir,
        benchmark_root=benchmark_root,
        tables_dir=tables_dir,
        metrics_dir=benchmark_root / "metrics",
        predictions_dir=benchmark_root / "predictions",
        logs_dir=benchmark_root / "logs",
        datasets_dir=datasets_dir,
        workflow_dir=workflow_dir,
        source_memberships_csv=source_audit_dir / "source_memberships.csv",
        source_components_csv=source_audit_dir / "source_components.csv",
        source_group_audit_json=source_audit_dir / "source_group_audit.json",
        fold_assignments_csv=tables_dir / "fold_assignments.csv",
        inner_validation_folds_csv=tables_dir / "inner_validation_folds.csv",
        fold_balance_csv=tables_dir / "fold_balance.csv",
        deep_jsonl=datasets_dir / "perfume_text_dataset_v1.jsonl",
        deep_chunks_csv=datasets_dir / "perfume_text_dataset_v1_chunks.csv",
        deep_summary_json=datasets_dir / "perfume_text_dataset_v1_summary.json",
        command_manifest_json=workflow_dir / "command_manifest.json",
        run_status_json=workflow_dir / "run_status.json",
    )


@dataclass(frozen=True)
class StageSpec:
    name: str
    description: str
    commands: Tuple[Tuple[str, ...], ...] = ()
    expected_outputs: Tuple[Path, ...] = ()
    gate_description: str | None = None

    @property
    def is_gate(self) -> bool:
        return self.gate_description is not None


@dataclass(frozen=True)
class WorkflowPlan:
    project_root: Path
    python_executable: str
    config_path: Path
    raw_csv: Path
    labels_csv: Path
    feature_label_csv: Path
    paths: WorkflowPaths
    n_splits: int
    inner_n_splits: int
    seed: int
    device: str
    stages: Tuple[StageSpec, ...]


def _model_outputs(paths: WorkflowPaths, model_id: str) -> Tuple[Path, ...]:
    return (
        paths.model_metrics(model_id),
        paths.model_per_label(model_id),
        paths.model_predictions(model_id),
        paths.model_run_manifest(model_id),
        paths.model_per_label_figure(model_id),
        paths.model_curve_figure(model_id),
        paths.model_log(model_id),
        paths.model_note(model_id),
    )


def build_workflow_plan(
    *,
    project_root: Path,
    output_root: Path,
    config_path: Path,
    raw_csv: Path,
    labels_csv: Path,
    feature_label_csv: Path,
    python_executable: str,
    n_splits: int = 5,
    inner_n_splits: int = 10,
    seed: int = 42,
    device: str = "auto",
) -> WorkflowPlan:
    """Build the immutable, ordered command plan."""

    project_root = project_root.resolve()
    output_root = resolve_project_path(output_root, project_root=project_root)
    config_path = resolve_project_path(config_path, project_root=project_root)
    raw_csv = resolve_project_path(raw_csv, project_root=project_root)
    labels_csv = resolve_project_path(labels_csv, project_root=project_root)
    feature_label_csv = resolve_project_path(feature_label_csv, project_root=project_root)
    paths = build_workflow_paths(output_root)

    def p(path: Path) -> str:
        return stringify_path(path, project_root=project_root)

    python_path = Path(python_executable).expanduser()
    if python_path.is_absolute():
        python_token = str(python_path)
    else:
        python_token = str(python_path)

    preparation_outputs = (
        paths.source_memberships_csv,
        paths.source_components_csv,
        paths.source_group_audit_json,
        paths.fold_assignments_csv,
        paths.inner_validation_folds_csv,
        paths.fold_balance_csv,
    )
    classical_outputs = tuple(
        output
        for model_id in ("logistic_regression", "random_forest", "xgboost")
        for output in _model_outputs(paths, model_id)
    )
    deep_outputs = (paths.deep_jsonl, paths.deep_chunks_csv, paths.deep_summary_json)
    all_model_outputs = tuple(
        output for model_id in MODEL_IDS for output in _model_outputs(paths, model_id)
    )

    stages = (
        StageSpec(
            name="prepare",
            description="Build exact-source connected components and frozen grouped folds.",
            commands=(
                (
                    python_token,
                    "-m",
                    "src.training.source_grouped_folds",
                    "--config",
                    p(config_path),
                    "--raw-csv",
                    p(raw_csv),
                    "--labels-csv",
                    p(labels_csv),
                    "--feature-label-csv",
                    p(feature_label_csv),
                    "--out-root",
                    p(paths.output_root),
                    "--n-splits",
                    str(n_splits),
                    "--inner-n-splits",
                    str(inner_n_splits),
                    "--seed",
                    str(seed),
                ),
            ),
            expected_outputs=preparation_outputs,
        ),
        StageSpec(
            name="source_audit_gate",
            description="Fail unless exact videos, transcript hashes, and components are fold-safe.",
            expected_outputs=preparation_outputs,
            gate_description=(
                "Require status=passed, validation_passed=true, zero cross-fold video IDs, "
                "zero cross-fold transcript hashes, and directly validate outer/inner group integrity."
            ),
        ),
        StageSpec(
            name="dummy_prevalence_baselines",
            description="Generate trivial and training-fold-prevalence OOF reference rows.",
            commands=(
                (
                    python_token,
                    "-m",
                    "src.training.run_dummy_prevalence_baselines",
                    "--input-csv",
                    p(feature_label_csv),
                    "--fold-manifest-csv",
                    p(paths.fold_assignments_csv),
                    "--out-dir",
                    p(paths.baseline_dir),
                    "--overwrite",
                ),
            ),
            expected_outputs=paths.baseline_outputs,
        ),
        StageSpec(
            name="logistic_regression_smoke",
            description="Run the full Logistic Regression benchmark as the cheap pipeline smoke test.",
            commands=(
                (
                    python_token,
                    "-m",
                    "src.training.run_logistic_regression_benchmark",
                    "--config",
                    p(config_path),
                    "--input-csv",
                    p(feature_label_csv),
                    "--out-root",
                    p(paths.benchmark_root),
                ),
            ),
            expected_outputs=_model_outputs(paths, "logistic_regression"),
        ),
        StageSpec(
            name="random_forest",
            description="Run Random Forest only after the LR smoke artifacts pass validation.",
            commands=(
                (
                    python_token,
                    "-m",
                    "src.training.run_random_forest_benchmark",
                    "--config",
                    p(config_path),
                    "--input-csv",
                    p(feature_label_csv),
                    "--out-root",
                    p(paths.benchmark_root),
                ),
            ),
            expected_outputs=_model_outputs(paths, "random_forest"),
        ),
        StageSpec(
            name="xgboost",
            description="Run XGBoost after the LR-first smoke gate.",
            commands=(
                (
                    python_token,
                    "-m",
                    "src.training.run_xgboost_benchmark",
                    "--config",
                    p(config_path),
                    "--input-csv",
                    p(feature_label_csv),
                    "--out-root",
                    p(paths.benchmark_root),
                ),
            ),
            expected_outputs=_model_outputs(paths, "xgboost"),
        ),
        StageSpec(
            name="deep_dataset",
            description="Build the source-grouped deep-text JSONL and chunk audit table.",
            commands=(
                (
                    python_token,
                    "-m",
                    "src.training.build_deep_model_text_dataset",
                    "--raw-csv",
                    p(raw_csv),
                    "--labels-csv",
                    p(labels_csv),
                    "--folds-csv",
                    p(paths.fold_assignments_csv),
                    "--output-jsonl",
                    p(paths.deep_jsonl),
                    "--output-chunks-csv",
                    p(paths.deep_chunks_csv),
                    "--output-summary-json",
                    p(paths.deep_summary_json),
                ),
            ),
            expected_outputs=deep_outputs,
        ),
        StageSpec(
            name="deep_dataset_audit_gate",
            description="Verify deep rows, chunks, folds, groups, and frozen manifest hashes.",
            expected_outputs=deep_outputs,
            gate_description=(
                "Require one deep-text record per perfume, exact outer-fold agreement, "
                "source-group preservation, and zero exact-source cross-fold counts."
            ),
        ),
        StageSpec(
            name="bert",
            description="Run BERT on the audited source-grouped deep-text dataset.",
            commands=(
                (
                    python_token,
                    "-m",
                    "src.training.run_bert_benchmark",
                    "--config",
                    p(config_path),
                    "--input-jsonl",
                    p(paths.deep_jsonl),
                    "--out-root",
                    p(paths.benchmark_root),
                    "--device",
                    device,
                ),
            ),
            expected_outputs=_model_outputs(paths, "bert"),
        ),
        StageSpec(
            name="bilstm_attention",
            description="Run BiLSTM + Attention after the BERT stage.",
            commands=(
                (
                    python_token,
                    "-m",
                    "src.training.run_bilstm_attention_benchmark",
                    "--config",
                    p(config_path),
                    "--input-jsonl",
                    p(paths.deep_jsonl),
                    "--out-root",
                    p(paths.benchmark_root),
                    "--device",
                    device,
                ),
            ),
            expected_outputs=_model_outputs(paths, "bilstm_attention"),
        ),
        StageSpec(
            name="source_grouped_diagnostics",
            description="Generate rare-label, component-weighted, and cluster-bootstrap diagnostics.",
            commands=(
                (
                    python_token,
                    "-m",
                    "src.training.generate_source_grouped_diagnostics",
                    "--feature-label-csv",
                    p(feature_label_csv),
                    "--outer-manifest-csv",
                    p(paths.fold_assignments_csv),
                    "--inner-manifest-csv",
                    p(paths.inner_validation_folds_csv),
                    "--benchmark-root",
                    p(paths.benchmark_root),
                    "--out-dir",
                    p(paths.tables_dir),
                    "--threshold",
                    "0.3",
                    "--seed",
                    str(seed),
                    "--bootstrap-reps",
                    "2000",
                    "--overwrite",
                ),
            ),
            expected_outputs=paths.diagnostic_outputs,
        ),
        StageSpec(
            name="final_thesis_export",
            description="Export the validated five-model comparison using thesis-facing terminology.",
            commands=(
                (
                    python_token,
                    "-m",
                    "src.training.generate_final_benchmark_figures",
                    "--config",
                    p(config_path),
                    "--source-table",
                    p(paths.tables_dir / "comparison_table.csv"),
                    "--output-dir",
                    p(paths.benchmark_root),
                ),
            ),
            expected_outputs=paths.final_export_outputs,
        ),
        StageSpec(
            name="final_audit_gate",
            description="Verify all five models used the same frozen source-grouped manifests.",
            expected_outputs=(
                all_model_outputs
                + paths.baseline_outputs
                + paths.diagnostic_outputs
                + paths.final_export_outputs
                + (paths.tables_dir / "comparison_table.csv",)
            ),
            gate_description=(
                "Require complete artifacts and OOF fold agreement for all five models, "
                "matching outer/inner/audit hashes, and complete comparison-table rows."
            ),
        ),
    )

    # Keep these expressions referenced in the plan construction: they document
    # the dependency bundles and catch accidental path drift during development.
    assert classical_outputs
    assert deep_outputs

    return WorkflowPlan(
        project_root=project_root,
        python_executable=python_token,
        config_path=config_path,
        raw_csv=raw_csv,
        labels_csv=labels_csv,
        feature_label_csv=feature_label_csv,
        paths=paths,
        n_splits=n_splits,
        inner_n_splits=inner_n_splits,
        seed=seed,
        device=device,
        stages=stages,
    )


def canonical_stage_names(plan: WorkflowPlan) -> List[str]:
    return [stage.name for stage in plan.stages]


def _normalize_stage_args(values: Sequence[str] | None) -> List[str]:
    normalized: List[str] = []
    for value in values or []:
        for part in str(value).split(","):
            name = part.strip()
            if name and name not in normalized:
                normalized.append(name)
    return normalized


def select_stages(
    *,
    plan: WorkflowPlan,
    only: Sequence[str] | None = None,
    from_stage: str | None = None,
    to_stage: str | None = None,
) -> List[StageSpec]:
    names = canonical_stage_names(plan)
    requested = _normalize_stage_args(only)
    unknown = [name for name in [*requested, from_stage, to_stage] if name and name not in names]
    if unknown:
        raise ValueError(f"Unknown workflow stage(s): {', '.join(unknown)}")
    if requested and (from_stage or to_stage):
        raise ValueError("--only cannot be combined with --from-stage/--to-stage")

    if requested:
        requested_set = set(requested)
        return [stage for stage in plan.stages if stage.name in requested_set]

    start = names.index(from_stage) if from_stage else 0
    end = names.index(to_stage) if to_stage else len(names) - 1
    if start > end:
        raise ValueError("--from-stage must appear before --to-stage")
    return list(plan.stages[start : end + 1])


def validate_output_root(*, output_root: Path, project_root: Path) -> None:
    """Reject legacy or non-output destinations before creating anything."""

    project_root = project_root.resolve()
    output_root = output_root.resolve()
    outputs_root = (project_root / "outputs").resolve()
    if output_root == outputs_root or outputs_root not in output_root.parents:
        raise ValueError(f"Output root must be a new child of {outputs_root}: {output_root}")

    protected_roots = (
        outputs_root / "gemini_browser_og784",
        outputs_root / "model_benchmark",
        outputs_root / "gemini_judge",
    )
    for protected in protected_roots:
        protected = protected.resolve()
        if output_root == protected or protected in output_root.parents:
            raise ValueError(f"Refusing to write inside protected legacy output root: {protected}")


def ensure_fresh_or_resumable_root(
    *,
    paths: WorkflowPaths,
    resume: bool,
    dry_run: bool,
) -> None:
    if resume or dry_run or not paths.output_root.exists():
        return
    # A dry run intentionally persists the resolved command manifest and its
    # status.  Those planning records must not make the otherwise-new result
    # root ineligible for the first real execution.
    ignored = {
        paths.command_manifest_json.resolve(),
        paths.run_status_json.resolve(),
    }
    payload_files = [
        path
        for path in paths.output_root.rglob("*")
        if path.is_file() and path.resolve() not in ignored and path.name != ".DS_Store"
    ]
    if payload_files:
        raise FileExistsError(
            f"Output root is not fresh: {paths.output_root}. Use --resume to reuse validated stages."
        )


def current_code_provenance(
    *,
    project_root: Path,
    config_path: Path,
) -> Dict[str, Any]:
    return {
        "base_git_commit": git_head(project_root),
        "file_sha256": {
            relative_path: sha256_file(project_root / relative_path)
            for relative_path in CODE_PROVENANCE_PATHS
            if (project_root / relative_path).is_file()
        },
        "config_sha256": sha256_file(config_path),
    }


def _stage_fingerprint(
    stage: StageSpec,
    *,
    project_root: Path,
    config_path: Path,
) -> str:
    payload = {
        "name": stage.name,
        "description": stage.description,
        "commands": [list(command) for command in stage.commands],
        "expected_outputs": [stringify_path(path, project_root=project_root) for path in stage.expected_outputs],
        "gate_description": stage.gate_description,
        "code_provenance": current_code_provenance(
            project_root=project_root,
            config_path=config_path,
        ),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_command_manifest(
    *,
    plan: WorkflowPlan,
    selected_stages: Sequence[StageSpec],
    dry_run: bool,
    resume: bool,
) -> Dict[str, Any]:
    selected_names = {stage.name for stage in selected_stages}
    stage_rows: List[Dict[str, Any]] = []
    resolved_commands: List[Dict[str, Any]] = []
    for ordinal, stage in enumerate(plan.stages, start=1):
        commands = [list(command) for command in stage.commands]
        stage_rows.append(
            {
                "ordinal": ordinal,
                "name": stage.name,
                "selected": stage.name in selected_names,
                "kind": "hard_gate" if stage.is_gate else "command",
                "description": stage.description,
                "commands": commands,
                "command_texts": [quote_command(command) for command in stage.commands],
                "expected_outputs": [
                    stringify_path(path, project_root=plan.project_root)
                    for path in stage.expected_outputs
                ],
                "gate_description": stage.gate_description,
                "fingerprint": _stage_fingerprint(
                    stage,
                    project_root=plan.project_root,
                    config_path=plan.config_path,
                ),
            }
        )
        if stage.name in selected_names:
            for command in commands:
                resolved_commands.append(
                    {
                        "stage": stage.name,
                        "argv": command,
                        "command": quote_command(command),
                    }
                )

    return {
        "schema_version": 1,
        "workflow_id": WORKFLOW_ID,
        "generated_at": now_utc(),
        "project_root": str(plan.project_root),
        "output_root": stringify_path(plan.paths.output_root, project_root=plan.project_root),
        "config_path": stringify_path(plan.config_path, project_root=plan.project_root),
        "dry_run": dry_run,
        "resume": resume,
        "n_splits": plan.n_splits,
        "inner_n_splits": plan.inner_n_splits,
        "seed": plan.seed,
        "device": plan.device,
        "code_provenance": current_code_provenance(
            project_root=plan.project_root,
            config_path=plan.config_path,
        ),
        "stage_order": canonical_stage_names(plan),
        "selected_stages": [stage.name for stage in selected_stages],
        "resolved_commands": resolved_commands,
        "existing_stage_receipts": {
            stage.name: read_json(plan.paths.stage_receipt(stage.name))
            for stage in plan.stages
            if plan.paths.stage_receipt(stage.name).is_file()
        },
        "stages": stage_rows,
        "safety": {
            "new_output_root_only": True,
            "fail_fast": True,
            "source_audit_cannot_be_bypassed": True,
            "lr_smoke_required_before_rf_xgb_and_deep": True,
        },
    }


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False)


def _validated_outer_manifest(
    *,
    paths: WorkflowPaths,
    labels_csv: Path,
    n_splits: int,
) -> Tuple[pd.DataFrame, str]:
    labels_df = _read_csv(labels_csv)
    if "perfume_id" not in labels_df.columns:
        raise ValueError(f"Labels CSV lacks perfume_id: {labels_csv}")
    label_ids = _normalize_id_series(labels_df["perfume_id"])
    if (label_ids == "").any() or label_ids.duplicated().any():
        raise ValueError("Labels CSV must contain unique, non-empty perfume_id values")

    outer_df = _read_csv(paths.fold_assignments_csv)
    required = {"perfume_id", "fold_id"}
    if not required.issubset(outer_df.columns):
        raise ValueError(f"Outer fold manifest lacks columns: {sorted(required - set(outer_df.columns))}")
    outer_df = outer_df.copy()
    outer_df["perfume_id"] = _normalize_id_series(outer_df["perfume_id"])
    outer_df["fold_id"] = _normalize_fold_series(
        outer_df["fold_id"], context="Outer fold manifest"
    )
    if (outer_df["perfume_id"] == "").any() or outer_df["perfume_id"].duplicated().any():
        raise ValueError("Outer fold manifest must contain unique, non-empty perfume_id values")
    expected_folds = set(range(1, n_splits + 1))
    if set(outer_df["fold_id"].unique()) != expected_folds:
        raise ValueError(f"Outer fold manifest must contain folds {sorted(expected_folds)}")
    if set(outer_df["perfume_id"]) != set(label_ids):
        raise ValueError("Outer fold perfume_id universe does not exactly match the frozen labels")

    group_col = _find_column(
        outer_df,
        SOURCE_GROUP_COLUMN_CANDIDATES,
        context="Outer fold manifest",
    )
    outer_df[group_col] = _normalize_id_series(outer_df[group_col])
    if (outer_df[group_col] == "").any():
        raise ValueError("Outer fold manifest contains empty source-group IDs")
    if int(outer_df.groupby(group_col)["fold_id"].nunique().max()) != 1:
        raise ValueError("At least one source component crosses outer folds")
    return outer_df, group_col


def _validate_source_memberships(
    *,
    paths: WorkflowPaths,
    outer_df: pd.DataFrame,
    outer_group_col: str,
) -> Dict[str, int]:
    memberships = _read_csv(paths.source_memberships_csv)
    if "perfume_id" not in memberships.columns:
        raise ValueError("source_memberships.csv lacks perfume_id")
    memberships = memberships.copy()
    memberships["perfume_id"] = _normalize_id_series(memberships["perfume_id"])
    if (memberships["perfume_id"] == "").any():
        raise ValueError("source_memberships.csv contains empty perfume_id values")
    if set(memberships["perfume_id"]) != set(outer_df["perfume_id"]):
        raise ValueError("Source membership perfume universe does not match outer folds")

    membership_group_col = _find_column(
        memberships,
        SOURCE_GROUP_COLUMN_CANDIDATES,
        context="source_memberships.csv",
    )
    memberships[membership_group_col] = _normalize_id_series(memberships[membership_group_col])
    if (memberships[membership_group_col] == "").any():
        raise ValueError("source_memberships.csv contains empty source-group IDs")

    fold_map = outer_df.set_index("perfume_id")["fold_id"]
    group_map = outer_df.set_index("perfume_id")[outer_group_col]
    memberships["_outer_fold_id"] = memberships["perfume_id"].map(fold_map)
    memberships["_outer_group_id"] = memberships["perfume_id"].map(group_map)
    if memberships["_outer_fold_id"].isna().any():
        raise ValueError("Source membership contains an unknown perfume_id")
    if not (
        memberships[membership_group_col].astype(str)
        == memberships["_outer_group_id"].astype(str)
    ).all():
        raise ValueError("Source membership group IDs disagree with the outer manifest")

    if "fold_id" in memberships.columns:
        membership_folds = _normalize_fold_series(
            memberships["fold_id"], context="source_memberships.csv"
        )
        if not (membership_folds.to_numpy() == memberships["_outer_fold_id"].to_numpy()).all():
            raise ValueError("Source membership fold IDs disagree with the outer manifest")

    if int(memberships.groupby(membership_group_col)["_outer_fold_id"].nunique().max()) != 1:
        raise ValueError("A source membership component crosses outer folds")

    source_checks: Dict[str, int] = {}
    has_wide_sources = False
    for label, candidates in (
        ("video", VIDEO_COLUMN_CANDIDATES),
        ("transcript", TRANSCRIPT_HASH_COLUMN_CANDIDATES),
    ):
        source_col = next((column for column in candidates if column in memberships.columns), None)
        if source_col is None:
            continue
        has_wide_sources = True
        values = _normalize_id_series(memberships[source_col])
        populated = memberships.loc[values != ""].copy()
        populated["_source_value"] = values.loc[values != ""]
        leaking = populated.groupby("_source_value")["_outer_fold_id"].nunique()
        leak_count = int((leaking > 1).sum())
        if leak_count:
            raise ValueError(f"{leak_count} exact {label} source(s) cross outer folds")
        source_checks[f"checked_{label}_sources"] = int(populated["_source_value"].nunique())

    if not has_wide_sources and {"source_type", "source_value"}.issubset(memberships.columns):
        long_df = memberships.copy()
        long_df["_source_type"] = _normalize_id_series(long_df["source_type"]).str.lower()
        long_df["_source_value"] = _normalize_id_series(long_df["source_value"])
        long_df = long_df.loc[long_df["_source_value"] != ""]
        leaking = long_df.groupby(["_source_type", "_source_value"])["_outer_fold_id"].nunique()
        leak_count = int((leaking > 1).sum())
        if leak_count:
            raise ValueError(f"{leak_count} exact source membership(s) cross outer folds")
        source_checks["checked_long_form_sources"] = int(len(leaking))
        has_wide_sources = True

    if not has_wide_sources:
        raise ValueError(
            "source_memberships.csv must expose video/transcript columns or source_type/source_value"
        )
    return source_checks


def _validate_source_components(
    *,
    paths: WorkflowPaths,
    outer_df: pd.DataFrame,
    outer_group_col: str,
) -> int:
    components = _read_csv(paths.source_components_csv)
    component_group_col = _find_column(
        components,
        SOURCE_GROUP_COLUMN_CANDIDATES,
        context="source_components.csv",
    )
    components = components.copy()
    components[component_group_col] = _normalize_id_series(components[component_group_col])
    if (components[component_group_col] == "").any() or components[component_group_col].duplicated().any():
        raise ValueError("source_components.csv must have one non-empty row per source group")
    if set(components[component_group_col]) != set(outer_df[outer_group_col]):
        raise ValueError("Source component universe disagrees with the outer manifest")
    if "fold_id" in components.columns:
        components["fold_id"] = _normalize_fold_series(
            components["fold_id"], context="source_components.csv"
        )
        expected = outer_df.groupby(outer_group_col)["fold_id"].first()
        observed = components.set_index(component_group_col)["fold_id"]
        if any(int(observed[group]) != int(expected[group]) for group in expected.index):
            raise ValueError("Source component fold IDs disagree with the outer manifest")
    return int(len(components))


def _validate_inner_manifest(
    *,
    paths: WorkflowPaths,
    outer_df: pd.DataFrame,
    outer_group_col: str,
    n_splits: int,
) -> Dict[str, int]:
    inner = _read_csv(paths.inner_validation_folds_csv)
    required = {"outer_fold_id", "perfume_id", "inner_role"}
    if not required.issubset(inner.columns):
        raise ValueError(f"Inner manifest lacks columns: {sorted(required - set(inner.columns))}")
    inner = inner.copy()
    inner["outer_fold_id"] = _normalize_fold_series(
        inner["outer_fold_id"], context="Inner validation manifest"
    )
    inner["perfume_id"] = _normalize_id_series(inner["perfume_id"])
    inner["inner_role"] = _normalize_id_series(inner["inner_role"]).str.lower()
    if inner.duplicated(["outer_fold_id", "perfume_id"]).any():
        raise ValueError("Inner manifest duplicates an outer_fold_id/perfume_id pair")
    if set(inner["outer_fold_id"].unique()) != set(range(1, n_splits + 1)):
        raise ValueError("Inner manifest does not cover every outer fold")
    if not set(inner["inner_role"]).issubset({"train", "val"}):
        raise ValueError("Inner manifest roles must be train or val")

    outer_fold_map = outer_df.set_index("perfume_id")["fold_id"]
    group_map = outer_df.set_index("perfume_id")[outer_group_col]
    inner["_held_out_fold"] = inner["perfume_id"].map(outer_fold_map)
    inner["_source_group_id"] = inner["perfume_id"].map(group_map)
    if inner[["_held_out_fold", "_source_group_id"]].isna().any().any():
        raise ValueError("Inner manifest contains an unknown perfume_id")
    if (inner["outer_fold_id"] == inner["_held_out_fold"]).any():
        raise ValueError("An outer-test perfume appears in its inner training/validation manifest")

    for outer_fold_id in range(1, n_splits + 1):
        observed = inner.loc[inner["outer_fold_id"] == outer_fold_id]
        expected_ids = set(outer_df.loc[outer_df["fold_id"] != outer_fold_id, "perfume_id"])
        if set(observed["perfume_id"]) != expected_ids:
            raise ValueError(f"Inner manifest perfume universe is incomplete for outer fold {outer_fold_id}")
        if set(observed["inner_role"]) != {"train", "val"}:
            raise ValueError(f"Outer fold {outer_fold_id} must contain both inner train and val roles")
        if int(observed.groupby("_source_group_id")["inner_role"].nunique().max()) != 1:
            raise ValueError(f"A source group is split between inner train/val for outer fold {outer_fold_id}")

    return {
        "inner_rows": int(len(inner)),
        "inner_validation_rows": int((inner["inner_role"] == "val").sum()),
    }


def validate_source_group_audit(
    *,
    paths: WorkflowPaths,
    labels_csv: Path,
    n_splits: int,
    raw_csv: Path | None = None,
    feature_label_csv: Path | None = None,
    config_path: Path | None = None,
) -> Dict[str, Any]:
    """Run the non-bypassable source and fold integrity gate."""

    required_paths = (
        paths.source_memberships_csv,
        paths.source_components_csv,
        paths.source_group_audit_json,
        paths.fold_assignments_csv,
        paths.inner_validation_folds_csv,
        paths.fold_balance_csv,
        labels_csv,
    )
    require_paths(required_paths, context="Source audit gate")
    report = read_json(paths.source_group_audit_json)
    if report.get("status") != "passed":
        raise ValueError("Source audit gate requires source_group_audit.json status='passed'")
    if report.get("validation_passed") is not True:
        raise ValueError("Source audit gate requires validation_passed=true")
    if report.get("grouping_version") != EXPECTED_GROUPING_VERSION:
        raise ValueError("Source audit gate requires source_identity_v1 grouping")
    if int(report.get("n_splits", -1)) != n_splits:
        raise ValueError("Source audit n_splits disagrees with the workflow")
    if raw_csv is not None and feature_label_csv is not None and config_path is not None:
        expected_inputs = {
            "raw_csv": sha256_file(raw_csv),
            "labels_csv": sha256_file(labels_csv),
            "feature_label_csv": sha256_file(feature_label_csv),
        }
        recorded_inputs = report.get("input_sha256")
        if not isinstance(recorded_inputs, Mapping):
            raise ValueError("Source audit lacks input_sha256 provenance")
        for key, expected_hash in expected_inputs.items():
            if recorded_inputs.get(key) != expected_hash:
                raise ValueError(f"Source audit has mismatched input hash {key}")
        frozen_validation = report.get("frozen_input_validation")
        if not isinstance(frozen_validation, Mapping) or frozen_validation.get("enforced") is not True:
            raise ValueError("Source audit did not enforce configured frozen input hashes")
        if frozen_validation.get("config_sha256") != sha256_file(config_path):
            raise ValueError("Source audit was generated with a different config")
    for field in ("cross_fold_video_id_count", "cross_fold_transcript_hash_count"):
        try:
            value = int(report.get(field))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Source audit gate requires integer field {field}") from exc
        if value != 0:
            raise ValueError(f"Source audit gate requires {field}=0, found {value}")
    manifest_hashes = report.get("manifest_sha256")
    if not isinstance(manifest_hashes, Mapping):
        raise ValueError("Source audit lacks manifest_sha256 provenance")
    expected_manifest_hashes = {
        "outer": sha256_file(paths.fold_assignments_csv),
        "inner": sha256_file(paths.inner_validation_folds_csv),
        "fold_balance": sha256_file(paths.fold_balance_csv),
    }
    for key, expected_hash in expected_manifest_hashes.items():
        if manifest_hashes.get(key) != expected_hash:
            raise ValueError(f"Source audit has mismatched manifest hash {key}")

    outer_df, group_col = _validated_outer_manifest(
        paths=paths,
        labels_csv=labels_csv,
        n_splits=n_splits,
    )
    group_sizes = outer_df.groupby(group_col)["perfume_id"].nunique()
    largest_report = report.get("largest_component")
    if not isinstance(largest_report, Mapping):
        raise ValueError("Source audit lacks largest_component disclosure")
    if int(largest_report.get("source_group_size", -1)) != int(group_sizes.max()):
        raise ValueError("Source audit largest_component size disagrees with manifests")
    membership_checks = _validate_source_memberships(
        paths=paths,
        outer_df=outer_df,
        outer_group_col=group_col,
    )
    component_count = _validate_source_components(
        paths=paths,
        outer_df=outer_df,
        outer_group_col=group_col,
    )
    inner_checks = _validate_inner_manifest(
        paths=paths,
        outer_df=outer_df,
        outer_group_col=group_col,
        n_splits=n_splits,
    )

    balance = _read_csv(paths.fold_balance_csv)
    required_balance_columns = {
        "fold_id",
        "test_perfume_count",
        "test_source_group_count",
        "train_perfume_count",
        "train_source_group_count",
        *{f"test_{label}_positive" for label in OFFICIAL_LABELS},
        *{f"train_{label}_positive" for label in OFFICIAL_LABELS},
    }
    missing_balance = required_balance_columns - set(balance.columns)
    if missing_balance:
        raise ValueError(
            f"fold_balance.csv lacks required columns: {sorted(missing_balance)}"
        )
    balance = balance.copy()
    balance["fold_id"] = _normalize_fold_series(
        balance["fold_id"], context="fold_balance.csv"
    )
    if balance["fold_id"].duplicated().any() or set(balance["fold_id"]) != set(
        range(1, n_splits + 1)
    ):
        raise ValueError("fold_balance.csv must contain exactly one row per outer fold")

    labels = _read_csv(labels_csv).copy()
    if not {"perfume_id", *OFFICIAL_LABELS}.issubset(labels.columns):
        raise ValueError("Frozen labels CSV lacks official labels required for fold balance")
    labels["perfume_id"] = _normalize_id_series(labels["perfume_id"])
    merged = outer_df.merge(
        labels[["perfume_id", *OFFICIAL_LABELS]],
        on="perfume_id",
        how="left",
        validate="one_to_one",
    )
    for fold_id in range(1, n_splits + 1):
        saved = balance.loc[balance["fold_id"] == fold_id].iloc[0]
        test = merged.loc[merged["fold_id"] == fold_id]
        train = merged.loc[merged["fold_id"] != fold_id]
        expected_counts = {
            "test_perfume_count": len(test),
            "test_source_group_count": test[group_col].nunique(),
            "train_perfume_count": len(train),
            "train_source_group_count": train[group_col].nunique(),
        }
        for label in OFFICIAL_LABELS:
            expected_counts[f"test_{label}_positive"] = int(
                pd.to_numeric(test[label], errors="raise").sum()
            )
            expected_counts[f"train_{label}_positive"] = int(
                pd.to_numeric(train[label], errors="raise").sum()
            )
        for column, expected_value in expected_counts.items():
            actual_value = int(pd.to_numeric(saved[column], errors="raise"))
            if actual_value != int(expected_value):
                raise ValueError(
                    f"fold_balance.csv {column} mismatch for fold {fold_id}: "
                    f"saved={actual_value} expected={expected_value}"
                )

    return {
        "status": "passed",
        "validation_passed": True,
        "n_perfumes": int(len(outer_df)),
        "n_source_components": component_count,
        "n_splits": n_splits,
        "cross_fold_video_id_count": 0,
        "cross_fold_transcript_hash_count": 0,
        "outer_manifest_sha256": sha256_file(paths.fold_assignments_csv),
        "inner_manifest_sha256": sha256_file(paths.inner_validation_folds_csv),
        "source_audit_sha256": sha256_file(paths.source_group_audit_json),
        **membership_checks,
        **inner_checks,
    }


def validate_deep_dataset(
    *,
    plan: WorkflowPlan,
    source_gate: Mapping[str, Any],
) -> Dict[str, Any]:
    paths = plan.paths
    require_paths(
        (paths.deep_jsonl, paths.deep_chunks_csv, paths.deep_summary_json),
        context="Deep dataset gate",
    )
    outer_df, group_col = _validated_outer_manifest(
        paths=paths,
        labels_csv=plan.labels_csv,
        n_splits=plan.n_splits,
    )
    outer_by_id = outer_df.set_index("perfume_id")

    rows: List[Dict[str, Any]] = []
    with paths.deep_jsonl.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Deep JSONL line {line_number} is not an object")
            rows.append(payload)
    if not rows:
        raise ValueError("Deep JSONL is empty")
    deep_df = pd.DataFrame(rows)
    required = {"perfume_id", "fold_id"}
    if not required.issubset(deep_df.columns):
        raise ValueError(f"Deep JSONL lacks fields: {sorted(required - set(deep_df.columns))}")
    deep_df["perfume_id"] = _normalize_id_series(deep_df["perfume_id"])
    deep_df["fold_id"] = _normalize_fold_series(deep_df["fold_id"], context="Deep JSONL")
    if deep_df["perfume_id"].duplicated().any():
        raise ValueError("Deep JSONL must have one record per perfume_id")
    if set(deep_df["perfume_id"]) != set(outer_df["perfume_id"]):
        raise ValueError("Deep JSONL perfume universe disagrees with outer folds")
    expected_folds = deep_df["perfume_id"].map(outer_by_id["fold_id"])
    if not (deep_df["fold_id"].to_numpy() == expected_folds.to_numpy()).all():
        raise ValueError("Deep JSONL fold IDs disagree with outer folds")

    deep_group_col = next(
        (column for column in SOURCE_GROUP_COLUMN_CANDIDATES if column in deep_df.columns),
        None,
    )
    if deep_group_col is None:
        raise ValueError("Deep JSONL lacks source-group IDs")
    expected_groups = deep_df["perfume_id"].map(outer_by_id[group_col]).astype(str)
    if not (_normalize_id_series(deep_df[deep_group_col]) == expected_groups).all():
        raise ValueError("Deep JSONL source-group IDs disagree with outer folds")

    canonical_labels = _read_csv(plan.labels_csv).copy()
    required_canonical = {
        "perfume_id",
        *OFFICIAL_LABELS,
        "anger",
        "valence",
        "arousal",
        "dominance",
    }
    missing_canonical = required_canonical - set(canonical_labels.columns)
    if missing_canonical:
        raise ValueError(
            f"Frozen label CSV lacks deep-target fields: {sorted(missing_canonical)}"
        )
    canonical_labels["perfume_id"] = _normalize_id_series(
        canonical_labels["perfume_id"]
    )
    canonical_by_id = canonical_labels.set_index("perfume_id")
    excluded_labels = ("anger",)
    vad_dimensions = ("valence", "arousal", "dominance")
    expected_chunk_counts: Dict[str, int] = {}
    for payload in rows:
        perfume_id = str(payload["perfume_id"]).strip()
        label_payload = payload.get("labels")
        excluded_payload = payload.get("excluded_labels")
        vad_payload = payload.get("vad")
        review_chunks = payload.get("review_chunks")
        if not isinstance(label_payload, Mapping):
            raise ValueError(f"Deep JSONL lacks labels mapping for {perfume_id}")
        if not isinstance(excluded_payload, Mapping):
            raise ValueError(f"Deep JSONL lacks excluded_labels mapping for {perfume_id}")
        if not isinstance(vad_payload, Mapping):
            raise ValueError(f"Deep JSONL lacks VAD mapping for {perfume_id}")
        if not isinstance(review_chunks, list) or not review_chunks:
            raise ValueError(f"Deep JSONL lacks review chunks for {perfume_id}")
        expected_chunk_counts[perfume_id] = len(review_chunks)
        for label in OFFICIAL_LABELS:
            actual = int(pd.to_numeric(label_payload.get(label), errors="raise"))
            expected_label = int(pd.to_numeric(canonical_by_id.loc[perfume_id, label]))
            if actual not in (0, 1) or actual != expected_label:
                raise ValueError(
                    f"Deep JSONL label {label} disagrees with frozen labels for {perfume_id}"
                )
        for label in excluded_labels:
            actual = int(pd.to_numeric(excluded_payload.get(label), errors="raise"))
            expected_label = int(pd.to_numeric(canonical_by_id.loc[perfume_id, label]))
            if actual not in (0, 1) or actual != expected_label:
                raise ValueError(
                    f"Deep JSONL excluded label {label} disagrees with frozen labels for {perfume_id}"
                )
        for dimension in vad_dimensions:
            actual_vad = float(pd.to_numeric(vad_payload.get(dimension), errors="raise"))
            expected_vad = float(pd.to_numeric(canonical_by_id.loc[perfume_id, dimension]))
            if not np.isfinite(actual_vad) or not np.isclose(
                actual_vad, expected_vad, rtol=0.0, atol=1e-12
            ):
                raise ValueError(
                    f"Deep JSONL VAD {dimension} disagrees with frozen labels for {perfume_id}"
                )
        for chunk in review_chunks:
            if not isinstance(chunk, Mapping):
                raise ValueError(f"Deep JSONL has a non-object chunk for {perfume_id}")
            chunk_text = str(chunk.get("chunk_text", "")).strip()
            if not chunk_text:
                raise ValueError(f"Deep JSONL has empty chunk text for {perfume_id}")
            if "TITLE:" in chunk_text or "DESCRIPTION:" in chunk_text:
                raise ValueError(
                    f"Deep JSONL includes forbidden title/description model text for {perfume_id}"
                )

    summary = read_json(paths.deep_summary_json)
    if summary.get("source_grouped") is not True:
        raise ValueError("Deep dataset summary must declare source_grouped=true")
    if int(summary.get("cross_fold_video_id_count", -1)) != 0:
        raise ValueError("Deep dataset summary reports cross-fold video leakage")
    if int(summary.get("cross_fold_transcript_hash_count", -1)) != 0:
        raise ValueError("Deep dataset summary reports cross-fold transcript leakage")
    if int(summary.get("n_perfumes", -1)) != len(outer_df):
        raise ValueError("Deep dataset summary n_perfumes disagrees with the outer manifest")
    if summary.get("fold_manifest_sha256") != source_gate["outer_manifest_sha256"]:
        raise ValueError("Deep dataset summary was built from a different outer manifest")
    if summary.get("raw_csv_sha256") != sha256_file(plan.raw_csv):
        raise ValueError("Deep dataset summary has the wrong frozen raw CSV hash")
    if summary.get("labels_csv_sha256") != sha256_file(plan.labels_csv):
        raise ValueError("Deep dataset summary has the wrong frozen label CSV hash")
    if summary.get("model_input_fields") != ["Final Transcript"]:
        raise ValueError("Deep dataset must declare Final Transcript as its only model input")
    if not str(summary.get("model_input_policy", "")).startswith("transcript_excerpt_only"):
        raise ValueError("Deep dataset summary lacks the transcript-only model-input policy")

    chunks = _read_csv(paths.deep_chunks_csv)
    if not {"perfume_id", "fold_id"}.issubset(chunks.columns):
        raise ValueError("Deep chunk audit table lacks perfume_id/fold_id")
    chunks["perfume_id"] = _normalize_id_series(chunks["perfume_id"])
    chunks["fold_id"] = _normalize_fold_series(chunks["fold_id"], context="Deep chunk table")
    if set(chunks["perfume_id"]) != set(outer_df["perfume_id"]):
        raise ValueError("Deep chunk perfume universe disagrees with outer folds")
    expected_chunk_folds = chunks["perfume_id"].map(outer_by_id["fold_id"])
    if not (chunks["fold_id"].to_numpy() == expected_chunk_folds.to_numpy()).all():
        raise ValueError("Deep chunk fold IDs disagree with outer folds")
    chunk_counts = chunks.groupby("perfume_id").size().to_dict()
    if chunk_counts != expected_chunk_counts:
        raise ValueError("Deep chunk table counts disagree with JSONL review_chunks")
    for label in (*OFFICIAL_LABELS, *excluded_labels):
        if label not in chunks.columns:
            raise ValueError(f"Deep chunk audit table lacks label column {label}")
        actual_values = pd.to_numeric(chunks[label], errors="raise").astype(int)
        expected_values = chunks["perfume_id"].map(canonical_by_id[label]).astype(int)
        if not set(actual_values.unique()).issubset({0, 1}) or not (
            actual_values.to_numpy() == expected_values.to_numpy()
        ).all():
            raise ValueError(f"Deep chunk label {label} disagrees with frozen labels")
    for dimension in vad_dimensions:
        if dimension not in chunks.columns:
            raise ValueError(f"Deep chunk audit table lacks VAD column {dimension}")
        actual_values = pd.to_numeric(chunks[dimension], errors="raise").to_numpy(float)
        expected_values = pd.to_numeric(
            chunks["perfume_id"].map(canonical_by_id[dimension]), errors="raise"
        ).to_numpy(float)
        if not np.isfinite(actual_values).all() or not np.allclose(
            actual_values, expected_values, rtol=0.0, atol=1e-12
        ):
            raise ValueError(f"Deep chunk VAD {dimension} disagrees with frozen labels")
    if "chunk_text" not in chunks.columns:
        raise ValueError("Deep chunk audit table lacks chunk_text")
    if chunks["chunk_text"].astype(str).str.contains(
        r"(?:^|\s)(?:TITLE|DESCRIPTION):", regex=True
    ).any():
        raise ValueError("Deep chunk audit table includes forbidden title/description text")
    for candidates, label in (
        (VIDEO_COLUMN_CANDIDATES, "video"),
        (TRANSCRIPT_HASH_COLUMN_CANDIDATES, "transcript"),
    ):
        column = next((candidate for candidate in candidates if candidate in chunks.columns), None)
        if column is None:
            continue
        values = _normalize_id_series(chunks[column])
        populated = chunks.loc[values != ""].copy()
        populated["_source"] = values.loc[values != ""]
        if int((populated.groupby("_source")["fold_id"].nunique() > 1).sum()):
            raise ValueError(f"Deep chunk table contains cross-fold {label} leakage")

    return {
        "status": "passed",
        "n_perfumes": int(len(deep_df)),
        "n_chunks": int(len(chunks)),
        "deep_jsonl_sha256": sha256_file(paths.deep_jsonl),
        "outer_manifest_sha256": source_gate["outer_manifest_sha256"],
    }


def validate_baseline_outputs(
    *,
    plan: WorkflowPlan,
    source_gate: Mapping[str, Any],
) -> Dict[str, Any]:
    require_nonempty_paths(plan.paths.baseline_outputs, context="Baseline diagnostics")
    report_path = plan.paths.baseline_dir / "dummy_prevalence_report.json"
    report = read_json(report_path)
    if report.get("diagnostic_id") != "dummy_prevalence_baselines_source_grouped_v1":
        raise ValueError("Baseline report has the wrong diagnostic_id")
    _assert_close(
        actual=report.get("threshold"),
        expected=FIXED_THRESHOLD,
        context="Baseline report threshold",
    )
    inputs = report.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("Baseline report lacks input provenance")
    expected_inputs = {
        "feature_label_csv_sha256": sha256_file(plan.feature_label_csv),
        "fold_manifest_csv_sha256": source_gate["outer_manifest_sha256"],
    }
    for field, expected_hash in expected_inputs.items():
        if inputs.get(field) != expected_hash:
            raise ValueError(f"Baseline report has mismatched {field}")
    recomputed = compute_dummy_prevalence_diagnostics(
        feature_label_csv=plan.feature_label_csv,
        fold_manifest_csv=plan.paths.fold_assignments_csv,
    )
    saved_summary = _read_csv(plan.paths.baseline_dir / "dummy_prevalence_summary.csv")
    if "baseline" not in saved_summary.columns or set(saved_summary["baseline"]) != set(
        BASELINE_ORDER
    ):
        raise ValueError("Baseline summary must contain all three required baselines")
    expected_summary = {row["baseline"]: row for row in recomputed["summary"]}
    for row in saved_summary.to_dict(orient="records"):
        baseline = str(row["baseline"])
        expected = expected_summary[baseline]
        for field in ("macro_f1", "macro_roc_auc", "macro_average_precision"):
            _assert_close(
                actual=row[field],
                expected=float(expected[field]),
                context=f"Baseline {baseline} {field}",
                atol=5e-12,
            )
    return {
        "status": "passed",
        "baselines": list(BASELINE_ORDER),
        "report_sha256": sha256_file(report_path),
        "outer_manifest_sha256": source_gate["outer_manifest_sha256"],
    }


def validate_source_grouped_diagnostics(
    *,
    plan: WorkflowPlan,
    source_gate: Mapping[str, Any],
) -> Dict[str, Any]:
    require_nonempty_paths(plan.paths.diagnostic_outputs, context="Source-grouped diagnostics")
    report_path = plan.paths.tables_dir / "source_grouped_diagnostics_report.json"
    report = read_json(report_path)
    if report.get("status") != "passed" or report.get("model_ids") != list(MODEL_IDS):
        raise ValueError("Source-grouped diagnostics report is incomplete")
    if int(report.get("n_perfumes", -1)) != int(source_gate["n_perfumes"]):
        raise ValueError("Diagnostics report has the wrong perfume count")
    _assert_close(
        actual=report.get("threshold"),
        expected=0.3,
        context="Diagnostics threshold",
    )
    bootstrap = report.get("bootstrap")
    if not isinstance(bootstrap, Mapping) or int(bootstrap.get("seed", -1)) != plan.seed:
        raise ValueError("Diagnostics report has the wrong bootstrap seed")
    if int(bootstrap.get("reps", -1)) != 2000:
        raise ValueError("Diagnostics report has the wrong bootstrap repetition count")
    provenance = report.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("Diagnostics report lacks provenance")
    provenance_inputs = provenance.get("inputs")
    provenance_outputs = provenance.get("outputs")
    if not isinstance(provenance_inputs, Mapping) or not isinstance(provenance_outputs, Mapping):
        raise ValueError("Diagnostics report has incomplete input/output hashes")
    expected_inputs = {
        "feature_label_csv": plan.feature_label_csv,
        "outer_manifest_csv": plan.paths.fold_assignments_csv,
        "inner_manifest_csv": plan.paths.inner_validation_folds_csv,
        **{
            f"oof_{model_id}": plan.paths.model_predictions(model_id)
            for model_id in MODEL_IDS
        },
    }
    for key, path in expected_inputs.items():
        entry = provenance_inputs.get(key)
        if not isinstance(entry, Mapping) or entry.get("sha256") != sha256_file(path):
            raise ValueError(f"Diagnostics report has mismatched input hash {key}")
    csv_paths = plan.paths.diagnostic_outputs[:-1]
    for path in csv_paths:
        entry = provenance_outputs.get(path.name)
        if not isinstance(entry, Mapping) or entry.get("sha256") != sha256_file(path):
            raise ValueError(f"Diagnostics report has mismatched output hash {path.name}")
    expected_rows = {
        "inner_fold_balance.csv": 10,
        "outer_fold_per_label_metrics.csv": len(MODEL_IDS) * plan.n_splits * len(OFFICIAL_LABELS),
        "component_weighted_model_metrics.csv": len(MODEL_IDS),
        "paired_component_bootstrap_macro_f1.csv": 10,
    }
    for path in csv_paths:
        if len(_read_csv(path)) != expected_rows[path.name]:
            raise ValueError(f"Diagnostics artifact has the wrong row count: {path.name}")
    return {
        "status": "passed",
        "report_sha256": sha256_file(report_path),
        "outer_manifest_sha256": source_gate["outer_manifest_sha256"],
    }


def validate_final_thesis_export(*, plan: WorkflowPlan) -> Dict[str, Any]:
    require_nonempty_paths(plan.paths.final_export_outputs, context="Final thesis export")
    table_path = plan.paths.tables_dir / "final_model_comparison_for_thesis.csv"
    table = _read_csv(table_path)
    required = {
        "model_id",
        "model_name",
        "macro_f1",
        "macro_roc_auc",
        "macro_average_precision",
    }
    if set(table.columns) != required:
        raise ValueError(
            "Thesis comparison export must use macro_average_precision and exact final columns"
        )
    if table["model_id"].duplicated().any() or set(table["model_id"]) != set(MODEL_IDS):
        raise ValueError("Thesis comparison export must contain exactly the five models")
    return {
        "status": "passed",
        "table_sha256": sha256_file(table_path),
    }


def recompute_oof_metrics(
    predictions: pd.DataFrame,
    *,
    labels: Sequence[str] = OFFICIAL_LABELS,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Recompute the fixed-threshold primary metrics from saved OOF rows."""

    per_label_rows: List[Dict[str, Any]] = []
    y_true_columns: List[np.ndarray] = []
    y_pred_columns: List[np.ndarray] = []
    for label in labels:
        truth = pd.to_numeric(predictions[f"true_{label}"], errors="raise").to_numpy(int)
        probability = pd.to_numeric(
            predictions[f"prob_{label}"], errors="raise"
        ).to_numpy(float)
        predicted = pd.to_numeric(
            predictions[f"pred_{label}"], errors="raise"
        ).to_numpy(int)
        y_true_columns.append(truth)
        y_pred_columns.append(predicted)
        if len(np.unique(truth)) != 2:
            roc_auc = float("nan")
            average_precision = float("nan") if int(truth.sum()) == 0 else float(
                average_precision_score(truth, probability)
            )
        else:
            roc_auc = float(roc_auc_score(truth, probability))
            average_precision = float(average_precision_score(truth, probability))
        per_label_rows.append(
            {
                "label": label,
                "support_pos": int(truth.sum()),
                "support_neg": int(len(truth) - truth.sum()),
                "precision": float(precision_score(truth, predicted, zero_division=0)),
                "recall": float(recall_score(truth, predicted, zero_division=0)),
                "f1": float(f1_score(truth, predicted, zero_division=0)),
                "roc_auc": roc_auc,
                "pr_auc": average_precision,
            }
        )
    per_label = pd.DataFrame(per_label_rows)
    y_true = np.column_stack(y_true_columns)
    y_pred = np.column_stack(y_pred_columns)
    metrics = {
        "macro_f1": float(per_label["f1"].mean()),
        "macro_roc_auc": float(per_label["roc_auc"].mean()),
        "macro_pr_auc": float(per_label["pr_auc"].mean()),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "subset_accuracy": float(np.mean(np.all(y_true == y_pred, axis=1))),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "label_cardinality": float(np.mean(np.sum(y_true, axis=1))),
    }
    return metrics, per_label


def validate_model_outputs(
    *,
    plan: WorkflowPlan,
    model_id: str,
    source_gate: Mapping[str, Any],
) -> Dict[str, Any]:
    paths = plan.paths
    required = _model_outputs(paths, model_id)
    require_nonempty_paths(required, context=f"{model_id} validation")
    config = load_yaml_config(plan.config_path)
    evaluation_cfg = _nested_mapping(config, "evaluation", context="Benchmark config")
    outer_cfg = _nested_mapping(
        _nested_mapping(config, "splits", context="Benchmark config"),
        "outer_cv",
        context="Benchmark config splits",
    )
    expected_threshold = float(evaluation_cfg.get("probability_threshold"))
    expected_split_design = str(outer_cfg.get("split_design_label", ""))
    if not expected_split_design:
        raise ValueError("Benchmark config lacks splits.outer_cv.split_design_label")
    outer_df, outer_group_col = _validated_outer_manifest(
        paths=paths,
        labels_csv=plan.labels_csv,
        n_splits=plan.n_splits,
    )
    expected_fold_by_id = outer_df.set_index("perfume_id")["fold_id"]

    predictions = _read_csv(paths.model_predictions(model_id))
    if not {"perfume_id", "fold_id"}.issubset(predictions.columns):
        raise ValueError(f"{model_id} OOF predictions lack perfume_id/fold_id")
    predictions["perfume_id"] = _normalize_id_series(predictions["perfume_id"])
    predictions["fold_id"] = _normalize_fold_series(
        predictions["fold_id"], context=f"{model_id} OOF predictions"
    )
    if predictions["perfume_id"].duplicated().any():
        raise ValueError(f"{model_id} OOF predictions duplicate perfume_id")
    if set(predictions["perfume_id"]) != set(outer_df["perfume_id"]):
        raise ValueError(f"{model_id} OOF perfume universe disagrees with outer folds")
    expected = predictions["perfume_id"].map(expected_fold_by_id)
    if not (predictions["fold_id"].to_numpy() == expected.to_numpy()).all():
        raise ValueError(f"{model_id} OOF fold IDs disagree with the frozen outer manifest")
    required_prediction_columns = {"source_group_id"}
    for label in OFFICIAL_LABELS:
        required_prediction_columns.update(
            {f"true_{label}", f"prob_{label}", f"pred_{label}"}
        )
    missing_prediction_columns = required_prediction_columns - set(predictions.columns)
    if missing_prediction_columns:
        raise ValueError(
            f"{model_id} OOF predictions lack columns: {sorted(missing_prediction_columns)}"
        )
    expected_groups = predictions["perfume_id"].map(
        outer_df.set_index("perfume_id")[outer_group_col]
    )
    if not (
        _normalize_id_series(predictions["source_group_id"]) == expected_groups.astype(str)
    ).all():
        raise ValueError(f"{model_id} OOF source-group IDs disagree with the frozen outer manifest")

    feature_labels = _read_csv(plan.feature_label_csv).set_index("perfume_id")
    for label in OFFICIAL_LABELS:
        probability = pd.to_numeric(predictions[f"prob_{label}"], errors="raise").to_numpy(float)
        if not np.isfinite(probability).all() or ((probability < 0.0) | (probability > 1.0)).any():
            raise ValueError(f"{model_id} OOF probability is invalid for {label}")
        truth = pd.to_numeric(predictions[f"true_{label}"], errors="raise").astype(int)
        expected_truth = predictions["perfume_id"].map(feature_labels[label]).astype(int)
        if not (truth.to_numpy() == expected_truth.to_numpy()).all():
            raise ValueError(f"{model_id} OOF true labels disagree with the frozen table for {label}")
        predicted = pd.to_numeric(predictions[f"pred_{label}"], errors="raise").astype(int)
        if not set(predicted.unique()).issubset({0, 1}):
            raise ValueError(f"{model_id} OOF predictions are non-binary for {label}")
        expected_predicted = (probability >= expected_threshold).astype(int)
        if not (predicted.to_numpy() == expected_predicted).all():
            raise ValueError(
                f"{model_id} OOF pred_{label} disagrees with probability >= "
                f"{expected_threshold}"
            )

    recomputed_metrics, recomputed_per_label = recompute_oof_metrics(predictions)

    expected_hashes = {
        "outer_manifest_sha256": source_gate["outer_manifest_sha256"],
        "inner_manifest_sha256": source_gate["inner_manifest_sha256"],
        "source_audit_sha256": source_gate["source_audit_sha256"],
    }
    is_deep_model = model_id in {"bert", "bilstm_attention"}
    expected_dataset_sha256 = sha256_file(
        paths.deep_jsonl if is_deep_model else plan.feature_label_csv
    )
    expected_frozen_inputs = (
        {
            "raw_csv": sha256_file(plan.raw_csv),
            "labels_csv": sha256_file(plan.labels_csv),
        }
        if is_deep_model
        else {"feature_label_csv": sha256_file(plan.feature_label_csv)}
    )
    expected_config_sha256 = sha256_file(plan.config_path)
    for artifact_name, artifact_path in (
        ("metrics", paths.model_metrics(model_id)),
        ("run manifest", paths.model_run_manifest(model_id)),
    ):
        payload = read_json(artifact_path)
        if payload.get("model_id") != model_id:
            raise ValueError(f"{model_id} {artifact_name} has the wrong model_id")
        if payload.get("dataset_sha256") != expected_dataset_sha256:
            raise ValueError(f"{model_id} {artifact_name} has the wrong dataset_sha256")
        if payload.get("config_sha256") != expected_config_sha256:
            raise ValueError(f"{model_id} {artifact_name} has the wrong config_sha256")
        _assert_close(
            actual=payload.get("threshold"),
            expected=expected_threshold,
            context=f"{model_id} {artifact_name} threshold",
        )
        if payload.get("split_design") != expected_split_design:
            raise ValueError(f"{model_id} {artifact_name} has the wrong split_design")
        frozen_inputs = payload.get("frozen_input_sha256")
        if not isinstance(frozen_inputs, dict):
            raise ValueError(f"{model_id} {artifact_name} lacks frozen_input_sha256")
        for key, expected_digest in expected_frozen_inputs.items():
            if frozen_inputs.get(key) != expected_digest:
                raise ValueError(
                    f"{model_id} {artifact_name} has mismatched frozen input hash {key}"
                )
        provenance = payload.get("split_provenance")
        if not isinstance(provenance, dict):
            raise ValueError(f"{model_id} {artifact_name} lacks split_provenance")
        expected_provenance = {
            "type": EXPECTED_SPLIT_TYPE,
            "split_design": expected_split_design,
            "grouping_version": EXPECTED_GROUPING_VERSION,
            "group_column": EXPECTED_GROUP_COLUMN,
        }
        for field, expected_value in expected_provenance.items():
            if provenance.get(field) != expected_value:
                raise ValueError(
                    f"{model_id} {artifact_name} has mismatched split provenance {field}"
                )
        for field, expected_hash in expected_hashes.items():
            if provenance.get(field) != expected_hash:
                raise ValueError(f"{model_id} {artifact_name} has mismatched {field}")

    metrics = read_json(paths.model_metrics(model_id))
    for field in ("macro_f1", "macro_roc_auc", "macro_pr_auc"):
        _assert_close(
            actual=metrics.get(field),
            expected=recomputed_metrics[field],
            context=f"{model_id} metrics JSON {field}",
        )
    secondary = metrics.get("secondary_metrics")
    if not isinstance(secondary, Mapping):
        raise ValueError(f"{model_id} metrics JSON lacks secondary_metrics")
    for field in ("micro_f1", "subset_accuracy", "hamming_loss", "label_cardinality"):
        _assert_close(
            actual=secondary.get(field),
            expected=recomputed_metrics[field],
            context=f"{model_id} metrics JSON secondary_metrics.{field}",
        )

    saved_per_label = _read_csv(paths.model_per_label(model_id))
    if "label" not in saved_per_label.columns:
        raise ValueError(f"{model_id} per-label table lacks label")
    if saved_per_label["label"].duplicated().any() or set(saved_per_label["label"]) != set(
        OFFICIAL_LABELS
    ):
        raise ValueError(f"{model_id} per-label table must contain every official label once")
    saved_per_label = saved_per_label.set_index("label")
    recomputed_by_label = recomputed_per_label.set_index("label")
    for label in OFFICIAL_LABELS:
        for field in ("support_pos", "support_neg"):
            actual = int(pd.to_numeric(saved_per_label.loc[label, field], errors="raise"))
            expected_value = int(recomputed_by_label.loc[label, field])
            if actual != expected_value:
                raise ValueError(f"{model_id} per-label {field} mismatch for {label}")
        for field in ("f1", "roc_auc", "pr_auc"):
            _assert_close(
                actual=saved_per_label.loc[label, field],
                expected=float(recomputed_by_label.loc[label, field]),
                context=f"{model_id} per-label {field} for {label}",
            )

    json_per_label = metrics.get("per_label")
    if not isinstance(json_per_label, list):
        raise ValueError(f"{model_id} metrics JSON lacks per_label rows")
    json_rows = {
        str(row.get("label")): row for row in json_per_label if isinstance(row, Mapping)
    }
    if set(json_rows) != set(OFFICIAL_LABELS):
        raise ValueError(f"{model_id} metrics JSON per_label set is incomplete")
    for label in OFFICIAL_LABELS:
        row = json_rows[label]
        for field in ("support_pos", "support_neg"):
            if int(row.get(field, -1)) != int(recomputed_by_label.loc[label, field]):
                raise ValueError(f"{model_id} metrics JSON per_label {field} mismatch for {label}")
        for field in ("f1", "roc_auc", "pr_auc"):
            _assert_close(
                actual=row.get(field),
                expected=float(recomputed_by_label.loc[label, field]),
                context=f"{model_id} metrics JSON per_label {field} for {label}",
            )
    return {
        "status": "passed",
        "model_id": model_id,
        "n_oof_rows": int(len(predictions)),
        "threshold": expected_threshold,
        "split_design": expected_split_design,
        **recomputed_metrics,
        **expected_hashes,
    }


def validate_final_outputs(
    *,
    plan: WorkflowPlan,
    source_gate: Mapping[str, Any],
) -> Dict[str, Any]:
    model_checks = {
        model_id: validate_model_outputs(plan=plan, model_id=model_id, source_gate=source_gate)
        for model_id in MODEL_IDS
    }
    comparison_path = plan.paths.tables_dir / "comparison_table.csv"
    require_paths((comparison_path,), context="Final audit")
    comparison = _read_csv(comparison_path)
    if "model_id" not in comparison.columns:
        raise ValueError("Comparison table lacks model_id")
    final_rows = comparison.loc[comparison["model_id"].isin(MODEL_IDS)].copy()
    if final_rows["model_id"].duplicated().any() or set(final_rows["model_id"]) != set(MODEL_IDS):
        raise ValueError("Comparison table must contain exactly one row for every final model")
    required_comparison_columns = {
        "dataset",
        "target_columns",
        "split_design",
        "threshold",
        "macro_f1",
        "macro_roc_auc",
        "macro_pr_auc",
        "reproducible_run",
        "saved_metrics",
        "comparable_plots",
        "interpretation_note",
        "row_complete",
        "note_path",
    }
    missing_columns = required_comparison_columns - set(final_rows.columns)
    if missing_columns:
        raise ValueError(
            f"Comparison table lacks required columns: {sorted(missing_columns)}"
        )
    truthy = {"true", "1", "yes"}
    for flag in (
        "reproducible_run",
        "saved_metrics",
        "comparable_plots",
        "interpretation_note",
        "row_complete",
    ):
        if not final_rows[flag].astype(str).str.lower().isin(truthy).all():
            raise ValueError(f"Comparison table contains false/incomplete {flag}")
    target_string = ";".join(OFFICIAL_LABELS)
    for model_id in MODEL_IDS:
        row = final_rows.loc[final_rows["model_id"] == model_id].iloc[0]
        check = model_checks[model_id]
        expected_dataset = stringify_path(
            plan.paths.deep_jsonl if model_id in {"bert", "bilstm_attention"} else plan.feature_label_csv,
            project_root=plan.project_root,
        )
        if str(row["dataset"]) != expected_dataset:
            raise ValueError(f"Comparison table has the wrong dataset for {model_id}")
        if str(row["target_columns"]) != target_string:
            raise ValueError(f"Comparison table has the wrong target columns for {model_id}")
        if str(row["split_design"]) != check["split_design"]:
            raise ValueError(f"Comparison table has the wrong split design for {model_id}")
        _assert_close(
            actual=row["threshold"],
            expected=float(check["threshold"]),
            context=f"Comparison table threshold for {model_id}",
        )
        for field in ("macro_f1", "macro_roc_auc", "macro_pr_auc"):
            _assert_close(
                actual=row[field],
                expected=float(check[field]),
                context=f"Comparison table {field} for {model_id}",
                atol=5e-7,
            )
        expected_note = stringify_path(
            plan.paths.model_note(model_id), project_root=plan.project_root
        )
        if str(row["note_path"]) != expected_note:
            raise ValueError(f"Comparison table has the wrong note path for {model_id}")
    return {
        "status": "passed",
        "models": list(MODEL_IDS),
        "outer_manifest_sha256": source_gate["outer_manifest_sha256"],
        "model_checks": model_checks,
    }


CommandExecutor = Callable[[str, Sequence[str], Path, Path], None]


class SourceGroupedBenchmarkRunner:
    """Fail-fast command runner with fingerprinted stage receipts."""

    def __init__(
        self,
        *,
        plan: WorkflowPlan,
        selected_stages: Sequence[StageSpec],
        dry_run: bool,
        resume: bool,
        command_executor: CommandExecutor | None = None,
    ) -> None:
        self.plan = plan
        self.selected_stages = list(selected_stages)
        self.dry_run = dry_run
        self.resume = resume
        self.command_executor = command_executor or self._default_command_executor
        self.history: List[Dict[str, Any]] = []
        self._source_gate_cache: Dict[str, Any] | None = None
        self._deep_gate_cache: Dict[str, Any] | None = None
        self._baseline_gate_cache: Dict[str, Any] | None = None
        self._diagnostics_gate_cache: Dict[str, Any] | None = None

    def _default_command_executor(
        self,
        stage_name: str,
        command: Sequence[str],
        cwd: Path,
        log_path: Path,
    ) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        with log_path.open("a", encoding="utf-8") as log_handle:
            log_handle.write(f"[{now_utc()}] {quote_command(command)}\n")
            log_handle.flush()
            process = subprocess.Popen(
                list(command),
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
            assert process.stdout is not None
            try:
                for raw_line in process.stdout:
                    print(raw_line, end="")
                    log_handle.write(raw_line)
                    log_handle.flush()
                return_code = process.wait()
            except BaseException:
                # Do not leave a costly model process orphaned if the workflow
                # receives Ctrl-C or another non-standard interruption.
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                raise
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, list(command))

    def write_manifest(self) -> Dict[str, Any]:
        manifest = build_command_manifest(
            plan=self.plan,
            selected_stages=self.selected_stages,
            dry_run=self.dry_run,
            resume=self.resume,
        )
        write_json(self.plan.paths.command_manifest_json, manifest)
        return manifest

    def _write_status(self, *, status: str, current_stage: str | None = None, error: str | None = None) -> None:
        payload: Dict[str, Any] = {
            "workflow_id": WORKFLOW_ID,
            "status": status,
            "updated_at": now_utc(),
            "dry_run": self.dry_run,
            "resume": self.resume,
            "selected_stages": [stage.name for stage in self.selected_stages],
            "current_stage": current_stage,
            "history": self.history,
            "command_manifest": stringify_path(
                self.plan.paths.command_manifest_json,
                project_root=self.plan.project_root,
            ),
        }
        if error is not None:
            payload["error"] = error
        write_json(self.plan.paths.run_status_json, payload)

    def _receipt_is_valid(self, stage: StageSpec) -> bool:
        receipt_path = self.plan.paths.stage_receipt(stage.name)
        if not receipt_path.is_file():
            return False
        try:
            receipt = read_json(receipt_path)
        except (OSError, ValueError, json.JSONDecodeError):
            return False
        return (
            receipt.get("status") == "completed"
            and receipt.get("fingerprint")
            == _stage_fingerprint(
                stage,
                project_root=self.plan.project_root,
                config_path=self.plan.config_path,
            )
            and all(path.is_file() for path in stage.expected_outputs)
        )

    def _write_receipt(
        self,
        *,
        stage: StageSpec,
        status: str,
        checks: Mapping[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        payload: Dict[str, Any] = {
            "workflow_id": WORKFLOW_ID,
            "stage": stage.name,
            "status": status,
            "updated_at": now_utc(),
            "fingerprint": _stage_fingerprint(
                stage,
                project_root=self.plan.project_root,
                config_path=self.plan.config_path,
            ),
            "code_provenance": current_code_provenance(
                project_root=self.plan.project_root,
                config_path=self.plan.config_path,
            ),
            "commands": [list(command) for command in stage.commands],
            "expected_outputs": [
                stringify_path(path, project_root=self.plan.project_root)
                for path in stage.expected_outputs
            ],
        }
        if checks is not None:
            payload["checks"] = dict(checks)
        if error is not None:
            payload["error"] = error
        write_json(self.plan.paths.stage_receipt(stage.name), payload)

    def _require_current_receipts(self, stage_names: Sequence[str]) -> None:
        stages_by_name = {stage.name: stage for stage in self.plan.stages}
        stale = [
            name
            for name in stage_names
            if name not in stages_by_name or not self._receipt_is_valid(stages_by_name[name])
        ]
        if stale:
            raise ValueError(
                "Final audit requires current code/config-bound stage receipts: "
                + ", ".join(stale)
            )

    def _source_gate(self, *, force: bool = False) -> Dict[str, Any]:
        if self._source_gate_cache is None or force:
            self._source_gate_cache = validate_source_group_audit(
                paths=self.plan.paths,
                labels_csv=self.plan.labels_csv,
                n_splits=self.plan.n_splits,
                raw_csv=self.plan.raw_csv,
                feature_label_csv=self.plan.feature_label_csv,
                config_path=self.plan.config_path,
            )
        return self._source_gate_cache

    def _deep_gate(self, *, force: bool = False) -> Dict[str, Any]:
        if self._deep_gate_cache is None or force:
            self._deep_gate_cache = validate_deep_dataset(
                plan=self.plan,
                source_gate=self._source_gate(),
            )
        return self._deep_gate_cache

    def _baseline_gate(self, *, force: bool = False) -> Dict[str, Any]:
        if self._baseline_gate_cache is None or force:
            self._baseline_gate_cache = validate_baseline_outputs(
                plan=self.plan,
                source_gate=self._source_gate(),
            )
        return self._baseline_gate_cache

    def _diagnostics_gate(self, *, force: bool = False) -> Dict[str, Any]:
        if self._diagnostics_gate_cache is None or force:
            self._diagnostics_gate_cache = validate_source_grouped_diagnostics(
                plan=self.plan,
                source_gate=self._source_gate(),
            )
        return self._diagnostics_gate_cache

    def _validate_inputs_for_prepare(self) -> None:
        require_paths(
            (
                self.plan.config_path,
                self.plan.raw_csv,
                self.plan.labels_csv,
                self.plan.feature_label_csv,
            ),
            context="Preparation",
        )
        validate_configured_frozen_inputs(
            config_path=self.plan.config_path,
            input_paths={
                "raw_csv": self.plan.raw_csv,
                "labels_csv": self.plan.labels_csv,
                "feature_label_csv": self.plan.feature_label_csv,
            },
        )

    def _validate_completed_stage(self, stage: StageSpec) -> Dict[str, Any] | None:
        name = stage.name
        if name == "prepare":
            require_paths(stage.expected_outputs, context="Resumed preparation")
            return None
        if name == "source_audit_gate":
            return self._source_gate(force=True)
        if name == "dummy_prevalence_baselines":
            return self._baseline_gate(force=True)
        if name == "logistic_regression_smoke":
            return validate_model_outputs(
                plan=self.plan,
                model_id="logistic_regression",
                source_gate=self._source_gate(),
            )
        if name in {"random_forest", "xgboost", "bert", "bilstm_attention"}:
            return validate_model_outputs(
                plan=self.plan,
                model_id=name,
                source_gate=self._source_gate(),
            )
        if name == "deep_dataset":
            require_paths(stage.expected_outputs, context="Resumed deep dataset")
            return None
        if name == "deep_dataset_audit_gate":
            return self._deep_gate(force=True)
        if name == "source_grouped_diagnostics":
            return self._diagnostics_gate(force=True)
        if name == "final_thesis_export":
            return validate_final_thesis_export(plan=self.plan)
        if name == "final_audit_gate":
            self._baseline_gate(force=True)
            self._diagnostics_gate(force=True)
            validate_final_thesis_export(plan=self.plan)
            return validate_final_outputs(plan=self.plan, source_gate=self._source_gate(force=True))
        raise ValueError(f"Unsupported stage: {name}")

    def _preflight(self, stage: StageSpec) -> None:
        name = stage.name
        if name == "prepare":
            self._validate_inputs_for_prepare()
            return
        self._source_gate(force=name == "source_audit_gate")
        require_paths((self.plan.config_path,), context=f"{name} configuration") if name not in {
            "source_audit_gate",
            "deep_dataset",
            "deep_dataset_audit_gate",
        } else None
        if name in {
            "random_forest",
            "xgboost",
            "deep_dataset",
            "deep_dataset_audit_gate",
            "bert",
            "bilstm_attention",
            "source_grouped_diagnostics",
            "final_thesis_export",
            "final_audit_gate",
        }:
            validate_model_outputs(
                plan=self.plan,
                model_id="logistic_regression",
                source_gate=self._source_gate(),
            )
        if name in {
            "deep_dataset",
            "deep_dataset_audit_gate",
            "bert",
            "bilstm_attention",
            "source_grouped_diagnostics",
            "final_thesis_export",
            "final_audit_gate",
        }:
            for model_id in ("random_forest", "xgboost"):
                validate_model_outputs(
                    plan=self.plan,
                    model_id=model_id,
                    source_gate=self._source_gate(),
                )
        if name in {
            "bert",
            "bilstm_attention",
            "source_grouped_diagnostics",
            "final_thesis_export",
            "final_audit_gate",
        }:
            self._deep_gate()
        if name in {
            "bilstm_attention",
            "source_grouped_diagnostics",
            "final_thesis_export",
            "final_audit_gate",
        }:
            validate_model_outputs(
                plan=self.plan,
                model_id="bert",
                source_gate=self._source_gate(),
            )
        if name in {"source_grouped_diagnostics", "final_thesis_export", "final_audit_gate"}:
            self._baseline_gate()
            validate_model_outputs(
                plan=self.plan,
                model_id="bilstm_attention",
                source_gate=self._source_gate(),
            )
        if name in {"final_thesis_export", "final_audit_gate"}:
            self._diagnostics_gate()
        if name == "final_audit_gate":
            self._require_current_receipts(
                (
                    "source_audit_gate",
                    "dummy_prevalence_baselines",
                    "logistic_regression_smoke",
                    "random_forest",
                    "xgboost",
                    "deep_dataset",
                    "deep_dataset_audit_gate",
                    "bert",
                    "bilstm_attention",
                    "source_grouped_diagnostics",
                    "final_thesis_export",
                )
            )

    def _execute_stage(self, stage: StageSpec) -> Dict[str, Any] | None:
        self._preflight(stage)
        if stage.is_gate:
            return self._validate_completed_stage(stage)

        log_path = self.plan.paths.stage_log(stage.name)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        for command in stage.commands:
            print(f"[{stage.name}] {quote_command(command)}")
            self.command_executor(stage.name, command, self.plan.project_root, log_path)
        require_paths(stage.expected_outputs, context=f"Completed stage {stage.name}")
        return self._validate_completed_stage(stage)

    def run(self) -> Dict[str, Any]:
        manifest = self.write_manifest()
        if self.dry_run:
            for item in manifest["resolved_commands"]:
                print(item["command"])
            self._write_status(status="dry_run")
            return {"status": "dry_run", "command_manifest": self.plan.paths.command_manifest_json}

        self._write_status(status="running")
        for stage in self.selected_stages:
            self._write_status(status="running", current_stage=stage.name)
            if self.resume and not stage.is_gate and self._receipt_is_valid(stage):
                try:
                    checks = self._validate_completed_stage(stage)
                except Exception:
                    # Stale or corrupt artifacts invalidate an otherwise valid receipt.
                    pass
                else:
                    self.history.append(
                        {"stage": stage.name, "status": "skipped_completed", "updated_at": now_utc()}
                    )
                    self._write_status(status="running", current_stage=stage.name)
                    continue
            try:
                checks = self._execute_stage(stage)
            except Exception as exc:
                self._write_receipt(stage=stage, status="failed", error=str(exc))
                self.history.append(
                    {"stage": stage.name, "status": "failed", "error": str(exc), "updated_at": now_utc()}
                )
                self._write_status(status="failed", current_stage=stage.name, error=str(exc))
                raise
            self._write_receipt(stage=stage, status="completed", checks=checks)
            self.history.append({"stage": stage.name, "status": "completed", "updated_at": now_utc()})

        self._write_status(status="completed")
        return {
            "status": "completed",
            "selected_stages": [stage.name for stage in self.selected_stages],
            "command_manifest": self.plan.paths.command_manifest_json,
            "run_status": self.plan.paths.run_status_json,
        }


def list_stages(plan: WorkflowPlan) -> None:
    print("Ordered source-grouped benchmark stages:")
    for index, stage in enumerate(plan.stages, start=1):
        kind = "hard gate" if stage.is_gate else "command"
        print(f"{index:>2}. {stage.name:<27} [{kind}]")
        print(f"    {stage.description}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Gemini-browser OG784 source-grouped benchmark in a new output root.",
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--raw-csv", type=Path, default=DEFAULT_RAW_CSV)
    parser.add_argument("--labels-csv", type=Path, default=DEFAULT_LABELS_CSV)
    parser.add_argument("--feature-label-csv", type=Path, default=DEFAULT_FEATURE_LABEL_CSV)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--inner-n-splits", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument(
        "--python-executable",
        default=None,
        help="Interpreter for child stages (default: <project-root>/.venv/bin/python).",
    )
    parser.add_argument("--list-stages", action="store_true")
    parser.add_argument("--only", nargs="+", default=None)
    parser.add_argument("--from-stage", "--from", dest="from_stage", default=None)
    parser.add_argument("--to-stage", "--to", dest="to_stage", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = args.project_root.resolve()
    python_executable = args.python_executable or str(project_root / ".venv/bin/python")
    plan = build_workflow_plan(
        project_root=project_root,
        output_root=args.out_root,
        config_path=args.config,
        raw_csv=args.raw_csv,
        labels_csv=args.labels_csv,
        feature_label_csv=args.feature_label_csv,
        python_executable=python_executable,
        n_splits=args.n_splits,
        inner_n_splits=args.inner_n_splits,
        seed=args.seed,
        device=args.device,
    )
    try:
        validate_output_root(output_root=plan.paths.output_root, project_root=project_root)
        selected = select_stages(
            plan=plan,
            only=args.only,
            from_stage=args.from_stage,
            to_stage=args.to_stage,
        )
        ensure_fresh_or_resumable_root(
            paths=plan.paths,
            resume=args.resume,
            dry_run=args.dry_run,
        )
    except (ValueError, FileExistsError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    if args.list_stages:
        list_stages(plan)
        return 0
    if not selected:
        print("No stages selected.")
        return 0

    print(f"Experiment root: {stringify_path(plan.paths.output_root, project_root=project_root)}")
    print("Stage plan: " + " -> ".join(stage.name for stage in selected))
    runner = SourceGroupedBenchmarkRunner(
        plan=plan,
        selected_stages=selected,
        dry_run=args.dry_run,
        resume=args.resume,
    )
    try:
        result = runner.run()
    except Exception as exc:  # noqa: BLE001 - top-level orchestration boundary
        print(f"Workflow failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"Workflow status: {result['status']}")
    print(
        "Command manifest: "
        + stringify_path(plan.paths.command_manifest_json, project_root=project_root)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

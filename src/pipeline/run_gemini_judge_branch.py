# Date Created: 2026-05-20

"""Standalone Phase 1 Gemini judge branch runner that wraps existing repo scripts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence

import pandas as pd
import yaml

from src.utils.paths import PROJECT_ROOT


CANONICAL_STAGES = [
    "dry-run",
    "labeling",
    "merge",
    "gold_eval",
    "benchmark_prep",
    "benchmark",
    "decision",
]
BENCHMARK_MODELS = ["logistic_regression", "xgboost", "random_forest", "bert", "bilstm_attention"]


@dataclass(frozen=True)
class BranchConfig:
    config_path: Path
    branch_id: str
    output_tag: str
    stage_order: tuple[str, ...]
    baseline_label_csv: Path
    baseline_provenance_jsonl: Path
    baseline_merge_report_json: Path
    baseline_benchmark_comparison_csv: Path
    baseline_fold_manifest_csv: Path
    target_csv: Path
    packet_csv: Path
    raw_csv: Path
    unified_csv: Path
    perfume_col: str
    text_col: str
    gold_packet_csv: Path
    evaluation_config_yaml: Path
    judge_model_name: str
    judge_temperature: float
    judge_max_retries: int
    judge_backoff_base_sec: float
    judge_backoff_cap_sec: float
    judge_rate_limit_sleep_sec: float
    judge_batch_size: int
    judge_seed: int
    judge_snippet_max_chars: int
    benchmark_threshold: float
    run_threshold_sweep: bool
    benchmark_models: tuple[str, ...]
    benchmark_device_default: str
    output_root: Path
    analysis_out_dir: Path
    gold_eval_out_dir: Path
    benchmark_root: Path
    decision_out_dir: Path
    progress_dir: Path
    logs_dir: Path
    judge_labels_jsonl: Path
    judge_labels_csv: Path
    judge_labeling_report_json: Path
    judge_labeling_errors_jsonl: Path
    judge_labeling_run_log: Path
    judge_merge_report_json: Path
    judge_merged_csv: Path
    benchmark_dataset_jsonl: Path
    benchmark_dataset_chunks_csv: Path
    benchmark_dataset_summary_json: Path
    judge_benchmark_comparison_csv: Path
    baseline_gold_metrics_json: Path
    judge_gold_metrics_json: Path
    decision_summary_json: Path
    branch_status_json: Path
    stage_status_jsons: dict[str, Path]


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def stringify_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def quote_command(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid YAML config: {path}")
    return payload


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def ensure_paths_exist(paths: Iterable[Path], *, allow_missing_packet: bool = False) -> None:
    for path in paths:
        if allow_missing_packet and path.name.startswith("Annotation_Packet_"):
            continue
        if not path.exists():
            raise FileNotFoundError(f"Required path not found: {path}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_tree(root: Path) -> Dict[str, Dict[str, int]]:
    snapshot: Dict[str, Dict[str, int]] = {}
    if not root.exists():
        return snapshot
    for file_path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = str(file_path.relative_to(root))
        stat = file_path.stat()
        snapshot[rel] = {"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}
    return snapshot


def load_branch_config(path: Path) -> BranchConfig:
    payload = read_yaml(path)

    branch = payload["branch"]
    baseline = payload["baseline_anchors"]
    raw_inputs = payload["raw_inputs"]
    gold_eval = payload["gold_eval"]
    judge_model = payload["judge_model"]
    benchmark = payload["benchmark"]
    output_roots = payload["output_roots"]
    artifacts = payload["artifacts"]
    status_json = payload["status_json"]

    stage_order = tuple(branch["stage_order"])
    stage_status_jsons = {
        "labeling": resolve_path(status_json["labeling_status_json"]),
        "merge": resolve_path(status_json["merge_status_json"]),
        "gold_eval": resolve_path(status_json["gold_eval_status_json"]),
        "benchmark_prep": resolve_path(status_json["benchmark_prep_status_json"]),
        "benchmark": resolve_path(status_json["benchmark_status_json"]),
        "decision": resolve_path(status_json["decision_status_json"]),
    }

    config = BranchConfig(
        config_path=path,
        branch_id=str(branch["branch_id"]),
        output_tag=str(branch["output_tag"]),
        stage_order=stage_order,
        baseline_label_csv=resolve_path(baseline["canonical_label_csv"]),
        baseline_provenance_jsonl=resolve_path(baseline["provenance_jsonl"]),
        baseline_merge_report_json=resolve_path(baseline["merge_report_json"]),
        baseline_benchmark_comparison_csv=resolve_path(baseline["benchmark_comparison_table_csv"]),
        baseline_fold_manifest_csv=resolve_path(baseline["benchmark_fold_manifest_csv"]),
        target_csv=resolve_path(raw_inputs["target_csv"]),
        packet_csv=resolve_path(raw_inputs["packet_csv"]),
        raw_csv=resolve_path(raw_inputs["raw_csv"]),
        unified_csv=resolve_path(raw_inputs["unified_csv"]),
        perfume_col=str(raw_inputs["perfume_col"]),
        text_col=str(raw_inputs["text_col"]),
        gold_packet_csv=resolve_path(gold_eval["gold_packet_csv"]),
        evaluation_config_yaml=resolve_path(gold_eval["evaluation_config_yaml"]),
        judge_model_name=str(judge_model["model"]),
        judge_temperature=float(judge_model["temperature"]),
        judge_max_retries=int(judge_model["max_retries"]),
        judge_backoff_base_sec=float(judge_model["backoff_base_sec"]),
        judge_backoff_cap_sec=float(judge_model["backoff_cap_sec"]),
        judge_rate_limit_sleep_sec=float(judge_model["rate_limit_sleep_sec"]),
        judge_batch_size=int(judge_model["batch_size"]),
        judge_seed=int(judge_model["seed"]),
        judge_snippet_max_chars=int(judge_model["snippet_max_chars"]),
        benchmark_threshold=float(benchmark["probability_threshold"]),
        run_threshold_sweep=bool(benchmark["run_threshold_sweep"]),
        benchmark_models=tuple(str(model_id) for model_id in benchmark["models"]),
        benchmark_device_default=str(benchmark["device_default"]),
        output_root=resolve_path(output_roots["root_dir"]),
        analysis_out_dir=resolve_path(output_roots["analysis_out_dir"]),
        gold_eval_out_dir=resolve_path(output_roots["gold_eval_out_dir"]),
        benchmark_root=resolve_path(output_roots["model_benchmark_root"]),
        decision_out_dir=resolve_path(output_roots["decision_out_dir"]),
        progress_dir=resolve_path(output_roots["progress_dir"]),
        logs_dir=resolve_path(output_roots["logs_dir"]),
        judge_labels_jsonl=resolve_path(artifacts["judge_labels_jsonl"]),
        judge_labels_csv=resolve_path(artifacts["judge_labels_csv"]),
        judge_labeling_report_json=resolve_path(artifacts["judge_labeling_report_json"]),
        judge_labeling_errors_jsonl=resolve_path(artifacts["judge_labeling_errors_jsonl"]),
        judge_labeling_run_log=resolve_path(artifacts["judge_labeling_run_log"]),
        judge_merge_report_json=resolve_path(artifacts["judge_merge_report_json"]),
        judge_merged_csv=resolve_path(artifacts["judge_merged_csv"]),
        benchmark_dataset_jsonl=resolve_path(artifacts["benchmark_dataset_jsonl"]),
        benchmark_dataset_chunks_csv=resolve_path(artifacts["benchmark_dataset_chunks_csv"]),
        benchmark_dataset_summary_json=resolve_path(artifacts["benchmark_dataset_summary_json"]),
        judge_benchmark_comparison_csv=resolve_path(artifacts["benchmark_comparison_table_csv"]),
        baseline_gold_metrics_json=resolve_path(artifacts["baseline_gold_metrics_json"]),
        judge_gold_metrics_json=resolve_path(artifacts["judge_gold_metrics_json"]),
        decision_summary_json=resolve_path(artifacts["decision_summary_json"]),
        branch_status_json=resolve_path(status_json["branch_status_json"]),
        stage_status_jsons=stage_status_jsons,
    )
    validate_config(config)
    return config


def validate_config(config: BranchConfig) -> None:
    if list(config.stage_order) != CANONICAL_STAGES:
        raise ValueError(f"Stage order must match {CANONICAL_STAGES}, found {list(config.stage_order)}")
    if config.judge_model_name != "gemini-2.5-pro":
        raise ValueError("Phase 1 judge branch must use judge_model.model=gemini-2.5-pro")
    if round(config.benchmark_threshold, 6) != 0.3:
        raise ValueError("Phase 1 benchmark threshold must remain 0.3")
    if config.run_threshold_sweep:
        raise ValueError("Phase 1 judge branch must keep run_threshold_sweep=false")
    if list(config.benchmark_models) != BENCHMARK_MODELS:
        raise ValueError(f"Benchmark models must be {BENCHMARK_MODELS}")

    ensure_paths_exist(
        [
            config.baseline_label_csv,
            config.baseline_provenance_jsonl,
            config.baseline_merge_report_json,
            config.baseline_benchmark_comparison_csv,
            config.baseline_fold_manifest_csv,
            config.target_csv,
            config.packet_csv,
            config.raw_csv,
            config.unified_csv,
            config.gold_packet_csv,
            config.evaluation_config_yaml,
        ],
    )

    venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
    if not venv_python.exists():
        raise FileNotFoundError(f"Expected project interpreter not found: {venv_python}")


def normalize_stage_args(values: Sequence[str] | None) -> List[str]:
    if not values:
        return []
    out: List[str] = []
    for value in values:
        for part in str(value).split(","):
            stage = part.strip()
            if not stage:
                continue
            if stage not in CANONICAL_STAGES:
                raise ValueError(f"Unknown stage: {stage}")
            if stage not in out:
                out.append(stage)
    return out


def select_stages(
    *,
    config: BranchConfig,
    only: Sequence[str] | None,
    from_stage: str | None,
    to_stage: str | None,
) -> List[str]:
    only_stages = normalize_stage_args(only)
    ordered = list(config.stage_order)

    if only_stages:
        if from_stage or to_stage:
            raise ValueError("--only cannot be combined with --from-stage/--to-stage")
        return [stage for stage in ordered if stage in set(only_stages)]

    start_idx = ordered.index(from_stage) if from_stage else 0
    end_idx = ordered.index(to_stage) if to_stage else len(ordered) - 1
    if start_idx > end_idx:
        raise ValueError("--from-stage must come before --to-stage")
    return ordered[start_idx : end_idx + 1]


class BranchRunner:
    def __init__(
        self,
        *,
        config: BranchConfig,
        selected_stages: Sequence[str],
        dry_run: bool,
        resume: bool,
        benchmark_device: str,
    ) -> None:
        self.config = config
        self.selected_stages = list(selected_stages)
        self.dry_run = dry_run
        self.resume = resume
        self.benchmark_device = benchmark_device or config.benchmark_device_default
        self.branch_started_at = now_utc()
        self.stage_history: List[Dict[str, Any]] = []

    def stage_log_path(self, stage: str) -> Path:
        return self.config.logs_dir / f"{stage}.log"

    def update_branch_status(
        self,
        *,
        current_stage: str,
        stage_status: str,
        latest_artifact: str | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        payload: Dict[str, Any] = {
            "branch_id": self.config.branch_id,
            "config_path": stringify_path(self.config.config_path),
            "selected_stages": list(self.selected_stages),
            "dry_run": self.dry_run,
            "resume": self.resume,
            "started_at": self.branch_started_at,
            "updated_at": now_utc(),
            "current_stage": current_stage,
            "stage_status": stage_status,
            "latest_artifact": latest_artifact,
            "stage_history": self.stage_history,
        }
        if extra:
            payload.update(dict(extra))
        write_json(self.config.branch_status_json, payload)

    def update_stage_status(
        self,
        stage: str,
        *,
        status: str,
        log_path: Path | None = None,
        latest_artifact: Path | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        status_path = self.config.stage_status_jsons.get(stage)
        if status_path is None:
            return
        payload: Dict[str, Any] = {
            "branch_id": self.config.branch_id,
            "stage": stage,
            "status": status,
            "updated_at": now_utc(),
        }
        if log_path is not None:
            payload["log_path"] = stringify_path(log_path)
        if latest_artifact is not None:
            payload["latest_artifact"] = stringify_path(latest_artifact)
        if extra:
            payload.update(dict(extra))
        write_json(status_path, payload)

    def append_log(self, stage: str, message: str) -> None:
        log_path = self.stage_log_path(stage)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"[{now_utc()}] {message}\n")

    def plan_stage_actions(self, stage: str) -> List[str]:
        if stage == "dry-run":
            actions: List[str] = []
            for next_stage in self.selected_stages:
                if next_stage == "dry-run":
                    continue
                actions.extend(self.plan_stage_actions(next_stage))
            return actions
        if stage == "labeling":
            return [quote_command(self.build_labeling_command())]
        if stage == "merge":
            return [quote_command(self.build_merge_command())]
        if stage == "gold_eval":
            return [quote_command(self.build_gold_eval_command())]
        if stage == "benchmark_prep":
            return [
                f"COPY {stringify_path(self.config.baseline_fold_manifest_csv)} -> "
                f"{stringify_path(self.config.benchmark_root / 'tables' / 'fold_assignments.csv')}",
                quote_command(self.build_benchmark_prep_command()),
            ]
        if stage == "benchmark":
            return [quote_command(command) for command in self.build_benchmark_commands().values()]
        if stage == "decision":
            return [quote_command(self.build_decision_command())]
        raise ValueError(f"Unsupported stage: {stage}")

    def build_labeling_command(self) -> List[str]:
        return [
            str(PROJECT_ROOT / ".venv" / "bin" / "python"),
            "-m",
            "src.labeling.run_gemini_labeling",
            "--target-csv",
            stringify_path(self.config.target_csv),
            "--packet-csv",
            stringify_path(self.config.packet_csv),
            "--raw-csv",
            stringify_path(self.config.raw_csv),
            "--perfume-col",
            self.config.perfume_col,
            "--text-col",
            self.config.text_col,
            "--model",
            self.config.judge_model_name,
            "--temperature",
            str(self.config.judge_temperature),
            "--max-retries",
            str(self.config.judge_max_retries),
            "--backoff-base-sec",
            str(self.config.judge_backoff_base_sec),
            "--backoff-cap-sec",
            str(self.config.judge_backoff_cap_sec),
            "--rate-limit-sleep-sec",
            str(self.config.judge_rate_limit_sleep_sec),
            "--batch-size",
            str(self.config.judge_batch_size),
            "--seed",
            str(self.config.judge_seed),
            "--snippet-max-chars",
            str(self.config.judge_snippet_max_chars),
            "--output-tag",
            self.config.output_tag,
        ]

    def build_merge_command(self) -> List[str]:
        return [
            str(PROJECT_ROOT / ".venv" / "bin" / "python"),
            "-m",
            "src.analysis.analyze_gemini_unified",
            "--unified-csv",
            stringify_path(self.config.unified_csv),
            "--gemini-csv",
            stringify_path(self.config.judge_labels_csv),
            "--out-dir",
            stringify_path(self.config.analysis_out_dir),
            "--perfume-col",
            self.config.perfume_col,
            "--strict",
        ]

    def build_gold_eval_command(self) -> List[str]:
        return [
            str(PROJECT_ROOT / ".venv" / "bin" / "python"),
            "-m",
            "src.analysis.compare_label_branch_to_gold43",
            "--baseline-csv",
            stringify_path(self.config.baseline_label_csv),
            "--judge-csv",
            stringify_path(self.config.judge_labels_csv),
            "--gold-csv",
            stringify_path(self.config.gold_packet_csv),
            "--config",
            stringify_path(self.config.evaluation_config_yaml),
            "--out-dir",
            stringify_path(self.config.gold_eval_out_dir),
        ]

    def build_benchmark_prep_command(self) -> List[str]:
        return [
            str(PROJECT_ROOT / ".venv" / "bin" / "python"),
            "-m",
            "src.training.build_deep_model_text_dataset",
            "--raw-csv",
            stringify_path(self.config.raw_csv),
            "--labels-csv",
            stringify_path(self.config.judge_labels_csv),
            "--folds-csv",
            stringify_path(self.config.benchmark_root / "tables" / "fold_assignments.csv"),
            "--output-jsonl",
            stringify_path(self.config.benchmark_dataset_jsonl),
            "--output-chunks-csv",
            stringify_path(self.config.benchmark_dataset_chunks_csv),
            "--output-summary-json",
            stringify_path(self.config.benchmark_dataset_summary_json),
        ]

    def build_benchmark_commands(self) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        base_python = str(PROJECT_ROOT / ".venv" / "bin" / "python")
        merged_csv = stringify_path(self.config.judge_merged_csv)
        benchmark_root = stringify_path(self.config.benchmark_root)
        benchmark_jsonl = stringify_path(self.config.benchmark_dataset_jsonl)
        config_path = stringify_path(self.config.evaluation_config_yaml)
        for model_id in self.config.benchmark_models:
            if model_id in {"logistic_regression", "xgboost", "random_forest"}:
                out[model_id] = [
                    base_python,
                    "-m",
                    f"src.training.run_{model_id}_benchmark",
                    "--config",
                    config_path,
                    "--input-csv",
                    merged_csv,
                    "--out-root",
                    benchmark_root,
                ]
            else:
                out[model_id] = [
                    base_python,
                    "-m",
                    f"src.training.run_{model_id}_benchmark",
                    "--config",
                    config_path,
                    "--input-jsonl",
                    benchmark_jsonl,
                    "--out-root",
                    benchmark_root,
                    "--device",
                    self.benchmark_device,
                ]
        return out

    def build_decision_command(self) -> List[str]:
        return [
            str(PROJECT_ROOT / ".venv" / "bin" / "python"),
            "-m",
            "src.analysis.compare_gemini_branches",
            "--baseline-label-csv",
            stringify_path(self.config.baseline_label_csv),
            "--judge-label-csv",
            stringify_path(self.config.judge_labels_csv),
            "--baseline-merge-report-json",
            stringify_path(self.config.baseline_merge_report_json),
            "--judge-merge-report-json",
            stringify_path(self.config.judge_merge_report_json),
            "--baseline-benchmark-comparison-csv",
            stringify_path(self.config.baseline_benchmark_comparison_csv),
            "--judge-benchmark-comparison-csv",
            stringify_path(self.config.judge_benchmark_comparison_csv),
            "--baseline-gold-metrics-json",
            stringify_path(self.config.baseline_gold_metrics_json),
            "--judge-gold-metrics-json",
            stringify_path(self.config.judge_gold_metrics_json),
            "--config",
            stringify_path(self.config.evaluation_config_yaml),
            "--out-dir",
            stringify_path(self.config.decision_out_dir),
        ]

    def start_stage(self, stage: str, *, command_texts: Sequence[str], extra: Mapping[str, Any] | None = None) -> float:
        log_path = self.stage_log_path(stage)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("", encoding="utf-8")
        self.append_log(stage, f"Starting stage {stage}")
        for command_text in command_texts:
            self.append_log(stage, f"Resolved action: {command_text}")
        start_monotonic = time.monotonic()
        stage_extra = {
            "started_at": now_utc(),
            "dry_run": self.dry_run,
            "command_texts": list(command_texts),
        }
        if extra:
            stage_extra.update(dict(extra))
        self.update_stage_status(stage, status="running", log_path=log_path, extra=stage_extra)
        self.update_branch_status(
            current_stage=stage,
            stage_status="running",
            latest_artifact=stringify_path(log_path),
            extra={"active_stage_log": stringify_path(log_path)},
        )
        return start_monotonic

    def finish_stage(
        self,
        stage: str,
        *,
        stage_start_monotonic: float,
        latest_artifact: Path | None,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        status = "dry_run" if self.dry_run else "completed"
        duration_sec = round(time.monotonic() - stage_start_monotonic, 3)
        log_path = self.stage_log_path(stage)
        stage_extra = {"completed_at": now_utc(), "duration_sec": duration_sec}
        if extra:
            stage_extra.update(dict(extra))
        self.update_stage_status(
            stage,
            status=status,
            log_path=log_path,
            latest_artifact=latest_artifact,
            extra=stage_extra,
        )
        self.stage_history.append(
            {
                "stage": stage,
                "status": status,
                "completed_at": stage_extra["completed_at"],
                "duration_sec": duration_sec,
                "latest_artifact": stringify_path(latest_artifact) if latest_artifact else None,
            }
        )
        self.update_branch_status(
            current_stage=stage,
            stage_status=status,
            latest_artifact=stringify_path(latest_artifact) if latest_artifact else stringify_path(log_path),
        )
        self.append_log(stage, f"Finished stage {stage} status={status}")

    def fail_stage(self, stage: str, *, stage_start_monotonic: float, error: Exception) -> None:
        log_path = self.stage_log_path(stage)
        duration_sec = round(time.monotonic() - stage_start_monotonic, 3)
        self.append_log(stage, f"Stage failed: {error}")
        self.update_stage_status(
            stage,
            status="failed",
            log_path=log_path,
            extra={
                "failed_at": now_utc(),
                "duration_sec": duration_sec,
                "error": str(error),
            },
        )
        self.update_branch_status(
            current_stage=stage,
            stage_status="failed",
            latest_artifact=stringify_path(log_path),
            extra={"error": str(error)},
        )

    def maybe_skip_stage(self, stage: str) -> bool:
        if self.dry_run or not self.resume:
            return False
        status_path = self.config.stage_status_jsons.get(stage)
        if status_path is None or not status_path.exists():
            return False
        payload = json.loads(status_path.read_text(encoding="utf-8"))
        if str(payload.get("status")) != "completed":
            return False
        self.stage_history.append(
            {
                "stage": stage,
                "status": "skipped_completed",
                "completed_at": now_utc(),
                "duration_sec": 0.0,
                "latest_artifact": payload.get("latest_artifact"),
            }
        )
        self.update_branch_status(
            current_stage=stage,
            stage_status="skipped_completed",
            latest_artifact=str(payload.get("latest_artifact") or ""),
        )
        return True

    def run_command(
        self,
        *,
        stage: str,
        command: Sequence[str],
        line_callback: Callable[[str], None] | None = None,
    ) -> None:
        log_path = self.stage_log_path(stage)
        command_text = quote_command(command)
        self.append_log(stage, f"Executing command: {command_text}")
        if self.dry_run:
            return

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        proc = subprocess.Popen(
            list(command),
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert proc.stdout is not None
        with log_path.open("a", encoding="utf-8") as handle:
            for raw_line in proc.stdout:
                line = raw_line.rstrip("\n")
                handle.write(line + "\n")
                handle.flush()
                if line_callback is not None:
                    line_callback(line)
        return_code = proc.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, list(command))

    def run(self) -> None:
        for stage in self.selected_stages:
            if stage != "dry-run" and self.maybe_skip_stage(stage):
                continue

            command_texts = self.plan_stage_actions(stage)
            stage_start = self.start_stage(stage, command_texts=command_texts) if stage != "dry-run" else time.monotonic()
            if stage == "dry-run":
                try:
                    self.update_branch_status(
                        current_stage="dry-run",
                        stage_status="running",
                        latest_artifact=stringify_path(self.stage_log_path("dry-run")),
                    )
                    self.run_dry_run_stage(command_texts=command_texts, stage_start_monotonic=stage_start)
                except Exception as error:  # pragma: no cover - defensive orchestration path
                    self.update_branch_status(
                        current_stage=stage,
                        stage_status="failed",
                        latest_artifact=stringify_path(self.stage_log_path(stage)),
                        extra={"error": str(error)},
                    )
                    raise
                continue

            try:
                self.dispatch_stage(stage=stage, stage_start_monotonic=stage_start)
            except Exception as error:
                self.fail_stage(stage, stage_start_monotonic=stage_start, error=error)
                raise

        self.update_branch_status(
            current_stage=self.selected_stages[-1] if self.selected_stages else "",
            stage_status="completed" if not self.dry_run else "dry_run",
            latest_artifact=stringify_path(self.config.branch_status_json),
            extra={"completed_at": now_utc()},
        )

    def dispatch_stage(self, *, stage: str, stage_start_monotonic: float) -> None:
        if stage == "labeling":
            self.run_labeling_stage(stage_start_monotonic=stage_start_monotonic)
            return
        if stage == "merge":
            self.run_merge_stage(stage_start_monotonic=stage_start_monotonic)
            return
        if stage == "gold_eval":
            self.run_gold_eval_stage(stage_start_monotonic=stage_start_monotonic)
            return
        if stage == "benchmark_prep":
            self.run_benchmark_prep_stage(stage_start_monotonic=stage_start_monotonic)
            return
        if stage == "benchmark":
            self.run_benchmark_stage(stage_start_monotonic=stage_start_monotonic)
            return
        if stage == "decision":
            self.run_decision_stage(stage_start_monotonic=stage_start_monotonic)
            return
        raise ValueError(f"Unsupported stage: {stage}")

    def run_dry_run_stage(self, *, command_texts: Sequence[str], stage_start_monotonic: float) -> None:
        log_path = self.stage_log_path("dry-run")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as handle:
            handle.write(f"[{now_utc()}] Dry-run command resolution for {self.config.branch_id}\n")
            for command_text in command_texts:
                handle.write(command_text + "\n")
                print(command_text)
        self.stage_history.append(
            {
                "stage": "dry-run",
                "status": "completed" if not self.dry_run else "dry_run",
                "completed_at": now_utc(),
                "duration_sec": round(time.monotonic() - stage_start_monotonic, 3),
                "latest_artifact": stringify_path(log_path),
            }
        )
        self.update_branch_status(
            current_stage="dry-run",
            stage_status="completed" if not self.dry_run else "dry_run",
            latest_artifact=stringify_path(log_path),
        )

    def run_labeling_stage(self, *, stage_start_monotonic: float) -> None:
        ensure_paths_exist([self.config.target_csv, self.config.raw_csv])
        total_state = {"done": 0, "total": None}

        def parse_line(line: str) -> None:
            total_match = re.search(r"Target perfumes selected:\s*(\d+)", line)
            if total_match:
                total_state["total"] = int(total_match.group(1))
            progress_match = re.search(r"Progress:\s*(\d+)/(\d+)\s+processed", line)
            if progress_match:
                total_state["done"] = int(progress_match.group(1))
                total_state["total"] = int(progress_match.group(2))
            fail_match = re.search(r"\[(\d+)/(\d+)\]\s+FAIL", line)
            if fail_match:
                total_state["done"] = int(fail_match.group(1))
                total_state["total"] = int(fail_match.group(2))
            elapsed = round(time.monotonic() - stage_start_monotonic, 3)
            total = total_state["total"] or 0
            done = total_state["done"]
            percent = round((done / total) * 100.0, 3) if total else None
            eta_sec = round((elapsed / done) * (total - done), 3) if total and done > 0 and done < total else None
            self.update_stage_status(
                "labeling",
                status="running",
                log_path=self.stage_log_path("labeling"),
                extra={
                    "child_log_path": stringify_path(self.config.judge_labeling_run_log),
                    "done": done,
                    "total": total,
                    "percent": percent,
                    "elapsed_sec": elapsed,
                    "eta_sec": eta_sec,
                },
            )

        command = self.build_labeling_command()
        self.run_command(stage="labeling", command=command, line_callback=parse_line)
        integrity = {} if self.dry_run else self.validate_labeling_outputs()
        final_total = int(integrity.get("expected_rows", total_state["total"] or 0))
        self.finish_stage(
            "labeling",
            stage_start_monotonic=stage_start_monotonic,
            latest_artifact=self.config.judge_labels_csv,
            extra={
                "child_log_path": stringify_path(self.config.judge_labeling_run_log),
                "done": final_total,
                "total": final_total,
                "percent": 100.0 if final_total else None,
                "elapsed_sec": round(time.monotonic() - stage_start_monotonic, 3),
                "eta_sec": 0.0 if final_total else None,
                "integrity_checks": integrity,
            },
        )

    def validate_labeling_outputs(self) -> Dict[str, Any]:
        ensure_paths_exist(
            [
                self.config.judge_labels_jsonl,
                self.config.judge_labels_csv,
                self.config.judge_labeling_report_json,
            ]
        )

        baseline_ids: set[str] = set()
        with self.config.baseline_label_csv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                baseline_ids.add(str(row.get("perfume_id", "")).strip())
        expected_rows = len(baseline_ids)

        jsonl_ids: set[str] = set()
        jsonl_models: set[str] = set()
        jsonl_rows = 0
        with self.config.judge_labels_jsonl.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                jsonl_rows += 1
                perfume_id = str(payload.get("perfume_id", "")).strip()
                if not perfume_id:
                    raise ValueError("Judge JSONL record is missing perfume_id")
                if perfume_id in jsonl_ids:
                    raise ValueError(f"Duplicate perfume_id in judge JSONL: {perfume_id}")
                jsonl_ids.add(perfume_id)
                jsonl_models.add(str(payload.get("metadata", {}).get("model", "")).strip())

        csv_ids: set[str] = set()
        csv_rows = 0
        with self.config.judge_labels_csv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                csv_rows += 1
                perfume_id = str(row.get("perfume_id", "")).strip()
                if not perfume_id:
                    raise ValueError("Judge CSV row is missing perfume_id")
                if perfume_id in csv_ids:
                    raise ValueError(f"Duplicate perfume_id in judge CSV: {perfume_id}")
                csv_ids.add(perfume_id)

        if jsonl_rows != expected_rows or csv_rows != expected_rows:
            raise ValueError(
                f"Judge labeling row count mismatch. JSONL={jsonl_rows} CSV={csv_rows} expected={expected_rows}"
            )
        if jsonl_ids != csv_ids:
            raise ValueError("Judge JSONL and CSV perfume_id sets do not match")
        if csv_ids != baseline_ids:
            raise ValueError("Judge CSV perfume_id set does not match frozen baseline CSV perfume_id set")
        if jsonl_models != {self.config.judge_model_name}:
            raise ValueError(f"Judge JSONL metadata model mismatch: {sorted(jsonl_models)}")

        report = json.loads(self.config.judge_labeling_report_json.read_text(encoding="utf-8"))
        failed_count = int(report.get("counts", {}).get("failed", -1))
        missing_ids = list(report.get("missing_ids", []))
        failed_ids = list(report.get("failed_ids", []))
        if failed_count != 0 or failed_ids:
            raise ValueError(f"Judge labeling report indicates failures: failed_count={failed_count}")
        if missing_ids:
            raise ValueError(f"Judge labeling report indicates missing ids: {missing_ids[:10]}")

        return {
            "expected_rows": expected_rows,
            "jsonl_rows": jsonl_rows,
            "csv_rows": csv_rows,
            "failed_count": failed_count,
            "missing_ids_count": len(missing_ids),
            "metadata_models": sorted(jsonl_models),
        }

    def run_merge_stage(self, *, stage_start_monotonic: float) -> None:
        required = [self.config.unified_csv]
        if not self.dry_run:
            required.append(self.config.judge_labels_csv)
        ensure_paths_exist(required)
        command = self.build_merge_command()
        self.run_command(stage="merge", command=command)
        integrity = {} if self.dry_run else self.validate_merge_outputs()
        self.finish_stage(
            "merge",
            stage_start_monotonic=stage_start_monotonic,
            latest_artifact=self.config.judge_merged_csv,
            extra={"integrity_checks": integrity},
        )

    def validate_merge_outputs(self) -> Dict[str, Any]:
        ensure_paths_exist([self.config.judge_merge_report_json, self.config.judge_merged_csv])
        payload = json.loads(self.config.judge_merge_report_json.read_text(encoding="utf-8"))
        counts = payload.get("counts", {})
        integrity = payload.get("integrity", {})
        matched = int(counts.get("matched", -1))
        unified_only = int(counts.get("unified_only", -1))
        gemini_only = int(counts.get("gemini_only", -1))
        duplicate_total = int(integrity.get("duplicate_ids_unified_count", 0)) + int(
            integrity.get("duplicate_ids_gemini_count", 0)
        )
        hash_mismatch_count = int(integrity.get("hash_mismatch_count", -1))
        if matched != 784 or unified_only != 0 or gemini_only != 0 or duplicate_total != 0 or hash_mismatch_count != 0:
            raise ValueError(
                "Judge merge integrity check failed. "
                f"matched={matched} unified_only={unified_only} gemini_only={gemini_only} "
                f"duplicate_total={duplicate_total} hash_mismatch_count={hash_mismatch_count}"
            )
        return {
            "matched": matched,
            "unified_only": unified_only,
            "gemini_only": gemini_only,
            "duplicate_total": duplicate_total,
            "hash_mismatch_count": hash_mismatch_count,
        }

    def run_gold_eval_stage(self, *, stage_start_monotonic: float) -> None:
        required = [self.config.baseline_label_csv, self.config.gold_packet_csv, self.config.evaluation_config_yaml]
        if not self.dry_run:
            required.append(self.config.judge_labels_csv)
        ensure_paths_exist(required)
        command = self.build_gold_eval_command()
        self.run_command(stage="gold_eval", command=command)
        integrity = {} if self.dry_run else self.validate_gold_eval_outputs()
        self.finish_stage(
            "gold_eval",
            stage_start_monotonic=stage_start_monotonic,
            latest_artifact=self.config.judge_gold_metrics_json,
            extra={"integrity_checks": integrity},
        )

    def validate_gold_eval_outputs(self) -> Dict[str, Any]:
        required_paths = [
            self.config.baseline_gold_metrics_json,
            self.config.gold_eval_out_dir / "baseline_vs_gold_per_label.csv",
            self.config.judge_gold_metrics_json,
            self.config.gold_eval_out_dir / "judge_vs_gold_per_label.csv",
            self.config.gold_eval_out_dir / "baseline_vs_judge_gold_comparison.csv",
            self.config.gold_eval_out_dir / "gold_packet_match_report.json",
        ]
        ensure_paths_exist(required_paths)
        baseline_payload = json.loads(self.config.baseline_gold_metrics_json.read_text(encoding="utf-8"))
        judge_payload = json.loads(self.config.judge_gold_metrics_json.read_text(encoding="utf-8"))
        matched_rows = int(baseline_payload.get("matched_gold_rows", -1))
        judge_rows = int(judge_payload.get("matched_gold_rows", -1))
        if matched_rows != judge_rows or matched_rows <= 0:
            raise ValueError("Gold evaluation outputs do not agree on matched gold row count")
        return {
            "matched_gold_rows": matched_rows,
            "baseline_primary_macro_f1": float(baseline_payload["primary_decision_metric"]["value"]),
            "judge_primary_macro_f1": float(judge_payload["primary_decision_metric"]["value"]),
        }

    def run_benchmark_prep_stage(self, *, stage_start_monotonic: float) -> None:
        required = [self.config.raw_csv, self.config.baseline_fold_manifest_csv]
        if not self.dry_run:
            required.append(self.config.judge_labels_csv)
        ensure_paths_exist(required)
        copied_manifest = self.config.benchmark_root / "tables" / "fold_assignments.csv"
        if not self.dry_run:
            copied_manifest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.config.baseline_fold_manifest_csv, copied_manifest)
        else:
            self.append_log(
                "benchmark_prep",
                f"Dry-run copy: {stringify_path(self.config.baseline_fold_manifest_csv)} -> {stringify_path(copied_manifest)}",
            )
        command = self.build_benchmark_prep_command()
        self.run_command(stage="benchmark_prep", command=command)
        integrity = {} if self.dry_run else self.validate_benchmark_prep_outputs()
        self.finish_stage(
            "benchmark_prep",
            stage_start_monotonic=stage_start_monotonic,
            latest_artifact=self.config.benchmark_dataset_summary_json,
            extra={"integrity_checks": integrity},
        )

    def validate_benchmark_prep_outputs(self) -> Dict[str, Any]:
        copied_manifest = self.config.benchmark_root / "tables" / "fold_assignments.csv"
        ensure_paths_exist([copied_manifest, self.config.benchmark_dataset_summary_json, self.config.benchmark_dataset_jsonl])
        source_hash = sha256_file(self.config.baseline_fold_manifest_csv)
        copied_hash = sha256_file(copied_manifest)
        if source_hash != copied_hash:
            raise ValueError("Copied fold manifest does not exactly match the frozen baseline fold manifest")
        summary = json.loads(self.config.benchmark_dataset_summary_json.read_text(encoding="utf-8"))
        n_perfumes = int(summary.get("n_perfumes", -1))
        if n_perfumes != 784:
            raise ValueError(f"Judge deep-text dataset summary must report n_perfumes=784, found {n_perfumes}")
        return {
            "copied_fold_manifest_sha256": copied_hash,
            "n_perfumes": n_perfumes,
        }

    def run_benchmark_stage(self, *, stage_start_monotonic: float) -> None:
        required = [self.config.evaluation_config_yaml]
        if not self.dry_run:
            required.extend([self.config.judge_merged_csv, self.config.benchmark_dataset_jsonl])
        ensure_paths_exist(required)
        commands = self.build_benchmark_commands()
        baseline_snapshot = snapshot_tree(self.config.baseline_benchmark_comparison_csv.parent.parent)
        total_models = len(commands)
        current_state: Dict[str, Any] = {
            "models_done": 0,
            "models_total": total_models,
            "current_model": None,
            "current_model_log_path": None,
        }

        def update_progress(*, current_model: str | None = None, fold: int | None = None, epoch: int | None = None) -> None:
            payload = {
                "models_done": current_state["models_done"],
                "models_total": total_models,
                "current_model": current_model if current_model is not None else current_state["current_model"],
                "elapsed_sec": round(time.monotonic() - stage_start_monotonic, 3),
                "current_model_log_path": current_state["current_model_log_path"],
            }
            if fold is not None:
                payload["current_fold"] = fold
            if epoch is not None:
                payload["current_epoch"] = epoch
            self.update_stage_status("benchmark", status="running", log_path=self.stage_log_path("benchmark"), extra=payload)

        fold_pattern = re.compile(r"\bfold=(\d+)\b", flags=re.IGNORECASE)
        epoch_pattern = re.compile(r"\bepoch=(\d+)\b", flags=re.IGNORECASE)

        for model_index, (model_id, command) in enumerate(commands.items(), start=1):
            current_state["current_model"] = model_id
            current_state["current_model_log_path"] = stringify_path(self.config.benchmark_root / "logs" / f"{model_id}.log")
            update_progress(current_model=model_id)

            def parse_line(line: str) -> None:
                fold_match = fold_pattern.search(line)
                epoch_match = epoch_pattern.search(line)
                update_progress(
                    current_model=model_id,
                    fold=int(fold_match.group(1)) if fold_match else None,
                    epoch=int(epoch_match.group(1)) if epoch_match else None,
                )

            self.run_command(stage="benchmark", command=command, line_callback=parse_line)
            current_state["models_done"] = model_index
            update_progress(current_model=model_id)

        integrity = {} if self.dry_run else self.validate_benchmark_outputs(baseline_snapshot=baseline_snapshot)
        self.finish_stage(
            "benchmark",
            stage_start_monotonic=stage_start_monotonic,
            latest_artifact=self.config.judge_benchmark_comparison_csv,
            extra={
                "models_done": total_models,
                "models_total": total_models,
                "current_model": None,
                "elapsed_sec": round(time.monotonic() - stage_start_monotonic, 3),
                "integrity_checks": integrity,
            },
        )

    def validate_benchmark_outputs(self, *, baseline_snapshot: Mapping[str, Mapping[str, int]]) -> Dict[str, Any]:
        ensure_paths_exist([self.config.judge_benchmark_comparison_csv])
        table = pd.read_csv(self.config.judge_benchmark_comparison_csv)
        required_models = set(self.config.benchmark_models)
        filtered = table.loc[table["model_id"].isin(required_models)].copy()
        if len(filtered) != len(required_models):
            raise ValueError(
                f"Judge benchmark comparison table must contain {len(required_models)} model rows, found {len(filtered)}"
            )
        filtered["threshold"] = pd.to_numeric(filtered["threshold"], errors="coerce")
        filtered["row_complete"] = filtered["row_complete"].astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
        if not filtered["row_complete"].all():
            incomplete = filtered.loc[~filtered["row_complete"], "model_id"].astype(str).tolist()
            raise ValueError(f"Judge benchmark comparison table has incomplete rows: {incomplete}")
        if sorted(filtered["threshold"].dropna().round(6).unique().tolist()) != [0.3]:
            raise ValueError("Judge benchmark comparison table must contain threshold=0.3 for all completed rows")

        after_snapshot = snapshot_tree(self.config.baseline_benchmark_comparison_csv.parent.parent)
        if dict(baseline_snapshot) != dict(after_snapshot):
            raise ValueError("Frozen baseline benchmark output root changed during judge benchmark stage")

        return {
            "complete_rows": int(filtered["row_complete"].sum()),
            "thresholds": sorted(filtered["threshold"].round(6).unique().tolist()),
            "baseline_output_root_unchanged": True,
        }

    def run_decision_stage(self, *, stage_start_monotonic: float) -> None:
        required = [
            self.config.baseline_label_csv,
            self.config.baseline_merge_report_json,
            self.config.baseline_benchmark_comparison_csv,
            self.config.evaluation_config_yaml,
        ]
        if not self.dry_run:
            required.extend(
                [
                    self.config.judge_labels_csv,
                    self.config.judge_merge_report_json,
                    self.config.judge_benchmark_comparison_csv,
                    self.config.baseline_gold_metrics_json,
                    self.config.judge_gold_metrics_json,
                ]
            )
        ensure_paths_exist(required)
        command = self.build_decision_command()
        self.run_command(stage="decision", command=command)
        integrity = {} if self.dry_run else self.validate_decision_outputs()
        self.finish_stage(
            "decision",
            stage_start_monotonic=stage_start_monotonic,
            latest_artifact=self.config.decision_summary_json,
            extra={"integrity_checks": integrity},
        )

    def validate_decision_outputs(self) -> Dict[str, Any]:
        required_paths = [
            self.config.decision_out_dir / "label_prevalence_delta.csv",
            self.config.decision_out_dir / "branch_label_drift_summary.json",
            self.config.decision_out_dir / "gold_side_by_side.csv",
            self.config.decision_out_dir / "benchmark_side_by_side.csv",
            self.config.decision_out_dir / "branch_decision_summary.md",
            self.config.decision_summary_json,
        ]
        ensure_paths_exist(required_paths)
        payload = json.loads(self.config.decision_summary_json.read_text(encoding="utf-8"))
        decision = str(payload.get("decision", "")).strip()
        if decision not in {"go", "no_go", "inconclusive"}:
            raise ValueError(f"Decision summary contains invalid outcome: {decision!r}")
        return {"decision": decision}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "gemini_judge" / "judge_g25pro_v1.yaml",
        help="Judge branch config YAML.",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        help="Optional stage subset. Accepts canonical stage names or comma-separated values.",
    )
    parser.add_argument("--from-stage", choices=CANONICAL_STAGES, default=None)
    parser.add_argument("--to-stage", choices=CANONICAL_STAGES, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Resolve commands and update status JSON without executing them.")
    parser.add_argument("--resume", action="store_true", help="Skip already completed stages based on status JSON files.")
    parser.add_argument(
        "--benchmark-device",
        choices=["auto", "cpu", "cuda", "mps"],
        default=None,
        help="Device override for BERT and BiLSTM judge benchmark runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_branch_config(args.config)
    selected_stages = select_stages(
        config=config,
        only=args.only,
        from_stage=args.from_stage,
        to_stage=args.to_stage,
    )
    runner = BranchRunner(
        config=config,
        selected_stages=selected_stages,
        dry_run=bool(args.dry_run),
        resume=bool(args.resume),
        benchmark_device=args.benchmark_device or config.benchmark_device_default,
    )
    runner.run()


if __name__ == "__main__":
    main()

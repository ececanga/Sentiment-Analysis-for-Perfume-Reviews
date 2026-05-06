from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

import pandas as pd
import yaml

from src.training.generate_final_benchmark_figures import (
    METRIC_SPECS,
    configure_plot_style,
    export_final_table,
    load_final_model_comparison,
    save_grouped_metric_figure,
    save_single_metric_figure,
)
from src.training.generate_threshold_sweep_summary import (
    DEFAULT_SUMMARY_ROOT,
    generate_threshold_sweep_summary,
    threshold_to_tag,
)
from src.training.model_benchmark_runner import load_config
from src.training.run_bert_benchmark import load_bert_settings, run_bert_benchmark
from src.training.run_bilstm_attention_benchmark import run_bilstm_attention_benchmark
from src.training.run_logistic_regression_benchmark import run_logistic_regression_benchmark
from src.training.run_random_forest_benchmark import run_random_forest_benchmark
from src.training.run_xgboost_benchmark import run_xgboost_benchmark
from src.utils.paths import PROJECT_ROOT

OFFICIAL_MODEL_ORDER = [
    "logistic_regression",
    "random_forest",
    "xgboost",
    "bert",
    "bilstm_attention",
]
DEEP_MODEL_IDS = {"bert", "bilstm_attention"}
DEFAULT_THRESHOLDS = [0.2, 0.3, 0.5, 0.7]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the official model benchmarks across multiple fixed baseline thresholds."
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=PROJECT_ROOT / "config" / "evaluation_config.yaml",
        help="Base evaluation config used to generate threshold-specific config copies.",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "analysis_out" / "gemini_unified_merged.csv",
        help="Canonical classical-model benchmark dataset CSV.",
    )
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "model_benchmark" / "datasets" / "perfume_text_dataset_v1.jsonl",
        help="Canonical deep-model benchmark dataset JSONL.",
    )
    parser.add_argument(
        "--summary-root",
        type=Path,
        default=DEFAULT_SUMMARY_ROOT,
        help="Root directory for sweep-level configs, manifests, tables, and plots.",
    )
    parser.add_argument(
        "--deep-device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="cpu",
        help="Device preference for BERT and BiLSTM threshold runs.",
    )
    parser.add_argument(
        "--threshold",
        action="append",
        type=float,
        help="Fixed baseline threshold to run. Repeat for multiple thresholds.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Optional subset of official model ids to run.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Allow existing threshold roots and skip model runs that are already complete.",
    )
    return parser.parse_args()


def stringify_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def normalize_models(models: Sequence[str] | None) -> List[str]:
    if models is None:
        return list(OFFICIAL_MODEL_ORDER)

    requested = [str(model_id) for model_id in models]
    invalid = sorted(set(requested) - set(OFFICIAL_MODEL_ORDER))
    if invalid:
        raise ValueError(f"Unknown model ids: {', '.join(invalid)}")

    deduped: List[str] = []
    for model_id in requested:
        if model_id not in deduped:
            deduped.append(model_id)
    return [model_id for model_id in OFFICIAL_MODEL_ORDER if model_id in deduped]


def normalize_thresholds(thresholds: Sequence[float] | None) -> List[float]:
    raw_values = list(DEFAULT_THRESHOLDS if thresholds is None else thresholds)
    if not raw_values:
        raise ValueError("At least one threshold must be provided.")

    normalized: List[float] = []
    seen_tags: set[str] = set()
    for value in raw_values:
        threshold = float(value)
        if not 0.0 < threshold < 1.0:
            raise ValueError(f"Threshold must be in (0, 1): {threshold}")
        tag = threshold_to_tag(threshold)
        if tag in seen_tags:
            raise ValueError(f"Threshold tag collision detected for {threshold}.")
        seen_tags.add(tag)
        normalized.append(threshold)
    return normalized


def threshold_output_root(summary_root: Path, threshold: float) -> Path:
    return summary_root.parent / f"model_benchmark_{threshold_to_tag(threshold)}"


def threshold_config_path(summary_root: Path, threshold: float) -> Path:
    return summary_root / "configs" / f"evaluation_config_{threshold_to_tag(threshold)}.yaml"


def write_threshold_config(*, base_config_path: Path, destination_path: Path, threshold: float) -> Path:
    config = load_config(base_config_path)
    evaluation_cfg = config.setdefault("evaluation", {})
    evaluation_cfg["probability_threshold"] = float(threshold)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    destination_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return destination_path


def ensure_required_inputs(
    *,
    base_config_path: Path,
    input_csv: Path,
    input_jsonl: Path,
    models: Sequence[str],
) -> None:
    if not base_config_path.exists():
        raise FileNotFoundError(f"Missing base config: {base_config_path}")
    if not input_csv.exists():
        raise FileNotFoundError(f"Missing input CSV: {input_csv}")
    if any(model_id in DEEP_MODEL_IDS for model_id in models) and not input_jsonl.exists():
        raise FileNotFoundError(f"Missing input JSONL: {input_jsonl}")


def assert_safe_threshold_roots(*, summary_root: Path, thresholds: Sequence[float], resume: bool) -> None:
    if resume:
        return
    for threshold in thresholds:
        out_root = threshold_output_root(summary_root, threshold)
        if out_root.exists() and any(out_root.iterdir()):
            raise ValueError(f"Threshold output root already exists and is non-empty: {out_root}")


def is_complete_model_run(out_root: Path, model_id: str) -> bool:
    metrics_path = out_root / "metrics" / f"{model_id}.json"
    per_label_path = out_root / "metrics" / f"{model_id}_per_label.csv"
    predictions_path = out_root / "predictions" / f"{model_id}_oof.csv"
    comparison_table_path = out_root / "tables" / "comparison_table.csv"
    required_paths = [metrics_path, per_label_path, predictions_path, comparison_table_path]
    if not all(path.exists() for path in required_paths):
        return False

    comparison_df = pd.read_csv(comparison_table_path)
    row = comparison_df.loc[comparison_df["model_id"] == model_id]
    if row.empty:
        return False
    return str(row.iloc[0]["row_complete"]).lower() in {"true", "1"}


def maybe_assert_local_bert_checkpoint(base_config_path: Path, models: Sequence[str]) -> None:
    if "bert" not in models:
        return
    settings = load_bert_settings(load_config(base_config_path))
    if not settings.local_files_only:
        return

    from transformers import AutoTokenizer

    try:
        AutoTokenizer.from_pretrained(settings.checkpoint, local_files_only=True)
    except Exception as exc:
        raise RuntimeError(
            f"BERT local checkpoint is not available for local_files_only loading: {settings.checkpoint}"
        ) from exc


def _run_logistic(*, input_csv: Path, out_root: Path, config_path: Path, deep_device: str) -> Dict[str, Any]:
    return run_logistic_regression_benchmark(input_csv=input_csv, out_root=out_root, config_path=config_path)


def _run_random_forest(*, input_csv: Path, out_root: Path, config_path: Path, deep_device: str) -> Dict[str, Any]:
    return run_random_forest_benchmark(input_csv=input_csv, out_root=out_root, config_path=config_path)


def _run_xgboost(*, input_csv: Path, out_root: Path, config_path: Path, deep_device: str) -> Dict[str, Any]:
    return run_xgboost_benchmark(input_csv=input_csv, out_root=out_root, config_path=config_path)


def _run_bert(*, input_jsonl: Path, out_root: Path, config_path: Path, deep_device: str) -> Dict[str, Any]:
    return run_bert_benchmark(
        input_jsonl=input_jsonl,
        out_root=out_root,
        config_path=config_path,
        device_preference=deep_device,
    )


def _run_bilstm(*, input_jsonl: Path, out_root: Path, config_path: Path, deep_device: str) -> Dict[str, Any]:
    return run_bilstm_attention_benchmark(
        input_jsonl=input_jsonl,
        out_root=out_root,
        config_path=config_path,
        device_preference=deep_device,
    )


MODEL_RUNNERS: Dict[str, Callable[..., Dict[str, Any]]] = {
    "logistic_regression": _run_logistic,
    "random_forest": _run_random_forest,
    "xgboost": _run_xgboost,
    "bert": _run_bert,
    "bilstm_attention": _run_bilstm,
}


def build_model_command(
    *,
    model_id: str,
    config_path: Path,
    input_csv: Path,
    input_jsonl: Path,
    out_root: Path,
    deep_device: str,
) -> str:
    module_name = f"src.training.run_{model_id}_benchmark"
    if model_id in DEEP_MODEL_IDS:
        return (
            f".venv/bin/python -m {module_name} "
            f"--config {stringify_path(config_path)} "
            f"--input-jsonl {stringify_path(input_jsonl)} "
            f"--out-root {stringify_path(out_root)} "
            f"--device {deep_device}"
        )
    return (
        f".venv/bin/python -m {module_name} "
        f"--config {stringify_path(config_path)} "
        f"--input-csv {stringify_path(input_csv)} "
        f"--out-root {stringify_path(out_root)}"
    )


def build_summary_command(*, summary_root: Path, threshold_roots: Sequence[Path], models: Sequence[str]) -> str:
    parts = [
        ".venv/bin/python -m src.training.generate_threshold_sweep_summary",
        f"--summary-root {stringify_path(summary_root)}",
    ]
    for threshold_root in threshold_roots:
        parts.append(f"--threshold-root {stringify_path(threshold_root)}")
    if models and list(models) != OFFICIAL_MODEL_ORDER:
        parts.append("--models " + " ".join(models))
    return " ".join(parts)


def write_command_manifest(
    *,
    summary_root: Path,
    thresholds: Sequence[float],
    models: Sequence[str],
    input_csv: Path,
    input_jsonl: Path,
    deep_device: str,
) -> Path:
    commands_dir = summary_root / "commands"
    commands_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = commands_dir / "fixed_threshold_sweep_commands.txt"
    lines: List[str] = []
    for threshold in thresholds:
        config_path = threshold_config_path(summary_root, threshold)
        out_root = threshold_output_root(summary_root, threshold)
        lines.append(f"# threshold={threshold}")
        for model_id in models:
            lines.append(
                build_model_command(
                    model_id=model_id,
                    config_path=config_path,
                    input_csv=input_csv,
                    input_jsonl=input_jsonl,
                    out_root=out_root,
                    deep_device=deep_device,
                )
            )
        lines.append("")
    lines.append(build_summary_command(summary_root=summary_root, threshold_roots=[threshold_output_root(summary_root, value) for value in thresholds], models=models))
    manifest_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return manifest_path


def generate_threshold_root_figures(out_root: Path) -> Dict[str, Any]:
    configure_plot_style()
    comparison_df = load_final_model_comparison(out_root / "tables" / "comparison_table.csv")
    figures_dir = out_root / "figures"
    tables_dir = out_root / "tables"
    export_final_table(comparison_df, path=tables_dir / "final_model_comparison_for_thesis.csv")
    for metric, metric_label, filename in METRIC_SPECS:
        save_single_metric_figure(
            comparison_df,
            metric=metric,
            metric_label=metric_label,
            path=figures_dir / filename,
        )
    save_grouped_metric_figure(comparison_df, path=figures_dir / "all_models_grouped_metric_comparison.png")
    return {
        "comparison_table_path": str(out_root / "tables" / "comparison_table.csv"),
        "final_table_path": str(tables_dir / "final_model_comparison_for_thesis.csv"),
    }


def run_fixed_threshold_sweep(
    *,
    base_config_path: Path,
    input_csv: Path,
    input_jsonl: Path,
    summary_root: Path,
    deep_device: str = "cpu",
    thresholds: Sequence[float] | None = None,
    models: Sequence[str] | None = None,
    resume: bool = False,
) -> Dict[str, Any]:
    ordered_models = normalize_models(models)
    ordered_thresholds = normalize_thresholds(thresholds)

    ensure_required_inputs(
        base_config_path=base_config_path,
        input_csv=input_csv,
        input_jsonl=input_jsonl,
        models=ordered_models,
    )
    assert_safe_threshold_roots(summary_root=summary_root, thresholds=ordered_thresholds, resume=resume)
    maybe_assert_local_bert_checkpoint(base_config_path, ordered_models)

    summary_root.mkdir(parents=True, exist_ok=True)
    logs_dir = summary_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    command_manifest_path = write_command_manifest(
        summary_root=summary_root,
        thresholds=ordered_thresholds,
        models=ordered_models,
        input_csv=input_csv,
        input_jsonl=input_jsonl,
        deep_device=deep_device,
    )

    manifest: Dict[str, Any] = {
        "summary_root": str(summary_root),
        "base_config_path": str(base_config_path),
        "input_csv": str(input_csv),
        "input_jsonl": str(input_jsonl),
        "deep_device": deep_device,
        "thresholds": ordered_thresholds,
        "models": ordered_models,
        "resume": resume,
        "command_manifest_path": str(command_manifest_path),
        "threshold_runs": [],
    }

    for threshold in ordered_thresholds:
        out_root = threshold_output_root(summary_root, threshold)
        config_path = write_threshold_config(
            base_config_path=base_config_path,
            destination_path=threshold_config_path(summary_root, threshold),
            threshold=threshold,
        )
        threshold_record: Dict[str, Any] = {
            "threshold": threshold,
            "threshold_tag": threshold_to_tag(threshold),
            "out_root": str(out_root),
            "config_path": str(config_path),
            "models": [],
        }

        for model_id in ordered_models:
            if resume and is_complete_model_run(out_root, model_id):
                threshold_record["models"].append({"model_id": model_id, "status": "skipped_resume"})
                continue

            if model_id in DEEP_MODEL_IDS:
                result = MODEL_RUNNERS[model_id](
                    input_jsonl=input_jsonl,
                    out_root=out_root,
                    config_path=config_path,
                    deep_device=deep_device,
                )
            else:
                result = MODEL_RUNNERS[model_id](
                    input_csv=input_csv,
                    out_root=out_root,
                    config_path=config_path,
                    deep_device=deep_device,
                )

            threshold_record["models"].append(
                {
                    "model_id": model_id,
                    "status": "completed",
                    "metrics_path": str(result["metrics_path"]),
                    "comparison_table_path": str(result["comparison_table_path"]),
                }
            )

        if ordered_models == OFFICIAL_MODEL_ORDER:
            figure_outputs = generate_threshold_root_figures(out_root)
            threshold_record["final_figures"] = {"status": "generated", **figure_outputs}
        else:
            threshold_record["final_figures"] = {
                "status": "skipped",
                "reason": "subset_run_requires_only_cross_threshold_summary",
            }

        manifest["threshold_runs"].append(threshold_record)

    threshold_roots = [threshold_output_root(summary_root, threshold) for threshold in ordered_thresholds]
    summary_outputs = generate_threshold_sweep_summary(
        summary_root=summary_root,
        threshold_roots=threshold_roots,
        models=ordered_models,
    )
    manifest["summary_outputs"] = summary_outputs

    manifest_path = logs_dir / "fixed_threshold_sweep_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def main() -> None:
    args = parse_args()
    manifest = run_fixed_threshold_sweep(
        base_config_path=args.base_config,
        input_csv=args.input_csv,
        input_jsonl=args.input_jsonl,
        summary_root=args.summary_root,
        deep_device=args.deep_device,
        thresholds=args.threshold,
        models=args.models,
        resume=args.resume,
    )
    print("Fixed-threshold sweep complete")
    print(f"- Command manifest: {manifest['command_manifest_path']}")
    print(f"- Sweep manifest: {manifest['manifest_path']}")
    for threshold_run in manifest["threshold_runs"]:
        print(
            f"- Threshold {threshold_run['threshold']}: "
            f"{', '.join(entry['model_id'] + '=' + entry['status'] for entry in threshold_run['models'])}"
        )
    summary_outputs = manifest["summary_outputs"]
    print(f"- Threshold summary table: {summary_outputs['wide_table_path']}")
    print(f"- Threshold summary figure: {summary_outputs['line_plot_path']}")


if __name__ == "__main__":
    main()

"""Run the leakage-repaired engineered-feature LR without class weighting.

This is a sensitivity analysis, not a replacement for the accepted
class-weighted Logistic Regression benchmark. It reuses the frozen feature
table and the exact same source-grouped outer and inner manifests while
writing to an isolated experiment root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.training.model_benchmark_runner import BenchmarkModelSpec, run_classical_model_benchmark
from src.utils.paths import PROJECT_ROOT


DEFAULT_BASE_CONFIG = (
    PROJECT_ROOT / "config" / "evaluation_config_gemini_browser_og784_source_grouped_v1.yaml"
)
DEFAULT_INPUT_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "gemini_browser_og784"
    / "analysis_out"
    / "gemini_browser_og784_feature_labels.csv"
)
DEFAULT_EXPERIMENT_ROOT = (
    PROJECT_ROOT / "outputs" / "logistic_regression_unweighted_source_grouped_v1"
)
ACCEPTED_METRICS = (
    PROJECT_ROOT
    / "outputs"
    / "gemini_browser_og784_source_grouped_v1"
    / "model_benchmark"
    / "metrics"
    / "logistic_regression.json"
)


MODEL_SPEC = BenchmarkModelSpec(
    model_id="logistic_regression",
    model_family="Logistic Regression (unweighted sensitivity)",
    build_model=lambda seed, _y_train: Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight=None,
                    solver="liblinear",
                    random_state=seed,
                ),
            ),
        ]
    ),
    model_params={
        "imputer_strategy": "median",
        "scaler": "standard",
        "classifier": {
            "type": "LogisticRegression",
            "max_iter": 2000,
            "class_weight": None,
            "solver": "liblinear",
        },
        "analysis_role": "unweighted_sensitivity_not_primary_benchmark",
    },
    package_names=(),
)


def build_sensitivity_config(*, base_config: Path, experiment_root: Path) -> Path:
    """Create a traceable derived config with isolated output paths."""

    payload = yaml.safe_load(base_config.read_text(encoding="utf-8"))
    benchmark_root = experiment_root / "model_benchmark"
    experiment_rel = experiment_root.relative_to(PROJECT_ROOT).as_posix()
    benchmark_rel = benchmark_root.relative_to(PROJECT_ROOT).as_posix()

    payload["benchmark"]["name"] = "logistic_regression_unweighted_source_grouped_v1"
    payload["benchmark"]["status"] = "post_hoc_unweighted_sensitivity"
    payload["benchmark"]["interpretation"] = "sensitivity_only_not_primary_benchmark"
    payload["benchmark"]["experiment_root"] = experiment_rel
    payload["benchmark"]["benchmark_root"] = benchmark_rel
    payload["artifacts"]["experiment_root"] = experiment_rel
    payload["artifacts"]["root_dir"] = benchmark_rel

    required = payload["artifacts"].get("required_per_model", {})
    for key, value in required.items():
        required[key] = str(value).replace(
            "outputs/gemini_browser_og784_source_grouped_v1/model_benchmark",
            benchmark_rel,
        )
    wide = payload["artifacts"].get("benchmark_wide", {})
    if "comparison_table_csv" in wide:
        wide["comparison_table_csv"] = f"{benchmark_rel}/tables/comparison_table.csv"

    payload.setdefault("sensitivity_analysis", {})["class_weight"] = None
    payload["sensitivity_analysis"]["comparison_against"] = str(
        ACCEPTED_METRICS.relative_to(PROJECT_ROOT)
    )
    payload["sensitivity_analysis"]["shared_outer_and_inner_manifests"] = True

    config_path = experiment_root / "evaluation_config_unweighted_lr.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return config_path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_comparison_report(*, result: Dict[str, Any], experiment_root: Path) -> tuple[Path, Path]:
    accepted = json.loads(ACCEPTED_METRICS.read_text(encoding="utf-8"))
    sensitivity = json.loads(Path(result["metrics_path"]).read_text(encoding="utf-8"))

    accepted_split = accepted["split_provenance"]
    sensitivity_split = sensitivity["split_provenance"]
    for key in ["outer_manifest_sha256", "inner_manifest_sha256", "source_audit_sha256"]:
        if accepted_split[key] != sensitivity_split[key]:
            raise ValueError(f"Sensitivity run does not share accepted split provenance: {key}")

    pred_df = pd.read_csv(result["predictions_path"])
    if len(pred_df) != 784 or pred_df["perfume_id"].nunique() != 784:
        raise ValueError("Sensitivity OOF predictions must contain exactly 784 unique perfumes")
    if pred_df.groupby("source_group_id")["fold_id"].nunique().max() != 1:
        raise ValueError("A source group crosses sensitivity OOF folds")

    metrics = ["macro_f1", "macro_roc_auc", "macro_pr_auc"]
    rows = []
    for metric in metrics:
        weighted = float(accepted[metric])
        unweighted = float(sensitivity[metric])
        rows.append((metric, weighted, unweighted, unweighted - weighted))

    lines = [
        "# Unweighted Engineered-Feature Logistic Regression Sensitivity",
        "",
        "The accepted class-weighted LR and this unweighted LR use the same frozen 784-perfume feature table, source-grouped outer folds, shared inner-validation manifest, seed, features, and threshold. The only intended model change is `class_weight=\"balanced\"` to `class_weight=None`.",
        "",
        "| Metric | Class-weighted LR | Unweighted LR | Unweighted minus weighted |",
        "| --- | ---: | ---: | ---: |",
    ]
    display = {
        "macro_f1": "Macro-F1",
        "macro_roc_auc": "Macro ROC-AUC",
        "macro_pr_auc": "Macro average precision",
    }
    for metric, weighted, unweighted, delta in rows:
        lines.append(f"| {display[metric]} | {weighted:.4f} | {unweighted:.4f} | {delta:+.4f} |")
    lines.extend(
        [
            "",
            "This is a post-hoc sensitivity comparison. It does not replace the prespecified class-weighted LR row in the primary five-model benchmark.",
            "",
        ]
    )
    report_path = experiment_root / "SENSITIVITY_REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    validation_path = experiment_root / "validation_report.json"
    validation_path.write_text(
        json.dumps(
            {
                "status": "passed",
                "rows": int(len(pred_df)),
                "unique_perfumes": int(pred_df["perfume_id"].nunique()),
                "source_groups": int(pred_df["source_group_id"].nunique()),
                "cross_fold_source_groups": 0,
                "shared_split_provenance_verified": True,
                "outer_manifest_sha256": sensitivity_split["outer_manifest_sha256"],
                "inner_manifest_sha256": sensitivity_split["inner_manifest_sha256"],
                "source_audit_sha256": sensitivity_split["source_audit_sha256"],
                "class_weight": None,
                "threshold": float(sensitivity["threshold"]),
                "metrics": {metric: float(sensitivity[metric]) for metric in metrics},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return report_path, validation_path


def write_checksum_manifest(experiment_root: Path) -> Path:
    manifest_path = experiment_root / "SHA256SUMS"
    paths = sorted(
        path
        for path in experiment_root.rglob("*")
        if path.is_file() and path != manifest_path
    )
    manifest_path.write_text(
        "".join(
            f"{sha256_file(path)}  ./{path.relative_to(experiment_root).as_posix()}\n"
            for path in paths
        ),
        encoding="utf-8",
    )
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_root = args.experiment_root.resolve()
    if experiment_root == (
        PROJECT_ROOT / "outputs" / "gemini_browser_og784_source_grouped_v1"
    ).resolve():
        raise ValueError("The unweighted sensitivity may not overwrite the accepted benchmark root")

    config_path = build_sensitivity_config(
        base_config=args.base_config.resolve(),
        experiment_root=experiment_root,
    )
    result = run_classical_model_benchmark(
        input_csv=args.input_csv.resolve(),
        out_root=experiment_root / "model_benchmark",
        config_path=config_path,
        model_spec=MODEL_SPEC,
    )
    report_path, validation_path = write_comparison_report(
        result=result,
        experiment_root=experiment_root,
    )
    checksum_path = write_checksum_manifest(experiment_root)
    metrics = result["macro_metrics"]
    print("Unweighted Logistic Regression sensitivity complete")
    print(f"- OOF predictions: {result['predictions_path']}")
    print(f"- Metrics: {result['metrics_path']}")
    print(f"- Comparison report: {report_path}")
    print(f"- Validation: {validation_path}")
    print(f"- Checksums: {checksum_path}")
    print(
        f"- Macro-F1={metrics['macro_f1']:.4f}, "
        f"macro ROC-AUC={metrics['macro_roc_auc']:.4f}, "
        f"macro average precision={metrics['macro_pr_auc']:.4f}"
    )


if __name__ == "__main__":
    main()

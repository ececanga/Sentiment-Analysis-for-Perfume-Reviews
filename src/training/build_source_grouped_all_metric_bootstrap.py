# Date Created: 2026-07-15

"""Build paired source-component bootstrap intervals for all headline metrics.

The accepted benchmark's original macro-F1 bootstrap remains immutable.  This
supplement reuses the same frozen OOF files and source-component weighting, but
adds model-level and paired-difference intervals for macro-F1, macro ROC-AUC,
and macro average precision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from src.training.generate_source_grouped_diagnostics import (
    MODEL_IDS,
    OFFICIAL_LABELS,
    SOURCE_GROUP_COLUMN,
    ValidatedInputs,
    load_and_validate_inputs,
    sha256_file,
)
from src.utils.paths import PROJECT_ROOT


DEFAULT_FEATURE_LABEL_CSV = (
    PROJECT_ROOT
    / "outputs/gemini_browser_og784/analysis_out/gemini_browser_og784_feature_labels.csv"
)
DEFAULT_OUTER_MANIFEST_CSV = (
    PROJECT_ROOT
    / "outputs/gemini_browser_og784_source_grouped_v1/model_benchmark/tables/fold_assignments.csv"
)
DEFAULT_INNER_MANIFEST_CSV = (
    PROJECT_ROOT
    / "outputs/gemini_browser_og784_source_grouped_v1/model_benchmark/tables/inner_validation_folds.csv"
)
DEFAULT_BENCHMARK_ROOT = (
    PROJECT_ROOT / "outputs/gemini_browser_og784_source_grouped_v1/model_benchmark"
)
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs/source_grouped_all_metric_bootstrap_v1"

OUTPUT_FILENAMES = {
    "model_intervals": "model_metric_bootstrap_intervals.csv",
    "paired_differences": "paired_model_metric_bootstrap_differences.csv",
    "report": "REPORT.json",
    "checksums": "SHA256SUMS",
}
METRIC_ORDER = (
    "component_equal_macro_f1",
    "component_equal_macro_roc_auc",
    "component_equal_macro_average_precision",
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _component_layout(inputs: ValidatedInputs) -> tuple[list[str], np.ndarray, np.ndarray]:
    component_ids = sorted(inputs.outer[SOURCE_GROUP_COLUMN].unique().tolist())
    component_index = {group_id: index for index, group_id in enumerate(component_ids)}
    group_sizes = inputs.outer.groupby(SOURCE_GROUP_COLUMN)["perfume_id"].size().to_dict()
    reference = inputs.oof_by_model[MODEL_IDS[0]]
    row_component_index = reference[SOURCE_GROUP_COLUMN].map(component_index).to_numpy(int)
    inverse_group_size = reference[SOURCE_GROUP_COLUMN].map(
        {group_id: 1.0 / int(size) for group_id, size in group_sizes.items()}
    ).to_numpy(float)
    return component_ids, row_component_index, inverse_group_size


def _draw_counts(*, n_components: int, reps: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, n_components, size=(reps, n_components))
    counts = np.zeros((reps, n_components), dtype=np.int32)
    np.add.at(
        counts,
        (np.repeat(np.arange(reps), n_components), sampled.ravel()),
        1,
    )
    return counts


def _metric_matrix_for_draws(
    inputs: ValidatedInputs,
    *,
    draw_counts: np.ndarray,
    row_component_index: np.ndarray,
    inverse_group_size: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Return metric matrices and defined-label counts for each component draw."""

    n_draws = int(draw_counts.shape[0])
    metric_values = {
        metric: np.zeros((n_draws, len(MODEL_IDS)), dtype=float)
        for metric in METRIC_ORDER
    }
    defined_counts = {
        metric: np.zeros(n_draws, dtype=int)
        for metric in METRIC_ORDER
    }
    truth_reference = inputs.oof_by_model[MODEL_IDS[0]]

    for draw_index in range(n_draws):
        weights = (
            draw_counts[draw_index, row_component_index].astype(float)
            * inverse_group_size
        )
        f1_by_model: list[list[float]] = [[] for _ in MODEL_IDS]
        roc_by_model: list[list[float]] = [[] for _ in MODEL_IDS]
        ap_by_model: list[list[float]] = [[] for _ in MODEL_IDS]
        defined_f1 = defined_roc = defined_ap = 0

        for label in OFFICIAL_LABELS:
            truth = truth_reference[f"true_{label}"].to_numpy(dtype=int)
            positive_weight = float(weights[truth == 1].sum())
            negative_weight = float(weights[truth == 0].sum())
            # Match the accepted benchmark's fixed seven-label macro-F1
            # denominator. Labels absent from a bootstrap draw contribute
            # 0.0 through zero_division=0 instead of being omitted.
            f1_defined = True
            roc_defined = positive_weight > 0.0 and negative_weight > 0.0
            ap_defined = positive_weight > 0.0
            defined_f1 += int(f1_defined)
            defined_roc += int(roc_defined)
            defined_ap += int(ap_defined)

            for model_index, model_id in enumerate(MODEL_IDS):
                oof = inputs.oof_by_model[model_id]
                if f1_defined:
                    f1_by_model[model_index].append(
                        float(
                            f1_score(
                                truth,
                                oof[f"pred_{label}"].to_numpy(dtype=int),
                                sample_weight=weights,
                                zero_division=0,
                            )
                        )
                    )
                if roc_defined:
                    roc_by_model[model_index].append(
                        float(
                            roc_auc_score(
                                truth,
                                oof[f"prob_{label}"].to_numpy(dtype=float),
                                sample_weight=weights,
                            )
                        )
                    )
                if ap_defined:
                    ap_by_model[model_index].append(
                        float(
                            average_precision_score(
                                truth,
                                oof[f"prob_{label}"].to_numpy(dtype=float),
                                sample_weight=weights,
                            )
                        )
                    )

        if not defined_f1 or not defined_roc or not defined_ap:
            raise RuntimeError("A component draw left a headline metric wholly undefined")
        defined_counts[METRIC_ORDER[0]][draw_index] = defined_f1
        defined_counts[METRIC_ORDER[1]][draw_index] = defined_roc
        defined_counts[METRIC_ORDER[2]][draw_index] = defined_ap
        for model_index in range(len(MODEL_IDS)):
            metric_values[METRIC_ORDER[0]][draw_index, model_index] = float(
                np.mean(f1_by_model[model_index])
            )
            metric_values[METRIC_ORDER[1]][draw_index, model_index] = float(
                np.mean(roc_by_model[model_index])
            )
            metric_values[METRIC_ORDER[2]][draw_index, model_index] = float(
                np.mean(ap_by_model[model_index])
            )
    return metric_values, defined_counts


def build_all_metric_bootstrap(
    inputs: ValidatedInputs,
    *,
    seed: int,
    reps: int,
    confidence_level: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if reps < 2:
        raise ValueError("bootstrap reps must be at least 2")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be strictly between 0 and 1")

    component_ids, row_component_index, inverse_group_size = _component_layout(inputs)
    draws = _draw_counts(n_components=len(component_ids), reps=reps, seed=seed)
    point_draw = np.ones((1, len(component_ids)), dtype=np.int32)
    bootstrap_values, defined_counts = _metric_matrix_for_draws(
        inputs,
        draw_counts=draws,
        row_component_index=row_component_index,
        inverse_group_size=inverse_group_size,
    )
    point_values, point_defined_counts = _metric_matrix_for_draws(
        inputs,
        draw_counts=point_draw,
        row_component_index=row_component_index,
        inverse_group_size=inverse_group_size,
    )

    alpha = 1.0 - confidence_level
    model_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for metric in METRIC_ORDER:
        values = bootstrap_values[metric]
        point = point_values[metric][0]
        metric_defined = defined_counts[metric]
        for model_index, model_id in enumerate(MODEL_IDS):
            lower, upper = np.quantile(
                values[:, model_index],
                [alpha / 2.0, 1.0 - (alpha / 2.0)],
                method="linear",
            )
            model_rows.append(
                {
                    "model_id": model_id,
                    "metric": metric,
                    "point_estimate": float(point[model_index]),
                    "bootstrap_mean": float(values[:, model_index].mean()),
                    "bootstrap_standard_error": float(values[:, model_index].std(ddof=1)),
                    "confidence_level": float(confidence_level),
                    "ci_lower": float(lower),
                    "ci_upper": float(upper),
                    "bootstrap_reps": int(reps),
                    "seed": int(seed),
                    "n_source_components": int(len(component_ids)),
                    "defined_label_count_point": int(point_defined_counts[metric][0]),
                    "defined_label_count_min": int(metric_defined.min()),
                    "defined_label_count_max": int(metric_defined.max()),
                }
            )
        for left_index, right_index in combinations(range(len(MODEL_IDS)), 2):
            differences = values[:, left_index] - values[:, right_index]
            lower, upper = np.quantile(
                differences,
                [alpha / 2.0, 1.0 - (alpha / 2.0)],
                method="linear",
            )
            pair_rows.append(
                {
                    "model_a": MODEL_IDS[left_index],
                    "model_b": MODEL_IDS[right_index],
                    "metric": metric,
                    "difference_definition": "model_a_minus_model_b",
                    "point_estimate_difference": float(
                        point[left_index] - point[right_index]
                    ),
                    "bootstrap_mean_difference": float(differences.mean()),
                    "bootstrap_standard_error": float(differences.std(ddof=1)),
                    "confidence_level": float(confidence_level),
                    "ci_lower": float(lower),
                    "ci_upper": float(upper),
                    "ci_excludes_zero": bool(lower > 0.0 or upper < 0.0),
                    "bootstrap_reps": int(reps),
                    "seed": int(seed),
                    "n_source_components": int(len(component_ids)),
                    "resampling_unit": SOURCE_GROUP_COLUMN,
                    "defined_label_count_min": int(metric_defined.min()),
                    "defined_label_count_max": int(metric_defined.max()),
                }
            )
    details = {
        "n_source_components": len(component_ids),
        "component_ids_sha256": _sha256_bytes("\n".join(component_ids).encode("utf-8")),
        "defined_label_counts": {
            metric: {
                "point": int(point_defined_counts[metric][0]),
                "bootstrap_min": int(defined_counts[metric].min()),
                "bootstrap_max": int(defined_counts[metric].max()),
            }
            for metric in METRIC_ORDER
        },
    }
    return pd.DataFrame(model_rows), pd.DataFrame(pair_rows), details


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(
        index=False, lineterminator="\n", na_rep="", float_format="%.12g"
    ).encode("utf-8")


def _write_atomically(payloads: Mapping[Path, bytes]) -> None:
    temporary: dict[Path, Path] = {}
    try:
        for path, payload in payloads.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_name(f".{path.name}.tmp")
            temp.write_bytes(payload)
            temporary[path] = temp
        for path, temp in temporary.items():
            temp.replace(path)
    finally:
        for temp in temporary.values():
            if temp.exists():
                temp.unlink()


def run_all_metric_bootstrap(
    *,
    feature_label_csv: Path,
    outer_manifest_csv: Path,
    inner_manifest_csv: Path,
    benchmark_root: Path,
    out_dir: Path,
    threshold: float = 0.3,
    seed: int = 42,
    reps: int = 2000,
    confidence_level: float = 0.95,
    overwrite: bool = False,
) -> dict[str, Path]:
    paths = {key: Path(out_dir) / name for key, name in OUTPUT_FILENAMES.items()}
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Refusing to overwrite bootstrap artifact: {existing[0]}")

    inputs = load_and_validate_inputs(
        feature_label_csv=Path(feature_label_csv),
        outer_manifest_csv=Path(outer_manifest_csv),
        inner_manifest_csv=Path(inner_manifest_csv),
        benchmark_root=Path(benchmark_root),
        threshold=threshold,
    )
    model_intervals, paired_differences, details = build_all_metric_bootstrap(
        inputs, seed=seed, reps=reps, confidence_level=confidence_level
    )
    csv_payloads = {
        paths["model_intervals"]: _csv_bytes(model_intervals),
        paths["paired_differences"]: _csv_bytes(paired_differences),
    }
    input_paths: dict[str, Path] = {
        "feature_label_csv": Path(feature_label_csv),
        "outer_manifest_csv": Path(outer_manifest_csv),
        "inner_manifest_csv": Path(inner_manifest_csv),
    }
    input_paths.update({f"oof_{key}": value for key, value in inputs.oof_paths.items()})
    report = {
        "schema_version": 1,
        "status": "passed",
        "diagnostic_id": "source_grouped_all_metric_bootstrap_v1",
        "accepted_benchmark_modified": False,
        "model_ids": list(MODEL_IDS),
        "official_labels": list(OFFICIAL_LABELS),
        "metrics": list(METRIC_ORDER),
        "method": (
            "paired source-group cluster bootstrap with identical component draws for "
            "all models; each sampled component contributes total weight one"
        ),
        "seed": int(seed),
        "bootstrap_reps": int(reps),
        "confidence_level": float(confidence_level),
        "quantile_method": "linear percentile interval",
        "details": details,
        "inputs": {
            key: {"path": str(path), "sha256": sha256_file(path)}
            for key, path in input_paths.items()
        },
        "artifacts": {
            path.name: {"sha256": _sha256_bytes(payload), "bytes": len(payload)}
            for path, payload in csv_payloads.items()
        },
    }
    report_payload = (
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    _write_atomically({**csv_payloads, paths["report"]: report_payload})
    checksum_keys = ("model_intervals", "paired_differences", "report")
    paths["checksums"].write_text(
        "".join(f"{sha256_file(paths[key])}  {paths[key].name}\n" for key in checksum_keys),
        encoding="utf-8",
    )
    return paths


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-label-csv", type=Path, default=DEFAULT_FEATURE_LABEL_CSV)
    parser.add_argument("--outer-manifest-csv", type=Path, default=DEFAULT_OUTER_MANIFEST_CSV)
    parser.add_argument("--inner-manifest-csv", type=Path, default=DEFAULT_INNER_MANIFEST_CSV)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reps", type=int, default=2000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paths = run_all_metric_bootstrap(
        feature_label_csv=args.feature_label_csv,
        outer_manifest_csv=args.outer_manifest_csv,
        inner_manifest_csv=args.inner_manifest_csv,
        benchmark_root=args.benchmark_root,
        out_dir=args.out_dir,
        threshold=args.threshold,
        seed=args.seed,
        reps=args.reps,
        confidence_level=args.confidence_level,
        overwrite=args.overwrite,
    )
    print("All-metric source-grouped bootstrap complete")
    for key, path in paths.items():
        print(f"- {key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

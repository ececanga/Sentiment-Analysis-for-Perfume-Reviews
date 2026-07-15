# Date Created: 2026-07-15

"""Tests for all-headline-metric paired source-component bootstrap outputs."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.training.build_source_grouped_all_metric_bootstrap import (
    METRIC_ORDER,
    build_all_metric_bootstrap,
    run_all_metric_bootstrap,
)
from src.training.generate_source_grouped_diagnostics import (
    MODEL_IDS,
    OFFICIAL_LABELS,
    build_component_weighted_model_metrics,
    load_and_validate_inputs,
)


class AllMetricBootstrapTests(unittest.TestCase):
    def _fixture(self, root: Path) -> dict[str, Path]:
        feature_path = root / "features.csv"
        outer_path = root / "outer.csv"
        inner_path = root / "inner.csv"
        benchmark_root = root / "benchmark"
        predictions = benchmark_root / "predictions"
        predictions.mkdir(parents=True)

        perfume_ids = [f"p{i:02d}" for i in range(15)]
        group_ids = [f"g{i:02d}" for i in range(10)]
        group_by_perfume = {
            perfume_id: group_ids[min(index // 2, 9)]
            for index, perfume_id in enumerate(perfume_ids)
        }
        fold_by_group = {group_id: (index % 5) + 1 for index, group_id in enumerate(group_ids)}
        feature_rows = []
        for index, perfume_id in enumerate(perfume_ids):
            feature_rows.append(
                {
                    "perfume_id": perfume_id,
                    "joy": int(index % 2 == 0),
                    "sadness": int(index % 5 == 0),
                    "fear": int(index in {0, 9}),
                    "trust": int(index % 3 != 0),
                    "surprise": int(index in {2, 7, 12}),
                    "anticipation": int(index % 2 == 0),
                    "disgust": int(index in {4, 11}),
                }
            )
        features = pd.DataFrame(feature_rows)
        features.to_csv(feature_path, index=False)

        outer_rows = []
        for perfume_id in perfume_ids:
            group_id = group_by_perfume[perfume_id]
            outer_rows.append(
                {
                    "perfume_id": perfume_id,
                    "fold_id": fold_by_group[group_id],
                    "source_group_id": group_id,
                    "source_group_size": sum(
                        value == group_id for value in group_by_perfume.values()
                    ),
                }
            )
        outer = pd.DataFrame(outer_rows)
        outer.to_csv(outer_path, index=False)

        inner_rows = []
        for outer_fold in range(1, 6):
            eligible = sorted(
                group for group, fold in fold_by_group.items() if fold != outer_fold
            )
            validation_group = eligible[0]
            for perfume_id in perfume_ids:
                group_id = group_by_perfume[perfume_id]
                if fold_by_group[group_id] == outer_fold:
                    continue
                inner_rows.append(
                    {
                        "outer_fold_id": outer_fold,
                        "perfume_id": perfume_id,
                        "source_group_id": group_id,
                        "inner_role": "val" if group_id == validation_group else "train",
                    }
                )
        pd.DataFrame(inner_rows).to_csv(inner_path, index=False)

        for model_index, model_id in enumerate(MODEL_IDS):
            oof = outer.copy()
            for label_index, label in enumerate(OFFICIAL_LABELS):
                truth = features[label].to_numpy(int)
                noise = ((np.arange(len(truth)) + model_index + label_index) % 5 == 0)
                prediction = np.where(noise, 1 - truth, truth).astype(int)
                probability = np.where(prediction == 1, 0.75, 0.15).astype(float)
                oof[f"true_{label}"] = truth
                oof[f"prob_{label}"] = probability
                oof[f"pred_{label}"] = prediction
            oof.to_csv(predictions / f"{model_id}_oof.csv", index=False)
        return {
            "feature_label_csv": feature_path,
            "outer_manifest_csv": outer_path,
            "inner_manifest_csv": inner_path,
            "benchmark_root": benchmark_root,
        }

    def test_metric_shapes_point_estimates_and_determinism(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = self._fixture(root)
            inputs = load_and_validate_inputs(**paths, threshold=0.3)
            model_rows, pair_rows, _ = build_all_metric_bootstrap(
                inputs, seed=17, reps=32, confidence_level=0.95
            )
            self.assertEqual(len(model_rows), len(MODEL_IDS) * len(METRIC_ORDER))
            self.assertEqual(len(pair_rows), 10 * len(METRIC_ORDER))
            self.assertEqual(set(model_rows["metric"]), set(METRIC_ORDER))
            component = build_component_weighted_model_metrics(inputs, threshold=0.3)
            observed = model_rows.loc[
                model_rows["metric"] == "component_equal_macro_f1"
            ].set_index("model_id")["point_estimate"]
            expected = component.set_index("model_id")["macro_f1"]
            self.assertTrue(np.allclose(observed.loc[expected.index], expected))

            first = run_all_metric_bootstrap(
                **paths, out_dir=root / "first", seed=17, reps=32
            )
            second = run_all_metric_bootstrap(
                **paths, out_dir=root / "second", seed=17, reps=32
            )
            for key in first:
                self.assertEqual(first[key].read_bytes(), second[key].read_bytes(), key)


if __name__ == "__main__":
    unittest.main()

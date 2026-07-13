"""Tests for the unweighted engineered-feature LR sensitivity runner."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from src.training.run_logistic_regression_unweighted_sensitivity import (
    DEFAULT_BASE_CONFIG,
    MODEL_SPEC,
    build_sensitivity_config,
)


class UnweightedLogisticRegressionSensitivityTests(unittest.TestCase):
    def test_model_spec_disables_class_weighting(self) -> None:
        model = MODEL_SPEC.build_model(42, None)
        self.assertIsNone(model.named_steps["clf"].class_weight)
        self.assertIsNone(MODEL_SPEC.model_params["classifier"]["class_weight"])

    def test_derived_config_isolates_outputs_and_reuses_folds(self) -> None:
        base = yaml.safe_load(DEFAULT_BASE_CONFIG.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmpdir:
            experiment_root = Path(tmpdir).resolve()
            path = build_sensitivity_config(
                base_config=DEFAULT_BASE_CONFIG,
                experiment_root=experiment_root,
            )
            derived = yaml.safe_load(path.read_text(encoding="utf-8"))

        self.assertNotEqual(derived["artifacts"]["root_dir"], base["artifacts"]["root_dir"])
        self.assertEqual(
            derived["splits"]["outer_cv"]["manifest_path"],
            base["splits"]["outer_cv"]["manifest_path"],
        )
        self.assertEqual(
            derived["splits"]["inner_validation"]["manifest_path"],
            base["splits"]["inner_validation"]["manifest_path"],
        )


if __name__ == "__main__":
    unittest.main()

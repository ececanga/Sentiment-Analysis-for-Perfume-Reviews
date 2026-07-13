# Date Created: 2026-07-13

"""Focused tests for Phase-2 thesis comparison exports."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.training.generate_final_benchmark_figures import (
    MODEL_ORDER,
    export_final_table,
    load_final_model_comparison,
)


class FinalBenchmarkFigureTests(unittest.TestCase):
    def _comparison(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "model_id": model_id,
                    "macro_f1": 0.2 + index / 100,
                    "macro_roc_auc": 0.5 + index / 100,
                    "macro_pr_auc": 0.3 + index / 100,
                    "row_complete": True,
                    "split_design": "5-fold source-component grouped CV",
                    "threshold": 0.3,
                }
                for index, model_id in enumerate(MODEL_ORDER)
            ]
        )

    def test_export_uses_average_precision_terminology_and_split_binding(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source = tmp / "comparison.csv"
            output = tmp / "thesis.csv"
            self._comparison().to_csv(source, index=False)
            loaded = load_final_model_comparison(
                source,
                expected_split_design="5-fold source-component grouped CV",
                expected_threshold=0.3,
            )
            export_final_table(loaded, path=output)
            exported = pd.read_csv(output)
            self.assertIn("macro_average_precision", exported.columns)
            self.assertNotIn("macro_pr_auc", exported.columns)

            corrupt = self._comparison()
            corrupt.loc[0, "split_design"] = "legacy KFold"
            corrupt.to_csv(source, index=False)
            with self.assertRaisesRegex(ValueError, "non-configured split design"):
                load_final_model_comparison(
                    source,
                    expected_split_design="5-fold source-component grouped CV",
                    expected_threshold=0.3,
                )


if __name__ == "__main__":
    unittest.main()

# Date Created: 2026-04-13 23:02:44+03:00
# Date Last Modified: 2026-05-15

"""Tests for Stage 7 Gemini 784 training pipeline."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.training.train_eval_gemini784 import run_training


def _make_synth_df(n: int = 20) -> pd.DataFrame:
    rows = []
    for i in range(n):
        row = {
            "Perfume": f"Perfume {i}",
            "perfume_id": f"p-{i}",
            "NRC_anger": i % 5,
            "NRC_anticipation": i % 7,
            "NRC_disgust": i % 3,
            "NRC_fear": i % 4,
            "NRC_joy": (i + 1) % 6,
            "NRC_sadness": (i + 2) % 4,
            "NRC_surprise": i % 2,
            "NRC_trust": (i + 3) % 5,
            "NRC_positive": i % 10,
            "NRC_negative": (i + 1) % 8,
            "VAD_valence": 0.1 + (i % 10) * 0.05,
            "VAD_arousal": 0.2 + (i % 9) * 0.04,
            "VAD_dominance": 0.3 + (i % 8) * 0.03,
            "Hartmann_joy": 0.1 * ((i + 1) % 5),
            "Hartmann_neutral": 0.1 * (i % 5),
            "Hartmann_surprise": 0.05 * (i % 6),
            "Hartmann_sadness": 0.05 * ((i + 2) % 6),
            "Hartmann_anger": 0.03 * (i % 7),
            "Hartmann_disgust": 0.02 * (i % 5),
            "Hartmann_fear": 0.04 * (i % 4),
            "Savani_joy": 0.8 if i % 2 == 0 else 0.2,
            "Savani_sadness": 0.7 if i % 9 == 0 else 0.1,
            "Savani_love": 0.5 if i % 3 == 0 else 0.2,
            "Savani_anger": 0.0,
            "Savani_fear": 0.6 if i in (0, 1) else 0.0,
            "Savani_surprise": 0.6 if i % 7 == 0 else 0.1,
            "GoEmotions_joy": 0.7 if i % 2 == 0 else 0.2,
            "GoEmotions_sadness": 0.8 if i % 9 == 0 else 0.1,
            "GoEmotions_anger": 0.0,
            "GoEmotions_fear": 0.7 if i == 0 else 0.1,
            "GoEmotions_surprise": 0.8 if i % 7 == 0 else 0.1,
            "GoEmotions_trust": 0.7 if i % 4 == 0 else 0.2,
            "Cardiff_joy": 0.6 if i % 2 == 0 else 0.1,
            "Cardiff_optimism": 0.6 if i % 3 == 0 else 0.1,
            "Cardiff_anger": 0.0,
            "Cardiff_sadness": 0.7 if i % 9 == 0 else 0.1,
            "joy": 1 if i % 2 == 0 else 0,
            "sadness": 1 if i % 9 == 0 else 0,
            "anger": 0,  # degenerate target by design
            "fear": 1 if i == 0 else 0,  # very rare target by design
            "trust": 1 if i % 4 == 0 else 0,
            "surprise": 1 if i % 7 == 0 else 0,
            "anticipation": 1 if i % 3 == 0 else 0,
            "disgust": 1 if i in (0, 9) else 0,
            "valence": 4.5 + (i % 5) * 0.5,
            "arousal": 4.0 + (i % 4) * 0.5,
            "dominance": 3.5 + (i % 6) * 0.5,
        }
        rows.append(row)
    return pd.DataFrame(rows)


class TrainEvalGemini784Tests(unittest.TestCase):
    def test_happy_path_outputs_and_degenerate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_csv = tmp / "merged.csv"
            out_dir = tmp / "out"
            _make_synth_df().to_csv(input_csv, index=False)

            metrics = run_training(
                input_csv=input_csv,
                out_dir=out_dir,
                seed=42,
                emotion_threshold=0.5,
                max_folds=5,
                save_models=False,
            )

            self.assertEqual(metrics["n_rows"], 20)
            self.assertTrue((out_dir / "stage7_gemini784_metrics.json").exists())
            self.assertTrue((out_dir / "stage7_gemini784_oof_predictions.csv").exists())
            self.assertTrue((out_dir / "stage7_gemini784_label_diagnostics.csv").exists())
            self.assertTrue((out_dir / "stage7_gemini784_vad_metrics.csv").exists())
            self.assertFalse((out_dir / "stage7_gemini784_models.joblib").exists())

            diag = pd.read_csv(out_dir / "stage7_gemini784_label_diagnostics.csv")
            anger_row = diag.loc[diag["emotion"] == "anger"].iloc[0]
            self.assertEqual(anger_row["status"], "degenerate_constant")

    def test_save_models_optional(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_csv = tmp / "merged.csv"
            out_dir = tmp / "out"
            _make_synth_df().to_csv(input_csv, index=False)

            run_training(
                input_csv=input_csv,
                out_dir=out_dir,
                save_models=True,
            )
            self.assertTrue((out_dir / "stage7_gemini784_models.joblib").exists())

    def test_oof_row_count_and_unique_perfume_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_csv = tmp / "merged.csv"
            out_dir = tmp / "out"
            df = _make_synth_df()
            df.to_csv(input_csv, index=False)

            run_training(input_csv=input_csv, out_dir=out_dir, save_models=False)
            oof = pd.read_csv(out_dir / "stage7_gemini784_oof_predictions.csv")

            self.assertEqual(len(oof), len(df))
            self.assertEqual(oof["perfume_id"].nunique(), len(df))


if __name__ == "__main__":
    unittest.main()


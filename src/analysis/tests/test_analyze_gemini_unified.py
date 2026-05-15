# Date Created: 2026-04-13 23:02:44+03:00
# Date Last Modified: 2026-05-15

"""Tests for minimal Gemini + unified analysis module."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.analyze_gemini_unified import build_ids_from_names, run_analysis


def _make_unified_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Perfume": "Alpha Eau de Parfum",
                "NRC_joy": 1,
                "NRC_sadness": 0,
                "NRC_anger": 0,
                "NRC_fear": 0,
                "NRC_trust": 1,
                "NRC_surprise": 0,
                "NRC_anticipation": 1,
                "NRC_disgust": 0,
                "VAD_valence": 0.80,
                "VAD_arousal": 0.60,
                "VAD_dominance": 0.70,
                "JHartmann_Label": "joy",
                "Savani_Label": "joy",
                "GoEmotions_Label": "admiration",
                "Cardiff_Label": "optimism",
            },
            {
                "Perfume": "Beta Extrait",
                "NRC_joy": 0,
                "NRC_sadness": 1,
                "NRC_anger": 1,
                "NRC_fear": 0,
                "NRC_trust": 0,
                "NRC_surprise": 0,
                "NRC_anticipation": 0,
                "NRC_disgust": 1,
                "VAD_valence": 0.30,
                "VAD_arousal": 0.50,
                "VAD_dominance": 0.20,
                "JHartmann_Label": "sadness",
                "Savani_Label": "sadness",
                "GoEmotions_Label": "sadness",
                "Cardiff_Label": "sadness",
            },
            {
                "Perfume": "Gamma",
                "NRC_joy": 0,
                "NRC_sadness": 0,
                "NRC_anger": 0,
                "NRC_fear": 1,
                "NRC_trust": 0,
                "NRC_surprise": 1,
                "NRC_anticipation": 0,
                "NRC_disgust": 0,
                "VAD_valence": 0.20,
                "VAD_arousal": 0.90,
                "VAD_dominance": 0.10,
                "JHartmann_Label": "fear",
                "Savani_Label": "fear",
                "GoEmotions_Label": "fear",
                "Cardiff_Label": "anger",
            },
        ]
    )


def _make_gemini_df_names_only() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "perfume_name": "Alpha Eau de Parfum",
                "joy": 1,
                "sadness": 0,
                "anger": 0,
                "fear": 0,
                "trust": 1,
                "surprise": 0,
                "anticipation": 1,
                "disgust": 0,
                "valence": 0.78,
                "arousal": 0.65,
                "dominance": 0.74,
            },
            {
                "perfume_name": "Beta Extrait",
                "joy": 0,
                "sadness": 1,
                "anger": 1,
                "fear": 0,
                "trust": 0,
                "surprise": 0,
                "anticipation": 0,
                "disgust": 1,
                "valence": 0.35,
                "arousal": 0.45,
                "dominance": 0.22,
            },
            {
                "perfume_name": "Gamma",
                "joy": 0,
                "sadness": 0,
                "anger": 0,
                "fear": 1,
                "trust": 0,
                "surprise": 1,
                "anticipation": 0,
                "disgust": 0,
                "valence": 0.25,
                "arousal": 0.85,
                "dominance": 0.15,
            },
        ]
    )


class AnalyzeGeminiUnifiedTests(unittest.TestCase):
    def test_happy_path_and_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            unified_path = tmp / "unified.csv"
            gemini_path = tmp / "gemini.csv"
            out_dir = tmp / "out"

            _make_unified_df().to_csv(unified_path, index=False)
            _make_gemini_df_names_only().to_csv(gemini_path, index=False)

            summary = run_analysis(
                unified_csv=unified_path,
                gemini_csv=gemini_path,
                out_dir=out_dir,
                perfume_col="Perfume",
                strict=True,
            )

            self.assertEqual(summary["join_counts"]["matched"], 3)
            self.assertEqual(summary["join_counts"]["unified_only"], 0)
            self.assertEqual(summary["join_counts"]["gemini_only"], 0)
            self.assertTrue((out_dir / "gemini_unified_merged.csv").exists())
            self.assertTrue((out_dir / "gemini_unified_join_report.json").exists())
            self.assertTrue((out_dir / "agreement_binary_nrc_vs_gemini.csv").exists())
            self.assertTrue((out_dir / "agreement_toplabel_models_vs_gemini.csv").exists())
            self.assertTrue((out_dir / "agreement_vad_continuous.csv").exists())
            self.assertTrue((out_dir / "gemini_unified_summary_report.json").exists())

    def test_duplicate_perfume_id_strict_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            unified = _make_unified_df()
            gemini = _make_gemini_df_names_only()
            ids = build_ids_from_names(gemini["perfume_name"])
            gemini["perfume_id"] = ids["perfume_id"]
            gemini["perfume_id_hash"] = ids["perfume_id_hash"]
            gemini.loc[1, "perfume_id"] = gemini.loc[0, "perfume_id"]

            unified_path = tmp / "unified.csv"
            gemini_path = tmp / "gemini.csv"
            out_dir = tmp / "out"
            unified.to_csv(unified_path, index=False)
            gemini.to_csv(gemini_path, index=False)

            with self.assertRaises(ValueError):
                run_analysis(
                    unified_csv=unified_path,
                    gemini_csv=gemini_path,
                    out_dir=out_dir,
                    perfume_col="Perfume",
                    strict=True,
                )

    def test_hash_mismatch_strict_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            unified = _make_unified_df()
            gemini = _make_gemini_df_names_only()
            ids = build_ids_from_names(gemini["perfume_name"])
            gemini["perfume_id"] = ids["perfume_id"]
            gemini["perfume_id_hash"] = ids["perfume_id_hash"]
            gemini.loc[0, "perfume_id_hash"] = "deadbeef00"

            unified_path = tmp / "unified.csv"
            gemini_path = tmp / "gemini.csv"
            out_dir = tmp / "out"
            unified.to_csv(unified_path, index=False)
            gemini.to_csv(gemini_path, index=False)

            with self.assertRaises(ValueError):
                run_analysis(
                    unified_csv=unified_path,
                    gemini_csv=gemini_path,
                    out_dir=out_dir,
                    perfume_col="Perfume",
                    strict=True,
                )

    def test_legacy_fallback_without_perfume_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            unified_path = tmp / "unified.csv"
            gemini_path = tmp / "gemini.csv"
            out_dir = tmp / "out"

            _make_unified_df().to_csv(unified_path, index=False)
            _make_gemini_df_names_only().to_csv(gemini_path, index=False)

            summary = run_analysis(
                unified_csv=unified_path,
                gemini_csv=gemini_path,
                out_dir=out_dir,
                perfume_col="Perfume",
                strict=True,
            )
            self.assertEqual(summary["join_counts"]["matched"], 3)

    def test_vad_agreement_has_required_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            unified_path = tmp / "unified.csv"
            gemini_path = tmp / "gemini.csv"
            out_dir = tmp / "out"

            _make_unified_df().to_csv(unified_path, index=False)
            _make_gemini_df_names_only().to_csv(gemini_path, index=False)
            run_analysis(
                unified_csv=unified_path,
                gemini_csv=gemini_path,
                out_dir=out_dir,
                perfume_col="Perfume",
                strict=True,
            )

            vad_df = pd.read_csv(out_dir / "agreement_vad_continuous.csv")
            req = {"dimension", "n", "pearson_r", "pearson_p", "spearman_r", "spearman_p", "mae"}
            self.assertTrue(req.issubset(set(vad_df.columns)))
            sub = vad_df[vad_df["n"] >= 2]
            self.assertTrue(np.isfinite(sub["mae"]).all())


if __name__ == "__main__":
    unittest.main()


# Date Created: 2026-04-13 23:02:44+03:00
# Date Last Modified: 2026-05-15

"""Tests for Stage 8A feature importance analysis."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.analyze_stage8_feature_importance import run_stage8_analysis

EMOTIONS = ["joy", "sadness", "anger", "fear", "trust", "surprise", "anticipation", "disgust"]
VAD = ["valence", "arousal", "dominance"]


def _make_synth_stage8_inputs(n: int = 30) -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame]:
    rows = []
    for i in range(n):
        x = i / max(1, (n - 1))
        joy = 1 if i % 2 == 0 else 0
        sadness = 1 if i in (1, 11) else 0
        anger = 0  # degenerate by design
        fear = 1 if i in (2, 17) else 0
        trust = 1 if i % 3 == 0 else 0
        surprise = 1 if i % 5 == 0 else 0
        anticipation = 1 if i % 4 < 2 else 0
        disgust = 1 if i in (3, 13) else 0

        row = {
            "perfume_id": f"p-{i}",
            "Perfume": f"Perfume {i}",
            "NRC_joy": 1.0 * joy + x,
            "VAD_valence": 0.2 + 0.6 * x,
            "Hartmann_joy": 0.1 + 0.7 * x,
            "Savani_joy": 0.2 + 0.5 * x,
            "GoEmotions_joy": 0.3 + 0.4 * x,
            "Cardiff_joy": 0.4 + 0.3 * x,
            "joy": joy,
            "sadness": sadness,
            "anger": anger,
            "fear": fear,
            "trust": trust,
            "surprise": surprise,
            "anticipation": anticipation,
            "disgust": disgust,
            "valence": 4.0 + 2.0 * x,
            "arousal": 3.5 + 1.5 * x,
            "dominance": 3.0 + 1.2 * x,
        }
        rows.append(row)

    merged_df = pd.DataFrame(rows)
    feature_cols = ["NRC_joy", "VAD_valence", "Hartmann_joy", "Savani_joy", "GoEmotions_joy", "Cardiff_joy"]

    oof_df = merged_df[["perfume_id", "Perfume"]].copy()
    for emo in EMOTIONS:
        y = merged_df[emo].to_numpy(dtype=int)
        oof_df[f"true_{emo}"] = y
        oof_df[f"prob_{emo}"] = 0.5 + 0.2 * (y - 0.5)
        oof_df[f"pred_{emo}"] = y
    for dim in VAD:
        y = merged_df[dim].to_numpy(dtype=float)
        oof_df[f"true_{dim}"] = y
        oof_df[f"pred_{dim}"] = y + np.random.default_rng(42).normal(0, 0.01, size=len(y))

    label_diag_rows = []
    for emo in EMOTIONS:
        y = merged_df[emo].to_numpy(dtype=int)
        n_pos = int(y.sum())
        n_neg = int(len(y) - n_pos)
        is_const = n_pos == 0 or n_neg == 0
        status = "degenerate_constant" if is_const else "trained"
        n_splits = 0 if is_const else int(min(3, n_pos, n_neg))
        label_diag_rows.append(
            {
                "emotion": emo,
                "n": len(y),
                "n_pos": n_pos,
                "n_neg": n_neg,
                "prevalence": float(n_pos / len(y)),
                "is_constant": is_const,
                "n_splits": n_splits,
                "status": status,
            }
        )
    label_diag_df = pd.DataFrame(label_diag_rows)

    metrics = {
        "n_rows": len(merged_df),
        "n_features": len(feature_cols),
        "feature_columns": feature_cols,
        "config": {"seed": 42, "max_folds": 3, "save_models": False},
    }
    return merged_df, oof_df, metrics, label_diag_df


class Stage8FeatureImportanceTests(unittest.TestCase):
    def test_happy_path_outputs_and_rank_columns(self) -> None:
        merged_df, oof_df, metrics, diag_df = _make_synth_stage8_inputs()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            merged_csv = tmp / "merged.csv"
            oof_csv = tmp / "oof.csv"
            metrics_json = tmp / "metrics.json"
            diag_csv = tmp / "diag.csv"
            out_dir = tmp / "out"

            merged_df.to_csv(merged_csv, index=False)
            oof_df.to_csv(oof_csv, index=False)
            metrics_json.write_text(json.dumps(metrics), encoding="utf-8")
            diag_df.to_csv(diag_csv, index=False)

            run_stage8_analysis(
                merged_csv=merged_csv,
                stage7_metrics_json=metrics_json,
                stage7_label_diagnostics_csv=diag_csv,
                stage7_oof_csv=oof_csv,
                out_dir=out_dir,
                n_repeats=2,
                max_folds=3,
                seed=42,
                rare_pos_threshold=3,
                strict=True,
            )

            expected = [
                "stage8_input_alignment_report.json",
                "stage8_emotion_feature_importance.csv",
                "stage8_emotion_family_importance.csv",
                "stage8_vad_feature_importance.csv",
                "stage8_vad_family_importance.csv",
                "stage8_target_interpretation_flags.csv",
                "stage8_importance_summary_report.json",
            ]
            for name in expected:
                self.assertTrue((out_dir / name).exists(), f"Missing artifact: {name}")

            em_feat = pd.read_csv(out_dir / "stage8_emotion_feature_importance.csv")
            em_fam = pd.read_csv(out_dir / "stage8_emotion_family_importance.csv")
            vad_feat = pd.read_csv(out_dir / "stage8_vad_feature_importance.csv")
            vad_fam = pd.read_csv(out_dir / "stage8_vad_family_importance.csv")
            flags = pd.read_csv(out_dir / "stage8_target_interpretation_flags.csv")

            required_feature_cols = {
                "target",
                "feature",
                "family",
                "metric_name",
                "importance_mean",
                "importance_std",
                "importance_median",
                "baseline_metric_mean",
                "n_folds_used",
                "n_repeats",
                "status",
                "interpretation_level",
                "rank_within_target",
            }
            required_family_cols = {
                "target",
                "family",
                "metric_name",
                "importance_mean",
                "importance_std",
                "importance_median",
                "baseline_metric_mean",
                "n_folds_used",
                "n_repeats",
                "status",
                "interpretation_level",
                "rank_within_target",
            }
            self.assertTrue(required_feature_cols.issubset(set(em_feat.columns)))
            self.assertTrue(required_feature_cols.issubset(set(vad_feat.columns)))
            self.assertTrue(required_family_cols.issubset(set(em_fam.columns)))
            self.assertTrue(required_family_cols.issubset(set(vad_fam.columns)))

            for df in (em_feat, em_fam, vad_feat, vad_fam):
                for target, grp in df.groupby("target"):
                    ranks = sorted(grp["rank_within_target"].astype(int).tolist())
                    self.assertEqual(ranks, list(range(1, len(grp) + 1)), f"Bad ranks for {target}")

            anger_rows = em_feat.loc[em_feat["target"] == "anger"]
            self.assertTrue((anger_rows["status"] == "degenerate_constant").all())
            self.assertTrue(anger_rows["importance_mean"].isna().all())

            family_set = set(em_fam["family"].unique())
            self.assertEqual(
                family_set,
                {"NRC", "VAD", "Hartmann", "Savani", "GoEmotions", "Cardiff"},
            )

            rare_flags = flags[(flags["target_type"] == "emotion") & (flags["target"].isin(["sadness", "fear", "disgust"]))]
            self.assertTrue((rare_flags["interpretation_level"] == "exploratory_rare").all())

    def test_strict_duplicate_perfume_id_fails(self) -> None:
        merged_df, oof_df, metrics, diag_df = _make_synth_stage8_inputs()
        merged_df = pd.concat([merged_df, merged_df.iloc[[0]]], ignore_index=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            merged_csv = tmp / "merged.csv"
            oof_csv = tmp / "oof.csv"
            metrics_json = tmp / "metrics.json"
            diag_csv = tmp / "diag.csv"
            out_dir = tmp / "out"

            merged_df.to_csv(merged_csv, index=False)
            oof_df.to_csv(oof_csv, index=False)
            metrics_json.write_text(json.dumps(metrics), encoding="utf-8")
            diag_df.to_csv(diag_csv, index=False)

            with self.assertRaises(ValueError):
                run_stage8_analysis(
                    merged_csv=merged_csv,
                    stage7_metrics_json=metrics_json,
                    stage7_label_diagnostics_csv=diag_csv,
                    stage7_oof_csv=oof_csv,
                    out_dir=out_dir,
                    n_repeats=2,
                    max_folds=3,
                    seed=42,
                    rare_pos_threshold=3,
                    strict=True,
                )


if __name__ == "__main__":
    unittest.main()

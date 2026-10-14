"""Tests for the source-grouped supervised feature-family ablation."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path

from src.utils.paths import PROJECT_ROOT

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from src.training.run_source_grouped_feature_ablation import (
    BOOTSTRAP_REPS,
    CODE_PROVENANCE_PATHS,
    CONFIDENCE_LEVEL,
    DEFAULT_ACCEPTED_METRICS,
    DEFAULT_ACCEPTED_OOF,
    DEFAULT_ACCEPTED_CHECKSUMS,
    DEFAULT_ACCEPTED_EXPERIMENT_ROOT,
    DEFAULT_CONFIG,
    DEFAULT_FEATURE_CSV,
    DEFAULT_MANIFEST,
    DEFAULT_OUT_DIR,
    DEFAULT_TFIDF_METRICS,
    DEFAULT_TFIDF_OOF,
    FULL_VARIANT,
    PERMUTATION_REPEATS,
    SEED,
    build_coefficient_stability,
    prepare_inputs,
    prepare_output_root,
    train_variant,
    validate_canonical_run_configuration,
    validate_full_hybrid_reproduction,
    weighted_binary_ranking_metrics,
)
from src.training.source_grouped_folds import SOURCE_GROUP_COLUMN, sha256_file


EXPECTED_FAMILIES = OrderedDict(
    [
        (
            "NRC",
            [
                "NRC_anger",
                "NRC_anticipation",
                "NRC_disgust",
                "NRC_fear",
                "NRC_joy",
                "NRC_sadness",
                "NRC_surprise",
                "NRC_trust",
                "NRC_positive",
                "NRC_negative",
            ],
        ),
        ("VAD", ["VAD_valence", "VAD_arousal", "VAD_dominance"]),
        (
            "Hartmann",
            [
                "Hartmann_joy",
                "Hartmann_neutral",
                "Hartmann_surprise",
                "Hartmann_sadness",
                "Hartmann_anger",
                "Hartmann_disgust",
                "Hartmann_fear",
            ],
        ),
        (
            "Savani",
            [
                "Savani_joy",
                "Savani_sadness",
                "Savani_love",
                "Savani_anger",
                "Savani_fear",
                "Savani_surprise",
            ],
        ),
        (
            "GoEmotions",
            [
                "GoEmotions_admiration",
                "GoEmotions_amusement",
                "GoEmotions_anger",
                "GoEmotions_annoyance",
                "GoEmotions_approval",
                "GoEmotions_caring",
                "GoEmotions_confusion",
                "GoEmotions_curiosity",
                "GoEmotions_desire",
                "GoEmotions_disappointment",
                "GoEmotions_disapproval",
                "GoEmotions_disgust",
                "GoEmotions_embarrassment",
                "GoEmotions_excitement",
                "GoEmotions_fear",
                "GoEmotions_gratitude",
                "GoEmotions_grief",
                "GoEmotions_joy",
                "GoEmotions_love",
                "GoEmotions_nervousness",
                "GoEmotions_optimism",
                "GoEmotions_pride",
                "GoEmotions_realization",
                "GoEmotions_relief",
                "GoEmotions_remorse",
                "GoEmotions_sadness",
                "GoEmotions_surprise",
                "GoEmotions_neutral",
            ],
        ),
        (
            "Cardiff",
            [
                "Cardiff_joy",
                "Cardiff_optimism",
                "Cardiff_anger",
                "Cardiff_sadness",
            ],
        ),
    ]
)

EXPECTED_VARIANT_FAMILIES = OrderedDict(
    [
        ("full_hybrid", tuple(EXPECTED_FAMILIES)),
        ("nrc_only", ("NRC",)),
        ("vad_only", ("VAD",)),
        ("lexicon_only", ("NRC", "VAD")),
        ("hartmann_only", ("Hartmann",)),
        ("savani_only", ("Savani",)),
        ("goemotions_only", ("GoEmotions",)),
        ("cardiff_only", ("Cardiff",)),
        (
            "transformer_only",
            ("Hartmann", "Savani", "GoEmotions", "Cardiff"),
        ),
        ("no_nrc", ("VAD", "Hartmann", "Savani", "GoEmotions", "Cardiff")),
        ("no_vad", ("NRC", "Hartmann", "Savani", "GoEmotions", "Cardiff")),
        ("no_hartmann", ("NRC", "VAD", "Savani", "GoEmotions", "Cardiff")),
        ("no_savani", ("NRC", "VAD", "Hartmann", "GoEmotions", "Cardiff")),
        ("no_goemotions", ("NRC", "VAD", "Hartmann", "Savani", "Cardiff")),
        ("no_cardiff", ("NRC", "VAD", "Hartmann", "Savani", "GoEmotions")),
    ]
)


def _ordered_columns_for_families(families: tuple[str, ...]) -> list[str]:
    allowed = {
        feature
        for family in families
        for feature in EXPECTED_FAMILIES[family]
    }
    canonical = [
        feature
        for family_features in EXPECTED_FAMILIES.values()
        for feature in family_features
    ]
    return [feature for feature in canonical if feature in allowed]


class SourceGroupedFeatureAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # This loads only frozen inputs and accepted artifacts; it does not create
        # or modify any Phase 5 output.
        cls.prepared = prepare_inputs(
            config_path=DEFAULT_CONFIG,
            feature_csv=DEFAULT_FEATURE_CSV,
            manifest_csv=DEFAULT_MANIFEST,
            accepted_oof_csv=DEFAULT_ACCEPTED_OOF,
            accepted_metrics_json=DEFAULT_ACCEPTED_METRICS,
            tfidf_oof_csv=DEFAULT_TFIDF_OOF,
            tfidf_metrics_json=DEFAULT_TFIDF_METRICS,
        )
        cls.full_run = train_variant(
            cls.prepared,
            variant=FULL_VARIANT,
            feature_columns=cls.prepared.variants[FULL_VARIANT],
            retain_models=False,
        )

    def test_real_frozen_family_and_variant_schema_is_exact(self) -> None:
        self.assertEqual(list(self.prepared.families), list(EXPECTED_FAMILIES))
        self.assertEqual(dict(self.prepared.families), dict(EXPECTED_FAMILIES))
        self.assertEqual(
            self.prepared.feature_columns,
            [
                feature
                for family_features in EXPECTED_FAMILIES.values()
                for feature in family_features
            ],
        )

        self.assertEqual(
            list(self.prepared.variants), list(EXPECTED_VARIANT_FAMILIES)
        )
        for variant, family_names in EXPECTED_VARIANT_FAMILIES.items():
            expected_columns = _ordered_columns_for_families(family_names)
            with self.subTest(variant=variant):
                self.assertEqual(
                    self.prepared.variants[variant], expected_columns
                )
                self.assertEqual(
                    len(self.prepared.variants[variant]), len(expected_columns)
                )

        self.assertEqual(
            {name: len(columns) for name, columns in self.prepared.variants.items()},
            {
                "full_hybrid": 58,
                "nrc_only": 10,
                "vad_only": 3,
                "lexicon_only": 13,
                "hartmann_only": 7,
                "savani_only": 6,
                "goemotions_only": 28,
                "cardiff_only": 4,
                "transformer_only": 45,
                "no_nrc": 48,
                "no_vad": 55,
                "no_hartmann": 51,
                "no_savani": 52,
                "no_goemotions": 30,
                "no_cardiff": 54,
            },
        )

    def test_weighted_ranking_metrics_match_sklearn_with_ties_and_zero_weights(self) -> None:
        truth = np.asarray([0, 1, 0, 1, 1, 0, 0, 1, 0, 1], dtype=int)
        probability = np.asarray(
            [0.1, 0.8, 0.8, 0.4, 0.4, 0.1, 0.6, 0.6, 0.4, 0.8],
            dtype=float,
        )
        weights = np.asarray(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 2, 1.5, 0, 3, 1, 0, 4, 2, 0],
                [2, 0, 0, 1, 0, 3, 1, 2, 0, 4],
                [1, 2, 3, 0, 0, 4, 5, 6, 0, 7],
            ],
            dtype=float,
        )

        observed_roc, observed_ap = weighted_binary_ranking_metrics(
            truth, probability, weights
        )
        expected_roc = np.asarray(
            [
                roc_auc_score(truth, probability, sample_weight=row)
                for row in weights
            ]
        )
        expected_ap = np.asarray(
            [
                average_precision_score(truth, probability, sample_weight=row)
                for row in weights
            ]
        )
        np.testing.assert_allclose(observed_roc, expected_roc, rtol=0.0, atol=1e-15)
        np.testing.assert_allclose(observed_ap, expected_ap, rtol=0.0, atol=1e-15)

        one_roc, one_ap = weighted_binary_ranking_metrics(
            truth, probability, weights[1]
        )
        self.assertEqual(one_roc.shape, (1,))
        self.assertEqual(one_ap.shape, (1,))
        self.assertAlmostEqual(one_roc[0], expected_roc[1], places=15)
        self.assertAlmostEqual(one_ap[0], expected_ap[1], places=15)

    def test_real_full_hybrid_replays_accepted_lr_without_writes(self) -> None:
        reproduction = validate_full_hybrid_reproduction(
            self.full_run.oof,
            self.prepared.accepted_oof,
            self.prepared.target_columns,
        )
        self.assertEqual(reproduction["status"], "passed")
        self.assertEqual(reproduction["truth_mismatches"], 0)
        self.assertEqual(reproduction["prediction_mismatches"], 0)
        self.assertLessEqual(reproduction["max_probability_difference"], 1e-12)

        accepted = self.prepared.accepted_metrics
        self.assertAlmostEqual(
            self.full_run.summary["macro_f1"], accepted["macro_f1"], places=15
        )
        self.assertAlmostEqual(
            self.full_run.summary["macro_average_precision"],
            accepted["macro_pr_auc"],
            places=15,
        )
        self.assertAlmostEqual(
            self.full_run.summary["macro_roc_auc"],
            accepted["macro_roc_auc"],
            places=15,
        )
        self.assertEqual(len(self.full_run.oof), 784)
        self.assertEqual(self.full_run.oof["perfume_id"].nunique(), 784)
        self.assertTrue(
            (self.full_run.fold_diagnostics["training_status"] == "trained").all()
        )
        self.assertEqual(int(self.full_run.fold_diagnostics["convergence_warning_count"].sum()), 0)

    def test_full_hybrid_coefficient_rows_and_stability_shapes(self) -> None:
        n_folds = int(self.prepared.data["fold_id"].nunique())
        n_labels = len(self.prepared.target_columns)
        n_features = len(self.prepared.feature_columns)
        expected_raw_rows = n_folds * n_labels * n_features

        self.assertEqual(len(self.full_run.coefficient_rows), expected_raw_rows)
        raw, stability, family, vectors = build_coefficient_stability(
            self.full_run.coefficient_rows
        )
        self.assertEqual(raw.shape[0], expected_raw_rows)
        self.assertEqual(stability.shape[0], n_labels * n_features)
        self.assertEqual(family.shape[0], n_labels * len(EXPECTED_FAMILIES))
        self.assertEqual(vectors.shape[0], n_labels)

        self.assertTrue((stability["n_folds"] == n_folds).all())
        sign_total = stability[
            ["positive_fold_count", "negative_fold_count", "zero_fold_count"]
        ].sum(axis=1)
        self.assertTrue((sign_total == n_folds).all())
        self.assertTrue(
            np.isfinite(stability["coefficient_mean"].to_numpy(float)).all()
        )
        self.assertTrue(
            np.isfinite(stability["dominant_sign_fraction"].to_numpy(float)).all()
        )

        expected_family_counts = {
            family_name: len(columns)
            for family_name, columns in EXPECTED_FAMILIES.items()
        }
        for label in self.prepared.target_columns:
            observed = (
                family.loc[family["label"] == label]
                .set_index("family")["feature_count"]
                .astype(int)
                .to_dict()
            )
            self.assertEqual(observed, expected_family_counts)

        self.assertEqual(vectors["label"].tolist(), self.prepared.target_columns)
        self.assertTrue((vectors["fold_pair_count"] == 10).all())
        for column in (
            "median_signed_coefficient_spearman",
            "mean_signed_coefficient_spearman",
            "median_top_k_jaccard",
            "mean_top_k_jaccard",
        ):
            self.assertTrue(np.isfinite(vectors[column].to_numpy(float)).all())

    def test_generated_phase5_artifact_contract(self) -> None:
        root = Path(DEFAULT_OUT_DIR)
        report = json.loads((root / "REPORT.json").read_text(encoding="utf-8"))
        summary = pd.read_csv(root / "ablation_summary.csv")
        interval_table = pd.read_csv(root / "model_metric_bootstrap_intervals.csv")
        paired_table = pd.read_csv(root / "paired_bootstrap_vs_full.csv")
        fear_summary = pd.read_csv(root / "fear_excluded_ablation_summary.csv")
        fear_intervals = pd.read_csv(
            root / "fear_excluded_model_metric_bootstrap_intervals.csv"
        )
        fear_paired = pd.read_csv(
            root / "fear_excluded_paired_bootstrap_vs_full.csv"
        )
        coefficients = pd.read_csv(root / "standardized_lr_coefficients.csv")
        permutation_summary = pd.read_csv(root / "family_permutation_importance.csv")
        permutation_per_label = pd.read_csv(
            root / "family_permutation_importance_per_label.csv"
        )
        permutation_per_label_repeats = pd.read_csv(
            root / "family_permutation_importance_per_label_repeats.csv"
        )
        fear_permutation_summary = pd.read_csv(
            root / "fear_excluded_family_permutation_importance.csv"
        )
        fear_permutation_repeats = pd.read_csv(
            root / "fear_excluded_family_permutation_importance_repeats.csv"
        )

        expected_variants = list(EXPECTED_VARIANT_FAMILIES)
        oof_paths = sorted((root / "oof_predictions").glob("*_oof.csv"))
        self.assertEqual(len(oof_paths), 15)
        self.assertEqual(
            {path.stem.removesuffix("_oof") for path in oof_paths},
            set(expected_variants),
        )
        self.assertEqual(summary["variant"].tolist(), expected_variants)
        self.assertEqual(len(summary), 15)
        self.assertEqual(
            {
                column
                for column in summary.columns
                if column.startswith("variant_minus_full_")
            },
            {
                "variant_minus_full_macro_average_precision",
                "variant_minus_full_macro_f1",
                "variant_minus_full_macro_roc_auc",
            },
        )

        expected_ids = self.prepared.data["perfume_id"].astype(str).to_numpy()
        expected_folds = self.prepared.data["fold_id"].to_numpy(int)
        expected_groups = self.prepared.data[SOURCE_GROUP_COLUMN].astype(str).to_numpy()
        expected_truth = {
            label: self.prepared.data[label].to_numpy(int)
            for label in self.prepared.target_columns
        }
        summary_by_variant = summary.set_index("variant")
        fear_summary_by_variant = fear_summary.set_index("variant")

        for variant in expected_variants:
            with self.subTest(variant=variant):
                oof = pd.read_csv(root / "oof_predictions" / f"{variant}_oof.csv")
                self.assertEqual(len(oof), 784)
                self.assertEqual(oof["perfume_id"].nunique(), 784)
                np.testing.assert_array_equal(
                    oof["perfume_id"].astype(str).to_numpy(), expected_ids
                )
                np.testing.assert_array_equal(
                    oof["fold_id"].to_numpy(int), expected_folds
                )
                np.testing.assert_array_equal(
                    oof[SOURCE_GROUP_COLUMN].astype(str).to_numpy(), expected_groups
                )
                self.assertEqual(
                    int(oof.groupby(SOURCE_GROUP_COLUMN)["fold_id"].nunique().max()),
                    1,
                )

                f1_values: list[float] = []
                ap_values: list[float] = []
                roc_values: list[float] = []
                for label in self.prepared.target_columns:
                    truth = oof[f"true_{label}"].to_numpy(int)
                    probability = oof[f"prob_{label}"].to_numpy(float)
                    prediction = oof[f"pred_{label}"].to_numpy(int)
                    np.testing.assert_array_equal(truth, expected_truth[label])
                    self.assertTrue(np.isfinite(probability).all())
                    self.assertTrue(((probability >= 0.0) & (probability <= 1.0)).all())
                    np.testing.assert_array_equal(
                        prediction, (probability >= 0.3).astype(int)
                    )
                    f1_values.append(f1_score(truth, prediction, zero_division=0))
                    ap_values.append(average_precision_score(truth, probability))
                    roc_values.append(roc_auc_score(truth, probability))

                saved = summary_by_variant.loc[variant]
                self.assertAlmostEqual(
                    float(saved["macro_f1"]), float(np.mean(f1_values)), places=15
                )
                self.assertAlmostEqual(
                    float(saved["macro_average_precision"]),
                    float(np.mean(ap_values)),
                    places=15,
                )
                self.assertAlmostEqual(
                    float(saved["macro_roc_auc"]),
                    float(np.mean(roc_values)),
                    places=15,
                )
                self.assertEqual(float(saved["threshold"]), 0.3)
                self.assertEqual(int(saved["seed"]), 42)
                self.assertEqual(saved["class_weight"], "balanced")

                included_labels = [
                    label for label in self.prepared.target_columns if label != "fear"
                ]
                fear_saved = fear_summary_by_variant.loc[variant]
                fear_f1 = [
                    f1_score(
                        oof[f"true_{label}"].to_numpy(int),
                        oof[f"pred_{label}"].to_numpy(int),
                        zero_division=0,
                    )
                    for label in included_labels
                ]
                fear_ap = [
                    average_precision_score(
                        oof[f"true_{label}"].to_numpy(int),
                        oof[f"prob_{label}"].to_numpy(float),
                    )
                    for label in included_labels
                ]
                fear_roc = [
                    roc_auc_score(
                        oof[f"true_{label}"].to_numpy(int),
                        oof[f"prob_{label}"].to_numpy(float),
                    )
                    for label in included_labels
                ]
                self.assertEqual(fear_saved["excluded_label"], "fear")
                self.assertEqual(int(fear_saved["included_label_count"]), 6)
                self.assertAlmostEqual(
                    float(fear_saved["macro_f1"]), float(np.mean(fear_f1)), places=15
                )
                self.assertAlmostEqual(
                    float(fear_saved["macro_average_precision"]),
                    float(np.mean(fear_ap)),
                    places=15,
                )
                self.assertAlmostEqual(
                    float(fear_saved["macro_roc_auc"]),
                    float(np.mean(fear_roc)),
                    places=15,
                )

        self.assertEqual(len(interval_table), 45)
        self.assertEqual(len(paired_table), 42)
        self.assertEqual(len(fear_summary), 15)
        self.assertEqual(len(fear_intervals), 45)
        self.assertEqual(len(fear_paired), 42)
        self.assertEqual(len(coefficients), 2030)
        self.assertEqual(len(permutation_summary), 12)
        self.assertEqual(len(permutation_per_label), 84)
        self.assertEqual(len(permutation_per_label_repeats), 2520)
        self.assertEqual(len(fear_permutation_summary), 12)
        self.assertEqual(len(fear_permutation_repeats), 360)
        self.assertEqual(interval_table["variant"].nunique(), 15)
        self.assertEqual(interval_table["metric"].nunique(), 3)
        self.assertEqual(paired_table["comparison_variant"].nunique(), 14)
        self.assertEqual(paired_table["metric"].nunique(), 3)
        self.assertEqual(permutation_summary["family"].nunique(), 6)
        self.assertEqual(permutation_summary["estimator"].nunique(), 2)
        self.assertEqual(permutation_per_label["family"].nunique(), 6)
        self.assertEqual(permutation_per_label["estimator"].nunique(), 2)
        self.assertEqual(permutation_per_label["label"].nunique(), 7)
        self.assertEqual(set(permutation_per_label["repeats"].astype(int)), {30})
        self.assertEqual(permutation_per_label_repeats["family"].nunique(), 6)
        self.assertEqual(permutation_per_label_repeats["estimator"].nunique(), 2)
        self.assertEqual(permutation_per_label_repeats["label"].nunique(), 7)
        self.assertEqual(
            set(permutation_per_label_repeats["repeat"].astype(int)), set(range(1, 31))
        )

        included_permutation = permutation_per_label_repeats.loc[
            permutation_per_label_repeats["label"] != "fear"
        ]
        derived_fear_repeats = (
            included_permutation.groupby(
                ["family", "repeat", "estimator"], sort=False, as_index=False
            )
            .agg(
                macro_average_precision_drop=("average_precision_drop", "mean"),
                macro_f1_drop=("f1_drop", "mean"),
                macro_roc_auc_drop=("roc_auc_drop", "mean"),
            )
        )
        derived_fear_repeats["excluded_label"] = "fear"
        derived_fear_repeats["included_label_count"] = 6
        repeat_columns = fear_permutation_repeats.columns.tolist()
        derived_fear_repeats = derived_fear_repeats[repeat_columns]
        repeat_sort = ["family", "repeat", "estimator"]
        pd.testing.assert_frame_equal(
            fear_permutation_repeats.sort_values(repeat_sort).reset_index(drop=True),
            derived_fear_repeats.sort_values(repeat_sort).reset_index(drop=True),
            check_dtype=False,
            check_exact=False,
            rtol=0.0,
            atol=1e-15,
        )

        derived_fear_summary_rows: list[dict[str, object]] = []
        for (family_name, estimator), group in derived_fear_repeats.groupby(
            ["family", "estimator"], sort=False
        ):
            row: dict[str, object] = {
                "family": family_name,
                "estimator": estimator,
                "excluded_label": "fear",
                "included_label_count": 6,
                "repeats": int(len(group)),
                "importance_definition": "unpermuted_metric_minus_permuted_metric",
            }
            for metric in (
                "macro_average_precision_drop",
                "macro_f1_drop",
                "macro_roc_auc_drop",
            ):
                values = group[metric].to_numpy(float)
                row[f"{metric}_mean"] = float(values.mean())
                row[f"{metric}_std"] = float(values.std(ddof=1))
                row[f"{metric}_q025"] = float(np.quantile(values, 0.025))
                row[f"{metric}_q975"] = float(np.quantile(values, 0.975))
            derived_fear_summary_rows.append(row)
        derived_fear_summary = pd.DataFrame(derived_fear_summary_rows)[
            fear_permutation_summary.columns
        ]
        summary_sort = ["family", "estimator"]
        pd.testing.assert_frame_equal(
            fear_permutation_summary.sort_values(summary_sort).reset_index(drop=True),
            derived_fear_summary.sort_values(summary_sort).reset_index(drop=True),
            check_dtype=False,
            check_exact=False,
            rtol=0.0,
            atol=1e-15,
        )

        expected_components = int(
            self.prepared.data[SOURCE_GROUP_COLUMN].nunique()
        )
        self.assertEqual(expected_components, 359)
        for table in (interval_table, paired_table, fear_intervals, fear_paired):
            self.assertEqual(set(table["bootstrap_reps"].astype(int)), {2000})
            self.assertEqual(set(table["seed"].astype(int)), {42})
            self.assertEqual(
                set(table["n_source_components"].astype(int)), {expected_components}
            )
        self.assertEqual(set(permutation_summary["repeats"].astype(int)), {30})

        self.assertEqual(report["status"], "completed")
        self.assertNotIn("generated_at_utc", report)
        self.assertFalse(report["accepted_benchmark_root_modified"])
        self.assertEqual(report["design"]["variant_count"], 15)
        self.assertEqual(report["design"]["seed"], 42)
        self.assertEqual(report["design"]["threshold"], 0.3)
        self.assertEqual(report["bootstrap"]["reps"], 2000)
        self.assertEqual(report["bootstrap"]["seed"], 42)
        self.assertEqual(
            report["bootstrap"]["n_source_components"], expected_components
        )
        component_id_hash = hashlib.sha256(
            "\n".join(sorted(set(expected_groups))).encode("utf-8")
        ).hexdigest()
        self.assertEqual(
            report["bootstrap"]["interval_method"],
            "percentile interval with linear quantile interpolation",
        )
        self.assertEqual(
            report["bootstrap"]["source_component_ids_sha256"], component_id_hash
        )
        fear_report = report["fear_excluded_sensitivity"]
        self.assertEqual(fear_report["excluded_label"], "fear")
        self.assertEqual(fear_report["fear_positive_support"], 3)
        self.assertEqual(len(fear_report["included_labels"]), 6)
        self.assertNotIn("fear", fear_report["included_labels"])
        self.assertEqual(fear_report["reps"], 2000)
        self.assertEqual(fear_report["seed"], 42)
        self.assertEqual(fear_report["n_source_components"], expected_components)
        self.assertEqual(fear_report["source_component_ids_sha256"], component_id_hash)
        self.assertEqual(
            fear_report["interval_method"], report["bootstrap"]["interval_method"]
        )
        report_fear_permutation = pd.DataFrame(fear_report["permutation_summary"])[
            fear_permutation_summary.columns
        ]
        pd.testing.assert_frame_equal(
            report_fear_permutation.sort_values(summary_sort).reset_index(drop=True),
            fear_permutation_summary.sort_values(summary_sort).reset_index(drop=True),
            check_dtype=False,
            check_exact=False,
            rtol=0.0,
            atol=1e-15,
        )
        self.assertEqual(report["permutation"]["repeats"], 30)
        self.assertEqual(report["permutation"]["seed"], 42)
        reproduction = report["full_hybrid_reproduction_gate"]
        self.assertEqual(reproduction["status"], "passed")
        self.assertEqual(reproduction["truth_mismatches"], 0)
        self.assertEqual(reproduction["prediction_mismatches"], 0)
        self.assertLessEqual(
            float(reproduction["max_probability_difference"]),
            float(reproduction["probability_atol"]),
        )

        output_safety = report["output_safety"]
        self.assertEqual(
            output_safety["supplemental_output_root"],
            "outputs/source_grouped_feature_ablation_v1",
        )
        self.assertEqual(
            output_safety["protected_checksum_manifest_sha256_before"],
            output_safety["protected_checksum_manifest_sha256_after"],
        )
        self.assertEqual(
            output_safety["protected_checksum_manifest_sha256_before"],
            sha256_file(DEFAULT_ACCEPTED_CHECKSUMS),
        )
        self.assertEqual(
            set(report["software"]["code_sha256"]), set(CODE_PROVENANCE_PATHS)
        )
        for name, path in CODE_PROVENANCE_PATHS.items():
            # Historical reports must retain historical code hashes after a
            # reviewed implementation revision; they do not describe current code.
            revisions = json.loads((PROJECT_ROOT / "config/thesis_three_seed_code_revisions.json").read_text())["files"]
            revision = revisions.get(str(path.relative_to(PROJECT_ROOT)))
            expected = revision["historical_sha256"] if revision else sha256_file(path)
            self.assertEqual(report["software"]["code_sha256"][name], expected)

        checksum_path = root / "SHA256SUMS"
        recorded: dict[str, str] = {}
        for line in checksum_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            digest, relative = line.split(maxsplit=1)
            self.assertNotIn(relative, recorded)
            recorded[relative] = digest
        expected_files = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file() and path != checksum_path
        }
        self.assertEqual(set(recorded), expected_files)
        for relative, digest in recorded.items():
            self.assertEqual(sha256_file(root / relative), digest)

    def test_prepare_output_root_rejects_collision_and_nonempty_root(self) -> None:
        with self.assertRaisesRegex(ValueError, "collides with a protected input root"):
            prepare_output_root(
                DEFAULT_ACCEPTED_EXPERIMENT_ROOT / "phase5-output", overwrite=False
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "phase5"
            root.mkdir()
            (root / "unexpected.txt").write_text("data\n", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "non-empty"):
                prepare_output_root(root, overwrite=False)
            with self.assertRaisesRegex(ValueError, "without SHA256SUMS"):
                prepare_output_root(root, overwrite=True)

    def test_prepare_output_root_rejects_checksum_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "phase5"
            root.mkdir()
            payload = root / "owned.txt"
            payload.write_text("current data\n", encoding="utf-8")
            (root / "SHA256SUMS").write_text(
                f"{'0' * 64}  owned.txt\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "do not match SHA256SUMS"):
                prepare_output_root(root, overwrite=True)

    def test_canonical_output_root_rejects_changed_analysis_parameters(self) -> None:
        canonical = {
            "out_dir": DEFAULT_OUT_DIR,
            "config_path": DEFAULT_CONFIG,
            "feature_csv": DEFAULT_FEATURE_CSV,
            "manifest_csv": DEFAULT_MANIFEST,
            "accepted_oof_csv": DEFAULT_ACCEPTED_OOF,
            "accepted_metrics_json": DEFAULT_ACCEPTED_METRICS,
            "tfidf_oof_csv": DEFAULT_TFIDF_OOF,
            "tfidf_metrics_json": DEFAULT_TFIDF_METRICS,
            "bootstrap_reps": BOOTSTRAP_REPS,
            "permutation_repeats": PERMUTATION_REPEATS,
            "seed": SEED,
            "confidence_level": CONFIDENCE_LEVEL,
            "enforce_expected_schema": True,
        }
        validate_canonical_run_configuration(**canonical)
        changed = dict(canonical)
        changed["bootstrap_reps"] = BOOTSTRAP_REPS - 1
        with self.assertRaisesRegex(
            ValueError, "frozen analysis parameters; changed: bootstrap_reps"
        ):
            validate_canonical_run_configuration(**changed)


if __name__ == "__main__":
    unittest.main()

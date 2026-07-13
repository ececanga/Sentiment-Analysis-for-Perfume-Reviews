# Date Created: 2026-07-13
# Date Last Modified: 2026-07-13

"""Focused tests for deterministic source-connected benchmark folds."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.labeling.strict_schema import normalize_perfume_id
from src.training.source_grouped_folds import (
    OFFICIAL_LABELS,
    SOURCE_GROUP_COLUMN,
    build_grouped_inner_validation_manifest,
    build_outer_fold_manifest,
    build_source_groups,
    build_source_memberships,
    load_and_validate_outer_fold_manifest,
    normalize_transcript,
    normalize_video_id,
    prepare_source_grouped_folds,
    sha256_file,
    transcript_sha256,
    validate_grouped_inner_manifest,
    validate_source_disjointness,
)


def _raw_source_frame() -> pd.DataFrame:
    """Seven perfumes forming one three-member component and four singletons."""

    return pd.DataFrame(
        [
            {
                "Perfume": "Alpha Eau de Parfum",
                "Video ID": "AAAAAAAAAAA",
                "Final Transcript": "Alpha opens with a bright citrus note.",
            },
            {
                "Perfume": "Beta Eau de Parfum",
                "Video ID": "AAAAAAAAAAA",
                "Final Transcript": "Beta is soft and calm.",
            },
            {
                "Perfume": "Beta Eau de Parfum",
                "Video ID": "BBBBBBBBBBB",
                "Final Transcript": "A shared bridge transcript!",
            },
            {
                "Perfume": "Gamma Eau de Parfum",
                "Video ID": "CCCCCCCCCCC",
                "Final Transcript": "A SHARED---bridge transcript",
            },
            {
                "Perfume": "Delta Eau de Parfum",
                "Video ID": "DDDDDDDDDDD",
                "Final Transcript": "Delta stands alone.",
            },
            {
                "Perfume": "Epsilon Eau de Parfum",
                "Video ID": "EEEEEEEEEEE",
                "Final Transcript": "Epsilon stands alone.",
            },
            {
                "Perfume": "Zeta Eau de Parfum",
                "Video ID": "FFFFFFFFFFF",
                "Final Transcript": "Zeta stands alone.",
            },
            {
                "Perfume": "Eta Eau de Parfum",
                "Video ID": "GGGGGGGGGGG",
                "Final Transcript": "Eta stands alone.",
            },
        ]
    )


def _canonical_ids(raw_df: pd.DataFrame) -> list[str]:
    return sorted(raw_df["Perfume"].map(normalize_perfume_id).unique().tolist())


def _source_tables(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    memberships, _ = build_source_memberships(
        raw_df,
        canonical_perfume_ids=_canonical_ids(raw_df),
    )
    return build_source_groups(memberships)


def _perfume_group_frame(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    memberships, components = _source_tables(raw_df)
    names = {
        normalize_perfume_id(name): name
        for name in raw_df["Perfume"].drop_duplicates().tolist()
    }
    perfume_groups = (
        memberships[["perfume_id", SOURCE_GROUP_COLUMN, "source_group_size"]]
        .drop_duplicates("perfume_id")
        .sort_values("perfume_id")
        .reset_index(drop=True)
    )
    perfume_groups["Perfume"] = perfume_groups["perfume_id"].map(names)
    return perfume_groups, memberships


def _feature_frame(raw_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for index, perfume_name in enumerate(raw_df["Perfume"].drop_duplicates().tolist()):
        row = {
            "perfume_id": normalize_perfume_id(perfume_name),
            "Perfume": perfume_name,
        }
        for label_index, label in enumerate(OFFICIAL_LABELS):
            row[label] = int((index + label_index) % 3 == 0)
        rows.append(row)
    return pd.DataFrame(rows)


def _write_hash_config(
    path: Path,
    *,
    raw_csv: Path,
    labels_csv: Path,
    feature_label_csv: Path,
    overrides: dict[str, str] | None = None,
) -> None:
    hashes = {
        "raw_csv": sha256_file(raw_csv),
        "labels_csv": sha256_file(labels_csv),
        "feature_label_csv": sha256_file(feature_label_csv),
    }
    hashes.update(overrides or {})
    path.write_text(
        json.dumps({"provenance": {"frozen_input_sha256": hashes}}, indent=2) + "\n",
        encoding="utf-8",
    )


class SourceIdentityTests(unittest.TestCase):
    def test_normalization_and_hash_are_prespecified(self) -> None:
        self.assertEqual(
            normalize_transcript("  Caf&eacute; & JOY---Now\n"),
            "café joy now",
        )
        self.assertEqual(
            transcript_sha256("A SHARED---bridge transcript"),
            transcript_sha256(" a shared bridge TRANSCRIPT! "),
        )
        self.assertEqual(normalize_video_id(" A1b2C3d4E5_ "), "A1b2C3d4E5_")
        with self.assertRaises(ValueError):
            normalize_video_id("too-short")
        self.assertEqual(normalize_video_id("too-short", allow_empty=True), "")

    def test_transitive_components_and_group_ids_ignore_raw_row_order(self) -> None:
        raw_df = _raw_source_frame()
        memberships, components = _source_tables(raw_df)
        reversed_memberships, reversed_components = _source_tables(
            raw_df.iloc[::-1].reset_index(drop=True)
        )

        mapping = memberships.drop_duplicates("perfume_id").set_index("perfume_id")
        alpha = normalize_perfume_id("Alpha Eau de Parfum")
        beta = normalize_perfume_id("Beta Eau de Parfum")
        gamma = normalize_perfume_id("Gamma Eau de Parfum")
        self.assertEqual(mapping.loc[alpha, SOURCE_GROUP_COLUMN], mapping.loc[beta, SOURCE_GROUP_COLUMN])
        self.assertEqual(mapping.loc[beta, SOURCE_GROUP_COLUMN], mapping.loc[gamma, SOURCE_GROUP_COLUMN])
        self.assertEqual(int(mapping.loc[alpha, "source_group_size"]), 3)

        component_contract = [
            SOURCE_GROUP_COLUMN,
            "source_group_size",
            "member_perfume_ids",
            "valid_video_id_count",
            "transcript_sha256_count",
        ]
        pd.testing.assert_frame_equal(
            components[component_contract].reset_index(drop=True),
            reversed_components[component_contract].reset_index(drop=True),
        )
        self.assertTrue(components[SOURCE_GROUP_COLUMN].str.match(r"^sg_[0-9a-f]{64}$").all())
        self.assertEqual(len(components), 5)
        self.assertEqual(
            set(memberships[["perfume_id", SOURCE_GROUP_COLUMN]].itertuples(index=False, name=None)),
            set(
                reversed_memberships[["perfume_id", SOURCE_GROUP_COLUMN]].itertuples(
                    index=False,
                    name=None,
                )
            ),
        )


class GroupedManifestTests(unittest.TestCase):
    def test_outer_and_inner_manifests_are_deterministic_and_group_disjoint(self) -> None:
        raw_df = _raw_source_frame()
        perfume_groups, memberships = _perfume_group_frame(raw_df)
        outer = build_outer_fold_manifest(perfume_groups, n_splits=3)
        outer_again = build_outer_fold_manifest(
            perfume_groups.sample(frac=1.0, random_state=19),
            n_splits=3,
        )
        pd.testing.assert_frame_equal(outer, outer_again)
        self.assertEqual(int(outer.groupby(SOURCE_GROUP_COLUMN)["fold_id"].nunique().max()), 1)

        disjointness = validate_source_disjointness(memberships, outer)
        self.assertTrue(disjointness["validation_passed"])
        self.assertEqual(disjointness["cross_fold_video_id_count"], 0)
        self.assertEqual(disjointness["cross_fold_transcript_hash_count"], 0)

        inner = build_grouped_inner_validation_manifest(
            outer,
            inner_n_splits=2,
            seed=42,
        )
        inner_again = build_grouped_inner_validation_manifest(
            outer,
            inner_n_splits=2,
            seed=42,
        )
        pd.testing.assert_frame_equal(inner, inner_again)
        summary = validate_grouped_inner_manifest(inner, outer)
        self.assertGreater(summary["inner_validation_rows"], 0)

        giant_group = outer.loc[outer["source_group_size"] == 3, SOURCE_GROUP_COLUMN].iloc[0]
        giant_inner = inner.loc[inner[SOURCE_GROUP_COLUMN] == giant_group]
        self.assertTrue((giant_inner["inner_role"] == "train").all())

    def test_validators_reject_split_groups_and_inner_corruption(self) -> None:
        raw_df = _raw_source_frame()
        perfume_groups, _ = _perfume_group_frame(raw_df)
        outer = build_outer_fold_manifest(perfume_groups, n_splits=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "folds.csv"
            outer.to_csv(path, index=False)
            loaded = load_and_validate_outer_fold_manifest(
                path,
                expected_ids=outer["perfume_id"].tolist(),
                n_splits=3,
            )
            self.assertEqual(len(loaded), len(outer))

            corrupt = outer.copy()
            giant_rows = corrupt.index[corrupt["source_group_size"] == 3].tolist()
            current_fold = int(corrupt.loc[giant_rows[0], "fold_id"])
            corrupt.loc[giant_rows[0], "fold_id"] = 1 + (current_fold % 3)
            corrupt.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "source group crosses outer folds"):
                load_and_validate_outer_fold_manifest(
                    path,
                    expected_ids=outer["perfume_id"].tolist(),
                    n_splits=3,
                )

        inner = build_grouped_inner_validation_manifest(outer, inner_n_splits=2, seed=42)
        corrupt_inner = inner.copy()
        giant_group = outer.loc[outer["source_group_size"] == 3, SOURCE_GROUP_COLUMN].iloc[0]
        candidate_rows = corrupt_inner.index[corrupt_inner[SOURCE_GROUP_COLUMN] == giant_group].tolist()
        self.assertGreaterEqual(len(candidate_rows), 2)
        corrupt_inner.loc[candidate_rows[0], "inner_role"] = "val"
        with self.assertRaisesRegex(ValueError, "crosses inner train/val"):
            validate_grouped_inner_manifest(corrupt_inner, outer)


class ArtifactBuilderTests(unittest.TestCase):
    def test_prepare_writes_valid_reproducible_artifacts_and_refuses_overwrite(self) -> None:
        raw_df = _raw_source_frame()
        feature_df = _feature_frame(raw_df)
        labels_df = feature_df[["perfume_id"]].copy()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            raw_csv = tmp / "raw.csv"
            labels_csv = tmp / "labels.csv"
            features_csv = tmp / "features.csv"
            config_path = tmp / "source_grouped.yaml"
            raw_df.to_csv(raw_csv, index=False)
            labels_df.to_csv(labels_csv, index=False)
            feature_df.to_csv(features_csv, index=False)
            _write_hash_config(
                config_path,
                raw_csv=raw_csv,
                labels_csv=labels_csv,
                feature_label_csv=features_csv,
            )

            first_root = tmp / "run_one"
            second_root = tmp / "run_two"
            first = prepare_source_grouped_folds(
                raw_csv=raw_csv,
                labels_csv=labels_csv,
                feature_label_csv=features_csv,
                out_root=first_root,
                config_path=config_path,
                n_splits=3,
                inner_n_splits=2,
                seed=42,
            )
            second = prepare_source_grouped_folds(
                raw_csv=raw_csv,
                labels_csv=labels_csv,
                feature_label_csv=features_csv,
                out_root=second_root,
                n_splits=3,
                inner_n_splits=2,
                seed=42,
            )

            for key in ("memberships", "components", "outer", "inner", "balance", "audit"):
                self.assertTrue(first[key].exists(), key)
            for key in ("memberships", "components", "outer", "inner", "balance"):
                self.assertEqual(sha256_file(first[key]), sha256_file(second[key]), key)

            audit = json.loads(first["audit"].read_text(encoding="utf-8"))
            self.assertEqual(audit["status"], "passed")
            self.assertTrue(audit["validation_passed"])
            self.assertEqual(audit["cross_fold_source_group_count"], 0)
            self.assertEqual(audit["cross_fold_video_id_count"], 0)
            self.assertEqual(audit["cross_fold_transcript_hash_count"], 0)
            self.assertEqual(
                audit["deep_model_input_policy"]["included"],
                ["Final Transcript excerpt"],
            )
            self.assertIn("audit_only_metadata_cross_fold_diagnostic", audit)
            frozen_validation = audit["frozen_input_validation"]
            self.assertTrue(frozen_validation["enforced"])
            self.assertEqual(frozen_validation["config_path"], str(config_path))
            self.assertEqual(frozen_validation["config_sha256"], sha256_file(config_path))
            self.assertEqual(frozen_validation["expected_sha256"], audit["input_sha256"])

            with self.assertRaises(FileExistsError):
                prepare_source_grouped_folds(
                    raw_csv=raw_csv,
                    labels_csv=labels_csv,
                    feature_label_csv=features_csv,
                    out_root=first_root,
                    n_splits=3,
                    inner_n_splits=2,
                    seed=42,
                )

    def test_hash_mismatch_fails_before_creating_outputs(self) -> None:
        raw_df = _raw_source_frame()
        feature_df = _feature_frame(raw_df)
        labels_df = feature_df[["perfume_id"]].copy()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            raw_csv = tmp / "raw.csv"
            labels_csv = tmp / "labels.csv"
            features_csv = tmp / "features.csv"
            config_path = tmp / "source_grouped_bad.yaml"
            out_root = tmp / "must_not_exist"
            raw_df.to_csv(raw_csv, index=False)
            labels_df.to_csv(labels_csv, index=False)
            feature_df.to_csv(features_csv, index=False)
            _write_hash_config(
                config_path,
                raw_csv=raw_csv,
                labels_csv=labels_csv,
                feature_label_csv=features_csv,
                overrides={"raw_csv": "0" * 64},
            )

            with self.assertRaisesRegex(
                ValueError,
                "Frozen input SHA-256 mismatch for raw_csv",
            ):
                prepare_source_grouped_folds(
                    raw_csv=raw_csv,
                    labels_csv=labels_csv,
                    feature_label_csv=features_csv,
                    out_root=out_root,
                    config_path=config_path,
                    n_splits=3,
                    inner_n_splits=2,
                    seed=42,
                )

            self.assertFalse(out_root.exists())


if __name__ == "__main__":
    unittest.main()

"""Tests for the deep-model text dataset builder."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.labeling.strict_schema import normalize_perfume_id
from src.training.build_deep_model_text_dataset import BuildConfig, build_dataset


OFFICIAL_LABELS = ["joy", "sadness", "fear", "trust", "surprise", "anticipation", "disgust"]


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class BuildDeepModelTextDatasetTests(unittest.TestCase):
    def test_builder_writes_canonical_jsonl_and_chunk_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            raw_csv = tmp / "data" / "raw.csv"
            labels_csv = tmp / "labels.csv"
            folds_csv = tmp / "folds.csv"
            output_jsonl = tmp / "outputs" / "perfume_text_dataset_v1.jsonl"
            output_chunks_csv = tmp / "outputs" / "perfume_text_dataset_v1_chunks.csv"
            output_summary_json = tmp / "outputs" / "perfume_text_dataset_v1_summary.json"

            rose = "rose rouge eau de parfum"
            iris = "iris smoke eau de parfum"
            rose_id = normalize_perfume_id(rose)
            iris_id = normalize_perfume_id(iris)

            _write_csv(
                raw_csv,
                [
                    "Perfume",
                    "Video ID",
                    "Title",
                    "Description",
                    "URL",
                    "Final Transcript",
                ],
                [
                    {
                        "Perfume": rose,
                        "Video ID": "vid-002",
                        "Title": "Rose Rouge review",
                        "Description": "A direct review of Rose Rouge.",
                        "URL": "https://example.com/rose-1",
                        "Final Transcript": (
                            "intro opening words for context and setup before we say rose rouge "
                            "smells joyful creamy and bright in the drydown with soft musk"
                        ),
                    },
                    {
                        "Perfume": rose,
                        "Video ID": "vid-003",
                        "Title": "Rose Rouge duplicate review",
                        "Description": "Duplicate transcript should be removed.",
                        "URL": "https://example.com/rose-2",
                        "Final Transcript": (
                            "intro opening words for context and setup before we say rose rouge "
                            "smells joyful creamy and bright in the drydown with soft musk"
                        ),
                    },
                    {
                        "Perfume": iris,
                        "Video ID": "vid-001",
                        "Title": "Obscure smoky floral review",
                        "Description": "",
                        "URL": "https://example.com/iris-1",
                        "Final Transcript": (
                            "first words only with no direct perfume mention but enough content "
                            "to test the prefix fallback branch for excerpt extraction"
                        ),
                    },
                ],
            )

            _write_csv(
                labels_csv,
                [
                    "perfume_id",
                    "perfume_id_hash",
                    "perfume_name",
                    *OFFICIAL_LABELS,
                    "anger",
                    "valence",
                    "arousal",
                    "dominance",
                ],
                [
                    {
                        "perfume_id": rose_id,
                        "perfume_id_hash": "rosehash01",
                        "perfume_name": rose,
                        "joy": 1,
                        "sadness": 0,
                        "fear": 0,
                        "trust": 1,
                        "surprise": 0,
                        "anticipation": 1,
                        "disgust": 0,
                        "anger": 0,
                        "valence": 8.0,
                        "arousal": 6.0,
                        "dominance": 5.5,
                    },
                    {
                        "perfume_id": iris_id,
                        "perfume_id_hash": "irishash02",
                        "perfume_name": iris,
                        "joy": 0,
                        "sadness": 1,
                        "fear": 0,
                        "trust": 0,
                        "surprise": 0,
                        "anticipation": 0,
                        "disgust": 0,
                        "anger": 0,
                        "valence": 4.0,
                        "arousal": 5.0,
                        "dominance": 4.5,
                    },
                ],
            )

            _write_csv(
                folds_csv,
                ["perfume_id", "fold_id"],
                [
                    {"perfume_id": rose_id, "fold_id": 2},
                    {"perfume_id": iris_id, "fold_id": 1},
                ],
            )

            result = build_dataset(
                BuildConfig(
                    raw_csv=raw_csv,
                    labels_csv=labels_csv,
                    folds_csv=folds_csv,
                    output_jsonl=output_jsonl,
                    output_chunks_csv=output_chunks_csv,
                    output_summary_json=output_summary_json,
                    title_max_words=8,
                    description_max_words=8,
                    transcript_window_words=12,
                    transcript_pre_context_words=2,
                )
            )

            self.assertTrue(result["output_jsonl"].exists())
            self.assertTrue(result["output_chunks_csv"].exists())
            self.assertTrue(result["output_summary_json"].exists())

            jsonl_records = [
                json.loads(line)
                for line in output_jsonl.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(jsonl_records), 2)

            rose_record = next(rec for rec in jsonl_records if rec["perfume_id"] == rose_id)
            iris_record = next(rec for rec in jsonl_records if rec["perfume_id"] == iris_id)

            self.assertEqual(rose_record["fold_id"], 2)
            self.assertEqual(rose_record["source_review_count"], 2)
            self.assertEqual(rose_record["deduplicated_review_count"], 1)
            self.assertEqual(rose_record["review_chunks"][0]["match_strategy"], "alias_phrase_window")
            self.assertIn("rose rouge", rose_record["review_chunks"][0]["chunk_text"].lower())

            self.assertEqual(iris_record["fold_id"], 1)
            self.assertEqual(iris_record["source_review_count"], 1)
            self.assertEqual(iris_record["review_chunks"][0]["match_strategy"], "prefix_fallback")
            self.assertIn("first words only", iris_record["review_chunks"][0]["chunk_text"].lower())

            with output_chunks_csv.open("r", encoding="utf-8", newline="") as handle:
                chunk_rows = list(csv.DictReader(handle))
            self.assertEqual(len(chunk_rows), 2)
            self.assertEqual(
                sorted((row["perfume_id"], row["match_strategy"]) for row in chunk_rows),
                sorted(
                    [
                        (iris_id, "prefix_fallback"),
                        (rose_id, "alias_phrase_window"),
                    ]
                ),
            )

            summary = json.loads(output_summary_json.read_text(encoding="utf-8"))
            self.assertEqual(summary["n_perfumes"], 2)
            self.assertEqual(summary["n_total_chunks"], 2)
            self.assertEqual(summary["deduplicated_transcript_rows_dropped"], 1)


if __name__ == "__main__":
    unittest.main()

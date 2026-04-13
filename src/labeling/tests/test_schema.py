"""Minimal tests for strict schema and snippet aggregation."""

from __future__ import annotations

import unittest

from src.labeling.aggregate_snippets import aggregate_emotions_majority, aggregate_vad_mean
from src.labeling.strict_schema import validate_label_payload


class SchemaTests(unittest.TestCase):
    def test_validate_success(self) -> None:
        payload = {
            "joy": 1,
            "sadness": 0,
            "anger": 0,
            "fear": 0,
            "trust": 1,
            "surprise": 0,
            "anticipation": 1,
            "disgust": 0,
            "valence": 8,
            "arousal": 7.5,
            "dominance": 6,
        }
        out = validate_label_payload(payload)
        self.assertEqual(out["joy"], 1)
        self.assertAlmostEqual(out["arousal"], 7.5)

    def test_validate_missing_key(self) -> None:
        payload = {
            "joy": 1,
            "sadness": 0,
            "anger": 0,
            "fear": 0,
            "trust": 1,
            "surprise": 0,
            "anticipation": 1,
            "valence": 8,
            "arousal": 7,
            "dominance": 6,
        }
        with self.assertRaises(ValueError):
            validate_label_payload(payload)

    def test_majority_tie_defaults_zero(self) -> None:
        labels = [
            {
                "joy": 1,
                "sadness": 0,
                "anger": 0,
                "fear": 0,
                "trust": 0,
                "surprise": 0,
                "anticipation": 0,
                "disgust": 0,
                "valence": 5.0,
                "arousal": 5.0,
                "dominance": 5.0,
            },
            {
                "joy": 0,
                "sadness": 0,
                "anger": 0,
                "fear": 0,
                "trust": 1,
                "surprise": 0,
                "anticipation": 0,
                "disgust": 0,
                "valence": 5.0,
                "arousal": 5.0,
                "dominance": 5.0,
            },
        ]
        agg = aggregate_emotions_majority(labels)
        self.assertEqual(agg["joy"], 0)
        self.assertEqual(agg["trust"], 0)

    def test_vad_mean(self) -> None:
        labels = [
            {
                "joy": 0,
                "sadness": 0,
                "anger": 0,
                "fear": 0,
                "trust": 0,
                "surprise": 0,
                "anticipation": 0,
                "disgust": 0,
                "valence": 2.0,
                "arousal": 4.0,
                "dominance": 6.0,
            },
            {
                "joy": 0,
                "sadness": 0,
                "anger": 0,
                "fear": 0,
                "trust": 0,
                "surprise": 0,
                "anticipation": 0,
                "disgust": 0,
                "valence": 4.0,
                "arousal": 6.0,
                "dominance": 8.0,
            },
        ]
        agg = aggregate_vad_mean(labels, round_ndigits=2)
        self.assertEqual(agg["valence"], 3.0)
        self.assertEqual(agg["arousal"], 5.0)
        self.assertEqual(agg["dominance"], 7.0)


if __name__ == "__main__":
    unittest.main()


"""Tests for thesis workflow orchestrator selection/command logic."""

from __future__ import annotations

import unittest
from pathlib import Path

from src.pipeline.run_thesis_workflow import (
    StageCommand,
    build_command,
    build_stage_registry,
    select_stages,
)


class RunThesisWorkflowTests(unittest.TestCase):
    def test_stage_registry_contains_expected_order(self) -> None:
        stages = build_stage_registry()
        names = [s.name for s in stages]
        expected = [
            "scrape",
            "format",
            "lexicon_vad",
            "transformers",
            "context_terms",
            "top_terms",
            "reprocess_optional",
            "fusion_merge",
            "fusion_aggregate",
            "gemini_labeling",
            "gemini_agreement",
            "supervised_train_eval",
            "feature_importance",
            "evaluation_reports",
        ]
        self.assertEqual(names, expected)

    def test_default_selection_excludes_scrape(self) -> None:
        stages = build_stage_registry()
        selected = select_stages(stages=stages)
        names = [s.name for s in selected]
        self.assertNotIn("scrape", names)
        self.assertIn("format", names)
        self.assertIn("feature_importance", names)

    def test_only_and_skip_selection(self) -> None:
        stages = build_stage_registry()
        selected = select_stages(
            stages=stages,
            only=["feature_importance", "gemini_agreement", "supervised_train_eval"],
            skip=["supervised_train_eval"],
        )
        self.assertEqual([s.name for s in selected], ["gemini_agreement", "feature_importance"])

    def test_from_to_selection(self) -> None:
        stages = build_stage_registry()
        selected = select_stages(
            stages=stages,
            from_stage="fusion_aggregate",
            to_stage="feature_importance",
        )
        self.assertEqual(
            [s.name for s in selected],
            [
                "fusion_aggregate",
                "gemini_labeling",
                "gemini_agreement",
                "supervised_train_eval",
                "feature_importance",
            ],
        )

    def test_build_command(self) -> None:
        cmd = build_command(
            stage_command=StageCommand("src/analysis/analyze_stage8_feature_importance.py", ("--dry-run",)),
            python_executable="/usr/bin/python3",
            project_root=Path("/repo"),
        )
        self.assertEqual(
            cmd,
            [
                "/usr/bin/python3",
                "/repo/src/analysis/analyze_stage8_feature_importance.py",
                "--dry-run",
            ],
        )


if __name__ == "__main__":
    unittest.main()

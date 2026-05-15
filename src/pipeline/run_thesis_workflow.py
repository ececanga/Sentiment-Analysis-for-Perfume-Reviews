# Date Created: 2026-04-13 23:02:44+03:00
# Date Last Modified: 2026-05-15

"""Top-level thesis workflow orchestrator.

This runner is additive and does not replace `run_pipeline.py`.
It chains the legacy/base stages and newer post-base stages with explicit
stage selection controls.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from src.utils.paths import PROJECT_ROOT


@dataclass(frozen=True)
class StageCommand:
    """Single script invocation inside a stage."""

    script_relpath: str
    args: Tuple[str, ...] = ()


@dataclass(frozen=True)
class StageSpec:
    """Workflow stage definition."""

    name: str
    description: str
    commands: Tuple[StageCommand, ...]
    enabled_by_default: bool = True
    is_manual_optional: bool = False


def build_stage_registry() -> List[StageSpec]:
    """Create ordered thesis workflow stages."""

    return [
        StageSpec(
            name="scrape",
            description="Scrape YouTube metadata + transcripts (manual/optional in current project).",
            commands=(StageCommand("src/preprocessing/perfume_emotion_scraper.py"),),
            enabled_by_default=False,
            is_manual_optional=True,
        ),
        StageSpec(
            name="format",
            description="Normalize transcript formatting into immutable formatted CSV.",
            commands=(StageCommand("src/preprocessing/format_transcripts.py"),),
        ),
        StageSpec(
            name="lexicon_vad",
            description="Run NRC emotion counting and VAD lexicon mapping.",
            commands=(StageCommand("src/lexicon/apply_emotion_vad_lexicons.py"),),
        ),
        StageSpec(
            name="transformers",
            description="Run all configured transformer emotion inference scripts.",
            commands=(
                StageCommand("src/transformers/analyze_emotions_distilroberta_jhartmann.py"),
                StageCommand("src/transformers/analyze_emotions_distilbert_savani.py"),
                StageCommand("src/transformers/analyze_emotions_goemotions_multi.py"),
                StageCommand("src/transformers/analyze_emotions_roberta_cardiffnlp.py"),
            ),
        ),
        StageSpec(
            name="context_terms",
            description="Extract context keywords around emotion anchors.",
            commands=(StageCommand("src/descriptive_terms/extract_context_terms.py"),),
        ),
        StageSpec(
            name="top_terms",
            description="Analyze top context terms and optional intensifier counts.",
            commands=(StageCommand("src/descriptive_terms/analyze_top_terms.py"),),
        ),
        StageSpec(
            name="reprocess_optional",
            description="Optional scent/emotion lexicon enrichment pass.",
            commands=(StageCommand("src/lexicon/reprocess_scent_emotion.py"),),
            is_manual_optional=True,
        ),
        StageSpec(
            name="fusion_merge",
            description="Merge aggregated lexicon and transformer label-level outputs.",
            commands=(StageCommand("src/aggregation_fusion/merge_emotion_profiles.py"),),
        ),
        StageSpec(
            name="fusion_aggregate",
            description="Aggregate full numeric feature families into unified profile.",
            commands=(StageCommand("src/aggregation_fusion/aggregate_and_merge_emotions.py"),),
        ),
        StageSpec(
            name="gemini_labeling",
            description="Run strict Gemini perfume-level labeling with resumability.",
            commands=(StageCommand("src/labeling/run_gemini_labeling.py"),),
            is_manual_optional=True,
        ),
        StageSpec(
            name="gemini_agreement",
            description="Join unified profile with Gemini labels and produce agreement tables.",
            commands=(StageCommand("src/analysis/analyze_gemini_unified.py"),),
        ),
        StageSpec(
            name="supervised_train_eval",
            description="Train Stage 7 supervised baselines on merged Gemini+unified data.",
            commands=(StageCommand("src/training/train_eval_gemini784.py"),),
        ),
        StageSpec(
            name="feature_importance",
            description="Run Stage 8A held-out permutation importance analysis.",
            commands=(StageCommand("src/analysis/analyze_stage8_feature_importance.py"),),
        ),
        StageSpec(
            name="evaluation_reports",
            description="Generate legacy agreement/visual evaluation reports and figures.",
            commands=(StageCommand("src/evaluation/evaluate_models.py"),),
            is_manual_optional=True,
        ),
    ]


def _build_stage_index(stages: Sequence[StageSpec]) -> Dict[str, int]:
    return {stage.name: i for i, stage in enumerate(stages)}


def validate_stage_names(
    *,
    stages: Sequence[StageSpec],
    names: Iterable[str],
    arg_name: str,
) -> None:
    known = {s.name for s in stages}
    bad = [n for n in names if n not in known]
    if bad:
        raise ValueError(f"Unknown stage(s) for {arg_name}: {bad}. Use --list-stages to inspect valid names.")


def select_stages(
    *,
    stages: Sequence[StageSpec],
    only: Sequence[str] | None = None,
    from_stage: str | None = None,
    to_stage: str | None = None,
    skip: Sequence[str] | None = None,
) -> List[StageSpec]:
    """Select stages from ordered registry according to CLI filters."""

    if only and (from_stage or to_stage):
        raise ValueError("--only cannot be combined with --from/--to.")

    stage_index = _build_stage_index(stages)
    stage_by_name = {s.name: s for s in stages}

    validate_stage_names(stages=stages, names=only or [], arg_name="--only")
    validate_stage_names(stages=stages, names=skip or [], arg_name="--skip")
    validate_stage_names(stages=stages, names=[from_stage] if from_stage else [], arg_name="--from")
    validate_stage_names(stages=stages, names=[to_stage] if to_stage else [], arg_name="--to")

    if only:
        selected_names = [s.name for s in stages if s.name in set(only)]
    else:
        all_names = [s.name for s in stages]
        if from_stage or to_stage:
            lo = stage_index[from_stage] if from_stage else 0
            hi = stage_index[to_stage] if to_stage else len(stages) - 1
            if lo > hi:
                raise ValueError(f"--from {from_stage} appears after --to {to_stage}.")
            selected_names = all_names[lo : hi + 1]
        else:
            selected_names = [s.name for s in stages if s.enabled_by_default]

    skip_set = set(skip or [])
    selected_names = [n for n in selected_names if n not in skip_set]
    return [stage_by_name[n] for n in selected_names]


def build_command(
    *,
    stage_command: StageCommand,
    python_executable: str,
    project_root: Path,
) -> List[str]:
    """Build subprocess command list for one script call."""

    script_path = project_root / stage_command.script_relpath
    return [python_executable, str(script_path), *stage_command.args]


def _banner(text: str) -> str:
    return f"\n{'=' * 88}\n{text}\n{'=' * 88}"


def run_stage(
    *,
    stage: StageSpec,
    python_executable: str,
    project_root: Path,
    dry_run: bool,
) -> None:
    """Execute all commands in a stage."""

    print(_banner(f"[START] stage={stage.name}"))
    print(f"Description: {stage.description}")
    if stage.is_manual_optional:
        print("Note: this stage is marked optional/manual in current project usage.")

    for idx, stage_cmd in enumerate(stage.commands, start=1):
        cmd = build_command(
            stage_command=stage_cmd,
            python_executable=python_executable,
            project_root=project_root,
        )
        pretty = shlex.join(cmd)
        print(f"  ({idx}/{len(stage.commands)}) {pretty}")
        if dry_run:
            continue
        subprocess.run(cmd, cwd=project_root, check=True)

    print(_banner(f"[DONE] stage={stage.name}"))


def list_stages(stages: Sequence[StageSpec]) -> None:
    """Print ordered stage list with defaults and optional flags."""

    print("Ordered thesis workflow stages:")
    for i, s in enumerate(stages, start=1):
        default_flag = "default" if s.enabled_by_default else "not-default"
        optional_flag = "optional/manual" if s.is_manual_optional else "core"
        print(f"{i:>2}. {s.name:<22} [{default_flag}; {optional_flag}]")
        print(f"    {s.description}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run full thesis workflow with explicit stage selection.",
    )
    ap.add_argument("--list-stages", action="store_true", help="List ordered stages and exit.")
    ap.add_argument("--only", nargs="+", default=None, help="Run only selected stage name(s).")
    ap.add_argument("--from", dest="from_stage", default=None, help="Run from this stage (inclusive).")
    ap.add_argument("--to", dest="to_stage", default=None, help="Run to this stage (inclusive).")
    ap.add_argument("--skip", nargs="+", default=None, help="Skip stage name(s) from the selected run.")
    ap.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    ap.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue to next stage if a stage fails (default: fail fast).",
    )
    ap.add_argument(
        "--python-executable",
        default=sys.executable,
        help="Python executable used to invoke stage scripts (default: current interpreter).",
    )
    ap.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Project root path (default: repository root).",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    stages = build_stage_registry()

    if args.list_stages:
        list_stages(stages)
        return 0

    try:
        selected = select_stages(
            stages=stages,
            only=args.only,
            from_stage=args.from_stage,
            to_stage=args.to_stage,
            skip=args.skip,
        )
    except ValueError as exc:
        print(f"Selection error: {exc}")
        return 2

    if not selected:
        print("No stages selected. Use --list-stages to inspect valid stage names.")
        return 0

    print(_banner("Thesis Workflow"))
    print(f"Project root: {args.project_root}")
    print(f"Python executable: {args.python_executable}")
    print(f"Dry run: {args.dry_run}")
    print("Stage plan:")
    for s in selected:
        print(f"  - {s.name}")

    failures: List[Tuple[str, str]] = []
    for stage in selected:
        try:
            run_stage(
                stage=stage,
                python_executable=args.python_executable,
                project_root=args.project_root,
                dry_run=args.dry_run,
            )
        except subprocess.CalledProcessError as exc:
            msg = f"Command failed with exit code {exc.returncode}"
            failures.append((stage.name, msg))
            print(_banner(f"[FAILED] stage={stage.name} :: {msg}"))
            if not args.continue_on_error:
                break
        except Exception as exc:  # noqa: BLE001
            msg = f"{type(exc).__name__}: {exc}"
            failures.append((stage.name, msg))
            print(_banner(f"[FAILED] stage={stage.name} :: {msg}"))
            if not args.continue_on_error:
                break

    if failures:
        print("\nWorkflow finished with failures:")
        for stage_name, msg in failures:
            print(f"  - {stage_name}: {msg}")
        return 1

    print(_banner("Workflow completed successfully"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

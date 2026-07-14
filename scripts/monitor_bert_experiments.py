#!/usr/bin/env python3
"""Display a quota-free live dashboard for the BERT experiment suite."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo


DEFAULT_OUTPUT_ROOT = Path("outputs/bert_source_grouped_experiments_v1")
LOCAL_ZONE = ZoneInfo("Europe/Istanbul")
STALE_AFTER_SECONDS = 15 * 60


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monitor the source-grouped BERT experiments from progress files."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Experiment output directory (default: {DEFAULT_OUTPUT_ROOT}).",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=10.0,
        help="Refresh interval in seconds (default: 10).",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Print one snapshot and exit.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_summary(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def local_timestamp(utc_value: object) -> str:
    if not utc_value:
        return "unknown"
    parsed = datetime.fromisoformat(str(utc_value).replace("Z", "+00:00"))
    return parsed.astimezone(LOCAL_ZONE).strftime("%Y-%m-%d %H:%M:%S %Z")


def update_age_seconds(utc_value: object) -> float | None:
    if not utc_value:
        return None
    parsed = datetime.fromisoformat(str(utc_value).replace("Z", "+00:00"))
    return max(0.0, (datetime.now(timezone.utc) - parsed).total_seconds())


def human_age(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    rounded = int(seconds)
    minutes, secs = divmod(rounded, 60)
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def current_step(progress: dict[str, object]) -> str:
    fold = progress.get("fold")
    fold_total = progress.get("fold_total")
    epoch = progress.get("epoch")
    if fold is None:
        return "Preparing the next condition"
    if epoch is None:
        action = "initializing / training epoch 1/3"
    elif int(epoch) < 3:
        action = f"training epoch {int(epoch) + 1}/3 (last completed: {epoch}/3)"
    else:
        action = "finalizing fold predictions"
    return f"Fold {fold}/{fold_total} - {action}"


def progress_bar(completed: int, total: int, width: int = 28) -> str:
    filled = round(width * completed / total) if total else 0
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def metric(value: str) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "n/a"


def render(progress: dict[str, object], summary: list[dict[str, str]]) -> str:
    total = int(progress.get("condition_total") or 0)
    completed = len(summary)
    condition_index = int(progress.get("condition_index") or min(completed + 1, total))
    age = update_age_seconds(progress.get("updated_utc"))
    status = str(progress.get("status", "unknown")).upper()
    if status == "COMPLETE":
        stage = "Completed"
        condition_display = f"{total}/{total}  All conditions completed"
        step_display = "No training remains"
    else:
        stage = str(progress.get("stage", "unknown")).replace("_", " ").title()
        condition = str(progress.get("condition_id", "unknown"))
        condition_display = f"{condition_index}/{total}  {condition}"
        step_display = current_step(progress)

    lines = [
        "BERT SOURCE-GROUPED EXPERIMENT MONITOR",
        "=" * 58,
        f"Status:             {status}",
        f"Device:             {str(progress.get('device', 'unknown')).upper()}",
        f"Last update:        {local_timestamp(progress.get('updated_utc'))}",
        f"Update age:         {human_age(age)}",
        "",
        f"Overall:            {progress_bar(completed, total)} {completed}/{total} complete",
        f"Current condition:  {condition_display}",
        f"Stage:              {stage}",
        f"Current step:       {step_display}",
        "",
        f"Stage ETA:          {progress.get('stage_eta', 'unknown')}",
        f"Stage finish:       {local_timestamp(progress.get('stage_estimated_finish_utc'))}",
        f"Full-suite ETA:     {progress.get('experiment_eta', 'unknown')}",
        f"Estimated finish:   {local_timestamp(progress.get('experiment_estimated_finish_utc'))}",
    ]

    if age is not None and age > STALE_AFTER_SECONDS and status == "RUNNING":
        lines.extend(
            [
                "",
                "WARNING: No progress update for more than 15 minutes.",
                "Check the training process before restarting anything.",
            ]
        )

    lines.extend(["", "COMPLETED CONDITIONS", "-" * 58])
    if not summary:
        lines.append("None yet.")
    else:
        lines.append(f"{'#':>2}  {'Condition':<34} {'F1':>6} {'ROC':>6} {'PR':>6}")
        for index, row in enumerate(summary, start=1):
            condition_name = row.get("condition_id", "unknown")
            if len(condition_name) > 34:
                condition_name = condition_name[:31] + "..."
            lines.append(
                f"{index:>2}  {condition_name:<34} "
                f"{metric(row.get('macro_f1', '')):>6} "
                f"{metric(row.get('macro_roc_auc', '')):>6} "
                f"{metric(row.get('macro_pr_auc', '')):>6}"
            )

    now = datetime.now(LOCAL_ZONE).strftime("%Y-%m-%d %H:%M:%S %Z")
    lines.extend(
        [
            "",
            "-" * 58,
            f"Dashboard refreshed: {now}",
            "This monitor only reads files; it does not affect training.",
            "Press Ctrl-C to close the dashboard (training will continue).",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    progress_path = args.output_root / "progress.json"
    summary_path = args.output_root / "experiment_summary.csv"

    while True:
        try:
            progress = read_json(progress_path)
            summary = read_summary(summary_path)
            output = render(progress, summary)
        except (OSError, json.JSONDecodeError) as exc:
            output = f"Waiting for readable progress files...\n\n{exc}"
            progress = {}

        if args.once:
            print(output)
            return 0

        title = "BERT experiments - " + str(progress.get("status", "waiting"))
        sys.stdout.write(f"\033]0;{title}\007\033[2J\033[H{output}\n")
        sys.stdout.flush()

        if progress.get("status") in {"complete", "failed"}:
            sys.stdout.write("\a\nMonitor finished. The terminal may now be closed.\n")
            sys.stdout.flush()
            return 0 if progress.get("status") == "complete" else 1
        time.sleep(max(1.0, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())

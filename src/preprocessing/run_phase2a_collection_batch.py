# Date Created: 2026-06-01
# Date Last Modified: 2026-06-01

"""Run capped Phase 2A YouTube collection batches across a local key pool."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.format_transcripts import format_transcripts
from src.preprocessing.perfume_emotion_scraper import (
    QUERY_TEMPLATES,
    ScraperConfig,
    build_queries,
    collect_for_candidate,
    get_youtube_transcript,
    key_fingerprint,
    load_candidate_queue,
    load_status_maps,
    load_youtube_key_pool,
    normalize_name_key,
    select_candidates,
    split_status_values,
    update_status,
    write_raw_rows_incremental,
)
from src.utils.paths import data_path


DEFAULT_QUEUE_V2 = PROJECT_ROOT / "outputs/gemini_judge/judge_g25pro_2k_v1/planning/phase2a_next_candidate_queue_v2.csv"
DEFAULT_QUEUE_V3 = PROJECT_ROOT / "outputs/gemini_judge/judge_g25pro_2k_v1/planning/phase2a_next_candidate_queue_v3.csv"
DEFAULT_QUEUE = DEFAULT_QUEUE_V2
DEFAULT_SUMMARY_JSON = PROJECT_ROOT / "outputs/gemini_judge/judge_g25pro_2k_v1/planning/phase2a_collection_run_summary.json"
DEFAULT_RETRY_STATUSES = {"quota_error", "auth_error", "youtube_api_error", "api_error"}
IP_BLOCK_RECOMMENDATION = (
    "Stop collection. Wait 4-6 hours or switch network. Run transcript diagnostic before next batch."
)
TRANSCRIPT_DIAGNOSTIC_COMMAND = (
    "python3 src/preprocessing/run_phase2a_collection_batch.py --diagnose-transcript-error YXcy3En2gCA"
)


def count_csv_rows(path: Path) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    with path.open(encoding="utf-8", errors="replace", newline="") as handle:
        rows = list(csv.DictReader(handle))
    perfumes = {
        str(row.get("Perfume", "")).strip()
        for row in rows
        if str(row.get("Perfume", "")).strip()
    }
    return len(rows), len(perfumes)


def count_statuses(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", errors="replace", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return dict(Counter(str(row.get("Status", "")).strip() for row in rows))


def transcript_error_category(error: str) -> str:
    text = str(error or "")
    lowered = text.lower()
    if "ipblocked" in text or "requestblocked" in text or "blocked" in lowered or "too many requests" in lowered:
        return "blocking_or_rate_limit"
    if "notranscriptfound" in text:
        return "no_transcript_found"
    if "transcriptsdisabled" in text:
        return "transcripts_disabled"
    if "attributeerror" in lowered or "typeerror" in lowered:
        return "library_api_mismatch_or_bug"
    if "could not retrieve" in lowered:
        return "transcript_retrieval_error"
    if not text.strip():
        return "blank"
    return text.split(":", 1)[0].strip() or "unknown"


def diagnose_transcript_error(video_id: str) -> int:
    transcript, error = get_youtube_transcript(video_id, max_retries=1)
    if transcript:
        print("transcript_diagnostic: success")
        print("category=ok")
        print(f"word_count={len(transcript.split())}")
        return 0
    print("transcript_diagnostic: failure")
    print(f"category={transcript_error_category(error)}")
    print(f"reason={str(error).replace(chr(10), ' ')[:500]}")
    return 1


def resolve_retry_statuses(args: argparse.Namespace) -> set[str]:
    retry_statuses = split_status_values(args.retry_status) if args.retry_status else set(DEFAULT_RETRY_STATUSES)
    if args.retry_transcript_errors and args.pace != "old":
        retry_statuses.add("transcript_fetch_error")
    return retry_statuses


def recommendation_for_stop(stop_reason: str) -> str:
    if stop_reason in {
        "blocking_or_rate_limit",
        "transcript_fetch_error_threshold",
        "transcript_fetch_error_rate",
        "no_raw_rows_after_three_attempts",
    }:
        return IP_BLOCK_RECOMMENDATION
    return ""


def init_run_state(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "pace": args.pace,
        "attempted": 0,
        "written": 0,
        "done": 0,
        "transcript_fetch_error": 0,
        "stop_reason": "",
        "recommendation": "",
        "key_results": [],
    }


def old_pace_stop_reason(state: dict[str, Any], status: str, last_error: str) -> str:
    if status == "transcript_fetch_error" and transcript_error_category(last_error) == "blocking_or_rate_limit":
        return "blocking_or_rate_limit"
    if state["transcript_fetch_error"] >= 3:
        return "transcript_fetch_error_threshold"
    if state["attempted"] >= 3 and state["transcript_fetch_error"] / max(state["attempted"], 1) >= 0.5:
        return "transcript_fetch_error_rate"
    if state["attempted"] >= 3 and state["written"] == 0:
        return "no_raw_rows_after_three_attempts"
    return ""


def write_summary_json(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pace": state["pace"],
        "attempted": state["attempted"],
        "written": state["written"],
        "done_count": state["done"],
        "transcript_fetch_error_count": state["transcript_fetch_error"],
        "stop_reason": state["stop_reason"],
        "recommendation": state["recommendation"],
        "key_results": state["key_results"],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_config(args: argparse.Namespace, *, api_key: str, key_label: str, retry_statuses: set[str]) -> ScraperConfig:
    return ScraperConfig(
        target_csv=data_path("perfume_level", "final_perfume_data.csv"),
        perfume_col="Name",
        candidate_queue_csv=args.candidate_queue_csv,
        raw_output_csv=args.raw_output_csv,
        status_csv=args.status_csv,
        seed_status_csv=args.seed_status_csv,
        batch_size=args.per_key_limit,
        limit=args.per_key_limit,
        resume=True,
        retry_statuses=retry_statuses,
        dry_run=args.dry_run,
        max_results=args.max_results,
        max_workers=args.max_workers,
        min_words=args.min_words,
        sleep_sec=args.sleep_sec,
        max_query_templates_per_perfume=args.max_query_templates_per_perfume,
        stop_after_valid_videos_per_perfume=args.stop_after_valid_videos_per_perfume,
        query_templates=QUERY_TEMPLATES,
        use_brand_queries=True,
        api_key=api_key,
        api_key_env=args.api_key_env,
        key_label=key_label,
        verbose=False,
        debug_quota=args.debug_quota,
    )


def format_if_possible(raw_output_csv: Path, formatted_output_csv: Path) -> None:
    if raw_output_csv.exists():
        format_transcripts(raw_output_csv, formatted_output_csv)


def run_for_key(
    args: argparse.Namespace,
    *,
    key_label: str,
    api_key: str,
    retry_statuses: set[str],
    run_state: dict[str, Any],
) -> dict[str, Any]:
    cfg = build_config(args, api_key=api_key, key_label=key_label, retry_statuses=retry_statuses)
    candidates = load_candidate_queue(cfg.candidate_queue_csv) if cfg.candidate_queue_csv else []
    to_process = select_candidates(candidates, cfg)

    attempted = 0
    rows_written = 0
    batch_statuses: Counter[str] = Counter()
    stopped_on = ""

    if args.dry_run:
        return {
            "key_label": key_label,
            "planned": len(to_process),
            "attempted": 0,
            "written": 0,
            "batch_statuses": {},
            "stopped_on": "",
            "queries_first_candidate": build_queries(to_process[0], cfg) if to_process else (),
            "query_count_first_candidate": len(build_queries(to_process[0], cfg)) if to_process else 0,
            "key_fingerprint": key_fingerprint(api_key),
        }

    for candidate in to_process:
        previous = load_status_maps([cfg.status_csv]).get(normalize_name_key(candidate.perfume), {})
        attempts = int(previous.get("Attempts", 0) or 0) + 1
        status, valid_rows, last_error = collect_for_candidate(candidate, cfg)
        written = write_raw_rows_incremental(cfg.raw_output_csv, valid_rows)
        update_status(cfg.status_csv, candidate.perfume, status, attempts, last_error)

        attempted += 1
        rows_written += written
        batch_statuses[status] += 1
        run_state["attempted"] += 1
        run_state["written"] += written
        if status == "done":
            run_state["done"] += 1
        if status == "transcript_fetch_error":
            run_state["transcript_fetch_error"] += 1

        if status in {"quota_error", "auth_error"}:
            stopped_on = status
            break
        if args.pace == "old":
            stop_reason = old_pace_stop_reason(run_state, status, last_error)
            if stop_reason:
                stopped_on = stop_reason
                run_state["stop_reason"] = stop_reason
                run_state["recommendation"] = recommendation_for_stop(stop_reason)
                break

    format_if_possible(cfg.raw_output_csv, args.formatted_output_csv)
    return {
        "key_label": key_label,
        "planned": len(to_process),
        "attempted": attempted,
        "written": rows_written,
        "batch_statuses": dict(batch_statuses),
        "stopped_on": stopped_on,
        "key_fingerprint": key_fingerprint(api_key),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run safe capped Phase 2A collection batches across YouTube keys.")
    parser.add_argument("--youtube-key-file", type=Path, default=PROJECT_ROOT / ".env.youtube_keys")
    parser.add_argument("--candidate-queue-csv", type=Path, default=None)
    parser.add_argument("--per-key-limit", type=int, default=None)
    parser.add_argument("--max-keys", type=int, default=None)
    parser.add_argument("--old-pace", action="store_true", help="Use conservative old 784-style Phase 2A pacing.")
    parser.add_argument("--pace", choices=("normal", "old"), default="normal")
    parser.add_argument("--api-key-env", default="YOUTUBE_API_KEY")
    parser.add_argument("--raw-output-csv", type=Path, default=data_path("raw", "YouTube_Perfume_Reviews_Phase2A_Partial.csv"))
    parser.add_argument(
        "--formatted-output-csv",
        type=Path,
        default=data_path("raw", "YouTube_Perfume_Reviews_Phase2A_FullyFormatted.csv"),
    )
    parser.add_argument("--status-csv", type=Path, default=data_path("intermediate", "phase2a_processed_perfumes.csv"))
    parser.add_argument("--seed-status-csv", type=Path, default=data_path("intermediate", "processed_perfumes.csv"))
    parser.add_argument("--retry-status", action="append", default=[])
    parser.add_argument(
        "--retry-transcript-errors",
        action="store_true",
        help="Opt in to retrying transcript_fetch_error candidates across keys. Default skips them.",
    )
    parser.add_argument(
        "--diagnose-transcript-error",
        metavar="VIDEO_ID",
        help="Fetch transcript for one video ID only. Does not use YouTube search or API keys.",
    )
    parser.add_argument("--max-results", type=int, default=None)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--min-words", type=int, default=20)
    parser.add_argument("--sleep-sec", type=float, default=None)
    parser.add_argument(
        "--max-query-templates-per-perfume",
        type=int,
        default=None,
        help="Limit search.list query templates per perfume. Phase 2A default uses the two strongest brand-review queries.",
    )
    parser.add_argument(
        "--stop-after-valid-videos-per-perfume",
        type=int,
        default=None,
        help="Stop searching/fetching a perfume after this many valid transcript videos.",
    )
    parser.add_argument(
        "--debug-quota",
        action="store_true",
        help="Print safe per-query quota diagnostics without exposing API key values.",
    )
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.old_pace:
        args.pace = "old"
    if args.candidate_queue_csv is None:
        args.candidate_queue_csv = DEFAULT_QUEUE_V3 if args.pace == "old" else DEFAULT_QUEUE
    if args.per_key_limit is None:
        args.per_key_limit = 3 if args.pace == "old" else 25
    if args.max_keys is None and args.pace == "old":
        args.max_keys = 1
    if args.max_results is None:
        args.max_results = 5
    if args.max_workers is None:
        args.max_workers = 1 if args.pace == "old" else 10
    if args.sleep_sec is None:
        args.sleep_sec = 30.0 if args.pace == "old" else 1.0
    if args.max_query_templates_per_perfume is None:
        args.max_query_templates_per_perfume = 2
    if args.stop_after_valid_videos_per_perfume is None:
        args.stop_after_valid_videos_per_perfume = 2 if args.pace == "old" else 5
    return args


def main() -> None:
    args = parse_args()
    if args.diagnose_transcript_error:
        raise SystemExit(diagnose_transcript_error(args.diagnose_transcript_error))
    if args.per_key_limit < 1:
        raise ValueError("--per-key-limit must be at least 1")
    if args.pace == "old" and args.retry_transcript_errors:
        print("old_pace_notice=transcript_fetch_error retries are disabled in old-pace mode")

    keys = load_youtube_key_pool(
        api_key_env=args.api_key_env,
        youtube_key_file=args.youtube_key_file,
        max_keys=args.max_keys,
    )
    retry_statuses = resolve_retry_statuses(args)
    run_state = init_run_state(args)

    print(f"pace={args.pace}")
    print(f"keys_found={len(keys)}")
    print(f"per_key_limit={args.per_key_limit}")
    print(f"retry_statuses={sorted(retry_statuses)}")
    print(f"candidate_queue_csv={args.candidate_queue_csv}")
    print(f"max_results={args.max_results}")
    print(f"max_query_templates_per_perfume={args.max_query_templates_per_perfume}")
    print(f"stop_after_valid_videos_per_perfume={args.stop_after_valid_videos_per_perfume}")
    print(f"debug_quota={args.debug_quota}")
    if args.pace == "old":
        print(f"transcript_preflight_recommendation={TRANSCRIPT_DIAGNOSTIC_COMMAND}")
    for item in keys:
        if run_state["stop_reason"]:
            break
        result = run_for_key(
            args,
            key_label=item.label,
            api_key=item.value,
            retry_statuses=retry_statuses,
            run_state=run_state,
        )
        run_state["key_results"].append(
            {
                "key_label": result["key_label"],
                "key_fingerprint": result["key_fingerprint"],
                "planned": result["planned"],
                "attempted": result["attempted"],
                "written": result["written"],
                "statuses": result["batch_statuses"],
                "stopped_on": result["stopped_on"],
            }
        )
        print(
            f"key_label={result['key_label']} key_fingerprint={result['key_fingerprint']} planned={result['planned']} "
            f"attempted={result['attempted']} written={result['written']} "
            f"statuses={result['batch_statuses']} stopped_on={result['stopped_on'] or 'none'}"
        )
        if args.dry_run and result["queries_first_candidate"]:
            print(f"first_candidate_query_count={result['query_count_first_candidate']}")
            print(f"first_candidate_queries={result['queries_first_candidate']}")
    if run_state["stop_reason"]:
        print(f"stop_reason={run_state['stop_reason']}")
        if run_state["recommendation"]:
            print(run_state["recommendation"])

    raw_rows, raw_unique = count_csv_rows(args.raw_output_csv)
    formatted_rows, formatted_unique = count_csv_rows(args.formatted_output_csv)
    print(f"status_counts={count_statuses(args.status_csv)}")
    print(f"raw_rows={raw_rows} raw_unique_perfumes={raw_unique}")
    print(f"formatted_rows={formatted_rows} formatted_unique_perfumes={formatted_unique}")
    if args.pace == "old" and not args.dry_run:
        write_summary_json(args.summary_json, run_state)
        print(f"summary_json={args.summary_json}")


if __name__ == "__main__":
    main()

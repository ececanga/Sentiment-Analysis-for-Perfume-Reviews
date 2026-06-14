# Date Created: 2026-06-12
# Date Last Modified: 2026-06-12

"""Run strict Gemini labeling over review-level synthetic perfume reviews."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.labeling.gemini_client import GeminiLabelConfig, GeminiLabelingError, GeminiStrictLabelClient
from src.labeling.strict_schema import EMOTIONS, VAD_DIMS, validate_label_payload
from src.utils.paths import PROJECT_ROOT, labels_path


@dataclass(frozen=True)
class ReviewLevelRunConfig:
    input_csv: Path
    model: str
    temperature: float
    max_retries: int
    backoff_base_sec: float
    backoff_cap_sec: float
    rate_limit_sleep_sec: float
    output_tag: str
    limit: int | None
    env_file: Path | None


class RunLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a", encoding="utf-8")

    def log(self, message: str) -> None:
        line = f"[{datetime.now(timezone.utc).isoformat()}] {message}"
        print(line)
        self.handle.write(line + "\n")
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


def stringify_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def clean_text(value: object) -> str:
    return " ".join(str(value or "").replace("\r", " ").replace("\n", " ").split())


def load_env_file(path: Path | None) -> list[str]:
    if path is None or not path.exists():
        return []
    loaded: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded


def output_paths(tag: str) -> dict[str, Path]:
    out_dir = labels_path("gemini_review_level")
    return {
        "jsonl": out_dir / f"gemini_review_labels_{tag}.jsonl",
        "csv": out_dir / f"gemini_review_labels_{tag}.csv",
        "pipeline_csv": out_dir / f"gemini_review_labels_{tag}_pipeline.csv",
        "report": out_dir / f"gemini_review_labeling_report_{tag}.json",
        "errors": out_dir / f"gemini_review_labeling_errors_{tag}.jsonl",
        "run_log": out_dir / f"gemini_review_labeling_run_{tag}.log",
    }


def load_reviews(path: Path, limit: int | None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    csv.field_size_limit(sys.maxsize)
    df = pd.read_csv(path)
    required = {"synthetic_review_id", "perfume_id", "Perfume", "Final Transcript"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Input CSV missing required columns: {missing}")
    df = df.copy()
    df["synthetic_review_id"] = df["synthetic_review_id"].map(clean_text)
    df["perfume_id"] = df["perfume_id"].map(clean_text)
    df["Perfume"] = df["Perfume"].map(clean_text)
    df["Final Transcript"] = df["Final Transcript"].map(clean_text)
    if "review_index" not in df.columns:
        df["review_index"] = df["synthetic_review_id"].astype(str).str.rsplit("-", n=1).str[-1]
    if df["Final Transcript"].eq("").any():
        raise ValueError("Input CSV contains empty Final Transcript rows.")
    if not df["synthetic_review_id"].is_unique:
        raise ValueError("synthetic_review_id must be unique.")
    if limit is not None:
        df = df.head(limit).copy()
    return df.reset_index(drop=True)


def append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(obj, ensure_ascii=False) + "\n")
        handle.flush()


def load_existing_records(path: Path) -> tuple[list[dict[str, Any]], set[str]]:
    if not path.exists():
        return [], set()
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            review_id = str(record.get("synthetic_review_id", "")).strip()
            if not review_id or review_id in seen:
                continue
            validate_label_payload(record["label"])
            records.append(record)
            seen.add(review_id)
    return records, seen


def record_to_csv_row(record: dict[str, Any]) -> dict[str, Any]:
    label = validate_label_payload(record["label"])
    return {
        "synthetic_review_id": record["synthetic_review_id"],
        "source_type": record.get("source_type", "synthetic"),
        "perfume_id": record["perfume_id"],
        "perfume_name": record["perfume_name"],
        "review_index": record.get("review_index", ""),
        **{column: label[column] for column in EMOTIONS + VAD_DIMS},
    }


def write_csv_outputs(records: list[dict[str, Any]], csv_path: Path, pipeline_path: Path) -> None:
    fieldnames = [
        "synthetic_review_id",
        "source_type",
        "perfume_id",
        "perfume_name",
        "review_index",
        *EMOTIONS,
        *VAD_DIMS,
    ]
    rows = [record_to_csv_row(record) for record in records]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    pipeline_rows = []
    for row in rows:
        pipeline_rows.append(
            {
                "synthetic_review_id": row["synthetic_review_id"],
                "perfume_id": row["perfume_id"],
                "perfume_name": row["perfume_name"],
                **{column: row[column] for column in EMOTIONS + VAD_DIMS},
            }
        )
    pipeline_fieldnames = ["synthetic_review_id", "perfume_id", "perfume_name", *EMOTIONS, *VAD_DIMS]
    with pipeline_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=pipeline_fieldnames)
        writer.writeheader()
        writer.writerows(pipeline_rows)


def build_report(
    *,
    cfg: ReviewLevelRunConfig,
    paths: dict[str, Path],
    target_ids: list[str],
    attempted_ids: list[str],
    skipped_ids: list[str],
    failed_ids: list[str],
    records: list[dict[str, Any]],
    loaded_env_keys: list[str],
) -> dict[str, Any]:
    labels = [validate_label_payload(record["label"]) for record in records]
    label_df = pd.DataFrame(labels) if labels else pd.DataFrame(columns=EMOTIONS + VAD_DIMS)
    emotion_counts = {emotion: int(label_df[emotion].sum()) for emotion in EMOTIONS} if labels else {}
    vad_summary = (
        {
            dim: {
                "mean": float(label_df[dim].mean()),
                "min": float(label_df[dim].min()),
                "max": float(label_df[dim].max()),
            }
            for dim in VAD_DIMS
        }
        if labels
        else {}
    )
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_csv": stringify_path(cfg.input_csv),
        "model": cfg.model,
        "temperature": cfg.temperature,
        "target_review_count": len(target_ids),
        "completed_review_count": len(records),
        "attempted_review_count": len(attempted_ids),
        "skipped_existing_count": len(skipped_ids),
        "failed_review_count": len(set(failed_ids)),
        "failed_review_ids": sorted(set(failed_ids)),
        "emotion_positive_counts": emotion_counts,
        "vad_summary": vad_summary,
        "loaded_env_keys": loaded_env_keys,
        "outputs": {key: stringify_path(value) for key, value in paths.items()},
        "note": "Review-level Gemini labels. These are pseudo-labels from gemini-2.5-pro for comparability with the OG judge run.",
    }


def parse_args() -> ReviewLevelRunConfig:
    parser = argparse.ArgumentParser(description="Run strict Gemini review-level labeling.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "synthetic_review_level_tests"
        / "emotion_rich_v2_pilot100"
        / "annotation"
        / "review_level_annotation_template.csv",
    )
    parser.add_argument("--model", type=str, default="gemini-2.5-pro")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--backoff-base-sec", type=float, default=1.0)
    parser.add_argument("--backoff-cap-sec", type=float, default=32.0)
    parser.add_argument("--rate-limit-sleep-sec", type=float, default=0.5)
    parser.add_argument("--output-tag", type=str, default="emotion_rich_v2_pilot100_g25pro_v1")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--env-file", type=Path, default=PROJECT_ROOT / ".env")
    args = parser.parse_args()
    return ReviewLevelRunConfig(
        input_csv=args.input_csv,
        model=args.model,
        temperature=args.temperature,
        max_retries=args.max_retries,
        backoff_base_sec=args.backoff_base_sec,
        backoff_cap_sec=args.backoff_cap_sec,
        rate_limit_sleep_sec=args.rate_limit_sleep_sec,
        output_tag=args.output_tag,
        limit=args.limit,
        env_file=args.env_file,
    )


def main() -> None:
    cfg = parse_args()
    loaded_env_keys = load_env_file(cfg.env_file)
    paths = output_paths(cfg.output_tag)
    logger = RunLogger(paths["run_log"])
    logger.log("Starting Gemini review-level labeling run.")
    logger.log(f"Input CSV: {cfg.input_csv}")
    logger.log(f"Model: {cfg.model}")
    if loaded_env_keys:
        safe_keys = [key for key in loaded_env_keys if "KEY" not in key and "TOKEN" not in key]
        logger.log(f"Loaded env keys from file: {safe_keys if safe_keys else '[secret keys hidden]'}")

    reviews = load_reviews(cfg.input_csv, cfg.limit)
    target_ids = reviews["synthetic_review_id"].astype(str).tolist()
    paths["jsonl"].parent.mkdir(parents=True, exist_ok=True)
    paths["jsonl"].touch(exist_ok=True)

    records, done_ids = load_existing_records(paths["jsonl"])
    logger.log(f"Review targets: {len(target_ids)}")
    logger.log(f"Existing successful records found: {len(done_ids)}")

    gemini_cfg = GeminiLabelConfig(
        model_name=cfg.model,
        temperature=cfg.temperature,
        max_retries=cfg.max_retries,
        backoff_base_sec=cfg.backoff_base_sec,
        backoff_cap_sec=cfg.backoff_cap_sec,
        rate_limit_sleep_sec=cfg.rate_limit_sleep_sec,
    )
    client: GeminiStrictLabelClient | None = None
    attempted_ids: list[str] = []
    skipped_ids: list[str] = []
    failed_ids: list[str] = []

    for idx, row in reviews.iterrows():
        review_id = str(row["synthetic_review_id"])
        if review_id in done_ids:
            skipped_ids.append(review_id)
            continue

        attempted_ids.append(review_id)
        try:
            if client is None:
                client = GeminiStrictLabelClient(config=gemini_cfg)
            result = client.label_snippet(
                perfume_name=str(row["Perfume"]),
                snippet_text=str(row["Final Transcript"]),
            )
            label = validate_label_payload(result.label)
            record = {
                "synthetic_review_id": review_id,
                "source_type": str(row.get("source_type", "synthetic")),
                "perfume_id": str(row["perfume_id"]),
                "perfume_name": str(row["Perfume"]),
                "review_index": int(row["review_index"]) if str(row["review_index"]).isdigit() else row["review_index"],
                "review_text": str(row["Final Transcript"]),
                "label": label,
                "attempts": result.attempts,
                "raw_response": result.raw_response,
                "metadata": {
                    "model": cfg.model,
                    "temperature": cfg.temperature,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "prompt_protocol": "STRICT_PROMPT_RULES",
                    "unit": "single_synthetic_review",
                },
            }
            append_jsonl(paths["jsonl"], record)
            records.append(record)
            done_ids.add(review_id)
            logger.log(f"[{idx + 1}/{len(reviews)}] OK {review_id}")
        except (GeminiLabelingError, Exception) as exc:  # noqa: BLE001
            failed_ids.append(review_id)
            append_jsonl(
                paths["errors"],
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "synthetic_review_id": review_id,
                    "perfume_id": str(row["perfume_id"]),
                    "perfume_name": str(row["Perfume"]),
                    "stage": "gemini_review_labeling",
                    "error": str(exc),
                    "attempts": getattr(exc, "attempts", None),
                    "last_error": getattr(exc, "last_error", None),
                    "last_response": getattr(exc, "last_response", None),
                },
            )
            logger.log(f"[{idx + 1}/{len(reviews)}] FAIL {review_id} error={exc}")
            break

    records = [record for record in records if record.get("synthetic_review_id") in set(target_ids)]
    write_csv_outputs(records, paths["csv"], paths["pipeline_csv"])
    report = build_report(
        cfg=cfg,
        paths=paths,
        target_ids=target_ids,
        attempted_ids=attempted_ids,
        skipped_ids=skipped_ids,
        failed_ids=failed_ids,
        records=records,
        loaded_env_keys=loaded_env_keys,
    )
    paths["report"].write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    logger.log(f"Wrote JSONL: {paths['jsonl']}")
    logger.log(f"Wrote CSV: {paths['csv']}")
    logger.log(f"Wrote pipeline CSV: {paths['pipeline_csv']}")
    logger.log(f"Wrote report: {paths['report']}")
    logger.log("Run complete.")
    logger.close()

    if failed_ids:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

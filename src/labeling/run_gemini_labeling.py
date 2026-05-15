# Date Created: 2026-04-13 23:02:44+03:00
# Date Last Modified: 2026-05-15

"""Run strict Gemini labeling over perfume-level targets with resumability."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from src.labeling.aggregate_snippets import aggregate_perfume
from src.labeling.gemini_client import GeminiLabelConfig, GeminiLabelingError, GeminiStrictLabelClient
from src.labeling.reporting import build_labeling_report, flatten_records_to_csv_rows, print_distribution_summary
from src.labeling.strict_schema import (
    EMOTIONS,
    VAD_DIMS,
    normalize_perfume_id,
    perfume_name_hash10,
    perfume_name_hash4,
    validate_label_payload,
)
from src.utils.paths import PROJECT_ROOT, data_path, labels_path


@dataclass(frozen=True)
class RunConfig:
    target_csv: Path
    packet_csv: Path
    raw_csv: Path
    perfume_col: str
    text_col: str
    model: str
    temperature: float
    max_retries: int
    backoff_base_sec: float
    backoff_cap_sec: float
    rate_limit_sleep_sec: float
    batch_size: int
    seed: int
    snippet_max_chars: int
    output_tag: str
    limit: Optional[int]
    dry_run: bool


class RunLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = self.path.open("a", encoding="utf-8")

    def log(self, message: str) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        line = f"[{ts}] {message}"
        print(line)
        self.fh.write(line + "\n")
        self.fh.flush()

    def close(self) -> None:
        self.fh.close()


def normalize_name_key(name: str) -> str:
    text = str(name or "").strip().lower()
    return re.sub(r"\s+", " ", text)


def clean_and_shorten(text: str, max_chars: int) -> str:
    out = str(text or "").replace("\n", " ").replace("\r", " ").strip()
    out = " ".join(out.split())
    if len(out) > max_chars:
        return out[:max_chars] + "..."
    return out


def stable_seed(seed: int, perfume_id: str) -> int:
    hex_seed = hashlib.sha1(f"{seed}:{perfume_id}".encode("utf-8")).hexdigest()[:8]
    return int(hex_seed, 16)


def read_csv_dicts(path: Path) -> List[Dict[str, str]]:
    csv.field_size_limit(sys.maxsize)
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        return list(csv.DictReader(f))


def build_target_entries(target_csv: Path, perfume_col: str) -> List[Dict[str, str]]:
    rows = read_csv_dicts(target_csv)
    dedup: Dict[str, str] = {}
    for row in rows:
        raw_name = str(row.get(perfume_col, "")).strip()
        key = normalize_name_key(raw_name)
        if raw_name and key and key not in dedup:
            dedup[key] = raw_name

    entries = [{"perfume_key": k, "perfume_name": v} for k, v in dedup.items()]

    slug_groups: Dict[str, List[Dict[str, str]]] = {}
    for e in entries:
        slug = normalize_perfume_id(e["perfume_name"])
        slug_groups.setdefault(slug, []).append(e)

    used_ids: set[str] = set()
    for slug, group in slug_groups.items():
        if len(group) == 1:
            perfume_id = slug
            if perfume_id in used_ids:
                perfume_id = f"{slug}-{perfume_name_hash4(group[0]['perfume_name'])}"
            group[0]["perfume_id"] = perfume_id
            used_ids.add(perfume_id)
            continue

        for e in group:
            candidate = f"{slug}-{perfume_name_hash4(e['perfume_name'])}"
            if candidate in used_ids:
                candidate = f"{slug}-{perfume_name_hash10(e['perfume_name'])[:6]}"
            e["perfume_id"] = candidate
            used_ids.add(candidate)

    for e in entries:
        e["perfume_id_hash"] = perfume_name_hash10(e["perfume_name"])

    entries.sort(key=lambda x: x["perfume_id"])
    return entries


def load_packet_snippets(
    packet_csv: Path,
    perfume_col: str,
    max_chars: int,
    logger: RunLogger,
) -> Dict[str, List[str]]:
    if not packet_csv.exists():
        logger.log(f"Packet CSV not found (optional), fallback to raw: {packet_csv}")
        return {}

    rows = read_csv_dicts(packet_csv)
    if not rows:
        return {}
    cols = list(rows[0].keys())
    snippet_cols = sorted(
        [c for c in cols if c.startswith("Snippet_")],
        key=lambda c: int(c.split("_", 1)[1]) if c.split("_", 1)[1].isdigit() else c,
    )
    if not snippet_cols:
        logger.log(f"No Snippet_* columns found in packet; fallback to raw: {packet_csv}")
        return {}

    out: Dict[str, List[str]] = {}
    for row in rows:
        raw_name = str(row.get(perfume_col, "")).strip()
        key = normalize_name_key(raw_name)
        if not key:
            continue
        snippets = [clean_and_shorten(str(row.get(c, "")), max_chars) for c in snippet_cols[:3]]
        snippets = [s for s in snippets if s]
        out[key] = snippets
    logger.log(f"Loaded packet snippets for {len(out)} perfumes from {packet_csv}")
    return out


def load_raw_reviews(
    raw_csv: Path,
    perfume_col: str,
    text_col: str,
    max_chars: int,
) -> Dict[str, List[str]]:
    rows = read_csv_dicts(raw_csv)
    out: Dict[str, List[str]] = {}
    for row in rows:
        name = str(row.get(perfume_col, "")).strip()
        text = clean_and_shorten(str(row.get(text_col, "")), max_chars)
        if not name or not text:
            continue
        key = normalize_name_key(name)
        out.setdefault(key, []).append(text)
    return out


def select_three_snippets(snippets: Sequence[str], perfume_id: str, seed: int) -> List[str]:
    clean = [s for s in snippets if str(s).strip()]
    k = len(clean)
    if k == 0:
        raise ValueError("no_text")
    if k == 1:
        return [clean[0], clean[0], clean[0]]
    if k == 2:
        rng = random.Random(stable_seed(seed, perfume_id))
        return [clean[rng.randrange(2)] for _ in range(3)]
    if k == 3:
        return clean[:3]

    rng = random.Random(stable_seed(seed, perfume_id))
    idxs = rng.sample(range(k), 3)
    return [clean[i] for i in idxs]


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()


def load_existing_jsonl(path: Path, logger: RunLogger) -> Tuple[List[Dict[str, Any]], set[str], Dict[str, str]]:
    if not path.exists():
        return [], set(), {}
    records: List[Dict[str, Any]] = []
    seen: set[str] = set()
    id_to_hash: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            perfume_id = str(rec.get("perfume_id", "")).strip()
            perfume_hash = str(rec.get("perfume_id_hash", "")).strip()
            if not perfume_id:
                continue
            if perfume_id in id_to_hash and perfume_hash and id_to_hash[perfume_id] != perfume_hash:
                logger.log(
                    f"WARNING collision detected in existing JSONL for perfume_id='{perfume_id}': "
                    f"{id_to_hash[perfume_id]} vs {perfume_hash}"
                )
            if perfume_id not in seen:
                seen.add(perfume_id)
                records.append(rec)
                if perfume_hash:
                    id_to_hash[perfume_id] = perfume_hash
    return records, seen, id_to_hash


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["perfume_id", "perfume_id_hash", "perfume_name"] + EMOTIONS + VAD_DIMS
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def validate_output_files(jsonl_path: Path, csv_path: Path) -> None:
    ids_jsonl: set[str] = set()
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            perfume_id = str(rec.get("perfume_id", "")).strip()
            if not perfume_id:
                raise ValueError("JSONL record missing perfume_id")
            if perfume_id in ids_jsonl:
                raise ValueError(f"Duplicate perfume_id in JSONL: {perfume_id}")
            validate_label_payload(rec["aggregated_labels"])
            ids_jsonl.add(perfume_id)

    ids_csv: set[str] = set()
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            perfume_id = str(row.get("perfume_id", "")).strip()
            if not perfume_id:
                raise ValueError("CSV row missing perfume_id")
            if perfume_id in ids_csv:
                raise ValueError(f"Duplicate perfume_id in CSV: {perfume_id}")
            payload = {k: row.get(k) for k in EMOTIONS + VAD_DIMS}
            validate_label_payload(payload)
            ids_csv.add(perfume_id)

    if ids_jsonl != ids_csv:
        raise ValueError("Mismatch between JSONL and CSV perfume_id sets")


def resolve_output_paths(tag: str, dry_run: bool) -> Dict[str, Path]:
    if dry_run:
        labels_debug_dir = PROJECT_ROOT / "labels" / "debug"
        logs_debug_dir = PROJECT_ROOT / "logs" / "debug"
        return {
            "jsonl": labels_debug_dir / f"gemini_labels_debug_{tag}.jsonl",
            "csv": labels_debug_dir / f"gemini_labels_debug_{tag}.csv",
            "report": labels_debug_dir / f"gemini_labeling_report_debug_{tag}.json",
            "errors": logs_debug_dir / f"gemini_labeling_errors_debug_{tag}.jsonl",
            "run_log": logs_debug_dir / f"gemini_labeling_run_debug_{tag}.log",
        }
    return {
        "jsonl": labels_path(f"gemini_labels_784_{tag}.jsonl"),
        "csv": labels_path(f"gemini_labels_784_{tag}.csv"),
        "report": labels_path(f"gemini_labeling_report_{tag}.json"),
        "errors": labels_path(f"gemini_labeling_errors_{tag}.jsonl"),
        "run_log": labels_path(f"gemini_labeling_run_{tag}.log"),
    }


def parse_args() -> RunConfig:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--target-csv",
        type=Path,
        default=data_path("perfume_level", "Unified_Emotion_Profile_Perfume.csv"),
        help="Target perfume-level CSV (default: Unified_Emotion_Profile_Perfume.csv)",
    )
    ap.add_argument(
        "--packet-csv",
        type=Path,
        default=data_path(
            "annotation_packets",
            "Annotation_Packet_TOSKOVAT_ORTO_PARISI_ELO_NASOMATTO_SERGE_LUTENS.csv",
        ),
        help="Optional packet CSV with Snippet_1..Snippet_3; fallback to raw for missing perfumes or missing file.",
    )
    ap.add_argument(
        "--raw-csv",
        type=Path,
        default=data_path("raw", "YouTube_Perfume_Reviews_FullyFormatted.csv"),
        help="Raw review CSV used for snippet fallback/generation.",
    )
    ap.add_argument("--perfume-col", type=str, default="Perfume")
    ap.add_argument("--text-col", type=str, default="Final Transcript")
    ap.add_argument("--model", type=str, default="gemini-2.5-flash")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-retries", type=int, default=5)
    ap.add_argument("--backoff-base-sec", type=float, default=1.0)
    ap.add_argument("--backoff-cap-sec", type=float, default=32.0)
    ap.add_argument("--rate-limit-sleep-sec", type=float, default=0.5)
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--snippet-max-chars", type=int, default=650)
    ap.add_argument("--output-tag", type=str, default="v1")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    return RunConfig(
        target_csv=args.target_csv,
        packet_csv=args.packet_csv,
        raw_csv=args.raw_csv,
        perfume_col=args.perfume_col,
        text_col=args.text_col,
        model=args.model,
        temperature=args.temperature,
        max_retries=args.max_retries,
        backoff_base_sec=args.backoff_base_sec,
        backoff_cap_sec=args.backoff_cap_sec,
        rate_limit_sleep_sec=args.rate_limit_sleep_sec,
        batch_size=args.batch_size,
        seed=args.seed,
        snippet_max_chars=args.snippet_max_chars,
        output_tag=args.output_tag,
        limit=args.limit,
        dry_run=args.dry_run,
    )


def main() -> None:
    cfg = parse_args()
    out_paths = resolve_output_paths(cfg.output_tag, cfg.dry_run)
    logger = RunLogger(out_paths["run_log"])
    logger.log("Starting Gemini strict labeling run.")
    logger.log(f"Dry run: {cfg.dry_run}")

    if not cfg.target_csv.exists():
        raise FileNotFoundError(f"Target CSV not found: {cfg.target_csv}")
    if not cfg.raw_csv.exists():
        raise FileNotFoundError(f"Raw CSV not found: {cfg.raw_csv}")

    targets = build_target_entries(cfg.target_csv, cfg.perfume_col)
    if cfg.limit is not None:
        targets = targets[: cfg.limit]
    logger.log(f"Target perfumes selected: {len(targets)}")

    packet_map = load_packet_snippets(cfg.packet_csv, cfg.perfume_col, cfg.snippet_max_chars, logger)
    raw_map = load_raw_reviews(cfg.raw_csv, cfg.perfume_col, cfg.text_col, cfg.snippet_max_chars)
    logger.log(f"Raw perfume groups loaded: {len(raw_map)}")

    out_paths["jsonl"].parent.mkdir(parents=True, exist_ok=True)
    out_paths["jsonl"].touch(exist_ok=True)

    all_records, done_ids, done_hashes = load_existing_jsonl(out_paths["jsonl"], logger)
    logger.log(f"Existing successful records found: {len(done_ids)}")

    gem_cfg = GeminiLabelConfig(
        model_name=cfg.model,
        temperature=cfg.temperature,
        max_retries=cfg.max_retries,
        backoff_base_sec=cfg.backoff_base_sec,
        backoff_cap_sec=cfg.backoff_cap_sec,
        rate_limit_sleep_sec=cfg.rate_limit_sleep_sec,
    )
    client: Optional[GeminiStrictLabelClient] = None

    target_ids = [t["perfume_id"] for t in targets]
    attempted_ids: List[str] = []
    failed_ids: List[str] = []
    skipped_ids: List[str] = []

    for idx, target in enumerate(targets, start=1):
        perfume_id = target["perfume_id"]
        perfume_name = target["perfume_name"]
        perfume_key = target["perfume_key"]
        perfume_id_hash = target["perfume_id_hash"]

        if perfume_id in done_ids:
            skipped_ids.append(perfume_id)
            if perfume_id in done_hashes and done_hashes[perfume_id] != perfume_id_hash:
                logger.log(
                    f"WARNING hash mismatch for existing record perfume_id='{perfume_id}': "
                    f"{done_hashes[perfume_id]} vs {perfume_id_hash}"
                )
            continue

        attempted_ids.append(perfume_id)
        source = "packet" if perfume_key in packet_map else "raw"
        source_snippets = packet_map.get(perfume_key) if source == "packet" else raw_map.get(perfume_key, [])

        try:
            snippets = select_three_snippets(source_snippets, perfume_id, cfg.seed)
        except ValueError as exc:
            failed_ids.append(perfume_id)
            append_jsonl(
                out_paths["errors"],
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "perfume_id": perfume_id,
                    "perfume_id_hash": perfume_id_hash,
                    "perfume_name": perfume_name,
                    "stage": "snippet_selection",
                    "reason": str(exc),
                },
            )
            logger.log(f"[{idx}/{len(targets)}] FAIL {perfume_id} reason={exc}")
            continue

        snippet_outputs: List[Dict[str, Any]] = []
        failed = False
        for sidx, snippet_text in enumerate(snippets, start=1):
            try:
                if client is None:
                    client = GeminiStrictLabelClient(config=gem_cfg)
                res = client.label_snippet(perfume_name=perfume_name, snippet_text=snippet_text)
                snippet_outputs.append(
                    {
                        "snippet_index": sidx,
                        "source": source,
                        "snippet_text": snippet_text,
                        "label": res.label,
                        "attempts": res.attempts,
                        "raw_response": res.raw_response,
                    }
                )
            except GeminiLabelingError as exc:
                failed = True
                failed_ids.append(perfume_id)
                append_jsonl(
                    out_paths["errors"],
                    {
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "perfume_id": perfume_id,
                        "perfume_id_hash": perfume_id_hash,
                        "perfume_name": perfume_name,
                        "stage": "gemini_labeling",
                        "snippet_index": sidx,
                        "attempts": exc.attempts,
                        "last_error": exc.last_error,
                        "last_response": exc.last_response,
                    },
                )
                logger.log(f"[{idx}/{len(targets)}] FAIL {perfume_id} snippet={sidx} error={exc.last_error}")
                break

        if failed:
            continue

        aggregated = aggregate_perfume([s["label"] for s in snippet_outputs])
        aggregated = validate_label_payload(aggregated)

        record = {
            "perfume_id": perfume_id,
            "perfume_id_hash": perfume_id_hash,
            "perfume_name": perfume_name,
            "snippet_outputs": snippet_outputs,
            "aggregated_labels": aggregated,
            "metadata": {
                "model": cfg.model,
                "temperature": cfg.temperature,
                "source_policy": "packet_if_exists_else_raw",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            },
        }
        append_jsonl(out_paths["jsonl"], record)
        all_records.append(record)
        done_ids.add(perfume_id)
        done_hashes[perfume_id] = perfume_id_hash

        if idx % cfg.batch_size == 0 or idx == len(targets):
            logger.log(f"Progress: {idx}/{len(targets)} processed")

    rows = flatten_records_to_csv_rows([r for r in all_records if r.get("perfume_id") in set(target_ids)])
    write_csv(out_paths["csv"], rows)
    validate_output_files(out_paths["jsonl"], out_paths["csv"])

    report = build_labeling_report(
        target_ids=target_ids,
        attempted_ids=attempted_ids,
        skipped_ids=skipped_ids,
        failed_ids=failed_ids,
        records=all_records,
    )
    out_paths["report"].parent.mkdir(parents=True, exist_ok=True)
    out_paths["report"].write_text(json.dumps(report, indent=2), encoding="utf-8")

    print_distribution_summary(report)
    logger.log(f"Wrote JSONL: {out_paths['jsonl']}")
    logger.log(f"Wrote CSV: {out_paths['csv']}")
    logger.log(f"Wrote report: {out_paths['report']}")
    logger.log(f"Wrote errors: {out_paths['errors']}")
    logger.log("Run complete.")
    logger.close()


if __name__ == "__main__":
    main()

"""Audit the bounded YouTube annotation packets and sample their source traceability."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.labeling.prepare_og784_browser_labeling_packets import clean_text
from src.utils.paths import PROJECT_ROOT


DEFAULT_INPUT = (
    PROJECT_ROOT
    / "outputs/gemini_judge/judge_g25pro_v1/model_benchmark/datasets/"
    "perfume_text_dataset_v1.jsonl"
)
DEFAULT_PACKET_DIR = (
    PROJECT_ROOT / "data/annotation_packets/og784_gemini_browser_pro_review_level"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/annotation_packet_provenance_audit_v1"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def parse_packet_items(packet_dir: Path) -> dict[str, dict[str, Any]]:
    manifest_path = packet_dir / "manifest.csv"
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        manifest = list(csv.DictReader(handle))

    batch_files = sorted({row["batch_file"] for row in manifest})
    parsed: dict[str, dict[str, Any]] = {}
    for filename in batch_files:
        lines = (packet_dir / filename).read_text(encoding="utf-8").splitlines()
        item: dict[str, Any] | None = None
        for line in lines:
            if line == "BEGIN_ITEM":
                if item is not None:
                    raise ValueError(f"Nested BEGIN_ITEM in {filename}")
                item = {"batch_file": filename, "excerpts": []}
                continue
            if line == "END_ITEM":
                if item is None or "item_id" not in item:
                    raise ValueError(f"Malformed item in {filename}")
                if item["item_id"] in parsed:
                    raise ValueError(f"Duplicate packet item: {item['item_id']}")
                parsed[item["item_id"]] = item
                item = None
                continue
            if item is None or ": " not in line:
                continue
            key, value = line.split(": ", 1)
            if key.startswith("review_excerpt_"):
                item["excerpts"].append(value)
            else:
                item[key] = value
        if item is not None:
            raise ValueError(f"Unclosed item in {filename}")

    if len(parsed) != len(manifest):
        raise ValueError(f"Parsed {len(parsed)} packet items but manifest has {len(manifest)}")
    return parsed


def selected_chunks(record: dict[str, Any], max_excerpts: int) -> list[dict[str, Any]]:
    chunks = list(record.get("review_chunks") or [])[:max_excerpts]
    if not chunks:
        raise ValueError(f"No review chunks for {record['perfume_id']}")
    return chunks


def expected_excerpts(
    record: dict[str, Any], max_excerpts: int, max_chars_per_excerpt: int
) -> list[str]:
    return [
        clean_text(chunk.get("chunk_text", ""), max_chars_per_excerpt)
        for chunk in selected_chunks(record, max_excerpts)
        if clean_text(chunk.get("chunk_text", ""), max_chars_per_excerpt)
    ]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty audit table: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit(
    input_path: Path,
    packet_dir: Path,
    output_dir: Path,
    sample_size: int,
    seed: int,
) -> dict[str, Any]:
    summary = json.loads((packet_dir / "packet_summary.json").read_text(encoding="utf-8"))
    max_excerpts = int(summary["max_excerpts"])
    max_chars = int(summary["max_chars_per_excerpt"])
    records = read_jsonl(input_path)
    packets = parse_packet_items(packet_dir)
    if len(records) != 784 or len(packets) != 784:
        raise ValueError(f"Expected 784 records and packets, found {len(records)} and {len(packets)}")

    audit_rows: list[dict[str, Any]] = []
    for record in records:
        item_id = f"og::{record['perfume_id']}"
        packet = packets[item_id]
        chunks = selected_chunks(record, max_excerpts)
        expected = expected_excerpts(record, max_excerpts, max_chars)
        actual = list(packet["excerpts"])
        any_name_match = any(bool(chunk.get("any_name_match")) for chunk in chunks)
        audit_rows.append(
            {
                "item_id": item_id,
                "perfume_id": record["perfume_id"],
                "perfume_name": record["perfume_name"],
                "excerpt_count": len(actual),
                "selected_source_any_name_match": any_name_match,
                "packet_matches_selected_source": actual == expected,
                "max_packet_excerpt_chars": max(map(len, actual)),
                "chunk_1_video_id": chunks[0].get("video_id", ""),
                "chunk_1_url": chunks[0].get("url", ""),
                "chunk_1_match_strategy": chunks[0].get("match_strategy", ""),
                "chunk_2_video_id": chunks[1].get("video_id", "") if len(chunks) > 1 else "",
                "chunk_2_url": chunks[1].get("url", "") if len(chunks) > 1 else "",
                "chunk_2_match_strategy": (
                    chunks[1].get("match_strategy", "") if len(chunks) > 1 else ""
                ),
                "excerpt_1": actual[0],
                "excerpt_2": actual[1] if len(actual) > 1 else "",
            }
        )

    flagged = [row for row in audit_rows if not row["selected_source_any_name_match"]]
    if len(flagged) != 174:
        raise ValueError(f"Expected 174 no-match items, found {len(flagged)}")
    if sample_size < 1 or sample_size > len(flagged):
        raise ValueError(f"sample_size must be between 1 and {len(flagged)}")

    sampled_ids = {
        row["item_id"] for row in random.Random(seed).sample(flagged, sample_size)
    }
    sample_rows = [row for row in audit_rows if row["item_id"] in sampled_ids]
    if not all(row["packet_matches_selected_source"] for row in sample_rows):
        raise ValueError("At least one sampled packet did not match its selected source excerpt")

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "packet_provenance_audit.csv", audit_rows)
    write_csv(output_dir / "flagged_traceability_spot_check.csv", sample_rows)

    excerpt_counts: dict[str, int] = {}
    for row in audit_rows:
        key = str(row["excerpt_count"])
        excerpt_counts[key] = excerpt_counts.get(key, 0) + 1
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_jsonl": str(input_path.relative_to(PROJECT_ROOT)),
        "packet_dir": str(packet_dir.relative_to(PROJECT_ROOT)),
        "rows": len(audit_rows),
        "max_excerpts": max_excerpts,
        "max_chars_per_excerpt": max_chars,
        "excerpt_count_distribution": excerpt_counts,
        "items_with_detected_name_or_anchor_match": len(audit_rows) - len(flagged),
        "items_without_detected_name_or_anchor_match": len(flagged),
        "items_without_detected_name_or_anchor_match_rate": len(flagged) / len(audit_rows),
        "all_packets_match_selected_source": all(
            row["packet_matches_selected_source"] for row in audit_rows
        ),
        "spot_check": {
            "population": "items_without_detected_name_or_anchor_match",
            "sample_size": sample_size,
            "seed": seed,
            "all_sampled_packets_match_selected_source": True,
            "scope": (
                "Traceability from packet excerpts to stored selected source chunks only; "
                "not a judgment of topical relevance or label correctness."
            ),
        },
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "REPORT.md").write_text(
        "\n".join(
            [
                "# Annotation Packet Provenance Audit",
                "",
                f"- YouTube perfume packets: {len(audit_rows)}",
                f"- Packet policy: at most {max_excerpts} excerpts, at most {max_chars} characters each",
                f"- Two excerpts: {excerpt_counts.get('2', 0)}; one excerpt: {excerpt_counts.get('1', 0)}",
                f"- At least one detected perfume-name or anchor match: {len(audit_rows) - len(flagged)}",
                f"- No detected match in either selected source chunk: {len(flagged)} ({len(flagged) / len(audit_rows):.1%})",
                f"- Deterministic flagged-item traceability spot check: {sample_size} items at seed {seed}",
                "- All sampled packet excerpts exactly matched the bounded text preserved from their selected source chunks.",
                "",
                "The 174-item result is an evidence-relevance warning, not a label-error count. "
                "A missing mechanical name or anchor match does not prove that an excerpt is irrelevant, "
                "and exact source traceability does not prove that it discusses the target perfume. "
                "The spot check verifies packet-to-source serialization only; it does not assess emotion labels.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--packet-dir", type=Path, default=DEFAULT_PACKET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit(
        args.input_jsonl.resolve(),
        args.packet_dir.resolve(),
        args.output_dir.resolve(),
        args.sample_size,
        args.seed,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

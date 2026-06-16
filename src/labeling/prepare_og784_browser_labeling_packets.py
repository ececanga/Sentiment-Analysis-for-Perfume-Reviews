# Date Created: 2026-06-15
# Date Last Modified: 2026-06-15

"""Prepare browser Gemini labeling packets for the frozen OG 784 perfumes."""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from src.labeling.strict_schema import EMOTIONS, VAD_DIMS
from src.utils.paths import PROJECT_ROOT, data_path

OUTPUT_COLUMNS = [
    "item_id",
    "source_type",
    "perfume_id",
    "perfume_name",
    *EMOTIONS,
    *VAD_DIMS,
]

PROMPT_TEXT = """You are labeling perfume review text for an academic thesis.

Task: For each perfume item in the batch, assign emotion-tone labels using ONLY the review excerpts provided.

Rules:
- Label each perfume item independently.
- Do NOT infer from brand, perfume name, reputation, note list, marketing concept, or niche status.
- Labels reflect the reviewers' emotional tone expressed in the provided excerpts.
- Set an emotion to 1 only when there is clear linguistic evidence. Otherwise set it to 0.
- Multiple emotions can be 1.
- Surprise=1 only when the text explicitly expresses unexpectedness, such as "surprised", "did not expect", "caught me off guard", or similar language.
- Anticipation=1 when the text clearly expresses expectation, looking forward, curiosity, desire to keep testing, hype, or waiting for something to unfold.
- Trust=1 when the text clearly expresses comfort, safety, reliability, attachment, reassurance, confidence, or recommendation.
- Arousal means emotional activation or calmness, not perfume strength or projection.
- Dominance means confidence/control/assertiveness, not perfume loudness unless framed as personal power/control.
- If emotional evidence is weak or neutral, use all emotions=0 and valence=5, arousal=5, dominance=5.
- For VAD, judge the overall emotional tone across the excerpts.

Emotion columns, binary 0/1:
joy,sadness,anger,fear,trust,surprise,anticipation,disgust

VAD columns, integer 1-9:
valence: 1 very negative, 5 neutral, 9 very positive
arousal: 1 very calm, 5 neutral, 9 highly activated
dominance: 1 passive/gentle, 5 neutral, 9 commanding/confident

Return no markdown, no explanation, and no code fences.
CSV validity is mandatory: if any field contains a comma, wrap that field in double quotes.
Do not shorten, paraphrase, or add ellipses to item_id, source_type, perfume_id, or perfume_name.
Copy those identifier fields exactly from each input item.
Apart from the batch tracking line requested below, return only the CSV header and rows.
Use exactly this CSV header:
item_id,source_type,perfume_id,perfume_name,joy,sadness,anger,fear,trust,surprise,anticipation,disgust,valence,arousal,dominance

Return exactly one output row per input perfume item.
"""


def stringify_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def clean_text(value: object, max_chars: int) -> str:
    text = str(value or "").replace("\n", " ").replace("\r", " ").strip()
    text = re.sub(r"\s+", " ", text)
    if len(text) > max_chars:
        return text[:max_chars].rstrip()
    return text


def chunks(items: Sequence[dict[str, Any]], size: int) -> Iterable[Sequence[dict[str, Any]]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def select_excerpts(record: dict[str, Any], max_excerpts: int, max_chars_per_excerpt: int) -> list[str]:
    chunks_raw = record.get("review_chunks") or []
    selected = chunks_raw[:max_excerpts]
    excerpts = [clean_text(chunk.get("chunk_text", ""), max_chars_per_excerpt) for chunk in selected]
    excerpts = [text for text in excerpts if text]
    if excerpts:
        return excerpts
    return [clean_text(record.get("document_text", ""), max_chars_per_excerpt)]


def build_items(input_jsonl: Path, max_excerpts: int, max_chars_per_excerpt: int) -> list[dict[str, Any]]:
    records = read_jsonl(input_jsonl)
    items: list[dict[str, Any]] = []
    for record in records:
        perfume_id = str(record["perfume_id"])
        item = {
            "item_id": f"og::{perfume_id}",
            "source_type": "og_youtube",
            "perfume_id": perfume_id,
            "perfume_id_hash": str(record["perfume_id_hash"]),
            "perfume_name": str(record["perfume_name"]),
            "source_review_count": int(record.get("source_review_count", 0)),
            "deduplicated_review_count": int(record.get("deduplicated_review_count", 0)),
            "excerpts": select_excerpts(record, max_excerpts, max_chars_per_excerpt),
        }
        items.append(item)
    if len(items) != 784:
        raise ValueError(f"Expected 784 OG items, found {len(items)}")
    if len({item["item_id"] for item in items}) != len(items):
        raise ValueError("Duplicate item_id values found.")
    return items


def write_batch(path: Path, batch: Sequence[dict[str, Any]], batch_number: int) -> None:
    lines = [
        PROMPT_TEXT.strip(),
        "",
        "RESPONSE FORMAT",
        f"Start your response with exactly this tracking line: OG Batch {batch_number} output:",
        "Then immediately output the CSV header and one CSV row per input perfume item.",
        "",
        "INPUT BATCH",
        "",
    ]
    for item in batch:
        lines.extend(
            [
                "BEGIN_ITEM",
                f"item_id: {item['item_id']}",
                f"source_type: {item['source_type']}",
                f"perfume_id: {item['perfume_id']}",
                f"perfume_name: {item['perfume_name']}",
                f"source_review_count: {item['source_review_count']}",
            ]
        )
        for excerpt_index, excerpt in enumerate(item["excerpts"], start=1):
            lines.append(f"review_excerpt_{excerpt_index}: {excerpt}")
        lines.extend(["END_ITEM", ""])
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_manifest(path: Path, items: Sequence[dict[str, Any]]) -> None:
    fieldnames = [
        "item_id",
        "source_type",
        "perfume_id",
        "perfume_id_hash",
        "perfume_name",
        "source_review_count",
        "deduplicated_review_count",
        "excerpt_count",
        "batch_file",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in items:
            writer.writerow(
                {
                    **{key: item.get(key, "") for key in fieldnames},
                    "excerpt_count": len(item["excerpts"]),
                }
            )


def write_output_template(path: Path, items: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for item in items:
            writer.writerow(
                {
                    "item_id": item["item_id"],
                    "source_type": item["source_type"],
                    "perfume_id": item["perfume_id"],
                    "perfume_name": item["perfume_name"],
                    **{column: "" for column in EMOTIONS + VAD_DIMS},
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare OG 784 browser Gemini labeling packets.")
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "gemini_judge"
        / "judge_g25pro_v1"
        / "model_benchmark"
        / "datasets"
        / "perfume_text_dataset_v1.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=data_path("annotation_packets", "og784_gemini_browser_pro_review_level"),
    )
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--max-excerpts", type=int, default=2)
    parser.add_argument("--max-chars-per-excerpt", type=int, default=360)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    items = build_items(args.input_jsonl, args.max_excerpts, args.max_chars_per_excerpt)

    for batch_number, batch in enumerate(chunks(items, args.batch_size), start=1):
        first = (batch_number - 1) * args.batch_size + 1
        last = first + len(batch) - 1
        batch_file = f"og784_batch_{batch_number:02d}_perfumes_{first:03d}-{last:03d}.txt"
        for item in batch:
            item["batch_file"] = batch_file
        write_batch(args.out_dir / batch_file, batch, batch_number)

    write_manifest(args.out_dir / "manifest.csv", items)
    write_output_template(args.out_dir / "output_template.csv", items)
    (args.out_dir / "prompt.md").write_text(PROMPT_TEXT + "\n", encoding="utf-8")

    batch_count = len(list(chunks(items, args.batch_size)))
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_jsonl": stringify_path(args.input_jsonl),
        "out_dir": stringify_path(args.out_dir),
        "rows": len(items),
        "unique_perfumes": len({item["perfume_id"] for item in items}),
        "batch_size": args.batch_size,
        "batch_count": batch_count,
        "max_excerpts": args.max_excerpts,
        "max_chars_per_excerpt": args.max_chars_per_excerpt,
        "output_columns": OUTPUT_COLUMNS,
        "note": "Packets contain OG review excerpts only. Existing Gemini 2.5 Pro labels are not copied.",
    }
    (args.out_dir / "packet_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    readme = f"""# OG 784 Gemini Browser Pro Labeling Packets

Paste one batch file at a time into Gemini Pro browser.

- Batch files: `og784_batch_*.txt`
- Manifest: `manifest.csv`
- Empty output template: `output_template.csv`
- Prompt only: `prompt.md`

Each batch asks Gemini to start with `OG Batch N output:` and then return CSV rows.

Counts:
- Perfumes: {len(items)}
- Batch size: {args.batch_size}
- Batches: {batch_count}

These packets do not include old Gemini 2.5 Pro labels.
"""
    (args.out_dir / "README.md").write_text(readme, encoding="utf-8")
    print(f"Wrote {len(items)} OG perfume items to {args.out_dir}")
    print(f"Batch files: {batch_count}")


if __name__ == "__main__":
    main()

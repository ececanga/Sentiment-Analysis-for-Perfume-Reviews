# Date Created: 2026-06-08

"""Prepare paste-ready Claude browser labeling packets.

The packets are intentionally for manual/browser pseudo-labeling, not API ground
truth. They keep the first N perfumes from the frozen 784 identity list and
write source-separated batches for YouTube and synthetic review text.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import pandas as pd

from src.labeling.strict_schema import EMOTIONS, VAD_DIMS, normalize_perfume_id
from src.utils.paths import PROJECT_ROOT, data_path, labels_path


PROMPT_TEXT = """You are labeling perfume review text for an academic thesis.

Task: For each item in the batch, assign emotion-tone labels using ONLY the review text provided.

Rules:
- Label each item independently.
- Do NOT infer from brand, perfume name, reputation, note list, marketing concept, or niche status.
- Labels reflect the reviewer's emotional tone in the text.
- Set an emotion to 1 only when there is clear linguistic evidence. Otherwise set it to 0.
- Multiple emotions can be 1.
- Surprise=1 only when the text explicitly expresses unexpectedness.
- Arousal means emotional activation or calmness, not perfume strength/performance.
- Dominance means confidence/control/assertiveness, not projection unless framed as personal power/control.
- If evidence is weak or neutral, use all emotions=0 and valence=5, arousal=5, dominance=5.

Emotion columns, binary 0/1:
joy,sadness,anger,fear,trust,surprise,anticipation,disgust

VAD columns, integer 1-9:
valence: 1 very negative, 5 neutral, 9 very positive
arousal: 1 very calm, 5 neutral, 9 highly activated
dominance: 1 passive/gentle, 5 neutral, 9 commanding/confident

Return ONLY valid CSV. No markdown, no explanation.
Use exactly this header:
item_id,source_type,perfume_id,perfume_name,joy,sadness,anger,fear,trust,surprise,anticipation,disgust,valence,arousal,dominance

Return exactly one output row per input item.
"""


def clean_text(value: object, max_chars: int) -> str:
    text = str(value or "").replace("\n", " ").replace("\r", " ").strip()
    text = re.sub(r"\s+", " ", text)
    if len(text) > max_chars:
        return text[:max_chars].rstrip() + "..."
    return text


def stable_seed(seed: int, key: str) -> int:
    digest = hashlib.sha1(f"{seed}:{key}".encode("utf-8")).hexdigest()[:8]
    return int(digest, 16)


def select_snippets(snippets: Sequence[str], perfume_id: str, seed: int, target_count: int = 3) -> List[str]:
    clean = [s for s in snippets if s]
    if not clean:
        return []
    if len(clean) >= target_count:
        rng = random.Random(stable_seed(seed, perfume_id))
        idxs = rng.sample(range(len(clean)), target_count)
        return [clean[i] for i in idxs]
    out = clean[:]
    while len(out) < target_count:
        out.append(clean[len(out) % len(clean)])
    return out


def chunks(items: Sequence[dict], size: int) -> Iterable[Sequence[dict]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def write_batch_file(path: Path, batch: Sequence[dict], source_type: str) -> None:
    lines: List[str] = []
    lines.append(PROMPT_TEXT.strip())
    lines.append("")
    lines.append("INPUT BATCH")
    lines.append("")
    for item in batch:
        lines.append("BEGIN_ITEM")
        lines.append(f"item_id: {item['item_id']}")
        lines.append(f"source_type: {source_type}")
        lines.append(f"perfume_id: {item['perfume_id']}")
        lines.append(f"perfume_name: {item['perfume_name']}")
        for text_idx, text in enumerate(item["texts"], start=1):
            label = "youtube_snippet" if source_type == "youtube" else "synthetic_review"
            lines.append(f"{label}_{text_idx}: {text}")
        lines.append("END_ITEM")
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_manifest(path: Path, rows: Sequence[dict]) -> None:
    fieldnames = [
        "item_id",
        "source_type",
        "perfume_id",
        "perfume_id_hash",
        "perfume_name",
        "n_texts",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_output_template(path: Path, rows: Sequence[dict]) -> None:
    fieldnames = ["item_id", "source_type", "perfume_id", "perfume_name"] + EMOTIONS + VAD_DIMS
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "item_id": row["item_id"],
                    "source_type": row["source_type"],
                    "perfume_id": row["perfume_id"],
                    "perfume_name": row["perfume_name"],
                    **{key: "" for key in EMOTIONS + VAD_DIMS},
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Claude browser labeling packets.")
    parser.add_argument(
        "--identity-csv",
        type=Path,
        default=labels_path("gemini_labels_784_judge_g25pro_v1.csv"),
        help="Frozen 784 identity list. Label values are not copied into packets.",
    )
    parser.add_argument(
        "--youtube-csv",
        type=Path,
        default=data_path("raw", "YouTube_Perfume_Reviews_FullyFormatted.csv"),
    )
    parser.add_argument(
        "--synthetic-csv",
        type=Path,
        default=data_path("synthetic", "synthetic_perfume_reviews_784_unlabeled_v1.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=data_path("annotation_packets", "claude_browser_first100"),
    )
    parser.add_argument("--limit-perfumes", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--youtube-max-chars", type=int, default=650)
    parser.add_argument("--synthetic-max-chars", type=int, default=900)
    args = parser.parse_args()

    csv.field_size_limit(sys.maxsize)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    identities = pd.read_csv(args.identity_csv).head(args.limit_perfumes).copy()
    required_identity_cols = {"perfume_id", "perfume_id_hash", "perfume_name"}
    missing_identity = required_identity_cols - set(identities.columns)
    if missing_identity:
        raise ValueError(f"Identity CSV missing columns: {sorted(missing_identity)}")

    youtube = pd.read_csv(args.youtube_csv)
    if not {"Perfume", "Final Transcript"}.issubset(youtube.columns):
        raise ValueError("YouTube CSV must contain Perfume and Final Transcript.")
    youtube["perfume_id"] = youtube["Perfume"].map(normalize_perfume_id)
    youtube["clean_text"] = youtube["Final Transcript"].map(lambda t: clean_text(t, args.youtube_max_chars))

    youtube_groups: Dict[str, List[str]] = {}
    for perfume_id, group in youtube.groupby("perfume_id", sort=False):
        youtube_groups[str(perfume_id)] = [t for t in group["clean_text"].tolist() if t]

    synthetic = pd.read_csv(args.synthetic_csv)
    if not {"perfume_id", "Final Transcript"}.issubset(synthetic.columns):
        raise ValueError("Synthetic CSV must contain perfume_id and Final Transcript.")
    synthetic["clean_text"] = synthetic["Final Transcript"].map(lambda t: clean_text(t, args.synthetic_max_chars))

    youtube_items: List[dict] = []
    synthetic_items: List[dict] = []
    missing_youtube: List[str] = []
    missing_synthetic: List[str] = []

    for row in identities.to_dict(orient="records"):
        perfume_id = str(row["perfume_id"])
        perfume_name = str(row["perfume_name"])
        perfume_id_hash = str(row["perfume_id_hash"])

        snippets = select_snippets(youtube_groups.get(perfume_id, []), perfume_id, args.seed, target_count=3)
        if snippets:
            youtube_items.append(
                {
                    "item_id": f"youtube::{perfume_id}",
                    "source_type": "youtube",
                    "perfume_id": perfume_id,
                    "perfume_id_hash": perfume_id_hash,
                    "perfume_name": perfume_name,
                    "texts": snippets,
                    "n_texts": len(snippets),
                }
            )
        else:
            missing_youtube.append(perfume_id)

        syn_rows = synthetic.loc[synthetic["perfume_id"].astype(str) == perfume_id].copy()
        if "review_index" in syn_rows.columns:
            syn_rows = syn_rows.sort_values("review_index")
        syn_texts = [t for t in syn_rows["clean_text"].tolist() if t]
        if syn_texts:
            synthetic_items.append(
                {
                    "item_id": f"synthetic::{perfume_id}",
                    "source_type": "synthetic",
                    "perfume_id": perfume_id,
                    "perfume_id_hash": perfume_id_hash,
                    "perfume_name": perfume_name,
                    "texts": syn_texts,
                    "n_texts": len(syn_texts),
                }
            )
        else:
            missing_synthetic.append(perfume_id)

    for source_type, items in [("youtube", youtube_items), ("synthetic", synthetic_items)]:
        source_dir = args.out_dir / source_type
        source_dir.mkdir(parents=True, exist_ok=True)
        for idx, batch in enumerate(chunks(items, args.batch_size), start=1):
            first = (idx - 1) * args.batch_size + 1
            last = first + len(batch) - 1
            write_batch_file(
                source_dir / f"{source_type}_batch_{idx:02d}_perfumes_{first:03d}-{last:03d}.txt",
                batch,
                source_type,
            )
        write_manifest(source_dir / f"{source_type}_manifest.csv", items)
        write_output_template(source_dir / f"{source_type}_output_template.csv", items)

    (args.out_dir / "claude_browser_prompt.md").write_text(PROMPT_TEXT, encoding="utf-8")
    summary = {
        "identity_csv": str(args.identity_csv.relative_to(PROJECT_ROOT)),
        "youtube_csv": str(args.youtube_csv.relative_to(PROJECT_ROOT)),
        "synthetic_csv": str(args.synthetic_csv.relative_to(PROJECT_ROOT)),
        "limit_perfumes": args.limit_perfumes,
        "batch_size": args.batch_size,
        "youtube_items": len(youtube_items),
        "synthetic_items": len(synthetic_items),
        "missing_youtube_perfume_ids": missing_youtube,
        "missing_synthetic_perfume_ids": missing_synthetic,
        "schema": ["item_id", "source_type", "perfume_id", "perfume_name"] + EMOTIONS + VAD_DIMS,
        "note": "Packets contain no existing label values; Claude/browser labels should be treated as pseudo-labels.",
    }
    (args.out_dir / "packet_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    readme = f"""# Claude Browser Labeling Packets

Use one batch file at a time in Claude.

- YouTube batches: `youtube/youtube_batch_*.txt`
- Synthetic batches: `synthetic/synthetic_batch_*.txt`
- Prompt only: `claude_browser_prompt.md`
- Output templates: `youtube/youtube_output_template.csv`, `synthetic/synthetic_output_template.csv`

Each batch asks Claude to return CSV only. Keep the returned CSV rows from all batches together by source.

Counts:
- YouTube perfume items: {len(youtube_items)}
- Synthetic perfume items: {len(synthetic_items)}
- Batch size: {args.batch_size}

These files do not include existing Gemini labels. Claude browser outputs are pseudo-labels and should be stored as a separate annotation version.
"""
    (args.out_dir / "README.md").write_text(readme, encoding="utf-8")

    print(f"Wrote Claude browser packets to: {args.out_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

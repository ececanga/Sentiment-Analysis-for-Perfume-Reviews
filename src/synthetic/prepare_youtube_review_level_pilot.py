# Date Created: 2026-06-13
# Date Last Modified: 2026-06-13

"""Prepare a 100-row real YouTube review-level pilot for comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.labeling.strict_schema import normalize_perfume_id
from src.utils.paths import PROJECT_ROOT, data_path, labels_path


def stable_seed(seed: int, key: str) -> int:
    digest = hashlib.sha1(f"{seed}:{key}".encode("utf-8")).hexdigest()[:8]
    return int(digest, 16)


def clean_text(value: object) -> str:
    return " ".join(str(value or "").replace("\r", " ").replace("\n", " ").split())


def emotion_keyword_count(text: str) -> int:
    terms = [
        "love",
        "like",
        "hate",
        "sad",
        "happy",
        "joy",
        "comfort",
        "confident",
        "surprise",
        "unexpected",
        "disappoint",
        "tired",
        "overwhelmed",
        "nostalgic",
        "memory",
        "calm",
        "excited",
        "afraid",
        "fear",
        "disgust",
    ]
    lowered = text.lower()
    return sum(len(re.findall(rf"\b{re.escape(term)}\w*\b", lowered)) for term in terms)


def stringify_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def sample_group(group: pd.DataFrame, *, perfume_id: str, seed: int, count: int) -> pd.DataFrame:
    if len(group) <= count:
        return group.sort_values("source_row_index").copy()
    rng = random.Random(stable_seed(seed, perfume_id))
    selected = rng.sample(group.index.tolist(), count)
    return group.loc[selected].sort_values("source_row_index").copy()


def build_rows(
    *,
    raw_df: pd.DataFrame,
    identity_df: pd.DataFrame,
    primary_perfume_count: int,
    reviews_per_primary_perfume: int,
    total_reviews: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw_by_id = {str(pid): group.copy() for pid, group in raw_df.groupby("perfume_id", sort=False)}
    primary_ids = identity_df["perfume_id"].astype(str).head(primary_perfume_count).tolist()
    selected_parts: list[pd.DataFrame] = []
    selection_notes: list[dict[str, Any]] = []
    used_raw_indices: set[int] = set()

    for perfume_id in primary_ids:
        group = raw_by_id.get(perfume_id, pd.DataFrame())
        take = sample_group(group, perfume_id=perfume_id, seed=seed, count=reviews_per_primary_perfume) if not group.empty else group
        selected_parts.append(take)
        used_raw_indices.update(int(idx) for idx in take["source_row_index"].tolist())
        selection_notes.append(
            {
                "perfume_id": perfume_id,
                "available_reviews": int(len(group)),
                "selected_reviews": int(len(take)),
                "selection_role": "primary_same_as_synthetic_pilot",
            }
        )

    selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame(columns=raw_df.columns)
    needed = total_reviews - len(selected)
    filler_notes: list[dict[str, Any]] = []
    if needed > 0:
        for perfume_id in identity_df["perfume_id"].astype(str).tolist()[primary_perfume_count:]:
            if needed <= 0:
                break
            group = raw_by_id.get(perfume_id, pd.DataFrame())
            if group.empty:
                continue
            group = group.loc[~group["source_row_index"].isin(used_raw_indices)].copy()
            if group.empty:
                continue
            take = sample_group(group, perfume_id=perfume_id, seed=seed, count=1)
            selected = pd.concat([selected, take], ignore_index=True)
            used_raw_indices.update(int(idx) for idx in take["source_row_index"].tolist())
            needed -= len(take)
            filler_notes.append(
                {
                    "perfume_id": perfume_id,
                    "available_reviews": int(len(group)),
                    "selected_reviews": int(len(take)),
                    "selection_role": "filler_to_reach_100_without_duplicates",
                }
            )

    if len(selected) != total_reviews:
        raise ValueError(f"Could only select {len(selected)} reviews; requested {total_reviews}.")

    selected = selected.reset_index(drop=True)
    selected["review_index"] = selected.groupby("perfume_id").cumcount() + 1
    selected["synthetic_review_id"] = selected.apply(
        lambda row: f"youtube-review-{row['perfume_id']}-{int(row['review_index']):02d}",
        axis=1,
    )
    selected["source_type"] = "youtube"
    selected["dataset_version"] = "youtube_review_level_pilot100_g25pro_v1"
    selected["generation_policy"] = "real_youtube_review_text"
    selected["Final Transcript"] = selected["Final Transcript"].map(clean_text)
    selected["Transcript"] = selected.get("Transcript", selected["Final Transcript"]).map(clean_text)
    selected["Transcript Length"] = selected["Final Transcript"].map(lambda text: len(re.findall(r"\b\w+\b", text)))
    selected["Emotion Keyword Count"] = selected["Final Transcript"].map(emotion_keyword_count)

    out_cols = [
        "Perfume",
        "Video ID",
        "Title",
        "Description",
        "URL",
        "Channel Title",
        "Transcript",
        "Transcript Length",
        "Emotion Keyword Count",
        "Final Transcript",
        "source_type",
        "dataset_version",
        "generation_policy",
        "synthetic_review_id",
        "perfume_id",
        "perfume_id_hash",
        "review_index",
        "source_row_index",
    ]
    for col in out_cols:
        if col not in selected.columns:
            selected[col] = ""
    summary = {
        "selection_notes": selection_notes + filler_notes,
        "primary_perfume_count": primary_perfume_count,
        "reviews_per_primary_perfume": reviews_per_primary_perfume,
        "primary_selected_total": int(sum(item["selected_reviews"] for item in selection_notes)),
        "filler_selected_total": int(sum(item["selected_reviews"] for item in filler_notes)),
    }
    return selected[out_cols].copy(), summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare 100 real YouTube review rows for review-level comparison.")
    parser.add_argument("--raw-csv", type=Path, default=data_path("raw", "YouTube_Perfume_Reviews_FullyFormatted.csv"))
    parser.add_argument("--identity-csv", type=Path, default=labels_path("gemini_labels_784_judge_g25pro_v1.csv"))
    parser.add_argument("--output-csv", type=Path, default=data_path("raw", "youtube_review_level_pilot100_g25pro_v1.csv"))
    parser.add_argument("--summary-json", type=Path, default=data_path("raw", "youtube_review_level_pilot100_g25pro_v1_summary.json"))
    parser.add_argument("--primary-perfume-count", type=int, default=20)
    parser.add_argument("--reviews-per-primary-perfume", type=int, default=5)
    parser.add_argument("--total-reviews", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260613)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_df = pd.read_csv(args.raw_csv)
    identity_df = pd.read_csv(args.identity_csv)
    required_raw = {"Perfume", "Final Transcript"}
    required_identity = {"perfume_id", "perfume_id_hash", "perfume_name"}
    missing_raw = sorted(required_raw - set(raw_df.columns))
    missing_identity = sorted(required_identity - set(identity_df.columns))
    if missing_raw:
        raise ValueError(f"Raw CSV missing columns: {missing_raw}")
    if missing_identity:
        raise ValueError(f"Identity CSV missing columns: {missing_identity}")

    raw_df = raw_df.copy()
    raw_df["source_row_index"] = raw_df.index.astype(int)
    raw_df["perfume_id"] = raw_df["Perfume"].map(normalize_perfume_id)
    hash_map = identity_df.set_index("perfume_id")["perfume_id_hash"].astype(str).to_dict()
    name_map = identity_df.set_index("perfume_id")["perfume_name"].astype(str).to_dict()
    raw_df["perfume_id_hash"] = raw_df["perfume_id"].map(hash_map).fillna("")
    raw_df["Perfume"] = raw_df["perfume_id"].map(name_map).fillna(raw_df["Perfume"])
    raw_df = raw_df.loc[raw_df["Final Transcript"].map(clean_text).ne("")].copy()

    out, selection_summary = build_rows(
        raw_df=raw_df,
        identity_df=identity_df,
        primary_perfume_count=args.primary_perfume_count,
        reviews_per_primary_perfume=args.reviews_per_primary_perfume,
        total_reviews=args.total_reviews,
        seed=args.seed,
    )
    if not out["synthetic_review_id"].is_unique:
        raise ValueError("Generated review IDs are not unique.")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_csv": stringify_path(args.raw_csv),
        "identity_csv": stringify_path(args.identity_csv),
        "output_csv": stringify_path(args.output_csv),
        "rows": int(len(out)),
        "unique_perfumes": int(out["perfume_id"].nunique()),
        "seed": args.seed,
        **selection_summary,
    }
    args.summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote YouTube review pilot: {args.output_csv}")
    print(f"Rows: {len(out)} Unique perfumes: {out['perfume_id'].nunique()}")
    print(f"Primary selected: {selection_summary['primary_selected_total']} Filler selected: {selection_summary['filler_selected_total']}")


if __name__ == "__main__":
    main()

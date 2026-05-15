# Date Created: 2026-03-02 11:07:58+03:00
# Date Last Modified: 2026-05-15

# filter_reviews_by_brand.py
"""
Purpose
-------
Filter YouTube perfume review rows by selected brand keywords while keeping the per-review
structure intact (no concatenation). Optionally create an annotation-ready perfume-level
packet (30 perfumes -> 30 rows) with 3 snippets per perfume + empty label columns.

Input
-----
data/raw/YouTube_Perfume_Reviews_FullyFormatted.csv

Outputs (saved under data/annotation_packets/)
-------------------------------
1) Filtered_Reviews_<BRANDTAG>.csv
   - All matching review rows
   - Includes audit columns:
       Matched_Brands
       Match_Perfume / Match_Title / Match_Description / Match_Channel Title

2) Filtered_Perfume_Counts_<BRANDTAG>.csv
   - Unique perfume names
   - Review count per perfume

3) (Optional) Annotation_Packet_<BRANDTAG>.csv
   - One row per perfume
   - Snippet_1..Snippet_K from sampled reviews
   - Empty gold label columns: 8 emotions + VAD (1–9)

Optional Excel
--------------
If WRITE_EXCEL = True, also writes xlsx versions of the above (same basenames).

Notes
-----
- Default strictness is "perfume_or_title" to reduce false positives.
- If you want more recall, set REQUIRE_IN="any" and rely on audit columns.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from src.utils.paths import data_path

# -------------------- CONFIG --------------------

INPUT_CSV = data_path("raw", "YouTube_Perfume_Reviews_FullyFormatted.csv")
OUTPUT_DIR = data_path("annotation_packets")

BRANDS = [
    {"tag": "TOSKOVAT", "keywords": ["toskovat", "toskovat'"]},
    {"tag": "ORTO_PARISI", "keywords": ["orto parisi"]},
    {"tag": "ELO", "keywords": ["etat libre d'orange", "etat libre d’orange", "etat libre dorange", "etat libre", "eldo", "e l d o"]},
    {"tag": "NASOMATTO", "keywords": ["nasomatto"]},
    {"tag": "SERGE_LUTENS", "keywords": ["serge lutens", "lutens"]},
]

SEARCH_COLS = ["Perfume", "Title", "Description", "Channel Title"]

REQUIRE_IN = "any"   # "any", "title_only", "perfume_or_title"

# Optional per-review sampling output (keeps per-review structure)
SAMPLE_K_PER_PERFUME = None       # e.g., 3 (set None to disable)
SAMPLE_RANDOM_SEED = 42

# Perfume column name
PERFUME_COL = "Perfume"

# Annotation packet (perfume-level) settings
MAKE_ANNOTATION_PACKET = True
ANNOT_K_SNIPPETS = 3              # 2–3 recommended
SNIPPET_TEXT_COL = "Final Transcript"  # change if needed
SNIPPET_CHAR_LEN = 650            # snippet length for easy annotation

# Write Excel copies too
WRITE_EXCEL = False               # True -> also writes .xlsx

# ------------------------------------------------


def normalize(series: pd.Series) -> pd.Series:
    """Normalize a text Series for matching."""
    return (
        series.astype(str)
        .fillna("")
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def build_regex_and_tag(brands: List[Dict[str, List[str]]]) -> Tuple[re.Pattern, str, List[str]]:
    """
    Build a single regex matching any provided keyword.
    Adds word-boundary guards for short tags like 'mfk'/'pdm' to reduce collisions.
    Returns: (compiled_regex, file_tag, all_tags)
    """
    tags: List[str] = []
    keywords: List[str] = []

    for b in brands:
        tags.append(b["tag"])
        keywords.extend([k.strip() for k in b["keywords"] if k and k.strip()])

    parts: List[str] = []
    for k in keywords:
        k_esc = re.escape(k)
        # word-boundary guard for short acronyms
        if k.lower() in {"mfk", "pdm", "eldo"}:
            parts.append(rf"(?<!\w){k_esc}(?!\w)")
        else:
            parts.append(k_esc)

    if not parts:
        raise ValueError("No keywords provided in BRANDS.")

    pattern = r"(?:%s)" % "|".join(parts)
    return re.compile(pattern, flags=re.IGNORECASE), "_".join(tags), tags


def column_match(df: pd.DataFrame, column: str, regex: re.Pattern) -> pd.Series:
    """Boolean mask for whether regex appears in a given column."""
    return normalize(df[column]).str.contains(regex, na=False)


def detect_brands_in_text(text_lower: str, brands: List[Dict[str, List[str]]]) -> str:
    """
    Detect which brand keyword sets are present using simple substring checks.
    This is for audit reporting, not strict filtering.
    """
    hits: List[str] = []
    for b in brands:
        kws = [k.lower() for k in b["keywords"] if k and k.strip()]
        if any(kw in text_lower for kw in kws):
            hits.append(b["tag"])
    return ",".join(hits)


def shorten_text(text: str, n: int) -> str:
    t = str(text).replace("\n", " ").strip()
    return t[:n] + ("..." if len(t) > n else "")


def safe_to_excel(df: pd.DataFrame, out_xlsx: Path, sheet_name: str = "Sheet1") -> None:
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name=sheet_name)


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_CSV}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)

    regex, file_tag, _tags = build_regex_and_tag(BRANDS)

    # Build per-column masks
    masks: Dict[str, pd.Series] = {}
    for col in SEARCH_COLS:
        if col in df.columns:
            masks[col] = column_match(df, col, regex)

    if not masks:
        raise ValueError(f"None of SEARCH_COLS exist in dataset. Expected one of: {SEARCH_COLS}")

    # Apply strictness rule
    if REQUIRE_IN == "any":
        final_mask = None
        for m in masks.values():
            final_mask = m if final_mask is None else (final_mask | m)

    elif REQUIRE_IN == "title_only":
        if "Title" not in masks:
            raise ValueError("'Title' column required but not found.")
        final_mask = masks["Title"]

    elif REQUIRE_IN == "perfume_or_title":
        if "Perfume" not in masks or "Title" not in masks:
            raise ValueError("'Perfume' and 'Title' required for this mode.")
        final_mask = masks["Perfume"] | masks["Title"]

    else:
        raise ValueError("Invalid REQUIRE_IN setting. Use: any | title_only | perfume_or_title")

    filtered = df.loc[final_mask].copy()

    # Add audit columns
    present_cols = [c for c in SEARCH_COLS if c in filtered.columns]
    joined = normalize(filtered[present_cols[0]])
    for c in present_cols[1:]:
        joined = joined + " " + normalize(filtered[c])
    joined_lower = joined.str.lower()

    filtered["Matched_Brands"] = joined_lower.apply(lambda t: detect_brands_in_text(t, BRANDS))
    for col, mask in masks.items():
        filtered[f"Match_{col}"] = mask.loc[filtered.index].astype(int)

    # Aggregate counts per perfume
    if PERFUME_COL not in filtered.columns:
        raise ValueError(f"'{PERFUME_COL}' column not found in filtered dataset.")

    perfume_counts = (
        filtered.groupby(PERFUME_COL)
        .size()
        .reset_index(name="Review_Count")
        .sort_values("Review_Count", ascending=False)
    )

    # Save outputs
    reviews_out = OUTPUT_DIR / f"Filtered_Reviews_{file_tag}.csv"
    counts_out = OUTPUT_DIR / f"Filtered_Perfume_Counts_{file_tag}.csv"

    filtered.to_csv(reviews_out, index=False)
    perfume_counts.to_csv(counts_out, index=False)

    print(f"✅ Mode: {REQUIRE_IN}")
    print(f"✅ Filtered rows: {len(filtered)}")
    print(f"✅ Unique perfumes: {len(perfume_counts)}")
    print(f"📁 Saved:\n  - {reviews_out}\n  - {counts_out}")

    if WRITE_EXCEL:
        safe_to_excel(filtered, OUTPUT_DIR / f"Filtered_Reviews_{file_tag}.xlsx", sheet_name="FilteredReviews")
        safe_to_excel(perfume_counts, OUTPUT_DIR / f"Filtered_Perfume_Counts_{file_tag}.xlsx", sheet_name="PerfumeCounts")
        print("📁 Also saved Excel copies (.xlsx).")

    # Optional: sample K reviews per perfume (per-review structure)
    if SAMPLE_K_PER_PERFUME is not None:
        sampled = (
            filtered.groupby(PERFUME_COL, group_keys=False)
            .apply(lambda g: g.sample(
                n=min(SAMPLE_K_PER_PERFUME, len(g)),
                random_state=SAMPLE_RANDOM_SEED
            ))
        )
        sampled_out = OUTPUT_DIR / f"Filtered_Reviews_SampledPerPerfume_{file_tag}.csv"
        sampled.to_csv(sampled_out, index=False)
        print(f"📁 Also saved sampled file (K={SAMPLE_K_PER_PERFUME}):\n  - {sampled_out}")

        if WRITE_EXCEL:
            safe_to_excel(sampled, OUTPUT_DIR / f"Filtered_Reviews_SampledPerPerfume_{file_tag}.xlsx",
                          sheet_name="SampledReviews")

    # Optional: make perfume-level annotation packet
    if MAKE_ANNOTATION_PACKET:
        if SNIPPET_TEXT_COL not in filtered.columns:
            print(f"⚠️ Annotation packet skipped: '{SNIPPET_TEXT_COL}' column not found.")
        else:
            sampled_for_packet = (
                filtered.groupby(PERFUME_COL, group_keys=False)
                .apply(lambda g: g.sample(
                    n=min(ANNOT_K_SNIPPETS, len(g)),
                    random_state=SAMPLE_RANDOM_SEED
                ))
            ).copy()

            sampled_for_packet["Snippet"] = sampled_for_packet[SNIPPET_TEXT_COL].apply(
                lambda t: shorten_text(t, SNIPPET_CHAR_LEN)
            )
            sampled_for_packet["rank"] = sampled_for_packet.groupby(PERFUME_COL).cumcount() + 1

            packet = (
                sampled_for_packet.pivot(index=PERFUME_COL, columns="rank", values="Snippet")
                .reset_index()
                .rename(columns={i: f"Snippet_{i}" for i in range(1, ANNOT_K_SNIPPETS + 1)})
            )

            # Add gold columns (multi-label + VAD)
            gold_cols = [
                "joy", "sadness", "anger", "fear",
                "trust", "surprise", "anticipation", "disgust",
                "valence", "arousal", "dominance"
            ]
            for c in gold_cols:
                packet[c] = ""

            annotation_out = OUTPUT_DIR / f"Annotation_Packet_{file_tag}.csv"
            packet.to_csv(annotation_out, index=False)
            print(f"📁 Saved annotation packet ({len(packet)} perfumes):\n  - {annotation_out}")

            if WRITE_EXCEL:
                safe_to_excel(packet, OUTPUT_DIR / f"Annotation_Packet_{file_tag}.xlsx", sheet_name="AnnotationPacket")


if __name__ == "__main__":
    main()

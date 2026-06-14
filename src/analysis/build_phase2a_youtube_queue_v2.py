# Date Created: 2026-06-01
# Date Last Modified: 2026-06-01

"""Build the Phase 2A YouTube transcript candidate queue v2.

The queue is intentionally YouTube/review oriented. It uses product name and brand
metadata plus local collection status only; product descriptions and note pyramids
are not used as fallback text or ranking inputs.
"""

from __future__ import annotations

import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs/gemini_judge/judge_g25pro_2k_v1/planning/phase2a_next_candidate_queue_v2.csv"
)

FINAL_PERFUMES = PROJECT_ROOT / "data/perfume_level/final_perfume_data.csv"
LEGACY_STATUS = PROJECT_ROOT / "data/intermediate/processed_perfumes.csv"
PHASE2A_STATUS = PROJECT_ROOT / "data/intermediate/phase2a_processed_perfumes.csv"
PHASE2A_RAW_FILES = (
    PROJECT_ROOT / "data/raw/YouTube_Perfume_Reviews_Phase2A_FullyFormatted.csv",
    PROJECT_ROOT / "data/raw/YouTube_Perfume_Reviews_Phase2A_Partial.csv",
)
FROZEN_LABELS = PROJECT_ROOT / "labels/gemini_labels_784_judge_g25pro_v1.csv"
FROZEN_RAW = PROJECT_ROOT / "data/raw/YouTube_Perfume_Reviews_FullyFormatted.csv"
LEGACY_QUEUE = (
    PROJECT_ROOT
    / "outputs/gemini_judge/judge_g25pro_2k_v1/planning/phase2a_next_candidate_queue.csv"
)

FIELDNAMES = [
    "perfume_name",
    "brand",
    "normalized_name",
    "current_status",
    "legacy_status",
    "quality_score",
    "quality_flags",
    "exclusion_reason",
    "recommended_query_templates",
    "recommended_for_next_batch",
]

REVIEWABLE_BRANDS = {
    "aesop",
    "amouage",
    "bdk parfums",
    "by kilian",
    "byredo",
    "comme des garcons",
    "diptyque",
    "d.s. and durga",
    "electimuss",
    "escentric molecules",
    "etat libre d'orange",
    "ex nihilo",
    "fragrance du bois",
    "frederic malle",
    "heeley",
    "heretic",
    "initio parfums prives",
    "jovoy",
    "l'artisan parfumeur",
    "le labo",
    "les indemodables",
    "maison crivelli",
    "maison francis kurkdjian",
    "mancera",
    "memo",
    "montale",
    "nasomatto",
    "nishane",
    "ormonde jayne",
    "parfums de marly",
    "penhaligons",
    "profumum",
    "serge lutens",
    "stephane humbert lucas 777",
    "tauer perfumes",
    "the house of oud",
    "tom ford private blend",
    "vilhelm parfumerie",
    "xerjoff",
    "zoologist",
}

FRAGRANCE_TERMS = (
    "eau de parfum",
    "eau de toilette",
    "extrait de parfum",
    "extrait",
    "parfum extrait",
    "parfum",
    "cologne",
    "fragrance",
)

NON_FRAGRANCE_PATTERNS = (
    (r"\bhair\b", "non_fragrance_variant"),
    (r"\bhair mist\b", "non_fragrance_variant"),
    (r"\bhair perfume\b", "non_fragrance_variant"),
    (r"\bcandle\b", "non_fragrance_variant"),
    (r"\broom spray\b", "non_fragrance_variant"),
    (r"\bbody spray\b", "non_fragrance_variant"),
    (r"\bbody cream\b", "non_fragrance_variant"),
    (r"\bbody care\b", "non_fragrance_variant"),
    (r"\bbody lotion\b", "non_fragrance_variant"),
    (r"\bhand lotion\b", "non_fragrance_variant"),
    (r"\blotion\b", "non_fragrance_variant"),
    (r"\bdeodorant\b", "non_fragrance_variant"),
    (r"\bshower gel\b", "non_fragrance_variant"),
    (r"\bsample pack\b", "non_fragrance_variant"),
    (r"\bsample packs\b", "non_fragrance_variant"),
    (r"\bdiscovery set\b", "non_fragrance_variant"),
    (r"\baftershave\b", "non_fragrance_variant"),
)

VARIANT_PATTERNS = (
    r"\btravel spray\b",
    r"\brefill\b",
    r"\bcollector\b",
    r"\blimited edition bottle\b",
)


def normalize_name_key(name: str) -> str:
    return re.sub(r"\s+", " ", str(name or "").strip().lower())


def clean_text(value: str) -> str:
    return " ".join(str(value or "").replace("\r", " ").replace("\n", " ").split())


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    csv.field_size_limit(10_000_000)
    with path.open(encoding="utf-8", errors="replace", newline="") as handle:
        return list(csv.DictReader(handle))


def status_map(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for row in read_csv_rows(path):
        perfume = normalize_name_key(row.get("Perfume", ""))
        status = normalize_name_key(row.get("Status", ""))
        if perfume:
            out[perfume] = status
    return out


def perfume_names_from_rows(rows: Iterable[dict[str, str]], column: str) -> set[str]:
    return {normalize_name_key(row.get(column, "")) for row in rows if normalize_name_key(row.get(column, ""))}


def raw_perfume_names(paths: Iterable[Path]) -> set[str]:
    names: set[str] = set()
    for path in paths:
        names.update(perfume_names_from_rows(read_csv_rows(path), "Perfume"))
    return names


def frozen_perfumes() -> set[str]:
    names = perfume_names_from_rows(read_csv_rows(FROZEN_LABELS), "perfume_name")
    names.update(raw_perfume_names([FROZEN_RAW]))
    return names


def brand_key(brand: str) -> str:
    return normalize_name_key(brand)


def is_numeric_or_ambiguous(name: str) -> bool:
    text = normalize_name_key(name)
    base = re.sub(
        r"\b(eau de parfum|eau de toilette|extrait de parfum|parfum extrait|parfum|extrait|cologne|fragrance)\b",
        " ",
        text,
    )
    base = re.sub(r"[^a-z0-9]+", " ", base).strip()
    tokens = base.split()
    if not tokens:
        return True
    if len(tokens) == 1 and (tokens[0].isdigit() or len(tokens[0]) <= 2):
        return True
    if tokens[0].isdigit() and len(tokens) <= 2:
        return True
    digit_chars = sum(ch.isdigit() for ch in base)
    alpha_chars = sum(ch.isalpha() for ch in base)
    return bool(digit_chars and digit_chars >= alpha_chars)


def non_fragrance_reason(name: str) -> str:
    text = normalize_name_key(name)
    for pattern, reason in NON_FRAGRANCE_PATTERNS:
        if re.search(pattern, text):
            return reason
    if "oil" in text and not re.search(r"\bperfume oil\b|\battar\b|\bpure oud oil\b", text):
        return "non_fragrance_variant"
    return ""


def fragrance_strength(name: str) -> int:
    text = normalize_name_key(name)
    if "extrait de parfum" in text or "parfum extrait" in text:
        return 18
    if "eau de parfum" in text:
        return 16
    if re.search(r"\bparfum\b", text):
        return 14
    if "eau de toilette" in text:
        return 12
    if re.search(r"\bcologne\b", text):
        return 8
    if re.search(r"\bfragrance\b", text):
        return 6
    if "extrait" in text:
        return 6
    return 0


def make_queries(brand: str, perfume: str) -> str:
    return " | ".join(
        [
            f"{brand} {perfume} review",
            f"{brand} {perfume} perfume review",
            f"{brand} {perfume} fragrance review",
            f'"{brand}" "{perfume}" review',
            f"{perfume} {brand} review",
        ]
    )


def legacy_queue_bonus() -> set[str]:
    rows = read_csv_rows(LEGACY_QUEUE)
    return {
        normalize_name_key(row.get("perfume_name", ""))
        for row in rows
        if normalize_name_key(row.get("perfume_name", "")) and not row.get("exclusion_reason")
    }


def quality_for(
    *,
    name: str,
    brand: str,
    brand_counts: Counter[str],
    legacy_status: str,
    current_status: str,
    in_legacy_queue: bool,
) -> tuple[int, list[str]]:
    flags: list[str] = []
    score = 50
    bkey = brand_key(brand)
    brand_count = brand_counts[bkey]

    if bkey in REVIEWABLE_BRANDS:
        score += 35
        flags.append("reviewable_brand")
    if brand_count >= 20:
        score += 10
        flags.append("large_catalog_brand")
    elif brand_count >= 10:
        score += 5
        flags.append("mid_catalog_brand")

    strength = fragrance_strength(name)
    score += strength
    if strength:
        flags.append("clear_fragrance_product")
    else:
        score -= 15
        flags.append("missing_clear_fragrance_type")

    if in_legacy_queue:
        score += 8
        flags.append("previously_ranked_candidate")
    if legacy_status in {"no videos", "no_videos", "search_no_videos"}:
        score -= 12
        flags.append("legacy_no_videos")
    elif legacy_status in {"no results", "no_results", "transcript_unavailable"}:
        score -= 8
        flags.append("legacy_no_results")

    if current_status in {"quota_error", "auth_error", "api_error", "youtube_api_error"}:
        score += 4
        flags.append("retryable_current_status")

    if any(re.search(pattern, normalize_name_key(name)) for pattern in VARIANT_PATTERNS):
        score -= 10
        flags.append("low_review_variant")

    token_count = len(re.findall(r"[a-z0-9]+", normalize_name_key(name)))
    if token_count >= 3:
        score += 3
        flags.append("specific_name")

    return score, flags


def duplicate_base(name: str, brand: str) -> str:
    text = normalize_name_key(name)
    text = re.sub(
        r"\b(eau de parfum|eau de toilette|extrait de parfum|parfum extrait|parfum|extrait|cologne|fragrance|travel spray|intense|absolu|absolute)\b",
        " ",
        text,
    )
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return f"{brand_key(brand)}::{re.sub(r'\\s+', ' ', text).strip()}"


def build_rows() -> tuple[list[dict[str, str]], Counter[str]]:
    perfume_rows = read_csv_rows(FINAL_PERFUMES)
    legacy_status_by_name = status_map(LEGACY_STATUS)
    current_status_by_name = status_map(PHASE2A_STATUS)
    frozen = frozen_perfumes()
    phase2a_raw = raw_perfume_names(PHASE2A_RAW_FILES)
    legacy_bonus = legacy_queue_bonus()
    brand_counts = Counter(brand_key(row.get("Brand", "")) for row in perfume_rows if brand_key(row.get("Brand", "")))

    scored_rows: list[dict[str, str]] = []
    excluded_counts: Counter[str] = Counter()
    seen_exact: set[str] = set()

    for source_index, row in enumerate(perfume_rows):
        name = clean_text(row.get("Name", ""))
        brand = clean_text(row.get("Brand", ""))
        normalized = normalize_name_key(name)
        if not name or not brand:
            continue

        dedupe_key = f"{brand_key(brand)}::{normalized}"
        if dedupe_key in seen_exact:
            excluded_counts["duplicate_exact_name"] += 1
            continue
        seen_exact.add(dedupe_key)

        legacy_status = legacy_status_by_name.get(normalized, "")
        current_status = current_status_by_name.get(normalized, "")
        exclusion_reason = ""

        if normalized in frozen:
            exclusion_reason = "frozen_phase1_784"
        elif normalized in phase2a_raw or current_status == "done":
            exclusion_reason = "phase2a_done"
        elif current_status == "transcript_fetch_error":
            exclusion_reason = "transcript_fetch_error"
        elif current_status == "transcript_unavailable":
            exclusion_reason = "transcript_unavailable"
        elif non_fragrance_reason(name):
            exclusion_reason = non_fragrance_reason(name)
        elif is_numeric_or_ambiguous(name):
            exclusion_reason = "numeric_or_ambiguous_name"

        quality_score, flags = quality_for(
            name=name,
            brand=brand,
            brand_counts=brand_counts,
            legacy_status=legacy_status,
            current_status=current_status,
            in_legacy_queue=normalized in legacy_bonus,
        )

        if exclusion_reason:
            excluded_counts[exclusion_reason] += 1

        scored_rows.append(
            {
                "perfume_name": name,
                "brand": brand,
                "normalized_name": normalized,
                "current_status": current_status,
                "legacy_status": legacy_status,
                "quality_score": str(quality_score),
                "quality_flags": ";".join(flags),
                "exclusion_reason": exclusion_reason,
                "recommended_query_templates": make_queries(brand, name),
                "recommended_for_next_batch": "False",
                "_source_index": str(source_index),
                "_duplicate_base": duplicate_base(name, brand),
            }
        )

    eligible = [row for row in scored_rows if not row["exclusion_reason"]]
    best_by_base: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in eligible:
        best_by_base[row["_duplicate_base"]].append(row)
    for rows in best_by_base.values():
        rows.sort(key=lambda item: (-int(item["quality_score"]), int(item["_source_index"])))
        for duplicate in rows[1:]:
            duplicate["quality_score"] = str(int(duplicate["quality_score"]) - 18)
            duplicate["quality_flags"] = ";".join(
                part for part in [duplicate["quality_flags"], "duplicate_variant_penalty"] if part
            )

    scored_rows.sort(
        key=lambda item: (
            bool(item["exclusion_reason"]),
            -int(item["quality_score"]),
            brand_key(item["brand"]),
            item["normalized_name"],
        )
    )
    recommended = 0
    for row in scored_rows:
        if not row["exclusion_reason"] and recommended < 25:
            row["recommended_for_next_batch"] = "True"
            recommended += 1

    cleaned = [{key: row[key] for key in FIELDNAMES} for row in scored_rows]
    return cleaned, excluded_counts


def main() -> None:
    rows, excluded_counts = build_rows()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    eligible = sum(1 for row in rows if not row["exclusion_reason"])
    recommended = sum(1 for row in rows if row["recommended_for_next_batch"] == "True")
    print(f"wrote={OUTPUT_PATH}")
    print(f"total_rows={total}")
    print(f"eligible_rows={eligible}")
    print(f"recommended_for_next_batch={recommended}")
    print("excluded_counts=" + dict(excluded_counts).__repr__())


if __name__ == "__main__":
    main()

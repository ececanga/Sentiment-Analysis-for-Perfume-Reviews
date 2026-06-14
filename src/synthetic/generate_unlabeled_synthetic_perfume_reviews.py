# Date Created: 2026-06-08
# Date Last Modified: 2026-06-08

"""Generate unlabeled synthetic perfume reviews for later annotation.

This is the correct synthetic-data entry point when labels should be assigned
after generation. It uses only perfume identity fields, not emotion/VAD labels.
"""

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
from statistics import mean
from typing import Any, Dict, List, Sequence

from src.utils.paths import PROJECT_ROOT, data_path, labels_path

csv.field_size_limit(sys.maxsize)

RAW_COMPAT_COLUMNS = [
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
]

PROVENANCE_COLUMNS = [
    "source_type",
    "generation_policy",
    "synthetic_review_id",
    "perfume_id",
    "perfume_id_hash",
    "review_index",
    "synthetic_seed",
    "source_identity_file",
    "persona",
    "tone_profile",
    "length_profile",
    "accord_tags",
]

EMOTION_KEYWORDS = {
    "love",
    "hate",
    "beautiful",
    "fresh",
    "intoxicating",
    "romantic",
    "cozy",
    "sensual",
    "clean",
    "dark",
    "bright",
    "calm",
    "uplifting",
    "warm",
    "inviting",
    "alluring",
    "boring",
    "powerful",
    "peaceful",
    "happy",
    "sad",
    "serene",
    "energetic",
    "confusing",
    "disappointing",
    "comforting",
}

ACCORD_HINTS: Dict[str, List[str]] = {
    "rose": ["rose", "soft floral petals"],
    "roses": ["rose", "soft floral petals"],
    "florence": ["floral", "clean white flowers"],
    "gardenia": ["floral", "creamy white flowers"],
    "jasmine": ["floral", "warm jasmine"],
    "tuberose": ["floral", "creamy tuberose"],
    "osmanthe": ["floral", "apricot osmanthus"],
    "osmanthus": ["floral", "apricot osmanthus"],
    "amber": ["amber", "warm resin"],
    "ambre": ["amber", "warm resin"],
    "ambra": ["amber", "warm resin"],
    "vanilla": ["gourmand", "soft vanilla"],
    "cacao": ["gourmand", "cocoa sweetness"],
    "almond": ["gourmand", "almond cream"],
    "sugar": ["gourmand", "powdery sweetness"],
    "zucchero": ["gourmand", "powdery sweetness"],
    "honey": ["gourmand", "golden honey"],
    "oud": ["oud", "dark woods"],
    "oudh": ["oud", "dark woods"],
    "aoud": ["oud", "dark woods"],
    "wood": ["woody", "dry woods"],
    "woods": ["woody", "dry woods"],
    "bois": ["woody", "dry woods"],
    "cedar": ["woody", "cedar shavings"],
    "santal": ["woody", "smooth sandalwood"],
    "sandal": ["woody", "smooth sandalwood"],
    "cuir": ["leather", "polished leather"],
    "leather": ["leather", "polished leather"],
    "tobacco": ["tobacco", "warm tobacco"],
    "tabac": ["tobacco", "warm tobacco"],
    "incense": ["incense", "dry incense smoke"],
    "smoke": ["smoky", "faint smoke"],
    "bergamot": ["citrus", "sparkling bergamot"],
    "lemon": ["citrus", "bright lemon peel"],
    "orange": ["citrus", "sweet orange peel"],
    "mandarin": ["citrus", "juicy mandarin"],
    "grapefruit": ["citrus", "bitter grapefruit"],
    "aqua": ["aquatic", "clean watery air"],
    "acqua": ["aquatic", "clean watery air"],
    "ocean": ["marine", "salty air"],
    "sale": ["marine", "salted skin"],
    "sea": ["marine", "salty air"],
    "bamboo": ["green", "green watery stems"],
    "green": ["green", "crushed leaves"],
    "tea": ["aromatic", "soft tea steam"],
    "lavender": ["aromatic", "cool lavender"],
    "rosemary": ["aromatic", "herbal rosemary"],
    "pepper": ["spicy", "dry pepper"],
    "cardamom": ["spicy", "cool cardamom"],
    "saffron": ["spicy", "saffron warmth"],
    "cinnamon": ["spicy", "sweet spice"],
    "musk": ["musk", "clean musk"],
    "fig": ["fruity", "green fig"],
    "coconut": ["fruity", "creamy coconut"],
    "mango": ["fruity", "ripe fruit"],
}

GENERIC_ACCORDS = [
    ["woody", "soft woods"],
    ["amber", "warm resin"],
    ["floral", "muted petals"],
    ["musk", "clean musk"],
    ["spicy", "gentle spice"],
    ["citrus", "fresh brightness"],
]

PERSONAS = {
    "casual_wearer": {
        "openers": [
            "I tried {perfume_name} over a normal day and paid attention to how it felt after the opening.",
            "{perfume_name} was easy to test because I could imagine wearing it outside, not only smelling it on paper.",
        ],
        "focus": [
            "I mostly noticed whether it felt comfortable, noticeable, and easy to return to.",
            "My reaction was practical: would I reach for this again, and would the mood fit my day?",
        ],
    },
    "note_focused_reviewer": {
        "openers": [
            "With {perfume_name}, I started by following the note texture from the first spray to the drydown.",
            "I tested {perfume_name} slowly because the material impression changed as it settled.",
        ],
        "focus": [
            "The review is more about texture and balance than about compliments.",
            "I kept comparing the opening, heart, and base because each stage gave a different emphasis.",
        ],
    },
    "niche_collector": {
        "openers": [
            "{perfume_name} feels like something I would judge against other niche releases, not against designer basics.",
            "I approached {perfume_name} as a collector, looking for character rather than simple likability.",
        ],
        "focus": [
            "Originality matters here, but wearability still matters once the initial idea settles.",
            "I can respect a bold idea, but I still need the perfume to make emotional sense on skin.",
        ],
    },
    "compliment_seeker": {
        "openers": [
            "I wore {perfume_name} thinking about how it might come across around other people.",
            "For {perfume_name}, I paid attention to projection, approachability, and whether the mood felt inviting.",
        ],
        "focus": [
            "The scent needs enough presence to be noticed without becoming tiring.",
            "I kept asking whether it feels friendly, memorable, and easy for someone else to enjoy nearby.",
        ],
    },
    "critical_minimalist": {
        "openers": [
            "I tested {perfume_name} with a fairly strict standard because I prefer clear, uncluttered perfumes.",
            "{perfume_name} had to convince me without relying on loudness or novelty.",
        ],
        "focus": [
            "I noticed the parts that felt useful and the parts that felt unnecessary.",
            "The simpler it became, the easier it was to decide whether the composition worked.",
        ],
    },
}

TONE_PROFILES = {
    "mostly_positive": {
        "weight": 0.52,
        "mood_words": ["pleasant", "warm", "easygoing", "bright", "comforting", "polished"],
        "judgment": [
            "The final impression is positive, even if I would still choose the right season for it.",
            "I can see this being useful because the mood stays approachable and pleasant.",
            "It leaves me with a good impression, especially after the sharper parts calm down.",
        ],
    },
    "neutral_descriptive": {
        "weight": 0.28,
        "mood_words": ["balanced", "quiet", "measured", "soft", "restrained", "steady"],
        "judgment": [
            "The final impression is balanced rather than strongly emotional.",
            "I understand the style, but it does not push me strongly toward love or dislike.",
            "It feels more like a careful composition than a dramatic personal favorite.",
        ],
    },
    "mixed": {
        "weight": 0.14,
        "mood_words": ["interesting", "uneven", "unusual", "contrasting", "restless", "layered"],
        "judgment": [
            "I respect the idea, but my reaction changes depending on which part is most noticeable.",
            "Some moments are attractive, while others feel harder to wear.",
            "It is memorable, but not completely easy.",
        ],
    },
    "critical": {
        "weight": 0.06,
        "mood_words": ["sharp", "heavy", "flat", "awkward", "dry", "difficult"],
        "judgment": [
            "The final impression is more difficult than enjoyable for my taste.",
            "I can describe what it is doing, but I do not feel drawn to wear it again soon.",
            "It has character, but the balance does not fully work for me.",
        ],
    },
}

LENGTH_PROFILES = {
    "short": {"weight": 0.25, "extra_sentences": 1},
    "medium": {"weight": 0.55, "extra_sentences": 3},
    "long": {"weight": 0.20, "extra_sentences": 5},
}

EXTRA_SENTENCES = [
    "The opening is the most direct part, while the base feels slower and less obvious.",
    "On paper it reads cleaner, but on skin it becomes more personal.",
    "I would not judge it only from the first minute because the drydown changes the context.",
    "The projection feels controlled, so the mood sits close rather than filling the room.",
    "It has enough texture to talk about, even when the structure is not complicated.",
    "The sweet and dry parts seem to pull in different directions.",
    "There is a wearable side, but it still has a specific personality.",
    "I would choose it more carefully than a simple everyday fragrance.",
    "The name sets one expectation, but the actual wearing experience feels more practical.",
    "It works best when I stop looking for a single dominant note and follow the overall atmosphere.",
]


@dataclass(frozen=True)
class PerfumeIdentity:
    perfume_id: str
    perfume_id_hash: str
    perfume_name: str


def stringify_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def stable_int(*parts: object) -> int:
    raw = ":".join(str(part) for part in parts)
    return int(hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12], 16)


def clean_whitespace(text: object) -> str:
    return " ".join(str(text or "").replace("\r", " ").replace("\n", " ").split())


def tokenize_name(name: str) -> List[str]:
    return [tok for tok in re.split(r"[^a-z0-9]+", str(name).lower()) if tok]


def infer_accords(perfume_name: str, rng: random.Random, max_tags: int = 4) -> List[str]:
    tags: List[str] = []
    descriptors: List[str] = []
    for token in tokenize_name(perfume_name):
        hint = ACCORD_HINTS.get(token)
        if not hint:
            continue
        tag, descriptor = hint
        if tag not in tags:
            tags.append(tag)
            descriptors.append(descriptor)
    if not descriptors:
        fallback = rng.sample(GENERIC_ACCORDS, k=3)
        descriptors = [item[1] for item in fallback]
    return descriptors[:max_tags]


def weighted_choice(weights: Dict[str, float], rng: random.Random) -> str:
    total = sum(weights.values())
    point = rng.random() * total
    upto = 0.0
    for key, weight in weights.items():
        upto += weight
        if upto >= point:
            return key
    return next(reversed(weights))


def read_perfume_identities(path: Path) -> List[PerfumeIdentity]:
    if not path.exists():
        raise FileNotFoundError(f"Identity CSV not found: {path}")
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"perfume_id", "perfume_id_hash", "perfume_name"}
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"Missing required identity columns in {path}: {missing}")
        identities: List[PerfumeIdentity] = []
        seen: set[str] = set()
        for row in reader:
            perfume_id = clean_whitespace(row.get("perfume_id", ""))
            perfume_name = clean_whitespace(row.get("perfume_name", ""))
            if not perfume_id or not perfume_name:
                continue
            if perfume_id in seen:
                raise ValueError(f"Duplicate perfume_id in identity CSV: {perfume_id}")
            seen.add(perfume_id)
            identities.append(
                PerfumeIdentity(
                    perfume_id=perfume_id,
                    perfume_id_hash=clean_whitespace(row.get("perfume_id_hash", "")),
                    perfume_name=perfume_name,
                )
            )
    return identities


def emotion_keyword_count(text: str) -> int:
    tokens = re.findall(r"[a-z]+", str(text).lower())
    return sum(1 for token in tokens if token in EMOTION_KEYWORDS)


def describe_accords(accords: Sequence[str]) -> str:
    if len(accords) == 1:
        return accords[0]
    return ", ".join(accords[:-1]) + f", and {accords[-1]}"


def build_unlabeled_review(identity: PerfumeIdentity, review_index: int, seed: int) -> Dict[str, Any]:
    rng = random.Random(stable_int(seed, identity.perfume_id, review_index, "unlabeled"))
    persona = rng.choice(list(PERSONAS))
    tone = weighted_choice({key: spec["weight"] for key, spec in TONE_PROFILES.items()}, rng)
    length = weighted_choice({key: spec["weight"] for key, spec in LENGTH_PROFILES.items()}, rng)

    persona_spec = PERSONAS[persona]
    tone_spec = TONE_PROFILES[tone]
    length_spec = LENGTH_PROFILES[length]

    accords = infer_accords(identity.perfume_name, rng)
    if not accords:
        fallback = rng.sample(GENERIC_ACCORDS, k=3)
        accords = [item[1] for item in fallback]

    mood = rng.choice(tone_spec["mood_words"])
    sentences = [
        rng.choice(persona_spec["openers"]).format(perfume_name=identity.perfume_name),
        f"The main impression is {describe_accords(accords)}, and the mood comes across as {mood}.",
        rng.choice(persona_spec["focus"]),
    ]
    sentences.extend(rng.sample(EXTRA_SENTENCES, k=min(length_spec["extra_sentences"], len(EXTRA_SENTENCES))))
    sentences.append(rng.choice(tone_spec["judgment"]))

    text = clean_whitespace(" ".join(sentences))
    review_id = f"unlabeled-synthetic-{identity.perfume_id}-{review_index:02d}"
    return {
        "Perfume": identity.perfume_name,
        "Video ID": review_id,
        "Title": f"Unlabeled synthetic review {review_index} of {identity.perfume_name}",
        "Description": "Unlabeled synthetic perfume review for later annotation; no target labels were used.",
        "URL": f"synthetic://unlabeled-perfume-review/{identity.perfume_id}/{review_index}",
        "Channel Title": "Unlabeled Synthetic Review Generator",
        "Transcript": text,
        "Transcript Length": len(text.split()),
        "Emotion Keyword Count": emotion_keyword_count(text),
        "Final Transcript": text,
        "source_type": "unlabeled_synthetic_template_v1",
        "generation_policy": "identity_only_persona_tone_template_no_emotion_labels",
        "synthetic_review_id": review_id,
        "perfume_id": identity.perfume_id,
        "perfume_id_hash": identity.perfume_id_hash,
        "review_index": review_index,
        "synthetic_seed": seed,
        "source_identity_file": "",
        "persona": persona,
        "tone_profile": tone,
        "length_profile": length,
        "accord_tags": ";".join(accords),
    }


def generate_rows(identities: Sequence[PerfumeIdentity], reviews_per_perfume: int, seed: int, source_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for identity in identities:
        for review_index in range(1, reviews_per_perfume + 1):
            row = build_unlabeled_review(identity, review_index, seed)
            row["source_identity_file"] = stringify_path(source_path)
            rows.append(row)
    return rows


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fieldnames = RAW_COMPAT_COLUMNS + PROVENANCE_COLUMNS
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(
    *,
    path: Path,
    source_csv: Path,
    output_csv: Path,
    identities: Sequence[PerfumeIdentity],
    rows: Sequence[Dict[str, Any]],
    reviews_per_perfume: int,
    seed: int,
) -> None:
    word_counts = [int(row["Transcript Length"]) for row in rows]
    summary = {
        "dataset_version": "unlabeled_synthetic_perfume_reviews_784_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_identity_file": stringify_path(source_csv),
        "output_csv": stringify_path(output_csv),
        "source_type": "unlabeled_synthetic_template_v1",
        "generation_policy": "identity_only_persona_tone_template_no_emotion_labels",
        "n_perfumes": len(identities),
        "reviews_per_perfume": reviews_per_perfume,
        "n_synthetic_review_rows": len(rows),
        "seed": seed,
        "word_count_min": min(word_counts) if word_counts else 0,
        "word_count_mean": round(mean(word_counts), 3) if word_counts else 0,
        "word_count_max": max(word_counts) if word_counts else 0,
        "persona_counts": {key: sum(row["persona"] == key for row in rows) for key in PERSONAS},
        "tone_profile_counts": {key: sum(row["tone_profile"] == key for row in rows) for key in TONE_PROFILES},
        "length_profile_counts": {key: sum(row["length_profile"] == key for row in rows) for key in LENGTH_PROFILES},
        "warning": (
            "No emotion or VAD labels were used to generate this dataset. "
            "Use Final Transcript for later annotation; keep provenance columns out of model features."
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate unlabeled synthetic perfume reviews for later labeling.")
    parser.add_argument(
        "--source-csv",
        type=Path,
        default=labels_path("gemini_labels_784_judge_g25pro_v1.csv"),
        help="CSV containing perfume_id, perfume_id_hash, and perfume_name. Label columns are ignored.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=data_path("synthetic", "synthetic_perfume_reviews_784_unlabeled_v1.csv"),
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=data_path("synthetic", "synthetic_perfume_reviews_784_unlabeled_v1_summary.json"),
    )
    parser.add_argument("--reviews-per-perfume", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--expected-perfumes", type=int, default=784)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.reviews_per_perfume <= 0:
        raise ValueError("--reviews-per-perfume must be positive.")
    if not args.overwrite:
        existing = [path for path in [args.output_csv, args.summary_json] if path.exists()]
        if existing:
            raise FileExistsError(
                "Output already exists; pass --overwrite to replace: "
                + ", ".join(str(path) for path in existing)
            )

    identities = read_perfume_identities(args.source_csv)
    if args.expected_perfumes and len(identities) != args.expected_perfumes:
        raise ValueError(f"Expected {args.expected_perfumes} perfumes, found {len(identities)}.")
    if args.limit is not None:
        identities = identities[: args.limit]

    rows = generate_rows(identities, args.reviews_per_perfume, args.seed, args.source_csv)
    write_csv(args.output_csv, rows)
    write_summary(
        path=args.summary_json,
        source_csv=args.source_csv,
        output_csv=args.output_csv,
        identities=identities,
        rows=rows,
        reviews_per_perfume=args.reviews_per_perfume,
        seed=args.seed,
    )

    print(f"Wrote unlabeled synthetic review rows: {len(rows)}")
    print(f"Wrote CSV: {args.output_csv}")
    print(f"Wrote summary: {args.summary_json}")


if __name__ == "__main__":
    main()

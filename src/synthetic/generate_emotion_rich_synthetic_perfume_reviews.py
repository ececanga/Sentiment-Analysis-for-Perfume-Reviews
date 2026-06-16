# Date Created: 2026-06-12

"""Generate emotion-rich unlabeled synthetic perfume reviews.

This v2 generator keeps the synthetic reviews unlabeled, but makes the review
language more human, sensory, and subjective than the v1 template generator.
It uses only perfume identities plus local note-like/context keywords. Existing
emotion/VAD labels in the identity source are ignored.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Sequence

import pandas as pd

from src.utils.paths import PROJECT_ROOT, data_path, labels_path

csv.field_size_limit(sys.maxsize)

DATASET_VERSION = "unlabeled_synthetic_perfume_reviews_784_emotion_rich_v2"
SOURCE_TYPE = "unlabeled_synthetic_emotion_rich_v2"
GENERATION_POLICY = "local_note_inspired_human_scenario_templates_no_label_conditioning"
DEFAULT_SEED = 20260612

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
    "dataset_version",
    "generation_policy",
    "synthetic_review_id",
    "perfume_id",
    "perfume_id_hash",
    "review_index",
    "synthetic_seed",
    "source_identity_file",
    "note_inspiration_file",
    "persona",
    "style_family",
    "scenario_family",
    "length_profile",
    "sensory_source",
    "sensory_terms",
    "accord_tags",
]

LABEL_COLUMNS = {
    "joy",
    "sadness",
    "anger",
    "fear",
    "trust",
    "surprise",
    "anticipation",
    "disgust",
    "valence",
    "arousal",
    "dominance",
}

EMOTION_KEYWORDS = {
    "adore",
    "afraid",
    "annoying",
    "attached",
    "awkward",
    "calm",
    "calmer",
    "comfort",
    "comforted",
    "comforting",
    "confident",
    "cozy",
    "disappointed",
    "disappointing",
    "dislike",
    "dreamy",
    "excited",
    "fascinated",
    "feel",
    "feeling",
    "felt",
    "gentle",
    "happy",
    "headache",
    "hesitate",
    "hopeful",
    "joy",
    "love",
    "loved",
    "mood",
    "nauseous",
    "nervous",
    "nostalgic",
    "peaceful",
    "regret",
    "relaxed",
    "safe",
    "soft",
    "surprised",
    "tense",
    "tender",
    "tiring",
    "unexpected",
    "unexpectedly",
    "uneasy",
    "warm",
}


@dataclass(frozen=True)
class PerfumeIdentity:
    perfume_id: str
    perfume_id_hash: str
    perfume_name: str


@dataclass(frozen=True)
class ScentHint:
    tag: str
    phrase: str
    color: str
    sound: str
    place: str
    texture: str


def hint(tag: str, phrase: str, color: str, sound: str, place: str, texture: str) -> ScentHint:
    return ScentHint(tag=tag, phrase=phrase, color=color, sound=sound, place=place, texture=texture)


CURATED_HINTS: Dict[str, ScentHint] = {
    "aldehydic": hint("aldehydic", "sparkling aldehydes", "white glare", "glass clinking", "a bright dressing room", "starched fabric"),
    "almond": hint("almond", "soft almond cream", "warm ivory", "a spoon touching porcelain", "a small bakery", "powdery cream"),
    "amber": hint("amber", "warm amber resin", "orange lamp light", "a low cello note", "a quiet evening room", "thick warmth"),
    "ambery": hint("amber", "warm amber resin", "orange lamp light", "a low cello note", "a quiet evening room", "thick warmth"),
    "ambrette": hint("musk", "soft ambrette musk", "pale beige", "cotton brushing skin", "fresh sheets", "clean powder"),
    "aquatic": hint("aquatic", "clean watery air", "pale blue", "distant waves", "a hotel balcony", "cool mist"),
    "bergamot": hint("citrus", "sparkling bergamot peel", "clear yellow", "a bicycle bell", "a sunny kitchen", "zesty brightness"),
    "benzoin": hint("resin", "sweet benzoin warmth", "honey brown", "wood softly creaking", "a dim cafe", "sticky resin"),
    "bitter": hint("bitter", "a bitter green edge", "olive green", "dry leaves scraping", "a shaded garden", "dry bite"),
    "boozy": hint("boozy", "dark boozy warmth", "deep burgundy", "ice in a glass", "a late hotel bar", "velvet heat"),
    "cacao": hint("gourmand", "dark cocoa", "matte brown", "a cafe spoon", "a winter kitchen", "dusty chocolate"),
    "caramel": hint("gourmand", "burnt caramel", "golden brown", "sugar cracking", "a small dessert shop", "sticky sweetness"),
    "cardamom": hint("spice", "cool cardamom", "soft green", "paper tearing", "a spice drawer", "dry cool spice"),
    "cedar": hint("wood", "dry cedar shavings", "pale wood", "a drawer sliding open", "a quiet study", "dry grain"),
    "cedarwood": hint("wood", "dry cedar shavings", "pale wood", "a drawer sliding open", "a quiet study", "dry grain"),
    "cherry": hint("fruit", "dark cherry syrup", "red lacquer", "a spoon tapping glass", "a crowded dessert table", "glossy sweetness"),
    "chocolate": hint("gourmand", "dark chocolate", "matte brown", "a cafe spoon", "a winter kitchen", "dusty cocoa"),
    "cinnamon": hint("spice", "sweet cinnamon warmth", "rust orange", "a kettle starting to hum", "a holiday kitchen", "warm powder"),
    "citrus": hint("citrus", "bright citrus peel", "clear yellow", "a bicycle bell", "a sunny kitchen", "zesty brightness"),
    "clean": hint("clean", "clean laundry musk", "white cotton", "sheets snapping in air", "a sunny laundry room", "smooth cotton"),
    "coconut": hint("coconut", "creamy coconut", "sunlit cream", "water against tiles", "a quiet poolside", "milky softness"),
    "coffee": hint("coffee", "dark coffee bitterness", "espresso brown", "cups on a saucer", "a morning cafe", "roasted grain"),
    "cream": hint("cream", "soft creamy sweetness", "warm cream", "a spoon through custard", "a calm kitchen", "round softness"),
    "creamy": hint("cream", "soft creamy sweetness", "warm cream", "a spoon through custard", "a calm kitchen", "round softness"),
    "fig": hint("fig", "green fig skin", "leafy green", "wind in broad leaves", "a shaded terrace", "milky greenness"),
    "floral": hint("floral", "soft floral petals", "dusty pink", "quiet footsteps on carpet", "a florist window", "petal softness"),
    "flowers": hint("floral", "soft floral petals", "dusty pink", "quiet footsteps on carpet", "a florist window", "petal softness"),
    "gardenia": hint("white floral", "creamy gardenia", "ivory white", "silk moving", "a warm conservatory", "thick petals"),
    "ginger": hint("spice", "fresh ginger bite", "golden yellow", "a match striking", "a bright market stall", "sharp warmth"),
    "gourmand": hint("gourmand", "edible gourmand sweetness", "caramel gold", "wrappers rustling", "a bakery counter", "syrupy softness"),
    "grapefruit": hint("citrus", "bitter grapefruit peel", "pale yellow", "soda fizz", "a balcony at noon", "sharp sparkle"),
    "green": hint("green", "crushed green stems", "wet green", "grass being cut", "a garden after rain", "leafy snap"),
    "honey": hint("honey", "golden honey", "deep gold", "slow dripping", "a sun-warmed table", "sticky glow"),
    "incense": hint("incense", "dry incense smoke", "grey violet", "a door closing softly", "a stone chapel", "thin smoke"),
    "iris": hint("iris", "powdery iris", "mauve grey", "makeup brushes tapping", "an old vanity table", "dry powder"),
    "jasmine": hint("white floral", "warm jasmine", "creamy white", "night insects outside", "a humid garden at night", "heady petals"),
    "labdanum": hint("resin", "dark labdanum resin", "brown amber", "wood softly creaking", "a dim room", "sticky resin"),
    "lavender": hint("aromatic", "cool lavender", "dusty purple", "linen folding", "a clean bedroom", "herbal softness"),
    "leather": hint("leather", "polished leather", "black brown", "boots on pavement", "a coat closet", "smooth hide"),
    "leathery": hint("leather", "polished leather", "black brown", "boots on pavement", "a coat closet", "smooth hide"),
    "lemon": hint("citrus", "bright lemon peel", "clear yellow", "a knife on a cutting board", "a morning kitchen", "zesty brightness"),
    "mango": hint("fruit", "ripe mango", "orange gold", "market chatter", "a humid fruit stall", "juicy pulp"),
    "marine": hint("marine", "salty sea air", "blue grey", "waves under a pier", "a windy coast", "salt on skin"),
    "mint": hint("mint", "cool mint leaves", "icy green", "ice cracking", "a shaded glass of water", "cold snap"),
    "musk": hint("musk", "clean skin musk", "soft beige", "cotton brushing skin", "fresh sheets", "smooth powder"),
    "musky": hint("musk", "clean skin musk", "soft beige", "cotton brushing skin", "fresh sheets", "smooth powder"),
    "neroli": hint("citrus floral", "bitter neroli", "white gold", "morning traffic far away", "a Mediterranean street", "soapy petals"),
    "oakmoss": hint("moss", "dry oakmoss", "forest green", "twigs under shoes", "an old garden wall", "mossy dryness"),
    "orange": hint("citrus", "sweet orange peel", "orange light", "soda fizz", "a kitchen in sunlight", "juicy zest"),
    "osmanthus": hint("floral fruit", "apricot osmanthus", "soft apricot", "silk moving", "a sunlit scarf drawer", "velvety fruit"),
    "oud": hint("oud", "dark oud wood", "near black", "a heavy door closing", "a shadowed room", "dense wood"),
    "oudh": hint("oud", "dark oud wood", "near black", "a heavy door closing", "a shadowed room", "dense wood"),
    "patchouli": hint("patchouli", "earthy patchouli", "dark green brown", "soil under rain", "a vintage shop", "damp earth"),
    "peach": hint("fruit", "soft peach skin", "blush peach", "summer fabric moving", "a warm window", "velvety fruit"),
    "pepper": hint("spice", "dry black pepper", "charcoal grey", "peppercorns cracking", "a busy kitchen", "dry prickle"),
    "powder": hint("powder", "soft cosmetic powder", "pale pink", "a compact clicking shut", "an old vanity table", "dry softness"),
    "powdery": hint("powder", "soft cosmetic powder", "pale pink", "a compact clicking shut", "an old vanity table", "dry softness"),
    "resin": hint("resin", "warm resin", "amber brown", "wood softly creaking", "a dim room", "sticky depth"),
    "resinous": hint("resin", "warm resin", "amber brown", "wood softly creaking", "a dim room", "sticky depth"),
    "rose": hint("rose", "jammy rose petals", "deep pink", "a vase being moved", "a florist window", "velvet petals"),
    "roses": hint("rose", "jammy rose petals", "deep pink", "a vase being moved", "a florist window", "velvet petals"),
    "saffron": hint("spice", "saffron warmth", "burnt gold", "silk rustling", "a fabric shop", "dry warmth"),
    "salt": hint("salt", "salted skin", "pale blue grey", "waves under a pier", "a windy coast", "mineral salt"),
    "salty": hint("salt", "salted skin", "pale blue grey", "waves under a pier", "a windy coast", "mineral salt"),
    "sandalwood": hint("wood", "smooth sandalwood", "pale tan", "wood being sanded", "a quiet studio", "creamy wood"),
    "smoke": hint("smoke", "faint smoke", "soft grey", "embers settling", "a fireplace after midnight", "dry haze"),
    "smoky": hint("smoke", "faint smoke", "soft grey", "embers settling", "a fireplace after midnight", "dry haze"),
    "spice": hint("spice", "warm spice", "rust orange", "a match striking", "a spice drawer", "dry warmth"),
    "spicy": hint("spice", "warm spice", "rust orange", "a match striking", "a spice drawer", "dry warmth"),
    "sugar": hint("gourmand", "powdered sugar", "white gold", "paper pastry bags", "a bakery counter", "soft sweetness"),
    "tea": hint("tea", "soft tea steam", "pale green", "a kettle cooling", "a quiet table", "transparent warmth"),
    "tobacco": hint("tobacco", "warm tobacco leaf", "dark honey", "paper slowly folding", "an old armchair", "dry sweetness"),
    "tonka": hint("tonka", "creamy tonka", "warm beige", "a spoon through custard", "a soft sweater", "powdered cream"),
    "tuberose": hint("white floral", "creamy tuberose", "ivory white", "silk moving", "a humid garden", "thick petals"),
    "vanilla": hint("vanilla", "soft vanilla", "warm cream", "a spoon through custard", "a winter kitchen", "creamy sweetness"),
    "vetiver": hint("vetiver", "dry vetiver roots", "olive brown", "dry grass underfoot", "a hot field", "rooty dryness"),
    "violet": hint("violet", "powdery violet", "soft purple", "paper flowers moving", "a vintage drawer", "dry petals"),
    "violets": hint("violet", "powdery violet", "soft purple", "paper flowers moving", "a vintage drawer", "dry petals"),
    "wood": hint("wood", "dry woods", "pale brown", "a drawer sliding open", "a quiet study", "dry grain"),
    "woods": hint("wood", "dry woods", "pale brown", "a drawer sliding open", "a quiet study", "dry grain"),
    "woody": hint("wood", "dry woods", "pale brown", "a drawer sliding open", "a quiet study", "dry grain"),
}

NAME_HINT_ALIASES = {
    "aoud": "oud",
    "ambre": "amber",
    "ambra": "amber",
    "acqua": "aquatic",
    "aqua": "aquatic",
    "sale": "salt",
    "cuir": "leather",
    "bois": "wood",
    "bergamote": "bergamot",
    "bergamotto": "bergamot",
    "zucchero": "sugar",
    "cacao": "cacao",
    "osmanthe": "osmanthus",
    "santal": "sandalwood",
    "tabac": "tobacco",
}

GENERIC_HINTS = [
    CURATED_HINTS["musk"],
    CURATED_HINTS["wood"],
    CURATED_HINTS["amber"],
    CURATED_HINTS["floral"],
    CURATED_HINTS["citrus"],
    CURATED_HINTS["spice"],
    CURATED_HINTS["clean"],
    CURATED_HINTS["powder"],
]

PERSONAS = [
    "casual_daily_wearer",
    "sensitive_nose",
    "niche_sampler",
    "memory_led_reviewer",
    "social_wearer",
    "quiet_observer",
    "collector_with_boundaries",
]

STYLE_FAMILIES = {
    "short_fragrantica_note": {"weight": 0.18, "length": "short"},
    "reflective_mini_review": {"weight": 0.28, "length": "medium"},
    "scene_based_review": {"weight": 0.23, "length": "medium"},
    "casual_blunt_review": {"weight": 0.16, "length": "short"},
    "long_personal_review": {"weight": 0.15, "length": "long"},
}

SCENARIO_SEQUENCE = [
    "sensory_memory",
    "comfort_routine",
    "surprising_drydown",
    "difficult_reaction",
    "social_identity",
]

OPTIONAL_DETAILS = [
    "On skin it became more personal than it seemed on paper.",
    "I kept checking my wrist, not because it was loud, but because the mood kept shifting.",
    "The opening and the drydown almost felt like two different opinions from the same person.",
    "It is not a simple compliment scent, but it does create a clear atmosphere.",
    "I can imagine someone loving this for exactly the reason another person would avoid it.",
    "The texture matters here as much as the smell itself.",
    "It made more sense after an hour, when the sharper edges had settled.",
    "I would not call it easy, but I would call it memorable.",
    "There is a little tension between comfort and strangeness.",
    "It feels tied to a specific weather, not to every day.",
    "The image in my head was stronger than the actual note list.",
    "I liked some moments more than others, which made the wearing feel alive.",
]

SENTENCE_STARTERS = [
    "For me,",
    "Strangely,",
    "At first,",
    "After a while,",
    "On my skin,",
    "In the air around me,",
    "By the drydown,",
]


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


def normalize_name_key(name: object) -> str:
    return re.sub(r"\s+", " ", str(name or "").strip().lower())


def tokenize_name(name: str) -> List[str]:
    return [tok for tok in re.split(r"[^a-z0-9]+", str(name).lower()) if tok]


def weighted_choice(weights: Dict[str, float], rng: random.Random) -> str:
    total = sum(weights.values())
    point = rng.random() * total
    upto = 0.0
    for key, weight in weights.items():
        upto += weight
        if upto >= point:
            return key
    return next(reversed(weights))


def dedupe_hints(hints: Iterable[ScentHint]) -> List[ScentHint]:
    seen: set[str] = set()
    out: List[ScentHint] = []
    for item in hints:
        if item.phrase in seen:
            continue
        seen.add(item.phrase)
        out.append(item)
    return out


def describe_list(values: Sequence[str]) -> str:
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} and {values[1]}"
    return ", ".join(values[:-1]) + f", and {values[-1]}"


def emotion_keyword_count(text: str) -> int:
    tokens = re.findall(r"[a-z]+", str(text).lower())
    return sum(1 for token in tokens if token in EMOTION_KEYWORDS)


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


def load_context_hints(path: Path) -> Dict[str, List[ScentHint]]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if df.empty:
        return {}
    name_col = df.columns[0]
    available_terms = [term for term in CURATED_HINTS if term in df.columns]
    out: Dict[str, List[ScentHint]] = {}
    for _, row in df.iterrows():
        name_key = normalize_name_key(row.get(name_col, ""))
        if not name_key:
            continue
        scored: List[tuple[int, str]] = []
        for term in available_terms:
            try:
                count = int(row.get(term, 0))
            except (TypeError, ValueError):
                count = 0
            if count > 0:
                scored.append((count, term))
        scored.sort(key=lambda item: (-item[0], item[1]))
        out[name_key] = dedupe_hints(CURATED_HINTS[term] for _, term in scored[:8])
    return out


def name_based_hints(perfume_name: str) -> List[ScentHint]:
    hints: List[ScentHint] = []
    for token in tokenize_name(perfume_name):
        term = token if token in CURATED_HINTS else NAME_HINT_ALIASES.get(token)
        if term and term in CURATED_HINTS:
            hints.append(CURATED_HINTS[term])
    return dedupe_hints(hints)


def choose_scent_hints(
    *,
    identity: PerfumeIdentity,
    context_hints: Dict[str, List[ScentHint]],
    rng: random.Random,
    min_count: int = 3,
    max_count: int = 5,
) -> tuple[List[ScentHint], str]:
    context = context_hints.get(normalize_name_key(identity.perfume_name), [])
    from_name = name_based_hints(identity.perfume_name)
    source = "context_keywords" if context else "name_or_generic"
    combined = dedupe_hints([*context, *from_name])
    if len(combined) < min_count:
        combined = dedupe_hints([*combined, *rng.sample(GENERIC_HINTS, k=len(GENERIC_HINTS))])
    count = min(max_count, max(min_count, len(combined)))
    if len(combined) > count:
        combined = rng.sample(combined, k=count)
    return combined, source


def first_hint(hints: Sequence[ScentHint], tag: str | None = None) -> ScentHint:
    if tag is not None:
        for item in hints:
            if item.tag == tag:
                return item
    return hints[0]


def scenario_sentences(
    *,
    scenario: str,
    perfume_name: str,
    hints: Sequence[ScentHint],
    persona: str,
    rng: random.Random,
) -> List[str]:
    main = first_hint(hints)
    second = hints[1] if len(hints) > 1 else main
    third = hints[2] if len(hints) > 2 else second
    phrases = describe_list([item.phrase for item in hints[:3]])

    if scenario == "sensory_memory":
        return [
            f"{perfume_name} immediately gave me a picture rather than a note list: {phrases}.",
            f"The color in my head was {main.color}, placed around {second.place}, with {third.sound} somewhere in the background.",
            rng.choice(
                [
                    "It felt nostalgic without being sweet about it, like remembering a place more than wanting to return there.",
                    "That image made me feel unexpectedly tender toward it, even though part of the scent stayed distant.",
                    "I did not feel simple happiness, more like a small private memory opening and closing.",
                ]
            ),
        ]

    if scenario == "comfort_routine":
        return [
            f"I tested {perfume_name} on a normal day, the kind where I wanted my perfume to change my mood a little.",
            f"The {main.phrase} and {second.phrase} made it feel like {main.texture} against the skin.",
            rng.choice(
                [
                    "It made me feel comforted in a quiet way, not like a dessert, more like clean clothes after a long shower.",
                    "I felt calmer and safer wearing it, as if the room around me had softer corners.",
                    "It made me feel relaxed enough to slow down, make tea, and stop checking my phone for a while.",
                ]
            ),
        ]

    if scenario == "surprising_drydown":
        return [
            f"I expected {perfume_name} to be mostly about {main.phrase}, but the drydown changed my mind.",
            f"After the first sharp moment, {second.phrase} came forward and the whole scent turned {rng.choice(['warmer', 'stranger', 'cleaner', 'darker', 'more tender'])}.",
            rng.choice(
                [
                    "That shift caught me off guard in a good way; I kept smelling my wrist because I did not expect to like the softer part.",
                    "I was surprised by how emotional the change felt, almost like the perfume had relaxed before I did.",
                    "The opening made me hesitate, but the base gave it a second chance.",
                ]
            ),
        ]

    if scenario == "difficult_reaction":
        return [
            f"I wanted to understand {perfume_name}, but {main.phrase} pushed harder than I expected.",
            f"The mix of {second.phrase} and {third.phrase} created a strong image, almost {rng.choice(['too close', 'too warm', 'too dry', 'too sharp', 'too heavy'])} for me.",
            rng.choice(
                [
                    "It is interesting, but I felt tense wearing it and would not reach for it when I need comfort.",
                    "I respect the idea, yet the texture made me feel uneasy after a while.",
                    "This is not automatically bad, but on my skin it became tiring and a little hard to escape.",
                ]
            ),
        ]

    if scenario == "social_identity":
        return [
            f"{perfume_name} made me think about the person I would become if I wore it outside.",
            f"The impression of {main.phrase} gives it presence, while {second.phrase} keeps the mood from feeling flat.",
            rng.choice(
                [
                    "I imagine it with a dark coat, clean shoes, and the confidence to enter a room without speaking first.",
                    "It made me feel more dressed than casual, like a scent for being seen rather than just feeling cozy.",
                    "I would wear it when I want to feel composed, even if I am not fully sure of myself.",
                ]
            ),
        ]

    raise ValueError(f"Unknown scenario: {scenario}")


def adapt_for_style(sentences: List[str], style_family: str, rng: random.Random) -> List[str]:
    out = sentences[:]
    if style_family == "short_fragrantica_note":
        if len(out) > 2:
            out = [out[0], out[-1]]
        out.append(rng.choice(["Very specific mood.", "Not an everyday reach.", "I would sample first.", "It has a clear personality."]))
    elif style_family == "casual_blunt_review":
        out.append(rng.choice(["I get the idea, but I would need the right mood for it.", "Pretty memorable, not necessarily easy.", "I can see the appeal, just not on every day."]))
    elif style_family == "scene_based_review":
        out.append(rng.choice(["The whole thing feels more like a scene than a product description.", "I remember the color of it more than any single note.", "It leaves a little atmosphere behind after the scent gets quiet."]))
    elif style_family == "reflective_mini_review":
        out.append(rng.choice(OPTIONAL_DETAILS))
    elif style_family == "long_personal_review":
        extras = rng.sample(OPTIONAL_DETAILS, k=2)
        out.insert(1, rng.choice(SENTENCE_STARTERS) + " " + extras[0][0].lower() + extras[0][1:])
        out.append(extras[1])
    return out


def build_review(identity: PerfumeIdentity, review_index: int, seed: int, context_hints: Dict[str, List[ScentHint]], source_csv: Path, context_csv: Path) -> Dict[str, Any]:
    rng = random.Random(stable_int(seed, identity.perfume_id, review_index, "emotion-rich-v2"))
    hints, sensory_source = choose_scent_hints(identity=identity, context_hints=context_hints, rng=rng)
    persona = rng.choice(PERSONAS)
    style_family = weighted_choice({key: float(spec["weight"]) for key, spec in STYLE_FAMILIES.items()}, rng)
    scenario_shift = stable_int(seed, identity.perfume_id, "scenario-shift") % len(SCENARIO_SEQUENCE)
    scenario_family = SCENARIO_SEQUENCE[(review_index - 1 + scenario_shift) % len(SCENARIO_SEQUENCE)]
    length_profile = str(STYLE_FAMILIES[style_family]["length"])

    sentences = scenario_sentences(
        scenario=scenario_family,
        perfume_name=identity.perfume_name,
        hints=hints,
        persona=persona,
        rng=rng,
    )
    sentences = adapt_for_style(sentences, style_family, rng)
    text = clean_whitespace(" ".join(sentences))
    review_id = f"emotion-rich-v2-{identity.perfume_id}-{review_index:02d}"

    sensory_terms = ";".join(item.phrase for item in hints)
    accord_tags = ";".join(item.tag for item in hints)
    return {
        "Perfume": identity.perfume_name,
        "Video ID": review_id,
        "Title": f"Emotion-rich unlabeled synthetic review {review_index} of {identity.perfume_name}",
        "Description": "Emotion-rich unlabeled synthetic perfume review for later annotation; no target labels were used.",
        "URL": f"synthetic://emotion-rich-v2/{identity.perfume_id}/{review_index}",
        "Channel Title": "Emotion Rich Synthetic Review Generator",
        "Transcript": text,
        "Transcript Length": len(text.split()),
        "Emotion Keyword Count": emotion_keyword_count(text),
        "Final Transcript": text,
        "source_type": SOURCE_TYPE,
        "dataset_version": DATASET_VERSION,
        "generation_policy": GENERATION_POLICY,
        "synthetic_review_id": review_id,
        "perfume_id": identity.perfume_id,
        "perfume_id_hash": identity.perfume_id_hash,
        "review_index": review_index,
        "synthetic_seed": seed,
        "source_identity_file": stringify_path(source_csv),
        "note_inspiration_file": stringify_path(context_csv),
        "persona": persona,
        "style_family": style_family,
        "scenario_family": scenario_family,
        "length_profile": length_profile,
        "sensory_source": sensory_source,
        "sensory_terms": sensory_terms,
        "accord_tags": accord_tags,
    }


def generate_rows(
    *,
    identities: Sequence[PerfumeIdentity],
    start_review_index: int,
    reviews_per_perfume: int,
    seed: int,
    source_csv: Path,
    context_csv: Path,
    context_hints: Dict[str, List[ScentHint]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for identity in identities:
        stop_review_index = start_review_index + reviews_per_perfume
        for review_index in range(start_review_index, stop_review_index):
            rows.append(build_review(identity, review_index, seed, context_hints, source_csv, context_csv))
    return rows


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fieldnames = RAW_COMPAT_COLUMNS + PROVENANCE_COLUMNS
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def top_counter_values(values: Iterable[str], limit: int = 20) -> Dict[str, int]:
    return dict(Counter(values).most_common(limit))


def validate_generated_rows(rows: Sequence[Dict[str, Any]], expected_rows: int | None) -> None:
    if expected_rows is not None and len(rows) != expected_rows:
        raise ValueError(f"Expected {expected_rows} generated rows, found {len(rows)}.")
    ids = [str(row["synthetic_review_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("synthetic_review_id values must be unique.")
    for row in rows:
        if any(col in row for col in LABEL_COLUMNS):
            raise ValueError("Generated rows must not include emotion or VAD label columns.")
        if not str(row.get("Final Transcript", "")).strip():
            raise ValueError("Generated rows must not contain empty Final Transcript values.")


def write_summary(
    *,
    path: Path,
    source_csv: Path,
    context_csv: Path,
    output_csv: Path,
    identities: Sequence[PerfumeIdentity],
    rows: Sequence[Dict[str, Any]],
    reviews_per_perfume: int,
    seed: int,
    expected_perfumes: int | None,
) -> None:
    word_counts = [int(row["Transcript Length"]) for row in rows]
    sensory_terms = [
        term
        for row in rows
        for term in str(row.get("sensory_terms", "")).split(";")
        if term
    ]
    source_counts = Counter(str(row["sensory_source"]) for row in rows)
    summary = {
        "dataset_version": DATASET_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_type": SOURCE_TYPE,
        "generation_policy": GENERATION_POLICY,
        "source_identity_file": stringify_path(source_csv),
        "note_inspiration_file": stringify_path(context_csv),
        "output_csv": stringify_path(output_csv),
        "n_perfumes": len(identities),
        "expected_perfumes": expected_perfumes,
        "reviews_per_perfume": reviews_per_perfume,
        "review_index_min": min(int(row["review_index"]) for row in rows) if rows else None,
        "review_index_max": max(int(row["review_index"]) for row in rows) if rows else None,
        "n_synthetic_review_rows": len(rows),
        "seed": seed,
        "word_count_min": min(word_counts) if word_counts else 0,
        "word_count_mean": round(mean(word_counts), 3) if word_counts else 0,
        "word_count_max": max(word_counts) if word_counts else 0,
        "emotion_keyword_count_mean": round(mean(int(row["Emotion Keyword Count"]) for row in rows), 3) if rows else 0,
        "persona_counts": Counter(str(row["persona"]) for row in rows),
        "style_family_counts": Counter(str(row["style_family"]) for row in rows),
        "scenario_family_counts": Counter(str(row["scenario_family"]) for row in rows),
        "length_profile_counts": Counter(str(row["length_profile"]) for row in rows),
        "sensory_source_counts": dict(source_counts),
        "top_sensory_terms": top_counter_values(sensory_terms),
        "label_columns_present": sorted(set(rows[0]) & LABEL_COLUMNS) if rows else [],
        "warning": (
            "This is an unlabeled synthetic review dataset. Existing identity-source labels are ignored. "
            "Scenario/style/sensory provenance columns are generation metadata only and must not be used as model features."
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, default=dict) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate emotion-rich unlabeled synthetic perfume reviews.")
    parser.add_argument(
        "--source-csv",
        type=Path,
        default=labels_path("gemini_labels_784_judge_g25pro_v1.csv"),
        help="CSV containing perfume_id, perfume_id_hash, and perfume_name. Label columns are ignored.",
    )
    parser.add_argument(
        "--context-keywords-csv",
        type=Path,
        default=data_path("perfume_level", "perfume_context_keywords.csv"),
        help="Local note-like/sensory keyword matrix derived from existing project text.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=data_path("synthetic", "synthetic_perfume_reviews_784_emotion_rich_v2_pilot100.csv"),
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=data_path("synthetic", "synthetic_perfume_reviews_784_emotion_rich_v2_pilot100_summary.json"),
    )
    parser.add_argument("--reviews-per-perfume", type=int, default=5)
    parser.add_argument("--start-review-index", type=int, default=1)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--limit-perfumes", type=int, default=20, help="Use 0 or a negative value for all perfumes.")
    parser.add_argument("--expected-perfumes", type=int, default=784)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.reviews_per_perfume <= 0:
        raise ValueError("--reviews-per-perfume must be positive.")
    if args.start_review_index <= 0:
        raise ValueError("--start-review-index must be positive.")
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
    if args.limit_perfumes and args.limit_perfumes > 0:
        identities = identities[: args.limit_perfumes]

    context_hints = load_context_hints(args.context_keywords_csv)
    rows = generate_rows(
        identities=identities,
        start_review_index=args.start_review_index,
        reviews_per_perfume=args.reviews_per_perfume,
        seed=args.seed,
        source_csv=args.source_csv,
        context_csv=args.context_keywords_csv,
        context_hints=context_hints,
    )
    expected_rows = len(identities) * args.reviews_per_perfume
    validate_generated_rows(rows, expected_rows=expected_rows)
    write_csv(args.output_csv, rows)
    write_summary(
        path=args.summary_json,
        source_csv=args.source_csv,
        context_csv=args.context_keywords_csv,
        output_csv=args.output_csv,
        identities=identities,
        rows=rows,
        reviews_per_perfume=args.reviews_per_perfume,
        seed=args.seed,
        expected_perfumes=args.expected_perfumes,
    )

    print(f"Wrote emotion-rich unlabeled synthetic review rows: {len(rows)}")
    print(f"Wrote CSV: {args.output_csv}")
    print(f"Wrote summary: {args.summary_json}")


if __name__ == "__main__":
    main()

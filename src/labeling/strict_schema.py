"""Schema and strict protocol utilities for Gemini labeling."""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from typing import Any, Dict

EMOTIONS = [
    "joy",
    "sadness",
    "anger",
    "fear",
    "trust",
    "surprise",
    "anticipation",
    "disgust",
]
VAD_DIMS = ["valence", "arousal", "dominance"]
REQUIRED_KEYS = EMOTIONS + VAD_DIMS

STRICT_PROMPT_RULES = """
STRICT emotion-tone protocol (tone-only):

- Labels reflect ONLY the reviewer’s emotional tone expressed in the snippet.
- Do NOT infer from brand identity, perfume name, marketing concept, reputation, or niche-ness.
- Set an emotion to 1 ONLY if there is explicit linguistic evidence in the snippet text.
- Surprise=1 ONLY if explicit unexpectedness language appears (e.g., "surprised", "didn't expect", "caught me off guard", "shocked", "unexpected").
- Dominance must reflect language of control/assertiveness/confidence (e.g., "commanding", "in control", "dominates"), NOT projection/strength/beast mode unless framed as personal power/control.
- Arousal must reflect emotional activation in language (e.g., "excited", "energized", "calming"), NOT perfume intensity/performance.
- If emotional evidence is weak/neutral, default: valence=5, arousal=5, dominance=5 and all emotions=0.
- Output must be valid JSON only.

Emotions (0/1): joy, sadness, anger, fear, trust, surprise, anticipation, disgust
VAD (1-9): valence, arousal, dominance
""".strip()


def normalize_perfume_id(name: str) -> str:
    """Slugify a perfume name into a stable ID."""
    text = str(name or "").strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text or "perfume"


def perfume_name_hash10(name: str) -> str:
    """Stable short hash for audit/collision checks."""
    return hashlib.sha1(str(name or "").encode("utf-8")).hexdigest()[:10]


def perfume_name_hash4(name: str) -> str:
    return hashlib.sha1(str(name or "").encode("utf-8")).hexdigest()[:4]


def validate_label_payload(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize one label payload."""
    if not isinstance(obj, dict):
        raise ValueError("Label payload must be a JSON object.")

    missing = [k for k in REQUIRED_KEYS if k not in obj]
    if missing:
        raise ValueError(f"Missing keys: {missing}")

    out: Dict[str, Any] = {}

    for emo in EMOTIONS:
        value = obj.get(emo)
        if isinstance(value, str):
            value = value.strip()
            if value in {"0", "1"}:
                value = int(value)
        if isinstance(value, bool):
            value = int(value)
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        if value not in (0, 1):
            raise ValueError(f"Invalid emotion value for '{emo}': {value}")
        out[emo] = int(value)

    for dim in VAD_DIMS:
        value = obj.get(dim)
        if isinstance(value, bool):
            raise ValueError(f"Invalid VAD value for '{dim}': {value}")
        try:
            fval = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid VAD value for '{dim}': {value}") from exc
        if not math.isfinite(fval):
            raise ValueError(f"Non-finite VAD value for '{dim}': {value}")
        if not (1.0 <= fval <= 9.0):
            raise ValueError(f"Out-of-range VAD value for '{dim}': {value}")
        out[dim] = fval

    return out

# Date Created: 2026-04-13 23:02:44+03:00
# Date Last Modified: 2026-05-15

"""Snippet-level aggregation to perfume-level labels."""

from __future__ import annotations

from typing import Dict, List

from src.labeling.strict_schema import EMOTIONS, VAD_DIMS


def aggregate_emotions_majority(snippet_labels: List[Dict[str, float | int]]) -> Dict[str, int]:
    if not snippet_labels:
        raise ValueError("snippet_labels is empty.")
    n = len(snippet_labels)
    half = n / 2.0
    out: Dict[str, int] = {}
    for emo in EMOTIONS:
        votes = sum(int(lbl[emo]) for lbl in snippet_labels)
        out[emo] = 1 if votes > half else 0
    return out


def aggregate_vad_mean(
    snippet_labels: List[Dict[str, float | int]],
    round_ndigits: int = 4,
) -> Dict[str, float]:
    if not snippet_labels:
        raise ValueError("snippet_labels is empty.")
    out: Dict[str, float] = {}
    n = float(len(snippet_labels))
    for dim in VAD_DIMS:
        out[dim] = round(sum(float(lbl[dim]) for lbl in snippet_labels) / n, round_ndigits)
    return out


def aggregate_perfume(snippet_labels: List[Dict[str, float | int]]) -> Dict[str, float | int]:
    agg = aggregate_emotions_majority(snippet_labels)
    agg.update(aggregate_vad_mean(snippet_labels))
    return agg


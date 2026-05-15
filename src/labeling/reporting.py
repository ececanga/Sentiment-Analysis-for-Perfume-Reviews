# Date Created: 2026-04-13 23:02:44+03:00
# Date Last Modified: 2026-05-15

"""Reporting helpers for Gemini labeling outputs."""

from __future__ import annotations

import math
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Sequence

from src.labeling.strict_schema import EMOTIONS, VAD_DIMS


def _safe_stats(values: Sequence[float]) -> Dict[str, float | None]:
    if not values:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {
        "mean": float(mean(values)),
        "std": float(pstdev(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def flatten_records_to_csv_rows(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for rec in records:
        agg = rec.get("aggregated_labels", {})
        row = {
            "perfume_id": rec.get("perfume_id"),
            "perfume_id_hash": rec.get("perfume_id_hash"),
            "perfume_name": rec.get("perfume_name"),
        }
        for key in EMOTIONS + VAD_DIMS:
            row[key] = agg.get(key)
        rows.append(row)
    rows.sort(key=lambda r: str(r.get("perfume_id", "")))
    return rows


def build_labeling_report(
    *,
    target_ids: Sequence[str],
    attempted_ids: Sequence[str],
    skipped_ids: Sequence[str],
    failed_ids: Sequence[str],
    records: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    target_set = set(target_ids)
    success_records = [r for r in records if r.get("perfume_id") in target_set]
    success_ids = {r.get("perfume_id") for r in success_records if r.get("perfume_id")}

    prevalence: Dict[str, float] = {}
    for emo in EMOTIONS:
        vals = [int(r["aggregated_labels"][emo]) for r in success_records]
        prevalence[emo] = float(sum(vals) / len(vals)) if vals else 0.0

    vad_stats: Dict[str, Dict[str, float | None]] = {}
    for dim in VAD_DIMS:
        vals = [float(r["aggregated_labels"][dim]) for r in success_records]
        vals = [v for v in vals if math.isfinite(v)]
        vad_stats[dim] = _safe_stats(vals)

    failed_set = set(failed_ids)
    missing_ids = sorted(list(target_set.difference(success_ids).difference(failed_set)))

    return {
        "counts": {
            "total_targets": len(target_set),
            "attempted": len(set(attempted_ids)),
            "succeeded": len(success_ids),
            "failed": len(failed_set),
            "skipped": len(set(skipped_ids)),
        },
        "emotion_prevalence": prevalence,
        "vad_stats": vad_stats,
        "failed_ids": sorted(list(failed_set)),
        "missing_ids": missing_ids,
    }


def print_distribution_summary(report: Dict[str, Any]) -> None:
    counts = report.get("counts", {})
    print(
        "Counts:",
        f"targets={counts.get('total_targets', 0)}",
        f"attempted={counts.get('attempted', 0)}",
        f"succeeded={counts.get('succeeded', 0)}",
        f"failed={counts.get('failed', 0)}",
        f"skipped={counts.get('skipped', 0)}",
    )
    print("Emotion prevalence:")
    for emo, rate in report.get("emotion_prevalence", {}).items():
        print(f"  {emo}: {rate:.4f}")
    print("VAD stats:")
    for dim, stats in report.get("vad_stats", {}).items():
        print(
            f"  {dim}: mean={stats.get('mean')} std={stats.get('std')} min={stats.get('min')} max={stats.get('max')}"
        )
    print(f"Failed IDs: {len(report.get('failed_ids', []))}")
    print(f"Missing IDs: {len(report.get('missing_ids', []))}")


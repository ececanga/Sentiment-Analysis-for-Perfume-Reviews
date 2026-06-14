"""Inventory local data readiness for the Phase 2A 2,000-perfume expansion."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

from src.labeling.run_gemini_labeling import normalize_name_key
from src.labeling.strict_schema import normalize_perfume_id, perfume_name_hash10
from src.utils.paths import PROJECT_ROOT

csv.field_size_limit(sys.maxsize)

TARGET_2K_COUNT = 2000
DEFAULT_NAME_UNIVERSE_CSV = PROJECT_ROOT / "data" / "perfume_level" / "final_perfume_data.csv"
DEFAULT_PROCESSED_CSV = PROJECT_ROOT / "data" / "intermediate" / "processed_perfumes.csv"


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def stringify_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def read_csv_dicts(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        return list(csv.DictReader(handle))


def require_columns(rows: List[Mapping[str, Any]], columns: Iterable[str], *, path: Path) -> None:
    if not rows:
        raise ValueError(f"CSV has no data rows: {path}")
    fieldnames = set(rows[0].keys())
    missing = [column for column in columns if column not in fieldnames]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").replace("\r", " ").replace("\n", " ").split())


def load_label_source(path: Path) -> Dict[str, Any]:
    rows = read_csv_dicts(path)
    require_columns(rows, ["perfume_id", "perfume_name"], path=path)

    normalized_names: set[str] = set()
    perfume_ids: set[str] = set()
    blank_names = 0
    blank_ids = 0

    for row in rows:
        perfume_name = str(row.get("perfume_name", "")).strip()
        perfume_id = str(row.get("perfume_id", "")).strip()
        if perfume_name:
            normalized_names.add(normalize_name_key(perfume_name))
        else:
            blank_names += 1
        if perfume_id:
            perfume_ids.add(perfume_id)
        else:
            blank_ids += 1

    return {
        "path": stringify_path(path),
        "rows": len(rows),
        "unique_normalized_perfumes": len(normalized_names),
        "unique_perfume_ids": len(perfume_ids),
        "blank_perfume_names": blank_names,
        "blank_perfume_ids": blank_ids,
        "normalized_names": normalized_names,
    }


def load_raw_inventory(
    *,
    raw_csv: Path,
    baseline_names: set[str],
    phase1_pro_names: set[str],
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    rows = read_csv_dicts(raw_csv)
    require_columns(rows, ["Perfume", "Final Transcript"], path=raw_csv)

    grouped: Dict[str, Dict[str, Any]] = {}
    blank_perfume_rows = 0
    usable_rows = 0

    for row in rows:
        perfume_name = str(row.get("Perfume", "")).strip()
        if not perfume_name:
            blank_perfume_rows += 1
            continue

        normalized_name = normalize_name_key(perfume_name)
        entry = grouped.setdefault(
            normalized_name,
            {
                "perfume_name": perfume_name,
                "normalized_name": normalized_name,
                "perfume_id": normalize_perfume_id(perfume_name),
                "perfume_id_hash": perfume_name_hash10(perfume_name),
                "review_count": 0,
                "usable_review_count": 0,
                "total_text_chars": 0,
            },
        )
        entry["review_count"] += 1

        text = clean_text(row.get("Final Transcript", ""))
        if text:
            usable_rows += 1
            entry["usable_review_count"] += 1
            entry["total_text_chars"] += len(text)

    candidates: List[Dict[str, Any]] = []
    for normalized_name, entry in grouped.items():
        usable_count = int(entry["usable_review_count"])
        total_chars = int(entry["total_text_chars"])
        in_frozen = normalized_name in baseline_names
        in_phase1_pro = normalized_name in phase1_pro_names
        candidate_for_2k = usable_count > 0 and not in_frozen
        candidates.append(
            {
                "perfume_name": entry["perfume_name"],
                "normalized_name": normalized_name,
                "perfume_id": entry["perfume_id"],
                "perfume_id_hash": entry["perfume_id_hash"],
                "review_count": int(entry["review_count"]),
                "usable_review_count": usable_count,
                "total_text_chars": total_chars,
                "avg_text_chars": round(total_chars / usable_count, 3) if usable_count else 0.0,
                "in_frozen_784": bool(in_frozen),
                "in_phase1_pro_784": bool(in_phase1_pro),
                "candidate_for_2k": bool(candidate_for_2k),
            }
        )

    candidates.sort(
        key=lambda row: (
            not bool(row["candidate_for_2k"]),
            -int(row["usable_review_count"]),
            str(row["normalized_name"]),
        )
    )

    raw_names = set(grouped)
    usable_names = {
        normalized_name for normalized_name, entry in grouped.items() if int(entry["usable_review_count"]) > 0
    }
    candidate_names = {row["normalized_name"] for row in candidates if row["candidate_for_2k"]}

    summary = {
        "path": stringify_path(raw_csv),
        "rows": len(rows),
        "blank_perfume_rows": blank_perfume_rows,
        "usable_review_rows": usable_rows,
        "unique_perfumes": len(raw_names),
        "perfumes_with_usable_text": len(usable_names),
        "perfumes_with_usable_text_outside_frozen_784": len(usable_names - baseline_names),
        "candidate_perfumes_outside_frozen_784": len(candidate_names),
        "raw_perfumes_missing_from_frozen_784": len(raw_names - baseline_names),
        "frozen_784_missing_from_raw": len(baseline_names - raw_names),
    }
    return summary, candidates


def load_name_universe(path: Path, *, baseline_names: set[str]) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    rows = read_csv_dicts(path)
    require_columns(rows, ["Name"], path=path)

    names = {normalize_name_key(row.get("Name", "")) for row in rows if str(row.get("Name", "")).strip()}
    return {
        "path": stringify_path(path),
        "rows": len(rows),
        "unique_names": len(names),
        "unique_names_outside_frozen_784": len(names - baseline_names),
        "unique_names_inside_frozen_784": len(names & baseline_names),
    }


def load_processed_status(path: Path, *, baseline_names: set[str]) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    rows = read_csv_dicts(path)
    require_columns(rows, ["Perfume", "Status"], path=path)

    status_counts: Counter[str] = Counter()
    status_counts_outside_frozen: Counter[str] = Counter()
    names: set[str] = set()
    names_outside_frozen: set[str] = set()

    for row in rows:
        status = str(row.get("Status", "")).strip() or "UNKNOWN"
        raw_name = str(row.get("Perfume", "")).strip()
        if not raw_name:
            continue
        normalized_name = normalize_name_key(raw_name)
        names.add(normalized_name)
        status_counts[status] += 1
        if normalized_name not in baseline_names:
            names_outside_frozen.add(normalized_name)
            status_counts_outside_frozen[status] += 1

    return {
        "path": stringify_path(path),
        "rows": len(rows),
        "unique_names": len(names),
        "unique_names_outside_frozen_784": len(names_outside_frozen),
        "status_counts": dict(sorted(status_counts.items())),
        "status_counts_outside_frozen_784": dict(sorted(status_counts_outside_frozen.items())),
    }


def write_candidate_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "perfume_name",
        "normalized_name",
        "perfume_id",
        "perfume_id_hash",
        "review_count",
        "usable_review_count",
        "total_text_chars",
        "avg_text_chars",
        "in_frozen_784",
        "in_phase1_pro_784",
        "candidate_for_2k",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fieldnames})


def build_inventory(args: argparse.Namespace) -> Dict[str, Any]:
    baseline_path = resolve_path(args.baseline_labels)
    phase1_pro_path = resolve_path(args.phase1_pro_labels)
    raw_csv = resolve_path(args.raw_csv)
    name_universe_csv = resolve_path(args.name_universe_csv)
    processed_csv = resolve_path(args.processed_csv)

    baseline = load_label_source(baseline_path)
    phase1_pro = load_label_source(phase1_pro_path)
    baseline_names = set(baseline.pop("normalized_names"))
    phase1_pro_names = set(phase1_pro.pop("normalized_names"))

    raw_summary, candidate_rows = load_raw_inventory(
        raw_csv=raw_csv,
        baseline_names=baseline_names,
        phase1_pro_names=phase1_pro_names,
    )

    usable_perfumes = int(raw_summary["perfumes_with_usable_text"])
    candidates_outside_frozen = int(raw_summary["candidate_perfumes_outside_frozen_784"])
    local_data_supports_2000 = usable_perfumes >= TARGET_2K_COUNT

    inventory = {
        "phase": "phase2a_2000_perfume_expansion",
        "status": "blocked_pending_raw_text_expansion" if not local_data_supports_2000 else "ready_for_target_build",
        "target_perfumes": TARGET_2K_COUNT,
        "baseline_labels": baseline,
        "phase1_pro_labels": phase1_pro,
        "label_source_name_sets_match": baseline_names == phase1_pro_names,
        "baseline_only_names_count": len(baseline_names - phase1_pro_names),
        "phase1_pro_only_names_count": len(phase1_pro_names - baseline_names),
        "raw_transcripts": raw_summary,
        "name_only_universe": load_name_universe(name_universe_csv, baseline_names=baseline_names),
        "processed_perfumes": load_processed_status(processed_csv, baseline_names=baseline_names),
        "phase2a_readiness": {
            "local_data_supports_2000": local_data_supports_2000,
            "usable_perfumes_available": usable_perfumes,
            "candidate_perfumes_outside_frozen_784": candidates_outside_frozen,
            "additional_usable_perfumes_required_to_reach_2000": max(0, TARGET_2K_COUNT - usable_perfumes),
            "additional_usable_perfumes_required_beyond_frozen_784": max(0, TARGET_2K_COUNT - len(baseline_names)),
            "can_create_2k_target_builder_now": bool(local_data_supports_2000),
            "blocker": None
            if local_data_supports_2000
            else "Local raw transcript coverage does not contain enough usable perfumes for a 2,000-perfume target set.",
        },
        "outputs": {
            "inventory_json": stringify_path(resolve_path(args.out_json)),
            "candidate_csv": stringify_path(resolve_path(args.out_csv)),
        },
    }

    write_candidate_csv(resolve_path(args.out_csv), candidate_rows)
    out_json = resolve_path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return inventory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inventory local readiness for Phase 2A controlled 2,000-perfume Gemini Pro expansion."
    )
    parser.add_argument("--baseline-labels", type=Path, required=True, help="Frozen Gemini Flash 784 label CSV.")
    parser.add_argument("--phase1-pro-labels", type=Path, required=True, help="Frozen Phase 1 Gemini Pro label CSV.")
    parser.add_argument("--raw-csv", type=Path, required=True, help="Raw review/transcript CSV.")
    parser.add_argument("--out-json", type=Path, required=True, help="Output JSON inventory summary.")
    parser.add_argument("--out-csv", type=Path, required=True, help="Output candidate perfume CSV.")
    parser.add_argument(
        "--name-universe-csv",
        type=Path,
        default=DEFAULT_NAME_UNIVERSE_CSV,
        help="Optional name-only perfume universe CSV with a Name column.",
    )
    parser.add_argument(
        "--processed-csv",
        type=Path,
        default=DEFAULT_PROCESSED_CSV,
        help="Optional processed perfume status CSV with Perfume and Status columns.",
    )
    return parser.parse_args()


def main() -> None:
    inventory = build_inventory(parse_args())
    readiness = inventory["phase2a_readiness"]
    raw = inventory["raw_transcripts"]
    print("Phase 2A inventory complete")
    print(f"Raw unique perfumes: {raw['unique_perfumes']}")
    print(f"Raw perfumes with usable text: {raw['perfumes_with_usable_text']}")
    print(f"Candidates outside frozen 784: {raw['candidate_perfumes_outside_frozen_784']}")
    print(f"Local data supports 2,000: {readiness['local_data_supports_2000']}")
    print(
        "Additional usable perfumes required: "
        f"{readiness['additional_usable_perfumes_required_to_reach_2000']}"
    )


if __name__ == "__main__":
    main()

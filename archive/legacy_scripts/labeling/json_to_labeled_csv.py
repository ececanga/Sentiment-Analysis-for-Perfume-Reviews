"""json_to_labeled_csv.py

Join LLM label JSON to an annotation packet CSV (or create perfume-only labels).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd

from src.utils.paths import data_path, labels_path

EMOTIONS = ["joy", "sadness", "anger", "fear", "trust", "surprise", "anticipation", "disgust"]
VAD = ["valence", "arousal", "dominance"]


def normalize_name(name: str) -> str:
    text = str(name).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def find_perfume_column(columns: Iterable[str]) -> str:
    matches = [c for c in columns if c.strip().lower() == "perfume"]
    if not matches:
        raise ValueError("No perfume column found. Expected 'Perfume' (case-insensitive).")
    return matches[0]


def validate_labels(labels: Dict[str, Dict[str, object]]) -> None:
    required = set(EMOTIONS + VAD)
    for perfume, payload in labels.items():
        if not isinstance(payload, dict):
            raise ValueError(f"Label payload must be object for perfume: {perfume}")
        missing = required.difference(payload.keys())
        if missing:
            raise ValueError(f"Missing keys for perfume '{perfume}': {sorted(missing)}")
        for emo in EMOTIONS:
            val = payload.get(emo)
            if val not in (0, 1):
                raise ValueError(f"Invalid emotion value for '{perfume}'/{emo}: {val}")
        for dim in VAD:
            val = payload.get(dim)
            if not isinstance(val, int) or not (1 <= val <= 9):
                raise ValueError(f"Invalid VAD value for '{perfume}'/{dim}: {val}")


def build_label_index(labels: Dict[str, Dict[str, object]]) -> Tuple[Dict[str, Dict[str, object]], Dict[str, str]]:
    norm_to_label: Dict[str, Dict[str, object]] = {}
    norm_to_original: Dict[str, str] = {}
    for perfume, payload in labels.items():
        key = normalize_name(perfume)
        if key in norm_to_label:
            continue
        norm_to_label[key] = payload
        norm_to_original[key] = perfume
    return norm_to_label, norm_to_original


def collect_snippet_columns(columns: Iterable[str]) -> List[str]:
    snippet_cols = [c for c in columns if c.startswith("Snippet_")]
    return sorted(snippet_cols, key=lambda c: int(c.split("_", 1)[1]) if c.split("_", 1)[1].isdigit() else c)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--labels",
        type=Path,
        default=labels_path("llm", "strict", "labels_strict_v1.json"),
        help="Path to labels JSON",
        required=False,
    )
    ap.add_argument(
        "--packet",
        type=str,
        default=str(
            data_path(
                "annotation_packets",
                "Annotation_Packet_TOSKOVAT_ORTO_PARISI_ELO_NASOMATTO_SERGE_LUTENS.csv",
            )
        ),
        help="Annotation packet CSV path (optional)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=data_path("annotation_packets", "Labeled_Annotation_Packet_strict_v1.csv"),
        help="Output labeled CSV path",
    )
    args = ap.parse_args()

    labels_path_in = args.labels
    if not labels_path_in.is_absolute():
        labels_path_in = labels_path_in.resolve()
    if not labels_path_in.exists():
        raise FileNotFoundError(f"Labels JSON not found: {labels_path_in}")

    labels = json.loads(labels_path_in.read_text(encoding="utf-8"))
    validate_labels(labels)
    norm_to_label, norm_to_original = build_label_index(labels)

    packet_path = None
    if args.packet is not None:
        packet_str = args.packet.strip()
        if packet_str:
            packet_path = Path(packet_str)
            if not packet_path.is_absolute():
                packet_path = packet_path.resolve()

    if packet_path is not None and not packet_path.exists():
        raise FileNotFoundError(f"Packet CSV not found: {packet_path}")

    labels_norm_keys = set(norm_to_label.keys())

    if packet_path is None:
        labeled_rows = []
        for norm_key, payload in norm_to_label.items():
            row = {"Perfume": norm_to_original[norm_key]}
            row.update({k: payload[k] for k in EMOTIONS + VAD})
            labeled_rows.append(row)
        labeled_df = pd.DataFrame(labeled_rows)
        labeled_out = args.out
    else:
        packet_df = pd.read_csv(packet_path)
        perfume_col = find_perfume_column(packet_df.columns)
        packet_df["_perfume_key"] = packet_df[perfume_col].apply(normalize_name)

        labels_rows = []
        for norm_key, payload in norm_to_label.items():
            row = {"Perfume": norm_to_original[norm_key], "_perfume_key": norm_key}
            row.update({k: payload[k] for k in EMOTIONS + VAD})
            labels_rows.append(row)
        labels_df = pd.DataFrame(labels_rows)

        packet_norm_keys = set(packet_df["_perfume_key"].dropna().unique())
        matched = packet_norm_keys.intersection(labels_norm_keys)
        missing_in_labels = sorted(packet_norm_keys.difference(labels_norm_keys))
        extra_in_labels = sorted(labels_norm_keys.difference(packet_norm_keys))

        print(f"Total packet perfumes: {len(packet_norm_keys)}")
        print(f"Total labels perfumes: {len(labels_norm_keys)}")
        print(f"Matched count: {len(matched)}")
        print(f"Missing in labels (first 20): {missing_in_labels[:20]}")
        print(f"Extra in labels (first 20): {extra_in_labels[:20]}")

        labels_map = labels_df.set_index("_perfume_key")[EMOTIONS + VAD].to_dict(orient="index")
        for col in EMOTIONS + VAD:
            packet_df[col] = packet_df["_perfume_key"].map(
                lambda k: labels_map.get(k, {}).get(col)
            )

        for col in EMOTIONS + VAD:
            dup_col = f"{col}.1"
            if dup_col in packet_df.columns:
                packet_df = packet_df.drop(columns=[dup_col])

        snippet_cols = collect_snippet_columns(packet_df.columns)

        out_cols = [perfume_col] + snippet_cols + EMOTIONS + VAD
        labeled_df = packet_df[out_cols]
        labeled_out = args.out

    labeled_out.parent.mkdir(parents=True, exist_ok=True)
    labeled_df.to_csv(labeled_out, index=False)
    for col in EMOTIONS + VAD:
        if col in labeled_df.columns:
            print(f"{col}: non-null count = {labeled_df[col].notna().sum()}")

    perfume_only_rows = []
    for norm_key, payload in norm_to_label.items():
        row = {"Perfume": norm_to_original[norm_key]}
        row.update({k: payload[k] for k in EMOTIONS + VAD})
        perfume_only_rows.append(row)

    perfume_only_df = pd.DataFrame(perfume_only_rows)
    perfume_only_path = labeled_out.parent / "perfume_only_labels.csv"
    perfume_only_df.to_csv(perfume_only_path, index=False)


if __name__ == "__main__":
    main()

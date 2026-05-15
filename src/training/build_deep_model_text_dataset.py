# Date Created: 2026-04-13 23:02:44+03:00
# Date Last Modified: 2026-05-15

"""Build the canonical deep-model text dataset for the frozen perfume benchmark."""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from src.labeling.strict_schema import normalize_perfume_id
from src.utils.paths import PROJECT_ROOT, data_path, labels_path, outputs_path

csv.field_size_limit(sys.maxsize)

OFFICIAL_LABELS = ["joy", "sadness", "fear", "trust", "surprise", "anticipation", "disgust"]
EXCLUDED_LABELS = ["anger"]
VAD_DIMS = ["valence", "arousal", "dominance"]

DEFAULT_DROP_TOKENS = {
    "absolu",
    "absolue",
    "body",
    "cologne",
    "cream",
    "de",
    "eau",
    "edp",
    "edt",
    "extrait",
    "femme",
    "for",
    "fraiche",
    "fraichee",
    "homme",
    "mist",
    "oil",
    "parfum",
    "perfume",
    "pour",
    "scented",
    "set",
    "spray",
    "toilette",
    "travel",
}

REVIEW_SEPARATOR = "<REVIEW_SEP>"


@dataclass(frozen=True)
class BuildConfig:
    raw_csv: Path
    labels_csv: Path
    folds_csv: Path
    output_jsonl: Path
    output_chunks_csv: Path
    output_summary_json: Path
    title_max_words: int = 24
    description_max_words: int = 48
    transcript_window_words: int = 160
    transcript_pre_context_words: int = 32


def stringify_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def clean_whitespace(text: str) -> str:
    return " ".join(html.unescape(str(text or "")).replace("\r", " ").replace("\n", " ").split())


def normalize_token(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text or "").lower())


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(text or "").lower())).strip()


def tokenize(text: str) -> List[str]:
    cleaned = clean_whitespace(text)
    return cleaned.split() if cleaned else []


def truncate_tokens(text: str, max_words: int) -> str:
    if max_words <= 0:
        return ""
    tokens = tokenize(text)
    return " ".join(tokens[:max_words])


def read_csv_dicts(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        return list(csv.DictReader(handle))


def perfume_alias_tokens(name: str, drop_tokens: Iterable[str] = DEFAULT_DROP_TOKENS) -> List[str]:
    base_tokens = normalize_text(name).split()
    filtered = [tok for tok in base_tokens if tok not in set(drop_tokens)]
    return filtered or base_tokens


def search_tokens_with_mapping(text: str) -> Tuple[List[str], List[str], List[int]]:
    raw_tokens = tokenize(text)
    search_tokens: List[str] = []
    raw_indices: List[int] = []
    for idx, raw in enumerate(raw_tokens):
        norm = normalize_token(raw)
        if not norm:
            continue
        search_tokens.append(norm)
        raw_indices.append(idx)
    return raw_tokens, search_tokens, raw_indices


def find_subsequence(tokens: Sequence[str], pattern: Sequence[str]) -> int:
    if not tokens or not pattern or len(pattern) > len(tokens):
        return -1
    stop = len(tokens) - len(pattern) + 1
    for start in range(stop):
        if list(tokens[start : start + len(pattern)]) == list(pattern):
            return start
    return -1


def find_first_token(tokens: Sequence[str], token: str) -> int:
    for idx, item in enumerate(tokens):
        if item == token:
            return idx
    return -1


def compute_field_match(text: str, alias_tokens: Sequence[str], anchor_token: str) -> bool:
    norm = normalize_text(text)
    alias_phrase = " ".join(alias_tokens).strip()
    if alias_phrase and alias_phrase in norm:
        return True
    return bool(anchor_token and anchor_token in norm.split())


def search_slice_to_raw_bounds(
    raw_indices: Sequence[int],
    start_search: int,
    end_search: int,
    raw_token_count: int,
) -> Tuple[int, int]:
    if not raw_indices:
        return 0, raw_token_count
    raw_start = raw_indices[start_search]
    raw_end = raw_indices[end_search - 1] + 1
    return raw_start, min(raw_end, raw_token_count)


def extract_transcript_excerpt(
    perfume_name: str,
    transcript: str,
    *,
    window_words: int,
    pre_context_words: int,
) -> Dict[str, Any]:
    raw_tokens, search_tokens, raw_indices = search_tokens_with_mapping(transcript)
    if not raw_tokens:
        return {
            "excerpt_text": "",
            "match_strategy": "empty_transcript",
            "transcript_phrase_match": False,
            "transcript_anchor_match": False,
            "alias_phrase": "",
            "anchor_token": "",
        }

    alias_tokens = perfume_alias_tokens(perfume_name)
    alias_phrase = " ".join(alias_tokens)
    anchor_token = max(alias_tokens, key=len) if alias_tokens else ""

    phrase_idx = find_subsequence(search_tokens, alias_tokens) if alias_tokens else -1
    anchor_idx = find_first_token(search_tokens, anchor_token) if anchor_token else -1

    if phrase_idx >= 0:
        start_search = max(0, phrase_idx - pre_context_words)
        end_search = min(len(search_tokens), start_search + window_words)
        start_search = max(0, end_search - window_words)
        raw_start, raw_end = search_slice_to_raw_bounds(raw_indices, start_search, end_search, len(raw_tokens))
        excerpt_tokens = raw_tokens[raw_start:raw_end]
        strategy = "alias_phrase_window"
    elif anchor_idx >= 0:
        start_search = max(0, anchor_idx - pre_context_words)
        end_search = min(len(search_tokens), start_search + window_words)
        start_search = max(0, end_search - window_words)
        raw_start, raw_end = search_slice_to_raw_bounds(raw_indices, start_search, end_search, len(raw_tokens))
        excerpt_tokens = raw_tokens[raw_start:raw_end]
        strategy = "anchor_token_window"
    else:
        excerpt_tokens = raw_tokens[:window_words]
        strategy = "prefix_fallback"

    return {
        "excerpt_text": " ".join(excerpt_tokens),
        "match_strategy": strategy,
        "transcript_phrase_match": phrase_idx >= 0,
        "transcript_anchor_match": anchor_idx >= 0,
        "alias_phrase": alias_phrase,
        "anchor_token": anchor_token,
    }


def build_chunk_text(
    *,
    title: str,
    description: str,
    transcript_excerpt: str,
    title_max_words: int,
    description_max_words: int,
) -> str:
    parts: List[str] = []
    title_short = truncate_tokens(title, title_max_words)
    description_short = truncate_tokens(description, description_max_words)
    review_short = clean_whitespace(transcript_excerpt)

    if title_short:
        parts.append(f"TITLE: {title_short}")
    if description_short:
        parts.append(f"DESCRIPTION: {description_short}")
    if review_short:
        parts.append(f"REVIEW: {review_short}")
    return " || ".join(parts)


def parse_binary_label(value: str) -> int:
    text = str(value or "").strip()
    if text not in {"0", "1"}:
        raise ValueError(f"Expected binary label 0/1, got {value!r}")
    return int(text)


def parse_float(value: str) -> float:
    return float(str(value or "").strip())


def load_labels(labels_csv: Path) -> Dict[str, Dict[str, Any]]:
    rows = read_csv_dicts(labels_csv)
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        perfume_id = str(row.get("perfume_id", "")).strip()
        if not perfume_id:
            raise ValueError(f"Missing perfume_id in labels file: {labels_csv}")
        if perfume_id in out:
            raise ValueError(f"Duplicate perfume_id in labels file: {perfume_id}")
        payload = {
            "perfume_id": perfume_id,
            "perfume_id_hash": str(row.get("perfume_id_hash", "")).strip(),
            "perfume_name": str(row.get("perfume_name", "")).strip(),
            "labels": {label: parse_binary_label(row.get(label, "")) for label in OFFICIAL_LABELS},
            "excluded_labels": {label: parse_binary_label(row.get(label, "")) for label in EXCLUDED_LABELS},
            "vad": {dim: parse_float(row.get(dim, "")) for dim in VAD_DIMS},
        }
        out[perfume_id] = payload
    return out


def load_folds(folds_csv: Path) -> Dict[str, int]:
    rows = read_csv_dicts(folds_csv)
    out: Dict[str, int] = {}
    for row in rows:
        perfume_id = str(row.get("perfume_id", "")).strip()
        if not perfume_id:
            raise ValueError(f"Missing perfume_id in fold manifest: {folds_csv}")
        if perfume_id in out:
            raise ValueError(f"Duplicate perfume_id in fold manifest: {perfume_id}")
        out[perfume_id] = int(str(row.get("fold_id", "")).strip())
    return out


def load_raw_groups(raw_csv: Path) -> Dict[str, List[Dict[str, str]]]:
    rows = read_csv_dicts(raw_csv)
    groups: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        perfume_name = str(row.get("Perfume", "")).strip()
        perfume_id = normalize_perfume_id(perfume_name)
        if not perfume_id:
            continue
        row_copy = dict(row)
        row_copy["perfume_id"] = perfume_id
        row_copy["Perfume"] = perfume_name
        groups[perfume_id].append(row_copy)
    return groups


def validate_universe(
    *,
    labels_by_id: Dict[str, Dict[str, Any]],
    fold_by_id: Dict[str, int],
    raw_groups: Dict[str, List[Dict[str, str]]],
) -> List[str]:
    label_ids = set(labels_by_id)
    fold_ids = set(fold_by_id)
    raw_ids = set(raw_groups)

    if label_ids != fold_ids:
        raise ValueError(
            "Mismatch between labels and fold manifest perfume_id sets: "
            f"labels_only={sorted(label_ids - fold_ids)[:10]} "
            f"folds_only={sorted(fold_ids - label_ids)[:10]}"
        )
    if label_ids != raw_ids:
        raise ValueError(
            "Mismatch between labels and raw text perfume_id sets: "
            f"labels_only={sorted(label_ids - raw_ids)[:10]} "
            f"raw_only={sorted(raw_ids - label_ids)[:10]}"
        )
    return sorted(label_ids)


def sort_group_rows(rows: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    return sorted(
        rows,
        key=lambda row: (
            clean_whitespace(row.get("Video ID", "")),
            clean_whitespace(row.get("Title", "")).lower(),
            clean_whitespace(row.get("URL", "")),
            clean_whitespace(row.get("Final Transcript", "")),
        ),
    )


def deduplicate_group_rows(rows: Sequence[Dict[str, str]]) -> Tuple[List[Dict[str, str]], int]:
    seen_transcripts: set[str] = set()
    deduped: List[Dict[str, str]] = []
    dropped = 0
    for row in rows:
        transcript = clean_whitespace(row.get("Final Transcript", ""))
        if not transcript:
            continue
        if transcript in seen_transcripts:
            dropped += 1
            continue
        seen_transcripts.add(transcript)
        deduped.append(row)
    return deduped, dropped


def build_dataset(cfg: BuildConfig) -> Dict[str, Any]:
    labels_by_id = load_labels(cfg.labels_csv)
    fold_by_id = load_folds(cfg.folds_csv)
    raw_groups = load_raw_groups(cfg.raw_csv)
    perfume_ids = validate_universe(labels_by_id=labels_by_id, fold_by_id=fold_by_id, raw_groups=raw_groups)

    cfg.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    cfg.output_chunks_csv.parent.mkdir(parents=True, exist_ok=True)
    cfg.output_summary_json.parent.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {
        "dataset_version": "v1",
        "raw_csv": stringify_path(cfg.raw_csv),
        "labels_csv": stringify_path(cfg.labels_csv),
        "fold_manifest_csv": stringify_path(cfg.folds_csv),
        "output_jsonl": stringify_path(cfg.output_jsonl),
        "output_chunks_csv": stringify_path(cfg.output_chunks_csv),
        "title_max_words": cfg.title_max_words,
        "description_max_words": cfg.description_max_words,
        "transcript_window_words": cfg.transcript_window_words,
        "transcript_pre_context_words": cfg.transcript_pre_context_words,
    }

    match_strategy_counts: Counter[str] = Counter()
    chunk_counts: List[int] = []
    source_review_counts: List[int] = []
    dedup_drops = 0
    total_chunks = 0
    chunks_with_any_name_match = 0
    perfumes_with_any_name_match = 0
    perfumes_with_only_prefix_fallback = 0

    chunk_fieldnames = [
        "perfume_id",
        "perfume_name",
        "perfume_id_hash",
        "fold_id",
        "source_review_count",
        "chunk_count",
        "chunk_index",
        "video_id",
        "url",
        "match_strategy",
        "title_match",
        "description_match",
        "transcript_phrase_match",
        "transcript_anchor_match",
        "any_name_match",
        "chunk_word_count",
        "chunk_text",
        *OFFICIAL_LABELS,
        *EXCLUDED_LABELS,
        *VAD_DIMS,
    ]

    with cfg.output_jsonl.open("w", encoding="utf-8") as jsonl_handle, cfg.output_chunks_csv.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as chunks_handle:
        chunk_writer = csv.DictWriter(chunks_handle, fieldnames=chunk_fieldnames)
        chunk_writer.writeheader()

        for perfume_id in perfume_ids:
            label_payload = labels_by_id[perfume_id]
            fold_id = fold_by_id[perfume_id]
            raw_rows = sort_group_rows(raw_groups[perfume_id])
            deduped_rows, dropped = deduplicate_group_rows(raw_rows)
            dedup_drops += dropped
            if not deduped_rows:
                raise ValueError(f"No usable transcript rows remain for perfume_id={perfume_id}")

            review_chunks: List[Dict[str, Any]] = []
            perfume_has_any_name_match = False
            for chunk_index, row in enumerate(deduped_rows):
                perfume_name = row.get("Perfume", "") or label_payload["perfume_name"]
                alias_tokens = perfume_alias_tokens(perfume_name)
                title = clean_whitespace(row.get("Title", ""))
                description = clean_whitespace(row.get("Description", ""))
                transcript = clean_whitespace(row.get("Final Transcript", ""))
                excerpt = extract_transcript_excerpt(
                    perfume_name,
                    transcript,
                    window_words=cfg.transcript_window_words,
                    pre_context_words=cfg.transcript_pre_context_words,
                )
                title_match = compute_field_match(title, alias_tokens, excerpt["anchor_token"])
                description_match = compute_field_match(
                    description,
                    alias_tokens,
                    excerpt["anchor_token"],
                )
                any_name_match = bool(
                    title_match
                    or description_match
                    or excerpt["transcript_phrase_match"]
                    or excerpt["transcript_anchor_match"]
                )
                chunk_text = build_chunk_text(
                    title=title,
                    description=description,
                    transcript_excerpt=excerpt["excerpt_text"],
                    title_max_words=cfg.title_max_words,
                    description_max_words=cfg.description_max_words,
                )
                if not chunk_text:
                    raise ValueError(f"Empty chunk text for perfume_id={perfume_id} chunk_index={chunk_index}")

                chunk_record = {
                    "chunk_index": chunk_index,
                    "video_id": clean_whitespace(row.get("Video ID", "")),
                    "url": clean_whitespace(row.get("URL", "")),
                    "match_strategy": excerpt["match_strategy"],
                    "title_match": title_match,
                    "description_match": description_match,
                    "transcript_phrase_match": excerpt["transcript_phrase_match"],
                    "transcript_anchor_match": excerpt["transcript_anchor_match"],
                    "any_name_match": any_name_match,
                    "chunk_word_count": len(tokenize(chunk_text)),
                    "chunk_text": chunk_text,
                }
                review_chunks.append(chunk_record)
                match_strategy_counts[excerpt["match_strategy"]] += 1
                if any_name_match:
                    chunks_with_any_name_match += 1
                    perfume_has_any_name_match = True

            chunk_count = len(review_chunks)
            total_chunks += chunk_count
            chunk_counts.append(chunk_count)
            source_review_counts.append(len(raw_rows))
            if perfume_has_any_name_match:
                perfumes_with_any_name_match += 1
            if all(chunk["match_strategy"] == "prefix_fallback" for chunk in review_chunks):
                perfumes_with_only_prefix_fallback += 1

            document_text = f" {REVIEW_SEPARATOR} ".join(chunk["chunk_text"] for chunk in review_chunks)
            record = {
                "perfume_id": perfume_id,
                "perfume_name": label_payload["perfume_name"],
                "perfume_id_hash": label_payload["perfume_id_hash"],
                "fold_id": fold_id,
                "source_review_count": len(raw_rows),
                "deduplicated_review_count": chunk_count,
                "labels": label_payload["labels"],
                "excluded_labels": label_payload["excluded_labels"],
                "vad": label_payload["vad"],
                "document_text": document_text,
                "review_chunks": review_chunks,
            }
            jsonl_handle.write(json.dumps(record, ensure_ascii=False) + "\n")

            for chunk in review_chunks:
                row_out: Dict[str, Any] = {
                    "perfume_id": perfume_id,
                    "perfume_name": label_payload["perfume_name"],
                    "perfume_id_hash": label_payload["perfume_id_hash"],
                    "fold_id": fold_id,
                    "source_review_count": len(raw_rows),
                    "chunk_count": chunk_count,
                    "chunk_index": chunk["chunk_index"],
                    "video_id": chunk["video_id"],
                    "url": chunk["url"],
                    "match_strategy": chunk["match_strategy"],
                    "title_match": chunk["title_match"],
                    "description_match": chunk["description_match"],
                    "transcript_phrase_match": chunk["transcript_phrase_match"],
                    "transcript_anchor_match": chunk["transcript_anchor_match"],
                    "any_name_match": chunk["any_name_match"],
                    "chunk_word_count": chunk["chunk_word_count"],
                    "chunk_text": chunk["chunk_text"],
                }
                for label in OFFICIAL_LABELS:
                    row_out[label] = label_payload["labels"][label]
                for label in EXCLUDED_LABELS:
                    row_out[label] = label_payload["excluded_labels"][label]
                for dim in VAD_DIMS:
                    row_out[dim] = label_payload["vad"][dim]
                chunk_writer.writerow(row_out)

    summary.update(
        {
            "n_perfumes": len(perfume_ids),
            "n_total_chunks": total_chunks,
            "deduplicated_transcript_rows_dropped": dedup_drops,
            "chunks_with_any_name_match": chunks_with_any_name_match,
            "perfumes_with_any_name_match": perfumes_with_any_name_match,
            "perfumes_with_only_prefix_fallback": perfumes_with_only_prefix_fallback,
            "source_review_count_min": min(source_review_counts),
            "source_review_count_median": median(source_review_counts),
            "source_review_count_mean": round(mean(source_review_counts), 3),
            "source_review_count_max": max(source_review_counts),
            "chunk_count_min": min(chunk_counts),
            "chunk_count_median": median(chunk_counts),
            "chunk_count_mean": round(mean(chunk_counts), 3),
            "chunk_count_max": max(chunk_counts),
            "match_strategy_counts": dict(match_strategy_counts),
        }
    )

    cfg.output_summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    return {
        "output_jsonl": cfg.output_jsonl,
        "output_chunks_csv": cfg.output_chunks_csv,
        "output_summary_json": cfg.output_summary_json,
        "summary": summary,
    }


def parse_args() -> BuildConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-csv",
        type=Path,
        default=data_path("raw", "YouTube_Perfume_Reviews_FullyFormatted.csv"),
        help="Immutable review-level source CSV.",
    )
    parser.add_argument(
        "--labels-csv",
        type=Path,
        default=labels_path("gemini_labels_784_v1.csv"),
        help="Perfume-level Gemini label CSV.",
    )
    parser.add_argument(
        "--folds-csv",
        type=Path,
        default=outputs_path("model_benchmark", "tables", "fold_assignments.csv"),
        help="Frozen perfume-level fold manifest CSV.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=outputs_path("model_benchmark", "datasets", "perfume_text_dataset_v1.jsonl"),
        help="Canonical perfume-level JSONL for future deep-model runs.",
    )
    parser.add_argument(
        "--output-chunks-csv",
        type=Path,
        default=outputs_path("model_benchmark", "datasets", "perfume_text_dataset_v1_chunks.csv"),
        help="Flat chunk-level CSV audit view.",
    )
    parser.add_argument(
        "--output-summary-json",
        type=Path,
        default=outputs_path("model_benchmark", "datasets", "perfume_text_dataset_v1_summary.json"),
        help="Summary JSON with coverage and chunk statistics.",
    )
    parser.add_argument("--title-max-words", type=int, default=24)
    parser.add_argument("--description-max-words", type=int, default=48)
    parser.add_argument("--transcript-window-words", type=int, default=160)
    parser.add_argument("--transcript-pre-context-words", type=int, default=32)
    args = parser.parse_args()

    return BuildConfig(
        raw_csv=args.raw_csv,
        labels_csv=args.labels_csv,
        folds_csv=args.folds_csv,
        output_jsonl=args.output_jsonl,
        output_chunks_csv=args.output_chunks_csv,
        output_summary_json=args.output_summary_json,
        title_max_words=args.title_max_words,
        description_max_words=args.description_max_words,
        transcript_window_words=args.transcript_window_words,
        transcript_pre_context_words=args.transcript_pre_context_words,
    )


def main() -> None:
    cfg = parse_args()
    result = build_dataset(cfg)
    summary = result["summary"]
    print(
        "Built deep-model text dataset: "
        f"{stringify_path(result['output_jsonl'])} "
        f"({summary['n_perfumes']} perfumes, {summary['n_total_chunks']} chunks)"
    )


if __name__ == "__main__":
    main()

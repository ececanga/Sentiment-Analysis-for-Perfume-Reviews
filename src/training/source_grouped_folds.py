# Date Created: 2026-07-13
# Date Last Modified: 2026-07-13

"""Build deterministic source-isolated outer and inner split manifests.

Perfumes are joined into one atomic component whenever they share a valid
YouTube video ID or the SHA-256 of a normalized full transcript.  Components
are never cut across an outer fold or an inner train/validation partition.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import GroupKFold

from src.labeling.strict_schema import normalize_perfume_id


SOURCE_IDENTITY_VERSION = "source_identity_v1"
SOURCE_GROUP_COLUMN = "source_group_id"
VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
OFFICIAL_LABELS = (
    "joy",
    "sadness",
    "fear",
    "trust",
    "surprise",
    "anticipation",
    "disgust",
)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of exact file bytes."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_configured_frozen_inputs(
    *,
    config_path: Path | None,
    input_paths: Mapping[str, Path],
) -> tuple[Dict[str, str], Dict[str, Any]]:
    """Hash canonical inputs and enforce a config's frozen digests when supplied."""

    actual = {key: sha256_file(path) for key, path in input_paths.items()}
    validation: Dict[str, Any] = {
        "enforced": config_path is not None,
        "config_path": None,
        "config_sha256": None,
    }
    if config_path is None:
        return actual, validation

    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, Mapping):
        raise ValueError(f"Source-grouped config must contain a YAML mapping: {config_path}")
    provenance = config.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("Source-grouped config lacks a provenance mapping")
    expected_values = provenance.get("frozen_input_sha256")
    if not isinstance(expected_values, Mapping):
        raise ValueError("Source-grouped config lacks provenance.frozen_input_sha256")

    expected: Dict[str, str] = {}
    for key in input_paths:
        if key not in expected_values:
            raise ValueError(
                f"Source-grouped config lacks provenance.frozen_input_sha256.{key}"
            )
        digest = str(expected_values[key]).strip().lower()
        if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError(
                f"Invalid SHA-256 in provenance.frozen_input_sha256.{key}: {digest!r}"
            )
        expected[key] = digest
        if actual[key] != digest:
            raise ValueError(
                f"Frozen input SHA-256 mismatch for {key}: "
                f"expected {digest}, got {actual[key]} ({input_paths[key]})"
            )

    validation.update(
        {
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
            "expected_sha256": expected,
        }
    )
    return actual, validation


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(missing) if isinstance(missing, (bool, np.bool_)) else False


def normalize_video_id(value: Any, *, allow_empty: bool = False) -> str:
    """Strip and validate a case-sensitive 11-character YouTube video ID."""

    text = "" if _is_missing(value) else str(value).strip()
    if text and VIDEO_ID_RE.fullmatch(text):
        return text
    if allow_empty:
        return ""
    if not text:
        raise ValueError("Video ID is empty")
    raise ValueError(f"Invalid YouTube video ID: {text!r}")


def normalize_transcript(value: Any) -> str:
    """Apply the prespecified Unicode/punctuation transcript normalization."""

    text = "" if _is_missing(value) else str(value)
    text = unicodedata.normalize("NFKC", html.unescape(text)).casefold()
    output: list[str] = []
    separator_pending = False
    for char in text:
        if char.isalnum():
            if separator_pending and output:
                output.append(" ")
            output.append(char)
            separator_pending = False
        else:
            separator_pending = True
    return "".join(output).strip()


def transcript_sha256(value: Any) -> str:
    """Hash a normalized transcript, returning empty for empty normalized text."""

    normalized = normalize_transcript(value)
    if not normalized:
        return ""
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class _UnionFind:
    def __init__(self, items: Iterable[str]) -> None:
        self.parent = {item: item for item in items}
        self.rank = {item: 0 for item in items}

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def _validate_canonical_ids(values: Sequence[Any], *, context: str) -> list[str]:
    ids = [str(value).strip() for value in values]
    if any(not value for value in ids):
        raise ValueError(f"{context} contains an empty perfume_id")
    if len(ids) != len(set(ids)):
        raise ValueError(f"{context} contains duplicate perfume_id values")
    return ids


def build_source_memberships(
    raw_df: pd.DataFrame,
    *,
    canonical_perfume_ids: Sequence[str],
) -> tuple[pd.DataFrame, Dict[str, int]]:
    """Normalize raw source keys and return one deduplicated membership row."""

    required = {"Perfume", "Video ID", "Final Transcript"}
    missing = required - set(raw_df.columns)
    if missing:
        raise ValueError(f"Raw source table lacks columns: {sorted(missing)}")
    canonical_ids = set(_validate_canonical_ids(canonical_perfume_ids, context="Canonical universe"))

    rows: list[Dict[str, Any]] = []
    invalid_video_count = 0
    empty_transcript_count = 0
    unresolved_ids: set[str] = set()
    for raw_index, row in raw_df.reset_index(drop=True).iterrows():
        perfume_name = str(row["Perfume"]).strip()
        perfume_id = normalize_perfume_id(perfume_name)
        if perfume_id not in canonical_ids:
            unresolved_ids.add(perfume_id)
            continue
        raw_video = row["Video ID"]
        video_id = normalize_video_id(raw_video, allow_empty=True)
        if not video_id:
            invalid_video_count += 1
        transcript_hash = transcript_sha256(row["Final Transcript"])
        if not transcript_hash:
            empty_transcript_count += 1
        rows.append(
            {
                "perfume_id": perfume_id,
                "Perfume": perfume_name,
                "video_id": video_id,
                "transcript_sha256": transcript_hash,
                "raw_row_index": int(raw_index),
            }
        )

    if unresolved_ids:
        preview = sorted(unresolved_ids)[:10]
        raise ValueError(f"Raw rows resolve outside the canonical perfume universe: {preview}")
    observed = {row["perfume_id"] for row in rows}
    missing_ids = sorted(canonical_ids - observed)
    if missing_ids:
        raise ValueError(f"Canonical perfumes have no raw source row: {missing_ids[:10]}")

    frame = pd.DataFrame(rows)
    dedupe_columns = ["perfume_id", "video_id", "transcript_sha256"]
    duplicate_count = int(frame.duplicated(dedupe_columns).sum())
    frame = (
        frame.sort_values(["perfume_id", "video_id", "transcript_sha256", "raw_row_index"])
        .drop_duplicates(dedupe_columns, keep="first")
        .reset_index(drop=True)
    )
    audit = {
        "raw_row_count": int(len(raw_df)),
        "resolved_raw_row_count": int(len(rows)),
        "deduplicated_membership_row_count": int(len(frame)),
        "duplicate_membership_row_count": duplicate_count,
        "invalid_or_empty_video_id_row_count": invalid_video_count,
        "empty_normalized_transcript_row_count": empty_transcript_count,
    }
    return frame, audit


def _group_digest(
    members: Sequence[str],
    *,
    video_ids: Sequence[str],
    transcript_hashes: Sequence[str],
) -> str:
    payload = {
        "version": SOURCE_IDENTITY_VERSION,
        "members": sorted(members),
        "video_ids": sorted(video_ids),
        "transcript_sha256": sorted(transcript_hashes),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sg_" + hashlib.sha256(encoded).hexdigest()


def build_source_groups(
    memberships_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute transitive source components and deterministic group IDs."""

    required = {"perfume_id", "video_id", "transcript_sha256"}
    missing = required - set(memberships_df.columns)
    if missing:
        raise ValueError(f"Membership table lacks columns: {sorted(missing)}")
    perfume_ids = sorted(set(memberships_df["perfume_id"].astype(str)))
    union_find = _UnionFind(perfume_ids)

    for source_col in ("video_id", "transcript_sha256"):
        populated = memberships_df.loc[
            memberships_df[source_col].astype(str).str.len().gt(0),
            ["perfume_id", source_col],
        ].drop_duplicates()
        for _, source_rows in populated.groupby(source_col, sort=True):
            source_ids = sorted(set(source_rows["perfume_id"].astype(str)))
            for perfume_id in source_ids[1:]:
                union_find.union(source_ids[0], perfume_id)

    members_by_root: Dict[str, list[str]] = defaultdict(list)
    for perfume_id in perfume_ids:
        members_by_root[union_find.find(perfume_id)].append(perfume_id)

    group_by_perfume: Dict[str, str] = {}
    component_rows: list[Dict[str, Any]] = []
    for members in sorted((sorted(values) for values in members_by_root.values()), key=lambda x: x[0]):
        component_memberships = memberships_df.loc[memberships_df["perfume_id"].isin(members)]
        video_ids = sorted(
            value for value in set(component_memberships["video_id"].astype(str)) if value
        )
        transcript_hashes = sorted(
            value
            for value in set(component_memberships["transcript_sha256"].astype(str))
            if value
        )
        group_id = _group_digest(
            members,
            video_ids=video_ids,
            transcript_hashes=transcript_hashes,
        )
        for perfume_id in members:
            group_by_perfume[perfume_id] = group_id
        component_rows.append(
            {
                SOURCE_GROUP_COLUMN: group_id,
                "source_group_size": len(members),
                "member_perfume_ids": "|".join(members),
                "valid_video_id_count": len(video_ids),
                "transcript_sha256_count": len(transcript_hashes),
            }
        )

    enriched = memberships_df.copy()
    enriched[SOURCE_GROUP_COLUMN] = enriched["perfume_id"].map(group_by_perfume)
    group_sizes = {row[SOURCE_GROUP_COLUMN]: row["source_group_size"] for row in component_rows}
    enriched["source_group_size"] = enriched[SOURCE_GROUP_COLUMN].map(group_sizes).astype(int)
    enriched = enriched.sort_values(
        [SOURCE_GROUP_COLUMN, "perfume_id", "video_id", "transcript_sha256"]
    ).reset_index(drop=True)
    components = pd.DataFrame(component_rows).sort_values(SOURCE_GROUP_COLUMN).reset_index(drop=True)
    return enriched, components


def build_outer_fold_manifest(
    perfume_df: pd.DataFrame,
    *,
    n_splits: int,
    unit_id_col: str = "perfume_id",
    group_col: str = SOURCE_GROUP_COLUMN,
    display_name_col: str | None = "Perfume",
) -> pd.DataFrame:
    """Assign atomic source groups with deterministic ``GroupKFold``."""

    required = {unit_id_col, group_col}
    missing = required - set(perfume_df.columns)
    if missing:
        raise ValueError(f"Source-group table lacks columns: {sorted(missing)}")
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    frame = perfume_df.copy()
    frame[unit_id_col] = frame[unit_id_col].astype(str).str.strip()
    frame[group_col] = frame[group_col].astype(str).str.strip()
    if frame[unit_id_col].duplicated().any() or (frame[unit_id_col] == "").any():
        raise ValueError("Outer split input requires unique, non-empty perfume IDs")
    if (frame[group_col] == "").any():
        raise ValueError("Outer split input contains empty source-group IDs")
    if frame[group_col].nunique() < n_splits:
        raise ValueError("Fewer source groups than requested outer folds")
    frame = frame.sort_values(unit_id_col).reset_index(drop=True)

    fold_ids = np.zeros(len(frame), dtype=int)
    splitter = GroupKFold(n_splits=n_splits)
    for fold_id, (_, test_idx) in enumerate(
        splitter.split(frame, groups=frame[group_col]),
        start=1,
    ):
        fold_ids[test_idx] = fold_id
    if (fold_ids == 0).any():
        raise RuntimeError("GroupKFold did not assign every perfume")

    columns = [unit_id_col]
    if display_name_col and display_name_col in frame.columns:
        columns.append(display_name_col)
    columns.append(group_col)
    manifest = frame[columns].copy()
    size_map = frame.groupby(group_col)[unit_id_col].nunique()
    manifest["source_group_size"] = manifest[group_col].map(size_map).astype(int)
    manifest["fold_id"] = fold_ids
    return manifest


def load_and_validate_outer_fold_manifest(
    path: Path,
    *,
    expected_ids: Sequence[str],
    n_splits: int,
    unit_id_col: str = "perfume_id",
    group_col: str = SOURCE_GROUP_COLUMN,
) -> pd.DataFrame:
    """Load a prebuilt outer manifest and enforce its hard constraints."""

    frame = pd.read_csv(path, keep_default_na=False)
    required = {unit_id_col, group_col, "fold_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Outer fold manifest lacks columns: {sorted(missing)}")
    frame[unit_id_col] = frame[unit_id_col].astype(str).str.strip()
    frame[group_col] = frame[group_col].astype(str).str.strip()
    frame["fold_id"] = pd.to_numeric(frame["fold_id"], errors="raise").astype(int)
    expected = _validate_canonical_ids(expected_ids, context="Expected outer-fold universe")
    if frame[unit_id_col].duplicated().any() or (frame[unit_id_col] == "").any():
        raise ValueError("Outer manifest contains duplicate or empty perfume IDs")
    if set(frame[unit_id_col]) != set(expected):
        raise ValueError("Outer manifest does not match the expected perfume universe")
    if set(frame["fold_id"]) != set(range(1, n_splits + 1)):
        raise ValueError(f"Outer manifest fold IDs must be exactly 1..{n_splits}")
    if (frame[group_col] == "").any():
        raise ValueError("Outer manifest contains an empty source-group ID")
    if int(frame.groupby(group_col)["fold_id"].nunique().max()) != 1:
        raise ValueError("A source group crosses outer folds")
    if "source_group_size" in frame.columns:
        observed_sizes = pd.to_numeric(frame["source_group_size"], errors="raise").astype(int)
        expected_sizes = frame[group_col].map(frame.groupby(group_col)[unit_id_col].nunique()).astype(int)
        if not observed_sizes.equals(expected_sizes):
            raise ValueError("Outer manifest source_group_size values are inconsistent")
        frame["source_group_size"] = observed_sizes
    return frame


def _stable_tie_break(seed: int, outer_fold_id: int, candidate_id: int) -> str:
    return hashlib.sha256(f"{seed}:{outer_fold_id}:{candidate_id}".encode("utf-8")).hexdigest()


def build_grouped_inner_validation_manifest(
    outer_manifest_df: pd.DataFrame,
    *,
    unit_id_col: str = "perfume_id",
    group_col: str = SOURCE_GROUP_COLUMN,
    fold_id_col: str = "fold_id",
    display_name_col: str | None = "Perfume",
    inner_n_splits: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """Build one deterministic, component-safe validation holdout per outer fold."""

    required = {unit_id_col, group_col, fold_id_col}
    missing = required - set(outer_manifest_df.columns)
    if missing:
        raise ValueError(f"Outer manifest lacks columns for inner split: {sorted(missing)}")
    if inner_n_splits < 2:
        raise ValueError("inner_n_splits must be >= 2")
    frame = outer_manifest_df.copy()
    frame[unit_id_col] = frame[unit_id_col].astype(str)
    frame[group_col] = frame[group_col].astype(str)
    frame[fold_id_col] = pd.to_numeric(frame[fold_id_col], errors="raise").astype(int)

    rows: list[Dict[str, Any]] = []
    for outer_fold_id in sorted(frame[fold_id_col].unique()):
        train_pool = frame.loc[frame[fold_id_col] != outer_fold_id].sort_values(unit_id_col).reset_index(drop=True)
        group_count = int(train_pool[group_col].nunique())
        actual_splits = min(inner_n_splits, group_count)
        if actual_splits < 2:
            raise ValueError(f"Outer fold {outer_fold_id} has too few training source groups")
        candidates: list[tuple[tuple[Any, ...], np.ndarray]] = []
        target_size = len(train_pool) / actual_splits
        splitter = GroupKFold(n_splits=actual_splits)
        for candidate_id, (_, val_idx) in enumerate(
            splitter.split(train_pool, groups=train_pool[group_col]),
            start=1,
        ):
            score = (
                abs(len(val_idx) - target_size),
                len(val_idx),
                _stable_tie_break(seed, int(outer_fold_id), candidate_id),
            )
            candidates.append((score, val_idx))
        _, selected_val_idx = min(candidates, key=lambda item: item[0])
        val_ids = set(train_pool.iloc[selected_val_idx][unit_id_col].astype(str))
        for record in train_pool.to_dict("records"):
            row: Dict[str, Any] = {
                "outer_fold_id": int(outer_fold_id),
                unit_id_col: str(record[unit_id_col]),
            }
            if display_name_col and display_name_col in train_pool.columns:
                row[display_name_col] = str(record[display_name_col])
            row[group_col] = str(record[group_col])
            row["inner_role"] = "val" if str(record[unit_id_col]) in val_ids else "train"
            rows.append(row)
    result = pd.DataFrame(rows)
    return result.sort_values(["outer_fold_id", unit_id_col]).reset_index(drop=True)


def validate_grouped_inner_manifest(
    inner_manifest_df: pd.DataFrame,
    outer_manifest_df: pd.DataFrame,
    *,
    unit_id_col: str = "perfume_id",
    group_col: str = SOURCE_GROUP_COLUMN,
    fold_id_col: str = "fold_id",
) -> Dict[str, int]:
    """Validate outer-test exclusion, universe coverage, and group isolation."""

    required_inner = {"outer_fold_id", unit_id_col, "inner_role"}
    missing = required_inner - set(inner_manifest_df.columns)
    if missing:
        raise ValueError(f"Inner manifest lacks columns: {sorted(missing)}")
    required_outer = {unit_id_col, group_col, fold_id_col}
    missing = required_outer - set(outer_manifest_df.columns)
    if missing:
        raise ValueError(f"Outer manifest lacks columns: {sorted(missing)}")

    inner = inner_manifest_df.copy()
    outer = outer_manifest_df.copy()
    inner["outer_fold_id"] = pd.to_numeric(inner["outer_fold_id"], errors="raise").astype(int)
    inner[unit_id_col] = inner[unit_id_col].astype(str)
    inner["inner_role"] = inner["inner_role"].astype(str).str.lower()
    outer[unit_id_col] = outer[unit_id_col].astype(str)
    outer[group_col] = outer[group_col].astype(str)
    outer[fold_id_col] = pd.to_numeric(outer[fold_id_col], errors="raise").astype(int)
    if inner.duplicated(["outer_fold_id", unit_id_col]).any():
        raise ValueError("Inner manifest duplicates an outer-fold/perfume pair")
    if not set(inner["inner_role"]).issubset({"train", "val"}):
        raise ValueError("Inner manifest roles must be train or val")
    if set(inner["outer_fold_id"]) != set(outer[fold_id_col]):
        raise ValueError("Inner manifest does not cover every outer fold")

    group_by_id = outer.set_index(unit_id_col)[group_col]
    validation_rows = 0
    for outer_fold_id in sorted(outer[fold_id_col].unique()):
        observed = inner.loc[inner["outer_fold_id"] == outer_fold_id].copy()
        expected_ids = set(outer.loc[outer[fold_id_col] != outer_fold_id, unit_id_col])
        if set(observed[unit_id_col]) != expected_ids:
            raise ValueError(f"Inner universe is incorrect for outer fold {outer_fold_id}")
        if set(observed["inner_role"]) != {"train", "val"}:
            raise ValueError(f"Outer fold {outer_fold_id} lacks train or val rows")
        observed["_expected_group"] = observed[unit_id_col].map(group_by_id)
        if observed["_expected_group"].isna().any():
            raise ValueError("Inner manifest contains an unknown perfume ID")
        if group_col in observed.columns:
            if not (
                observed[group_col].astype(str) == observed["_expected_group"].astype(str)
            ).all():
                raise ValueError("Inner manifest source-group IDs disagree with the outer manifest")
        if int(observed.groupby("_expected_group")["inner_role"].nunique().max()) != 1:
            raise ValueError(f"A source group crosses inner train/val in outer fold {outer_fold_id}")
        validation_rows += int((observed["inner_role"] == "val").sum())
    return {"inner_rows": int(len(inner)), "inner_validation_rows": validation_rows}


def validate_source_disjointness(
    memberships_df: pd.DataFrame,
    outer_manifest_df: pd.DataFrame,
    *,
    unit_id_col: str = "perfume_id",
    group_col: str = SOURCE_GROUP_COLUMN,
) -> Dict[str, Any]:
    """Directly count cross-fold component, video, and transcript overlap."""

    fold_by_id = outer_manifest_df.set_index(unit_id_col)["fold_id"]
    working = memberships_df.copy()
    working["fold_id"] = working[unit_id_col].map(fold_by_id)
    if working["fold_id"].isna().any():
        raise ValueError("Membership table contains an ID missing from the outer manifest")
    cross_group = int((working.groupby(group_col)["fold_id"].nunique() > 1).sum())
    counts: Dict[str, Any] = {"cross_fold_source_group_count": cross_group}
    for column, output_name in (
        ("video_id", "cross_fold_video_id_count"),
        ("transcript_sha256", "cross_fold_transcript_hash_count"),
    ):
        populated = working.loc[working[column].astype(str).str.len().gt(0)]
        counts[output_name] = int((populated.groupby(column)["fold_id"].nunique() > 1).sum())
    counts["validation_passed"] = all(
        counts[key] == 0
        for key in (
            "cross_fold_source_group_count",
            "cross_fold_video_id_count",
            "cross_fold_transcript_hash_count",
        )
    )
    return counts


def _build_fold_balance(
    fold_manifest: pd.DataFrame,
    feature_df: pd.DataFrame,
    *,
    labels: Sequence[str],
) -> pd.DataFrame:
    merged = fold_manifest.merge(
        feature_df[["perfume_id", *labels]],
        on="perfume_id",
        how="left",
        validate="one_to_one",
    )
    if merged[list(labels)].isna().any().any():
        raise ValueError("Fold balance could not resolve all labels")
    rows: list[Dict[str, Any]] = []
    for fold_id in sorted(merged["fold_id"].unique()):
        test = merged.loc[merged["fold_id"] == fold_id]
        train = merged.loc[merged["fold_id"] != fold_id]
        row: Dict[str, Any] = {
            "fold_id": int(fold_id),
            "test_perfume_count": int(len(test)),
            "test_source_group_count": int(test[SOURCE_GROUP_COLUMN].nunique()),
            "train_perfume_count": int(len(train)),
            "train_source_group_count": int(train[SOURCE_GROUP_COLUMN].nunique()),
        }
        for label in labels:
            row[f"test_{label}_positive"] = int(pd.to_numeric(test[label]).sum())
            row[f"train_{label}_positive"] = int(pd.to_numeric(train[label]).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def _shared_source_count(memberships: pd.DataFrame, column: str) -> int:
    populated = memberships.loc[memberships[column].astype(str).str.len().gt(0)]
    return int((populated.groupby(column)["perfume_id"].nunique() > 1).sum())


def build_cross_fold_metadata_diagnostic(
    raw_df: pd.DataFrame,
    fold_manifest: pd.DataFrame,
) -> Dict[str, Dict[str, int]]:
    """Count repeated normalized audit metadata without making it a split key."""

    fold_by_id = fold_manifest.set_index("perfume_id")["fold_id"]
    working = raw_df.copy()
    working["perfume_id"] = working["Perfume"].map(normalize_perfume_id)
    working["fold_id"] = working["perfume_id"].map(fold_by_id)
    result: Dict[str, Dict[str, int]] = {}
    for column in ("Title", "Description", "Channel Title"):
        if column not in working.columns:
            continue
        normalized = working[column].map(normalize_transcript)
        populated = working.loc[(normalized != "") & working["fold_id"].notna()].copy()
        populated["_normalized_metadata"] = normalized.loc[populated.index]
        fold_counts = populated.groupby("_normalized_metadata")["fold_id"].nunique()
        crossing = set(fold_counts.loc[fold_counts > 1].index)
        result[column] = {
            "unique_normalized_value_count": int(populated["_normalized_metadata"].nunique()),
            "cross_fold_normalized_value_count": int(len(crossing)),
            "affected_perfume_count": int(
                populated.loc[
                    populated["_normalized_metadata"].isin(crossing), "perfume_id"
                ].nunique()
            ),
        }
    return result


def prepare_source_grouped_folds(
    *,
    raw_csv: Path,
    labels_csv: Path,
    feature_label_csv: Path,
    out_root: Path,
    config_path: Path | None = None,
    n_splits: int = 5,
    inner_n_splits: int = 10,
    seed: int = 42,
) -> Dict[str, Path]:
    """Build source-audit/split artifacts, optionally enforcing frozen hashes."""

    raw_csv = Path(raw_csv)
    labels_csv = Path(labels_csv)
    feature_label_csv = Path(feature_label_csv)
    out_root = Path(out_root)
    config_path = Path(config_path) if config_path is not None else None
    input_paths = {
        "raw_csv": raw_csv,
        "labels_csv": labels_csv,
        "feature_label_csv": feature_label_csv,
    }
    input_sha256, frozen_input_validation = validate_configured_frozen_inputs(
        config_path=config_path,
        input_paths=input_paths,
    )
    source_dir = out_root / "source_audit"
    tables_dir = out_root / "model_benchmark" / "tables"
    paths = {
        "memberships": source_dir / "source_memberships.csv",
        "components": source_dir / "source_components.csv",
        "audit": source_dir / "source_group_audit.json",
        "outer": tables_dir / "fold_assignments.csv",
        "inner": tables_dir / "inner_validation_folds.csv",
        "balance": tables_dir / "fold_balance.csv",
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite source-grouped artifacts: {existing[0]}")

    labels_df = pd.read_csv(labels_csv, keep_default_na=False)
    feature_df = pd.read_csv(feature_label_csv, keep_default_na=False)
    raw_df = pd.read_csv(raw_csv, keep_default_na=False)
    if "perfume_id" not in labels_df or "perfume_id" not in feature_df:
        raise ValueError("Labels and feature-label inputs must contain perfume_id")
    label_ids = _validate_canonical_ids(labels_df["perfume_id"].tolist(), context="Labels CSV")
    feature_ids = _validate_canonical_ids(
        feature_df["perfume_id"].tolist(), context="Feature-label CSV"
    )
    if set(label_ids) != set(feature_ids):
        raise ValueError("Labels and feature-label perfume universes differ")
    missing_labels = set(OFFICIAL_LABELS) - set(feature_df.columns)
    if missing_labels:
        raise ValueError(f"Feature-label CSV lacks official labels: {sorted(missing_labels)}")

    memberships, membership_audit = build_source_memberships(
        raw_df,
        canonical_perfume_ids=label_ids,
    )
    memberships, components = build_source_groups(memberships)
    names = feature_df.set_index("perfume_id")["Perfume"].astype(str)
    perfume_groups = (
        memberships[["perfume_id", SOURCE_GROUP_COLUMN, "source_group_size"]]
        .drop_duplicates("perfume_id")
        .sort_values("perfume_id")
        .reset_index(drop=True)
    )
    perfume_groups["Perfume"] = perfume_groups["perfume_id"].map(names)
    if perfume_groups["Perfume"].isna().any():
        raise ValueError("Could not resolve display names for every perfume")
    fold_manifest = build_outer_fold_manifest(
        perfume_groups,
        n_splits=n_splits,
        display_name_col="Perfume",
    )
    inner_manifest = build_grouped_inner_validation_manifest(
        fold_manifest,
        inner_n_splits=inner_n_splits,
        seed=seed,
    )
    validate_grouped_inner_manifest(inner_manifest, fold_manifest)
    disjointness = validate_source_disjointness(memberships, fold_manifest)
    if not disjointness["validation_passed"]:
        raise ValueError(f"Generated source-grouped folds failed validation: {disjointness}")

    fold_by_id = fold_manifest.set_index("perfume_id")["fold_id"]
    memberships["fold_id"] = memberships["perfume_id"].map(fold_by_id).astype(int)
    group_fold = fold_manifest.groupby(SOURCE_GROUP_COLUMN)["fold_id"].first()
    components["fold_id"] = components[SOURCE_GROUP_COLUMN].map(group_fold).astype(int)
    fold_balance = _build_fold_balance(
        fold_manifest,
        feature_df,
        labels=OFFICIAL_LABELS,
    )

    component_sizes = components["source_group_size"].astype(int)
    largest = components.sort_values(
        ["source_group_size", SOURCE_GROUP_COLUMN], ascending=[False, True]
    ).iloc[0]
    source_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    memberships.to_csv(paths["memberships"], index=False, lineterminator="\n")
    components.to_csv(paths["components"], index=False, lineterminator="\n")
    fold_manifest.to_csv(paths["outer"], index=False, lineterminator="\n")
    inner_manifest.to_csv(paths["inner"], index=False, lineterminator="\n")
    fold_balance.to_csv(paths["balance"], index=False, lineterminator="\n")

    audit: Dict[str, Any] = {
        "schema_version": 1,
        "status": "passed",
        "validation_passed": True,
        "grouping_version": SOURCE_IDENTITY_VERSION,
        "seed": int(seed),
        "n_splits": int(n_splits),
        "inner_n_splits": int(inner_n_splits),
        "input_paths": {
            "raw_csv": str(raw_csv),
            "labels_csv": str(labels_csv),
            "feature_label_csv": str(feature_label_csv),
        },
        "input_sha256": input_sha256,
        "frozen_input_validation": frozen_input_validation,
        "n_perfumes": int(len(fold_manifest)),
        "n_source_components": int(len(components)),
        "largest_component": {
            SOURCE_GROUP_COLUMN: str(largest[SOURCE_GROUP_COLUMN]),
            "source_group_size": int(largest["source_group_size"]),
            "share_of_perfumes": float(int(largest["source_group_size"]) / len(fold_manifest)),
        },
        "component_size_distribution": {
            str(size): int(count)
            for size, count in sorted(Counter(component_sizes.tolist()).items())
        },
        "valid_video_id_count": int(
            memberships.loc[memberships["video_id"].astype(str).str.len().gt(0), "video_id"].nunique()
        ),
        "normalized_transcript_hash_count": int(
            memberships.loc[
                memberships["transcript_sha256"].astype(str).str.len().gt(0),
                "transcript_sha256",
            ].nunique()
        ),
        "shared_video_id_count": _shared_source_count(memberships, "video_id"),
        "shared_transcript_hash_count": _shared_source_count(
            memberships, "transcript_sha256"
        ),
        "audit_only_metadata_cross_fold_diagnostic": build_cross_fold_metadata_diagnostic(
            raw_df,
            fold_manifest,
        ),
        "deep_model_input_policy": {
            "included": ["Final Transcript excerpt"],
            "excluded_audit_metadata": ["Title", "Description", "URL", "Channel Title"],
            "reason": "repeated reviewer/channel metadata is not covered by source_identity_v1",
        },
        "fold_sizes": {
            str(int(row.fold_id)): int(row.test_perfume_count)
            for row in fold_balance.itertuples(index=False)
        },
        "fold_source_group_counts": {
            str(int(row.fold_id)): int(row.test_source_group_count)
            for row in fold_balance.itertuples(index=False)
        },
        "cross_fold_source_group_count": int(disjointness["cross_fold_source_group_count"]),
        "cross_fold_video_id_count": int(disjointness["cross_fold_video_id_count"]),
        "cross_fold_transcript_hash_count": int(
            disjointness["cross_fold_transcript_hash_count"]
        ),
        "normalization": {
            "video_id": "strip; case-sensitive ^[A-Za-z0-9_-]{11}$; ignore invalid/empty",
            "transcript": (
                "html_unescape; unicode_nfkc; casefold; replace non-alphanumeric runs "
                "with spaces; collapse whitespace; sha256 utf-8"
            ),
            "component_rule": "undirected transitive closure over either shared source key",
        },
        "manifest_sha256": {
            "outer": sha256_file(paths["outer"]),
            "inner": sha256_file(paths["inner"]),
            "fold_balance": sha256_file(paths["balance"]),
        },
        **membership_audit,
    }
    paths["audit"].write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return paths


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        help="YAML config whose provenance.frozen_input_sha256 values must match.",
    )
    parser.add_argument("--raw-csv", type=Path, required=True)
    parser.add_argument("--labels-csv", type=Path, required=True)
    parser.add_argument("--feature-label-csv", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--inner-n-splits", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paths = prepare_source_grouped_folds(
        raw_csv=args.raw_csv,
        labels_csv=args.labels_csv,
        feature_label_csv=args.feature_label_csv,
        out_root=args.out_root,
        config_path=args.config,
        n_splits=args.n_splits,
        inner_n_splits=args.inner_n_splits,
        seed=args.seed,
    )
    print(f"Source-group audit: {paths['audit']}")
    print(f"Outer fold manifest: {paths['outer']}")
    print(f"Inner validation manifest: {paths['inner']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

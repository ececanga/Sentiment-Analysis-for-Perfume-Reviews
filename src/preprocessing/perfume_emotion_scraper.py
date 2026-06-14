# Date Created: 2025-09-20 17:23:40+03:00
# Date Last Modified: 2026-05-15

"""Collect YouTube perfume review metadata and transcripts.

The original script was a top-level scraper. It is now import-safe and exposes
the same collection logic through a CLI so Phase 2A can resume safely without
touching the frozen 784-row raw transcript files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import pandas as pd

PROJECT_ROOT_FALLBACK = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_FALLBACK) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FALLBACK))

from src.labeling.run_gemini_labeling import normalize_name_key
from src.utils.paths import PROJECT_ROOT, data_path

csv.field_size_limit(sys.maxsize)

BATCH_SIZE = 95
POPULAR_BRANDS = [
    "chanel",
    "dior",
    "versace",
    "armani",
    "ysl",
    "tom ford",
    "gucci",
    "lancome",
    "burberry",
    "jo malone",
    "byredo",
    "creed",
    "le labo",
    "narciso",
    "zara",
    "marc jacobs",
    "maison francis kurkdjian",
    "mugler",
    "valentino",
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
    "sexy",
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
}
RAW_COLUMNS = [
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
STATUS_COLUMNS = ["Perfume", "Status", "Attempts", "Last Error"]
QUERY_TEMPLATES = ("{perfume} perfume", "{perfume} fragrance", "{perfume} review")
BRAND_QUERY_TEMPLATES = (
    "{brand} {perfume} review",
    "{brand} {perfume} perfume review",
    "{brand} {perfume} fragrance review",
    "{perfume} {brand} review",
)
STATUS_ALIASES = {
    "done": "done",
    "no videos": "search_no_videos",
    "no_videos": "search_no_videos",
    "search_no_videos": "search_no_videos",
    "no results": "transcript_unavailable",
    "no_results": "transcript_unavailable",
    "transcript_unavailable": "transcript_unavailable",
    "too_short": "too_short",
    "auth_error": "auth_error",
    "api_error": "api_error",
    "quota_error": "quota_error",
    "network_error": "network_error",
    "youtube_api_error": "youtube_api_error",
    "transcript_fetch_error": "transcript_fetch_error",
}
API_ERROR_STATUSES = (
    "auth_error",
    "quota_error",
    "network_error",
    "youtube_api_error",
    "api_error",
)


@dataclass(frozen=True)
class Candidate:
    perfume: str
    brand: str = ""
    quality_flags: str = ""
    source: str = "target_csv"


@dataclass(frozen=True)
class YouTubeKey:
    label: str
    value: str


@dataclass(frozen=True)
class ScraperConfig:
    target_csv: Path
    perfume_col: str | None
    candidate_queue_csv: Path | None
    raw_output_csv: Path
    status_csv: Path
    seed_status_csv: Path | None
    batch_size: int
    limit: int | None
    resume: bool
    retry_statuses: set[str]
    dry_run: bool
    max_results: int
    max_workers: int
    min_words: int
    sleep_sec: float
    max_query_templates_per_perfume: int | None
    stop_after_valid_videos_per_perfume: int | None
    query_templates: tuple[str, ...]
    use_brand_queries: bool
    api_key: str
    api_key_env: str = "YOUTUBE_API_KEY"
    key_label: str = "key_env"
    verbose: bool = True
    diagnose_youtube_key: bool = False
    diagnose_youtube_keys: bool = False
    youtube_key_file: Path | None = None
    max_keys: int | None = None
    debug_quota: bool = False


def score_perfume(name: str) -> bool:
    return any(brand in str(name).lower() for brand in POPULAR_BRANDS)


def clean_text(text: Any) -> str:
    return " ".join(str(text or "").replace("\r", " ").replace("\n", " ").split())


def redact_secret(text: str, secret: str) -> str:
    if not secret or secret == "YOUR_API_KEY_HERE":
        return text
    return str(text).replace(secret, "<REDACTED_API_KEY>")


def key_fingerprint(secret: str) -> str:
    if not secret or secret == "YOUR_API_KEY_HERE":
        return "none"
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()[:12]


def normalize_status(status: str) -> str:
    return STATUS_ALIASES.get(str(status or "").strip().lower(), str(status or "").strip().lower())


def split_status_values(values: Sequence[str] | None) -> set[str]:
    out: set[str] = set()
    for value in values or []:
        for part in str(value).split(","):
            status = normalize_status(part)
            if status:
                out.add(status)
    return out


def load_env_fallback(env_name: str) -> None:
    if os.getenv(env_name):
        return
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key != env_name:
            continue
        value = value.strip().strip('"').strip("'")
        if value:
            os.environ[env_name] = value
        return


def clean_env_value(value: str) -> str:
    return str(value or "").strip().strip('"').strip("'")


def key_label_from_name(name: str, default: str = "key") -> str:
    suffix = str(name).removeprefix("YOUTUBE_API_KEY").strip("_").lower()
    return f"key_{suffix}" if suffix else default


def parse_youtube_key_file(path: Path | None) -> List[YouTubeKey]:
    if path is None or not path.exists():
        return []

    keys: List[YouTubeKey] = []
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").strip()
        if "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        if not name.startswith("YOUTUBE_API_KEY"):
            continue
        value = clean_env_value(value)
        if not value or value == "YOUR_API_KEY_HERE":
            continue
        keys.append(YouTubeKey(label=key_label_from_name(name), value=value))
    return keys


def load_youtube_key_pool(
    *,
    api_key_env: str,
    youtube_key_file: Path | None,
    max_keys: int | None = None,
    key_label: str | None = None,
) -> List[YouTubeKey]:
    if max_keys is not None and max_keys <= 0:
        return []

    keys: List[YouTubeKey] = []

    shell_key = clean_env_value(os.getenv(api_key_env, ""))
    if shell_key and shell_key != "YOUR_API_KEY_HERE":
        keys.append(YouTubeKey(label=key_label or "key_env", value=shell_key))

    prefix = f"{api_key_env}_"
    for name in sorted(os.environ):
        if not name.startswith(prefix):
            continue
        value = clean_env_value(os.environ.get(name, ""))
        if value and value != "YOUR_API_KEY_HERE":
            keys.append(YouTubeKey(label=key_label_from_name(name), value=value))

    keys.extend(parse_youtube_key_file(youtube_key_file))

    out: List[YouTubeKey] = []
    seen_values: set[str] = set()
    seen_labels: set[str] = set()
    for item in keys:
        if item.value in seen_values:
            continue
        seen_values.add(item.value)
        label = item.label
        if label in seen_labels:
            suffix = 2
            while f"{label}_{suffix}" in seen_labels:
                suffix += 1
            label = f"{label}_{suffix}"
        seen_labels.add(label)
        out.append(YouTubeKey(label=label, value=item.value))
        if max_keys is not None and len(out) >= max_keys:
            break
    return out


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8", engine="python")


def load_perfume_names(path: Path, perfume_col: str | None) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Target perfume CSV not found: {path}")
    df = pd.read_csv(path, encoding="ISO-8859-1")
    if df.empty:
        return []
    if perfume_col:
        if perfume_col not in df.columns:
            raise ValueError(f"Target CSV is missing perfume column '{perfume_col}': {path}")
        series = df[perfume_col]
    else:
        series = df.iloc[:, 0]

    names = [str(value).strip().lower() for value in series.dropna().tolist() if str(value).strip()]
    names = list(dict.fromkeys(names))
    return sorted(names, key=lambda name: (not score_perfume(name), name))


def load_target_candidates(path: Path, perfume_col: str | None) -> List[Candidate]:
    return [Candidate(perfume=name) for name in load_perfume_names(path, perfume_col)]


def parse_boolish(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def load_candidate_queue(path: Path) -> List[Candidate]:
    if not path.exists():
        raise FileNotFoundError(f"Candidate queue CSV not found: {path}")
    df = pd.read_csv(path, encoding="utf-8", engine="python")
    if df.empty:
        return []
    if "perfume_name" not in df.columns:
        raise ValueError(f"Candidate queue CSV must include perfume_name column: {path}")
    rows = df.to_dict(orient="records")
    if "recommended_for_next_batch" in df.columns:
        recommended = [row for row in rows if parse_boolish(row.get("recommended_for_next_batch"))]
        rows = recommended if recommended else rows

    out: List[Candidate] = []
    seen: set[str] = set()
    for row in rows:
        perfume = clean_text(row.get("perfume_name", ""))
        if not perfume:
            continue
        key = normalize_name_key(perfume)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            Candidate(
                perfume=perfume,
                brand=clean_text(row.get("brand", "")),
                quality_flags=clean_text(row.get("candidate_quality_flags", "")),
                source="candidate_queue",
            )
        )
    return out


def load_status_maps(paths: Iterable[Path]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for path in paths:
        if path is None or not path.exists():
            continue
        df = read_csv(path)
        if df.empty:
            continue
        if "Perfume" not in df.columns or "Status" not in df.columns:
            raise ValueError(f"Status CSV must include Perfume and Status columns: {path}")
        for _, row in df.iterrows():
            perfume = str(row.get("Perfume", "")).strip()
            if not perfume:
                continue
            key = normalize_name_key(perfume)
            out[key] = {
                "Perfume": perfume,
                "Status": normalize_status(str(row.get("Status", ""))),
                "Attempts": int(float(row.get("Attempts", 1) or 1)) if "Attempts" in df.columns else 1,
                "Last Error": str(row.get("Last Error", "") or "") if "Last Error" in df.columns else "",
            }
    return out


def select_candidates(candidates: Sequence[Candidate], cfg: ScraperConfig) -> List[Candidate]:
    status_paths = []
    if cfg.seed_status_csv is not None:
        status_paths.append(cfg.seed_status_csv)
    status_paths.append(cfg.status_csv)
    status_by_name = load_status_maps(status_paths) if cfg.resume else {}

    selected: List[Candidate] = []
    for candidate in candidates:
        key = normalize_name_key(candidate.perfume)
        status = status_by_name.get(key, {}).get("Status")
        if cfg.resume and status and status not in cfg.retry_statuses:
            continue
        selected.append(candidate)

    max_count = cfg.batch_size
    if cfg.limit is not None:
        max_count = min(max_count, cfg.limit)
    return selected[:max_count]


def query_templates_for_candidate(candidate: Candidate, cfg: ScraperConfig) -> tuple[str, ...]:
    templates: List[str] = []
    if cfg.use_brand_queries and candidate.brand:
        templates.extend(BRAND_QUERY_TEMPLATES)
    templates.extend(cfg.query_templates)
    seen: set[str] = set()
    out: List[str] = []
    for template in templates:
        if template not in seen:
            seen.add(template)
            out.append(template)
    if cfg.max_query_templates_per_perfume and cfg.max_query_templates_per_perfume > 0:
        out = out[: cfg.max_query_templates_per_perfume]
    return tuple(out)


def build_queries(candidate: Candidate, cfg: ScraperConfig) -> List[str]:
    return [
        template.format(perfume=candidate.perfume, brand=candidate.brand)
        for template in query_templates_for_candidate(candidate, cfg)
    ]


def valid_rows_from_videos(candidate: Candidate, videos: Sequence[Dict[str, Any]], cfg: ScraperConfig) -> tuple[List[Dict[str, Any]], int, List[str]]:
    if not videos:
        return [], 0, []

    if cfg.sleep_sec > 0:
        time.sleep(cfg.sleep_sec)
    with ThreadPoolExecutor(max_workers=cfg.max_workers) as executor:
        fetched = list(executor.map(fetch_transcript, videos))

    valid: List[Dict[str, Any]] = []
    non_empty_transcripts = 0
    transcript_errors = [row.get("_transcript_error", "") for row in fetched if row.get("_transcript_error")]
    for video in fetched:
        txt = clean_text(video.get("Transcript", ""))
        if txt:
            non_empty_transcripts += 1
        word_count = len(txt.split())
        if word_count < cfg.min_words:
            continue
        valid.append(
            {
                "Perfume": candidate.perfume,
                "Video ID": video.get("Video ID", ""),
                "Title": video.get("Title", ""),
                "Description": video.get("Description", ""),
                "URL": video.get("URL", ""),
                "Channel Title": video.get("Channel Title", ""),
                "Transcript": txt,
                "Transcript Length": word_count,
                "Emotion Keyword Count": emotion_keyword_count(txt),
                "Final Transcript": txt,
            }
        )

    return valid, non_empty_transcripts, transcript_errors


def existing_raw_keys(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    df = read_csv(path)
    if df.empty:
        return set()
    if "Perfume" not in df.columns or "Video ID" not in df.columns:
        raise ValueError(f"Raw output CSV must include Perfume and Video ID columns: {path}")
    keys: set[tuple[str, str]] = set()
    for _, row in df.iterrows():
        perfume_key = normalize_name_key(str(row.get("Perfume", "")))
        video_id = str(row.get("Video ID", "")).strip()
        if perfume_key and video_id:
            keys.add((perfume_key, video_id))
    return keys


def write_raw_rows_incremental(path: Path, rows: Sequence[Dict[str, Any]]) -> int:
    if not rows:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_keys = existing_raw_keys(path)
    new_rows: List[Dict[str, Any]] = []
    for row in rows:
        key = (normalize_name_key(str(row.get("Perfume", ""))), str(row.get("Video ID", "")).strip())
        if not key[0] or not key[1] or key in existing_keys:
            continue
        existing_keys.add(key)
        new_rows.append({column: row.get(column, "") for column in RAW_COLUMNS})
    if not new_rows:
        return 0

    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerows(new_rows)
    return len(new_rows)


def update_status(path: Path, perfume: str, status: str, attempts: int, last_error: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    if path.exists() and path.stat().st_size > 0:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if normalize_name_key(row.get("Perfume", "")) != normalize_name_key(perfume):
                    rows.append(
                        {
                            "Perfume": row.get("Perfume", ""),
                            "Status": normalize_status(row.get("Status", "")),
                            "Attempts": row.get("Attempts", ""),
                            "Last Error": row.get("Last Error", ""),
                        }
                    )
    rows.append(
        {
            "Perfume": perfume,
            "Status": status,
            "Attempts": attempts,
            "Last Error": last_error,
        }
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=STATUS_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def classify_api_error(payload: Dict[str, Any], response_status: int | None = None) -> str | None:
    error = payload.get("error")
    if not isinstance(error, dict):
        return None
    reason_text = " ".join(
        str(item.get("reason", "")) for item in error.get("errors", []) if isinstance(item, dict)
    ).lower()
    message = str(error.get("message", "")).lower()
    code = int(error.get("code", response_status or 0) or 0)
    combined = f"{reason_text} {message}"
    if any(
        token in combined
        for token in (
            "keynotvalid",
            "api key not valid",
            "invalid api key",
            "key invalid",
            "accessnotconfigured",
            "iprefererblocked",
            "referer",
            "credential",
            "unauthorized",
        )
    ):
        return "auth_error"
    if code in {401, 407}:
        return "auth_error"
    if code == 403 and any(token in combined for token in ("quota", "ratelimit", "rate limit", "dailylimit")):
        return "quota_error"
    if response_status is not None and response_status >= 400:
        return "youtube_api_error"
    return None


def search_youtube(query: str, *, api_key: str, max_results: int) -> Dict[str, Any]:
    try:
        import requests
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("api_error: missing requests dependency") from exc

    url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "part": "snippet",
        "q": query,
        "maxResults": max_results,
        "type": "video",
        "key": api_key,
    }
    try:
        response = requests.get(url, params=params, timeout=30)
    except requests.RequestException as exc:
        error = redact_secret(f"network_error: {type(exc).__name__}: {exc}", api_key)
        raise RuntimeError(error) from exc
    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError(f"api_error: invalid_json_response status={response.status_code}") from exc
    classified = classify_api_error(payload, response.status_code)
    if classified:
        message = payload.get("error", {}).get("message", "")
        raise RuntimeError(f"{classified}: {message}")
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(f"youtube_api_error: HTTP {response.status_code}") from exc
    return payload


def build_video_rows(perfume: str, query: str, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in payload.get("items", []):
        video_id = str(item.get("id", {}).get("videoId", "")).strip()
        if not video_id:
            continue
        snippet = item.get("snippet", {})
        rows.append(
            {
                "Perfume": perfume,
                "Video ID": video_id,
                "Title": snippet.get("title", ""),
                "Description": snippet.get("description", ""),
                "URL": f"https://www.youtube.com/watch?v={video_id}",
                "Channel Title": snippet.get("channelTitle", ""),
                "_query": query,
            }
        )
    return rows


def get_youtube_transcript(video_id: str, max_retries: int = 3) -> tuple[str, str]:
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        from youtube_transcript_api._errors import NoTranscriptFound, TranscriptsDisabled
    except ImportError:
        return "", "missing youtube_transcript_api dependency"

    last_error = ""
    for _ in range(max_retries):
        try:
            if hasattr(YouTubeTranscriptApi, "get_transcript"):
                transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
                return " ".join(entry["text"] for entry in transcript), ""
            transcript = YouTubeTranscriptApi().fetch(video_id, languages=["en"])
            return " ".join(entry["text"] for entry in transcript.to_raw_data()), ""
        except (TranscriptsDisabled, NoTranscriptFound) as exc:
            return "", type(exc).__name__
        except Exception as exc:  # noqa: BLE001 - external API can raise several transient types
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(1)
    return "", last_error or "transcript_fetch_failed"


def fetch_transcript(video: Dict[str, Any]) -> Dict[str, Any]:
    transcript, error = get_youtube_transcript(str(video["Video ID"]))
    out = dict(video)
    out["Transcript"] = transcript
    out["_transcript_error"] = error
    return out


def emotion_keyword_count(text: str) -> int:
    return sum(1 for word in str(text).lower().split() if word in EMOTION_KEYWORDS)


def collect_for_candidate(candidate: Candidate, cfg: ScraperConfig) -> tuple[str, List[Dict[str, Any]], str]:
    valid_rows: List[Dict[str, Any]] = []
    api_errors: List[str] = []
    seen_video_ids: set[str] = set()
    saw_videos = False
    non_empty_transcripts = 0
    transcript_errors: List[str] = []
    stop_after = cfg.stop_after_valid_videos_per_perfume if cfg.stop_after_valid_videos_per_perfume else None
    queries = build_queries(candidate, cfg)
    query_count = len(queries)

    for query_index, query in enumerate(queries, start=1):
        if cfg.verbose:
            print(f"Searching: {query}")
        if cfg.debug_quota:
            print(
                "debug_quota "
                f"key_label={cfg.key_label} key_fingerprint={key_fingerprint(cfg.api_key)} "
                f"candidate={candidate.perfume!r} query_index={query_index}/{query_count} "
                f"max_results={cfg.max_results} query={query!r}"
            )
        try:
            payload = search_youtube(query, api_key=cfg.api_key, max_results=cfg.max_results)
        except RuntimeError as exc:
            error_text = str(exc)
            if cfg.debug_quota:
                print(
                    "debug_quota "
                    f"key_label={cfg.key_label} key_fingerprint={key_fingerprint(cfg.api_key)} "
                    f"candidate={candidate.perfume!r} query_index={query_index}/{query_count} "
                    f"quota_error={str(error_text).startswith('quota_error')}"
                )
            api_errors.append(error_text)
            for status in ("auth_error", "quota_error", "network_error"):
                if error_text.startswith(status):
                    error_with_query = (
                        f"query_index={query_index}/{query_count}; query={query!r}; {error_text}"
                    )
                    if valid_rows:
                        return status, valid_rows, error_with_query
                    return status, [], error_with_query
            continue
        if cfg.debug_quota:
            print(
                "debug_quota "
                f"key_label={cfg.key_label} key_fingerprint={key_fingerprint(cfg.api_key)} "
                f"candidate={candidate.perfume!r} query_index={query_index}/{query_count} "
                f"quota_error=false result_items={len(payload.get('items', []))}"
            )
        videos = build_video_rows(candidate.perfume, query, payload)
        saw_videos = saw_videos or bool(videos)
        new_videos: List[Dict[str, Any]] = []
        for video in videos:
            video_id = str(video.get("Video ID", "")).strip()
            if not video_id or video_id in seen_video_ids:
                continue
            seen_video_ids.add(video_id)
            new_videos.append(video)

        query_valid, query_non_empty, query_errors = valid_rows_from_videos(candidate, new_videos, cfg)
        valid_rows.extend(query_valid)
        non_empty_transcripts += query_non_empty
        transcript_errors.extend(query_errors)

        if stop_after is not None and len(valid_rows) >= stop_after:
            return "done", valid_rows[:stop_after], ""

    if api_errors and not saw_videos and not valid_rows:
        status = "api_error"
        for candidate_status in API_ERROR_STATUSES:
            if any(error.startswith(candidate_status) for error in api_errors):
                status = candidate_status
                break
        return status, [], "; ".join(api_errors)[:500]
    if not saw_videos:
        return "search_no_videos", [], ""
    if valid_rows:
        return "done", valid_rows, ""
    if non_empty_transcripts:
        return "too_short", [], ""
    if transcript_errors and any(
        "NoTranscriptFound" not in error and "TranscriptsDisabled" not in error for error in transcript_errors
    ):
        return "transcript_fetch_error", [], "; ".join(transcript_errors)[:500]
    return "transcript_unavailable", [], "; ".join(transcript_errors)[:500]


def diagnose_youtube_key(cfg: ScraperConfig) -> int:
    if cfg.api_key == "YOUR_API_KEY_HERE":
        print("youtube_key_diagnostic: failure")
        print("category=auth_error")
        print("reason=missing_api_key")
        print("likely_problem=auth")
        return 1

    try:
        payload = search_youtube("perfume review", api_key=cfg.api_key, max_results=1)
    except RuntimeError as exc:
        error_text = redact_secret(str(exc), cfg.api_key)
        category = error_text.split(":", 1)[0] if ":" in error_text else "unknown_error"
        likely = {
            "auth_error": "auth",
            "quota_error": "quota",
            "network_error": "network",
            "youtube_api_error": "youtube_api",
            "api_error": "unknown",
        }.get(category, "unknown")
        print("youtube_key_diagnostic: failure")
        print(f"category={category}")
        print(f"reason={error_text[:240]}")
        print(f"likely_problem={likely}")
        return 1

    item_count = len(payload.get("items", []))
    print("youtube_key_diagnostic: success")
    print("category=ok")
    print(f"result_items={item_count}")
    print("likely_problem=none")
    return 0


def diagnostic_category(error_text: str) -> str:
    raw = str(error_text or "").split(":", 1)[0].strip()
    if raw in {"quota_error", "auth_error", "network_error", "youtube_api_error", "api_error"}:
        return raw
    return "unknown"


def diagnose_youtube_keys(keys: Sequence[YouTubeKey]) -> int:
    if not keys:
        print("youtube_key_diagnostics: no_keys_found")
        return 1

    usable = 0
    print(f"youtube_key_diagnostics: keys_found={len(keys)}")
    for item in keys:
        try:
            payload = search_youtube("perfume review", api_key=item.value, max_results=1)
        except RuntimeError as exc:
            category = diagnostic_category(redact_secret(str(exc), item.value))
            print(f"label={item.label} success=false category={category}")
            continue

        usable += 1
        print(f"label={item.label} success=true category=ok result_items={len(payload.get('items', []))}")
    print(f"youtube_key_diagnostics: usable_keys={usable}")
    return 0 if usable else 1


def parse_args() -> ScraperConfig:
    parser = argparse.ArgumentParser(description="Safely collect YouTube perfume review transcripts.")
    parser.add_argument(
        "--target-csv",
        type=Path,
        default=data_path("perfume_level", "final_perfume_data.csv"),
        help="Perfume source CSV.",
    )
    parser.add_argument(
        "--perfume-col",
        default="Name",
        help="Perfume name column in --target-csv. Use empty string to read the first column.",
    )
    parser.add_argument(
        "--candidate-queue-csv",
        type=Path,
        default=None,
        help="Optional curated queue CSV with perfume_name and optional brand/recommended fields.",
    )
    parser.add_argument(
        "--output-csv",
        "--raw-output-csv",
        dest="raw_output_csv",
        type=Path,
        default=data_path("raw", "YouTube_Perfume_Reviews_Phase2A_Partial.csv"),
        help="Append-only raw transcript output CSV. Defaults to a Phase 2A file.",
    )
    parser.add_argument(
        "--status-csv",
        type=Path,
        default=data_path("intermediate", "phase2a_processed_perfumes.csv"),
        help="Phase 2A collection status CSV.",
    )
    parser.add_argument(
        "--seed-status-csv",
        type=Path,
        default=data_path("intermediate", "processed_perfumes.csv"),
        help="Existing status CSV used for resume/exclusion, not modified.",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--limit", type=int, default=None, help="Optional cap within --batch-size.")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip already processed perfumes unless their status is listed in --retry-status.",
    )
    parser.add_argument(
        "--retry-status",
        action="append",
        default=[],
        help="Status to retry. May be repeated or comma-separated; accepts legacy No videos/No results aliases.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show planned perfumes and queries without API calls/writes.")
    parser.add_argument("--max-results", type=int, default=10)
    parser.add_argument("--max-workers", type=int, default=10)
    parser.add_argument("--min-words", type=int, default=20)
    parser.add_argument("--sleep-sec", type=float, default=1.0)
    parser.add_argument(
        "--max-query-templates-per-perfume",
        type=int,
        default=0,
        help="Limit search.list query templates per perfume. Use 0 for all templates.",
    )
    parser.add_argument(
        "--stop-after-valid-videos-per-perfume",
        type=int,
        default=0,
        help="Stop searching/fetching a perfume after this many valid transcript videos. Use 0 for no cap.",
    )
    parser.add_argument(
        "--query-template",
        action="append",
        default=[],
        help="Query template containing {perfume}. Repeat to override defaults.",
    )
    parser.add_argument(
        "--use-brand-queries",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When brand is available, prepend brand-aware query templates before generic templates.",
    )
    parser.add_argument(
        "--diagnose-youtube-key",
        action="store_true",
        help="Make one harmless YouTube search request and report sanitized auth/quota/network status.",
    )
    parser.add_argument(
        "--youtube-key-file",
        type=Path,
        default=PROJECT_ROOT / ".env.youtube_keys",
        help="Local env-style file containing YOUTUBE_API_KEY_01/YOUTUBE_API_KEY_02/etc.",
    )
    parser.add_argument(
        "--diagnose-youtube-keys",
        action="store_true",
        help="Make at most one small YouTube search request per loaded key and report label-only diagnostics.",
    )
    parser.add_argument(
        "--debug-quota",
        action="store_true",
        help="Print safe per-query quota diagnostics without exposing API key values.",
    )
    parser.add_argument(
        "--key-label",
        default=None,
        help="Optional safe label for the current shell key in diagnostic output.",
    )
    parser.add_argument("--max-keys", type=int, default=None, help="Maximum number of loaded keys to use.")
    parser.add_argument("--api-key-env", default="YOUTUBE_API_KEY")
    args = parser.parse_args()

    if not args.diagnose_youtube_keys and not os.getenv(args.api_key_env):
        try:
            from dotenv import load_dotenv

            load_dotenv(override=False)
        except ImportError:
            load_env_fallback(args.api_key_env)
    api_key = os.getenv(args.api_key_env) or "YOUR_API_KEY_HERE"
    perfume_col = args.perfume_col if str(args.perfume_col).strip() else None
    return ScraperConfig(
        target_csv=args.target_csv,
        perfume_col=perfume_col,
        candidate_queue_csv=args.candidate_queue_csv,
        raw_output_csv=args.raw_output_csv,
        status_csv=args.status_csv,
        seed_status_csv=args.seed_status_csv,
        batch_size=args.batch_size,
        limit=args.limit,
        resume=args.resume,
        retry_statuses=split_status_values(args.retry_status),
        dry_run=args.dry_run,
        max_results=args.max_results,
        max_workers=args.max_workers,
        min_words=args.min_words,
        sleep_sec=args.sleep_sec,
        max_query_templates_per_perfume=(
            args.max_query_templates_per_perfume if args.max_query_templates_per_perfume > 0 else None
        ),
        stop_after_valid_videos_per_perfume=(
            args.stop_after_valid_videos_per_perfume if args.stop_after_valid_videos_per_perfume > 0 else None
        ),
        query_templates=tuple(args.query_template) if args.query_template else QUERY_TEMPLATES,
        use_brand_queries=args.use_brand_queries,
        api_key=api_key,
        api_key_env=args.api_key_env,
        key_label=args.key_label or "key_env",
        verbose=True,
        diagnose_youtube_key=args.diagnose_youtube_key,
        diagnose_youtube_keys=args.diagnose_youtube_keys,
        youtube_key_file=args.youtube_key_file,
        max_keys=args.max_keys,
        debug_quota=args.debug_quota,
    )


def run_collection(cfg: ScraperConfig) -> None:
    if cfg.diagnose_youtube_keys:
        keys = load_youtube_key_pool(
            api_key_env=cfg.api_key_env,
            youtube_key_file=cfg.youtube_key_file,
            max_keys=cfg.max_keys,
            key_label=cfg.key_label,
        )
        raise SystemExit(diagnose_youtube_keys(keys))

    if cfg.diagnose_youtube_key:
        raise SystemExit(diagnose_youtube_key(cfg))

    candidates = (
        load_candidate_queue(cfg.candidate_queue_csv)
        if cfg.candidate_queue_csv is not None
        else load_target_candidates(cfg.target_csv, cfg.perfume_col)
    )
    to_process = select_candidates(candidates, cfg)
    print(f"Processing candidates selected: {len(to_process)}")
    print(f"Candidate source: {'candidate_queue' if cfg.candidate_queue_csv is not None else 'target_csv'}")
    print(f"Raw output CSV: {cfg.raw_output_csv}")
    print(f"Status CSV: {cfg.status_csv}")
    if cfg.retry_statuses:
        print(f"Retry statuses: {sorted(cfg.retry_statuses)}")
    print(f"Max query templates per perfume: {cfg.max_query_templates_per_perfume or 'all'}")
    print(f"Stop after valid videos per perfume: {cfg.stop_after_valid_videos_per_perfume or 'all'}")
    if cfg.debug_quota:
        print(f"Debug quota: enabled key_label={cfg.key_label} key_fingerprint={key_fingerprint(cfg.api_key)}")

    if cfg.dry_run:
        status_by_name = load_status_maps([cfg.status_csv]) if cfg.status_csv.exists() else {}
        for idx, candidate in enumerate(to_process, start=1):
            current_status = status_by_name.get(normalize_name_key(candidate.perfume), {}).get("Status", "")
            queries = build_queries(candidate, cfg)
            print(
                f"[{idx}/{len(to_process)}] "
                f"source={candidate.source} perfume={candidate.perfume} brand={candidate.brand or ''} "
                f"current_status={current_status} quality_flags={candidate.quality_flags or ''} "
                f"query_count={len(queries)} queries={queries}"
            )
        print("Dry run complete. No API calls or files were written.")
        return

    if cfg.api_key == "YOUR_API_KEY_HERE":
        raise RuntimeError("YOUTUBE_API_KEY is not configured. Use --dry-run or set the API key before collection.")

    for idx, candidate in enumerate(to_process, start=1):
        print(f"[{idx}/{len(to_process)}] Processing perfume: {candidate.perfume}")
        previous = load_status_maps([cfg.status_csv]).get(normalize_name_key(candidate.perfume), {})
        attempts = int(previous.get("Attempts", 0) or 0) + 1
        status, valid_rows, last_error = collect_for_candidate(candidate, cfg)
        written = write_raw_rows_incremental(cfg.raw_output_csv, valid_rows)
        update_status(cfg.status_csv, candidate.perfume, status, attempts, last_error)
        print(f"[{idx}/{len(to_process)}] status={status} valid={len(valid_rows)} written={written}")


def main() -> None:
    run_collection(parse_args())


if __name__ == "__main__":
    main()

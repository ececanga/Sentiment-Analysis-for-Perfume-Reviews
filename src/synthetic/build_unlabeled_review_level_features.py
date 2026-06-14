# Date Created: 2026-06-08
# Date Last Modified: 2026-06-08

"""Build review-level features from unlabeled synthetic perfume reviews.

This is the pre-labeling pipeline test for the review-level experiment:
one synthetic review in, one feature row out. Labels are intentionally absent.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pandas as pd

from src.labeling.strict_schema import EMOTIONS, VAD_DIMS
from src.utils.paths import PROJECT_ROOT, data_path

csv.field_size_limit(sys.maxsize)

NRC_EMOTIONS = [
    "anger",
    "anticipation",
    "disgust",
    "fear",
    "joy",
    "sadness",
    "surprise",
    "trust",
    "positive",
    "negative",
]

MODEL_NAMES = {
    "hartmann": "j-hartmann/emotion-english-distilroberta-base",
    "savani": "bhadresh-savani/distilbert-base-uncased-emotion",
    "goemotions": "monologg/bert-base-cased-goemotions-original",
    "cardiff": "cardiffnlp/twitter-roberta-base-emotion",
}
ALL_MODELS = tuple(MODEL_NAMES)

BASE_REVIEW_COLUMNS = [
    "synthetic_review_id",
    "perfume_id",
    "perfume_id_hash",
    "Perfume",
    "review_index",
    "source_type",
    "dataset_version",
    "generation_policy",
    "persona",
    "tone_profile",
    "style_family",
    "scenario_family",
    "length_profile",
    "sensory_source",
    "sensory_terms",
    "accord_tags",
    "Transcript Length",
    "Emotion Keyword Count",
    "Final Transcript",
]


def stringify_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def clean_whitespace(text: object) -> str:
    return " ".join(str(text or "").replace("\r", " ").replace("\n", " ").split())


def safe_col(text: object) -> str:
    value = str(text or "").strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "label"


def parse_models(values: Sequence[str]) -> List[str]:
    if not values:
        return list(ALL_MODELS)
    out: List[str] = []
    for value in values:
        for part in str(value).split(","):
            key = part.strip().lower()
            if not key:
                continue
            if key == "all":
                return list(ALL_MODELS)
            if key not in ALL_MODELS:
                raise ValueError(f"Unknown model key: {key}. Valid values: {ALL_MODELS}")
            out.append(key)
    return list(dict.fromkeys(out))


def load_reviews(path: Path, limit_reviews: int | None, limit_perfumes: int | None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Review CSV not found: {path}")
    df = pd.read_csv(path, encoding="utf-8", engine="python")
    required = {"synthetic_review_id", "perfume_id", "Perfume", "Final Transcript"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    if limit_perfumes is not None:
        keep_ids = sorted(df["perfume_id"].dropna().astype(str).unique().tolist())[:limit_perfumes]
        df = df.loc[df["perfume_id"].astype(str).isin(keep_ids)].copy()
    if limit_reviews is not None:
        df = df.head(limit_reviews).copy()

    df["synthetic_review_id"] = df["synthetic_review_id"].apply(clean_whitespace)
    df["perfume_id"] = df["perfume_id"].apply(clean_whitespace)
    df["Perfume"] = df["Perfume"].apply(clean_whitespace)
    df["Final Transcript"] = df["Final Transcript"].apply(clean_whitespace)
    if df.empty:
        raise ValueError("Review dataset is empty after limits.")
    if df["Final Transcript"].eq("").any():
        raise ValueError("Review dataset contains empty Final Transcript rows.")
    if not df["synthetic_review_id"].is_unique:
        raise ValueError("synthetic_review_id must be unique for review-level features.")
    return df.reset_index(drop=True)


def load_nrc_lexicon(path: Path) -> Dict[str, set[str]]:
    nrc = pd.read_csv(path, sep="\t", header=None, names=["word", "emotion", "association"])
    nrc = nrc[nrc["association"] == 1].drop("association", axis=1)
    return nrc.groupby("word")["emotion"].apply(set).to_dict()


def load_vad_lexicon(path: Path) -> Dict[str, Dict[str, float]]:
    vad = pd.read_csv(path, sep="\t")
    vad.columns = ["word", "valence", "arousal", "dominance"]
    return vad.set_index("word").T.to_dict()


def base_review_frame(review_df: pd.DataFrame) -> pd.DataFrame:
    out = review_df.copy()
    for col in BASE_REVIEW_COLUMNS:
        if col not in out.columns:
            out[col] = ""
    return out[BASE_REVIEW_COLUMNS].copy()


def build_lexicon_features(review_df: pd.DataFrame) -> pd.DataFrame:
    nrc_lexicon = load_nrc_lexicon(data_path("lexicons", "nrc_emotion", "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"))
    vad_lexicon = load_vad_lexicon(data_path("lexicons", "nrc_vad", "NRC-VAD-Lexicon-v2.1.txt"))

    rows: List[Dict[str, Any]] = []
    for _, row in review_df.iterrows():
        tokens = re.findall(r"\b\w+\b", str(row["Final Transcript"]).lower())
        emotion_counter: Dict[str, int] = defaultdict(int)
        vad_values: Dict[str, List[float]] = {"valence": [], "arousal": [], "dominance": []}
        for word in tokens:
            if word in nrc_lexicon:
                for emotion in nrc_lexicon[word]:
                    emotion_counter[emotion] += 1
            if word in vad_lexicon:
                for dim in VAD_DIMS:
                    vad_values[dim].append(float(vad_lexicon[word][dim]))

        out = {"synthetic_review_id": row["synthetic_review_id"]}
        out.update({f"NRC_{emotion}": emotion_counter[emotion] for emotion in NRC_EMOTIONS})
        for dim in VAD_DIMS:
            values = vad_values[dim]
            out[f"VAD_{dim}"] = sum(values) / len(values) if values else 0.0
        rows.append(out)
    return pd.DataFrame(rows)


def detect_device() -> int | str:
    import torch

    if torch.cuda.is_available():
        return 0
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return -1


def normalize_pipeline_output(output: Any) -> List[Dict[str, Any]]:
    if isinstance(output, list) and output and isinstance(output[0], list):
        return output[0]
    if isinstance(output, list) and output and isinstance(output[0], dict):
        return output
    return []


def classify_batched(pipe: Any, texts: Sequence[str], *, batch_size: int, max_chars: int) -> List[List[Dict[str, Any]]]:
    truncated = [str(text)[:max_chars] for text in texts]
    outputs = pipe(truncated, batch_size=batch_size)
    return [normalize_pipeline_output(output) for output in outputs]


def build_transformer_pipeline(model_key: str, *, local_files_only: bool) -> Any:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline, pipeline

    model_name = MODEL_NAMES[model_key]
    device = detect_device()
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, local_files_only=local_files_only)

    if model_key == "goemotions":
        return TextClassificationPipeline(
            model=model,
            tokenizer=tokenizer,
            return_all_scores=True,
            function_to_apply="sigmoid",
            device=device,
        )

    kwargs: Dict[str, Any] = {"model": model, "tokenizer": tokenizer, "device": device}
    if model_key in {"hartmann", "savani"}:
        kwargs["top_k"] = None
    elif model_key == "cardiff":
        kwargs["return_all_scores"] = True
    return pipeline("text-classification", **kwargs)


def transformer_prefix(model_key: str) -> str:
    return {
        "hartmann": "Hartmann",
        "savani": "Savani",
        "goemotions": "GoEmotions",
        "cardiff": "Cardiff",
    }[model_key]


def transformer_label_col(model_key: str) -> str:
    return {
        "hartmann": "JHartmann_Label",
        "savani": "Savani_Label",
        "goemotions": "GoEmotions_Label",
        "cardiff": "Cardiff_Label",
    }[model_key]


def score_transformer_model(
    review_df: pd.DataFrame,
    *,
    model_key: str,
    batch_size: int,
    max_chars: int,
    local_files_only: bool,
) -> pd.DataFrame:
    pipe = build_transformer_pipeline(model_key, local_files_only=local_files_only)
    outputs = classify_batched(pipe, review_df["Final Transcript"].tolist(), batch_size=batch_size, max_chars=max_chars)
    prefix = transformer_prefix(model_key)
    label_col = transformer_label_col(model_key)

    rows: List[Dict[str, Any]] = []
    for (_, source_row), scores_raw in zip(review_df.iterrows(), outputs):
        scores = {safe_col(item.get("label")): float(item.get("score", 0.0)) for item in scores_raw}
        top_label = max(scores, key=scores.get) if scores else ""
        out: Dict[str, Any] = {
            "synthetic_review_id": source_row["synthetic_review_id"],
            label_col: top_label,
        }
        out.update({f"{prefix}_{label}": score for label, score in scores.items()})
        rows.append(out)
    return pd.DataFrame(rows)


def merge_on_review_id(tables: Sequence[pd.DataFrame]) -> pd.DataFrame:
    if not tables:
        raise ValueError("No feature tables to merge.")
    merged = tables[0].copy()
    for table in tables[1:]:
        merged = merged.merge(table, on="synthetic_review_id", how="inner")
    return merged


def write_annotation_template(path: Path, feature_df: pd.DataFrame) -> None:
    columns = [
        "synthetic_review_id",
        "perfume_id",
        "Perfume",
        "Final Transcript",
        *EMOTIONS,
        *VAD_DIMS,
    ]
    out = feature_df[[col for col in columns if col in feature_df.columns]].copy()
    for col in EMOTIONS + VAD_DIMS:
        if col not in out.columns:
            out[col] = ""
    path.parent.mkdir(parents=True, exist_ok=True)
    out[columns].to_csv(path, index=False)


def write_summary(
    *,
    path: Path,
    input_csv: Path,
    output_csv: Path,
    annotation_template_csv: Path,
    review_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    models: Sequence[str],
) -> None:
    numeric_feature_cols = [
        col for col in feature_df.columns
        if col.startswith(("NRC_", "VAD_", "Hartmann_", "Savani_", "GoEmotions_", "Cardiff_"))
        and pd.api.types.is_numeric_dtype(feature_df[col])
    ]
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_csv": stringify_path(input_csv),
        "output_csv": stringify_path(output_csv),
        "annotation_template_csv": stringify_path(annotation_template_csv),
        "review_rows": int(len(feature_df)),
        "unique_perfumes": int(feature_df["perfume_id"].nunique()) if "perfume_id" in feature_df.columns else None,
        "models": list(models),
        "numeric_feature_count": len(numeric_feature_cols),
        "feature_prefix_counts": {
            prefix: sum(col.startswith(prefix) for col in numeric_feature_cols)
            for prefix in ["NRC_", "VAD_", "Hartmann_", "Savani_", "GoEmotions_", "Cardiff_"]
        },
        "persona_counts": review_df["persona"].value_counts(dropna=False).to_dict() if "persona" in review_df.columns else {},
        "tone_profile_counts": review_df["tone_profile"].value_counts(dropna=False).to_dict() if "tone_profile" in review_df.columns else {},
        "style_family_counts": review_df["style_family"].value_counts(dropna=False).to_dict() if "style_family" in review_df.columns else {},
        "scenario_family_counts": review_df["scenario_family"].value_counts(dropna=False).to_dict() if "scenario_family" in review_df.columns else {},
        "warning": "This is unlabeled review-level feature data. Train logistic regression only after adding review-level labels.",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build review-level features for unlabeled synthetic reviews.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=data_path("synthetic", "synthetic_perfume_reviews_784_unlabeled_v1.csv"),
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "synthetic_review_level_tests" / "unlabeled_v1",
    )
    parser.add_argument("--models", action="append", default=[], help="Transformer model keys. Use all, hartmann, savani, goemotions, cardiff.")
    parser.add_argument("--skip-transformers", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-chars", type=int, default=512)
    parser.add_argument("--limit-reviews", type=int, default=None)
    parser.add_argument("--limit-perfumes", type=int, default=None)
    parser.add_argument("--allow-network", action="store_true", help="Allow Hugging Face downloads if local cache is missing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = [] if args.skip_transformers else parse_models(args.models)
    out_root = args.out_root
    intermediate_dir = out_root / "intermediate"
    features_dir = out_root / "features"
    templates_dir = out_root / "annotation"
    for path in [intermediate_dir, features_dir, templates_dir]:
        path.mkdir(parents=True, exist_ok=True)

    review_df = load_reviews(args.input_csv, limit_reviews=args.limit_reviews, limit_perfumes=args.limit_perfumes)
    base_df = base_review_frame(review_df)
    lexicon_df = build_lexicon_features(review_df)
    lexicon_df.to_csv(intermediate_dir / "review_level_nrc_vad.csv", index=False)

    tables: List[pd.DataFrame] = [base_df, lexicon_df]
    for model_key in models:
        print(f"Scoring review-level synthetic reviews with {model_key}: {MODEL_NAMES[model_key]}")
        model_df = score_transformer_model(
            review_df,
            model_key=model_key,
            batch_size=args.batch_size,
            max_chars=args.max_chars,
            local_files_only=not args.allow_network,
        )
        model_df.to_csv(intermediate_dir / f"review_level_{model_key}.csv", index=False)
        tables.append(model_df)

    feature_df = merge_on_review_id(tables)
    output_csv = features_dir / "review_level_features_unlabeled.csv"
    annotation_template_csv = templates_dir / "review_level_annotation_template.csv"
    summary_json = out_root / "review_level_feature_summary.json"
    feature_df.to_csv(output_csv, index=False)
    write_annotation_template(annotation_template_csv, feature_df)
    write_summary(
        path=summary_json,
        input_csv=args.input_csv,
        output_csv=output_csv,
        annotation_template_csv=annotation_template_csv,
        review_df=review_df,
        feature_df=feature_df,
        models=models,
    )

    print(f"Wrote review-level features: {output_csv}")
    print(f"Wrote annotation template: {annotation_template_csv}")
    print(f"Wrote summary: {summary_json}")
    print(f"Review rows: {len(feature_df)} Models: {models if models else ['lexicon_only']}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# demo_pipeline_small.py
"""
Demo: run the full pipeline on 3+3=6 reviews (2 perfumes) and PRINT every stage's inputs/outputs.

This script is intentionally verbose, but bounded because it only uses 6 reviews.

It "emulates" your existing pipeline scripts:
  - src/preprocessing/format_transcripts.py
  - src/lexicon/apply_emotion_vad_lexicons.py (+ optional WordNet synonym expansion baseline)
  - src/transformers/analyze_emotions_distilroberta_jhartmann.py
  - src/transformers/analyze_emotions_distilbert_savani.py (optional)
  - src/transformers/analyze_emotions_goemotions_multi.py (optional)
  - src/transformers/analyze_emotions_roberta_cardiffnlp.py (optional)
  - src/aggregation_fusion/aggregate_and_merge_emotions.py (on the 2 perfumes subset)
  - Visual mapping: deterministic VAD -> color (HSV-like mapping)

Run examples:
  python src/pipeline/demo_pipeline_small.py
  python src/pipeline/demo_pipeline_small.py --transformers hartmann
  python src/pipeline/demo_pipeline_small.py --transformers hartmann,cardiff
  python src/pipeline/demo_pipeline_small.py --transformers none
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd

from src.utils.paths import data_path, outputs_path

# -----------------------------
# Utilities
# -----------------------------

TOKEN_RE = re.compile(r"[A-Za-z']+")  # keeps apostrophes inside words


def banner(stage_name: str, emulating_files: List[str]) -> None:
    line = "=" * 96
    print("\n" + line)
    print(f"{stage_name}")
    print("Emulating / corresponding .py:")
    for f in emulating_files:
        print(f"  - {f}")
    print(line)


def short_text(s: str, n: int = 220) -> str:
    s = "" if s is None else str(s)
    s = re.sub(r"\s+", " ", s).strip()
    return (s[:n] + "…") if len(s) > n else s


def safe_read_csv(path: Path) -> pd.DataFrame:
    """
    Robust CSV reader for very large transcript fields.

    - Prefer pandas C engine (usually faster, no python csv field-size limit).
    - If C engine fails (rare, e.g., malformed quoting), fall back to python engine
      AFTER raising csv.field_size_limit.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")

    import csv, sys

    # Raise python csv module field size limit for huge transcript cells
    max_int = sys.maxsize
    while True:
        try:
            csv.field_size_limit(max_int)
            break
        except OverflowError:
            max_int = int(max_int / 10)

    # Try fast C engine first
    try:
        return pd.read_csv(path, engine="c", low_memory=False)
    except Exception as e_c:
        # Fall back to python engine (now with increased field size limit)
        try:
            return pd.read_csv(path, engine="python", low_memory=False)
        except Exception as e_p:
            raise RuntimeError(
                f"Failed reading CSV: {path}\n"
                f"C engine error: {e_c}\n"
                f"Python engine error: {e_p}"
            )



def detect_device_for_transformers() -> int:
    """
    transformers pipeline uses:
      device=0 for CUDA
      device=-1 for CPU
    Many pipelines don't accept MPS directly as an int; safest is CPU unless CUDA.
    But on Mac you can still benefit via torch.backends.mps; some HF pipelines accept device="mps".
    We'll keep it robust: try MPS via string when available.
    """
    try:
        import torch  # noqa
        if torch.cuda.is_available():
            return 0
    except Exception:
        pass
    return -1


def detect_torch_backend() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def get_stopwords() -> set:
    # prefer sklearn's list if available (your code already uses sklearn stopwords elsewhere)
    try:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        return set(ENGLISH_STOP_WORDS)
    except Exception:
        # minimal fallback
        return {
            "the","a","an","and","or","but","if","so","to","of","in","on","for","as","at","with",
            "is","it","this","that","these","those","be","are","was","were","by","from","you","i",
            "we","they","he","she","me","my","your","our","their"
        }


# -----------------------------
# Stage 1: Preprocessing
# (Baseline: stopword removal, lowercasing, punctuation stripping)
# -----------------------------

@dataclass
class PreprocessResult:
    clean_text: str
    tokens: List[str]


def preprocess_text(text: str, stopwords: set) -> PreprocessResult:
    raw = "" if text is None else str(text)
    raw = raw.lower()
    tokens = [t.lower() for t in TOKEN_RE.findall(raw)]
    tokens = [t for t in tokens if t not in stopwords and len(t) >= 2]
    clean_text = " ".join(tokens)
    return PreprocessResult(clean_text=clean_text, tokens=tokens)


# -----------------------------
# Stage 2: NRC Lexicon + WordNet synonym expansion (baseline)
# -----------------------------

NRC_EMOTIONS_10 = [
    "anger", "anticipation", "disgust", "fear", "joy", "sadness",
    "surprise", "trust", "positive", "negative"
]


def load_nrc_lexicon(nrc_path: Path) -> Dict[str, set]:
    nrc = pd.read_csv(
        nrc_path,
        sep="\t",
        header=None,
        names=["word", "emotion", "association"],
    )
    nrc = nrc[nrc["association"] == 1].drop(columns=["association"])
    return nrc.groupby("word")["emotion"].apply(set).to_dict()


def try_wordnet_synonyms(word: str) -> List[str]:
    """
    Returns synonyms from WordNet if available.
    If NLTK or wordnet is not installed, returns empty list.
    """
    try:
        from nltk.corpus import wordnet as wn  # type: ignore
    except Exception:
        return []
    syns = set()
    for synset in wn.synsets(word):
        for lemma in synset.lemmas():
            w = lemma.name().replace("_", " ").lower()
            if w != word:
                syns.add(w)
    return sorted(syns)


def nrc_count_with_wordnet(
    tokens: List[str],
    nrc_lexicon: Dict[str, set],
    use_wordnet: bool,
    max_synonyms_per_token: int = 6
) -> Dict[str, int]:
    counts = {e: 0 for e in NRC_EMOTIONS_10}
    for t in tokens:
        if t in nrc_lexicon:
            for emo in nrc_lexicon[t]:
                if emo in counts:
                    counts[emo] += 1
            continue

        if not use_wordnet:
            continue

        syns = try_wordnet_synonyms(t)
        if syns:
            for s in syns[:max_synonyms_per_token]:
                # Only use synonym if it's directly in NRC lexicon
                if s in nrc_lexicon:
                    for emo in nrc_lexicon[s]:
                        if emo in counts:
                            counts[emo] += 1
                    break  # count at most one synonym hit per token (keeps it conservative)
    return counts


# -----------------------------
# Stage 3: VAD mapping
# -----------------------------

def load_vad_lexicon(vad_path: Path) -> Dict[str, Dict[str, float]]:
    vad = pd.read_csv(vad_path, sep="\t")
    vad.columns = ["word", "valence", "arousal", "dominance"]
    return vad.set_index("word").T.to_dict()


def vad_mean(tokens: List[str], vad_lexicon: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    vals, aros, doms = [], [], []
    for t in tokens:
        if t in vad_lexicon:
            vals.append(float(vad_lexicon[t]["valence"]))
            aros.append(float(vad_lexicon[t]["arousal"]))
            doms.append(float(vad_lexicon[t]["dominance"]))
    if not vals:
        return {"valence": 0.0, "arousal": 0.0, "dominance": 0.0, "matched_tokens": 0}
    return {
        "valence": sum(vals) / len(vals),
        "arousal": sum(aros) / len(aros),
        "dominance": sum(doms) / len(doms),
        "matched_tokens": len(vals),
    }


# -----------------------------
# Stage 4: Transformer inference (optional, local HF models)
# -----------------------------

def run_hartmann(texts: List[str], backend: str) -> pd.DataFrame:
    from transformers import pipeline
    # Your code: j-hartmann/emotion-english-distilroberta-base, all scores
    # We avoid GPU assumptions; HF generally runs fine on CPU; on Mac MPS may work.
    kwargs = dict(
        task="text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        return_all_scores=True,
        top_k=None,
    )
    if backend == "cuda":
        kwargs["device"] = 0
    # Many pipeline versions accept device="mps"; try it, fall back to CPU.
    elif backend == "mps":
        kwargs["device"] = "mps"

    try:
        clf = pipeline(**kwargs)
    except Exception:
        kwargs.pop("device", None)
        clf = pipeline(**kwargs)

    rows = []
    for i, t in enumerate(texts):
        out = clf(str(t)[:512])[0]  # list of dicts
        scores = {d["label"]: float(d["score"]) for d in out}
        top = max(scores, key=scores.get) if scores else None
        rows.append({"review_idx": i, "Transformer_Top_Emotion": top, **scores})
    return pd.DataFrame(rows)


def run_savani(texts: List[str], backend: str) -> pd.DataFrame:
    from transformers import pipeline
    kwargs = dict(
        task="text-classification",
        model="bhadresh-savani/distilbert-base-uncased-emotion",
        top_k=None,
    )
    if backend == "cuda":
        kwargs["device"] = 0
    elif backend == "mps":
        kwargs["device"] = "mps"
    try:
        clf = pipeline(**kwargs)
    except Exception:
        kwargs.pop("device", None)
        clf = pipeline(**kwargs)

    rows = []
    for i, t in enumerate(texts):
        out = clf(str(t)[:512])[0]  # list of dicts
        scores = {d["label"].lower(): float(d["score"]) for d in out}
        top = max(scores, key=scores.get) if scores else None
        rows.append({"review_idx": i, "Predicted Emotion": top, **scores})
    return pd.DataFrame(rows)


def run_cardiff(texts: List[str], backend: str) -> pd.DataFrame:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    model_name = "cardiffnlp/twitter-roberta-base-emotion"
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_name)
    kwargs = dict(task="text-classification", model=mdl, tokenizer=tok, return_all_scores=True)
    if backend == "cuda":
        kwargs["device"] = 0
    elif backend == "mps":
        kwargs["device"] = "mps"
    try:
        clf = pipeline(**kwargs)
    except Exception:
        kwargs.pop("device", None)
        clf = pipeline(**kwargs)

    rows = []
    for i, t in enumerate(texts):
        out = clf(str(t)[:512])[0]
        scores = {d["label"]: float(d["score"]) for d in out}
        top = max(scores, key=scores.get) if scores else None
        rows.append({"review_idx": i, "Predicted Emotion": top, **scores})
    return pd.DataFrame(rows)


def run_goemotions(texts: List[str], backend: str) -> pd.DataFrame:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

    model_name = "monologg/bert-base-cased-goemotions-original"
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_name)

    # Your code chooses device = 0 / mps / -1. We'll mirror that behavior safely.
    device = 0 if backend == "cuda" else ("mps" if backend == "mps" else -1)

    pipe = TextClassificationPipeline(
        model=mdl,
        tokenizer=tok,
        return_all_scores=True,
        function_to_apply="sigmoid",
        device=device,
    )

    rows = []
    for i, t in enumerate(texts):
        out = pipe(str(t)[:512])[0]
        scores = {d["label"]: float(d["score"]) for d in out}
        top = max(scores, key=scores.get) if scores else None
        rows.append({"review_idx": i, "Predicted Emotion": top, **scores})
    return pd.DataFrame(rows)


# -----------------------------
# Stage 5: Aggregation and merge (subset demo)
# -----------------------------

META_COLS = {"Perfume", "review_idx", "Predicted Emotion", "Transformer_Top_Emotion"}


def agg_numeric_means(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    numeric_cols = [c for c in df.columns if c not in META_COLS and pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        return df[["Perfume"]].drop_duplicates()
    out = df[["Perfume"] + numeric_cols].groupby("Perfume", as_index=False).mean(numeric_only=True)
    return out.rename(columns={c: f"{prefix}_{c}" for c in numeric_cols})


def agg_nrc_sum(df_nrc: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [c for c in df_nrc.columns if c in NRC_EMOTIONS_10]
    out = df_nrc[["Perfume"] + numeric_cols].groupby("Perfume", as_index=False).sum(numeric_only=True)
    return out.rename(columns={c: f"NRC_{c}" for c in numeric_cols})


def agg_vad_mean(df_vad: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = ["valence", "arousal", "dominance"]
    out = df_vad[["Perfume"] + numeric_cols].groupby("Perfume", as_index=False).mean(numeric_only=True)
    return out.rename(columns={c: f"VAD_{c}" for c in numeric_cols})


def nrc_posneg_proxy(nrc_row: pd.Series) -> float:
    pos = float(nrc_row.get("NRC_positive", 0.0))
    neg = float(nrc_row.get("NRC_negative", 0.0))
    denom = pos + neg
    return 0.0 if denom == 0 else (pos - neg) / denom  # in [-1, +1]


# -----------------------------
# Stage 6: Visual mapping (deterministic color)
# Map VAD in [0,1] approx to RGB via HSV-like mapping:
#   hue: valence (0=red, 120=green)
#   sat: arousal
#   val: dominance
# -----------------------------

def clamp01(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else x


def hsv_to_rgb(h: float, s: float, v: float) -> Tuple[int, int, int]:
    # h in [0,360), s,v in [0,1]
    import math
    h = h % 360.0
    c = v * s
    x = c * (1 - abs(((h / 60.0) % 2) - 1))
    m = v - c
    if 0 <= h < 60:
        rp, gp, bp = c, x, 0
    elif 60 <= h < 120:
        rp, gp, bp = x, c, 0
    elif 120 <= h < 180:
        rp, gp, bp = 0, c, x
    elif 180 <= h < 240:
        rp, gp, bp = 0, x, c
    elif 240 <= h < 300:
        rp, gp, bp = x, 0, c
    else:
        rp, gp, bp = c, 0, x
    r, g, b = (rp + m), (gp + m), (bp + m)
    return int(round(r * 255)), int(round(g * 255)), int(round(b * 255))


def vad_to_hex_color(valence: float, arousal: float, dominance: float) -> str:
    v = clamp01(valence)      # typically ~[0,1] in NRC-VAD
    a = clamp01(arousal)
    d = clamp01(dominance)
    hue = 120.0 * v           # 0..120
    sat = a                   # 0..1
    bri = d                   # 0..1
    r, g, b = hsv_to_rgb(hue, sat, bri)
    return f"#{r:02X}{g:02X}{b:02X}"


# -----------------------------
# Main flow
# -----------------------------

def select_two_perfumes_six_reviews(df: pd.DataFrame, n_per_perfume: int, seed: int) -> pd.DataFrame:
    if "Perfume" not in df.columns or "Final Transcript" not in df.columns:
        raise ValueError("Input CSV must contain at least columns: Perfume, Final Transcript")

    counts = df["Perfume"].value_counts()
    eligible = [p for p, c in counts.items() if c >= n_per_perfume]
    if len(eligible) < 2:
        raise RuntimeError(
            f"Need at least 2 perfumes with >= {n_per_perfume} reviews each. Eligible found: {len(eligible)}"
        )

    # deterministic pick: first two eligible in value_counts order
    perfumes = eligible[:2]
    sub = df[df["Perfume"].isin(perfumes)].copy()

    # deterministic sample within each perfume
    sub = (
        sub.groupby("Perfume", group_keys=False)
           .apply(lambda g: g.sample(n=n_per_perfume, random_state=seed))
           .reset_index(drop=True)
    )
    return sub


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--formatted_csv",
        type=Path,
        default=data_path("raw", "YouTube_Perfume_Reviews_FullyFormatted.csv"),
    )
    ap.add_argument(
        "--partial_csv",
        type=Path,
        default=data_path("raw", "YouTube_Perfume_Reviews_Partial.csv"),
    )
    ap.add_argument(
        "--nrc_path",
        type=Path,
        default=data_path("lexicons", "nrc_emotion", "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"),
    )
    ap.add_argument(
        "--vad_path",
        type=Path,
        default=data_path("lexicons", "nrc_vad", "NRC-VAD-Lexicon-v2.1.txt"),
    )
    ap.add_argument("--n_per_perfume", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_wordnet", action="store_true", help="Disable WordNet synonym expansion baseline.")
    ap.add_argument(
        "--transformers",
        default="hartmann",
        help="Comma list: none|hartmann|savani|goemotions|cardiff (example: hartmann,cardiff)",
    )
    ap.add_argument(
        "--save_demo_outputs",
        action="store_true",
        help="Write small CSVs into outputs/demo_outputs/.",
    )
    args = ap.parse_args()

    formatted_path = args.formatted_csv
    partial_path = args.partial_csv
    demo_out_dir = outputs_path("demo_outputs")

    # -----------------------------
    # Stage 0: Load + select demo subset
    # -----------------------------
    banner(
        "PIPELINE STAGE 0: Load data and select 3+3 reviews (2 perfumes) [DEMO SUBSET]",
        ["(selection helper inside demo_pipeline_small.py)"]
    )

    if formatted_path.exists():
        df_all = safe_read_csv(formatted_path)
        source_used = str(formatted_path)
    else:
        # If formatted doesn't exist, load partial and minimally emulate format_transcripts flattening
        # (but we still proceed; you should ideally run your format_transcripts.py first)
        df_all = safe_read_csv(partial_path)
        source_used = str(partial_path)

        # Minimal whitespace normalization (matches format_transcripts spirit)
        def fix_transcript_format(text: str) -> str:
            text = "" if text is None else str(text)
            text = re.sub(r"\s{3,}", " ", text)
            text = text.replace("\n", " ")
            text = re.sub(r"\s{2,}", " ", text)
            return text.strip()

        if "Final Transcript" in df_all.columns:
            df_all["Final Transcript"] = df_all["Final Transcript"].apply(fix_transcript_format)
        if "Transcript" in df_all.columns:
            df_all["Transcript"] = df_all["Transcript"].apply(fix_transcript_format)

    print(f"Data source used: {source_used}")
    print(f"Total rows loaded: {len(df_all):,}")
    print(f"Columns: {df_all.columns.tolist()}")

    df_demo = select_two_perfumes_six_reviews(df_all, n_per_perfume=args.n_per_perfume, seed=args.seed)

    # Keep a stable "review_idx" within the demo subset for merges
    df_demo = df_demo.reset_index(drop=True).copy()
    df_demo["review_idx"] = df_demo.index

    print("\nSelected perfumes:")
    print(df_demo["Perfume"].value_counts().to_string())

    show_cols = [c for c in ["Perfume", "Video ID", "Title", "URL", "Final Transcript", "review_idx"] if c in df_demo.columns]
    tmp = df_demo[show_cols].copy()
    if "Final Transcript" in tmp.columns:
        tmp["Final Transcript"] = tmp["Final Transcript"].apply(lambda s: short_text(s, 200))
    print("\nDEMO INPUT ROWS (truncated transcripts):")
    print(tmp.to_string(index=False))

    if args.save_demo_outputs:
        demo_out_dir.mkdir(parents=True, exist_ok=True)
        df_demo.to_csv(demo_out_dir / "stage0_demo_subset.csv", index=False)

    # -----------------------------
    # Stage 1: Preprocessing (baseline cleaning)
    # -----------------------------
    banner(
        "PIPELINE STAGE 1: Preprocessing (lowercase + punctuation stripping + stopword removal)",
        ["py files/Preprocessing_Formatting/format_transcripts.py (plus baseline token cleaning in thesis pipeline)"]
    )

    stopwords = get_stopwords()
    pre_rows = []
    for _, r in df_demo.iterrows():
        pr = preprocess_text(r["Final Transcript"], stopwords=stopwords)
        pre_rows.append({
            "review_idx": int(r["review_idx"]),
            "Perfume": r["Perfume"],
            "raw_excerpt": short_text(r["Final Transcript"], 160),
            "clean_excerpt": short_text(pr.clean_text, 160),
            "token_count": len(pr.tokens),
            "tokens_preview": " ".join(pr.tokens[:25]),
        })
    df_pre = pd.DataFrame(pre_rows)
    print(df_pre.to_string(index=False))

    # store tokens for later stages
    tokens_by_review = {
        int(r["review_idx"]): preprocess_text(r["Final Transcript"], stopwords=stopwords).tokens
        for _, r in df_demo.iterrows()
    }

    if args.save_demo_outputs:
        df_pre.to_csv(demo_out_dir / "stage1_preprocessing.csv", index=False)

    # -----------------------------
    # Stage 2: NRC Lexicon + WordNet expansion
    # -----------------------------
    banner(
        "PIPELINE STAGE 2: Lexicon-based emotion extraction (NRC) + WordNet synonym expansion (baseline)",
        ["py files/Lexicon_Based_Emotion/apply_emotion_vad_lexicons.py (extended here with optional WordNet expansion)"]
    )

    nrc_lex = load_nrc_lexicon(args.nrc_path)
    use_wordnet = (not args.no_wordnet)

    # Quick capability hint (doesn't stop execution)
    if use_wordnet:
        try:
            _ = try_wordnet_synonyms("good")
            print("WordNet: enabled (if you get LookupError, run: python -c \"import nltk; nltk.download('wordnet')\" )")
        except Exception as e:
            print(f"WordNet: attempted but may fail at runtime: {e}")

    nrc_rows = []
    for _, r in df_demo.iterrows():
        ridx = int(r["review_idx"])
        toks = tokens_by_review[ridx]
        counts = nrc_count_with_wordnet(toks, nrc_lexicon=nrc_lex, use_wordnet=use_wordnet)
        top3 = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:3]
        nrc_rows.append({
            "review_idx": ridx,
            "Perfume": r["Perfume"],
            **counts,
            "top3_nrc": ", ".join([f"{k}:{v}" for k, v in top3]),
        })

    df_nrc = pd.DataFrame(nrc_rows)
    print("\nPer-review NRC counts:")
    print(df_nrc.to_string(index=False))

    df_nrc_perf = df_nrc.groupby("Perfume", as_index=False)[NRC_EMOTIONS_10].sum(numeric_only=True)
    df_nrc_perf["top3_nrc_perfume"] = df_nrc_perf[NRC_EMOTIONS_10].apply(
        lambda s: ", ".join([f"{k}:{int(v)}" for k, v in s.sort_values(ascending=False).head(3).items()]),
        axis=1
    )
    print("\nPer-perfume NRC sums:")
    print(df_nrc_perf.to_string(index=False))

    if args.save_demo_outputs:
        df_nrc.to_csv(demo_out_dir / "stage2_nrc_per_review.csv", index=False)
        df_nrc_perf.to_csv(demo_out_dir / "stage2_nrc_per_perfume.csv", index=False)

    # -----------------------------
    # Stage 3: VAD mapping
    # -----------------------------
    banner(
        "PIPELINE STAGE 3: VAD mapping (mean valence/arousal/dominance over matched tokens)",
        ["py files/Lexicon_Based_Emotion/apply_emotion_vad_lexicons.py"]
    )

    vad_lex = load_vad_lexicon(args.vad_path)
    vad_rows = []
    for _, r in df_demo.iterrows():
        ridx = int(r["review_idx"])
        toks = tokens_by_review[ridx]
        m = vad_mean(toks, vad_lexicon=vad_lex)
        vad_rows.append({
            "review_idx": ridx,
            "Perfume": r["Perfume"],
            "valence": round(float(m["valence"]), 4),
            "arousal": round(float(m["arousal"]), 4),
            "dominance": round(float(m["dominance"]), 4),
            "matched_tokens": int(m["matched_tokens"]),
        })

    df_vad = pd.DataFrame(vad_rows)
    print("\nPer-review VAD means:")
    print(df_vad.to_string(index=False))

    df_vad_perf = df_vad.groupby("Perfume", as_index=False)[["valence","arousal","dominance"]].mean(numeric_only=True)
    df_vad_perf[["valence","arousal","dominance"]] = df_vad_perf[["valence","arousal","dominance"]].round(4)
    print("\nPer-perfume VAD means:")
    print(df_vad_perf.to_string(index=False))

    if args.save_demo_outputs:
        df_vad.to_csv(demo_out_dir / "stage3_vad_per_review.csv", index=False)
        df_vad_perf.to_csv(demo_out_dir / "stage3_vad_per_perfume.csv", index=False)

    # -----------------------------
    # Stage 4: Transformer inference (optional)
    # -----------------------------
    banner(
        "PIPELINE STAGE 4: Transformer-based emotion classification (optional models, local Hugging Face)",
        [
            "py files/Transformer_Based_Emotion_Classification/analyze_emotions_distilroberta_jhartmann.py",
            "py files/Transformer_Based_Emotion_Classification/analyze_emotions_distilbert_savani.py (optional)",
            "py files/Transformer_Based_Emotion_Classification/analyze_emotions_goemotions_multi.py (optional)",
            "py files/Transformer_Based_Emotion_Classification/analyze_emotions_roberta_cardiffnlp.py (optional)",
        ]
    )

    tf_spec = [s.strip().lower() for s in args.transformers.split(",") if s.strip()]
    if tf_spec == ["none"]:
        tf_spec = []

    backend = detect_torch_backend()
    print(f"Detected torch backend: {backend}")

    texts = df_demo["Final Transcript"].astype(str).tolist()
    base_meta = df_demo[["review_idx", "Perfume"]].copy()

    tf_outputs: Dict[str, pd.DataFrame] = {}

    if "hartmann" in tf_spec:
        df_h = run_hartmann(texts, backend=backend)
        df_h = base_meta.merge(df_h, on="review_idx", how="left")
        tf_outputs["Hartmann"] = df_h
        # Print top-3 scores per review
        score_cols = [c for c in df_h.columns if c not in {"review_idx","Perfume","Transformer_Top_Emotion"} and pd.api.types.is_numeric_dtype(df_h[c])]
        rows = []
        for _, rr in df_h.iterrows():
            scores = rr[score_cols].sort_values(ascending=False).head(3)
            rows.append({
                "review_idx": int(rr["review_idx"]),
                "Perfume": rr["Perfume"],
                "top3": ", ".join([f"{k}:{float(v):.4f}" for k, v in scores.items()]),
                "top1_label": rr["Transformer_Top_Emotion"],
            })
        print("\n[Hartmann] per-review top-3:")
        print(pd.DataFrame(rows).to_string(index=False))

    if "savani" in tf_spec:
        df_s = run_savani(texts, backend=backend)
        df_s = base_meta.merge(df_s, on="review_idx", how="left")
        tf_outputs["Savani"] = df_s
        score_cols = [c for c in df_s.columns if c not in {"review_idx","Perfume","Predicted Emotion"} and pd.api.types.is_numeric_dtype(df_s[c])]
        rows = []
        for _, rr in df_s.iterrows():
            scores = rr[score_cols].sort_values(ascending=False).head(3)
            rows.append({
                "review_idx": int(rr["review_idx"]),
                "Perfume": rr["Perfume"],
                "top3": ", ".join([f"{k}:{float(v):.4f}" for k, v in scores.items()]),
                "top1_label": rr["Predicted Emotion"],
            })
        print("\n[Savani] per-review top-3:")
        print(pd.DataFrame(rows).to_string(index=False))

    if "cardiff" in tf_spec:
        df_c = run_cardiff(texts, backend=backend)
        df_c = base_meta.merge(df_c, on="review_idx", how="left")
        tf_outputs["Cardiff"] = df_c
        score_cols = [c for c in df_c.columns if c not in {"review_idx","Perfume","Predicted Emotion"} and pd.api.types.is_numeric_dtype(df_c[c])]
        rows = []
        for _, rr in df_c.iterrows():
            scores = rr[score_cols].sort_values(ascending=False).head(3)
            rows.append({
                "review_idx": int(rr["review_idx"]),
                "Perfume": rr["Perfume"],
                "top3": ", ".join([f"{k}:{float(v):.4f}" for k, v in scores.items()]),
                "top1_label": rr["Predicted Emotion"],
            })
        print("\n[Cardiff] per-review top-3:")
        print(pd.DataFrame(rows).to_string(index=False))

    if "goemotions" in tf_spec:
        df_g = run_goemotions(texts, backend=backend)
        df_g = base_meta.merge(df_g, on="review_idx", how="left")
        tf_outputs["GoEmotions"] = df_g
        # GoEmotions has many labels: print top-5 per review
        score_cols = [c for c in df_g.columns if c not in {"review_idx","Perfume","Predicted Emotion"} and pd.api.types.is_numeric_dtype(df_g[c])]
        rows = []
        for _, rr in df_g.iterrows():
            scores = rr[score_cols].sort_values(ascending=False).head(5)
            rows.append({
                "review_idx": int(rr["review_idx"]),
                "Perfume": rr["Perfume"],
                "top5": ", ".join([f"{k}:{float(v):.4f}" for k, v in scores.items()]),
                "top1_label": rr["Predicted Emotion"],
            })
        print("\n[GoEmotions] per-review top-5:")
        print(pd.DataFrame(rows).to_string(index=False))

    if args.save_demo_outputs and tf_outputs:
        demo_out_dir.mkdir(parents=True, exist_ok=True)
        for name, dfx in tf_outputs.items():
            dfx.to_csv(demo_out_dir / f"stage4_transformer_{name.lower()}_per_review.csv", index=False)

    # -----------------------------
    # Stage 5: Aggregation + fusion (subset)
    # -----------------------------
    banner(
        "PIPELINE STAGE 5: Aggregate per perfume + merge NRC/VAD/Transformer (subset demo)",
        ["py files/Aggregation_Fusion/aggregate_and_merge_emotions.py"]
    )

    # Aggregate NRC + VAD
    nrc_agg = agg_nrc_sum(df_nrc)
    vad_agg = agg_vad_mean(df_vad)

    merged = nrc_agg.merge(vad_agg, on="Perfume", how="inner")

    # Aggregate each transformer output (mean of numeric score columns)
    for name, dfx in tf_outputs.items():
        merged = merged.merge(agg_numeric_means(dfx, prefix=name), on="Perfume", how="left")

    # Add a tiny "fusion" proxy (kept explicit and interpretable)
    merged["NRC_posneg_proxy"] = merged.apply(nrc_posneg_proxy, axis=1).round(4)
    merged["Unified_valence_proxy"] = (0.5 * merged["VAD_valence"] + 0.5 * (merged["NRC_posneg_proxy"] + 1) / 2).round(4)

    print("\nUNIFIED per-perfume profile (demo subset):")
    # widen display a bit
    with pd.option_context("display.max_columns", 200, "display.width", 160):
        print(merged.to_string(index=False))

    if args.save_demo_outputs:
        merged.to_csv(demo_out_dir / "stage5_unified_profile_demo.csv", index=False)

    # -----------------------------
    # Stage 6: Visual mapping (color)
    # -----------------------------
    banner(
        "PIPELINE STAGE 6: Visual mapping (deterministic VAD -> hex color per perfume)",
        ["(visual mapping helper inside demo_pipeline_small.py)"]
    )

    colors = []
    for _, r in merged.iterrows():
        hex_color = vad_to_hex_color(float(r["VAD_valence"]), float(r["VAD_arousal"]), float(r["VAD_dominance"]))
        colors.append({"Perfume": r["Perfume"], "VAD_valence": r["VAD_valence"], "VAD_arousal": r["VAD_arousal"], "VAD_dominance": r["VAD_dominance"], "hex_color": hex_color})
    df_colors = pd.DataFrame(colors)
    print(df_colors.to_string(index=False))

    if args.save_demo_outputs:
        df_colors.to_csv(demo_out_dir / "stage6_vad_colors.csv", index=False)

    print("\n✅ DEMO COMPLETED: ran all stages on 6 reviews (2 perfumes).")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        raise

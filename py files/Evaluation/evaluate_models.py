#!/usr/bin/env python3
"""
Evaluate and compare emotion pipelines on perfume YouTube reviews.

Inputs (expected in ./csv files/):
  - Unified_Emotion_Profile_Perfume.csv         (from aggregate_and_merge_emotions.py)
  - YouTube_Perfume_Reviews_FullyFormatted.csv  (optional, for counts)
  - YouTube_Perfume_Reviews_With_Savani.csv     (optional, for counts)
  - YouTube_Perfume_Reviews_With_Transformer.csv (optional, Hartmann; for counts)
  - YouTube_Perfume_Reviews_With_GoEmotions.csv (optional, for counts)
  - YouTube_Perfume_Reviews_With_CardiffRoberta.csv (optional, for counts)

Produces (in ./evaluation_out/):
  - pairwise_kappa.csv                        (Cohen's κ between models, top labels per perfume)
  - fleiss_kappa.txt                          (Fleiss' κ across models)
  - agreement_by_perfume.csv                  (% agreement & entropy per perfume)
  - label_distribution_by_model.csv           (coarse-label distribution per model)
  - jsd_pairwise.csv                          (pairwise JSD on coarse distributions)
  - vad_correlations.csv                      (VAD vs model-derived valence proxies)
  - score_vector_correlations.csv             (Pearson r on common score vectors across models)
  - jsd_common_score_vectors.csv              (JSD on common (non-coarse) score vectors across models)
  - vad_color_palette_hsl.csv                 (VAD→HSL palette per perfume)
  - Figures/*.png: κ heatmap, JSD heatmap, VAD scatters (bubble), NRC heatmap, stacked bars, color swatches
  - top_disagreements_topK.csv                (lowest agreement / highest entropy perfumes)

How it measures:
  - Agreement on TOP LABELS: uses per-perfume top labels saved in Unified_Emotion_Profile_Perfume.csv
    (columns: JHartmann_Label, GoEmotions_Label, Cardiff_Label, Savani_Label).
  - Distribution similarity: projects model-specific score columns into a shared 6-class space
    {joy, sadness, anger, fear, surprise, neutral} then compares (JSD).
  - VAD alignment: correlates VAD_valence with model-derived valence proxies (joy − anger/fear/sadness).

CLI:
  python "py files/Evaluation/evaluate_models.py" --topN 20 --topK 25 --normalize_vad
"""

from __future__ import annotations

import argparse
import colorsys
import numpy as np
import pandas as pd
import colorsys
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple
from scipy.spatial.distance import jensenshannon
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score

# ---------------------------- Paths & IO ----------------------------
ROOT = Path('.').resolve()
CSV_DIR = ROOT / 'csv files'
OUT_DIR = ROOT / 'evaluation_out'
FIG_DIR = OUT_DIR / 'Figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

UNIFIED_PATH = CSV_DIR / 'Unified_Emotion_Profile_Perfume.csv'
FF_PATH      = CSV_DIR / 'YouTube_Perfume_Reviews_FullyFormatted.csv'  # optional for counts

# ---------------------------- Coarse label space ----------------------------
COARSE = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'neutral']

# Map many fine labels → coarse 6 categories
LABEL_MAP = {
    # JOY / POSITIVE AFFECT
    'joy':'joy','love':'joy','optimism':'joy','admiration':'joy','approval':'joy','amusement':'joy',
    'excitement':'joy','enthusiasm':'joy','gratitude':'joy','relief':'joy','pride':'joy','desire':'joy',
    'affection':'joy','contentment':'joy','happiness':'joy','pleasure':'joy','satisfaction':'joy',
    'trust':'joy','positive':'joy','like':'joy','curiosity':'joy',
    # SADNESS
    'sadness':'sadness','disappointment':'sadness','grief':'sadness','remorse':'sadness','loneliness':'sadness',
    'regret':'sadness','homesickness':'sadness','sorrow':'sadness','depression':'sadness','melancholy':'sadness',
    # ANGER / DISGUST / CONTEMPT
    'anger':'anger','annoyance':'anger','disapproval':'anger','disgust':'anger','contempt':'anger',
    'frustration':'anger','irritation':'anger','hatred':'anger','resentment':'anger','outrage':'anger',
    # FEAR / ANXIETY
    'fear':'fear','nervousness':'fear','anxiety':'fear','worry':'fear','apprehension':'fear','panic':'fear',
    'dread':'fear','insecurity':'fear','embarrassment':'fear','shame':'fear',
    # SURPRISE / REALIZATION / CONFUSION
    'surprise':'surprise','astonishment':'surprise','realization':'surprise','confusion':'surprise',
    'amazement':'surprise','shock':'surprise',
    # NEUTRAL / OTHER
    'neutral':'neutral','no emotion':'neutral','other':'neutral','others':'neutral','unknown':'neutral'
}

# For plotting readability
MODEL_ORDER = ['JHartmann', 'Savani', 'Cardiff', 'GoEmotions']

# Canonical name → accepted column prefixes
ALIAS_PREFIXES = {
    'JHartmann': ['JHartmann', 'Hartmann'],   # <-- this is the important one
    'Savani':    ['Savani'],
    'Cardiff':   ['Cardiff', 'CardiffRoberta'],
    'GoEmotions':['GoEmotions']
}

# ---------------------------- Utilities ----------------------------

def coarsen(label: str | float | None) -> str:
    """Map any fine label to the COARSE set; ignore NaN by handling it upstream."""
    if not isinstance(label, str) or not label:
        return 'neutral'
    key = label.strip().lower()
    return LABEL_MAP.get(key, LABEL_MAP.get(key.replace('-', ' '), 'neutral'))

def norm_prob(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    v[v < 0] = 0
    s = v.sum()
    if s <= 1e-9:
        return np.ones_like(v) / len(v) if len(v) > 0 else np.array([])
    return v / s

def safe_savefig(path: Path):
    try:
        plt.tight_layout()
    except Exception:
        pass
    plt.savefig(path, dpi=180, bbox_inches='tight')
    plt.close()

def get_review_counts(unified: pd.DataFrame) -> pd.Series:
    """Infer review counts per perfume; align to unified perfumes and fill at least 1."""
    review_files = [
        CSV_DIR / 'YouTube_Perfume_Reviews_With_Savani.csv',
        CSV_DIR / 'YouTube_Perfume_Reviews_With_Transformer.csv',
        CSV_DIR / 'YouTube_Perfume_Reviews_With_GoEmotions.csv',
        CSV_DIR / 'YouTube_Perfume_Reviews_With_CardiffRoberta.csv',
        FF_PATH
    ]
    counts = None
    for p in review_files:
        if p.exists():
            try:
                tmp = pd.read_csv(p)
                if 'Perfume' in tmp.columns:
                    counts = tmp['Perfume'].value_counts()
                    break
            except Exception as e:
                print(f"Warning: Could not read counts from {p}: {e}")
    if counts is None:
        counts = pd.Series(dtype=int)
    counts = counts.reindex(unified['Perfume']).fillna(1).astype(int)
    counts.name = "ReviewCount"
    return counts

# ---------------------------- Agreement on top labels ----------------------------

def compute_pairwise_kappa(unified: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Cohen's κ over coarsened labels, IGNORING missing per-model labels per perfume."""
    present = [m for m in MODEL_ORDER if f"{m}_Label" in unified.columns]
    if not present:
        return pd.DataFrame()

    kappas = pd.DataFrame(index=present, columns=present, dtype=float)
    for a in present:
        Acol = f"{a}_Label"
        Avals = unified[Acol]
        for b in present:
            if a == b:
                kappas.loc[a, b] = 1.0
                continue
            Bcol = f"{b}_Label"
            if Bcol not in unified.columns:
                kappas.loc[a, b] = np.nan
                continue
            mask = Avals.notna() & unified[Bcol].notna()
            if not mask.any():
                kappas.loc[a, b] = np.nan
                continue
            y1 = Avals[mask].map(coarsen)
            y2 = unified[Bcol][mask].map(coarsen)
            kappas.loc[a, b] = cohen_kappa_score(y1, y2, labels=COARSE)
    return kappas

def fleiss_kappa_from_unified(unified: pd.DataFrame) -> float:
    """Fleiss' κ across all present models, ignoring missing per-perfume labels."""
    items, n_raters_per_item = [], []
    for _, row in unified.iterrows():
        counts = {c: 0 for c in COARSE}
        n_raters = 0
        for m in MODEL_ORDER:
            col = f'{m}_Label'
            if col in unified.columns and pd.notna(row[col]):
                lab = coarsen(row[col])
                counts[lab] += 1
                n_raters += 1
        if n_raters >= 2:
            items.append(list(counts.values()))
            n_raters_per_item.append(n_raters)

    if not items:
        return float('nan')

    M = np.array(items, dtype=float)
    n_vec = np.array(n_raters_per_item, dtype=float)

    p_j = M.sum(axis=0) / n_vec.sum()
    P_i = ((M**2).sum(axis=1) - n_vec) / (n_vec * (n_vec - 1))
    P_bar = np.nanmean(P_i)
    P_e = (p_j**2).sum()
    return (P_bar - P_e) / (1 - P_e) if (1 - P_e) > 0 else float('nan')

def agreement_by_perfume(unified: pd.DataFrame) -> pd.DataFrame:
    """Per-perfume agreement (majority fraction) & entropy, ignoring missing labels."""
    rows = []
    for _, r in unified.iterrows():
        labels = []
        for m in MODEL_ORDER:
            col = f"{m}_Label"
            if col in unified.columns and pd.notna(r[col]):
                labels.append(coarsen(r[col]))
        if not labels:
            rows.append({"Perfume": r["Perfume"], "AgreementRate": np.nan,
                         "NumModels": 0, "EntropyBits": np.nan, "MajorityLabel": None})
            continue
        counts = pd.Series(labels).value_counts()
        total = counts.sum()
        agreement = counts.max() / total
        probs = counts / total
        H = -(probs * np.log2(probs + 1e-12)).sum()
        rows.append({"Perfume": r["Perfume"], "AgreementRate": agreement,
                     "NumModels": int(total), "EntropyBits": float(H),
                     "MajorityLabel": counts.idxmax()})
    return pd.DataFrame(rows)

# ---------------------------- Distribution projection ----------------------------

PROJECTORS: Dict[str, Dict[str, str]] = {
    'JHartmann': {'joy':'joy','love':'joy','sadness':'sadness','anger':'anger','fear':'fear','surprise':'surprise'},
    'Savani':    {'joy':'joy','love':'joy','sadness':'sadness','anger':'anger','fear':'fear','surprise':'surprise'},
    'Cardiff':   {'joy':'joy','optimism':'joy','sadness':'sadness','anger':'anger'},
    'GoEmotions': {
        'admiration':'joy','approval':'joy','gratitude':'joy','joy':'joy','love':'joy','optimism':'joy',
        'amusement':'joy','excitement':'joy','relief':'joy','pride':'joy','desire':'joy','caring':'joy',
        'sadness':'sadness','disappointment':'sadness','grief':'sadness','remorse':'sadness',
        'anger':'anger','annoyance':'anger','disgust':'anger','disapproval':'anger','contempt':'anger',
        'fear':'fear','nervousness':'fear','anxiety':'fear','worry':'fear',
        'surprise':'surprise','realization':'surprise','confusion':'surprise',
        'neutral':'neutral','other':'neutral','others':'neutral'
    }
}

def project_to_coarse(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Return DataFrame with columns: Perfume, <prefix>_[joy|sadness|anger|fear|surprise|neutral] (row-normalized).
       Accepts alias prefixes (e.g., 'Hartmann_' treated as 'JHartmann_')."""
    alias_list = ALIAS_PREFIXES.get(prefix, [prefix])

    # Collect all candidate score columns for any accepted prefix
    cols = []
    for pfx in alias_list:
        cols += [c for c in df.columns if c.lower().startswith(pfx.lower() + '_')]

    if not cols:
        return pd.DataFrame({'Perfume': df['Perfume']})

    proj = pd.DataFrame({'Perfume': df['Perfume']})
    mapping = PROJECTORS.get(prefix, {})
    bucket = {k: np.zeros(len(df)) for k in COARSE}

    for c in cols:
        # Strip the FIRST underscore only, regardless of actual prefix used
        base = c.split('_', 1)[1]
        coarse_target = mapping.get(base.lower())
        if coarse_target in COARSE:
            bucket[coarse_target] += pd.to_numeric(df[c], errors='coerce').fillna(0).to_numpy()

    for k in COARSE:
        proj[f'{prefix}_{k}'] = bucket[k]

    vals = proj[[f'{prefix}_{k}' for k in COARSE]].to_numpy()
    proj[[f'{prefix}_{k}' for k in COARSE]] = np.apply_along_axis(norm_prob, 1, vals)
    return proj

def build_coarse_cube(unified: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    return {m: project_to_coarse(unified, m) for m in MODEL_ORDER}

def hue_from_valence_full(valence, offset=0.02):
    # rank in [0,1], guarantees uniform coverage across the wheel
    r = pd.Series(valence).rank(pct=True).to_numpy(dtype=float)
    return (r + offset) % 1.0

def pct_rescale(x, lo=1, hi=99):
    x = np.asarray(x, dtype=float)
    a, b = np.nanpercentile(x, lo), np.nanpercentile(x, hi)
    if b <= a:  # fallback
        mn, mx = np.nanmin(x), np.nanmax(x)
        return (x - mn) / (mx - mn + 1e-9)
    return (x - a) / (b - a)

def vad_to_hsv_full(valence, arousal, dominance, hue_offset=0.0):
    # spread hue around the whole wheel using valence ranks
    r = pd.Series(valence).rank(pct=True).to_numpy(dtype=float)
    H = (r + hue_offset) % 1.0

    # much brighter & more saturated
    S = 0.80 + 0.20 * pct_rescale(arousal, 5, 95)   # was 0.25 + 0.75 * ...
    V = 0.92 + 0.08 * pct_rescale(dominance, 5, 95) # was 0.55 + 0.40 * ...

    return H, np.clip(S, 0, 1), np.clip(V, 0, 1)

# ---------------------------- Dist. similarity & label distribution ----------------------------

def pairwise_jsd_coarse(cube: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    models = [m for m in MODEL_ORDER if m in cube and f'{m}_joy' in cube[m].columns]
    if not models:
        return pd.DataFrame()

    pairs = pd.DataFrame(index=models, columns=models, dtype=float)
    for a in models:
        for b in models:
            if a == b:
                pairs.loc[a, b] = 0.0
                continue
            A = cube[a][[f'{a}_{k}' for k in COARSE]].to_numpy()
            B = cube[b][[f'{b}_{k}' for k in COARSE]].to_numpy()
            jsds = [jensenshannon(A[i], B[i])**2 for i in range(len(A))]
            pairs.loc[a, b] = float(np.nanmean(jsds))
    return pairs

def label_distribution_by_model(unified: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m in MODEL_ORDER:
        col = f'{m}_Label'
        if col not in unified.columns:
            continue
        counts = unified[col].map(coarsen).value_counts()
        for lab in COARSE:
            rows.append({'Model': m, 'Label': lab, 'Count': int(counts.get(lab, 0))})
    return pd.DataFrame(rows)

# ---------------------------- VAD correlations ----------------------------
def vad_correlations(unified: pd.DataFrame, cube: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    out = []
    if 'VAD_valence' not in unified.columns:
        return pd.DataFrame()

    x = pd.to_numeric(unified['VAD_valence'], errors='coerce')
    for m in MODEL_ORDER:
        if m not in cube or f'{m}_joy' not in cube[m].columns:
            continue
        y = derive_valence_proxy(cube[m], m)
        df = pd.DataFrame({'x': x, 'y': pd.to_numeric(y, errors='coerce')}).dropna()

        # Guard: skip if either vector is ~constant (avoids ConstantInputWarning)
        if len(df) < 3 or df['x'].nunique() < 2 or df['y'].nunique() < 2:
            out.append({'Model': m, 'Pearson_r': np.nan, 'Pearson_p': np.nan,
                        'Spearman_r': np.nan, 'Spearman_p': np.nan})
            continue

        pr, pr_p = pearsonr(df['x'], df['y'])
        sr, sr_p = spearmanr(df['x'], df['y'])
        out.append({'Model': m, 'Pearson_r': pr, 'Pearson_p': pr_p,
                    'Spearman_r': sr, 'Spearman_p': sr_p})
    return pd.DataFrame(out)

# ---------------------------- Extra: common score vectors (non-coarse) ----------------------------

def collect_model_matrix(unified: pd.DataFrame, prefix: str) -> pd.DataFrame | None:
    cols = [c for c in unified.columns if c.startswith(prefix + "_")]
    if not cols:
        return None
    num = []
    for c in cols:
        if pd.api.types.is_numeric_dtype(unified[c]):
            num.append(c)
        else:
            coerced = pd.to_numeric(unified[c], errors='coerce')
            if coerced.notna().any():
                unified[c] = coerced
                num.append(c)
    if not num:
        return None
    return unified[['Perfume'] + num].set_index('Perfume').fillna(0.0)

def analyze_common_score_vectors(unified: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    M = {m: collect_model_matrix(unified, m) for m in MODEL_ORDER}
    models = [k for k, v in M.items() if v is not None]
    corr_rows, js_rows = [], []

    for i, a in enumerate(models):
        for b in models[i+1:]:
            A, B = M[a], M[b]
            A_suf = {c.split(f"{a}_", 1)[1] for c in A.columns}
            B_suf = {c.split(f"{b}_", 1)[1] for c in B.columns}
            common = sorted(A_suf & B_suf)
            if not common:
                continue

            A_cols, B_cols = [f"{a}_{s}" for s in common], [f"{b}_{s}" for s in common]
            merged = A[A_cols].join(B[B_cols], how='inner', lsuffix='_A', rsuffix='_B')
            if len(merged) < 2:
                continue

            vA = merged[[f"{a}_{s}" for s in common]].values.flatten()
            vB = merged[[f"{b}_{s}" for s in common]].values.flatten()
            if np.std(vA) > 1e-9 and np.std(vB) > 1e-9:
                r = np.corrcoef(vA, vB)[0, 1]
                corr_rows.append({"model_a": a, "model_b": b, "pearson_r": float(r), "emotions_compared": ",".join(common)})

            js_vals = []
            for idx in merged.index:
                p = merged.loc[idx, [f"{a}_{s}" for s in common]].to_numpy()
                q = merged.loc[idx, [f"{b}_{s}" for s in common]].to_numpy()
                js_vals.append(jensenshannon(norm_prob(p), norm_prob(q))**2)
            js_rows.append({
                "model_a": a, "model_b": b,
                "mean_JS": float(np.mean(js_vals)),
                "median_JS": float(np.median(js_vals)),
                "n_perfumes": int(len(js_vals)),
                "emotions": ",".join(common)
            })

    return pd.DataFrame(corr_rows), pd.DataFrame(js_rows)

# ---------------------------- Plotting ----------------------------

def plot_kappa_heatmap(kappa: pd.DataFrame):
    if kappa.empty:
        return
    plt.figure(figsize=(6, 5))
    im = plt.imshow(kappa.values.astype(float), cmap='viridis', vmin=0, vmax=1)
    plt.xticks(range(len(kappa.columns)), kappa.columns, rotation=45, ha='right')
    plt.yticks(range(len(kappa.index)), kappa.index)
    for i, j in np.ndindex(kappa.shape):
        val = kappa.iloc[i, j]
        plt.text(j, i, f"{val:.2f}", ha='center', va='center',
                 color='white' if val < 0.6 else 'black')
    plt.title("Cohen's κ (Coarsened Top Labels)")
    plt.colorbar(im, label="Kappa Score")
    safe_savefig(FIG_DIR / 'kappa_heatmap.png')

def plot_label_distribution(dist_df: pd.DataFrame):
    if dist_df.empty:
        return
    pivot = dist_df.pivot(index='Model', columns='Label', values='Count').reindex(MODEL_ORDER).fillna(0)[COARSE]
    pivot.plot(kind='bar', stacked=True, figsize=(8, 5))
    plt.ylabel('Perfume Count')
    plt.xlabel('Model')
    plt.xticks(rotation=0)
    plt.title('Coarse Label Distribution (Majority per Perfume)')
    plt.legend(title='Emotion', bbox_to_anchor=(1.05, 1), loc='upper left')
    safe_savefig(FIG_DIR / 'label_distribution_stacked.png')

def plot_jsd_heatmap(jsd: pd.DataFrame):
    if jsd.empty:
        return
    plt.figure(figsize=(6, 5))
    im = plt.imshow(jsd.values.astype(float), cmap='plasma', vmin=0)
    plt.xticks(range(len(jsd.columns)), jsd.columns, rotation=45, ha='right')
    plt.yticks(range(len(jsd.index)), jsd.index)
    for i, j in np.ndindex(jsd.shape):
        if i == j:
            continue
        plt.text(j, i, f"{jsd.iloc[i, j]:.3f}", ha='center', va='center',
                 color='white' if jsd.iloc[i, j] > jsd.values.mean() else 'black')
    plt.title('Avg. Jensen–Shannon Divergence (Coarse Distributions)')
    plt.colorbar(im, label="JSD")
    safe_savefig(FIG_DIR / 'jsd_heatmap.png')

def plot_vad_scatters(unified: pd.DataFrame, cube: Dict[str, pd.DataFrame], counts: pd.Series, vad_corr: pd.DataFrame):
    if 'VAD_valence' not in unified.columns or vad_corr.empty:
        return
    x = unified['VAD_valence'].astype(float)
    size = np.sqrt(counts.reindex(unified['Perfume']).fillna(1).astype(float)) * 20

    for m in MODEL_ORDER:
        if m not in cube or f'{m}_joy' not in cube[m].columns:
            continue
        y = derive_valence_proxy(cube[m], m).astype(float)
        plt.figure(figsize=(6, 5))
        plt.scatter(x, y, alpha=0.7, s=size)

        plt.ylim(-1.0, 1.0)  # same scale for all models

        rc = vad_corr[vad_corr['Model'] == m]
        subtitle = ''
        if not rc.empty and not np.isnan(rc['Pearson_r'].iloc[0]):
            subtitle = f' (r≈{rc["Pearson_r"].iloc[0]:.2f})'

        try:
            if y.nunique() >= 2 and x.nunique() >= 2:
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                xs = np.linspace(x.min(), x.max(), 100)
                plt.plot(xs, p(xs))
        except Exception:
            pass

        plt.xlabel('VAD Valence (Lexicon)')
        plt.ylabel(f'{m} Valence Proxy (joy − neg)')
        plt.title(f'VAD vs {m}{subtitle}')
        plt.grid(True, linestyle='--', alpha=0.6)
        safe_savefig(FIG_DIR / f'vad_vs_{m}.png')

def plot_topN_stacked_per_perfume(unified: pd.DataFrame, cube: Dict[str, pd.DataFrame], topN: int, counts: pd.Series):
    if 'JHartmann' not in cube or f'JHartmann_joy' not in cube['JHartmann'].columns:
        return
    perfumes = counts.sort_values(ascending=False).head(topN).index.tolist()
    df = cube['JHartmann']
    sub = df[df['Perfume'].isin(perfumes)].set_index('Perfume').loc[perfumes][[f'JHartmann_{k}' for k in COARSE]]
    if sub.empty:
        return
    sub.columns = COARSE
    sub.plot(kind='bar', stacked=True, figsize=(12, 6))
    plt.ylabel('Probability')
    plt.xlabel('Perfume')
    plt.xticks(rotation=45, ha='right')
    plt.title(f'JHartmann Coarse Emotion Distribution — Top {len(sub)} Perfumes by Review Count')
    plt.legend(title='Emotion', bbox_to_anchor=(1.05, 1), loc='upper left')
    safe_savefig(FIG_DIR / f'JHartmann_stacked_top{len(sub)}.png')

def plot_nrc_heatmap(unified: pd.DataFrame, topN: int):
    nrc_cols = [c for c in unified.columns if c.startswith("NRC_")]
    if not nrc_cols:
        return
    nrc = unified[["Perfume"] + nrc_cols].set_index("Perfume")
    totals = nrc.sum(axis=1).sort_values(ascending=False)
    keep = totals.head(topN).index
    H = nrc.loc[keep]
    H = H.div(H.sum(axis=1), axis=0).fillna(0)
    plt.figure(figsize=(12, 8))
    plt.imshow(H.values, aspect="auto", cmap='YlGnBu')
    plt.colorbar(label="Proportion")
    plt.yticks(range(len(H.index)), H.index, fontsize=9)
    plt.xticks(range(len(H.columns)), [c.replace("NRC_", "") for c in H.columns], rotation=45, ha="right")
    plt.title(f"NRC Emotion Profile (Top {len(H)} Perfumes, Row-Normalized)")
    safe_savefig(FIG_DIR / "nrc_heatmap.png")

# ---------------------------- VAD → Color Palette ----------------------------

def vad_to_hsl(v, a, d):
    v, a, d = [np.clip(val, 0, 1) for val in [v, a, d]]
    H = 0.70 * v        # hue: low valence→blue, high→yellow/green
    S = 0.3 + 0.7 * a   # saturation by arousal
    L = 0.35 + 0.3 * d  # lightness by dominance
    return H, S, L

def plot_vad_color_swatches(palette: pd.DataFrame, maxN: int = 40, order="hue"):
    if palette.empty:
        return

    df = palette.copy()
    if order == "hue" and "H" in df.columns:
        df = df.sort_values("H")

    N = min(maxN, len(df))
    idx = np.linspace(0, len(df) - 1, N, dtype=int)   # spread over all hues
    show = df.iloc[idx]

    plt.figure(figsize=(12, 1.5))
    for i, (_, row) in enumerate(show.iterrows()):
        r, g, b = colorsys.hsv_to_rgb(float(row["H"]), float(row["S"]), float(row["V"]))
        plt.fill_between([i, i + 1], 0, 1, color=(r, g, b))
        plt.text(i + 0.5, -0.1, row['Perfume'][:10]+'..', ha='center', va='top', rotation=90, fontsize=8)
    plt.xlim(0, N); plt.yticks([]); plt.xticks([])
    plt.title(f"VAD→Color Swatches (HSV, Top {N} Perfumes)")
    plt.subplots_adjust(bottom=0.4)
    safe_savefig(FIG_DIR / "vad_color_swatches.png")
    
# ---------------------------- Built-in: Top Disagreements ----------------------------

def compute_and_show_top_disagreements(unified: pd.DataFrame, agree_df: pd.DataFrame, topK: int) -> pd.DataFrame:
    """Merge agreement metrics with per-model labels, print & save the lowest-agreement perfumes."""
    if agree_df.empty:
        return pd.DataFrame()
    model_cols = [c for c in [f'{m}_Label' for m in MODEL_ORDER] if c in unified.columns]
    keep_cols = ['Perfume', 'AgreementRate', 'EntropyBits', 'NumModels', 'MajorityLabel'] + model_cols
    merged = agree_df.merge(unified[['Perfume'] + model_cols], on='Perfume', how='left')

    ranked = merged.sort_values(
        by=['AgreementRate', 'EntropyBits', 'NumModels'],
        ascending=[True, False, False]
    ).head(topK)

    out_csv = OUT_DIR / f'top_disagreements_top{topK}.csv'
    ranked[keep_cols].to_csv(out_csv, index=False)

    # Pretty print
    pd.set_option('display.max_colwidth', 36)
    print("\n=== Top Disagreements (lowest agreement / highest entropy) ===")
    if not ranked.empty:
        print(ranked[keep_cols].to_string(index=False, formatters={
            'AgreementRate': '{:.2f}'.format,
            'EntropyBits': '{:.2f}'.format
        }))
        print(f"\n↗ Saved to: {out_csv.relative_to(ROOT)}")
    else:
        print("(none)")
    return ranked[keep_cols]

#---------------------------------------------------------------

def ensure_top_labels(unified: pd.DataFrame,
                      cube: Dict[str, pd.DataFrame],
                      min_conf: float = 0.0,
                      overwrite: bool = False) -> pd.DataFrame:
    """
    If a model's '<Model>_Label' column is missing/all-NaN, or if overwrite=True,
    derive it from the model's coarse distribution (argmax over COARSE).
    If the max prob < min_conf, assign 'neutral'.
    """
    uni = unified.copy()
    for m in MODEL_ORDER:
        lab_col = f"{m}_Label"
        if m not in cube or f"{m}_joy" not in cube[m].columns:
            continue

        # derive from coarse distribution with confidence
        dist = cube[m][[f"{m}_{k}" for k in COARSE]].copy()
        top  = dist.idxmax(axis=1).str.split('_').str[1]
        conf = dist.max(axis=1)
        derived = np.where(conf >= float(min_conf), top, 'neutral')

        if overwrite or (lab_col not in uni.columns) or (uni[lab_col].isna().all()):
            uni[lab_col] = derived
        else:
            # fill only where missing
            uni[lab_col] = uni[lab_col].where(uni[lab_col].notna(), derived)
    return uni

def derive_valence_proxy(df_model: pd.DataFrame, prefix: str) -> pd.Series:
    j   = df_model.get(f'{prefix}_joy',      pd.Series(0, index=df_model.index))
    ang = df_model.get(f'{prefix}_anger',    pd.Series(0, index=df_model.index))
    fear= df_model.get(f'{prefix}_fear',     pd.Series(0, index=df_model.index))
    sad = df_model.get(f'{prefix}_sadness',  pd.Series(0, index=df_model.index))
    neu = df_model.get(f'{prefix}_neutral',  pd.Series(0, index=df_model.index))
    sup = df_model.get(f'{prefix}_surprise', pd.Series(0, index=df_model.index))

    denom = (1 - (neu + sup)).replace(0, np.nan)
    centered = (j/denom) - ((ang + fear + sad)/denom)
    return centered.fillna(j - (ang + fear + sad)).clip(-1, 1)

def vad_to_hsl_rescaled(valence, arousal, dominance):
    v = pct_rescale(valence, 5, 95)
    a = pct_rescale(arousal, 5, 95)
    d = pct_rescale(dominance, 5, 95)
    H = 0.0 + 0.7 * v
    S = 0.25 + 0.65 * a
    L = 0.30 + 0.40 * d
    return H, S, L

def build_vad_palette(unified: pd.DataFrame, counts: pd.Series) -> pd.DataFrame:
    need = {'VAD_valence','VAD_arousal','VAD_dominance'}
    if not need.issubset(unified.columns): 
        return pd.DataFrame()

    # Spread hue around the entire wheel; bright & saturated
    H, S, V = vad_to_hsv_full(
        unified['VAD_valence'], 
        unified['VAD_arousal'], 
        unified['VAD_dominance'],
        hue_offset=0.0
    )
    out = pd.DataFrame({
        "Perfume": unified["Perfume"],
        "H": np.clip(np.round(H, 3), 0, 1),
        "S": np.clip(np.round(S, 3), 0, 1),
        "V": np.clip(np.round(V, 3), 0, 1),
        "ReviewCount": unified["Perfume"].map(counts).fillna(0).astype(int).values
    })
    return out.sort_values("ReviewCount", ascending=False)  # or just `return out`

def plot_nrc_lift_heatmap(unified: pd.DataFrame, topN: int = 25):
    nrc_cols = [c for c in unified.columns if c.startswith("NRC_")]
    if not nrc_cols: return
    M = unified[["Perfume"] + nrc_cols].set_index("Perfume").astype(float)
    row_prop = M.div(M.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    col_mean = row_prop.mean(axis=0).replace(0, np.nan)
    lift = (row_prop / col_mean).replace([np.inf, -np.inf], np.nan).fillna(0)
    keep = row_prop.sum(axis=1).sort_values(ascending=False).head(topN).index
    H = lift.loc[keep]
    plt.figure(figsize=(12, 8))
    im = plt.imshow(H.values, aspect="auto", cmap='coolwarm', vmin=0.5, vmax=1.5)
    plt.colorbar(im, label="Lift vs Corpus Avg (1.0 = avg)")
    plt.yticks(range(len(H.index)), H.index, fontsize=9)
    plt.xticks(range(len(H.columns)), [c.replace("NRC_","") for c in H.columns], rotation=45, ha="right")
    plt.title(f"NRC Emotion *Lift* (Top {len(H)} Perfumes)")
    safe_savefig(FIG_DIR / "nrc_lift_heatmap.png")

# --------------------------------------------------------------
def apply_runtime_mappings(cardiff_optimism_neutral: bool = False):
    """
    Optionally remap Cardiff 'optimism' → neutral (default is joy).
    This helps test whether Cardiff optimism is inflating disagreements.
    """
    if 'Cardiff' not in PROJECTORS:
        return
    if cardiff_optimism_neutral:
        PROJECTORS['Cardiff']['optimism'] = 'neutral'
    else:
        PROJECTORS['Cardiff']['optimism'] = 'joy'

# ---------------------------- Main ----------------------------

def main(topN: int, topK: int, normalize_vad: bool = False,
         toplabel_min_conf: float = 0.0,
         cardiff_optimism_neutral: bool = False,
         overwrite_labels: bool = False):
    if not UNIFIED_PATH.exists():
        raise FileNotFoundError(f"Missing {UNIFIED_PATH}. Run the aggregation first.")
    unified = pd.read_csv(UNIFIED_PATH)

    # Apply runtime label mapping toggles BEFORE building the cube
    apply_runtime_mappings(cardiff_optimism_neutral)

    # Optional: normalize VAD to [0,1] for palette only
    if normalize_vad and {'VAD_valence','VAD_arousal','VAD_dominance'}.issubset(unified.columns):
        cols = ['VAD_valence','VAD_arousal','VAD_dominance']
        x = unified[cols].to_numpy(dtype=float)
        lo, hi = np.nanmin(x, axis=0), np.nanmax(x, axis=0)
        span = np.where((hi - lo) < 1e-9, 1.0, (hi - lo))
        unified[cols] = (x - lo) / span

    counts = get_review_counts(unified)

    # --- Build coarse distributions first ---
    cube = build_coarse_cube(unified)

    # --- Ensure top labels exist (with confidence threshold) ---
    unified_labels = ensure_top_labels(
        unified, cube,
        min_conf=toplabel_min_conf,
        overwrite=overwrite_labels
    )

    print("Example labels:",
      {m: unified_labels[f"{m}_Label"].iloc[0] for m in MODEL_ORDER if f"{m}_Label" in unified_labels})

    # 1) AGREEMENT (use unified_labels everywhere labels are needed)
    kappa = compute_pairwise_kappa(unified_labels)
    if not kappa.empty:
        kappa.to_csv(OUT_DIR / 'pairwise_kappa.csv')
        plot_kappa_heatmap(kappa)

    fleiss = fleiss_kappa_from_unified(unified_labels)
    with open(OUT_DIR / 'fleiss_kappa.txt', 'w') as f:
        f.write(f"Fleiss' kappa across models: {fleiss:.4f}\n")

    agree_df = agreement_by_perfume(unified_labels)
    if not agree_df.empty:
        agree_df.to_csv(OUT_DIR / 'agreement_by_perfume.csv', index=False)

    # 2) Distribution similarity (already uses cube)
    jsd_coarse = pairwise_jsd_coarse(cube)
    if not jsd_coarse.empty:
        jsd_coarse.to_csv(OUT_DIR / 'jsd_pairwise.csv')
        plot_jsd_heatmap(jsd_coarse)

    # Label distribution also needs unified_labels
    dist_df = label_distribution_by_model(unified_labels)
    if not dist_df.empty:
        dist_df.to_csv(OUT_DIR / 'label_distribution_by_model.csv', index=False)
        plot_label_distribution(dist_df)

    # 3) VAD alignment (uses unified + cube)
    vad_corr = vad_correlations(unified, cube)
    if not vad_corr.empty:
        vad_corr.to_csv(OUT_DIR / 'vad_correlations.csv', index=False)
        plot_vad_scatters(unified, cube, counts, vad_corr)

    # 4) Per-perfume stacked bars (uses cube)
    plot_topN_stacked_per_perfume(unified, cube, topN, counts)

    # 5) NRC heatmaps
    plot_nrc_heatmap(unified, topN=25)
    plot_nrc_lift_heatmap(unified, topN=25)

    # 6) Common score-vector analysis
    corr_df, js_common_df = analyze_common_score_vectors(unified)
    if not corr_df.empty:
        corr_df.to_csv(OUT_DIR / 'score_vector_correlations.csv', index=False)
    if not js_common_df.empty:
        js_common_df.to_csv(OUT_DIR / 'jsd_common_score_vectors.csv', index=False)

    # 7) VAD → color palette
    palette = build_vad_palette(unified, counts)
    if not palette.empty:
        palette.to_csv(OUT_DIR / 'vad_color_palette_hsv.csv', index=False)
        plot_vad_color_swatches(palette, maxN=100)

    print(palette["H"].min(), palette["H"].max())  # expect ~0.0 … ~1.0

    # 8) Top disagreements (use unified_labels + agree_df)
    if topK and topK > 0:
        compute_and_show_top_disagreements(unified_labels, agree_df, topK)

    # 9) Console summary
    print("\n" + "="*24)
    print("=== Evaluation Summary ===")
    print("="*24)
    if not kappa.empty:
        offdiag = kappa.values[np.triu_indices_from(kappa.values, k=1)]
        print(f"Cohen's κ (mean off-diagonal): {np.nanmean(offdiag):.3f}")
    print(f"Fleiss' κ (across models): {fleiss:.3f}")
    if not vad_corr.empty:
        print("\n--- VAD Valence Correlations ---")
        for _, r in vad_corr.iterrows():
            print(f"  {r['Model']:<10} | Pearson r={r['Pearson_r']:.2f} (p={r['Pearson_p']:.3g})")
    print(f"\n✅ All outputs saved to: {OUT_DIR.relative_to(ROOT)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate and compare emotion pipelines on perfume reviews.")
    parser.add_argument('--topN', type=int, default=15, help='Top-N perfumes for stacked distribution plots.')
    parser.add_argument('--topK', type=int, default=15, help='How many lowest-agreement perfumes to print & save.')
    parser.add_argument('--normalize_vad', action='store_true', help='Min-max normalize VAD columns to [0,1].')
    parser.add_argument('--toplabel_min_conf', type=float, default=0.00,
                        help='When deriving labels, require this min probability; otherwise assign neutral.')
    parser.add_argument('--cardiff_optimism_neutral', action='store_true',
                        help="Map Cardiff 'optimism' to neutral (default maps to joy).")
    parser.add_argument('--overwrite_labels', action='store_true',
    help='Overwrite existing *_Label columns from coarse distributions (after mappings/conf thresholds).')

    args = parser.parse_args()

    main(
        topN=args.topN,
        topK=args.topK,
        normalize_vad=args.normalize_vad,
        toplabel_min_conf=args.toplabel_min_conf,
        cardiff_optimism_neutral=args.cardiff_optimism_neutral,
        overwrite_labels=args.overwrite_labels
    )

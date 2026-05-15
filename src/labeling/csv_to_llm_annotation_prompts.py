# Date Created: 2026-03-02 11:07:58+03:00
# Date Last Modified: 2026-05-15

# csv_to_llm_annotation_prompts.py
"""
Convert a filtered per-review CSV (e.g., Filtered_Reviews_MFK_PDM.csv) into
one-perfume prompt blocks you can paste into ChatGPT/Gemini.

Input (per-review CSV):
- Must contain columns: Perfume, Final Transcript (and optionally Title/Description/etc.)

Output:
- A single UTF-8 .txt file containing 1 prompt block per perfume in the exact structure:
  Perfume: <NAME>
  Snippet 1: ...
  Snippet 2: ...
  Snippet 3: ...
  Task: ...

Default behavior:
- Sample K=3 reviews per perfume (deterministic seed)
- Truncate each snippet to N characters
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from src.utils.paths import data_path


DEFAULT_HEADER_PROMPT = """You are annotating perfumes for an academic MSc thesis.
Follow the rules strictly.

### Multi-label emotions (0 or 1 only)

Return 1 if clearly present in the overall emotional tone of the perfume discussion.
Return 0 otherwise.

Emotions:
- joy
- sadness
- anger
- fear
- trust
- surprise
- anticipation
- disgust

Multiple emotions can be 1.

### VAD scale (integers only, 1–9)

Valence:
1 = very negative
9 = very positive

Arousal:
1 = calm/soft
9 = intense/energetic

Dominance:
1 = gentle/subtle
9 = powerful/commanding

### Important rules

- Do NOT infer from brand reputation.
- Base decisions only on provided snippets.
- Avoid marking everything as joy.
- Use full range of 1–9 for VAD.
- Return ONLY a JSON object.
- No explanations.

### Output format (strict)

{
  "joy": 0,
  "sadness": 0,
  "anger": 0,
  "fear": 0,
  "trust": 0,
  "surprise": 0,
  "anticipation": 0,
  "disgust": 0,
  "valence": 5,
  "arousal": 5,
  "dominance": 5
}

Do not include any text outside JSON.
"""


def shorten(text: str, n: int) -> str:
    t = str(text).replace("\n", " ").replace("\r", " ").strip()
    # normalize whitespace
    t = " ".join(t.split())
    return t[:n] + ("..." if len(t) > n else "")


def make_blocks(
    df: pd.DataFrame,
    perfume_col: str,
    text_col: str,
    k: int,
    seed: int,
    max_chars: int,
) -> list[str]:
    if perfume_col not in df.columns:
        raise ValueError(f"Missing required column: {perfume_col}")
    if text_col not in df.columns:
        raise ValueError(f"Missing required column: {text_col}")

    # drop empty perfumes
    df = df.copy()
    df[perfume_col] = df[perfume_col].astype(str).fillna("").str.strip()
    df = df[df[perfume_col] != ""]

    # sample k reviews per perfume deterministically
    sampled = (
        df.groupby(perfume_col, group_keys=False)
          .apply(lambda g: g.sample(n=min(k, len(g)), random_state=seed))
          .reset_index(drop=True)
    )

    # build blocks, preserving a stable perfume order
    blocks: list[str] = []
    for perfume, g in sampled.groupby(perfume_col, sort=True):
        snippets = [shorten(t, max_chars) for t in g[text_col].tolist()]
        # pad if fewer than k available
        while len(snippets) < k:
            snippets.append("")

        block = (
            f"Perfume: {perfume}\n\n"
            f"Snippet 1:\n{snippets[0]}\n\n"
            f"Snippet 2:\n{snippets[1]}\n\n"
            f"Snippet 3:\n{snippets[2]}\n\n"
            "Task:\n"
            "Label the perfume using the defined emotion schema and VAD scale.\n"
        )
        blocks.append(block)

    return blocks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in_csv",
        type=Path,
        default=data_path(
            "annotation_packets",
            "Filtered_Reviews_TOSKOVAT_ORTO_PARISI_ELO_NASOMATTO_SERGE_LUTENS.csv",
        ),
        help="Input filtered per-review CSV",
    )
    ap.add_argument(
        "--out_txt",
        type=Path,
        default=data_path("annotation_packets", "LLM_Annotation_Prompts.txt"),
        help="Output .txt file containing prompt blocks",
    )
    ap.add_argument(
        "--perfume_col",
        type=str,
        default="Perfume",
        help="Column name for perfume",
    )
    ap.add_argument(
        "--text_col",
        type=str,
        default="Final Transcript",
        help="Column name for the review text",
    )
    ap.add_argument(
        "--k",
        type=int,
        default=3,
        help="Number of snippets per perfume",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling",
    )
    ap.add_argument(
        "--max_chars",
        type=int,
        default=650,
        help="Max characters per snippet",
    )
    ap.add_argument(
        "--include_header_prompt",
        action="store_true",
        help="If set, prepends the full instruction prompt at the top of the output file",
    )
    args = ap.parse_args()

    in_path = args.in_csv
    out_path = args.out_txt
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    df = pd.read_csv(in_path)

    blocks = make_blocks(
        df=df,
        perfume_col=args.perfume_col,
        text_col=args.text_col,
        k=args.k,
        seed=args.seed,
        max_chars=args.max_chars,
    )

    with out_path.open("w", encoding="utf-8") as f:
        if args.include_header_prompt:
            f.write(DEFAULT_HEADER_PROMPT.strip() + "\n\n")
            f.write("=" * 60 + "\n\n")

        for i, b in enumerate(blocks, start=1):
            f.write(f"### PERFUME {i}\n")
            f.write(b)
            f.write("\n" + "=" * 60 + "\n\n")

    print(f"✅ Wrote {len(blocks)} perfume prompt blocks to: {out_path}")


if __name__ == "__main__":
    main()

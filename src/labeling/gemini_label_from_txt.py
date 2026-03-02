# strict tone-only labeling v1
"""
File: gemini_label_from_txt.py

Purpose
-------
Automatically generate structured emotion labels for perfume review snippets
using Google Gemini.

This script:
1. Reads a plain-text file containing perfume blocks formatted as:

   ### PERFUME N
   Perfume: <name>
   Snippet 1: ...
   Snippet 2: ...
   Snippet 3: ...

2. Sends the content to Gemini with a strictly defined emotion schema:
   - Multi-label categorical emotions (0/1):
        joy, sadness, anger, fear,
        trust, surprise, anticipation, disgust
   - Dimensional affect scores (1–9 integers):
        valence, arousal, dominance

3. Forces structured JSON output.
4. Saves the labeled results to a reproducible machine-readable file.

Why this exists
---------------
- Enables scalable pseudo-annotation for niche perfume subsets.
- Supports experimental comparison between:
    • Human annotations
    • Transformer-based models
    • Lexicon-based outputs
    • LLM-generated labels
- Maintains reproducibility via fixed prompt + schema + deterministic format.

Input
-----
snippets.txt

Output
------
labels/llm/strict/labels_strict_v1.json
    {
      "<perfume_name>": {
          "joy": 0/1,
          "sadness": 0/1,
          "anger": 0/1,
          "fear": 0/1,
          "trust": 0/1,
          "surprise": 0/1,
          "anticipation": 0/1,
          "disgust": 0/1,
          "valence": 1–9,
          "arousal": 1–9,
          "dominance": 1–9
      },
      ...
    }

Configuration
-------------
- Requires environment variable:
      GEMINI_API_KEY
- Default model:
      gemini-2.5-flash
- JSON response enforced using:
      response_mime_type = "application/json"

Notes
-----
- Conservative labeling: emotion=1 only if clearly supported by snippets.
- If snippet blocks are incomplete, labeling is based on available text.
- Recommended for batch experiments and comparative evaluation.

Thesis Stage
------------
Supervised validation / LLM-assisted annotation generation.
"""
import os, json, re
from pathlib import Path
from google import genai

from src.utils.paths import data_path, labels_path

MODEL = "gemini-2.5-flash"
IN_TXT = data_path("annotation_packets", "LLM_Annotation_Prompts.txt")
OUT_JSON = labels_path("llm", "strict", "labels_strict_v1.json")

SCHEMA = """
STRICT emotion-tone protocol (tone-only):

- Labels reflect ONLY the reviewer’s emotional tone expressed in the snippets.
- Do NOT infer from brand identity, perfume name, marketing concept, reputation, or niche-ness.
- Set an emotion to 1 ONLY if there is explicit linguistic evidence in the snippet text.
- Surprise=1 ONLY if explicit unexpectedness language appears (e.g., “surprised”, “didn’t expect”, “caught me off guard”, “shocked”, “unexpected”).
- Dominance must reflect language of control/assertiveness/confidence (e.g., “commanding”, “in control”, “dominates”), NOT projection/strength/beast mode unless framed as personal power/control.
- Arousal must reflect emotional activation in language (e.g., “excited”, “energized”, “calming”), NOT perfume intensity/performance.
- If emotional evidence is weak/neutral, default: valence=5, arousal=5, dominance=5 and all emotions=0.
- Output must be valid JSON only.

Emotions (0/1): joy, sadness, anger, fear, trust, surprise, anticipation, disgust
VAD (1–9 integers): valence, arousal, dominance

Return ONLY valid JSON of the form:
{
  "<perfume>": {"joy":0/1,"sadness":0/1,"anger":0/1,"fear":0/1,"trust":0/1,"surprise":0/1,"anticipation":0/1,"disgust":0/1,"valence":1-9,"arousal":1-9,"dominance":1-9},
  ...
}
"""

def main():
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise SystemExit("Set GEMINI_API_KEY first.")

    client = genai.Client(api_key=key)
    text = IN_TXT.read_text(encoding="utf-8", errors="ignore")

    resp = client.models.generate_content(
        model=MODEL,
        contents=[{"role": "user", "parts": [{"text": SCHEMA + "\n\n" + text}]}],
        config={"response_mime_type": "application/json"},
    )

    data = json.loads(resp.text)
    OUT_JSON.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✅ wrote {OUT_JSON} with {len(data)} perfumes")

if __name__ == "__main__":
    main()

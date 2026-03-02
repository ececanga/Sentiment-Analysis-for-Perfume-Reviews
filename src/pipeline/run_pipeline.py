# 01) run_pipeline.py
# Purpose: Automates the pipeline execution

import subprocess
import shlex  # Handles paths with spaces
from pathlib import Path

from src.utils.paths import PROJECT_ROOT

def run_script(script_path: Path) -> None:
    print(f"\n🚀 Running: {script_path}")
    subprocess.run(f'python "{script_path}"', shell=True, check=True)

# ---------------------
# 🔹 MANUAL STEPS FIRST
# ---------------------

# 1. (Optional) Scrape YouTube data (skip if already done)
# run_script(PROJECT_ROOT / "src" / "preprocessing" / "perfume_emotion_scraper.py")

# 2. Fix transcript formatting
run_script(PROJECT_ROOT / "src" / "preprocessing" / "format_transcripts.py")

# -------------------------------------
# ✅ AUTOMATED FULL PIPELINE BEGINS HERE
# -------------------------------------

# 3. Apply lexicon-based emotion and VAD
run_script(PROJECT_ROOT / "src" / "lexicon" / "apply_emotion_vad_lexicons.py")

# 4. Transformer-based Emotion Classification (select models to include)

# 🟢 J-Hartmann DistilRoBERTa
run_script(PROJECT_ROOT / "src" / "transformers" / "analyze_emotions_distilroberta_jhartmann.py")

# 🟡 Savani DistilBERT (optional)
run_script(PROJECT_ROOT / "src" / "transformers" / "analyze_emotions_distilbert_savani.py")

# 🔵 GoEmotions multilabel (optional)
run_script(PROJECT_ROOT / "src" / "transformers" / "analyze_emotions_goemotions_multi.py")

# 🔴 CardiffNLP RoBERTa (optional)
run_script(PROJECT_ROOT / "src" / "transformers" / "analyze_emotions_roberta_cardiffnlp.py")

# 5. Extract scent+emotion context terms (±4 window)
run_script(PROJECT_ROOT / "src" / "descriptive_terms" / "extract_context_terms.py")

# 6. Analyze top terms per perfume
run_script(PROJECT_ROOT / "src" / "descriptive_terms" / "analyze_top_terms.py")

# 7. Apply scent-emotion reprocessing (optional enhancement)
run_script(PROJECT_ROOT / "src" / "lexicon" / "reprocess_scent_emotion.py")

# 8. Merge NRC, VAD, Transformer outputs
run_script(PROJECT_ROOT / "src" / "aggregation_fusion" / "merge_emotion_profiles.py")

# 9. Final aggregation into unified scores
run_script(PROJECT_ROOT / "src" / "aggregation_fusion" / "aggregate_and_merge_emotions.py")

print("\n✅ Pipeline execution completed successfully!")

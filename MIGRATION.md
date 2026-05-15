<!--
Date Created: 2026-03-02 11:07:58+03:00
Date Last Modified: 2026-05-15
-->

# Migration Notes

This document describes the new project layout and how legacy paths were moved.

## New structure (high level)
- `data/` for datasets and lexicons
- `labels/` for LLM and silver labels
- `src/` for all code
- `outputs/` for evaluation and ablation artifacts
- `docs/` for thesis, papers, presentations, and diagrams
- `notes/` for working notes

## Path mapping (old -> new)

### Code (py files/*)
- `py files/Aggregation_Fusion/` -> `src/aggregation_fusion/`
- `py files/Descriptive_Term_Context_Analysis/` -> `src/descriptive_terms/`
- `py files/Evaluation/` -> `src/evaluation/`
- `py files/Labeling/` -> `src/labeling/`
- `py files/Lexicon_Based_Emotion/` -> `src/lexicon/`
- `py files/Pipeline Controller/` -> `src/pipeline/`
- `py files/Preprocessing_Formatting/` -> `src/preprocessing/`
- `py files/Transformer_Based_Emotion_Classification/` -> `src/transformers/`
- `py files/documents/g_s.py` -> removed (superseded by `archive/legacy_scripts/utils/slide1_de.py`)
- `py files/documents/generate_fusion_diagram.py` -> `archive/legacy_scripts/utils/generate_fusion_diagram.py`
- `py files/documents/slide1_de.py` -> `archive/legacy_scripts/utils/slide1_de.py`
- `csv files/inspect_csv_headers.py` -> removed

### Archived after runner cleanup
- `src/labeling/gemini_label_from_txt.py` -> `archive/legacy_scripts/labeling/gemini_label_from_txt.py`
- `src/labeling/json_to_labeled_csv.py` -> `archive/legacy_scripts/labeling/json_to_labeled_csv.py`
- `src/pipeline/demo_pipeline_small.py` -> `archive/legacy_scripts/pipeline/demo_pipeline_small.py`
- `src/training/train_eval_gold43.py` -> `archive/legacy_scripts/training/train_eval_gold43.py`
- `src/utils/generate_fusion_diagram.py` -> `archive/legacy_scripts/utils/generate_fusion_diagram.py`
- `src/utils/slide1_de.py` -> `archive/legacy_scripts/utils/slide1_de.py`
- `src/labeling/label_report.py` -> removed
- `src/utils/inspect_csv_headers.py` -> removed

### Data
- `csv files/YouTube_Perfume_Reviews_FullyFormatted.csv` -> `data/raw/YouTube_Perfume_Reviews_FullyFormatted.csv`
- `csv files/YouTube_Perfume_Reviews_Partial.csv` -> `data/raw/YouTube_Perfume_Reviews_Partial.csv`
- `csv files/YouTube_Perfume_Reviews_With_*.csv` -> `data/intermediate/`
- `csv files/processed_perfumes.csv` -> `data/intermediate/processed_perfumes.csv`
- `csv files/Aggregated_*.csv` -> `data/perfume_level/`
- `csv files/Merged_Emotion_VAD_Profile.csv` -> `data/perfume_level/Merged_Emotion_VAD_Profile.csv`
- `csv files/NRC_Emotion_Labels_Perfume.csv` -> `data/perfume_level/NRC_Emotion_Labels_Perfume.csv`
- `csv files/Transformer_Emotion_Scores.csv` -> `data/perfume_level/Transformer_Emotion_Scores.csv`
- `csv files/Unified_Emotion_Profile_Perfume.csv` -> `data/perfume_level/Unified_Emotion_Profile_Perfume.csv`
- `csv files/VAD_Scores_Perfume.csv` -> `data/perfume_level/VAD_Scores_Perfume.csv`
- `csv files/final_perfume_data.csv` -> `data/perfume_level/final_perfume_data.csv`
- `csv files/perfume_context_keywords.csv` -> `data/perfume_level/perfume_context_keywords.csv`
- `csv files/perfume_intensifier_counts.csv` -> `data/perfume_level/perfume_intensifier_counts.csv`
- `csv files/Annotation_Packet_*.csv` -> `data/annotation_packets/`
- `csv files/Filtered_*.csv` -> `data/annotation_packets/`
- `csv files/LLM_Annotation_Prompts*.txt` -> `data/annotation_packets/`

### Labels
- `labels.json` -> `labels/llm/loose/labels.json`
- `json files/labels_strict_v1.json` -> `labels/llm/strict/labels_strict_v1.json`
- `csv files/Unified_With_Silver_*.csv` -> `labels/silver/`

### Outputs
- `evaluation_out/` -> `outputs/evaluation_out/`
- `evaluation_out/Figures/` -> `outputs/figures/`
- `csv files/ablation_results/` -> `outputs/ablation_results/`

### Lexicons
- `NRC-Emotion-Lexicon/` -> `data/lexicons/nrc_emotion/NRC-Emotion-Lexicon/`
- `NRC-Emotion-Lexicon-Wordlevel-v0.92.txt` -> `data/lexicons/nrc_emotion/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt`
- `NRC-VAD-Lexicon-v2.1/` -> `data/lexicons/nrc_vad/NRC-VAD-Lexicon-v2.1/`
- `NRC-VAD-Lexicon-v2.1.txt` -> `data/lexicons/nrc_vad/NRC-VAD-Lexicon-v2.1.txt`

### Docs (thesis/papers/presentations/diagrams)
- `GoEmotions Papers/` -> `docs/papers/GoEmotions Papers/`
- `Ömer Uzel Thesis.pdf` -> `docs/thesis/Ömer Uzel Thesis.pdf`
- `c2071011_thesis_231123.docx` -> `docs/thesis/c2071011_thesis_231123.docx`
- `thesis_sections_draft_v2.docx` -> `docs/thesis/thesis_sections_draft_v2.docx`
- `fullCode.docx` -> `docs/thesis/fullCode.docx`
- `fullCode.pages` -> `docs/thesis/fullCode.pages`
- `fullCode copy.pages` -> `docs/thesis/fullCode copy.pages`
- `notes/*.pptx|.key|.pdf|.pages` -> `docs/presentations/`
- `py files/documents/workedExample_slide_prompts.json` -> `docs/presentations/workedExample_slide_prompts.json`
- `ChatGPT Image Dec 24, 2025, 10_55_33 AM.png` -> `docs/diagrams/ChatGPT Image Dec 24, 2025, 10_55_33 AM.png`
- `notes/*.png|.jpg` -> `docs/diagrams/`
- `py files/documents/*.png` -> `docs/diagrams/`

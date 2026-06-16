# Emotion-Rich V2 Review-Level Labeling Packets

Paste one batch file at a time into the browser LLM.

- Batch files: `v2_review_batch_*.txt`
- Prompt only: `prompt.md`
- Manifest: `manifest.csv`
- Empty output template: `output_template.csv`

Each batch asks the browser LLM to start with `Batch N output:` and then return the CSV rows.
You can paste that whole response back into Codex for validation and file writing.

Expected output schema:

```csv
synthetic_review_id,source_type,perfume_id,perfume_name,review_index,joy,sadness,anger,fear,trust,surprise,anticipation,disgust,valence,arousal,dominance
```

When all batches are labeled, combine the returned CSV rows into one file with the same header.

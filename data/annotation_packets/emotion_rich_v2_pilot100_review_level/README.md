# Emotion-Rich V2 Review-Level Labeling Packets

Paste one batch file at a time into the browser LLM.

- Batch files: `v2_review_batch_*.txt`
- Prompt only: `prompt.md`
- Manifest: `manifest.csv`
- Empty output template: `output_template.csv`

Expected output schema:

```csv
synthetic_review_id,source_type,perfume_id,perfume_name,review_index,joy,sadness,anger,fear,trust,surprise,anticipation,disgust,valence,arousal,dominance
```

When all batches are labeled, combine the returned CSV rows into one file with the same header.

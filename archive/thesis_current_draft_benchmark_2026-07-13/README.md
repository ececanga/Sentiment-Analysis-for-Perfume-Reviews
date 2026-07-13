# Thesis current-draft benchmark snapshot (2026-07-13)

Status: **archived current-draft benchmark; non-final**.

This Phase 1 snapshot preserves the exact selected working artifacts before the
source-group leakage repair and benchmark rerun. It is non-overwriting and does
not convert the archived results into final thesis evidence.

## Archive contract

- `912` payload files (236.4 MiB) are inventoried in
  `experiment_manifest.json` and hashed in `SHA256SUMS`.
- Canonical labels, feature inputs, raw transcripts, fold assignments, five OOF
  prediction sets, metrics, run manifests, tables, figures, annotation packets,
  pasted browser outputs, synthetic experiments, source code, and the thesis
  draft are retained under `payload/` using their repository-relative paths.
- The payload is intentionally Git-ignored because of its size. The metadata
  files at this directory root are suitable for version control.
- Two commit-pinned copies of `config/evaluation_config.yaml` and a clearly
  marked reconstructed browser-specific configuration are included.

## Critical interpretation boundary

The existing fold manifest groups by perfume only. A source audit found 341 of
392 shared video IDs crossing folds (462 perfumes) and 340 normalized transcript
hashes crossing folds (459 perfumes). These values require a grouped-CV rerun
before the benchmark is used as final evidence.

The project log records the browser labeler as Gemini 3.1 Pro. The repository
preserves the prompt, packets, pasted CSV responses, validation outputs, and
final label tables, but the browser responses do not contain machine-verifiable
model-version or session metadata.

The field historically called `macro_pr_auc` was computed as average precision;
future thesis text should call it **macro average precision**.

## Non-final material

`payload/provenance/NON_FINAL_ARTIFACTS.md` identifies the legacy eight-label
trainer, historical result branches, invalid mixed-source threshold tables, and
the thesis-table status classification. `experiment_manifest.json` maps Tables
4.1-4.10 and 5.1 to their preserved artifacts.

## Verify

From the repository root:

```bash
.venv/bin/python scripts/freeze_thesis_current_draft_benchmark.py --verify-only
```

Verification checks every payload SHA-256 digest, file size, metadata checksum,
and rejects unrecorded payload files.

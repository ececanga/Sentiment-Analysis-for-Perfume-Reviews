# Public repository scope and local inputs

The repository retains the final thesis, experiment code, configurations, labels, numerical feature tables, saved predictions, metrics, and final figures. The five-seed predictions and numerical summaries are unchanged by the publication cleanup.

## Inputs kept locally

The following are not distributed here:

- NRC Emotion and VAD lexicons and their bundled documents.
- YouTube transcripts, generated text/chunk datasets, and annotation packets containing source excerpts.
- Scraped perfume descriptions in `final_perfume_data.csv`.
- Thesis drafts, similarity reports, downloaded papers, personal documents, and one-off document-editing scripts.
- Logs, token caches, checkpoints, and regenerable distance matrices.

Synthetic review datasets and their labels remain available. Removing source text from the public repository does not remove the local inputs used for the completed experiments.

## NRC resources

Download the resources directly from their authors and follow their terms of use. The lexicons are not covered by any license for this repository's code.

- [NRC Emotion Lexicon](https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm), version 0.92: place the word-level file at `data/lexicons/nrc_emotion/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt`.
- [NRC VAD Lexicon](https://saifmohammad.com/WebPages/nrc-vad.html), version 2.1: place the main file at `data/lexicons/nrc_vad/NRC-VAD-Lexicon-v2.1.txt`.

Both resources' terms prohibit redistribution. Keep downloaded copies local.

## Reproducing results

The saved out-of-fold predictions support recomputing the reported five-seed metrics without the private text inputs:

```bash
python scripts/audit_thesis_five_seed_results.py
```

This audit verifies all 195 jobs and recalculates the means, sample standard deviations, and paired differences. It writes the audited tables and report to `outputs/thesis_five_seed_revision_20261012/`.

Rerunning preprocessing, annotation-packet generation, or text-model training requires local source data matching the configurations. See [the five-seed protocol](thesis_three_seed_experiments.md). Some older reports and checksum manifests refer to inputs now kept locally; they describe the original experiment and do not imply those inputs are included in a fresh clone. Original provenance paths and checksums have been retained rather than changing completed experiment records.

## Repository history

The publication cleanup removes the excluded paths from the published branch and tag histories as well as the current tree. Earlier commit identifiers have changed. Local backups are ignored and must not be uploaded. A local archive intended to stay private must remain outside tracked paths; moving it into a tracked `archive/` directory does not make it private.

# Phase 2A 2,000-Perfume Expansion Plan

## Summary

Phase 2A is a controlled expansion from the frozen 784-perfume validation set to a 2,000-perfume target set using the winning Gemini 2.5 Pro judge policy from Phase 1.

This is not a 5,000-perfume run. The 2,000-perfume step is intended to expose data quality, runtime, quota, cost, and benchmark stability issues before a larger expansion.

Current local raw transcript coverage is insufficient for a 2,000-perfume expansion. The repo has a larger name-only perfume universe, but usable raw review/transcript text currently covers only the frozen 784 perfumes.

## Phase 1 Recap

Phase 1 compared the frozen Gemini 2.5 Flash baseline against a Gemini 2.5 Pro judge branch on the same 784 perfumes, raw review/transcript universe, snippet policy, strict prompt/schema, aggregation rule, folds, seeds, and benchmark protocol.

Phase 1 result: `GO`.

- Frozen Flash gold macro-F1: `0.26819727891156464`
- Gemini Pro gold macro-F1: `0.4329004329004329`
- Frozen Flash benchmark mean macro-F1: `0.238986`
- Gemini Pro benchmark mean macro-F1: `0.28399199999999997`
- Phase 1 Pro labels completed: `784/784`
- Merge integrity: passed
- Official labels collapsed to zero: none
- Benchmark rows completed: all 5
- Benchmark threshold: `0.3`
- Freeze manifest: 16 SHA-256 hashes, no warnings

The promoted Phase 2A policy is the Gemini 2.5 Pro judge policy, not a copy of the 784 labels.

## Frozen Artifacts

Do not overwrite the frozen Flash baseline:

- `labels/gemini_labels_784_v1.csv`
- `outputs/model_benchmark/`
- `outputs/analysis_out/`

Do not overwrite the frozen Gemini Pro Phase 1 branch:

- `labels/gemini_labels_784_judge_g25pro_v1.csv`
- `labels/gemini_labels_784_judge_g25pro_v1.jsonl`
- `outputs/gemini_judge/judge_g25pro_v1/`
- `docs/phase1_gemini_pro_judge_decision.md`
- `outputs/gemini_judge/judge_g25pro_v1/phase1_freeze_manifest.json`

Phase 2A must use new versioned roots only.

## Data Inventory

Verified local inventory:

- Frozen baseline perfumes: `784`
- Phase 1 Pro perfumes: `784`
- Raw review rows in `data/raw/YouTube_Perfume_Reviews_FullyFormatted.csv`: `4,881`
- Raw unique perfumes: `784`
- Raw perfumes with usable `Final Transcript`: `784`
- Usable candidates outside frozen 784: `0`
- Name-only universe in `data/perfume_level/final_perfume_data.csv`: `2,191` rows, `2,184` unique `Name` values
- Processed perfume status rows in `data/intermediate/processed_perfumes.csv`: `1,330`
- Processed status counts: `Done=784`, `No videos=533`, `No results=13`
- Additional usable transcript perfumes required to reach 2,000: `1,216`

The larger local perfume universe is currently name-only for Phase 2A purposes. It is not enough for Gemini review-tone labeling because the promoted policy labels review/transcript snippets.

## Current Blocker

Phase 2A cannot start labeling yet. Local usable transcript coverage is only 784 perfumes, and all 784 overlap the frozen baseline.

The next project task is transcript/review collection for at least 1,216 additional perfumes, followed by rerunning the Phase 2A inventory. A deterministic 2,000-perfume target list should not be built until the inventory confirms enough usable local transcript coverage.

## Collection Diagnostic After First 25-Batch

The first fresh Phase 2A 25-perfume collection batch added `0` new done perfumes. Local diagnostics are saved under `outputs/gemini_judge/judge_g25pro_2k_v1/planning/`.

- Fresh 25-batch statuses: `transcript_unavailable=19`, `api_error=1`, `quota_error=5`
- Numeric or ambiguous candidate names in the fresh batch: `20/25`
- Likely main issue: the current scraper queue reaches weak alphabetical/name-only candidates early and does not use the `Brand` column from `final_perfume_data.csv`.
- Additional issue: YouTube Data API quota was reached near the end of the batch, so retrying immediately is not useful.
- Next queue strategy: use brand-aware, locally ranked candidates; avoid numeric/very short names; do not retry `done`; retry `transcript_unavailable` only after query strategy improves; retry `api_error`/`quota_error` only after network/quota recovery.

Recommended next validation remains capped and should use rows marked `recommended_for_next_batch=true` in the latest queue file.

## Old YouTube Scraping Strategy

The old 784-perfume scraper used `data/perfume_level/final_perfume_data.csv` as the perfume universe but effectively treated it as a name-only source. It ignored `Brand`, `Description`, and `Notes`, and searched only broad name templates:

- `{perfume} perfume`
- `{perfume} fragrance`
- `{perfume} review`

This worked well enough for the frozen 784 mostly because the broad searches found many already-reviewable products, manual reruns/key swapping worked around search quota interruptions, and enough videos had accessible captions at the time. It was not a fully automatic key-rotation or transcript-recovery system, and it did not use brand context to disambiguate names.

## Why Phase 2A Failed Differently

The first Phase 2A queue reached weak alphabetical/name-only candidates too early, including numeric or ambiguous names and non-review targets such as candles or hair/body variants. Multi-key search improves YouTube Data API quota handling, but it does not solve transcript fetching.

The current `transcript_fetch_error` pattern is `IpBlocked` / `blocking_or_rate_limit`, which means the transcript request path is being blocked by YouTube for the current IP/network. API keys authenticate and quota-limit YouTube Data API search; they do not authorize transcript retrieval and therefore do not clear transcript IP blocking.

## Revised YouTube-Only Strategy

Phase 2A remains YouTube transcript/review only. Queue v2 improved precision but marked only a small recommended slice; after the optimized collection run, the v2 recommended candidates were exhausted while later backup rows were not selected by the batch loader.

Queue v2 changes:

- Use `Brand + Name` for every candidate and every recommended query.
- Exclude the frozen 784 transcript perfumes.
- Exclude current Phase 2A `done`, `transcript_fetch_error`, and `transcript_unavailable` candidates by default.
- Exclude numeric/ambiguous names and identifiable non-fragrance products such as candles, hair mists, sample packs, room sprays, body products, deodorants, shower gels, lotions, and similar variants.
- Prefer clear perfume/fragrance products: `eau de parfum`, `eau de toilette`, `extrait de parfum`, `parfum`, `cologne`, and `fragrance`.
- Prefer known/reviewable/popular brands using the `Brand` column instead of only perfume-name text.
- Use reviewer-focused queries only, for example `{brand} {perfume} review`, `{brand} {perfume} perfume review`, `{brand} {perfume} fragrance review`, `"{brand}" "{perfume}" review`, and `{perfume} {brand} review`.

Queue v3 is `outputs/gemini_judge/judge_g25pro_2k_v1/planning/phase2a_next_candidate_queue_v3.csv`. It expands the recommended pool to hundreds of locally ranked candidates while preserving the same YouTube-only review/transcript constraint and the same default exclusions for frozen 784, Phase 2A `done`, `transcript_fetch_error`, `transcript_unavailable`, numeric/ambiguous names, and obvious non-fragrance variants. It adds ranking tiers so collection can continue through popular clear-fragrance candidates first, then clear-fragrance candidates from less reviewed brands, then lower-confidence but acceptable perfume products.

## Rejected Fallback

Do not build a Description+Notes 2K fallback or inventory. Product descriptions and note pyramids are brand/marketing text, not reviewer emotional expression. The thesis constraint requires emotion extraction from review/transcript-style language, so Phase 2A must continue to require usable review/transcript text before Gemini labeling.

## Proposed 2K Selection Rule

After raw transcript coverage is expanded, build a deterministic 2,000-perfume target set:

- Include the frozen 784 perfumes.
- Add 1,216 perfumes outside the frozen 784 with non-empty usable raw review/transcript text.
- Use stable normalized perfume names and IDs from the existing labeling utilities.
- Prefer perfumes with stronger review coverage when choosing among more than 1,216 eligible additions.
- Write the target set to a new Phase 2A path only, such as `data/perfume_level/phase2a_2000_perfume_targets.csv`.

No target-builder should be created until the inventory confirms the local data can support this rule.

## Phase 2A Artifact Paths

Draft config:

- `config/gemini_judge/judge_g25pro_2k_v1.yaml`

Planning outputs:

- `outputs/gemini_judge/judge_g25pro_2k_v1/planning/phase2a_inventory.json`
- `outputs/gemini_judge/judge_g25pro_2k_v1/planning/phase2a_candidate_perfumes.csv`

Future label outputs:

- `labels/gemini_labels_2000_judge_g25pro_v1.csv`
- `labels/gemini_labels_2000_judge_g25pro_v1.jsonl`
- `labels/gemini_labeling_report_2000_judge_g25pro_v1.json`
- `labels/gemini_labeling_errors_2000_judge_g25pro_v1.jsonl`
- `labels/gemini_labeling_run_2000_judge_g25pro_v1.log`

Future output root:

- `outputs/gemini_judge/judge_g25pro_2k_v1/`

## Proposed Stage Order

1. Expand raw review/transcript collection for at least 1,216 additional perfumes.
2. Format and validate expanded transcripts.
3. Rerun `src.analysis.inventory_phase2a_2k_expansion`.
4. If inventory confirms at least 2,000 usable perfumes, build a deterministic 2K target list.
5. Dry-run the future judge branch command resolution without Gemini API calls.
6. Run Gemini labeling only after go/no-go checks pass.
7. Merge, inspect integrity, and benchmark at the fixed `0.3` threshold.
8. Review stability, data quality, runtime, quota behavior, and benchmark behavior before considering 5K.

## Calls, Cost, Runtime, And Quota Risks

The Phase 2A labeling estimate is fixed by policy:

- `2,000` perfumes
- `3` snippets per perfume
- `6,000` Gemini calls before retries

Retry overhead should be expected for transient API failures, schema validation failures, rate limits, and occasional malformed responses. Exact cost should be calculated from current Gemini 2.5 Pro pricing immediately before production labeling.

Runtime and quota risks:

- Gemini 2.5 Pro latency will make 6,000 calls materially slower than the 784-perfume Phase 1 run.
- Daily or per-minute request quota may require throttling or multiple sessions.
- Resume behavior must be verified before production.
- Failed JSON/schema responses must be captured without corrupting completed labels.
- Long transcript snippets must remain bounded by the existing snippet policy.

## Required Integrity Checks

Before labeling:

- Raw transcript inventory confirms at least 2,000 perfumes with non-empty usable text.
- Target list has exactly 2,000 unique normalized perfumes.
- Target list includes all frozen 784 perfumes.
- No target ID collisions after normalization/hash disambiguation.
- Output roots point only to Phase 2A versioned paths.
- Threshold remains `0.3`; threshold sweep remains disabled.

After labeling:

- CSV and JSONL contain exactly 2,000 unique perfume IDs.
- CSV and JSONL ID sets match.
- Label report has zero failed and zero missing perfumes.
- Label payloads satisfy the strict schema.
- No official label collapses to zero unexpectedly.
- Merge integrity passes against the Phase 2A target/unified feature table.
- Benchmark outputs are isolated under `outputs/gemini_judge/judge_g25pro_2k_v1/`.

## Go/No-Go Criteria

Do not start actual 2K labeling until all are true:

- At least 2,000 perfumes have usable local raw review/transcript text.
- A deterministic 2K target CSV exists and passes uniqueness checks.
- A dry-run resolves only Phase 2A output paths.
- API quota and runtime plan are acceptable for at least 6,000 calls plus retries.
- No frozen baseline or Phase 1 Pro paths are configured as write targets.

Current status: `NO-GO` for labeling because local usable transcript coverage is only 784 perfumes.

## Reuse And 784 Assumptions

Reusable safely for planning:

- `src.labeling.strict_schema`
- `src.labeling.run_gemini_labeling` normalization/snippet rules
- `src.analysis.analyze_gemini_unified` ID construction logic
- Existing Phase 1 docs/manifests as read-only anchors

Hard-coded 784 assumptions to avoid for 2K production:

- `src/labeling/run_gemini_labeling.py` writes production labels as `gemini_labels_784_{tag}.*`
- `src/pipeline/run_gemini_judge_branch.py` validates label, merge, and deep-text output against frozen baseline IDs and hard-coded `784`
- `src/analysis/freeze_phase1_gemini_pro_branch.py` is explicitly Phase 1 only
- `scripts/run_phase1_judge.sh` points to `judge_g25pro_v1`

# Phase 2A Transcript/IP Recovery Plan

## Current Diagnosis

`transcript_fetch_error` is currently caused by `IpBlocked` / `blocking_or_rate_limit` from transcript retrieval, not by the YouTube Data API search call.

`IpBlocked` means YouTube is refusing transcript requests from the current network/IP. The search API can still work with valid keys while transcript retrieval fails, because transcript retrieval is a separate non-keyed request path handled by the transcript library.

API key rotation helps only the YouTube Data API search quota. More API keys do not fix `transcript_fetch_error` when the root cause is transcript IP blocking.

Do not use Description+Notes fallback for Phase 2A. The thesis expansion requires emotion-bearing review/transcript language, not brand/product marketing text.

## Safe Recovery Test

After waiting or changing network, test exactly one previously failed video ID before running any candidate batch:

```bash
python3 src/preprocessing/run_phase2a_collection_batch.py --diagnose-transcript-error VIDEO_ID
```

Example using one locally observed `IpBlocked` video ID:

```bash
python3 src/preprocessing/run_phase2a_collection_batch.py --diagnose-transcript-error YXcy3En2gCA
```

This diagnostic fetches a transcript for one video ID only. It does not run YouTube search, does not need API key rotation, and should be treated as the gate before more transcript attempts.

## Recommended Recovery Steps

1. Wait/cooldown before making more transcript requests.
2. Switch network if cooldown does not clear the block.
3. Try a mobile hotspot if the normal network remains blocked.
4. Rerun one transcript-only diagnostic with `--diagnose-transcript-error VIDEO_ID`.
5. Only if that diagnostic succeeds, run a small queue-v2 batch.

## Retry Policy

Retryable by default:

- `quota_error`
- `auth_error` after fixing the key or key pool
- `api_error` if now reclassified or known to be transient
- `youtube_api_error`

Do not retry by default:

- `transcript_fetch_error`
- `transcript_unavailable`
- `done`

`transcript_fetch_error` should only be retried by explicit opt-in after the one-video diagnostic passes. `transcript_unavailable` should stay excluded unless the query strategy changes enough to justify a fresh search.

## Queue v2 Commands

Regenerate the local queue without network calls:

```bash
python3 src/analysis/build_phase2a_youtube_queue_v2.py
```

Dry-run the next queue-v2 batch:

```bash
python3 src/preprocessing/run_phase2a_collection_batch.py --candidate-queue-csv outputs/gemini_judge/judge_g25pro_2k_v1/planning/phase2a_next_candidate_queue_v2.csv --per-key-limit 25 --max-keys 1 --max-results 5 --max-query-templates-per-perfume 2 --stop-after-valid-videos-per-perfume 5 --dry-run
```

Run a live queue-v2 batch only after the transcript diagnostic passes:

```bash
python3 src/preprocessing/run_phase2a_collection_batch.py --candidate-queue-csv outputs/gemini_judge/judge_g25pro_2k_v1/planning/phase2a_next_candidate_queue_v2.csv --per-key-limit 25 --max-keys 1 --max-results 5 --max-query-templates-per-perfume 2 --stop-after-valid-videos-per-perfume 5 --max-workers 4 --sleep-sec 2
```

Do not add `--retry-transcript-errors` to the live command unless the transcript-only diagnostic has passed and the specific goal is to retest previously IP-blocked videos.

## Old-Pace Collection Mode

The original 784-perfume collection likely succeeded partly because it behaved like a slower manual process: one active key at a time, natural cooldowns between reruns, fewer concentrated transcript requests, and less retry pressure on the transcript endpoint.

Phase 2A now has an old-pace mode that keeps queue v3, brand-aware search, key loading, status safety, and transcript-only collection, but slows the run down to reduce the chance of `IpBlocked` / `blocking_or_rate_limit`.

Old-pace defaults:

- queue: `outputs/gemini_judge/judge_g25pro_2k_v1/planning/phase2a_next_candidate_queue_v3.csv`
- `per_key_limit=3`
- `max_keys=1`
- `max_workers=1`
- `sleep_sec=30`
- `max_results=5`
- `max_query_templates_per_perfume=2`
- `stop_after_valid_videos_per_perfume=2`
- no default retry for `transcript_fetch_error`

Old-pace safety stops:

- stop immediately when the transcript error category looks like IP blocking or rate limiting
- stop if `transcript_fetch_error >= 3` in one run
- stop if the transcript-fetch-error rate is at least `0.5` after at least `3` attempts
- stop if at least `3` perfumes were attempted and no raw rows were written

Recommended sequence after cooldown or network change:

```bash
python3 src/preprocessing/run_phase2a_collection_batch.py --diagnose-transcript-error YXcy3En2gCA
```

Then dry-run old-pace:

```bash
.venv/bin/python src/preprocessing/run_phase2a_collection_batch.py --old-pace --dry-run
```

Only if the diagnostic passes, run a tiny old-pace live batch:

```bash
.venv/bin/python src/preprocessing/run_phase2a_collection_batch.py --old-pace
```

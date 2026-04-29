# Autoresearch: DDTree DFlash throughput on Castle

## Objective
Improve DFlash + DDTree decode throughput for the Qwen3.5-27B llama.cpp server port on Castle without sacrificing greedy bit-equal correctness in the e2e harness. The current user-visible TPS is far below the standalone DFlash baseline and below the target-only llama.cpp server reference.

## Metrics
- **Primary**: `tps` (tok/s, higher is better) — e2e speculative decode throughput, computed as `spec_generated / spec_timing_sec`.
- **Secondary**: `step_ms`, `draft_ms`, `exact_decode_ms`, `pack_ms`, `topk_ms`, `acceptance`, `gen_tokens`, `spec_sec`.

## How to Run
`./autoresearch.sh`

The script syncs the local working tree source files to Castle, builds `test-speculative-tree-e2e` and `llama-server`, runs the Castle CUDA e2e benchmark, and prints `METRIC` lines.

## Files in Scope
- `common/speculative-tree-driver.cpp` / `.h` — DDTree proposal, target feature ring, exact validation, timing counters.
- `src/llama-context.cpp` / `.h` — decode path, DFlash persist/rollback, draft profiling.
- `src/llama-graph.cpp` — graph inputs and draft positional behavior.
- `src/models/dflash-draft.cpp` — DFlash draft model graph/input integration.
- `tools/server/server-context.cpp` — DDTree server integration and prompt cache rebuild policy.
- `tests/test-speculative-tree-e2e.cpp` — benchmark/correctness harness and metrics output.
- `ggml/src/ggml-cuda/*` and relevant `ggml/src/ggml-cpu/*` files — only for narrowly scoped kernels needed by DDTree/DFlash.

## Off Limits
- Do not change model files, benchmark prompts, or expected outputs.
- Do not relax correctness gates or remove bit-equal validation.
- Do not overfit to a single prompt by hard-coding token IDs, prompt lengths, or outputs.
- Do not touch unrelated llama.cpp features.

## Constraints
- Benchmark must pass e2e bit-equal greedy correctness.
- Keep `--require-full-prompt-ingest`; use `--require-ddtree` only when the tested mode should run batched tree verify.
- No new external dependencies.
- Castle is the source of performance truth; local CPU/Metal builds are not enough for keeps.
- If a Castle server is occupying the GPU, benchmark runs may stop it to free VRAM for e2e testing.

## Current Baseline Context
Status from `DDTREE_STATUS_2026-04-29.md`:
- Stable server config uses `LLAMA_DDTREE_TARGET_FEAT_CTX=1024`, `-ngl65 -ngld6 -c65536`, q4 KV, chain-only exact validation.
- Real task sample after 1024-window server: API TPS 6.38 tok/s, wall TPS 5.92 tok/s.
- Server logs imply raw DDTree decode around 7.7 tok/s with `exact_avg_commit ~= 3.5`, `step ~= 456 ms`.
- Main remaining costs: exact target 1-token decode and draft decode. Recent timing split showed `exact_decode` dominates exact validation.
- Fast rollback helps only when persist fits with full target offload; at 64k/full-draft on 24GB it currently OOMs, and reducing target offload loses the gain.

## What's Been Tried
- Shared draft lm_head with target output weight: correct and frees about 1 GiB duplicate GPU allocation.
- Full draft GPU offload (`-ngld6`) after avoiding unused persist allocation: correct and substantially faster draft compute.
- Server prompt cache/checkpoint restored with DDTree rebuild window: repeated long prompts now reuse cache and rebuild only last target-feature window.
- `LLAMA_DDTREE_TARGET_FEAT_CTX=1024`: kept; reduced pack/draft costs and improved repeated-request latency.
- Removing default batched tree verify from exact correctness path: kept; avoids diagnostic target-tree/snapshot overhead when exact chain validation is final authority.
- Exact batched spine: discarded; snapshot/restore/replay cost made it slower.
- Skipping logsumexp in top-k scores: discarded; top-k dominated by vocab scan/heap, not score normalization.
- Fast rollback at 64k/full-draft: not viable on current 24GB GPU because persist allocation needs about 1.7 GiB extra VRAM.

## Next Experiment Directions
- Lower target 1-token exact decode cost without changing outputs.
- Reduce draft decode graph compute or mixed backend overhead.
- Explore safe conditional batched acceptance only if correctness evidence supports it.
- Move or compress target feature data to reduce pack/upload cost after larger bottlenecks are addressed.

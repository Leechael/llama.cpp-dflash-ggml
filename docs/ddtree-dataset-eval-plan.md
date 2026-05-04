# DDTree Dataset Eval Plan

Goal: measure the llama.cpp DDTree port on the original DFlash public
benchmarks: HumanEval, GSM8K, and Math500. The previous numbers were produced
by `repo/dflash/scripts/bench_llm.py`; this plan measures the llama.cpp path on
Castle through `test-speculative-tree-e2e`.

## Scope

- Datasets: 10 prompts per dataset, `datasets.shuffle(seed=42)`, matching the
  original DFlash `bench_llm.py` sampling policy.
- Generation: greedy, `n_gen=256`.
- Primary Python-compatible config: `budget=22` non-root DDTree nodes,
  `top_k=0` (auto => K=8 when branching), `proposal_temp=1`,
  `target_feat_ctx=2048`, `q8_0` KV.
- Harness: `build-server/bin/test-speculative-tree-e2e`.
- Correctness gate: greedy token trajectory must be bit-equal between chain and
  spec output for every sample.

Current status after the paper verifier fix: the default correctness path does
not trust batched/tree logits. It skips the redundant tree verifier and gates
final acceptance through exact one-token chain validation. The Python
`bench_llm.py` headline uses the standalone fast batched path with Q8_0 KV and
does not check bit-equal token trajectories; direct standalone testing on
HumanEval_01 shows this fast path diverges from standalone AR at generated token
34 for `n_gen=64`.

Castle 4090 llama.cpp exact-gated result with the Python-compatible parameters,
Qwen3.5-27B Q4_K_M target + DFlash draft, `n_gen=256`, 10 prompts per dataset:

| dataset | AR tok/s | DFlash tok/s | AL | speedup | bit-equal |
|---|---:|---:|---:|---:|---:|
| HumanEval | 46.32 | 40.23 | 8.34 | 0.87x | 10/10 |
| GSM8K | 46.31 | 40.12 | 6.72 | 0.87x | 10/10 |
| Math500 | 46.33 | 40.05 | 7.30 | 0.86x | 10/10 |

The exact-gated path is slower than AR because it still runs one exact target
decode per generated token, then adds the draft pass. On the 30-prompt run the
target AR cost is ~21.59 ms/token; exact-gated DDTree costs ~24.9 ms/token
after draft overhead is amortized.

For throughput experiments, the unsafe fast batched path with
`LLAMA_DDTREE_UNSAFE_TRUST_BATCHED=1` is not a correctness row:

| dataset | AR tok/s | DFlash tok/s | AL | speedup | bit-equal |
|---|---:|---:|---:|---:|---:|
| HumanEval | 46.33 | 145.82 | 8.14 | 3.15x | 4/10 |
| GSM8K | 46.31 | 120.68 | 6.57 | 2.61x | 2/10 |
| Math500 | 46.31 | 131.32 | 7.20 | 2.84x | 5/10 |

For performance experiments, `LLAMA_DDTREE_UNSAFE_TRUST_BATCHED=1` restores the
fast batched posterior behavior. Rows where `bit_equal` is not 10/10 must be
treated as non-correctness-gated throughput rows, matching the limitation of the
standalone Python benchmark. `LLAMA_DDTREE_DIAG_BATCHED=1` restores the
diagnostic batched+exact path.

## Metrics

Primary metrics:

- `chain_decode_tps`: target-only greedy chain decode throughput from the same
  llama.cpp binary.
- `spec_decode_tps`: DDTree decode throughput from per-step timing, excluding
  model load and prompt prefill.
- `spec_acceptance`: exact average committed tokens per DDTree step.
- `speedup_decode`: `spec_decode_tps / chain_decode_tps`.
- `bit_equal_pass`: required for every sample.

Secondary metrics:

- `spec_e2e_tps`: includes prompt ingest and harness context setup; useful for
  sanity only, not directly comparable with the original DFlash headline.
- `step_ms`, `draft_ms`, `topk_ms`, `exact_ms`: diagnose where the llama.cpp
  port loses time or acceptance.

## Commands

From local Mac, sync the bench script to Castle if needed:

```sh
cd /Users/leechael/workshop/playgrounds/luceboxhub-castle/repo/dflash/deps/llama.cpp
rsync -az scripts/bench_dflash_datasets_llamacpp.py \
  castle.local:/home/leechael/workshop/lucebox-hub/dflash/deps/llama.cpp/scripts/
```

Build the harness on Castle:

```sh
/usr/bin/ssh castle.local \
  'cd /home/leechael/workshop/lucebox-hub/dflash/deps/llama.cpp && \
   cmake --build build-server -j 16 --target test-speculative-tree-e2e'
```

Run a one-prompt smoke first:

```sh
/usr/bin/ssh castle.local \
  'cd /home/leechael/workshop/lucebox-hub/dflash/deps/llama.cpp && \
   /home/leechael/workshop/lucebox-hub/sglang/.venv/bin/python \
   scripts/bench_dflash_datasets_llamacpp.py \
     --datasets HumanEval --n-sample 1 --gen 64 \
     --out-dir /tmp/llamacpp_dflash_dataset_smoke'
```

Before the full run, a short horizon check is useful after verifier changes:

```sh
for n in 32 64 128 256; do
  /usr/bin/ssh castle.local \
    "cd /home/leechael/workshop/lucebox-hub/dflash/deps/llama.cpp && \
     /home/leechael/workshop/lucebox-hub/sglang/.venv/bin/python \
     scripts/bench_dflash_datasets_llamacpp.py \
       --datasets HumanEval --n-sample 1 --gen $n \
       --out-dir /tmp/llamacpp_dflash_horizon_$n" || true
done
```

Run the full 30-prompt bench:

```sh
/usr/bin/ssh castle.local \
  'cd /home/leechael/workshop/lucebox-hub/dflash/deps/llama.cpp && \
   /home/leechael/workshop/lucebox-hub/sglang/.venv/bin/python \
   scripts/bench_dflash_datasets_llamacpp.py \
     --datasets HumanEval,GSM8K,Math500 \
     --n-sample 10 --gen 256 \
     --out-dir /tmp/llamacpp_dflash_dataset_full'
```

For faster trend data, use the short gated run:

```sh
/usr/bin/ssh castle.local \
  'cd /home/leechael/workshop/lucebox-hub/dflash/deps/llama.cpp && \
   /home/leechael/workshop/lucebox-hub/sglang/.venv/bin/python \
   scripts/bench_dflash_datasets_llamacpp.py \
     --datasets HumanEval,GSM8K,Math500 \
     --n-sample 10 --gen 32 \
     --out-dir /tmp/llamacpp_dflash_dataset_short_30_gen32'
```

For a reference-budget comparison, rerun with:

```sh
--budget 22 --top-k 22 --out-dir /tmp/llamacpp_dflash_dataset_budget22
```

For exact-path diagnostics, rerun a sample with:

```sh
--verifier exact --exact-validation --out-dir /tmp/llamacpp_dflash_dataset_exact_diag
```

## Result Files

Each run writes:

- `summary.md`: dataset-level table.
- `results.csv`: per-prompt flat table.
- `results.json`: full arguments and per-prompt metrics.
- `logs/*.log`: raw harness output for audit and parser repair.
- `tokens/*.bin`: chain/spec generated token files.

## Interpretation

Use `speedup_decode` and `spec_acceptance` for direct comparison against the
original DFlash benchmark only for rows where `bit_equal_pass` is true. If
long-generation rows fail the bit-equal gate, treat those rows as correctness
failures first and performance data second. Use server/API TPS only as a
follow-up after this harness confirms acceptance and decode speed, because API
TPS includes server queueing, prompt handling, and response streaming.

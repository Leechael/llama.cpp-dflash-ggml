# DDTree Phase 1 test fixtures

## Files

- `tree_5node.json` — 5-node tree fixture (1 root + 4 verify nodes). Used by `test-qwen35-tree --mode tree`. Token IDs are mid-range Qwen3.5 placeholders; regenerate from real sampled tokens as needed.

- `short_prompt.bin` — 16 int32 LE token IDs for chain-mode warm-up (Test 1.A). Hardcoded Qwen3.5 BOS/system-prompt tokens. Regenerate with `make_short_prompt.py`.

- `make_short_prompt.py` — standalone script; reads a text string and writes int32 LE tokens to `short_prompt.bin`. Requires `llama-cpp-python`. Falls back to the hardcoded 16-token fixture if the package is absent.

## Build

The test binary is gated behind a CMake option (off by default, not in ctest):

```
cmake -DLLAMA_BUILD_TESTS_QWEN35_TREE=ON <other flags> ..
make test-qwen35-tree
```

Requires a Qwen3.5-27B GGUF (~16 GB). Not available in CI.

## Running (castle only)

Test 1.A — chain mode does not regress vs current fork master:

```bash
./test-qwen35-tree --mode chain \
    --model /path/to/Qwen3.5-27B-Q4_K_M.gguf \
    --prompt-tokens fixtures/ddtree/short_prompt.bin \
    --out-logits /tmp/chain.bin

# Compare against a golden dump produced by an unmodified fork build:
python3 scripts/compare_logits.py /tmp/chain_golden.bin /tmp/chain.bin
```

Test 1.B — tree mode aligns with test_dflash (blocked, see below):

```bash
./test-qwen35-tree --mode tree \
    --model /path/to/Qwen3.5-27B-Q4_K_M.gguf \
    --tree-fixture fixtures/ddtree/tree_5node.json \
    --out-logits /tmp/tree.bin

python3 scripts/compare_logits.py /tmp/test_dflash_golden.bin /tmp/tree.bin
```

## Blocker: Test 1.B prerequisite

Test 1.B requires `test_dflash` to support `--dump-verify-logits`, which dumps the per-node logits produced by the dflash tree forward. This flag is a Phase 0 prerequisite listed in roadmap section 7.2 and is not yet implemented. Until it lands, Test 1.B cannot produce a golden reference and cannot be run end-to-end.

---

# DDTree Phase 2 test fixtures

## Build

Gated behind a CMake option (off by default, not in ctest):

```
cmake -DLLAMA_BUILD_TESTS_QWEN35_TREE_ROLLBACK=ON <other flags> ..
make test-qwen35-tree-rollback
```

Requires Phase 2 `llama_seq_snapshot` / `llama_seq_restore` / `llama_seq_release` API from the implementation agent.

## Test 2.A — snapshot/restore symmetry

```bash
./test-qwen35-tree-rollback \
    --model /path/to/Qwen3.5-27B-Q4_K_M.gguf \
    --prompt-tokens fixtures/ddtree/short_prompt.bin \
    --gen 8 \
    --out-logits-pre /tmp/pre.bin \
    --out-logits-post /tmp/post.bin

# Both logit dumps must be bit-equal:
python3 scripts/compare_logits.py /tmp/pre.bin /tmp/post.bin --abs-tol 0 --rel-tol 0
```

## Test 2.B — BLOCKED

Requires `test_dflash --dump-state-at-commit`, a Phase 0 prerequisite not yet implemented. Until that flag lands, Test 2.B (tree partial-accept vs sequential golden state) cannot be run.

## Test 2.C — deferred

Long-prompt OOM stress test deferred to Phase 5 server integration.

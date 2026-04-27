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

---

# DDTree Phase 3 test fixtures

## Files

- `dflash_draft_metadata_smoke.json` — expected GGUF metadata fields for a
  converted dflash-draft model.  Used by `check_dflash_draft_gguf.py`.

## Build

Both Phase 3 test binaries are gated behind a single CMake option (off by
default, not in ctest):

```
cmake -DLLAMA_BUILD_TESTS_DFLASH_DRAFT=ON <other flags> ..
make test-dflash-draft test-qwen35-chain-capture
```

Requires:
- A Qwen3.5-27B GGUF (~16 GB).
- A converted dflash-draft GGUF (see conversion step below).
- Phase 3 implementation API: `llama_model_token_embd_lookup`,
  `llama_set_capture_hidden`, `llama_get_hidden_capture`.

## Converting safetensors to dflash-draft GGUF

The conversion script is written by the implementation agent in parallel.
Once it lands at `repo/dflash/scripts/convert_dflash_draft.py`:

```bash
python repo/dflash/scripts/convert_dflash_draft.py \
    /path/to/dflash_draft/model.safetensors \
    -o /path/to/draft.gguf
```

Until the script lands this step is a TODO.

## Validating the converted GGUF (Test 3.A)

```bash
python repo/scripts/check_dflash_draft_gguf.py \
    /path/to/draft.gguf \
    tests/fixtures/ddtree/dflash_draft_metadata_smoke.json
# Exit 0: PASS. Exit 1: one line per discrepant field on stderr.
```

## Test 3.B — Draft forward bit-equal vs dflash reference

BLOCKED on Phase 0 prerequisite: `test_dflash --dump-draft-output` flag is
not yet implemented.  Until that flag lands, no golden reference exists and
the end-to-end comparison cannot be run.

The driver (`test-dflash-draft`) can still be used standalone to inspect
draft logits:

```bash
./test-dflash-draft \
    --target-model /path/to/Qwen3.5-27B-Q4_K_M.gguf \
    --draft-model  /path/to/draft.gguf \
    --last-tok     12345 \
    --target-feat-bin /path/to/target_feat.bin \
    --out-logits   /tmp/draft_logits.bin

# Once the Phase 0 flag lands, compare against the dflash reference dump:
python3 scripts/compare_logits.py /tmp/dflash_draft_golden.bin /tmp/draft_logits.bin
```

## Test 3.C — Hidden capture does not break chain mode

Run both capture and no-capture modes in a single invocation.  The driver
asserts logits are bit-equal and that the capture buffer contains valid
(non-NaN, non-zero) values.

```bash
./test-qwen35-chain-capture \
    --model        /path/to/Qwen3.5-27B-Q4_K_M.gguf \
    --prompt-tokens fixtures/ddtree/short_prompt.bin \
    --out-logits   /tmp/capture_logits.bin \
    --out-capture  /tmp/capture_buf.bin
# Exit 0: both assertions passed.
```

Regression-only mode (skips Mode A, writes no-capture logits for external
comparison against a Phase 1 chain golden dump):

```bash
./test-qwen35-chain-capture \
    --model        /path/to/Qwen3.5-27B-Q4_K_M.gguf \
    --prompt-tokens fixtures/ddtree/short_prompt.bin \
    --out-logits   /tmp/nocapture_logits.bin \
    --out-capture  /dev/null \
    --no-capture

# Compare against Phase 1 chain baseline:
python3 scripts/compare_logits.py /tmp/chain_golden.bin /tmp/nocapture_logits.bin --abs-tol 0 --rel-tol 0
```

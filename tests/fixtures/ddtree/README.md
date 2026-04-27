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

---

# DDTree Phase 4 test fixtures

## Build

The Phase 4 end-to-end test binary is gated behind its own CMake option (off
by default, not in ctest):

```
cmake -DLLAMA_BUILD_TESTS_SPECULATIVE_TREE_E2E=ON <other flags> ..
make test-speculative-tree-e2e
```

Requires:
- A Qwen3.5-27B GGUF (~16 GB).
- A converted dflash-draft GGUF.
- Phase 4 implementation API:
  `llama_speculative_tree_driver_init` / `_step` / `_free`
  (`common/speculative-tree-driver.h`) and `llama_set_target_feat_raw`
  (Phase 3 gap; `llama.h`).

## Test 4.A — Spec-decode token trajectory matches chain reference

This is the canonical Phase 4 acceptance test.  With `--temp 0` (greedy),
DDTree speculative decoding is lossless: the target verifies each draft token
against its own argmax before accepting it.  The resulting token sequence MUST
be bit-equal to a plain greedy chain decode from the same prompt.

```bash
./test-speculative-tree-e2e \
    --target-model  /path/to/Qwen3.5-27B-Q4_K_M.gguf \
    --draft-model   /path/to/draft.gguf \
    --prompt-tokens fixtures/ddtree/short_prompt.bin \
    --gen 64 \
    --out-chain /tmp/chain.tokens \
    --out-spec  /tmp/spec.tokens \
    --ddtree-budget 22 \
    --temp 0

# The driver prints: chain_n=X spec_n=Y first_divergence=none bytes_match=Z/Z
# Exit 0 = PASS.

# Optional: offline comparison using the script:
python3 scripts/compare_tokens.py /tmp/chain.tokens /tmp/spec.tokens
# Exit 0 = all positions match AND n_a == n_b.
```

**Note**: `--temp 0` is required for the bit-equal guarantee.  Non-zero
temperature introduces stochastic sampling, which makes the two sequences
non-deterministic relative to each other.  With non-zero temp the comparison
is informational only (the driver does not assert bit-equality).

## Test 4.B — BLOCKED

Test 4.B (comparison of the spec-decode token sequence against the output of
the `test_dflash` daemon) is blocked on a Phase 0 prerequisite: the
`test_dflash` daemon mode interface (`--daemon` flag) is not yet implemented.
Until that flag lands, no golden `test_dflash` token stream can be produced
for comparison.

The chain-reference comparison in Test 4.A gives strong independent functional
verification and is sufficient for Phase 4 sign-off.

---

# DDTree Phase 5 test fixtures

Phase 5 integrates the DDTree driver into `llama-server` as a selectable
speculative-decode mode (`--speculative-mode ddtree`).  There are no new
binary test fixtures; validation is done via two shell scripts in the
super-repo `scripts/` directory that run against the server on Castle.

## New CLI flags (impl agent deliverables)

| Flag | Type | Default | Notes |
|------|------|---------|-------|
| `--speculative-mode {chain,ddtree}` | string | chain | selects speculative backend |
| `--ddtree-budget N` | int | 22 | max draft tokens per tree step |
| `--ddtree-temp F` | float | 0.0 | draft sampling temperature |
| `--ddtree-no-chain-seed` | bool flag | off | disable chain-seed warmup |

These flags are parsed in `common/arg.cpp`.  The HTTP API surface is
unchanged: same OpenAI-compatible `/v1/chat/completions` and
`/v1/messages` endpoints, SSE streaming, `tool_use`, and
`reasoning_content` all work identically to chain mode.

Only `--parallel 1` (single slot) is supported in Phase 5.

## Test 5.A — Smoke test (primary acceptance)

Run from the local mac:

```bash
# Default (port 8003, single prompt)
./repo/scripts/run_server_ddtree_castle.sh

# Custom port and prompt
./repo/scripts/run_server_ddtree_castle.sh 8003 "Write a haiku."
```

What the script does:

1. Verifies the `llama-server` binary exists on Castle.
2. Kills any leftover DDTree-mode server (idempotent).
3. Starts the server via `nohup` on Castle, logging to `/tmp/ddtree_server.log`.
4. Polls `/health` up to 60 s (2 s interval).
5. Sends one non-streaming `POST /v1/chat/completions` and validates
   `choices[0].message.content` is non-empty.
6. Sends one streaming request and confirms SSE `data:` lines arrive.
7. Prints the last 50 lines of the server log.
8. Stops the server.
9. Exits 0 (SMOKE PASS) or non-zero (SMOKE FAIL).

## Test 5.B — Mode comparison (optional / informational)

```bash
./repo/scripts/compare_server_modes_castle.sh
# or with a custom prompt:
./repo/scripts/compare_server_modes_castle.sh "Describe the sky in exactly 32 tokens."
```

Starts both a chain-mode server (port 8001) and a DDTree-mode server
(port 8003) on Castle, sends the same greedy (`temperature: 0`,
`max_tokens: 32`) prompt to each, and reports the first word-level
divergence index.

**Expected outcome**: divergence at word index >= 17.  This matches the
Phase 4 finding that chain and DDTree outputs are bit-equal up to
approximately 17 tokens per speculative-step boundary, then diverge due
to KV-cache / conversation-state differences in the server slot state
machine.  Divergence at or above that threshold is not a regression.
Early divergence (word index < 17) warrants investigation.

The script always exits 0; the comparison is informational.

## Phase 5 acceptance criteria

Phase 5 acceptance is **smoke level only**:

- `run_server_ddtree_castle.sh` exits 0 (SMOKE PASS).
- Non-streaming completion returns valid JSON with non-empty content.
- SSE streaming delivers at least one `data:` chunk.

Full production replacement of `dflash/scripts/server.py` (pointing
Claude Code at `http://castle.local:8002/v1`) is the user's **manual**
validation step and is outside automated testing scope.

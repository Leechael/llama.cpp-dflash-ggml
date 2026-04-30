#!/usr/bin/env bash
set -euo pipefail

REMOTE=castle.local
REMOTE_DIR=/home/leechael/workshop/lucebox-hub/dflash/deps/llama.cpp
TARGET_MODEL=/home/leechael/workshop/lucebox-hub/dflash/models/Qwen3.5-27B-Q4_K_M.gguf
DRAFT_MODEL=/home/leechael/workshop/lucebox-hub/dflash/models/draft/model.gguf
PROMPT_TEXT=${AUTORESEARCH_PROMPT:-/tmp/real_rendered_prompt.txt}
GEN=${AUTORESEARCH_GEN:-32}
CTX=${AUTORESEARCH_CTX:-65536}
KV_TYPE=${AUTORESEARCH_KV_TYPE:-q4_0}
DRAFT_GPU_LAYERS=${AUTORESEARCH_DRAFT_GPU_LAYERS:-6}
N_BATCH=${AUTORESEARCH_N_BATCH:-512}
N_UBATCH=${AUTORESEARCH_N_UBATCH:-512}
BUDGET=${AUTORESEARCH_BUDGET:-40}
PROFILE=${LLAMA_DDTREE_PROFILE:-1}
BLOCK_SIZE=${LLAMA_DDTREE_BLOCK_SIZE:-}
TARGET_FEAT_CTX=${LLAMA_DDTREE_TARGET_FEAT_CTX:-128}
FAST_BATCHED=${LLAMA_DDTREE_FAST_BATCHED:-}
FAST_ROLLBACK=${LLAMA_DDTREE_FAST_ROLLBACK:-}
SNAPSHOT_FALLBACK=${LLAMA_DDTREE_SNAPSHOT_FALLBACK:-}
FORCE_CHAIN=${LLAMA_DDTREE_FORCE_CHAIN_KERNEL:-}
SKIP_EXACT_SEQ_RM=${LLAMA_DDTREE_SKIP_EXACT_SEQ_RM:-}
TOP_K=${LLAMA_DDTREE_TOP_K:-4}
TREE_ROWS=${LLAMA_DDTREE_TREE_ROWS:-}
CHAIN_SEED=${LLAMA_DDTREE_CHAIN_SEED:-0}
CHAIN_DEPTH_CAP=${LLAMA_DDTREE_CHAIN_DEPTH_CAP:-}
PROPOSAL_TEMP=${LLAMA_DDTREE_PROPOSAL_TEMP:-0.7}
TRACE=${LLAMA_DDTREE_TRACE:-}
CHAIN_CAPTURE=${LLAMA_DDTREE_CHAIN_CAPTURE:-}
CHAIN_SEQ_RM=${LLAMA_DDTREE_CHAIN_SEQ_RM:-}
NO_FLASH_ARG=${AUTORESEARCH_NO_FLASH_ATTN:+--no-flash-attn}

# Sync only source/control files needed for the benchmark. Avoid .git and build dirs.
rsync -az --delete \
  --exclude build --exclude build-server --exclude build-cpu --exclude .git \
  common include src tests tools ggml CMakeLists.txt cmake \
  "$REMOTE:$REMOTE_DIR/" >/dev/null

ssh "$REMOTE" "pgrep -f '[b]uild-server/bin/llama-server' | xargs -r kill"

ssh "$REMOTE" "cd '$REMOTE_DIR' && cmake --build build-server -j 16 --target test-speculative-tree-e2e llama-server" >/tmp/autoresearch_build.log 2>&1 || {
  tail -80 /tmp/autoresearch_build.log
  exit 1
}

out_file=$(mktemp /tmp/autoresearch_ddtree.XXXXXX)
ssh "$REMOTE" "cd '$REMOTE_DIR' && \
  LLAMA_DDTREE_PROFILE='$PROFILE' \
  LLAMA_DDTREE_BLOCK_SIZE='$BLOCK_SIZE' \
  LLAMA_DDTREE_TARGET_FEAT_CTX='$TARGET_FEAT_CTX' \
  LLAMA_DDTREE_FAST_BATCHED='$FAST_BATCHED' \
  LLAMA_DDTREE_FAST_ROLLBACK='$FAST_ROLLBACK' \
  LLAMA_DDTREE_SNAPSHOT_FALLBACK='$SNAPSHOT_FALLBACK' \
  LLAMA_DDTREE_FORCE_CHAIN_KERNEL='$FORCE_CHAIN' \
  LLAMA_DDTREE_SKIP_EXACT_SEQ_RM='$SKIP_EXACT_SEQ_RM' \
  LLAMA_DDTREE_TOP_K='$TOP_K' \
  LLAMA_DDTREE_TREE_ROWS='$TREE_ROWS' \
  LLAMA_DDTREE_CHAIN_SEED='$CHAIN_SEED' \
  LLAMA_DDTREE_CHAIN_DEPTH_CAP='$CHAIN_DEPTH_CAP' \
  LLAMA_DDTREE_PROPOSAL_TEMP='$PROPOSAL_TEMP' \
  ${TRACE:+LLAMA_DDTREE_TRACE='$TRACE'} \
  ${CHAIN_CAPTURE:+LLAMA_DDTREE_CHAIN_CAPTURE='$CHAIN_CAPTURE'} \
  ${CHAIN_SEQ_RM:+LLAMA_DDTREE_CHAIN_SEQ_RM='$CHAIN_SEQ_RM'} \
  ./build-server/bin/test-speculative-tree-e2e \
    --target-model '$TARGET_MODEL' \
    --draft-model '$DRAFT_MODEL' \
    --prompt-text '$PROMPT_TEXT' \
    --gen '$GEN' \
    --out-spec /tmp/autoresearch_spec.bin \
    --out-chain /tmp/autoresearch_chain.bin \
    --ddtree-budget '$BUDGET' \
    ${TOP_K:+--ddtree-top-k '$TOP_K'} \
    --require-full-prompt-ingest \
    --temp 0 \
    --n-gpu-layers ${AUTORESEARCH_N_GPU_LAYERS:-65} \
    --draft-gpu-layers '$DRAFT_GPU_LAYERS' \
    --n-ctx '$CTX' \
    --n-batch '$N_BATCH' \
    --n-ubatch '$N_UBATCH' \
    --kv-type '$KV_TYPE' \
    $NO_FLASH_ARG" >"$out_file" 2>&1 || {
  tail -120 "$out_file"
  exit 1
}

cat "$out_file" | tail -220
grep -E 'chain timing detail|chain timing:' "$out_file" || true

python3 - "$out_file" <<'PY'
import re, sys
text = open(sys.argv[1], 'r', errors='replace').read()

def last_float(pattern, default=0.0):
    vals = re.findall(pattern, text)
    return float(vals[-1]) if vals else default

def last_int(pattern, default=0):
    vals = re.findall(pattern, text)
    return int(vals[-1]) if vals else default

spec_sec = last_float(r"spec timing:\s*([0-9.]+)\s*sec")
gen_tokens = last_int(r"spec:\s*generated\s+(\d+)\s+tokens")
# Committed can be > requested generation because one speculative step may validate beyond the requested output.
steps = last_int(r"steps=(\d+)")
committed = last_int(r"(?:^|\s)committed=(\d+)")
step_ms = last_float(r"spec timing avg:.*?step=([0-9.]+)")
pack_ms = last_float(r"spec timing avg:.*?pack=([0-9.]+)")
draft_ms = last_float(r"spec timing avg:.*?draft=([0-9.]+)")
topk_ms = last_float(r"spec timing avg:.*?topk=([0-9.]+)")
exact_ms = last_float(r"spec timing avg:.*?exact=([0-9.]+)")
exact_decode_ms = last_float(r"spec timing avg:.*?exact_decode=([0-9.]+)")
acceptance = last_float(r"exact_avg_commit_per_step=([0-9.]+)")
if not spec_sec or not gen_tokens or not step_ms or not steps:
    print("Failed to parse spec timing/generated tokens/decode step timing", file=sys.stderr)
    sys.exit(2)
e2e_tps = gen_tokens / spec_sec
decode_tps = gen_tokens / (steps * step_ms / 1000.0)
print(f"METRIC tps={decode_tps:.6f}")
print(f"METRIC e2e_tps={e2e_tps:.6f}")
print(f"METRIC spec_sec={spec_sec:.6f}")
print(f"METRIC gen_tokens={gen_tokens}")
print(f"METRIC steps={steps}")
print(f"METRIC committed={committed}")
print(f"METRIC step_ms={step_ms:.6f}")
print(f"METRIC pack_ms={pack_ms:.6f}")
print(f"METRIC draft_ms={draft_ms:.6f}")
print(f"METRIC topk_ms={topk_ms:.6f}")
print(f"METRIC exact_ms={exact_ms:.6f}")
print(f"METRIC exact_decode_ms={exact_decode_ms:.6f}")
print(f"METRIC acceptance={acceptance:.6f}")
PY

rm -f "$out_file"

#!/usr/bin/env bash
set -euo pipefail

REMOTE=castle.local
REMOTE_DIR=/home/leechael/workshop/lucebox-hub/dflash/deps/llama.cpp
TARGET_MODEL=/home/leechael/workshop/lucebox-hub/dflash/models/Qwen3.5-27B-Q4_K_M.gguf
DRAFT_MODEL=/home/leechael/workshop/lucebox-hub/dflash/models/draft/model.gguf
PROMPT_TEXT=/tmp/real_rendered_prompt.txt
GEN=${AUTORESEARCH_GEN:-16}
CTX=${AUTORESEARCH_CTX:-65536}
TARGET_FEAT_CTX=${LLAMA_DDTREE_TARGET_FEAT_CTX:-1024}

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
  LLAMA_DDTREE_PROFILE=1 \
  LLAMA_DDTREE_TARGET_FEAT_CTX='$TARGET_FEAT_CTX' \
  env -u LLAMA_DDTREE_FAST_BATCHED -u LLAMA_DDTREE_FAST_ROLLBACK -u LLAMA_DDTREE_SNAPSHOT_FALLBACK -u LLAMA_DDTREE_FORCE_CHAIN_KERNEL \
  ./build-server/bin/test-speculative-tree-e2e \
    --target-model '$TARGET_MODEL' \
    --draft-model '$DRAFT_MODEL' \
    --prompt-text '$PROMPT_TEXT' \
    --gen '$GEN' \
    --out-spec /tmp/autoresearch_spec.bin \
    --out-chain /tmp/autoresearch_chain.bin \
    --ddtree-budget 22 \
    --require-full-prompt-ingest \
    --temp 0 \
    --n-gpu-layers 65 \
    --draft-gpu-layers 6 \
    --n-ctx '$CTX' \
    --n-batch 512 \
    --n-ubatch 512 \
    --kv-type q4_0" >"$out_file" 2>&1 || {
  tail -120 "$out_file"
  exit 1
}

cat "$out_file" | tail -120

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
committed = last_int(r"committed=(\d+)")
step_ms = last_float(r"spec timing avg:.*?step=([0-9.]+)")
pack_ms = last_float(r"spec timing avg:.*?pack=([0-9.]+)")
draft_ms = last_float(r"spec timing avg:.*?draft=([0-9.]+)")
topk_ms = last_float(r"spec timing avg:.*?topk=([0-9.]+)")
exact_ms = last_float(r"spec timing avg:.*?exact=([0-9.]+)")
exact_decode_ms = last_float(r"spec timing avg:.*?exact_decode=([0-9.]+)")
acceptance = last_float(r"exact_avg_commit_per_step=([0-9.]+)")
if not spec_sec or not gen_tokens:
    print("Failed to parse spec timing/generated tokens", file=sys.stderr)
    sys.exit(2)
tps = gen_tokens / spec_sec
print(f"METRIC tps={tps:.6f}")
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

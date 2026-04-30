#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

prompts="/tmp/real_rendered_prompt.txt /tmp/ddtree_prompt_task0_tok17350.txt /tmp/ddtree_prompt_task129_tok18108.txt /tmp/ddtree_prompt_task297_tok18689.txt /tmp/ddtree_prompt_task528_tok18918.txt /tmp/ddtree_prompt_task629_tok19171.txt /tmp/ddtree_prompt_task761_tok21321.txt /tmp/ddtree_prompt_task995_tok56813.txt"

for p in $prompts; do
    echo "=== PROMPT $p ==="
    out=$(AUTORESEARCH_PROMPT="$p" LLAMA_DDTREE_PROPOSAL_TEMP=0.7 ./autoresearch.sh 2>&1 || true)
    tps=$(echo "$out" | python3 -c "import sys,re; m=re.search(r'METRIC tps=([0-9.]+)', sys.stdin.read()); print(m.group(1) if m else '0')")
    same=$(echo "$out" | python3 -c "import sys,re; m=re.search(r'batched_exact_same=(\d+)', sys.stdin.read()); print(m.group(1) if m else '0')")
    diff=$(echo "$out" | python3 -c "import sys,re; m=re.search(r'batched_exact_diff=(\d+)', sys.stdin.read()); print(m.group(1) if m else '0')")
    steps=$(echo "$out" | python3 -c "import sys,re; m=re.search(r'steps=(\d+)', sys.stdin.read()); print(m.group(1) if m else '0')")
    committed=$(echo "$out" | python3 -c "import sys,re; m=re.search(r'committed=(\d+)', sys.stdin.read()); print(m.group(1) if m else '0')")
    echo "result: prompt=$p tps=$tps same=$same diff=$diff steps=$steps committed=$committed"
done
echo "METRIC tps=0"

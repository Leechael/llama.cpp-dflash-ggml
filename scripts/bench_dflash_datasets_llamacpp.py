#!/usr/bin/env python3
"""Run the DFlash/DDTree dataset bench through llama.cpp's e2e harness.

This reproduces the original dflash `scripts/bench_llm.py` prompt sampling
policy, but uses `test-speculative-tree-e2e` so the measured path is the
llama.cpp port: target chain decode plus DDTree speculative decode with a
greedy bit-equal gate.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


CASTLE_TARGET = "/home/leechael/workshop/lucebox-hub/dflash/models/Qwen3.5-27B-Q4_K_M.gguf"
CASTLE_DRAFT = "/home/leechael/workshop/lucebox-hub/dflash/models/draft/model.gguf"


@dataclass(frozen=True)
class Bench:
    name: str
    dataset: str
    config: str | None
    split: str
    extract: Callable[[dict], str]


BENCHES = [
    Bench("HumanEval", "openai_humaneval", None, "test", lambda x: x["prompt"]),
    Bench("GSM8K", "gsm8k", "main", "test", lambda x: f"Question: {x['question']}\nAnswer: "),
    Bench("Math500", "HuggingFaceH4/MATH-500", None, "test", lambda x: f"Problem: {x['problem']}\nSolution: "),
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default="./build-server/bin/test-speculative-tree-e2e")
    ap.add_argument("--target-model", default=CASTLE_TARGET)
    ap.add_argument("--draft-model", default=CASTLE_DRAFT)
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--datasets", default="HumanEval,GSM8K,Math500")
    ap.add_argument("--n-sample", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gen", type=int, default=256)
    ap.add_argument("--budget", type=int, default=16)
    ap.add_argument("--top-k", type=int, default=16)
    ap.add_argument("--proposal-temp", type=float, default=1.0)
    ap.add_argument("--target-feat-ctx", type=int, default=128)
    ap.add_argument("--verifier", default="paper", choices=["paper", "exact"])
    ap.add_argument("--exact-validation", action="store_true")
    ap.add_argument("--grammar-verify", action="store_true")
    ap.add_argument("--n-gpu-layers", type=int, default=65)
    ap.add_argument("--draft-gpu-layers", type=int, default=6)
    ap.add_argument("--n-ctx", type=int, default=4096)
    ap.add_argument("--n-batch", type=int, default=64)
    ap.add_argument("--n-ubatch", type=int, default=64)
    ap.add_argument("--kv-type", default="q4_0", choices=["f16", "q8_0", "q4_0"])
    ap.add_argument("--timeout-sec", type=int, default=900)
    ap.add_argument("--skip-longer-than", type=int, default=3500)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--keep-logs", action="store_true")
    return ap.parse_args()


def require_file(path: str, label: str) -> None:
    if not Path(path).is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def load_tokenizer():
    try:
        from transformers import AutoTokenizer
    except Exception as exc:  # pragma: no cover - depends on host env
        print(f"[bench] tokenizer unavailable, prompt length filter disabled: {exc}", file=sys.stderr)
        return None
    return AutoTokenizer.from_pretrained("Qwen/Qwen3.5-27B", trust_remote_code=True)


def token_count(tokenizer, prompt: str) -> int:
    if tokenizer is None:
        return -1
    return len(tokenizer.encode(prompt, add_special_tokens=False))


def selected_benches(names: str) -> list[Bench]:
    wanted = {x.strip() for x in names.split(",") if x.strip()}
    by_name = {b.name: b for b in BENCHES}
    unknown = wanted - set(by_name)
    if unknown:
        raise ValueError(f"unknown dataset(s): {', '.join(sorted(unknown))}")
    return [b for b in BENCHES if b.name in wanted]


def load_samples(bench: Bench, n_sample: int, seed: int) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset(bench.dataset, bench.config, split=bench.split)
    ds = ds.shuffle(seed=seed).select(range(n_sample))
    return list(ds)


def last_float(pattern: str, text: str, default: float = 0.0) -> float:
    vals = re.findall(pattern, text, flags=re.S | re.M)
    return float(vals[-1]) if vals else default


def last_int(pattern: str, text: str, default: int = 0) -> int:
    vals = re.findall(pattern, text, flags=re.S | re.M)
    return int(vals[-1]) if vals else default


def parse_e2e_output(text: str) -> dict:
    chain_decode_avg_ms = last_float(r"chain timing detail:.*?decode_avg=([0-9.]+)\s*ms", text)
    chain_decode_total_ms = last_float(r"chain timing detail:.*?decode_total=([0-9.]+)\s*ms", text)
    chain_decode_steps = last_int(r"chain timing detail:.*?decode_steps=(\d+)", text)
    chain_n = last_int(r"chain_n=(\d+)", text)
    spec_n = last_int(r"spec_n=(\d+)", text)
    spec_sec = last_float(r"spec timing:\s*([0-9.]+)\s*sec", text)
    spec_steps = last_int(r"spec stats: steps=(\d+)", text)
    committed = last_int(r"(?:^|\s)committed=(\d+)", text)
    step_ms = last_float(r"spec timing avg:.*?step=([0-9.]+)", text)
    draft_ms = last_float(r"spec timing avg:.*?draft=([0-9.]+)", text)
    topk_ms = last_float(r"spec timing avg:.*?topk=([0-9.]+)", text)
    exact_ms = last_float(r"spec timing avg:.*?exact=([0-9.]+)", text)
    acceptance = last_float(r"exact_avg_commit_per_step=([0-9.]+)", text)
    first_div = "none"
    m_div = re.search(r"first_divergence=([^ \n]+)", text)
    if m_div:
        first_div = m_div.group(1)
    passed = "PASS: all" in text and first_div == "none"

    chain_tps = 1000.0 / chain_decode_avg_ms if chain_decode_avg_ms else 0.0
    if chain_decode_total_ms and chain_decode_steps:
        chain_tps = chain_decode_steps / (chain_decode_total_ms / 1000.0)
    spec_decode_tps = spec_n / (spec_steps * step_ms / 1000.0) if spec_n and spec_steps and step_ms else 0.0
    spec_e2e_tps = spec_n / spec_sec if spec_n and spec_sec else 0.0

    return {
        "chain_n": chain_n,
        "spec_n": spec_n,
        "chain_decode_steps": chain_decode_steps,
        "chain_decode_avg_ms": chain_decode_avg_ms,
        "chain_decode_total_ms": chain_decode_total_ms,
        "chain_decode_tps": chain_tps,
        "spec_sec": spec_sec,
        "spec_steps": spec_steps,
        "spec_committed": committed,
        "spec_step_ms": step_ms,
        "spec_draft_ms": draft_ms,
        "spec_topk_ms": topk_ms,
        "spec_exact_ms": exact_ms,
        "spec_acceptance": acceptance,
        "spec_decode_tps": spec_decode_tps,
        "spec_e2e_tps": spec_e2e_tps,
        "speedup_decode": spec_decode_tps / chain_tps if chain_tps else 0.0,
        "first_divergence": first_div,
        "bit_equal_pass": passed,
    }


def run_one(args: argparse.Namespace, out_dir: Path, bench: Bench, idx: int, prompt: str) -> tuple[dict, str]:
    prompt_path = out_dir / "prompts" / f"{bench.name}_{idx:02d}.txt"
    spec_path = out_dir / "tokens" / f"{bench.name}_{idx:02d}.spec.bin"
    chain_path = out_dir / "tokens" / f"{bench.name}_{idx:02d}.chain.bin"
    log_path = out_dir / "logs" / f"{bench.name}_{idx:02d}.log"
    prompt_path.write_text(prompt, encoding="utf-8")

    cmd = [
        args.binary,
        "--target-model", args.target_model,
        "--draft-model", args.draft_model,
        "--prompt-text", str(prompt_path),
        "--gen", str(args.gen),
        "--out-spec", str(spec_path),
        "--out-chain", str(chain_path),
        "--ddtree-budget", str(args.budget),
        "--ddtree-top-k", str(args.top_k),
        "--require-full-prompt-ingest",
        "--temp", "0",
        "--n-gpu-layers", str(args.n_gpu_layers),
        "--draft-gpu-layers", str(args.draft_gpu_layers),
        "--n-ctx", str(args.n_ctx),
        "--n-batch", str(args.n_batch),
        "--n-ubatch", str(args.n_ubatch),
        "--kv-type", args.kv_type,
    ]
    env = os.environ.copy()
    env.update({
        "LLAMA_DDTREE_PROFILE": "1",
        "LLAMA_DDTREE_VERIFIER": args.verifier,
        "LLAMA_DDTREE_TARGET_FEAT_CTX": str(args.target_feat_ctx),
        "LLAMA_DDTREE_PROPOSAL_TEMP": str(args.proposal_temp),
    })
    if args.exact_validation:
        env["LLAMA_DDTREE_EXACT_VALIDATION"] = "1"
    if args.grammar_verify:
        env["LLAMA_DDTREE_GRAMMAR_VERIFY"] = "1"
    proc = subprocess.run(
        cmd,
        cwd=Path(args.binary).resolve().parent.parent.parent if not Path(args.binary).is_absolute() else None,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=args.timeout_sec,
    )
    log_path.write_text(proc.stdout, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        parsed = parse_e2e_output(proc.stdout)
        if parsed["chain_n"] and parsed["spec_n"]:
            parsed["harness_returncode"] = proc.returncode
            return parsed, str(log_path)
        raise RuntimeError(f"{bench.name} #{idx + 1} failed with exit {proc.returncode}; log={log_path}")
    parsed = parse_e2e_output(proc.stdout)
    parsed["harness_returncode"] = proc.returncode
    return parsed, str(log_path)


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def write_outputs(out_dir: Path, rows: list[dict], args: argparse.Namespace) -> None:
    json_path = out_dir / "results.json"
    csv_path = out_dir / "results.csv"
    md_path = out_dir / "summary.md"
    json_path.write_text(json.dumps({"args": vars(args), "rows": rows}, indent=2), encoding="utf-8")

    if rows:
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    lines = [
        "# llama.cpp DFlash Dataset Bench",
        "",
        f"- gen: {args.gen}",
        f"- budget/top_k: {args.budget}/{args.top_k}",
        f"- verifier: {args.verifier}",
        f"- ctx/batch/ubatch: {args.n_ctx}/{args.n_batch}/{args.n_ubatch}",
        f"- kv: {args.kv_type}",
        "",
        "| dataset | n | AR tok/s | DFlash tok/s | AL | speedup | bit-equal |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in sorted({r["dataset"] for r in rows}):
        ds = [r for r in rows if r["dataset"] == dataset and r.get("chain_decode_tps", 0) > 0]
        if not ds:
            continue
        lines.append(
            f"| {dataset} | {len(ds)} | "
            f"{mean([r['chain_decode_tps'] for r in ds]):.2f} | "
            f"{mean([r['spec_decode_tps'] for r in ds]):.2f} | "
            f"{mean([r['spec_acceptance'] for r in ds]):.2f} | "
            f"{mean([r['speedup_decode'] for r in ds]):.2f}x | "
            f"{sum(1 for r in ds if r['bit_equal_pass'])}/{len(ds)} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    require_file(args.binary, "test-speculative-tree-e2e binary")
    require_file(args.target_model, "target model")
    require_file(args.draft_model, "draft model")

    if not args.out_dir:
        args.out_dir = f"/tmp/llamacpp_dflash_dataset_bench_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(args.out_dir)
    for sub in ("prompts", "tokens", "logs"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer()
    rows: list[dict] = []
    result_path = out_dir / "results.json"
    if args.resume and result_path.is_file():
        rows = json.loads(result_path.read_text(encoding="utf-8"))["rows"]

    done = {(r["dataset"], r["sample_index"]) for r in rows if r.get("status") == "ok"}
    print(f"[bench] out_dir={out_dir}", flush=True)

    for bench in selected_benches(args.datasets):
        print(f"[bench] loading {bench.name}", flush=True)
        samples = load_samples(bench, args.n_sample, args.seed)
        for idx, sample in enumerate(samples):
            if (bench.name, idx) in done:
                continue
            prompt = bench.extract(sample)
            n_prompt = token_count(tokenizer, prompt)
            base = {
                "dataset": bench.name,
                "sample_index": idx,
                "prompt_tokens_est": n_prompt,
            }
            if n_prompt > args.skip_longer_than >= 0:
                row = {**base, "status": "skipped_long_prompt"}
                rows.append(row)
                write_outputs(out_dir, rows, args)
                print(f"[bench] {bench.name} {idx + 1:02d}: skipped n_prompt={n_prompt}", flush=True)
                continue
            print(f"[bench] {bench.name} {idx + 1:02d}/{len(samples)} n_prompt={n_prompt}", flush=True)
            try:
                parsed, log_path = run_one(args, out_dir, bench, idx, prompt)
                status = "ok" if parsed.get("harness_returncode") == 0 else "failed_harness"
                row = {**base, "status": status, "log": log_path, **parsed}
                print(
                    f"[bench]   AR={row['chain_decode_tps']:.2f} "
                    f"DFlash={row['spec_decode_tps']:.2f} "
                    f"AL={row['spec_acceptance']:.2f} "
                    f"speedup={row['speedup_decode']:.2f}x "
                    f"bit_equal={row['bit_equal_pass']}",
                    flush=True,
                )
            except Exception as exc:
                row = {**base, "status": "failed", "error": str(exc)}
                print(f"[bench]   FAILED: {exc}", flush=True)
            rows.append(row)
            write_outputs(out_dir, rows, args)

    write_outputs(out_dir, rows, args)
    print((out_dir / "summary.md").read_text(encoding="utf-8"), flush=True)
    return 0 if all(r.get("status") in {"ok", "skipped_long_prompt"} for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Batch benchmark runner for tree training methods across multiple .pt files.

Distributes .pt files across available GPUs (one process per GPU).
Each process benchmarks its assigned files sequentially and records
metrics to a shared JSONL output.

Output JSONL fields per file:
    pt_path, throughput (tok/s), method, peak_memory_gb, compressed_ratio

Usage:
    python areal/tests/bench_tree_training.py \\
        --data-dir /path/to/pt_files \\
        --method flex \\
        --output results.jsonl

    # All conftest-compatible options:
    python areal/tests/bench_tree_training.py \\
        --data-dir /path/to/pt_files \\
        --method stack \\
        --output results.jsonl \\
        --model-path /path/to/model \\
        --max-tokens-per-mb 24576 \\
        --prefix-len 10 \\
        --disable-dfn-mask \\
        --num-gpus 4
"""

import argparse
import fcntl
import gc
import json
import os
import time
from pathlib import Path

import torch
import torch.multiprocessing as mp

from areal.engine.fsdp_engine import FSDPEngine
from areal.models.tree_attn.tree import (
    _build_tries_from_trie_partition,
    _greedy_build_tries,
)
from areal.platforms import current_platform
from areal.tests.test_tree_training import (
    _build_input_from_token_lists,
    get_memory_stats,
    loss_fn,
    loss_weight_fn,
    reset_peak_memory,
    run_train_batch,
    setup_engine,
)
from areal.tests.utils import get_model_path


def _append_jsonl(output_path: str, record: dict):
    """Append a single JSON record to a JSONL file with exclusive file locking.

    Uses fcntl.flock so concurrent writers (this script's workers or
    external processes) never interleave partial lines.
    """
    with open(output_path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _compute_compression_ratio(input_data: dict) -> float:
    """Compute theoretical compression ratio from trie prefix sharing."""
    attn_mask = input_data["attention_mask"]
    seq_lens = attn_mask.sum(dim=1).cpu().tolist()
    total_original_tokens = sum(int(s) for s in seq_lens)
    if total_original_tokens == 0:
        return float("inf")
    tries_single, n_tokens_single = _greedy_build_tries(
        input_data, max_tokens_per_tree=total_original_tokens + 1
    )
    tree_tokens_single = sum(n_tokens_single)
    if tree_tokens_single == 0:
        return float("inf")
    return total_original_tokens / tree_tokens_single


def _compute_mb_tree_stats(
    input_data: dict,
    max_tokens_per_mb: int,
    use_trie_partition: bool,
) -> dict:
    """Compute tree token stats after microbatch partitioning.

    Splitting into microbatches duplicates shared prefixes at partition
    boundaries, so the sum of per-microbatch tree tokens is typically
    larger than a single-tree count.  This function replicates the
    partitioning logic to report the actual numbers.

    Returns dict with keys: mb_tree_tokens, mb_n_trees, mb_compressed_ratio.
    """
    attn_mask = input_data["attention_mask"]
    total_original = int(attn_mask.sum().item())
    if total_original == 0:
        return {"mb_tree_tokens": 0, "mb_n_trees": 0, "mb_compressed_ratio": float("inf")}

    if use_trie_partition:
        tries, n_tokens = _build_tries_from_trie_partition(
            input_data, max_tokens_per_tree=max_tokens_per_mb
        )
    else:
        tries, n_tokens = _greedy_build_tries(
            input_data, max_tokens_per_tree=max_tokens_per_mb
        )

    mb_tree_tokens = sum(n_tokens)
    mb_ratio = total_original / mb_tree_tokens if mb_tree_tokens > 0 else float("inf")
    return {
        "mb_tree_tokens": mb_tree_tokens,
        "mb_n_trees": len(tries),
        "mb_compressed_ratio": round(mb_ratio, 4),
    }


def _load_tree_data(
    data_path: str,
    prefix_len: int = -1,
) -> tuple[dict, int, int]:
    """Load a .pt file and prepare input data for benchmarking.

    Returns:
        (input_data dict, total_tokens, n_seqs)
    """
    raw = torch.load(data_path, weights_only=False)
    device = current_platform.device_type
    device_obj = device if isinstance(device, torch.device) else torch.device(device)

    if isinstance(raw, dict) and "input_data" in raw:
        input_data = raw["input_data"]
        result = {}
        for field_name, value in input_data.items():
            if isinstance(value, torch.Tensor):
                if prefix_len != -1 and value.size(0) >= prefix_len:
                    result[field_name] = value[:prefix_len].to(device_obj)
                else:
                    result[field_name] = value.to(device_obj)
            else:
                result[field_name] = value
        attn_mask = result.get("attention_mask")
        total_tokens = int(attn_mask.sum().item()) if attn_mask is not None else 0
        n_seqs = attn_mask.shape[0] if attn_mask is not None else 0
        return result, total_tokens, n_seqs

    if isinstance(raw, (list, tuple)) and len(raw) > 0 and torch.is_tensor(raw[0]):
        seqs = list(raw)
        if prefix_len != -1 and len(seqs) > prefix_len:
            seqs = seqs[:prefix_len]
        total_tokens = sum(t.numel() for t in seqs)
        n_seqs = len(seqs)
        input_data, _ = _build_input_from_token_lists(seqs, device_obj)
        return input_data, total_tokens, n_seqs

    raise ValueError(
        f"Unrecognised data format in {data_path}: "
        f"expected dict with 'input_data' or list[Tensor], got {type(raw)}"
    )


def run_single_benchmark(
    data_path: str,
    model_path: str,
    method: str,
    max_tokens_per_mb: int,
    gradient_checkpointing: bool,
    prefix_len: int,
    use_dfn_mask: bool,
    use_trie_partition: bool,
    cut_f1_tail: bool,
    master_port: str,
    gpu_id: int = 0,
) -> dict:
    """Run benchmark for a single .pt file. Returns a result dict."""
    input_data, total_tokens, n_seqs = _load_tree_data(data_path, prefix_len)
    compression_ratio = _compute_compression_ratio(input_data)

    enable_tree = method == "flex"
    enable_stack = method == "stack"
    gc_flag = gradient_checkpointing if method != "stack" else False
    dfn = use_dfn_mask if method == "flex" else False
    trie_part = use_trie_partition if method == "flex" else False

    # Compute microbatch-level tree stats for flex (prefix duplication from splitting)
    mb_stats: dict = {}
    if enable_tree:
        mb_stats = _compute_mb_tree_stats(input_data, max_tokens_per_mb, trie_part)

    gc.collect()
    torch.cuda.empty_cache()
    reset_peak_memory()

    assert "input_ids" in input_data, "input_ids not found in input_data"
    with setup_engine(
        FSDPEngine,
        experiment_name=f"bench_{method}",
        master_port=master_port,
        max_tokens_per_mb=max_tokens_per_mb,
        gradient_checkpointing=gc_flag,
        enable_tree_training=enable_tree,
        enable_tree_stack_training=enable_stack,
        model_path=model_path,
        use_dfn_mask=dfn,
        use_trie_partition=trie_part,
        cut_f1_tail=cut_f1_tail,
        local_rank=gpu_id,
    ) as engine:
        _, elapsed = run_train_batch(engine, input_data, loss_fn, loss_weight_fn)
        mem_stats = get_memory_stats(f"bench_{method}")

    throughput = total_tokens / elapsed if elapsed > 0 else 0.0

    result = {
        "pt_path": str(data_path),
        "method": method,
        "throughput": round(throughput, 2),
        "peak_memory_gb": round(mem_stats["peak_allocated_gb"], 4),
        "compressed_ratio": round(compression_ratio, 4),
        "elapsed_s": round(elapsed, 4),
        "n_seqs": n_seqs,
        "total_tokens": total_tokens,
    }
    result.update(mb_stats)
    return result


def _worker(gpu_id: int, pt_files: list[str], args, result_queue):
    """Worker process: benchmarks assigned .pt files on a single GPU.

    For flex method, a warmup run is performed on the first file to
    trigger torch.compile / flex_attention kernel compilation before
    any timed benchmarks.
    """
    torch.cuda.set_device(gpu_id)

    resolved_model_path = get_model_path(
        args.model_path or "/data/jiarui/dta/models/Qwen2.5-0.5B",
        "Qwen/Qwen2-0.5B",
    )

    base_port = 7800 + gpu_id * 100
    results = []

    # --- Warmup for flex (triggers torch.compile / flex_attention compilation) ---
    if args.method == "flex" and pt_files:
        warmup_file = pt_files[0]
        print(f"[GPU {gpu_id}] Warmup: {Path(warmup_file).name} (compiling flex kernels)...")
        t_warm = time.time()
        try:
            run_single_benchmark(
                data_path=warmup_file,
                model_path=resolved_model_path,
                method=args.method,
                max_tokens_per_mb=args.max_tokens_per_mb,
                gradient_checkpointing=not args.disable_gradient_checkpointing,
                prefix_len=args.prefix_len,
                use_dfn_mask=not args.disable_dfn_mask,
                use_trie_partition=args.use_trie_partition,
                cut_f1_tail=not args.no_cut_f1_tail,
                master_port=str(base_port + 99),
                gpu_id=gpu_id,
            )
        except Exception as e:
            print(f"[GPU {gpu_id}] Warmup failed (non-fatal): {e}")
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[GPU {gpu_id}] Warmup done ({time.time() - t_warm:.1f}s)")

    # --- Actual benchmarks ---
    for i, pt_path in enumerate(pt_files):
        port = str(base_port + (i % 50))
        t0 = time.time()
        try:
            result = run_single_benchmark(
                data_path=pt_path,
                model_path=resolved_model_path,
                method=args.method,
                max_tokens_per_mb=args.max_tokens_per_mb,
                gradient_checkpointing=not args.disable_gradient_checkpointing,
                prefix_len=args.prefix_len,
                use_dfn_mask=not args.disable_dfn_mask,
                use_trie_partition=args.use_trie_partition,
                cut_f1_tail=not args.no_cut_f1_tail,
                master_port=port,
                gpu_id=gpu_id,
            )
            results.append(result)
            _append_jsonl(args.output, result)
            mb_info = ""
            if "mb_compressed_ratio" in result:
                mb_info = (
                    f", mb_CR={result['mb_compressed_ratio']:.3f}x"
                    f" ({result['mb_n_trees']} trees, {result['mb_tree_tokens']} tok)"
                )
            print(
                f"[GPU {gpu_id}] {i + 1}/{len(pt_files)} done: "
                f"{Path(pt_path).name} — "
                f"{result['throughput']:.0f} tok/s, "
                f"{result['peak_memory_gb']:.2f} GB, "
                f"CR={result['compressed_ratio']:.3f}x"
                f"{mb_info} "
                f"({time.time() - t0:.1f}s)"
            )
        except Exception as e:
            import traceback

            err_record = {
                "pt_path": str(pt_path),
                "method": args.method,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
            results.append(err_record)
            _append_jsonl(args.output, err_record)
            print(f"[GPU {gpu_id}] ERROR: {Path(pt_path).name} — {e}")
            gc.collect()
            torch.cuda.empty_cache()

    result_queue.put(results)


def main():
    parser = argparse.ArgumentParser(
        description="Batch benchmark runner for tree training methods",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir", required=True, help="Directory containing .pt files"
    )
    parser.add_argument(
        "--method",
        required=True,
        choices=["baseline", "flex", "stack"],
        help="Training method to benchmark",
    )
    parser.add_argument("--output", required=True, help="Output JSONL file path")
    parser.add_argument(
        "--model-path", default=None, help="Model cgheckpoint path (default: Qwen2.5-0.5B)"
    )
    parser.add_argument("--max-tokens-per-mb", type=int, default=24576)
    parser.add_argument(
        "--disable-gradient-checkpointing", action="store_true", default=False
    )
    parser.add_argument("--prefix-len", type=int, default=-1)
    parser.add_argument("--disable-dfn-mask", action="store_true", default=False)
    parser.add_argument(
        "--use-trie-partition",
        action="store_true",
        default=False,
        help="Use TokenTrie-based partitioning (backward_permute + divide) "
        "instead of greedy first-fit for microbatch composition",
    )
    parser.add_argument(
        "--no-cut-f1-tail",
        action="store_true",
        default=False,
        help="Disable cut_f1_tail optimisation in tree stack training. "
        "When set, the entire pushed segment is cached instead of only "
        "the prefix needed for the next pop (useful for ablation)",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="Number of GPUs (default: all available)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    pt_files = sorted(data_dir.glob("*.pt"))
    if not pt_files:
        print(f"No .pt files found in {data_dir}")
        return

    num_gpus = args.num_gpus or torch.cuda.device_count()
    print(f"[Debug] num_gpus: {num_gpus}")
    if num_gpus <= 0:
        print("No GPUs available")
        return

    print(f"Found {len(pt_files)} .pt files in {data_dir}")
    print(f"Method: {args.method}, GPUs: {num_gpus}")
    print(f"max_tokens_per_mb={args.max_tokens_per_mb}, "
          f"dfn_mask={'OFF' if args.disable_dfn_mask else 'ON'}, "
          f"trie_partition={'ON' if args.use_trie_partition else 'OFF'}, "
          f"gradient_ckpt={'OFF' if args.disable_gradient_checkpointing else 'ON'}, "
          f"cut_f1_tail={'OFF' if args.no_cut_f1_tail else 'ON'}, "
          f"prefix_len={args.prefix_len}")

    chunks: list[list[str]] = [[] for _ in range(num_gpus)]
    for i, f in enumerate(pt_files):
        chunks[i % num_gpus].append(str(f))

    for gpu_id in range(num_gpus):
        print(f"  GPU {gpu_id}: {len(chunks[gpu_id])} files")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.touch(exist_ok=True)

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    processes = []

    for gpu_id in range(num_gpus):
        if not chunks[gpu_id]:
            continue
        p = ctx.Process(
            target=_worker,
            args=(gpu_id, chunks[gpu_id], args, result_queue),
        )
        p.start()
        processes.append(p)

    all_results = []
    for _ in processes:
        worker_results = result_queue.get(timeout=7200)
        all_results.extend(worker_results)

    for p in processes:
        p.join(timeout=120)
        if p.is_alive():
            p.terminate()
            p.join()

    print(f"\nResults written to {output_path} ({len(all_results)} records)")

    successful = [r for r in all_results if "error" not in r]
    failed = [r for r in all_results if "error" in r]
    if successful:
        avg_tp = sum(r["throughput"] for r in successful) / len(successful)
        avg_mem = sum(r["peak_memory_gb"] for r in successful) / len(successful)
        avg_cr = sum(r["compressed_ratio"] for r in successful) / len(successful)
        print(f"\nSummary ({len(successful)} succeeded, {len(failed)} failed):")
        print(f"  Avg throughput:     {avg_tp:,.0f} tok/s")
        print(f"  Avg peak memory:    {avg_mem:.2f} GB")
        print(f"  Avg compression:    {avg_cr:.3f}x")
        with_mb = [r for r in successful if "mb_compressed_ratio" in r]
        if with_mb:
            avg_mb_cr = sum(r["mb_compressed_ratio"] for r in with_mb) / len(with_mb)
            avg_mb_trees = sum(r["mb_n_trees"] for r in with_mb) / len(with_mb)
            avg_mb_tok = sum(r["mb_tree_tokens"] for r in with_mb) / len(with_mb)
            print(f"  Avg mb compression: {avg_mb_cr:.3f}x "
                  f"(avg {avg_mb_trees:.1f} trees, {avg_mb_tok:,.0f} tree tok)")
    if failed:
        print(f"\nFailed files ({len(failed)}):")
        for r in failed:
            print(f"  {r['pt_path']}: {r['error']}")


if __name__ == "__main__":
    main()

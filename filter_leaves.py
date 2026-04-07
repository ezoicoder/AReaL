#!/usr/bin/env python3
"""Filter list[Tensor] .pt files to keep only leaf (non-prefix) sequences.

A "leaf" sequence is one that is NOT a strict prefix of any other sequence
in the same file.  Uses ``TokenTrie`` for efficient trie-based detection.

Usage
-----
    python DynamicTreeAttn/filter_leaves.py \\
        --input-dir  DynamicTreeAttn/data \\
        --output-dir DynamicTreeAttn/data_leaves

    # dry-run (print stats without writing):
    python DynamicTreeAttn/filter_leaves.py \\
        --input-dir DynamicTreeAttn/data --dry-run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from areal.models.tree_attn.token_trie import TokenTrie


def longest_common_prefix_len(sequences: list[torch.Tensor]) -> int:
    """Return longest common prefix length shared by all sequences."""
    if not sequences:
        return 0
    min_len = min(int(seq.numel()) for seq in sequences)
    if min_len == 0:
        return 0

    for i in range(min_len):
        token = sequences[0][i]
        if any(seq[i] != token for seq in sequences[1:]):
            return i
    return min_len


def strip_common_prefix(
    sequences: list[torch.Tensor],
) -> tuple[list[torch.Tensor], int]:
    """Strip longest common prefix from all sequences and return prefix length."""
    prefix_len = longest_common_prefix_len(sequences)
    if prefix_len == 0:
        return list(sequences), 0
    return [seq[prefix_len:] for seq in sequences], prefix_len


def filter_leaves(sequences: list[torch.Tensor]) -> list[torch.Tensor]:
    """Return only the leaf sequences (drop strict-prefix duplicates).

    A sequence is a "leaf" iff it is not a strict prefix of another sequence
    in the list.  When multiple sequences are identical, only one copy is kept.

    The returned list preserves the original order of the surviving sequences.
    """
    if len(sequences) <= 1:
        return list(sequences)

    trie = TokenTrie(sequences)

    leaf_indices: set[int] = set()
    for attach_list in trie.attach_lists:
        leaf_indices.add(attach_list[-1][0]["_sequence_batch_id"])

    return [sequences[i] for i in range(len(sequences)) if i in leaf_indices]


def _compression_ratio(sequences: list[torch.Tensor]) -> tuple[int, int, float]:
    """Build a TokenTrie and return (n_tokens, n_tree_tokens, ratio)."""
    if not sequences:
        return 0, 0, 0.0
    trie = TokenTrie(sequences)
    ratio = trie.n_tokens / trie.n_tree_tokens if trie.n_tree_tokens > 0 else float("inf")
    return trie.n_tokens, trie.n_tree_tokens, ratio


def _attention_compression_ratio(sequences: list[torch.Tensor]) -> tuple[int, int, float]:
    """Return (dense_attn_ops, tree_attn_ops, dense/tree ratio) for sequences."""
    if not sequences:
        return 0, 0, 0.0

    dense_attn_ops = sum((L * (L + 1)) // 2 for L in (int(seq.numel()) for seq in sequences))
    trie = TokenTrie(sequences)
    tree_attn_ops = int(trie.get_stats("forward")["sum_depth"])
    ratio = (
        dense_attn_ops / tree_attn_ops
        if tree_attn_ops > 0
        else (float("inf") if dense_attn_ops > 0 else 0.0)
    )
    return dense_attn_ops, tree_attn_ops, ratio


def process_file(
    src: Path,
    dst: Path | None,
    *,
    dry_run: bool = False,
    strip_common_prefix_flag: bool = False,
) -> dict:
    """Load a .pt file, filter leaves, optionally save, and return stats."""
    raw = torch.load(src, weights_only=False)

    if not (isinstance(raw, (list, tuple)) and len(raw) > 0 and torch.is_tensor(raw[0])):
        return {"skipped": True, "reason": f"not list[Tensor] (got {type(raw).__name__})"}

    sequences: list[torch.Tensor] = list(raw)
    n_original = len(sequences)
    original_tokens = sum(t.numel() for t in sequences)
    orig_max_len = max((int(t.numel()) for t in sequences), default=0)

    orig_total, orig_tree, orig_ratio = _compression_ratio(sequences)

    leaves = filter_leaves(sequences)
    n_leaves = len(leaves)

    common_prefix_len = 0
    output_sequences = leaves
    if strip_common_prefix_flag:
        output_sequences, common_prefix_len = strip_common_prefix(leaves)

    output_tokens = sum(t.numel() for t in output_sequences)
    # `leaf_max_len` is defined on the final output stage.
    # If --strip-common-prefix is enabled, this is strip后的最大长度.
    leaf_max_len = max((int(t.numel()) for t in output_sequences), default=0)
    output_total, output_tree, output_ratio = _compression_ratio(output_sequences)
    output_attn_dense, output_attn_tree, output_attn_ratio = _attention_compression_ratio(
        output_sequences
    )
    avg_output_len = output_tokens / n_leaves if n_leaves > 0 else 0.0

    if not dry_run and dst is not None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        torch.save(output_sequences, dst)

    return {
        "skipped": False,
        "n_original": n_original,
        "n_leaves": n_leaves,
        "n_removed": n_original - n_leaves,
        "original_tokens": original_tokens,
        "orig_max_len": orig_max_len,
        "output_tokens": output_tokens,
        "leaf_max_len": leaf_max_len,
        "removed_tokens": original_tokens - output_tokens,
        "orig_tree_tokens": orig_tree,
        "orig_ratio": orig_ratio,
        "output_tree_tokens": output_tree,
        "output_ratio": output_ratio,
        "output_attn_dense": output_attn_dense,
        "output_attn_tree": output_attn_tree,
        "output_attn_ratio": output_attn_ratio,
        "avg_output_len": avg_output_len,
        "common_prefix_len": common_prefix_len,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter list[Tensor] .pt files to keep only leaf sequences."
    )
    parser.add_argument(
        "--input-dir", type=str, required=True, help="Directory with .pt files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (defaults to <input-dir>_leaves)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print stats without writing output files",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="*.pt",
        help="Glob pattern for input files (default: *.pt)",
    )
    parser.add_argument(
        "--strip-common-prefix",
        action="store_true",
        help="After leaf filtering, remove longest common prefix shared by all sequences",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"Error: {input_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else input_dir.with_name(input_dir.name + "_leaves")

    pt_files = sorted(input_dir.glob(args.glob))
    if not pt_files:
        print(f"No files matching '{args.glob}' in {input_dir}", file=sys.stderr)
        sys.exit(1)

    total_original = 0
    total_leaves = 0
    total_original_tokens = 0
    total_output_tokens = 0
    total_orig_tree_tokens = 0
    total_output_tree_tokens = 0
    total_output_attn_dense = 0
    total_output_attn_tree = 0
    total_common_prefix_len = 0
    total_orig_max_len = 0
    total_leaf_max_len = 0
    output_label = "Leaf+Strip" if args.strip_common_prefix else "Leaf"
    output_stage_label = (
        "leaves + common-prefix stripped"
        if args.strip_common_prefix
        else "leaves only"
    )

    header = (
        f"{'File':<20s} {'Orig':>5s} {'Leaf':>5s} {'Rm':>5s}"
        f" {'OrigTok':>10s} {'TreeTok':>10s} {'Ratio':>6s} {'OrigMax':>8s}"
        f" {output_label + 'Tok':>10s} {'TreeTok':>10s} {'Ratio':>6s} {'LeafMax':>8s} {'AvgOutLen':>10s}"
        f" {'AttnDense':>12s} {'AttnTree':>12s} {'AttnCR':>8s}"
    ) + (f"{' PrefixLen':>10s}" if args.strip_common_prefix else "")
    sub_header = (
        f"{'':<20s} {'':>5s} {'':>5s} {'':>5s}"
        f" {'--- original ---':^28s}"
        f" {f'--- {output_stage_label} ---':^75s}"
    )
    print(sub_header)
    print(header)
    print("-" * len(header))

    for pt_file in pt_files:
        dst = output_dir / pt_file.name if not args.dry_run else None
        stats = process_file(
            pt_file,
            dst,
            dry_run=args.dry_run,
            strip_common_prefix_flag=args.strip_common_prefix,
        )

        name = pt_file.name
        if stats["skipped"]:
            print(f"{name:<20s} SKIPPED: {stats['reason']}")
            continue

        total_original += stats["n_original"]
        total_leaves += stats["n_leaves"]
        total_original_tokens += stats["original_tokens"]
        total_output_tokens += stats["output_tokens"]
        total_orig_tree_tokens += stats["orig_tree_tokens"]
        total_output_tree_tokens += stats["output_tree_tokens"]
        total_output_attn_dense += stats["output_attn_dense"]
        total_output_attn_tree += stats["output_attn_tree"]
        total_common_prefix_len += stats["common_prefix_len"]
        total_orig_max_len = max(total_orig_max_len, int(stats["orig_max_len"]))
        total_leaf_max_len = max(total_leaf_max_len, int(stats["leaf_max_len"]))

        row = (
            f"{name:<20s} {stats['n_original']:>5d} {stats['n_leaves']:>5d} {stats['n_removed']:>5d}"
            f" {stats['original_tokens']:>10,d} {stats['orig_tree_tokens']:>10,d} {stats['orig_ratio']:>6.2f}x {stats['orig_max_len']:>8d}"
            f" {stats['output_tokens']:>10,d} {stats['output_tree_tokens']:>10,d} {stats['output_ratio']:>6.2f}x"
            f" {stats['leaf_max_len']:>8d} {stats['avg_output_len']:>10.2f}"
            f" {stats['output_attn_dense']:>12,d} {stats['output_attn_tree']:>12,d} {stats['output_attn_ratio']:>8.2f}x"
        ) + (f"{stats['common_prefix_len']:>10d}" if args.strip_common_prefix else "")
        print(row)

    print("-" * len(header))
    removed_seqs = total_original - total_leaves
    removed_tokens = total_original_tokens - total_output_tokens
    total_orig_ratio = (
        total_original_tokens / total_orig_tree_tokens if total_orig_tree_tokens else 0
    )
    total_output_ratio = (
        total_output_tokens / total_output_tree_tokens if total_output_tree_tokens else 0
    )
    total_output_attn_ratio = (
        total_output_attn_dense / total_output_attn_tree
        if total_output_attn_tree
        else (float("inf") if total_output_attn_dense > 0 else 0.0)
    )
    total_avg_output_len = total_output_tokens / total_leaves if total_leaves > 0 else 0.0
    total_row = (
        f"{'TOTAL':<20s} {total_original:>5d} {total_leaves:>5d} {removed_seqs:>5d}"
        f" {total_original_tokens:>10,d} {total_orig_tree_tokens:>10,d} {total_orig_ratio:>6.2f}x {total_orig_max_len:>8d}"
        f" {total_output_tokens:>10,d} {total_output_tree_tokens:>10,d} {total_output_ratio:>6.2f}x"
        f" {total_leaf_max_len:>8d} {total_avg_output_len:>10.2f}"
        f" {total_output_attn_dense:>12,d} {total_output_attn_tree:>12,d} {total_output_attn_ratio:>8.2f}x"
    ) + (f"{total_common_prefix_len:>10d}" if args.strip_common_prefix else "")
    print(total_row)

    if total_original > 0:
        pct_seqs = 100.0 * removed_seqs / total_original
        pct_toks = (
            100.0 * removed_tokens / total_original_tokens if total_original_tokens else 0
        )
        if args.strip_common_prefix:
            print(
                f"\nRemoved {pct_seqs:.1f}% sequences, {pct_toks:.1f}% tokens "
                "(strict-prefix duplicates + common-prefix stripping)"
            )
            print(
                f"Total stripped common-prefix length across files: {total_common_prefix_len} tokens"
            )
        else:
            print(
                f"\nRemoved {pct_seqs:.1f}% sequences, {pct_toks:.1f}% tokens "
                "(strict-prefix duplicates)"
            )
        print(
            f"Trie compression: original {total_orig_ratio:.2f}x -> output {total_output_ratio:.2f}x"
        )
        print(
            "Filtered attention compression (dense/tree depth-sum): "
            f"{total_output_attn_ratio:.2f}x"
        )

    if not args.dry_run:
        print(f"\nOutput written to: {output_dir}")
    else:
        print("\n(dry-run — no files written)")


if __name__ == "__main__":
    main()

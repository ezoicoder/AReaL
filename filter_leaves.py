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
import os
import sys
from pathlib import Path

import torch

from areal.models.tree_attn.token_trie import TokenTrie


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


def process_file(src: Path, dst: Path | None, *, dry_run: bool = False) -> dict:
    """Load a .pt file, filter leaves, optionally save, and return stats."""
    raw = torch.load(src, weights_only=False)

    if not (isinstance(raw, (list, tuple)) and len(raw) > 0 and torch.is_tensor(raw[0])):
        return {"skipped": True, "reason": f"not list[Tensor] (got {type(raw).__name__})"}

    sequences: list[torch.Tensor] = list(raw)
    n_original = len(sequences)
    original_tokens = sum(t.numel() for t in sequences)

    orig_total, orig_tree, orig_ratio = _compression_ratio(sequences)

    leaves = filter_leaves(sequences)
    n_leaves = len(leaves)
    leaf_tokens = sum(t.numel() for t in leaves)

    leaf_total, leaf_tree, leaf_ratio = _compression_ratio(leaves)

    if not dry_run and dst is not None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        torch.save(leaves, dst)

    return {
        "skipped": False,
        "n_original": n_original,
        "n_leaves": n_leaves,
        "n_removed": n_original - n_leaves,
        "original_tokens": original_tokens,
        "leaf_tokens": leaf_tokens,
        "removed_tokens": original_tokens - leaf_tokens,
        "orig_tree_tokens": orig_tree,
        "orig_ratio": orig_ratio,
        "leaf_tree_tokens": leaf_tree,
        "leaf_ratio": leaf_ratio,
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
    total_leaf_tokens = 0
    total_orig_tree_tokens = 0
    total_leaf_tree_tokens = 0

    header = (
        f"{'File':<20s} {'Orig':>5s} {'Leaf':>5s} {'Rm':>5s}"
        f" {'OrigTok':>10s} {'TreeTok':>10s} {'Ratio':>6s}"
        f" {'LeafTok':>10s} {'TreeTok':>10s} {'Ratio':>6s}"
    )
    sub_header = (
        f"{'':<20s} {'':>5s} {'':>5s} {'':>5s}"
        f" {'--- original ---':^28s}"
        f" {'--- leaves only ---':^28s}"
    )
    print(sub_header)
    print(header)
    print("-" * len(header))

    for pt_file in pt_files:
        dst = output_dir / pt_file.name if not args.dry_run else None
        stats = process_file(pt_file, dst, dry_run=args.dry_run)

        name = pt_file.name
        if stats["skipped"]:
            print(f"{name:<20s} SKIPPED: {stats['reason']}")
            continue

        total_original += stats["n_original"]
        total_leaves += stats["n_leaves"]
        total_original_tokens += stats["original_tokens"]
        total_leaf_tokens += stats["leaf_tokens"]
        total_orig_tree_tokens += stats["orig_tree_tokens"]
        total_leaf_tree_tokens += stats["leaf_tree_tokens"]

        print(
            f"{name:<20s} {stats['n_original']:>5d} {stats['n_leaves']:>5d} {stats['n_removed']:>5d}"
            f" {stats['original_tokens']:>10,d} {stats['orig_tree_tokens']:>10,d} {stats['orig_ratio']:>6.2f}x"
            f" {stats['leaf_tokens']:>10,d} {stats['leaf_tree_tokens']:>10,d} {stats['leaf_ratio']:>6.2f}x"
        )

    print("-" * len(header))
    removed_seqs = total_original - total_leaves
    removed_tokens = total_original_tokens - total_leaf_tokens
    total_orig_ratio = total_original_tokens / total_orig_tree_tokens if total_orig_tree_tokens else 0
    total_leaf_ratio = total_leaf_tokens / total_leaf_tree_tokens if total_leaf_tree_tokens else 0
    print(
        f"{'TOTAL':<20s} {total_original:>5d} {total_leaves:>5d} {removed_seqs:>5d}"
        f" {total_original_tokens:>10,d} {total_orig_tree_tokens:>10,d} {total_orig_ratio:>6.2f}x"
        f" {total_leaf_tokens:>10,d} {total_leaf_tree_tokens:>10,d} {total_leaf_ratio:>6.2f}x"
    )

    if total_original > 0:
        pct_seqs = 100.0 * removed_seqs / total_original
        pct_toks = 100.0 * removed_tokens / total_original_tokens if total_original_tokens else 0
        print(f"\nRemoved {pct_seqs:.1f}% sequences, {pct_toks:.1f}% tokens (strict-prefix duplicates)")
        print(f"Trie compression: original {total_orig_ratio:.2f}x -> leaves-only {total_leaf_ratio:.2f}x")

    if not args.dry_run:
        print(f"\nOutput written to: {output_dir}")
    else:
        print("\n(dry-run — no files written)")


if __name__ == "__main__":
    main()

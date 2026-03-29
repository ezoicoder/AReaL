"""Compressed Trie with DFS-optimized traversal ordering.

Provides a lightweight compressed trie that computes optimal DFS orderings
for tree-stack forward and backward passes, minimizing KV cache stack depth
and improving memory efficiency.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from math import ceil
from typing import Optional


def _get_stats(
    lens: list[int],
    lcp_lens: list[int],
    mode: str,
    block_size: int | None = None,
) -> dict:
    """Compute statistics for a sequence of trie-ordered leaves.

    Returns a dict with keys like n_leaf_sequences, n_tree_tokens,
    sum_prefix_len, sum_depth, and (for backward) n_f1_tokens.
    """
    n_tree_tokens = sum(lens) - sum(lcp_lens)
    sum_depth = 0
    for i in range(len(lens)):
        start = lcp_lens[i - 1] if i > 0 else 0
        end = lens[i]
        sum_depth += (start + end - 1) * (end - start) // 2

    if mode == "forward":
        sum_prefix_len = sum(lcp_lens)
        return {
            "n_leaf_sequences": len(lens),
            "n_tree_tokens": n_tree_tokens,
            "sum_prefix_len": sum_prefix_len,
            "sum_depth": sum_depth,
        }

    if mode == "backward":
        sum_prefix_len = 0
        n_f1_tokens = 0
        for i in range(len(lens)):
            start = lcp_lens[i] if i < len(lcp_lens) else 0
            end = lens[i]
            pop_len = end - start
            f1_start = lcp_lens[i - 1] if i > 0 else 0

            if block_size is None or pop_len <= block_size:
                f1_end = lcp_lens[i] if i < len(lcp_lens) else 0
                sum_prefix_len += start
            else:
                n_blocks = ceil(pop_len / block_size)
                block_size_actual = ceil(pop_len / n_blocks)
                f1_end = end - block_size_actual
                for b in range(n_blocks):
                    pop_start = max(end - (b + 1) * block_size_actual, start)
                    sum_prefix_len += pop_start

            n_f1_tokens += max(f1_end - f1_start, 0)

        return {
            "n_leaf_sequences": len(lens),
            "n_tree_tokens": n_tree_tokens,
            "sum_prefix_len": sum_prefix_len,
            "sum_depth": sum_depth,
            "n_f1_tokens": n_f1_tokens,
        }

    raise ValueError(f"Unsupported mode: {mode}")


@dataclass(slots=True)
class CTNode:
    """Node in a compressed trie built from (lens, lcp_lens)."""

    depth: int = 0
    seq_id: int = -1  # -1 for internal nodes
    chain_tail_depth: int = 0
    child_ids: list[int] = field(default_factory=list)


class CompressedTrie:
    """Compressed trie that supports DFS-optimized traversal orderings.

    Built from pre-sorted sequence lengths and their pairwise LCP lengths.
    Provides forward-optimized, backward-optimized, and random orderings
    that minimize KV-cache stack depth during tree-stack training.
    """

    def __init__(self, lens: list[int], lcp_lens: list[int]):
        if len(lcp_lens) != len(lens) - 1:
            raise ValueError("len(lcp_lens) must equal len(lens) - 1")

        self.nodes: list[CTNode] = []
        self._build(lens, lcp_lens)

        self.lca_depth: int = 0
        self.order: list[int] | None = None
        self.lens: list[int] | None = None
        self.lcp_lens: list[int] | None = None

    def _new_node(self, depth: int, seq_id: int = -1) -> int:
        self.nodes.append(CTNode(depth=depth, seq_id=seq_id))
        return len(self.nodes) - 1

    def _build(self, lens: list[int], lcp_lens: list[int]) -> None:
        n_seqs = len(lens)
        root_id = self._new_node(depth=0, seq_id=-1)
        stack = [(root_id, 0)]
        nodes = self.nodes

        for seq_id in range(n_seqs):
            len_i = lens[seq_id]
            lcp = lcp_lens[seq_id - 1] if seq_id > 0 else 0

            if len(stack) >= 2:
                while stack[-2][1] > lcp:
                    child_id = stack.pop()[0]
                    parent_id = stack[-1][0]
                    nodes[parent_id].child_ids.append(child_id)

                child_id = stack.pop()[0]
                if stack[-1][1] < lcp:
                    lcp_node_id = self._new_node(depth=lcp, seq_id=-1)
                    stack.append((lcp_node_id, lcp))
                parent_id = stack[-1][0]
                nodes[parent_id].child_ids.append(child_id)
            else:
                if stack[-1][1] < lcp:
                    lcp_node_id = self._new_node(depth=lcp, seq_id=-1)
                    stack.append((lcp_node_id, lcp))

            stack.append(
                (self._new_node(depth=len_i, seq_id=seq_id), len_i)
            )

        while len(stack) >= 2:
            child_id = stack.pop()[0]
            parent_id = stack[-1][0]
            nodes[parent_id].child_ids.append(child_id)

    # ------------------------------------------------------------------
    # Chain-tail depth (used to pick DFS child ordering)
    # ------------------------------------------------------------------

    def _dfs_chain(self, node_id: int, child_order_func) -> None:
        node = self.nodes[node_id]
        if node.seq_id != -1:
            node.chain_tail_depth = node.depth
            return
        for cid in node.child_ids:
            self._dfs_chain(cid, child_order_func)
        ordered = child_order_func(node_id)
        node.chain_tail_depth = self.nodes[ordered[0]].chain_tail_depth

    # ------------------------------------------------------------------
    # DFS traversal helpers
    # ------------------------------------------------------------------

    def _dfs_get_lens(self, node_id: int, seq_set: set[int]) -> None:
        node = self.nodes[node_id]
        if node.seq_id != -1:
            if node.seq_id in seq_set:
                self.lens.append(node.depth)
                self.lcp_lens.append(self.lca_depth)
                self.lca_depth = node.depth
            return
        for cid in node.child_ids:
            self.lca_depth = min(self.lca_depth, node.depth)
            self._dfs_get_lens(cid, seq_set)

    def get_lens(self, seq_set: set[int]) -> tuple[list[int], list[int]]:
        """Get (lens, lcp_lens) for a subset of sequences."""
        self.lens = []
        self.lcp_lens = []
        self.lca_depth = 0
        self._dfs_get_lens(0, seq_set)
        return self.lens, self.lcp_lens[1:]

    def _dfs_get_order(self, node_id: int, child_order_func) -> None:
        node = self.nodes[node_id]
        if node.seq_id != -1:
            self.order.append(node.seq_id)
            self.lens.append(node.depth)
            self.lcp_lens.append(self.lca_depth)
            self.lca_depth = node.depth
            return
        ordered = child_order_func(node_id)
        for cid in ordered:
            self.lca_depth = min(self.lca_depth, node.depth)
            self._dfs_get_order(cid, child_order_func)

    def _get_order(self, child_order_func) -> None:
        self._dfs_chain(0, child_order_func)
        self.order = []
        self.lens = []
        self.lcp_lens = []
        self.lca_depth = 0
        self._dfs_get_order(0, child_order_func)

    # ------------------------------------------------------------------
    # Child ordering strategies
    # ------------------------------------------------------------------

    def _child_order_forward(self, node_id: int) -> list[int]:
        """Shortest chain-tail-depth first → minimises forward stack depth."""
        node = self.nodes[node_id]
        return sorted(
            node.child_ids,
            key=lambda cid: self.nodes[cid].chain_tail_depth,
        )

    def _child_order_backward(self, node_id: int) -> list[int]:
        """Leaves first, then shortest chain-tail-depth → optimises backward pops."""
        node = self.nodes[node_id]
        return sorted(
            node.child_ids,
            key=lambda cid: (
                1 if self.nodes[cid].child_ids else 0,
                self.nodes[cid].chain_tail_depth,
            ),
        )

    def _child_order_random(
        self, node_id: int, seed: int | None = None
    ) -> list[int]:
        node = self.nodes[node_id]
        ids = node.child_ids.copy()
        if seed is not None:
            random.Random(seed).shuffle(ids)
        else:
            random.shuffle(ids)
        return ids

    # ------------------------------------------------------------------
    # Public ordering APIs
    # ------------------------------------------------------------------

    def get_order_forward(self) -> tuple[list[int], list[int], list[int]]:
        """Return (order, lens, lcp_lens) optimised for forward pass."""
        self._get_order(self._child_order_forward)
        return self.order, self.lens, self.lcp_lens[1:]

    def get_order_backward(self) -> tuple[list[int], list[int], list[int]]:
        """Return (order, lens, lcp_lens) optimised for backward pass."""
        self._get_order(self._child_order_backward)
        return self.order[::-1], self.lens[::-1], self.lcp_lens[1:][::-1]

    def get_order_random(
        self, seed: int | None = None
    ) -> list[int]:
        """Return order with randomly shuffled children."""
        self._get_order(
            lambda nid: self._child_order_random(nid, seed)
        )
        return self.order


def _get_subtrie(
    trie: CompressedTrie, seq_set: set[int]
) -> CompressedTrie:
    """Extract a sub-trie containing only the given sequence IDs."""
    lens, lcp_lens = trie.get_lens(seq_set)
    return CompressedTrie(lens, lcp_lens)

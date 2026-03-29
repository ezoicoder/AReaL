from __future__ import annotations

from typing import Optional

import torch

from areal.models.tree_attn.compressed_trie import CompressedTrie, _get_stats
from areal.utils import logging

logger = logging.getLogger(__name__)


def _lcp_torch(a: torch.Tensor, b: torch.Tensor) -> int:
    """Compute the length of the longest common prefix of two 1D tensors."""
    L = min(a.numel(), b.numel())
    eq = a[:L] == b[:L]
    return L if eq.all() else int((~eq).to(torch.int32).argmax().item())


def _leafization(input_ids: list[torch.LongTensor], attachs: list[dict]):
    """Merge fully-overlapping prefixes and compute LCP lengths.

    Args:
        input_ids: Token tensors sorted in lexicographic order.
        attachs: Per-sequence attachment dicts.
    """
    lcp_lens = []
    for i in range(len(input_ids) - 1):
        seq_L, seq_R = input_ids[i], input_ids[i + 1]
        lcp = _lcp_torch(seq_L, seq_R)
        L = min(seq_L.numel(), seq_R.numel())
        if lcp < L and seq_L[lcp] > seq_R[lcp]:
            raise ValueError("Input_ids not sorted in lexicographic order.")
        lcp_lens.append(lcp)

    input_ids_leafed = []
    attach_lists = []
    lcp_lens_leafed = []

    fork = -1
    for i in range(len(input_ids)):
        if i == len(input_ids) - 1 or lcp_lens[i] < min(
            input_ids[i].numel(), input_ids[i + 1].numel()
        ):
            input_ids_leafed.append(input_ids[i])
            if i < len(input_ids) - 1:
                lcp_lens_leafed.append(lcp_lens[i])
            attach_list = []
            for k in range(fork + 1, i + 1):
                attach_list.append((attachs[k], input_ids[k].numel()))
            attach_lists.append(attach_list)
            fork = i

    return input_ids_leafed, attach_lists, lcp_lens_leafed


class TokenTrie:
    def __init__(
        self,
        inputs: list[torch.LongTensor],
        attachs: list[dict] | None = None,
        sorted: bool = False,
    ):
        if attachs is not None:
            assert len(inputs) == len(attachs), "Length of inputs and attachs must match."
        else:
            attachs = [{} for _ in range(len(inputs))]

        for seq_id in range(len(inputs)):
            attachs[seq_id]["_sequence_batch_id"] = seq_id

        # -------- sort by lexicographical order of input_ids --------
        if not sorted:
            pairs = list(zip(inputs, attachs))
            pairs.sort(key=lambda x: x[0].tolist())
            inputs_sorted = [p[0] for p in pairs]
            attachs_sorted = [p[1] for p in pairs]
        else:
            inputs_sorted, attachs_sorted = inputs, attachs

        # -------- leafization --------
        self.inputs, self.attach_lists, self.lcp_lens = _leafization(
            inputs_sorted, attachs_sorted
        )
        self.lens = [len(ids) for ids in self.inputs]

        # -------- stats --------
        self.n_sequences = len(inputs)
        self.n_tokens = sum(len(ids) for ids in inputs)
        self.max_seq_len = max(len(ids) for ids in inputs) if inputs else 0
        self.n_leafed_tokens = sum(self.lens)
        self.n_tree_tokens = self.n_leafed_tokens - sum(self.lcp_lens)

    def get_stats(self, mode: str, block_size: Optional[int] = None) -> dict:
        stats = _get_stats(self.lens, self.lcp_lens, mode, block_size)
        stats["n_sequences"] = self.n_sequences
        stats["n_tokens"] = self.n_tokens
        return stats

    def permute(self, order: list[int]) -> None:
        self.inputs = [self.inputs[i] for i in order]
        self.attach_lists = [self.attach_lists[i] for i in order]
        self.lens = [self.lens[i] for i in order]
        self.lcp_lens = [
            _lcp_torch(self.inputs[i], self.inputs[i + 1])
            for i in range(len(self.inputs) - 1)
        ]

    def forward_permute(self) -> None:
        if len(self.lens) <= 1:
            return
        compressed_trie = CompressedTrie(self.lens, self.lcp_lens)
        order, _, _ = compressed_trie.get_order_forward()
        self.permute(order)

    def backward_permute(self) -> None:
        if len(self.lens) <= 1:
            return
        compressed_trie = CompressedTrie(self.lens, self.lcp_lens)
        order, _, _ = compressed_trie.get_order_backward()
        self.permute(order)

    def random_permute(self) -> None:
        if len(self.lens) <= 1:
            return
        compressed_trie = CompressedTrie(self.lens, self.lcp_lens)
        order = compressed_trie.get_order_random()
        self.permute(order)

    # ------------------------------------------------------------------
    # Helpers used by fsdp_engine
    # ------------------------------------------------------------------

    def count_tokens_information(self) -> dict:
        return {
            "n_sequences": self.n_sequences,
            "n_tokens": self.n_tokens,
            "max_seq_len": self.max_seq_len,
            "n_leafed_tokens": self.n_leafed_tokens,
            "n_tree_tokens": self.n_tree_tokens,
            "compression_ratio": (
                self.n_tokens / self.n_tree_tokens if self.n_tree_tokens > 0 else None
            ),
            "leafed_compression_ratio": (
                self.n_leafed_tokens / self.n_tree_tokens
                if self.n_tree_tokens > 0
                else None
            ),
            "avg_compressed_length_leafed": (
                self.n_tree_tokens / len(self.inputs) if self.inputs else None
            ),
        }

    def try_devide(self, tree_token_limit: int) -> list[list[int]] | None:
        """Try to partition so each part stays within *tree_token_limit*."""
        divs = [0]
        cur_tree_tokens = self.lens[0]
        for i in range(1, len(self.lens)):
            new_tree_tokens = self.lens[i] - self.lcp_lens[i - 1]
            if cur_tree_tokens + new_tree_tokens > tree_token_limit:
                divs.append(i)
                cur_tree_tokens = self.lens[i]
                if cur_tree_tokens > tree_token_limit:
                    return None
            else:
                cur_tree_tokens += new_tree_tokens

        divs.append(len(self.lens))

        parts: list[list[int]] = []
        for i in range(len(divs) - 1):
            part: list[int] = []
            for j in range(divs[i], divs[i + 1]):
                for attachment in self.attach_lists[j]:
                    part.append(attachment[0]["_sequence_batch_id"])
            parts.append(part)

        return parts

    def divide(self, n_parts: int) -> list[list[int]]:
        """Partition into *n_parts* minimising max tree-token count."""
        assert n_parts > 0
        assert n_parts <= self.n_sequences

        L = max(self.n_tree_tokens // n_parts, self.max_seq_len)
        R = self.n_tree_tokens

        while L < R:
            mid = (L + R) // 2
            parts = self.try_devide(mid)
            if parts is not None and len(parts) <= n_parts:
                R = mid
            else:
                L = mid + 1

        parts = self.try_devide(R)
        if len(parts) < n_parts:
            logger.warning(
                "Could only produce %d parts (requested %d)",
                len(parts),
                n_parts,
            )
            pos = 0
            rem = n_parts - len(parts)
            for _ in range(rem):
                while len(parts[pos]) <= 1:
                    pos += 1
                parts.append([parts[pos].pop()])

        assert len(parts) == n_parts
        return parts

import torch
from typing import List, Union


def _lcp_torch(a: torch.Tensor, b: torch.Tensor) -> int:
    """
    Compute the length of the longest common prefix (LCP) of two 1D tensors.
    """
    L = min(a.numel(), b.numel())
    eq = a[:L] == b[:L]
    return L if eq.all() else int((~eq).to(torch.int32).argmax().item())


def _leafization(input_ids: List[torch.LongTensor], attachs: List[dict]):
    """
    Perform leafization on a list of token sequences.

    This function merges sequences that share identical prefixes and
    computes the LCP (longest common prefix) lengths between adjacent
    sequences. Only the longest sequence is kept for each fully overlapping
    prefix, while metadata (attachs) of merged sequences are grouped together.

    Args:
        input_ids:
            List of token tensors, sorted in lexicographic order.
        attachs:
            List of dictionaries containing loss or auxiliary configuration
            associated with each token sequence.

    Returns:
        input_ids_leafed:
            List of token tensors after leafization.
        attach_lists:
            For each leafed sequence, a list of (attach, sequence_length)
            tuples corresponding to the merged original sequences.
        lcp_lens_leafed:
            LCP lengths between adjacent leafed sequences.
    """

    # Compute LCP lengths between adjacent sequences and
    # verify lexicographic ordering.
    lcp_lens = []
    for i in range(len(input_ids) - 1):
        seq_L, seq_R = input_ids[i], input_ids[i + 1]
        lcp = _lcp_torch(seq_L, seq_R)
        L = min(seq_L.numel(), seq_R.numel())

        # Ensure lexicographic order: if two sequences differ at position lcp,
        # the left one must be smaller than the right one.
        if lcp < L and seq_L[lcp] > seq_R[lcp]:
            raise ValueError("Input_ids not sorted in lexicographic order.")

        lcp_lens.append(lcp)

    # Merge sequences with fully overlapping prefixes.
    # Only the longest sequence is retained.
    input_ids_leafed = []
    attach_lists = []
    lcp_lens_leafed = []

    fork = -1
    for i in range(len(input_ids)):
        if (
            i == len(input_ids) - 1
            or lcp_lens[i] < min(input_ids[i].numel(), input_ids[i + 1].numel())
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
    """
    TokenTrie is a prefix-sharing structure over token sequences.

    It organizes multiple token sequences into a trie-like representation
    by merging shared prefixes. This enables efficient storage and
    backward computation (e.g., log-probabilities or losses) that only
    need to be evaluated at branching points.
    """

    def __init__(
        self,
        inputs: List[torch.LongTensor],
        attachs: List[dict] | None = None,
        sorted: bool = False
    ):
        """
        Construct a TokenTrie from input token sequences.

        Args:
            inputs:
                List of token ID tensors.
            attachs:
                Optional list of dictionaries containing per-sequence
                auxiliary information (e.g., loss configuration).
            sorted:
                Whether the input sequences are already sorted in
                lexicographic order.
        """
        if attachs is not None:
            assert len(inputs) == len(attachs), \
                "Length of inputs and attachs must match."
        else:
            attachs = [{} for _ in range(len(inputs))]

        # Add sequence batch ID into attachs
        for seq_id in range(len(inputs)):
            attachs[seq_id]['_sequence_batch_id'] = seq_id

        # Sort by lexicographical order of input_ids
        if not sorted:
            pairs = list(zip(inputs, attachs))
            pairs.sort(key=lambda x: x[0].tolist())
            inputs_sorted = [p[0] for p in pairs]
            attachs_sorted = [p[1] for p in pairs]
        else:
            inputs_sorted, attachs_sorted = inputs, attachs

        # Leafization
        self.inputs, self.attach_lists, self.lcp_lens = \
            _leafization(inputs_sorted, attachs_sorted)

        # Statistics
        self.n_sequences = len(inputs)
        self.n_tokens = sum(len(ids) for ids in inputs)
        self.n_leafed_tokens = sum(len(ids) for ids in self.inputs)
        self.n_tree_tokens = self.n_leafed_tokens - sum(self.lcp_lens)
import torch
from transformers.cache_utils import DynamicCache
from typing import List, Optional, Tuple

import torch.nn.functional as F
from math import ceil
from bisect import bisect_left, bisect_right

from areal.utils.functional.vocab_parallel import gather_logprobs

def _get_forkpos(lcp_lens) -> set:
    """
    Compute all fork positions that TreeForwardEngine's Stack needs to track.

    Returns a sorted list of unique fork positions.
    """
    forkpos_list = []

    # Fork positions induced by branching (LCP boundaries)
    for lcp in lcp_lens:
        if lcp > 0:
            forkpos_list.append(lcp - 1)

    forkpos_list = list(set(forkpos_list))
    forkpos_list.sort()

    return forkpos_list

class TreeForwardEngine:
    """
    Engine for forward computation over sequences with shared prefixes.

    TreeForwardEngine stores only necessary KV caches, log-probs, and 
    logits at fork positions and to efficiently compute log-probs for
    multiple sequences while saving memory.
    """
    def __init__(self, model, dtype: torch.dtype, max_seq_len: int):
        """
        Initialize the engine with model, dtype, and maximum sequence length.

        Buffers for tokens, logprobs and KV caches are preallocated to max_seq_len.
        """
        self.model = model
        self.device = model.device
        self.dtype = dtype
        self.max_seq_len = max_seq_len

        # ------------------------------------------------------------------------
        # Initialize static stack buffers
        # ------------------------------------------------------------------------
        self.cur_len = 0
        
        self.tokens = torch.zeros((max_seq_len), device=self.device, dtype=torch.long)      # Token buffer
        self.logprobs = torch.zeros((max_seq_len), device=self.device, dtype=torch.float32) # Logprob buffer

        # Fork position logits buffer (store logits only at fork positions, others are None)
        self.forkpos_list = []                                                      # List of all fork positions
        self.forkpos_logits: List[Optional[torch.Tensor]] = [None] * max_seq_len    # Logits at fork positions for computing logprobs
        
        # KV cache buffers
        cfg = model.config
        self.n_layers = cfg.num_hidden_layers
        n_kv_heads = cfg.num_key_value_heads
        head_dim = cfg.hidden_size // cfg.num_attention_heads

        kv_buffer_shape = (1, n_kv_heads, max_seq_len, head_dim)
        
        self.kv_cache = (
            [
                torch.zeros(kv_buffer_shape, device=self.device, dtype=dtype)
                for _ in range(self.n_layers)
            ],
            [
                torch.zeros(kv_buffer_shape, device=self.device, dtype=dtype)
                for _ in range(self.n_layers)
            ],
        )

        self.ret_logprobs = []  # Store computed logprobs for each sequence

    def get_forkpos(self, start: int, end: int) -> List[int]:
        """
        Yield fork positions within the interval [start, end).

        Uses binary search on precomputed forkpos_list.
        """

        left = bisect_left(self.forkpos_list, start)
        right = bisect_right(self.forkpos_list, end - 1)
        yield from self.forkpos_list[left:right]

    @torch.no_grad()
    def push(
        self,
        new_tokens: torch.LongTensor,
        attach_list: List[Tuple[dict, int]],
    ):
        """
        Push new tokens into the stack with their attachments.

        Builds cache (KV, logprobs) up to cache_len.
        Updates logprobs for the previous token.
        """
        
        B = new_tokens.numel()
        assert self.cur_len + B <= self.max_seq_len, (
            f"Exceeds max_seq_len: cur_len={self.cur_len}, new_tokens={B}, max={self.max_seq_len}"
        )

        start, end = self.cur_len, self.cur_len + B

        # -------------------------------------------------------------
        # 1. Build prefix cache from existing KV
        # -------------------------------------------------------------
        prefix_cache = DynamicCache()
        for l in range(self.n_layers):
            prefix_cache.update(
                self.kv_cache[0][l][:, :, :start, :],
                self.kv_cache[1][l][:, :, :start, :],
                layer_idx=l,
            )

        # -------------------------------------------------------------
        # 2. Forward
        # -------------------------------------------------------------
        out = self.model(
            new_tokens.unsqueeze(0),
            past_key_values=prefix_cache,
            use_cache=True,
        )
        
        # Compute logprobs for new tokens
        logits = out.logits  # [1, B, vocab]
        logprobs = gather_logprobs(
            logits=logits,
            labels=new_tokens[1:].unsqueeze(0),
        )

        # -------------------------------------------------------------
        # 3. Write tokens, computed logprobs, and KV cache into stack
        # -------------------------------------------------------------

        # Write tokens into stack
        self.tokens[start:end] = new_tokens

        # Write logprobs into stack
        self.logprobs[start : end-1] = logprobs.squeeze(0)
        # Fill the logprob of the first token using self.forkpos_logits[start]
        if start > 0:   
            pre_logits = self.forkpos_logits[start-1].float()
            first_token = new_tokens[0].item()
            pre_logprob = F.log_softmax(pre_logits, dim=-1)[first_token].item()
            self.logprobs[start-1] = pre_logprob

        # Write KV cache into stack
        new_cache = out.past_key_values 
        for l, layer in enumerate(new_cache.layers):
            self.kv_cache[0][l][:, :, start:end, :] = layer.keys[:, :, start:end, :]
            self.kv_cache[1][l][:, :, start:end, :] = layer.values[:, :, start:end, :]

        # Write logits into stack (fork positions only)
        forkpos_slice = self.get_forkpos(start, end)
        for i in forkpos_slice:
            self.forkpos_logits[i] = logits[0, i - start].detach().clone()

        # -------------------------------------------------------------
        # 4. Store logprobs for sequences ending in attach_list
        # -------------------------------------------------------------
        for attachment, length in attach_list:
            seq_id = attachment['_sequence_batch_id']
            logprobs = self.logprobs[:length-1]
            self.returns[seq_id] = logprobs.clone()

        self.cur_len += B

    @torch.no_grad()
    def forward(self, token_trie):
        """
        Perform backward pass over all sequences in a TokenTrie.
        Compute logprobs for each sequence.
        The sequence ID is identified by attachment['_sequence_batch_id'], which TokenTrie automatically adds.

        Args:
            token_trie: TokenTrie containing input sequences and attachs.

        Returns:
            List of logprob tensors for each sequence in the TokenTrie.
        """

        self.returns = [None] * token_trie.n_sequences

        inputs, attach_lists, lcp_lens = token_trie.inputs, token_trie.attach_lists, token_trie.lcp_lens

        self.forkpos_list = _get_forkpos(lcp_lens)

        print(self.forkpos_list)

        for i in range(len(inputs)):
            input_ids = inputs[i].to(self.device)
            attach_list = attach_lists[i]
            seq_len = input_ids.size(0)

            # Pop diverged branch from previous sequence
            if i > 0:
                self.cur_len = lcp_lens[i - 1]

            # Push new tokens
            new_tokens = input_ids[self.cur_len :]
            self.push(new_tokens, attach_list)

        return self.returns
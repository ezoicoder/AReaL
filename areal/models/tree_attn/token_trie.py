import torch
from typing import List, Union

def _lcp_torch(a: torch.Tensor, b: torch.Tensor) -> int:
    """Compute the length of the longest common prefix of two 1D tensors."""
    L = min(a.numel(), b.numel())
    eq = a[:L] == b[:L]
    return L if eq.all() else int((~eq).to(torch.int32).argmax().item())

def _leafization(input_ids: List[torch.LongTensor], attachs: List[dict]):
    """
        参数：
            input_ids: List of Token Tensor（按字典序排序）
            attachs: List of dict，表示每个 Token Tensor 的 loss 配置

        将完全重叠的前缀合并，并计算 lcp_lens 列表。
    """

    # 计算相邻序列的 LCP 长度，同时检查字典序
    lcp_lens = []
    for i in range(len(input_ids)-1):
        seq_L, seq_R = input_ids[i], input_ids[i+1]
        lcp = _lcp_torch(seq_L, seq_R)
        L = min(seq_L.numel(), seq_R.numel())
        if lcp < L and seq_L[lcp] > seq_R[lcp]:
            raise ValueError("Input_ids not sorted in lexicographic order.")
        lcp_lens.append(lcp)

    # 合并完全重叠的前缀，计算时只保留最长序列
    input_ids_leafed = []
    attach_lists = []
    lcp_lens_leafed = []

    fork = -1
    for i in range(len(input_ids)):
        if i == len(input_ids)-1 or lcp_lens[i] < min(input_ids[i].numel(), input_ids[i+1].numel()):
            input_ids_leafed.append(input_ids[i])
            if i < len(input_ids)-1:
                lcp_lens_leafed.append(lcp_lens[i])
            attach_list = []
            for k in range(fork+1, i+1):
                attach_list.append((attachs[k], input_ids[k].numel()))
            attach_lists.append(attach_list)
            fork = i

    return input_ids_leafed, attach_lists, lcp_lens_leafed

class TokenTrie:
    def __init__(
        self,
        inputs: List[torch.LongTensor],
        attachs: List[dict] | None = None,
        sorted: bool = False,
        dtype: torch.dtype = None,
    ):
        if attachs is not None:
            assert len(inputs) == len(attachs), "Length of inputs and attachs must match."
        else:
            attachs = [{} for _ in range(len(inputs))]
        
        # 向 attachs 中添加序列编号
        for seq_id in range(len(inputs)):
            attachs[seq_id]['_sequence_batch_id'] = seq_id

        # -------- sort by lexicographical order of input_ids --------
        if not sorted:
            pairs = list(zip(inputs, attachs))
            pairs.sort(key=lambda x: x[0].tolist())
            inputs_sorted, attachs_sorted = [p[0] for p in pairs], [p[1] for p in pairs]
        else:
            inputs_sorted, attachs_sorted = inputs, attachs
            
        # -------- leafization --------
        self.inputs, self.attach_lists, self.lcp_lens = \
            _leafization(inputs_sorted, attachs_sorted)

        # -------- statistics --------
        self.n_sequences = len(inputs)
        self.n_tokens = sum(len(ids) for ids in inputs)
        self.max_seq_len = max(len(ids) for ids in inputs)
        self.n_leafed_tokens = sum(len(ids) for ids in self.inputs)
        self.n_tree_tokens = self.n_leafed_tokens - sum(self.lcp_lens)

        # Compute the token compression ratio (original tokens / tree tokens) and print it
        if self.n_tokens > 0:
            compression_ratio = self.n_tokens / self.n_tree_tokens
            print(f"[TokenTrie] Token compression ratio: {compression_ratio:.4f} ({self.n_tokens}/{self.n_tree_tokens})")
            print(f"[TokenTrie] Token compression ratio of leafed sequences: {self.n_leafed_tokens / self.n_tree_tokens:.4f} ({self.n_leafed_tokens}/{self.n_tree_tokens})")
            print(f"[TokenTrie] Average compressed length of leafed sequences {self.n_tree_tokens / len(self.inputs):.4f}")
        else:
            print("[TokenTrie] Warning: n_tokens is zero, cannot compute token compression ratio.")

    def try_devide(self, tree_token_limit: int) -> List[List[int]] | None:
        """
        Try to divide the sequences such that each part does not exceed
        the given tree_token_limit.
        
        If successful, return the division result (list of original 
        sequence IDs for each part); otherwise return None.
        """

        lens = [len(ids) for ids in self.inputs]
        divs = [0]

        cur_tree_tokens = lens[0]
        for i in range(1, len(lens)):
            new_tree_tokens = lens[i] - self.lcp_lens[i-1]
            if cur_tree_tokens + new_tree_tokens > tree_token_limit:
                divs.append(i)
                cur_tree_tokens = lens[i]
                if cur_tree_tokens > tree_token_limit:
                    return None
            else:
                cur_tree_tokens += new_tree_tokens

        divs.append(len(lens))
        
        parts = []
        for i in range(len(divs)-1):
            part = []
            for j in range(divs[i], divs[i+1]):
                for attachment in self.attach_lists[j]:
                    part.append(attachment[0]['_sequence_batch_id'])
            parts.append(part)
        
        return parts
                
    def divide(self, n_parts: int):
        """
        Divide the sequences into n_parts such that the maximum tree tokens
        in each part is minimized.
        """
        assert n_parts > 0, f"n_parts {n_parts} is not greater than 0"
        assert n_parts <= self.n_sequences, f"n_parts {n_parts} is not less than or equal to n_sequences {self.n_sequences}"

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
        print(parts)
        if len(parts) < n_parts:
            print(f"warning: len(parts) {len(parts)} is less than n_parts {n_parts}")
            pos = 0
            rem = n_parts - len(parts)
            for i in range(rem):
                while len(parts[pos]) <= 1: pos = pos +1
                last_elem = parts[pos].pop()
                parts.append([last_elem])

        assert len(parts) == n_parts, f"len(parts) {len(parts)} is not equal to n_parts {n_parts}"
        return parts
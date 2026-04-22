from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Protocol

import torch
from transformers import PretrainedConfig
from transformers.cache_utils import DynamicCache

from areal.experimental.dta.dta_engine import DTAEngine
from areal.experimental.dta.token_trie import TokenTrie
from areal.utils.data import extract_valid_token_sequences


class KVCacheModel(Protocol):
    """Structural contract for DTA-compatible models."""

    def forward(
        self,
        tokens: torch.LongTensor,
        past_key_values: DynamicCache | None = None,
        use_cache: bool = True,
    ) -> SimpleNamespace: ...


class DTAWrapper:
    """Engine-agnostic facade for DTA forward/backward paths."""

    def __init__(
        self,
        model: KVCacheModel,
        model_config: PretrainedConfig,
        device: torch.device,
        dtype: torch.dtype,
        max_seq_len: int,
        block_size: int,
    ) -> None:
        self.model = model
        self.device = device
        self.block_size = block_size
        self._engine = DTAEngine(
            model_config=model_config,
            device=device,
            dtype=dtype,
            max_seq_len=max_seq_len,
        )

    @torch.no_grad()
    def run_forward(
        self,
        input_ids_batch: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        input_ids_list, max_seq_len = extract_valid_token_sequences(
            input_ids_batch, attention_mask
        )
        input_data = [{} for _ in input_ids_list]
        trie = TokenTrie(input_ids_list, input_data, sorted=False)
        trie.forward_permute()

        output = self._engine.forward(model=self.model, token_trie=trie)
        batch_size = len(output)
        output_padded = torch.zeros(
            (batch_size, max_seq_len),
            dtype=output[0].dtype,
            device=output[0].device,
        )
        for i, seq in enumerate(output):
            seq_len = seq.shape[0]
            output_padded[i, :seq_len] = seq
        return output_padded

    def run_backward(
        self,
        input_ids_batch: torch.Tensor,
        attention_mask: torch.Tensor,
        per_seq_input_data: list[dict[str, Any]],
        loss_fn: Any,
        block_size: int | None = None,
    ) -> dict[str, float]:
        input_ids_list, _ = extract_valid_token_sequences(
            input_ids_batch, attention_mask
        )

        trie = TokenTrie(input_ids_list, per_seq_input_data, sorted=False)
        trie.backward_permute()

        total_loss = self._engine.backward(
            model=self.model,
            token_trie=trie,
            block_size=block_size or self.block_size,
            loss_fn=loss_fn,
        )
        return {"dta_loss": float(total_loss)}

    def run_backward_with_scaled_loss(
        self,
        input_ids_batch: torch.Tensor,
        attention_mask: torch.Tensor,
        mb_list: Any,
        prepare_mb_inputs_fn: Any,
        loss_fn: Any,
        loss_weight_fn: Any,
        total_loss_weight: torch.Tensor,
        block_size: int | None = None,
    ) -> dict[str, float]:
        per_seq_input_data: list[dict[str, Any]] = []
        for mb_item in mb_list:
            _, ctx = prepare_mb_inputs_fn(mb_item)
            loss_scale = loss_weight_fn(ctx.mb_input) / total_loss_weight
            if isinstance(loss_scale, torch.Tensor):
                loss_scale = loss_scale.item()
            per_seq_input_data.append({"original": ctx.mb_input, "scale": loss_scale})

        def scaled_loss_fn(
            logprobs: torch.Tensor,
            entropy: torch.Tensor,
            seq_input_data: dict[str, Any],
        ) -> torch.Tensor:
            # Keep current behavior: DTA engine expects one extra position.
            logprobs = torch.cat([logprobs, logprobs.new_zeros(1)], dim=0)
            return (
                loss_fn(logprobs, entropy, seq_input_data["original"])
                * seq_input_data["scale"]
            )

        return self.run_backward(
            input_ids_batch=input_ids_batch,
            attention_mask=attention_mask,
            per_seq_input_data=per_seq_input_data,
            loss_fn=scaled_loss_fn,
            block_size=block_size,
        )

from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist
from torchdata.stateful_dataloader import StatefulDataLoader

from areal.api.engine_api import InferenceEngine, TrainEngine
from areal.api.workflow_api import WorkflowLike
from areal.infra.platforms import current_platform
from areal.utils import stats_tracker
from areal.utils.data import (
    all_gather_tensor_container,
    broadcast_tensor_container,
    concat_padded_tensors,
    extract_single_valid_token_sequence,
    get_total_valid_tokens,
    tensor_container_to,
)
from areal.utils.datapack import ffd_allocate


class _TreeTokenOnlyTimeModel:
    def pred(self, stats: dict[str, Any]) -> float:
        return float(stats["n_tree_tokens"])


def _validate_group_indices(
    group_indices: list[list[int]], n_groups: int, n_items: int
) -> None:
    if len(group_indices) != n_groups:
        raise ValueError(
            f"group_indices must contain exactly {n_groups} groups, got {len(group_indices)}."
        )
    flat_indices = [idx for group in group_indices for idx in group]
    if len(flat_indices) != n_items:
        raise ValueError(
            f"group_indices must assign exactly {n_items} items, got {len(flat_indices)}."
        )
    if sorted(flat_indices) != list(range(n_items)):
        raise ValueError(
            "group_indices must be a permutation of [0, ..., n_items-1] "
            "(no duplicates, no missing/out-of-range indices)."
        )


@dataclass
class RedistributedData:
    all_data: list[dict[str, Any]]
    data: dict[str, Any]
    rank: int
    group_indices: list[list[int]]
    dta_metrics: "DTAMetrics | None" = None


@dataclass(slots=True)
class DTAMetrics:
    n_tokens: float
    n_tree_tokens_before_allocation: float
    n_tree_tokens_after_allocation: float
    compression_ratio_before_allocation: float
    compression_ratio_after_allocation: float

    def to_stats(self) -> dict[str, float]:
        return {
            "dta/n_tokens": self.n_tokens,
            "dta/n_tree_tokens_before_allocation": self.n_tree_tokens_before_allocation,
            "dta/n_tree_tokens_after_allocation": self.n_tree_tokens_after_allocation,
            "dta/compression_ratio_before_allocation": self.compression_ratio_before_allocation,
            "dta/compression_ratio_after_allocation": self.compression_ratio_after_allocation,
        }


@dataclass(slots=True)
class DTAAllocationResult:
    group_indices: list[list[int]]
    metrics: DTAMetrics


def _remove_padding_from_trajectory(d: dict[str, Any]) -> dict[str, Any]:
    """Remove padding from a single trajectory dict based on attention_mask.

    Modifies the dict in-place and returns it.
    """
    if "attention_mask" not in d:
        return d.copy()
    new_d = {}
    max_sequence_length = int(d["attention_mask"].sum(-1).max().item())
    attn_mask_shape = d["attention_mask"].shape
    for k, v in d.items():
        if (
            torch.is_tensor(v)
            and len(v.shape) >= 2
            and v.shape[:2] == attn_mask_shape[:2]
        ):
            new_d[k] = v[:, :max_sequence_length]
        else:
            new_d[k] = v
    return new_d


def _dta_allocate(
    trajectories: list[dict[str, Any]],
    n_groups: int,
) -> DTAAllocationResult:
    from areal.experimental.dta.dp import LB_by_DFS_and_TM
    from areal.experimental.dta.token_trie import TokenTrie

    token_seqs: list[torch.Tensor] = []
    for idx, trajectory in enumerate(trajectories):
        try:
            seq = extract_single_valid_token_sequence(trajectory)
        except (TypeError, ValueError) as err:
            raise ValueError(
                f"Invalid trajectory format at index {idx} for DTA partitioning."
            ) from err
        token_seqs.append(seq)

    all_stats = TokenTrie(token_seqs).get_stats(mode="backward")
    n_total_tokens = float(all_stats["n_tokens"])
    n_tree_tokens_before = float(all_stats["n_tree_tokens"])

    config = SimpleNamespace(K=n_groups, mode="backward", block_size=None)
    group_indices = LB_by_DFS_and_TM(token_seqs, _TreeTokenOnlyTimeModel(), config)

    n_tree_tokens_after = 0.0
    for group in group_indices:
        if not group:
            continue
        group_token_seqs = [token_seqs[idx] for idx in group]
        group_stats = TokenTrie(group_token_seqs).get_stats(mode="backward")
        n_tree_tokens_after += float(group_stats["n_tree_tokens"])

    compression_ratio_before = (
        n_total_tokens / n_tree_tokens_before
        if n_tree_tokens_before > 0
        else float("nan")
    )
    compression_ratio_after = (
        n_total_tokens / n_tree_tokens_after
        if n_tree_tokens_after > 0
        else float("nan")
    )
    metrics = DTAMetrics(
        n_tokens=n_total_tokens,
        n_tree_tokens_before_allocation=n_tree_tokens_before,
        n_tree_tokens_after_allocation=n_tree_tokens_after,
        compression_ratio_before_allocation=compression_ratio_before,
        compression_ratio_after_allocation=compression_ratio_after,
    )
    return DTAAllocationResult(group_indices=group_indices, metrics=metrics)


def redistribute_trajectories(
    trajectories: list[dict[str, Any]],
    group=None,
    partition_mode: str = "seqlen",
) -> RedistributedData:
    """Redistribute a list of trajectory dicts across a process group.

    Each trajectory dict should contain tensors with shape [batch_size, seqlen, *],
    where batch_size can vary per trajectory. This function gathers trajectories
    from all ranks and redistributes them for load balancing based on sequence lengths.

    Parameters
    ----------
    trajectories : list[dict[str, Any]]
        List of trajectory dictionaries from the local rank. Each trajectory
        contains tensors with shape [batch_size, seqlen, ...].
    group : dist.ProcessGroup, optional
        The process group for communication. If None, uses the default group.
    partition_mode : str, optional
        Data-parallel partition policy. "seqlen" uses first-fit-decreasing on
        total sequence lengths. "dta" uses DTA DFS-order partitioning with
        `n_tree_tokens` as cost. Defaults to "seqlen".

    Returns
    -------
    RedistributedData
        Contains:
        - all_data: All trajectories gathered from all ranks (with padding removed)
        - data: Concatenated trajectories assigned to the local rank
        - rank: Local rank in the group
        - group_indices: Assignment of trajectory indices to each rank
    """
    # All-gather trajectories from all ranks
    all_gathered = all_gather_tensor_container(trajectories, group=group)

    # Flatten the list of lists into a single list of trajectories
    all_data = []
    for traj_list in all_gathered:
        all_data.extend(traj_list)

    # Compute sequence lengths for load balancing
    seqlens = [get_total_valid_tokens(d) for d in all_data]

    # Remove pad positions from each trajectory
    for d in all_data:
        _remove_padding_from_trajectory(d)

    n_groups = dist.get_world_size(group)
    if partition_mode == "dta":
        dta_result = _dta_allocate(all_data, n_groups)
        group_indices = dta_result.group_indices
        dta_metrics = dta_result.metrics
    elif partition_mode == "seqlen":
        group_indices = ffd_allocate(seqlens, capacity=int(1e12), min_groups=n_groups)
        dta_metrics = None
    else:
        raise ValueError(
            f"Unsupported partition_mode: {partition_mode}. "
            "Expected one of {'seqlen', 'dta'}."
        )
    _validate_group_indices(group_indices, n_groups=n_groups, n_items=len(all_data))
    local_indices = group_indices[dist.get_rank(group=group)]

    # Concatenate assigned trajectories for this rank
    data = concat_padded_tensors([all_data[i] for i in local_indices])
    return RedistributedData(
        all_data=all_data,
        data=data,
        rank=dist.get_rank(group=group),
        group_indices=group_indices,
        dta_metrics=dta_metrics,
    )


class DistRolloutCoordinator:
    def __init__(self, rollout_engine: InferenceEngine, train_engine: TrainEngine):
        self.rollout_engine = rollout_engine
        self.train_engine = train_engine

    def _broadcast_and_redistribute_trajectories(
        self,
        trajectories: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        """Broadcast and redistribute trajectories across distributed workers.

        This helper encapsulates:
        1. Redistribution within data parallel group (for load balancing)
        2. Broadcasting to context and model parallel group
        3. Synchronization barriers

        Parameters
        ----------
        trajectories : list[dict[str, Any]] | None
            List of trajectory dicts from data parallel head, None for other ranks.
            Each trajectory is a dict of tensors with shape [batch_size, seqlen, ...],
            where batch_size can vary per trajectory.

        Returns
        -------
        dict[str, Any]
            Redistributed and broadcast batch available on all ranks (concatenated)
        """
        partition_mode = self.train_engine.config.partition_mode

        if trajectories is not None:
            redist = redistribute_trajectories(
                trajectories,
                group=self.train_engine.data_parallel_group,
                partition_mode=partition_mode,
            )
            batch = redist.data
            dta_metrics_payload = [redist.dta_metrics]
        else:
            batch = None
            dta_metrics_payload = [None]

        current_platform.synchronize()
        dist.barrier(group=self.train_engine.cpu_group)

        dist.broadcast_object_list(
            dta_metrics_payload,
            src=self.train_engine.current_data_parallel_head(),
            group=self.train_engine.context_and_model_parallel_group,
        )
        dta_metrics = dta_metrics_payload[0]
        if dta_metrics is not None:
            stats_tracker.scalar(**dta_metrics.to_stats())

        batch = broadcast_tensor_container(
            batch,
            src_rank=self.train_engine.current_data_parallel_head(),
            group=self.train_engine.context_and_model_parallel_group,
        )

        current_platform.synchronize()
        dist.barrier(group=self.train_engine.cpu_group)

        return batch

    def rollout_batch(
        self,
        data: list[dict[str, Any]],
        workflow: WorkflowLike,
        workflow_kwargs: dict[str, Any] | None = None,
        group_size: int = 1,
    ) -> dict[str, Any]:
        """Generate rollout batch with distributed coordination (synchronous).

        This method orchestrates distributed rollout generation:
        - Only data parallel heads generate rollouts (avoid redundancy)
        - Results are transferred to device and redistributed
        - Batch is broadcast to all workers
        - Synchronization barriers ensure consistency

        Must call connect_engine() before using this method.

        Parameters
        ----------
        data : List[Dict[str, Any]]
            Input data batch for rollout generation
        workflow : WorkflowLike
            Workflow defining rollout logic
        workflow_kwargs : Dict[str, Any], optional
            Keyword arguments to pass to the workflow constructor
        group_size : int, optional
            Number of times to run the workflow per input and concatenate results.
            Default is 1 (no grouping).

        Returns
        -------
        Dict[str, Any]
            Generated rollout batch on all ranks

        Raises
        ------
        RuntimeError
            If rollout engine not connected via connect_engine()
        """

        trajectories = None
        if self.train_engine.is_data_parallel_head():
            trajectories = self.rollout_engine.rollout_batch(
                data,
                workflow=workflow,
                workflow_kwargs=workflow_kwargs,
                group_size=group_size,
            )
            trajectories = tensor_container_to(
                trajectories, current_platform.current_device()
            )

        return self._broadcast_and_redistribute_trajectories(trajectories)

    def prepare_batch(
        self,
        dataloader: StatefulDataLoader,
        workflow: WorkflowLike,
        workflow_kwargs: dict[str, Any] | None = None,
        should_accept_fn: Callable[[dict[str, Any]], bool] | str | None = None,
        group_size: int = 1,
        dynamic_bs: bool = False,
    ) -> dict[str, Any]:
        """Prepare async rollout batch with distributed coordination.

        Similar to rollout_batch but uses prepare_batch for async training,
        where rollout generation happens concurrently with training.

        Must call connect_engine() before using this method.

        Parameters
        ----------
        dataloader : StatefulDataLoader
            Dataloader to pull samples from
        workflow : WorkflowLike
            Workflow defining rollout logic
        workflow_kwargs : Dict[str, Any], optional
            Keyword arguments to pass to the workflow constructor
        should_accept_fn : Callable[[Dict[str, Any]], bool] | str, optional
            Filter function for accepting samples based on staleness
        group_size : int, optional
            Number of times to run the workflow per input and concatenate results.
            Default is 1 (no grouping).
        dynamic_bs : bool, optional
            If True, enables dynamic batch sizing. Default is False.

        Returns
        -------
        Dict[str, Any]
            Prepared rollout batch on all ranks

        Raises
        ------
        RuntimeError
            If rollout engine not connected via connect_engine()
        """

        trajectories = None
        if self.train_engine.is_data_parallel_head():
            trajectories = self.rollout_engine.prepare_batch(
                dataloader,
                workflow=workflow,
                workflow_kwargs=workflow_kwargs,
                should_accept_fn=should_accept_fn,
                group_size=group_size,
                dynamic_bs=dynamic_bs,
            )
            trajectories = tensor_container_to(
                trajectories, current_platform.current_device()
            )

        return self._broadcast_and_redistribute_trajectories(trajectories)

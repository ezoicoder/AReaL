"""Training-side smoke/benchmark runner for ``ArchonLMEngine``.

This script runs ``N`` training steps of ``ArchonLMEngine.train_batch`` under
``torch.distributed`` and records global per-step loss, elapsed time and peak
GPU memory. At the end it optionally dumps ``diff.pt`` (parameter update
statistics) so two runs can be compared offline via
:mod:`compare_training_dumps`.

Launch with torchrun::

    torchrun --nproc_per_node=$WORLD_SIZE \\
        tests/experimental/archon/torchrun/run_archon_training_test.py \\
        --config tests/experimental/archon/torchrun/archon_training_test.yaml \\
        test_config.step=4 \\
        test_config.data_dir=/path/to/data

Primary outputs land under ``<dump_dir>/``:

- ``stats.jsonl``  -- one global JSON record per step (rank-aggregated)
- ``diff.pt``      -- per-parameter update stats (saved on rank 0 only)

The runner is intentionally narrow: inputs are assumed to be ``list[Tensor]``
(1-D token ids) per ``.pt`` file, and the loss function is hard-wired to a
typical ``grpo_loss_fn`` setup.
"""

from __future__ import annotations

import functools
import glob
import json
import math
import os
import sys
import time
import types
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

# Make repo root importable when invoked via torchrun from any cwd.
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from areal.api.io_struct import FinetuneSpec  # noqa: E402
from areal.experimental.archon.torchrun.training_test_config import (  # noqa: E402
    ArchonTrainingTestConfig,
    ensure_dump_dir,
    load_training_test_config,
)
from areal.experimental.archon.utils import strip_wrapper_prefixes  # noqa: E402
from areal.experimental.engine.archon_engine import ArchonLMEngine  # noqa: E402
from areal.infra.dist_rollout import redistribute_trajectories  # noqa: E402
from areal.infra.platforms import current_platform  # noqa: E402
from areal.trainer.ppo.actor import grpo_loss_fn  # noqa: E402
from areal.utils.data import concat_batch  # noqa: E402
from areal.utils.logging import getLogger  # noqa: E402
from areal.utils.network import find_free_ports  # noqa: E402

# Fixed prompt ratio for synthetic loss mask construction.
_PROMPT_RATIO = 0.3

# -----------------------------------------------------------------------------
# Distributed setup
# -----------------------------------------------------------------------------


def _setup_distributed_environment() -> tuple[int, int]:
    """Initialize the global process group using torchrun env vars."""
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()

    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", str(find_free_ports(1)[0]))

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group(
        backend="nccl",
        init_method=(f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}"),
        world_size=world_size,
        rank=rank,
    )
    current_platform.set_device(int(os.environ["LOCAL_RANK"]))
    return rank, world_size


# -----------------------------------------------------------------------------
# Data loading / trajectory construction
# -----------------------------------------------------------------------------


def _list_step_files(data_dir: str) -> list[str]:
    """Sort .pt files in ``data_dir`` lexicographically (ascii dict order)."""
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"data_dir does not exist or is not a dir: {data_dir}")
    files = sorted(glob.glob(os.path.join(data_dir, "*.pt")))
    if not files:
        raise FileNotFoundError(f"No .pt files under {data_dir}")
    return files


def _load_sequences(pt_path: str) -> list[torch.Tensor]:
    seqs = torch.load(pt_path, map_location="cpu", weights_only=True)
    if not isinstance(seqs, list) or not seqs:
        raise ValueError(
            f"Expected non-empty list[Tensor] in {pt_path}, got {type(seqs)}"
        )
    for i, s in enumerate(seqs):
        if not isinstance(s, torch.Tensor) or s.ndim != 1:
            raise ValueError(
                f"Entry {i} of {pt_path} is not a 1-D tensor: "
                f"type={type(s)}, ndim={getattr(s, 'ndim', None)}"
            )
    return seqs


def _build_trajectory(
    input_ids: torch.Tensor,
    global_idx: int,
    base_seed: int,
    max_tokens: int,
    device: torch.device,
) -> dict[str, Any]:
    """Wrap one 1-D token sequence as a GRPO-ready trajectory dict.

    The per-sequence seed is derived from ``global_idx`` so that every rank
    produces identical synthetic fields for the same sequence in the .pt file.
    """
    assert input_ids.ndim == 1
    seq_len = int(min(int(input_ids.numel()), int(max_tokens)))
    if seq_len <= 0:
        raise ValueError(f"Sequence at idx {global_idx} has non-positive length.")

    ids = input_ids[:seq_len].long().unsqueeze(0).contiguous()
    attention_mask = torch.ones(1, seq_len, dtype=torch.long)
    loss_mask = torch.zeros(1, seq_len)
    prompt_len = max(1, int(seq_len * _PROMPT_RATIO))
    loss_mask[:, prompt_len:] = 1.0

    gen = torch.Generator(device="cpu").manual_seed(int(base_seed) + int(global_idx))
    logprobs = torch.randn(1, seq_len, generator=gen) * 0.5 - 2.0
    old_logprobs = logprobs.clone()
    advantages = torch.randn(1, seq_len, generator=gen)
    rewards = torch.randint(0, 2, (1,), generator=gen).float()
    values = torch.zeros(1, seq_len)

    traj = {
        "input_ids": ids,
        "attention_mask": attention_mask,
        "loss_mask": loss_mask,
        "logprobs": logprobs,
        "old_logprobs": old_logprobs,
        "advantages": advantages,
        "rewards": rewards,
        "values": values,
        "prox_logp": old_logprobs.clone(),
    }
    return {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in traj.items()
    }


def _build_local_trajectories(
    seqs: list[torch.Tensor],
    dp_rank: int,
    dp_world_size: int,
    base_seed: int,
    max_tokens: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Each rank owns a disjoint stride of the sequence list.

    ``all_gather_tensor_container`` inside ``redistribute_trajectories`` assumes
    every participant supplies the same number of trajectories. Striding by
    ``dp_rank::dp_world_size`` trivially satisfies that when the total count is
    divisible by ``dp_world_size``; any remainder lands on the lowest-ranked
    contributors first, so we truncate to the largest multiple of dp size.
    """
    trimmed_len = (len(seqs) // dp_world_size) * dp_world_size
    if trimmed_len == 0:
        raise ValueError(
            f"Need at least dp_world_size={dp_world_size} sequences, got {len(seqs)}."
        )
    seqs = seqs[:trimmed_len]

    out: list[dict[str, Any]] = []
    for local_i, global_i in enumerate(range(dp_rank, trimmed_len, dp_world_size)):
        del local_i
        out.append(
            _build_trajectory(
                input_ids=seqs[global_i],
                global_idx=global_i,
                base_seed=base_seed,
                max_tokens=max_tokens,
                device=device,
            )
        )
    return out


# -----------------------------------------------------------------------------
# Loss function / engine patching
# -----------------------------------------------------------------------------


# Reasonable defaults mirroring ``tests/experimental/archon/test_grpo.py``.
_GRPO_KW: dict[str, Any] = dict(
    eps_clip=0.2,
    eps_clip_higher=None,
    c_clip=None,
    importance_sampling_level="token",
    current_version=1,
    prox_logp_method="recompute",
    use_sapo_loss=False,
    use_decoupled_loss=False,
)


def _loss_weight_fn(input_data: dict[str, Any]) -> torch.Tensor:
    mask = input_data["loss_mask"]
    return mask.count_nonzero()


def _patch_engine_for_test(
    engine: ArchonLMEngine,
    disable_optimizer: bool,
) -> None:
    """Inject optional optimizer no-ops onto the engine."""
    if not disable_optimizer:
        return

    def _noop_zero_grad(self):
        for p in self._get_all_parameters():
            if p.grad is not None:
                p.grad = None

    def _noop_step(self):
        grad_norm = 0.0
        for p in self._get_all_parameters():
            if p.grad is not None:
                grad_norm += float(p.grad.detach().float().norm().item()) ** 2
        grad_norm = grad_norm**0.5
        _noop_zero_grad(self)
        return {
            "update_successful": 1.0,
            "grad_norm": grad_norm,
            "lr": 0.0,
        }

    engine.optimizer_zero_grad = types.MethodType(_noop_zero_grad, engine)
    engine.optimizer_step = types.MethodType(_noop_step, engine)


# -----------------------------------------------------------------------------
# Engine lifecycle
# -----------------------------------------------------------------------------


def _create_engine(cfg: ArchonTrainingTestConfig) -> ArchonLMEngine:
    """Construct + initialize an ``ArchonLMEngine`` from the test config."""
    parallel_strategy = cfg.parallel.to_parallel_strategy()

    engine_cfg = cfg.engine
    if cfg.test_config.disable_optimizer:
        # Skip optimizer creation entirely so no Adam state is allocated.
        engine_cfg.optimizer = None

    engine = ArchonLMEngine(engine_cfg)
    engine.create_process_group(parallel_strategy=parallel_strategy)

    ft_spec = FinetuneSpec(
        total_train_epochs=1,
        dataset_size=max(1, int(cfg.test_config.step)),
        train_batch_size=1,
    )
    engine.initialize(addr=None, ft_spec=ft_spec)
    return engine


def _destroy_engine(engine: ArchonLMEngine | None) -> None:
    if engine is not None:
        engine.destroy()
    if dist.is_initialized():
        dist.destroy_process_group()


# -----------------------------------------------------------------------------
# Parameter diff dump
# -----------------------------------------------------------------------------


def _materialize_full_param(param: torch.Tensor) -> torch.Tensor:
    """Return a full (unsharded) tensor for one parameter."""
    from torch.distributed.tensor import DTensor

    if isinstance(param, DTensor):
        return param.full_tensor()
    return param


def _to_dump_name_tensors(
    engine: ArchonLMEngine, raw_name: str, tensor: torch.Tensor
) -> list[tuple[str, torch.Tensor]]:
    """Convert one Archon parameter into dump-name/tensor pairs.

    Prefer HuggingFace keys when ``state_dict_adapter`` is available; otherwise
    use wrapper-stripped Archon keys.
    """
    adapter = engine.state_dict_adapter
    if adapter is not None:
        mapped = adapter.convert_single_to_hf(raw_name, tensor)
        if mapped:
            return [(strip_wrapper_prefixes(name), value) for name, value in mapped]
    return [(strip_wrapper_prefixes(raw_name), tensor)]


def _snapshot_initial_full_params(
    engine: ArchonLMEngine,
) -> dict[str, torch.Tensor] | None:
    """Capture initial full params on CPU (rank 0 only).

    Each parameter is materialized one-by-one via ``full_tensor()`` and moved to
    CPU immediately. This keeps extra GPU memory bounded by one parameter tensor.
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    out: dict[str, torch.Tensor] | None = {} if rank == 0 else None
    for raw_name, param in engine.model.named_parameters():
        full = _materialize_full_param(param)
        if rank == 0:
            assert out is not None
            for dump_name, dump_tensor in _to_dump_name_tensors(engine, raw_name, full):
                if dump_name in out:
                    raise ValueError(
                        f"Duplicate dump key '{dump_name}' from raw param '{raw_name}'."
                    )
                out[dump_name] = (
                    dump_tensor.detach().to(device="cpu", dtype=torch.float32).clone()
                )
        del full
    if dist.is_initialized():
        dist.barrier(group=engine.cpu_group)
    return out


def _save_diff_snapshot(
    engine: ArchonLMEngine,
    initial_params: dict[str, torch.Tensor] | None,
    dump_dir: str,
    filename: str,
) -> str | None:
    """Save ``diff.pt`` with per-parameter update metrics (rank 0 only)."""
    out_path: str | None = None
    rank = dist.get_rank() if dist.is_initialized() else 0
    if dist.is_initialized():
        dist.barrier(group=engine.cpu_group)

    if rank == 0 and initial_params is None:
        raise RuntimeError("Missing initial params on rank 0 for diff snapshot.")

    # Every rank participates in full_tensor() to ensure collective safety.
    if rank == 0:
        assert initial_params is not None
        per_param: dict[str, dict[str, float]] = {}
        global_numel = 0.0
        global_abs_sum = 0.0
        global_l2_sq = 0.0
        global_ref_l2_sq = 0.0
        global_max_abs = 0.0
    else:
        per_param = {}
        global_numel = 0.0
        global_abs_sum = 0.0
        global_l2_sq = 0.0
        global_ref_l2_sq = 0.0
        global_max_abs = 0.0

    for raw_name, param in engine.model.named_parameters():
        full = _materialize_full_param(param)
        if rank == 0:
            assert initial_params is not None
            for dump_name, dump_tensor in _to_dump_name_tensors(engine, raw_name, full):
                if dump_name not in initial_params:
                    raise KeyError(
                        f"Missing initial parameter for dump key '{dump_name}' "
                        f"(raw='{raw_name}')."
                    )
                initial = initial_params[dump_name]
                current = dump_tensor.detach().to(device="cpu", dtype=torch.float32)
                if current.shape != initial.shape:
                    raise ValueError(
                        f"Shape mismatch for '{dump_name}': current={tuple(current.shape)} "
                        f"vs initial={tuple(initial.shape)}"
                    )
                delta = current - initial
                abs_delta = delta.abs()
                numel = float(delta.numel())
                abs_sum = float(abs_delta.sum().item())
                l2_sq = float(delta.double().pow(2).sum().item())
                ref_l2_sq = float(initial.double().pow(2).sum().item())
                max_abs = float(abs_delta.max().item()) if delta.numel() > 0 else 0.0
                l2 = math.sqrt(max(l2_sq, 0.0))
                ref_l2 = math.sqrt(max(ref_l2_sq, 0.0))
                if dump_name in per_param:
                    raise ValueError(
                        f"Duplicate final dump key '{dump_name}' from raw param '{raw_name}'."
                    )
                per_param[dump_name] = {
                    "numel": numel,
                    "mean_abs_update": abs_sum / max(numel, 1.0),
                    "max_abs_update": max_abs,
                    "l2_update": l2,
                    "rel_l2_update": l2 / max(ref_l2, 1e-12),
                }
                global_numel += numel
                global_abs_sum += abs_sum
                global_l2_sq += l2_sq
                global_ref_l2_sq += ref_l2_sq
                global_max_abs = max(global_max_abs, max_abs)
        del full

    if rank == 0:
        payload = {
            "schema_version": 1,
            "aggregation": "full_tensor_one_param_peak",
            "params": per_param,
            "global": {
                "num_params": len(per_param),
                "numel": global_numel,
                "mean_abs_update": global_abs_sum / max(global_numel, 1.0),
                "max_abs_update": global_max_abs,
                "l2_update": math.sqrt(max(global_l2_sq, 0.0)),
                "rel_l2_update": math.sqrt(max(global_l2_sq, 0.0))
                / max(math.sqrt(max(global_ref_l2_sq, 0.0)), 1e-12),
            },
        }

        os.makedirs(dump_dir, exist_ok=True)
        out_path = os.path.join(dump_dir, filename)
        torch.save(payload, out_path)

    if dist.is_initialized():
        dist.barrier(group=engine.cpu_group)
    return out_path


# -----------------------------------------------------------------------------
# Per-step training
# -----------------------------------------------------------------------------


def _run_single_step(
    *,
    engine: ArchonLMEngine,
    cfg: ArchonTrainingTestConfig,
    step_idx: int,
    step_file: str,
    device: torch.device,
    loss_fn,
) -> dict[str, Any]:
    """Run one training step and return a global (all-rank) stats record."""
    dp_rank = engine.data_parallel_rank
    dp_world_size = engine.data_parallel_world_size
    dp_group = engine.data_parallel_group

    seqs = _load_sequences(step_file)
    max_tokens = int(engine.config.mb_spec.max_tokens_per_mb)

    trajectories = _build_local_trajectories(
        seqs=seqs,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        base_seed=cfg.test_config.seed + step_idx * 100003,
        max_tokens=max_tokens,
        device=device,
    )

    redist = redistribute_trajectories(
        trajectories=trajectories,
        group=dp_group,
        packing_algorithm=engine.config.packing_algorithm,
    )
    local_trajectories = redist.data
    if not local_trajectories:
        raise RuntimeError(
            f"Step {step_idx}: redistribute_trajectories returned no local trajectories. "
            f"all_data={len(redist.all_data)}, group_indices={redist.group_indices}, "
            f"rank={redist.rank}"
        )
    batch, _ = concat_batch(local_trajectories)
    batch = {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in batch.items()
    }

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    result = engine.train_batch(
        input_=batch,
        loss_fn=loss_fn,
        loss_weight_fn=_loss_weight_fn,
        return_loss=True,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_s = time.perf_counter() - t0
    peak_mem_mib = (
        float(torch.cuda.max_memory_allocated() / (1024**2))
        if torch.cuda.is_available()
        else 0.0
    )

    step_loss = float(result.get("loss", float("nan")))
    loss_source = "train_batch_return"

    num_local_seqs = int(batch["input_ids"].shape[0])
    num_local_tokens = int(batch["attention_mask"].sum().item())

    grad_norm_local = float(result.get("grad_norm", float("nan")))
    grad_norm_local_for_max = (
        grad_norm_local if math.isfinite(grad_norm_local) else float("-inf")
    )
    lr_local = float(result.get("lr", 0.0))
    update_successful_local = float(result.get("update_successful", 0.0))

    loss_weight_local = float(max(num_local_tokens, 1))
    weighted_loss_local = (
        float(step_loss) * loss_weight_local if math.isfinite(step_loss) else 0.0
    )
    loss_weight_local = loss_weight_local if math.isfinite(step_loss) else 0.0

    reduce_sum = torch.tensor(
        [
            weighted_loss_local,
            loss_weight_local,
            float(num_local_seqs),
            float(num_local_tokens),
        ],
        dtype=torch.float64,
        device=device,
    )
    reduce_max = torch.tensor(
        [
            float(elapsed_s),
            float(peak_mem_mib),
            float(grad_norm_local_for_max),
            float(lr_local),
        ],
        dtype=torch.float64,
        device=device,
    )
    reduce_min = torch.tensor(
        [float(update_successful_local)],
        dtype=torch.float64,
        device=device,
    )

    if dist.is_initialized():
        dist.all_reduce(reduce_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(reduce_max, op=dist.ReduceOp.MAX)
        dist.all_reduce(reduce_min, op=dist.ReduceOp.MIN)

    global_loss_weight = float(reduce_sum[1].item())
    global_loss = (
        float(reduce_sum[0].item() / global_loss_weight)
        if global_loss_weight > 0.0
        else float("nan")
    )
    global_grad_norm = float(reduce_max[2].item())
    if not math.isfinite(global_grad_norm):
        global_grad_norm = float("nan")

    return {
        "step": int(step_idx),
        "file": os.path.abspath(step_file),
        "world_size": int(dist.get_world_size()) if dist.is_initialized() else 1,
        "dp_world_size": int(dp_world_size),
        "num_global_sequences": int(round(float(reduce_sum[2].item()))),
        "num_global_tokens": int(round(float(reduce_sum[3].item()))),
        "elapsed_s_max": float(reduce_max[0].item()),
        "peak_mem_mib_max": float(reduce_max[1].item()),
        "loss": float(global_loss),
        "loss_source": f"{loss_source}_global_token_weighted",
        "grad_norm_max": float(global_grad_norm),
        "update_successful": float(reduce_min[0].item()),
        "lr_max": float(reduce_max[3].item()),
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    cfg, config_path = load_training_test_config(argv)

    rank, world_size = _setup_distributed_environment()
    device = torch.device(current_platform.device_type)
    logger = getLogger(f"[ArchonTrainingTest Rank {rank}]")

    dump_dir = ensure_dump_dir(cfg, rank=rank)
    stats_path = os.path.join(dump_dir, "stats.jsonl")
    if rank == 0:
        # Truncate any prior stats file.
        open(stats_path, "w").close()

    if rank == 0:
        logger.info(
            "config=%s dump_dir=%s world_size=%s",
            config_path,
            dump_dir,
            world_size,
        )

    step_files = _list_step_files(cfg.test_config.data_dir)
    if rank == 0:
        logger.info(
            "Found %d .pt files in %s",
            len(step_files),
            cfg.test_config.data_dir,
        )

    engine: ArchonLMEngine | None = None

    try:
        engine = _create_engine(cfg)
        _patch_engine_for_test(
            engine,
            disable_optimizer=cfg.test_config.disable_optimizer,
        )
        if rank == 0 and cfg.test_config.save_params:
            logger.warning(
                "test_config.save_params is deprecated in low-memory mode and "
                "ignored. Use diff.pt.",
            )
        if rank == 0 and cfg.test_config.save_initial_params:
            logger.warning(
                "test_config.save_initial_params is ignored in low-memory mode.",
            )

        initial_params: dict[str, torch.Tensor] | None = None
        if cfg.test_config.save_diff:
            initial_params = _snapshot_initial_full_params(engine)

        loss_fn = functools.partial(grpo_loss_fn, **_GRPO_KW)

        num_steps = int(cfg.test_config.step)
        for step_idx in range(num_steps):
            file_idx = step_idx % len(step_files)
            step_file = step_files[file_idx]
            if rank == 0:
                logger.info(
                    "Starting training step %d/%d (0-based index %d), data file=%s",
                    step_idx + 1,
                    num_steps,
                    step_idx,
                    os.path.abspath(step_file),
                )

            record = _run_single_step(
                engine=engine,
                cfg=cfg,
                step_idx=step_idx,
                step_file=step_file,
                device=device,
                loss_fn=loss_fn,
            )

            if rank == 0:
                with open(stats_path, "a") as fp:
                    fp.write(json.dumps(record) + "\n")
                logger.info(
                    "Step %03d done: file=%s loss=%.6f grad_norm(max)=%.4f "
                    "elapsed(max)=%.2fs peak_mem(max)=%.1fMiB",
                    step_idx,
                    os.path.basename(step_file),
                    record["loss"],
                    record["grad_norm_max"],
                    record["elapsed_s_max"],
                    record["peak_mem_mib_max"],
                )

        if cfg.test_config.save_diff:
            _save_diff_snapshot(engine, initial_params, dump_dir, "diff.pt")
            if initial_params is not None:
                initial_params.clear()
    finally:
        _destroy_engine(engine)


if __name__ == "__main__":
    main()

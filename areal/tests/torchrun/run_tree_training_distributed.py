"""
Distributed tree training test with torchrun.

This script tests tree attention training in a multi-GPU environment,
fully reusing the production code path including prepare_batch().
"""
import argparse
import os
import time
from contextlib import contextmanager
from typing import Any
import torch
import torch.distributed as dist

from areal.api.alloc_mode import AllocationMode
from areal.api.cli_args import TrainEngineConfig, MicroBatchSpec, FSDPEngineConfig, OptimizerConfig
from areal.api.io_struct import FinetuneSpec, WeightUpdateMeta
from areal.engine.fsdp_engine import FSDPEngine
from areal.platforms import current_platform
from areal.utils.data import tensor_container_to
from areal.utils import logging
from areal.tests.test_tree_training import (
    reset_peak_memory, get_memory_stats,
)
from areal.engine.ppo.actor import grpo_loss_fn
from areal.tests.utils import get_model_path
from functools import partial
loss_fn = partial(
    grpo_loss_fn,
    eps_clip=0.2,  # insert appropriate values for your test case
    eps_clip_higher=None,
    c_clip=None,
    behav_imp_weight_cap=None,
    m2_threshold=None,
    importance_sampling_level="token",
    current_version=None,
    prox_logp_method="recompute",
    use_sapo_loss=False,
    sapo_tau_pos=1.0,
    sapo_tau_neg=1.05,
    use_decoupled_loss=False,
    vocab_min_logits=None,
    vocab_max_logits=None,
)

def loss_weight_fn(input_data):
    """Default weight function based on attention_mask sum"""
    return input_data["loss_mask"].count_nonzero()

# Create logger for distributed tests
logger = logging.getLogger("TreeTrainingDistributed")

MODEL_PATH = get_model_path(
    "/data/tree/models/Qwen3-0.6B", "Qwen/Qwen3-0.6B"
)

# Path to real tree training data (prefix, each rank will append _rank{R}.pt)
TREE_DATA_PATH = "/data/tree/tree-data/tau2-16k-small/call2.pt"


def _collect_full_gradients_on_rank0(engine: FSDPEngine, rank: int) -> dict[str, torch.Tensor] | None:
    """Collect FULL gradients on rank 0 only (efficient).
    
    This function handles the gradient distribution differences:
    - ZeRO-1 (Tree Stack): All ranks have full gradients after all_reduce
      -> Only rank 0 collects, others skip
    - FSDP2 (Baseline): Gradients are sharded DTensors
      -> ALL ranks must call DTensor.full_tensor() (collective operation)
      -> But only rank 0 saves the result
    
    Note: Gradients are kept on GPU (cuda:0) for faster comparison.
    
    Returns:
        Dictionary of full gradients on GPU (rank 0 only), None on other ranks
    """
    # Initialize grads dict only on rank 0
    grads = {} if rank == 0 else None
    
    # Check if any parameter has DTensor gradients (FSDP2)
    has_dtensor = False
    for param in engine.model.parameters():
        if param.grad is not None and hasattr(param.grad, 'full_tensor'):
            has_dtensor = True
            break
    
    if not has_dtensor and rank != 0:
        # ZeRO-1/DDP mode: non-zero ranks can skip entirely
        return None
    
    # FSDP2 mode OR rank 0: need to participate
    for name, param in engine.model.named_parameters():
        if param.grad is None:
            continue
        
        # Check if this is a DTensor (FSDP2)
        if hasattr(param.grad, 'full_tensor'):
            # FSDP2: ALL ranks must call full_tensor() (collective operation)
            # This triggers all_gather internally across all ranks
            full_grad = param.grad.full_tensor()
            
            # Only rank 0 saves (keep on GPU for faster comparison)
            if rank == 0:
                clean_name = name.replace("module.", "", 1) if name.startswith("module.") else name
                grads[clean_name] = full_grad
        elif hasattr(param.grad, '_local_tensor'):
            # DTensor but no full_tensor method (shouldn't happen, but handle it)
            # Fallback: just get local tensor on rank 0
            if rank == 0:
                clean_name = name.replace("module.", "", 1) if name.startswith("module.") else name
                grads[clean_name] = param.grad._local_tensor
        else:
            # Regular tensor (ZeRO-1/DDP): already full gradient, only rank 0 collects
            # Note: param.grad is already detached (gradients don't need gradients)
            if rank == 0:
                clean_name = name.replace("module.", "", 1) if name.startswith("module.") else name
                grads[clean_name] = param.grad
    
    return grads


class MockRolloutEngine:
    """Mock rollout engine that loads data from file.
    
    Implements the InferenceEngine interface required by FSDPEngine.connect_engine().
    In real scenarios, this would be RemoteSGLangEngine generating rollouts.
    Here we just load pre-saved data from disk.
    
    Note: Data is loaded on CPU and will be moved to GPU by DistRolloutCoordinator
    via tensor_container_to(), matching the real code path.
    
    All ranks load from the same data file, then split data by rank.
    """
    def __init__(self, data_path: str, rank: int, world_size: int, prefix_len: int = -1):
        """
        Args:
            data_path: Path to data file (e.g., "/path/to/call2.pt")
            rank: Rank number to determine which slice to take
            world_size: Total number of ranks
            prefix_len: Number of sequences to keep (-1 means all)
        """
        self.data_path = data_path
        self.rank = rank
        self.world_size = world_size
        self.prefix_len = prefix_len
        self._data_cache = None
        
    def initialize(self, *args, **kwargs):
        """Empty implementation for interface compatibility."""
        pass
    
    def destroy(self):
        """Empty implementation for interface compatibility."""
        pass
    
    def get_version(self):
        """Return mock version."""
        return 0
    
    def set_version(self, version: int):
        """Empty implementation for interface compatibility."""
        pass
    
    def prepare_batch(self, dataloader, workflow, workflow_kwargs=None, 
                     should_accept_fn=None, group_size=1, dynamic_bs=False) -> list[dict[str, Any]]:
        """Load data from disk and return as trajectories list.

        Additionally, save the rank's split input as TREE_DATA_PATH_split{rank}.pt,
        where TREE_DATA_PATH is self.data_path with the trailing .pt removed.
        The data is saved under the 'input_data' key in the output file, 
        and all tensors are moved to cpu before saving.
        """
        import os

        if self._data_cache is None:
            # All ranks load from the same data file
            data = torch.load(self.data_path, map_location='cpu')
            input_data = data["input_data"]
            
            # Apply prefix_len filter if specified
            if self.prefix_len != -1:
                for key, value in input_data.items():
                    if isinstance(value, torch.Tensor) and value.size(0) >= self.prefix_len:
                        input_data[key] = value[:self.prefix_len]
            
            # Get total batch size
            total_batch_size = input_data["attention_mask"].shape[0]
            
            # Ensure batch size is divisible by world_size
            if total_batch_size % self.world_size != 0:
                raise ValueError(
                    f"Batch size {total_batch_size} must be divisible by world_size {self.world_size}. "
                    f"Consider using --prefix-len to adjust the batch size."
                )
            
            # Calculate per-rank batch size and this rank's slice
            per_rank_batch_size = total_batch_size // self.world_size
            start_idx = self.rank * per_rank_batch_size
            end_idx = start_idx + per_rank_batch_size
            
            print(f"[Rank {self.rank}] Total batch: {total_batch_size}, taking slice [{start_idx}:{end_idx}]")
            
            # Slice data for this rank
            rank_input_data = {}
            for key, value in input_data.items():
                if isinstance(value, torch.Tensor):
                    # Move to cpu (if not already), then slice
                    rank_input_data[key] = value[start_idx:end_idx].to('cpu')
                else:
                    rank_input_data[key] = value

            print(f"[Rank {self.rank}] input_data['attention_mask'] shape: {rank_input_data['attention_mask'].shape}")
            print(f"[Rank {self.rank}] input_data['attention_mask'] device: {rank_input_data['attention_mask'].device}")

            # if self.world_size > 1:
            #     # Save rank_input_data to TREE_DATA_PATH_split{rank}.pt, with all tensors on cpu 
            #     base_path = self.data_path
            #     if base_path.endswith('.pt'):
            #         base_path = base_path[:-3]
            #     split_save_path = f"{base_path}_split{self.rank}.pt"

            #     # Make sure all tensors in rank_input_data are on cpu
            #     cpu_rank_input_data = {}
            #     for k, v in rank_input_data.items():
            #         if isinstance(v, torch.Tensor):
            #             cpu_rank_input_data[k] = v.to('cpu')
            #         else:
            #             cpu_rank_input_data[k] = v

            #     torch.save({'input_data': cpu_rank_input_data}, split_save_path)
            #     print(f"[Rank {self.rank}] Saved split input_data to {split_save_path}")

            # Convert batch format to list of trajectories
            # Each trajectory is a dict with shape [1, seq_len, ...]
            trajectories = []
            for i in range(per_rank_batch_size):
                traj = {}
                for key, value in rank_input_data.items():
                    if isinstance(value, torch.Tensor):
                        # Each elem stays on cpu
                        traj[key] = value[i:i+1]
                    else:
                        traj[key] = value
                trajectories.append(traj)
            
            self._data_cache = trajectories

        return self._data_cache


@contextmanager
def setup_engine_with_mock_rollout(
    rank: int,
    world_size: int,
    experiment_name: str,
    max_tokens_per_mb: int,
    prefix_len: int = -1,
    enable_tree_stack_training: bool = False,
    enable_tree_training: bool = False,
    is_tree_distribution: bool = True,
    gradient_checkpointing: bool = False,
    disable_optimizer: bool = True,
):
    """Setup FSDP engine with mock rollout engine.
    
    This allows us to call engine.prepare_batch() and fully reuse the
    production code path including DistRolloutCoordinator logic.
    
    Args:
        enable_tree_stack_training: Enable tree stack (tree attention) training
        enable_tree_training: Enable tree training (flatten tree / flex)
        is_tree_distribution: Whether to use tree-based data distribution
        gradient_checkpointing: Enable gradient checkpointing (only for baseline/flex)
    
    Yields:
        Initialized engine instance
    """
    # Tree stack doesn't support gradient checkpointing
    actual_gradient_checkpointing = False if enable_tree_stack_training else gradient_checkpointing
    
    config = TrainEngineConfig(
        experiment_name=experiment_name,
        trial_name="distributed_test",
        path=MODEL_PATH,
        mb_spec=MicroBatchSpec(max_tokens_per_mb=max_tokens_per_mb),
        optimizer=None if disable_optimizer else OptimizerConfig(),
        enable_tree_training=enable_tree_training,
        enable_tree_stack_training=enable_tree_stack_training,
        gradient_checkpointing=actual_gradient_checkpointing,
        disable_optimizer=disable_optimizer,
        is_tree_distribution=is_tree_distribution,
        fsdp=FSDPEngineConfig(),
    )
    
    alloc_mode = AllocationMode.from_str(f"d{world_size}p1t1")
    ft_spec = FinetuneSpec(total_train_epochs=1, dataset_size=128, train_batch_size=8)
    
    engine = FSDPEngine(config)
    engine.create_process_group(alloc_mode.train)
    engine.initialize(addr=None, ft_spec=ft_spec, parallel_strategy=alloc_mode.train)
    
    # Create and connect mock rollout engine with prefix_len support
    # All ranks load from same file, then split by rank
    mock_engine = MockRolloutEngine(TREE_DATA_PATH, rank=rank, world_size=world_size, prefix_len=prefix_len)
    meta = WeightUpdateMeta(type="disk")
    engine.connect_engine(mock_engine, meta)
    
    print(f"[Rank {rank}] Engine initialized: is_DP_head={engine.is_data_parallel_head()}, prefix_len={prefix_len}")
    
    try:
        yield engine
    finally:
        # Clean up engine resources
        print(f"[Rank {rank}] Destroying engine: {experiment_name}")
        
        # Critical: Synchronize BEFORE destroy() to ensure all ranks are ready
        # After destroy(), process groups may be gone, so we can't call barrier()
        dist.barrier()
        
        # Destroy the model and optimizer (may destroy process groups if own_global_group=True)
        engine.destroy()
        
        # Clear the device mesh cache to allow creating a new engine
        # NOTE: We can't synchronize after this because destroy() may have killed process groups
        # Instead, we rely on the pre-destroy barrier to ensure all ranks proceed together
        
        print(f"[Rank {rank}] Engine cleanup complete: {experiment_name}")


def run_single_training(
    mode: str,
    max_tokens_per_mb: int,
    prefix_len: int,
    gradient_checkpointing: bool,
    save_grad_file: str,
    compare_grad_file: str | None = None,  # Kept for backward compatibility, not used
):
    """Run training for a single mode and save gradients to file.
    
    Args:
        mode: Training mode - "baseline", "flex", or "stack"
        max_tokens_per_mb: Max tokens per microbatch
        prefix_len: Number of sequences to keep
        gradient_checkpointing: Whether to enable gradient checkpointing (only for baseline/flex)
        save_grad_file: Path to save gradients (rank 0 only)
        compare_grad_file: Deprecated, kept for compatibility
    """
    # Get rank and world_size from environment (set by torchrun)
    # Don't use dist.get_rank() here as distributed hasn't been initialized yet
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # Map mode to engine config
    mode_config = {
        "baseline": {
            "enable_tree_stack_training": False,
            "enable_tree_training": False,
            "is_tree_distribution": False,
            "use_gradient_checkpointing": gradient_checkpointing,
        },
        "flex": {
            "enable_tree_stack_training": False,
            "enable_tree_training": True,
            "is_tree_distribution": False,
            "use_gradient_checkpointing": gradient_checkpointing,
        },
        "stack": {
            "enable_tree_stack_training": True,
            "enable_tree_training": False,
            "is_tree_distribution": True,
            "use_gradient_checkpointing": False,  # Stack doesn't support checkpointing
        },
    }
    
    if mode not in mode_config:
        raise ValueError(f"Invalid mode: {mode}. Must be one of {list(mode_config.keys())}")
    
    config = mode_config[mode]
    print(f"[Rank {rank}] Running {mode} training (world_size={world_size})")
    
    # Setup engine and run training
    with setup_engine_with_mock_rollout(
        rank, world_size, mode, max_tokens_per_mb,
        prefix_len=prefix_len,
        enable_tree_stack_training=config["enable_tree_stack_training"],
        enable_tree_training=config["enable_tree_training"],
        is_tree_distribution=config["is_tree_distribution"],
        gradient_checkpointing=config["use_gradient_checkpointing"],
    ) as engine:
        
        print(f"[Rank {rank}] Calling {mode}_engine.prepare_batch()...")
        input_data = engine.prepare_batch(dataloader=None, workflow=None)
        
        # Train
        reset_peak_memory()
        engine.train()
        torch.cuda.synchronize()
        start = time.time()
        
        _,loss = engine.train_batch(input_data, loss_fn=loss_fn, loss_weight_fn=loss_weight_fn, required_loss=True)
        
        torch.cuda.synchronize()
        local_train_time = time.time() - start
        
        # Synchronize across all ranks to wait for the slowest rank
        dist.barrier()
        
        # Collect training time from all ranks and take the maximum (slowest rank)
        time_tensor = torch.tensor([local_train_time], dtype=torch.float32, device='cuda')
        dist.all_reduce(time_tensor, op=dist.ReduceOp.MAX)
        train_time = time_tensor.item()  # Maximum training time across all ranks
        
        # Collect memory stats from all ranks and take the maximum
        local_mem = get_memory_stats(mode.capitalize())
        mem_tensor = torch.tensor([
            local_mem["allocated_gb"],
            local_mem["reserved_gb"],
            local_mem["peak_allocated_gb"]
        ], dtype=torch.float32, device='cuda')
        dist.all_reduce(mem_tensor, op=dist.ReduceOp.MAX)
        train_mem = {
            "stage": local_mem["stage"],
            "allocated_gb": mem_tensor[0].item(),
            "reserved_gb": mem_tensor[1].item(),
            "peak_allocated_gb": mem_tensor[2].item(),
        }
        
        print(f"[Rank {rank}] {mode.capitalize()}: local_time={local_train_time:.4f}s, max_time={train_time:.4f}s, loss={loss:.6f}")
        print(f"[Rank {rank}] {mode.capitalize()}: local_mem=[alloc={local_mem['allocated_gb']:.2f}GB, reserved={local_mem['reserved_gb']:.2f}GB, peak={local_mem['peak_allocated_gb']:.2f}GB]")
        if rank == 0:
            print(f"[Rank 0] {mode.capitalize()}: max_mem_across_ranks=[alloc={train_mem['allocated_gb']:.2f}GB, reserved={train_mem['reserved_gb']:.2f}GB, peak={train_mem['peak_allocated_gb']:.2f}GB]")
        
        # Collect loss from all ranks and compute average
        loss_tensor = torch.tensor([loss], dtype=torch.float32, device='cuda')
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        loss_sum = (loss_tensor.item() / world_size)  # Average loss across all ranks
        
        # Collect gradients on rank 0
        grads = _collect_full_gradients_on_rank0(engine, rank)
    
    # Save gradients (rank 0 only)
    if rank == 0 and save_grad_file and grads:
        print(f"[Rank 0] Saving {mode} gradients to {save_grad_file}")
        print(f"[Rank 0] Average loss across {world_size} ranks: {loss_sum:.6f}")
        torch.save({
            "mode": mode,
            "grads": grads,
            "time": train_time,
            "mem": train_mem,
            "loss_sum": loss_sum,  # Average loss across all ranks
            "config": {
                "max_tokens_per_mb": max_tokens_per_mb,
                "prefix_len": prefix_len,
                "gradient_checkpointing": config["use_gradient_checkpointing"],
            }
        }, save_grad_file)
        print(f"[Rank 0] ✓ Saved {len(grads)} gradient tensors")
    
    print(f"[Rank {rank}] {mode.capitalize()} training complete")




def main():
    parser = argparse.ArgumentParser(
        description="Run distributed tree training and save gradients to file"
    )
    parser.add_argument("--mode", type=str, 
                       choices=["baseline", "flex", "stack"], 
                       required=True,
                       help="Training mode: baseline (FSDP2), flex (flatten tree), or stack (tree attention)")
    parser.add_argument("--max_tokens_per_mb", type=int, default=16384,
                       help="Max tokens per microbatch")
    parser.add_argument("--prefix-len", type=int, default=-1,
                       help="Number of sequences to keep from data file. -1 means keep all")
    parser.add_argument("--save-grad-file", type=str, required=True,
                       help="Path to save gradients (required)")
    parser.add_argument("--disable-gradient-checkpointing", action="store_true", default=False,
                       help="Disable gradient checkpointing for baseline/flex")
    args = parser.parse_args()
    
    gradient_checkpointing = not args.disable_gradient_checkpointing
    
    run_single_training(
        mode=args.mode,
        max_tokens_per_mb=args.max_tokens_per_mb,
        prefix_len=args.prefix_len,
        gradient_checkpointing=gradient_checkpointing,
        save_grad_file=args.save_grad_file,
        compare_grad_file=None,  # No comparison, only save
    )


if __name__ == "__main__":
    main()

"""
Usage examples:

# Save baseline gradients (single GPU)
torchrun --nproc_per_node=1 --master_port=29500 \
  areal/tests/torchrun/run_tree_training_distributed.py \
  --mode=baseline \
  --save-grad-file=/tmp/baseline.pt \
  --max_tokens_per_mb=16384 \
  --prefix-len=30

# Save flex gradients (single GPU)
AREAL_FLEX_ATTENTION_BLOCK_SIZE=64 torchrun --nproc_per_node=1 --master_port=29501 \
  areal/tests/torchrun/run_tree_training_distributed.py \
  --mode=flex \
  --save-grad-file=/tmp/flex.pt \
  --max_tokens_per_mb=16384 \
  --prefix-len=10

# Save stack gradients (single GPU)
torchrun --nproc_per_node=1 --master_port=29502 \
  areal/tests/torchrun/run_tree_training_distributed.py \
  --mode=stack \
  --save-grad-file=/tmp/stack_1.pt \
  --max_tokens_per_mb=16384 \
  --prefix-len=10

# Save stack gradients (2 GPUs) - prefix-len must result in batch size divisible by 2
torchrun --nproc_per_node=2 --master_port=29502 \
  areal/tests/torchrun/run_tree_training_distributed.py \
  --mode=stack \
  --save-grad-file=/tmp/stack_2.pt \
  --max_tokens_per_mb=16384 \
  --prefix-len=10

# Then use compare_gradients.py to compare them
python areal/tests/compare_gradients.py \
  --reference /tmp/baseline.pt \
  --compare /tmp/stack.pt
"""

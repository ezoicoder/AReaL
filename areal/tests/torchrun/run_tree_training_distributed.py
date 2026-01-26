"""
Distributed tree training test with torchrun.

This script tests tree attention training in a multi-GPU environment,
fully reusing the production code path including prepare_batch().
"""
import argparse
import os
import sys
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

def print_flush(msg):
    """Print with immediate flush to ensure output ordering in distributed setting."""
    print(msg)
    sys.stdout.flush()

MODEL_PATH = get_model_path(
    "/data/tree/models/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"
)

# Path to real tree training data (prefix, each rank will append _rank{R}.pt)
TREE_DATA_PATH = "/data/tree/tree-data/tau2-16k-small/call2_rank0.pt"


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


def _collect_full_params_all_ranks(engine: FSDPEngine, rank: int) -> dict[str, torch.Tensor]:
    """Collect FULL parameters on ALL ranks.
    
    Similar to _collect_full_gradients_on_rank0 but collects on all ranks.
    Handles both FSDP2 (DTensor) and ZeRO-1 (full tensor) cases.
    
    Note: Parameters are kept on GPU for faster comparison.
    
    Returns:
        Dictionary of full parameters on GPU (all ranks)
    """
    params = {}
    
    # Check if any parameter is a DTensor (FSDP2)
    has_dtensor = False
    for param in engine.model.parameters():
        if hasattr(param, 'full_tensor'):
            has_dtensor = True
            break
    
    # All ranks need to participate in DTensor operations
    for name, param in engine.model.named_parameters():
        # Check if this is a DTensor (FSDP2)
        if hasattr(param, 'full_tensor'):
            # FSDP2: ALL ranks must call full_tensor() (collective operation)
            full_param = param.full_tensor()
            clean_name = name.replace("module.", "", 1) if name.startswith("module.") else name
            params[clean_name] = full_param.detach().clone()
        elif hasattr(param, '_local_tensor'):
            # DTensor but no full_tensor method (shouldn't happen, but handle it)
            clean_name = name.replace("module.", "", 1) if name.startswith("module.") else name
            params[clean_name] = param._local_tensor.detach().clone()
        else:
            # Regular tensor (ZeRO-1/DDP): already full parameter
            clean_name = name.replace("module.", "", 1) if name.startswith("module.") else name
            params[clean_name] = param.detach().clone()
    
    return params


def _check_params_consistency_across_ranks(params: dict[str, torch.Tensor], rank: int, world_size: int, stage: str) -> bool:
    """Check if parameters are consistent across all ranks.
    
    Args:
        params: Dictionary of parameter tensors on current rank
        rank: Current rank
        world_size: Total number of ranks
        stage: Description of when this check is performed (e.g., "before training", "after training")
    
    Returns:
        True if all ranks have identical parameters, False otherwise
    """
    if world_size == 1:
        print(f"[Rank {rank}] Single rank, skipping consistency check for {stage}")
        return True
    
    all_consistent = True
    inconsistent_params = []
    
    for name, param in params.items():
        # Compute hash of this rank's parameter
        param_flat = param.flatten()
        local_sum = param_flat.sum().item()
        local_max = param_flat.max().item()
        local_min = param_flat.min().item()
        
        # Gather stats from all ranks
        sum_tensor = torch.tensor([local_sum], dtype=torch.float32, device='cuda')
        max_tensor = torch.tensor([local_max], dtype=torch.float32, device='cuda')
        min_tensor = torch.tensor([local_min], dtype=torch.float32, device='cuda')
        
        # Gather to rank 0
        if rank == 0:
            sum_list = [torch.zeros_like(sum_tensor) for _ in range(world_size)]
            max_list = [torch.zeros_like(max_tensor) for _ in range(world_size)]
            min_list = [torch.zeros_like(min_tensor) for _ in range(world_size)]
        else:
            sum_list = None
            max_list = None
            min_list = None
        
        dist.gather(sum_tensor, sum_list, dst=0)
        dist.gather(max_tensor, max_list, dst=0)
        dist.gather(min_tensor, min_list, dst=0)
        
        # Check consistency on rank 0
        if rank == 0:
            sums = [t.item() for t in sum_list]
            maxs = [t.item() for t in max_list]
            mins = [t.item() for t in min_list]
            
            # Check if all ranks have the same values (with tolerance for floating point errors)
            sum_diff = max(sums) - min(sums)
            max_diff = max(maxs) - min(maxs)
            min_diff = max(mins) - min(mins)
            
            # Use relative tolerance
            tolerance = 1e-6
            is_consistent = (sum_diff < tolerance * abs(sums[0]) if sums[0] != 0 else sum_diff < tolerance)
            
            if not is_consistent:
                all_consistent = False
                inconsistent_params.append((name, sum_diff, max_diff, min_diff))
    
    # Broadcast result to all ranks
    result_tensor = torch.tensor([1.0 if all_consistent else 0.0], dtype=torch.float32, device='cuda')
    dist.broadcast(result_tensor, src=0)
    all_consistent = (result_tensor.item() > 0.5)
    
    # Print results on rank 0
    if rank == 0:
        print(f"[Rank 0] Parameter consistency check {stage}:")
        if all_consistent:
            print(f"[Rank 0]   ✓ All parameters are consistent across {world_size} ranks")
        else:
            print(f"[Rank 0]   ✗ Found {len(inconsistent_params)} inconsistent parameters:")
            for name, sum_diff, max_diff, min_diff in inconsistent_params[:5]:  # Show first 5
                print(f"[Rank 0]     - {name}: sum_diff={sum_diff:.6e}, max_diff={max_diff:.6e}, min_diff={min_diff:.6e}")
            if len(inconsistent_params) > 5:
                print(f"[Rank 0]     ... and {len(inconsistent_params) - 5} more")
    
    return all_consistent


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
        stack_block_size=stack_block_size,
        stack_depth=stack_depth,
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
    except Exception as e:
        # Print exception details before cleanup
        import traceback
        print_flush(f"[Rank {rank}] ⚠️  EXCEPTION in engine context: {type(e).__name__}: {e}")
        print_flush(f"[Rank {rank}] Traceback:\n{traceback.format_exc()}")
        raise  # Re-raise to let outer code handle it
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
    disable_optimizer: bool = True,
    skip_param_comparison: bool = False,
):
    """Run training for a single mode and save gradients to file.
    
    Args:
        mode: Training mode - "baseline", "flex", or "stack"
        max_tokens_per_mb: Max tokens per microbatch
        prefix_len: Number of sequences to keep
        gradient_checkpointing: Whether to enable gradient checkpointing (only for baseline/flex)
        save_grad_file: Path to save gradients (rank 0 only)
        compare_grad_file: Deprecated, kept for compatibility
        disable_optimizer: Whether to disable optimizer (if False, will check parameter updates)
        skip_param_comparison: If True, skip parameter collection/comparison (for memory profiling only)
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
    print(f"[Rank {rank}] Running {mode} training (world_size={world_size}, disable_optimizer={disable_optimizer})")
    
    # Setup engine and run training
    with setup_engine_with_mock_rollout(
        rank, world_size, mode, max_tokens_per_mb,
        prefix_len=prefix_len,
        enable_tree_stack_training=config["enable_tree_stack_training"],
        enable_tree_training=config["enable_tree_training"],
        is_tree_distribution=config["is_tree_distribution"],
        gradient_checkpointing=config["use_gradient_checkpointing"],
        disable_optimizer=disable_optimizer,
    ) as engine:
        
        print(f"[Rank {rank}] Calling {mode}_engine.prepare_batch()...")
        input_data = engine.prepare_batch(dataloader=None, workflow=None)
        
        # Collect parameters before training (ALL ranks if optimizer is enabled and comparison is not skipped)
        params_before = None
        if not disable_optimizer and not skip_param_comparison:
            print(f"[Rank {rank}] Collecting parameters before training...")
            params_before = _collect_full_params_all_ranks(engine, rank)
            print(f"[Rank {rank}] Collected {len(params_before)} parameter tensors before training")
            
            # Check parameter consistency across ranks
            _check_params_consistency_across_ranks(params_before, rank, world_size, "before training")
        elif not disable_optimizer and skip_param_comparison:
            print(f"[Rank {rank}] Optimizer enabled but parameter comparison skipped (--enable-optimizer-no-comp)")
        
        # Train
        reset_peak_memory()
        engine.train()
        torch.cuda.synchronize()
        start = time.time()
        
        _,loss = engine.train_batch(input_data, loss_fn=loss_fn, loss_weight_fn=loss_weight_fn, required_loss=True)
        for _ in range(repeat):
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
        
        # Check parameter updates (ALL ranks if optimizer is enabled and comparison is not skipped)
        params_updated = False
        params_after = None
        if not disable_optimizer and not skip_param_comparison:
            print(f"[Rank {rank}] Collecting parameters after training...")
            params_after = _collect_full_params_all_ranks(engine, rank)
            print(f"[Rank {rank}] Collected {len(params_after)} parameter tensors after training")
            
            # Check parameter consistency across ranks after training
            _check_params_consistency_across_ranks(params_after, rank, world_size, "after training")
            
            # Each rank checks if its own parameters were updated
            if params_before and params_after:
                print(f"[Rank {rank}] Checking if parameters were updated...")
                
                # Check if any parameter changed
                num_changed = 0
                num_unchanged = 0
                max_change = 0.0
                max_change_param = ""
                
                for name in params_before.keys():
                    if name in params_after:
                        param_before = params_before[name]
                        param_after = params_after[name]
                        
                        # Calculate absolute difference
                        diff = torch.abs(param_after - param_before)
                        max_diff = diff.max().item()
                        
                        if max_diff > max_change:
                            max_change = max_diff
                            max_change_param = name
                        
                        # Check if parameter changed (with small tolerance for numerical errors)
                        if max_diff > 1e-8:
                            num_changed += 1
                        else:
                            num_unchanged += 1
                
                params_updated = num_changed > 0
                
                print(f"[Rank {rank}] Parameter update check:")
                print(f"[Rank {rank}]   - Changed: {num_changed} parameters")
                print(f"[Rank {rank}]   - Unchanged: {num_unchanged} parameters")
                print(f"[Rank {rank}]   - Max change: {max_change:.6e} ({max_change_param})")
                print(f"[Rank {rank}]   - Parameters updated: {'✓ YES' if params_updated else '✗ NO'}")
                
                if not params_updated:
                    print(f"[Rank {rank}] ⚠️  WARNING: No parameters were updated despite optimizer being enabled!")
            
            # Synchronize params_updated across ranks and verify consistency
            if world_size > 1:
                updated_tensor = torch.tensor([1.0 if params_updated else 0.0], dtype=torch.float32, device='cuda')
                # Gather all ranks' params_updated status to rank 0
                if rank == 0:
                    updated_list = [torch.zeros_like(updated_tensor) for _ in range(world_size)]
                else:
                    updated_list = None
                dist.gather(updated_tensor, updated_list, dst=0)
                
                if rank == 0:
                    all_updated = [t.item() > 0.5 for t in updated_list]
                    if not all(u == all_updated[0] for u in all_updated):
                        print(f"[Rank 0] ⚠️  WARNING: params_updated status inconsistent across ranks: {all_updated}")
    
    # Save gradients and parameter update info (rank 0 only)
    if rank == 0 and save_grad_file and grads:
        print(f"[Rank 0] Saving {mode} gradients to {save_grad_file}")
        print(f"[Rank 0] Average loss across {world_size} ranks: {loss_sum:.6f}")
        
        save_data = {
            "mode": mode,
            "grads": grads,
            "time": train_time,
            "mem": train_mem,
            "loss_sum": loss_sum,  # Average loss across all ranks
            "config": {
                "max_tokens_per_mb": max_tokens_per_mb,
                "prefix_len": prefix_len,
                "gradient_checkpointing": config["use_gradient_checkpointing"],
                "disable_optimizer": disable_optimizer,
                "skip_param_comparison": skip_param_comparison,
                "world_size": world_size,
            }
        }
        
        # Add parameter update info if optimizer was enabled and comparison was not skipped
        # (rank 0's params represent all ranks after consistency check)
        if not disable_optimizer and not skip_param_comparison and params_before and params_after:
            save_data["params_updated"] = params_updated
            save_data["params_before"] = params_before
            save_data["params_after"] = params_after
            save_data["params_consistency_verified"] = True  # We verified consistency above
        
        torch.save(save_data, save_grad_file)
        print(f"[Rank 0] ✓ Saved {len(grads)} gradient tensors")
        if not disable_optimizer and not skip_param_comparison:
            print(f"[Rank 0] ✓ Saved parameter update info (consistency verified across {world_size} ranks)")
        elif not disable_optimizer and skip_param_comparison:
            print(f"[Rank 0] ✓ Optimizer enabled for memory profiling (parameter comparison skipped)")
    
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
    parser.add_argument("--enable-optimizer", action="store_true", default=False,
                       help="Enable optimizer to check parameter updates (default: disabled)")
    parser.add_argument("--enable-optimizer-no-comp", action="store_true", default=False,
                       help="Enable optimizer only for memory profiling, skip parameter collection/comparison (default: disabled)")
    parser.add_argument("--stack-block-size", type=int, default=4096,
                       help="Stack block size for tree stack training")
    parser.add_argument("--stack-depth", type=int, default=16384,
                       help="Stack depth for tree stack training")
    parser.add_argument("--repeat", type=int, default=0,
                       help="Repeat the training")
    args = parser.parse_args()
    
    # Check mutual exclusivity of --enable-optimizer and --enable-optimizer-no-comp
    if args.enable_optimizer and args.enable_optimizer_no_comp:
        parser.error("--enable-optimizer and --enable-optimizer-no-comp cannot be used together")
    
    global stack_block_size
    global stack_depth
    global repeat
    stack_block_size = args.stack_block_size
    stack_depth = args.stack_depth
    repeat = args.repeat

    gradient_checkpointing = not args.disable_gradient_checkpointing
    disable_optimizer = not (args.enable_optimizer or args.enable_optimizer_no_comp)
    skip_param_comparison = args.enable_optimizer_no_comp  # Skip comparison when using no-comp mode
    
    run_single_training(
        mode=args.mode,
        max_tokens_per_mb=args.max_tokens_per_mb,
        prefix_len=args.prefix_len,
        gradient_checkpointing=gradient_checkpointing,
        save_grad_file=args.save_grad_file,
        compare_grad_file=None,  # No comparison, only save
        disable_optimizer=disable_optimizer,
        skip_param_comparison=skip_param_comparison,
    )


if __name__ == "__main__":
    main()

"""
Usage examples:

torchrun --nproc_per_node=2 --master_port=29500 \
  areal/tests/torchrun/run_tree_training_distributed.py \
  --prefix-len=20 \
  --mode=stack \
  --stack-block-size=768 \
  --save-grad-file=/tmp/stack.pt \
  --enable-optimizer \
  --repeat=2

# Save baseline gradients (single GPU)
torchrun --nproc_per_node=1 --master_port=29500 \
  areal/tests/torchrun/run_tree_training_distributed.py \
  --prefix-len 20 \
  --mode=baseline \
  --save-grad-file=/tmp/baseline.pt \
  --max_tokens_per_mb=16384

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
  --prefix-len=10

# Save stack gradients (2 GPUs) - prefix-len must result in batch size divisible by 2
torchrun --nproc_per_node=2 --master_port=29502 \
  areal/tests/torchrun/run_tree_training_distributed.py \
  --mode=stack \
  --save-grad-file=/tmp/stack_2.pt \
  --max_tokens_per_mb=16384 \
  --prefix-len=10

# Enable optimizer to check parameter updates
torchrun --nproc_per_node=2 --master_port=29503 \
  areal/tests/torchrun/run_tree_training_distributed.py \
  --mode=stack \
  --save-grad-file=/tmp/stack_with_opt.pt \
  --max_tokens_per_mb=16384 \
  --enable-optimizer

# Enable optimizer only for memory profiling (skip parameter collection/comparison)
torchrun --nproc_per_node=2 --master_port=29504 \
  areal/tests/torchrun/run_tree_training_distributed.py \
  --mode=stack \
  --save-grad-file=/tmp/stack_mem_profile.pt \
  --max_tokens_per_mb=16384 \
  --enable-optimizer-no-comp

# Then use compare_gradients.py to compare them
python areal/tests/compare_gradients.py \
  --reference /tmp/baseline.pt \
  --compare /tmp/stack.pt
"""

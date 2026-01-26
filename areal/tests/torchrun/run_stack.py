"""
Stack training/forward with is_tree_distribution ablation study.

This script provides a high-level interface for running stack training or forward pass experiments.
The interface function handles torchrun initialization and cleanup automatically.

High-level API usage:
    from areal.tests.torchrun.run_stack import run_stack_experiment
    
    # Run training
    results = run_stack_experiment(
        model_path="/path/to/model",
        data_path="/path/to/data.pt",
        stack_block_size=768,
        world_size=2,
        is_tree_distribution=True,
        jsonl_path="/path/to/results.jsonl",
        run_forward=False
    )
    
    # Run forward pass only
    results = run_stack_experiment(
        model_path="/path/to/model",
        data_path="/path/to/data.pt",
        stack_block_size=768,
        world_size=2,
        is_tree_distribution=True,
        jsonl_path="/path/to/results.jsonl",
        run_forward=True
    )

Command-line usage (for testing):
    # Training
    python areal/tests/torchrun/run_stack.py \
      --model-path /data/tree/models/Qwen3-4B \
      --data-path /data/tree/tree-data/tau2-16k-small/call2.pt \
      --world-size 2 \
      --is-tree-distribution \
      --jsonl-path ../results.jsonl
    
    # Forward pass only
    python areal/tests/torchrun/run_stack.py \
      --model-path /data/tree/models/Qwen3-4B \
      --data-path /data/tree/tree-data/tau2-16k-small/call2.pt \
      --world-size 2 \
      --is-tree-distribution \
      --run-forward \
      --jsonl-path ../results.jsonl
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from areal.api.alloc_mode import AllocationMode
from areal.api.cli_args import TrainEngineConfig, MicroBatchSpec, FSDPEngineConfig
from areal.api.io_struct import FinetuneSpec, WeightUpdateMeta
from areal.engine.fsdp_engine import FSDPEngine
from areal.engine.ppo.actor import grpo_loss_fn
from areal.platforms import current_platform
from areal.utils.data import tensor_container_to
from areal.utils import logging
from functools import partial

# Create logger
logger = logging.getLogger("StackTraining")


def print_flush(msg):
    """Print with immediate flush to ensure output ordering in distributed setting."""
    print(msg)
    sys.stdout.flush()


# Fixed loss function for GRPO
def loss_fn(logprobs, entropy, input_data):
    return -logprobs.sum() / logprobs.shape[0]


def loss_weight_fn(input_data):
    """Default weight function based on loss_mask sum"""
    return torch.tensor(1.0)


class MockRolloutEngine:
    """Mock rollout engine that loads data from file.
    
    Implements the InferenceEngine interface required by FSDPEngine.connect_engine().
    All ranks load from the same data file, then split data by rank.
    """
    def __init__(self, data_path: str, rank: int, world_size: int):
        """
        Args:
            data_path: Path to data file (e.g., "/path/to/data.pt")
            rank: Rank number to determine which slice to take
            world_size: Total number of ranks
        """
        self.data_path = data_path
        self.rank = rank
        self.world_size = world_size
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
        """Load data from disk and return as trajectories list."""
        if self._data_cache is None:
            # All ranks load from the same data file
            data = torch.load(self.data_path, map_location='cpu')
            input_data = data["input_data"]
            
            # Get total batch size
            total_batch_size = input_data["attention_mask"].shape[0]
            
            # Ensure batch size is divisible by world_size
            if total_batch_size % self.world_size != 0:
                raise ValueError(
                    f"Batch size {total_batch_size} must be divisible by world_size {self.world_size}."
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
    model_path: str,
    data_path: str,
    stack_block_size: int,
    is_tree_distribution: bool,
):
    """Setup FSDP engine with mock rollout engine for stack training.
    
    Args:
        rank: Current rank
        world_size: Total number of ranks
        model_path: Path to model
        data_path: Path to data file
        stack_block_size: Stack block size for tree stack training
        is_tree_distribution: Whether to use tree-based data distribution
    
    Yields:
        Initialized engine instance
    """
    config = TrainEngineConfig(
        experiment_name="stack_ablation",
        trial_name="distributed_test",
        path=model_path,
        mb_spec=MicroBatchSpec(max_tokens_per_mb=16384),  # Fixed default
        optimizer=None,
        enable_tree_training=False,
        enable_tree_stack_training=True,
        gradient_checkpointing=False,  # Stack doesn't support checkpointing
        disable_optimizer=True,  # Always disabled for this ablation study
        is_tree_distribution=is_tree_distribution,
        stack_block_size=stack_block_size,
        stack_depth=16384,  # Fixed default
        fsdp=FSDPEngineConfig(),
    )
    
    alloc_mode = AllocationMode.from_str(f"d{world_size}p1t1")
    ft_spec = FinetuneSpec(total_train_epochs=1, dataset_size=128, train_batch_size=8)
    
    engine = FSDPEngine(config)
    engine.create_process_group(alloc_mode.train)
    engine.initialize(addr=None, ft_spec=ft_spec, parallel_strategy=alloc_mode.train)
    
    # Create and connect mock rollout engine
    mock_engine = MockRolloutEngine(data_path, rank=rank, world_size=world_size)
    meta = WeightUpdateMeta(type="disk")
    engine.connect_engine(mock_engine, meta)
    
    print(f"[Rank {rank}] Engine initialized: is_DP_head={engine.is_data_parallel_head()}, is_tree_distribution={is_tree_distribution}")
    
    try:
        yield engine
    except Exception as e:
        import traceback
        print_flush(f"[Rank {rank}] ⚠️  EXCEPTION in engine context: {type(e).__name__}: {e}")
        print_flush(f"[Rank {rank}] Traceback:\n{traceback.format_exc()}")
        raise
    finally:
        print(f"[Rank {rank}] Destroying engine")
        
        # Critical: Synchronize BEFORE destroy()
        dist.barrier()
        
        # Destroy the model and optimizer
        engine.destroy()
        
        print(f"[Rank {rank}] Engine cleanup complete")


def get_memory_stats(stage: str) -> dict:
    """Get current GPU memory statistics."""
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)  # GB
    peak_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB
    
    return {
        "stage": stage,
        "allocated_gb": allocated,
        "reserved_gb": reserved,
        "peak_allocated_gb": peak_allocated,
    }


def reset_peak_memory():
    """Reset peak memory statistics."""
    torch.cuda.reset_peak_memory_stats()


def _run_training_worker(
    model_path: str,
    data_path: str,
    stack_block_size: int,
    is_tree_distribution: bool,
    temp_result_file: str,
    run_forward: bool = False,
):
    """Internal worker function that runs in distributed environment.
    
    This function is called by torchrun and expects RANK/WORLD_SIZE to be set.
    It performs the actual training or forward pass and saves results to a temporary file.
    
    Args:
        model_path: Path to model
        data_path: Path to training data
        stack_block_size: Stack block size for tree attention
        is_tree_distribution: Whether to use tree-based data distribution
        temp_result_file: Path to temporary file for saving results
        run_forward: If True, run forward pass only; if False, run training (default: False)
    """
    # Get rank and world_size from environment (set by torchrun)
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    mode_str = "forward" if run_forward else "training"
    print(f"[Rank {rank}] Running stack {mode_str} (world_size={world_size}, is_tree_distribution={is_tree_distribution})")
    
    # Setup engine and run training/forward
    with setup_engine_with_mock_rollout(
        rank, world_size, model_path, data_path, stack_block_size, is_tree_distribution
    ) as engine:
        
        print(f"[Rank {rank}] Calling engine.prepare_batch()...")
        input_data = engine.prepare_batch(dataloader=None, workflow=None)
        
        # Reset peak memory before operation
        reset_peak_memory()
        
        if run_forward:
            # Forward pass mode
            engine.eval()
            torch.cuda.synchronize()
            
            # Run forward twice, only record second iteration
            print(f"[Rank {rank}] Running warmup forward iteration...")
            _ = engine.forward_batch(input_=input_data)
            
            # Synchronize all ranks after warmup
            torch.cuda.synchronize()
            dist.barrier()
            print(f"[Rank {rank}] Warmup complete, synchronizing...")
            
            # Reset peak memory again for accurate measurement
            reset_peak_memory()
            
            # Second iteration - record metrics
            print(f"[Rank {rank}] Running measured forward iteration...")
            start_time = time.time()
            
            logprobs = engine.forward_batch(input_=input_data)
            
            torch.cuda.synchronize()
            local_time = time.time() - start_time
            
            # Synchronize across all ranks
            dist.barrier()
            
            # Collect local metrics
            local_mem = get_memory_stats("Stack Forward")
            
            print(f"[Rank {rank}] Stack Forward: time={local_time:.4f}s, logprobs_shape={logprobs.shape if logprobs is not None else None}")
            print(f"[Rank {rank}] Stack Forward: mem=[alloc={local_mem['allocated_gb']:.2f}GB, reserved={local_mem['reserved_gb']:.2f}GB, peak={local_mem['peak_allocated_gb']:.2f}GB]")
            
            # For forward pass, we don't have loss or token_trie_info
            loss = 0.0
            token_trie_info = {}
            
        else:
            # Training mode
            engine.train()
            torch.cuda.synchronize()
            
            # Run training twice, only record second iteration
            print(f"[Rank {rank}] Running warmup training iteration...")
            _, _ = engine.train_batch(input_data, loss_fn=loss_fn, loss_weight_fn=loss_weight_fn, required_loss=True)
            
            # Synchronize all ranks after warmup
            torch.cuda.synchronize()
            dist.barrier()
            print(f"[Rank {rank}] Warmup complete, synchronizing...")
            
            # Reset peak memory again for accurate measurement
            reset_peak_memory()
            
            # Second iteration - record metrics
            print(f"[Rank {rank}] Running measured training iteration...")
            start_time = time.time()
            
            token_trie_info, loss = engine.train_batch(
                input_data, 
                loss_fn=loss_fn, 
                loss_weight_fn=loss_weight_fn, 
                required_loss=True,
                required_token_trie_info=True,
            )
            
            torch.cuda.synchronize()
            local_time = time.time() - start_time
            
            # Synchronize across all ranks
            dist.barrier()
            
            # Collect local metrics
            local_mem = get_memory_stats("Stack Training")
            
            print(f"[Rank {rank}] Stack Training: time={local_time:.4f}s, loss={loss:.6f}")
            print(f"[Rank {rank}] Stack Training: mem=[alloc={local_mem['allocated_gb']:.2f}GB, reserved={local_mem['reserved_gb']:.2f}GB, peak={local_mem['peak_allocated_gb']:.2f}GB]")
            print(f"[Rank {rank}] Stack Training: token_trie_info={token_trie_info}")
        
        # Gather all metrics to rank 0
        # Collect time from all ranks
        time_tensor = torch.tensor([local_time], dtype=torch.float32, device='cuda')
        if rank == 0:
            time_list = [torch.zeros_like(time_tensor) for _ in range(world_size)]
        else:
            time_list = None
        dist.gather(time_tensor, time_list, dst=0)
        
        # Collect memory from all ranks
        mem_tensor = torch.tensor([
            local_mem["allocated_gb"],
            local_mem["reserved_gb"],
            local_mem["peak_allocated_gb"]
        ], dtype=torch.float32, device='cuda')
        if rank == 0:
            mem_list = [torch.zeros_like(mem_tensor) for _ in range(world_size)]
        else:
            mem_list = None
        dist.gather(mem_tensor, mem_list, dst=0)
        
        # Collect loss from all ranks
        loss_tensor = torch.tensor([loss], dtype=torch.float32, device='cuda')
        if rank == 0:
            loss_list = [torch.zeros_like(loss_tensor) for _ in range(world_size)]
        else:
            loss_list = None
        dist.gather(loss_tensor, loss_list, dst=0)
        
        # Collect token_trie_info from all ranks
        # Convert token_trie_info dict to tensor for gathering
        if token_trie_info:
            token_info_keys = sorted(token_trie_info.keys())
            token_info_values = [token_trie_info[k] for k in token_info_keys]
            token_info_tensor = torch.tensor(token_info_values, dtype=torch.float32, device='cuda')
        else:
            token_info_keys = []
            token_info_tensor = torch.tensor([], dtype=torch.float32, device='cuda')
        
        if rank == 0:
            token_info_list = [torch.zeros_like(token_info_tensor) for _ in range(world_size)]
        else:
            token_info_list = None
        
        if token_info_tensor.numel() > 0:
            dist.gather(token_info_tensor, token_info_list, dst=0)
        
        # Build results on rank 0
        if rank == 0:
            per_rank_metrics = []
            
            for r in range(world_size):
                rank_time = time_list[r].item()
                rank_mem = {
                    "allocated_gb": mem_list[r][0].item(),
                    "reserved_gb": mem_list[r][1].item(),
                    "peak_allocated_gb": mem_list[r][2].item(),
                }
                rank_loss = loss_list[r].item()
                
                # Reconstruct token_trie_info
                if token_info_list and token_info_keys:
                    rank_token_info = {
                        token_info_keys[i]: token_info_list[r][i].item() 
                        for i in range(len(token_info_keys))
                    }
                else:
                    rank_token_info = {}
                
                per_rank_metrics.append({
                    "rank": r,
                    "time_seconds": rank_time,
                    "memory": rank_mem,
                    "loss": rank_loss,
                    "token_info": rank_token_info,
                })
            
            # Calculate aggregate metrics
            max_time = max(m["time_seconds"] for m in per_rank_metrics)
            max_memory = {
                "allocated_gb": max(m["memory"]["allocated_gb"] for m in per_rank_metrics),
                "reserved_gb": max(m["memory"]["reserved_gb"] for m in per_rank_metrics),
                "peak_allocated_gb": max(m["memory"]["peak_allocated_gb"] for m in per_rank_metrics),
            }
            avg_loss = sum(m["loss"] for m in per_rank_metrics) / world_size
            
            # Sum token_trie_info across ranks
            if token_info_keys:
                total_token_info = {}
                for key in token_info_keys:
                    total_token_info[key] = sum(m["token_info"].get(key, 0) for m in per_rank_metrics)
            else:
                total_token_info = {}
            
            results = {
                "per_rank_metrics": per_rank_metrics,
                "aggregate_metrics": {
                    "max_time_seconds": max_time,
                    "max_memory": max_memory,
                    "avg_loss": avg_loss,
                    "total_token_info": total_token_info,
                    "world_size": world_size,
                }
            }
            
            # Print aggregate results
            print(f"[Rank 0] ===== Aggregate Results =====")
            print(f"[Rank 0] Max time: {max_time:.4f}s")
            print(f"[Rank 0] Max memory: alloc={max_memory['allocated_gb']:.2f}GB, reserved={max_memory['reserved_gb']:.2f}GB, peak={max_memory['peak_allocated_gb']:.2f}GB")
            print(f"[Rank 0] Avg loss: {avg_loss:.6f}")
            print(f"[Rank 0] Total token info: {total_token_info}")
            
            # Save results to temporary file
            with open(temp_result_file, 'w') as f:
                json.dump(results, f)
            
            print(f"[Rank 0] Results saved to {temp_result_file}")
    
    print(f"[Rank {rank}] Stack {mode_str} complete")


def run_stack_experiment(
    model_path: str,
    data_path: str,
    stack_block_size: int,
    world_size: int,
    is_tree_distribution: bool,
    jsonl_path: str,
    master_port: int = 29500,
    run_forward: bool = False,
) -> dict:
    """High-level interface for running stack training/forward experiment.
    
    This function automatically launches torchrun, runs the training or forward pass,
    collects results, and cleans up the distributed environment.
    
    Args:
        model_path: Path to model directory
        data_path: Path to training data (.pt file)
        stack_block_size: Stack block size for tree attention
        world_size: Number of GPUs to use
        is_tree_distribution: Whether to use tree-based data distribution
        jsonl_path: Path to JSONL file for logging results
        master_port: Master port for distributed training (default: 29500)
        run_forward: If True, run forward pass only; if False, run training (default: False)
    
    Returns:
        Dictionary containing performance metrics:
        {
            "per_rank_metrics": [...],
            "aggregate_metrics": {...}
        }
    """
    # Check if we're already in a distributed environment
    if "RANK" in os.environ:
        raise RuntimeError(
            "run_stack_experiment() should not be called from within a distributed environment. "
            "It will launch torchrun internally."
        )
    
    # Validate inputs
    if not Path(model_path).exists():
        raise ValueError(f"Model path does not exist: {model_path}")
    if not Path(data_path).exists():
        raise ValueError(f"Data path does not exist: {data_path}")
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")
    
    # Create temporary file for results
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_result_file = f.name
    
    try:
        # Get the path to this script
        script_path = Path(__file__).resolve()
        
        # Build torchrun command
        cmd = [
            "torchrun",
            f"--nproc_per_node={world_size}",
            f"--master_port={master_port}",
            str(script_path),
            "--internal-worker",  # Special flag to indicate worker mode
            "--model-path", model_path,
            "--data-path", data_path,
            "--stack-block-size", str(stack_block_size),
            "--temp-result-file", temp_result_file,
            "--jsonl-path", jsonl_path,
        ]
        
        if is_tree_distribution:
            cmd.append("--is-tree-distribution")
        
        if run_forward:
            cmd.append("--run-forward")

        
        print(f"Launching torchrun with {world_size} GPUs...")
        print(f"Command: {' '.join(cmd)}")
        print()
        
        # Run torchrun
        result = subprocess.run(cmd, check=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"torchrun failed with exit code {result.returncode}")
        
        # Load results from temporary file
        with open(temp_result_file, 'r') as f:
            results = json.load(f)
        
        # Append to JSONL file
        log_entry = {
            "input": {
                "model_path": model_path,
                "data_path": data_path,
                "stack_block_size": stack_block_size,
                "world_size": world_size,
                "is_tree_distribution": is_tree_distribution,
                "run_forward": run_forward,
            },
            "output": results,
        }
        
        # Create parent directory if it doesn't exist
        Path(jsonl_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(jsonl_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        print()
        print(f"✓ Experiment complete!")
        print(f"✓ Results logged to {jsonl_path}")
        
        return results
        
    finally:
        # Clean up temporary file
        if Path(temp_result_file).exists():
            Path(temp_result_file).unlink()


def main():
    parser = argparse.ArgumentParser(
        description="Run stack training/forward with is_tree_distribution ablation"
    )
    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to model")
    parser.add_argument("--data-path", type=str, required=True,
                       help="Path to training data (e.g., /path/to/data.pt)")
    parser.add_argument("--stack-block-size", type=int, default=4096,
                       help="Stack block size for tree attention (default: 4096)")
    parser.add_argument("--world-size", type=int, default=2,
                       help="Number of GPUs to use (default: 2)")
    parser.add_argument("--is-tree-distribution", action="store_true", default=False,
                       help="Enable tree-based data distribution (default: False)")
    parser.add_argument("--run-forward", action="store_true", default=False,
                       help="Run forward pass only instead of training (default: False)")
    parser.add_argument("--jsonl-path", type=str, required=True,
                       help="Path to JSONL file for logging results")
    parser.add_argument("--master-port", type=int, default=29500,
                       help="Master port for distributed training (default: 29500)")
    parser.add_argument("--internal-worker", action="store_true", default=False,
                       help="Internal flag: indicates this is a worker process launched by torchrun")
    parser.add_argument("--temp-result-file", type=str, default=None,
                       help="Internal flag: temporary file for worker results")
    args = parser.parse_args()
    
    # Check if this is a worker process (launched by torchrun)
    if args.internal_worker:
        # This is a worker process - run the actual training/forward
        if args.temp_result_file is None:
            raise ValueError("--temp-result-file is required for worker mode")
        
        _run_training_worker(
            model_path=args.model_path,
            data_path=args.data_path,
            stack_block_size=args.stack_block_size,
            is_tree_distribution=args.is_tree_distribution,
            temp_result_file=args.temp_result_file,
            run_forward=args.run_forward,
        )
    else:
        # This is the main process - launch torchrun
        results = run_stack_experiment(
            model_path=args.model_path,
            data_path=args.data_path,
            stack_block_size=args.stack_block_size,
            world_size=args.world_size,
            is_tree_distribution=args.is_tree_distribution,
            jsonl_path=args.jsonl_path,
            master_port=args.master_port,
            run_forward=args.run_forward,
        )
        
        # Print final JSON results
        print("\n===== Final Results (JSON) =====")
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

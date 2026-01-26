#!/usr/bin/env python3
"""
Example: How to use run_stack_experiment for ablation study.

This script demonstrates the high-level API usage without needing to
manually call torchrun. Everything is handled automatically.
"""
from pathlib import Path
from areal.tests.torchrun.run_stack import run_stack_experiment


def main():
    # Configuration
    model_path = "/data/tree/models/Qwen2.5-1.5B-Instruct"
    data_path = "/data/tree/tree-data/tau2-16k-small/call2_rank0.pt"
    stack_block_size = 768
    world_size = 2  # Number of GPUs
    jsonl_path = "/tmp/stack_ablation_results.jsonl"
    
    # Validate paths
    if not Path(model_path).exists():
        print(f"Error: Model path does not exist: {model_path}")
        print("Please update the model_path in this script.")
        return
    
    if not Path(data_path).exists():
        print(f"Error: Data file does not exist: {data_path}")
        print("Please update the data_path in this script.")
        return
    
    print("=" * 80)
    print("Stack Training Ablation Study")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Data: {data_path}")
    print(f"Stack Block Size: {stack_block_size}")
    print(f"GPUs: {world_size}")
    print(f"Output: {jsonl_path}")
    print("=" * 80)
    print()
    
    # Clear previous results
    if Path(jsonl_path).exists():
        Path(jsonl_path).unlink()
    
    # Experiment 1: Without tree distribution
    print("[1/2] Running WITHOUT tree distribution...")
    print("-" * 80)
    results_baseline = run_stack_experiment(
        model_path=model_path,
        data_path=data_path,
        stack_block_size=stack_block_size,
        world_size=world_size,
        is_tree_distribution=False,
        jsonl_path=jsonl_path,
        master_port=29500,
    )
    
    print()
    print("[1/2] ✓ Baseline complete!")
    print(f"  Max time: {results_baseline['aggregate_metrics']['max_time_seconds']:.4f}s")
    print(f"  Peak memory: {results_baseline['aggregate_metrics']['max_memory']['peak_allocated_gb']:.2f} GB")
    print()
    
    # Experiment 2: With tree distribution
    print("[2/2] Running WITH tree distribution...")
    print("-" * 80)
    results_tree = run_stack_experiment(
        model_path=model_path,
        data_path=data_path,
        stack_block_size=stack_block_size,
        world_size=world_size,
        is_tree_distribution=True,
        jsonl_path=jsonl_path,
        master_port=29501,  # Different port to avoid conflicts
    )
    
    print()
    print("[2/2] ✓ Tree distribution complete!")
    print(f"  Max time: {results_tree['aggregate_metrics']['max_time_seconds']:.4f}s")
    print(f"  Peak memory: {results_tree['aggregate_metrics']['max_memory']['peak_allocated_gb']:.2f} GB")
    print()
    
    # Compare results
    print("=" * 80)
    print("Comparison Summary")
    print("=" * 80)
    
    time_baseline = results_baseline['aggregate_metrics']['max_time_seconds']
    time_tree = results_tree['aggregate_metrics']['max_time_seconds']
    time_speedup = (time_baseline / time_tree - 1) * 100
    
    mem_baseline = results_baseline['aggregate_metrics']['max_memory']['peak_allocated_gb']
    mem_tree = results_tree['aggregate_metrics']['max_memory']['peak_allocated_gb']
    mem_reduction = (1 - mem_tree / mem_baseline) * 100
    
    print(f"Time:")
    print(f"  Baseline: {time_baseline:.4f}s")
    print(f"  Tree: {time_tree:.4f}s")
    print(f"  → {'Speedup' if time_speedup > 0 else 'Slowdown'}: {abs(time_speedup):.2f}%")
    print()
    
    print(f"Peak Memory:")
    print(f"  Baseline: {mem_baseline:.2f} GB")
    print(f"  Tree: {mem_tree:.2f} GB")
    print(f"  → {'Reduction' if mem_reduction > 0 else 'Increase'}: {abs(mem_reduction):.2f}%")
    print()
    
    print("=" * 80)
    print(f"✓ All results saved to: {jsonl_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()


"""
Compare gradients from different training modes.

This script loads gradient files saved by run_tree_training_distributed.py
and compares them for correctness.

Usage:
    python areal/tests/compare_gradients.py \
        --reference /tmp/baseline.pt \
        --compare /tmp/flex.pt

    python areal/tests/compare_gradients.py \
        --reference /tmp/baseline.pt \
        --compare /tmp/stack.pt
"""

import argparse
from pathlib import Path

import torch

from areal.tests.test_tree_training import _compare_and_assert_gradients
from areal.utils import logging

logger = logging.getLogger("CompareGradients")


def load_gradient_file(filepath: str, device: str = 'cuda:0') -> dict:
    """Load gradient file and return data dict.
    
    Args:
        filepath: Path to gradient file
        device: Device to load tensors to (default: 'cuda:0')
    
    Returns:
        Dictionary with gradient data
    """
    with open(filepath, "rb") as f:
        # Load to specified device (default: cuda:0, same as saved)
        data = torch.load(f, map_location=device)
    return data


def compare_gradients(
    reference_file: str, 
    compare_file: str, 
    device: str = 'cuda:0',
    mean_rel_diff_threshold: float = 0.3,
):
    """Compare gradients from two files.
    
    Args:
        reference_file: Path to reference gradient file
        compare_file: Path to comparison gradient file
        device: Device to load gradients to (default: 'cuda:0')
        mean_rel_diff_threshold: Threshold for mean relative difference (default: 0.3)
    
    Returns:
        True if gradients match, False otherwise
    """
    print(f"Loading reference gradients from: {reference_file}")
    ref_data = load_gradient_file(reference_file, device=device)
    
    print(f"Loading comparison gradients from: {compare_file}")
    cmp_data = load_gradient_file(compare_file, device=device)
    
    ref_mode = ref_data["mode"]
    cmp_mode = cmp_data["mode"]
    ref_grads = ref_data["grads"]
    cmp_grads = cmp_data["grads"]
    ref_time = ref_data["time"]
    cmp_time = cmp_data["time"]
    ref_mem = ref_data["mem"]
    cmp_mem = cmp_data["mem"]
    ref_loss = ref_data.get("loss_sum", None)
    cmp_loss = cmp_data.get("loss_sum", None)
    
    # Verify all gradients are on the same device
    print(f"\nVerifying gradient tensors are on {device}...")
    for name, grad in ref_grads.items():
        if not isinstance(grad, torch.Tensor):
            raise TypeError(f"Reference gradient '{name}' is not a tensor: {type(grad)}")
        print(f"  Reference grad '{name}' device: {grad.device}")
    
    for name, grad in cmp_grads.items():
        if not isinstance(grad, torch.Tensor):
            raise TypeError(f"Comparison gradient '{name}' is not a tensor: {type(grad)}")
    
    print(f"✓ All gradients loaded on {device}")
    
    print(f"\n{'='*60}")
    print(f"Gradient Comparison: {ref_mode} vs {cmp_mode}")
    print(f"{'='*60}")
    print(f"Reference ({ref_mode}): {len(ref_grads)} parameters")
    print(f"Compare   ({cmp_mode}): {len(cmp_grads)} parameters")
    
    if "config" in ref_data and "config" in cmp_data:
        print(f"\nReference config: {ref_data['config']}")
        print(f"Compare config:   {cmp_data['config']}")
    
    print(f"\n{'='*60}")
    print(f"Performance Metrics")
    print(f"{'='*60}")
    print(f"{ref_mode.capitalize():10s}: {ref_time:.4f}s, {ref_mem['peak_allocated_gb']:.2f} GB")
    print(f"{cmp_mode.capitalize():10s}: {cmp_time:.4f}s, {cmp_mem['peak_allocated_gb']:.2f} GB")
    if ref_time > 0:
        print(f"Speedup:  {ref_time/cmp_time:.2f}x")
    print(f"Memory:   {ref_mem['peak_allocated_gb'] - cmp_mem['peak_allocated_gb']:.2f} GB saved")
    
    # Print loss comparison if available
    if ref_loss is not None and cmp_loss is not None:
        print(f"\n{'='*60}")
        print(f"Loss Comparison")
        print(f"{'='*60}")
        print(f"{ref_mode.capitalize():10s} loss: {ref_loss:.6f}")
        print(f"{cmp_mode.capitalize():10s} loss: {cmp_loss:.6f}")
        
        # Compute relative difference (normalized by max absolute value)
        max_abs_loss = max(abs(ref_loss), abs(cmp_loss))
        if max_abs_loss > 0:
            rel_diff = abs(ref_loss - cmp_loss) / max_abs_loss
            print(f"Relative diff: {rel_diff:.6f} (|diff| / max(|ref|, |cmp|))")
        else:
            print(f"Relative diff: 0.000000 (both losses are zero)")
    
    print(f"\n{'='*60}")
    print(f"Comparing Gradients")
    print(f"{'='*60}")
    
    try:
        
        # Use the same comparison logic as test_tree_training.py
        _compare_and_assert_gradients(
            baseline_grads=ref_grads,
            tree_grads=cmp_grads,
            baseline_params={},  # Not comparing params here
            tree_params={},      # Not comparing params here
            logger_instance=logger,
            max_mismatch_prints=5,
            mean_rel_diff_threshold=mean_rel_diff_threshold,
        )
        print(f"\n✓ Gradient check PASSED")
        print(f"  All {len(ref_grads)} parameters match within threshold")
        print(f"  mean_rel_diff_threshold={mean_rel_diff_threshold}")
        return True
    except AssertionError as e:
        print(f"\n✗ Gradient check FAILED")
        print(f"  Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Compare gradients from different training modes"
    )
    parser.add_argument("--reference", type=str, required=True,
                       help="Path to reference gradient file (e.g., baseline.pt)")
    parser.add_argument("--compare", type=str, required=True,
                       help="Path to comparison gradient file (e.g., flex.pt or stack.pt)")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device to load gradients to (default: cuda:0)")
    parser.add_argument("--mean-rel-diff-threshold", type=float, default=0.3,
                       help="Threshold for mean relative difference (default: 0.3)")
    args = parser.parse_args()
    
    # Validate files exist
    if not Path(args.reference).exists():
        print(f"Error: Reference file not found: {args.reference}")
        return 1
    
    if not Path(args.compare).exists():
        print(f"Error: Comparison file not found: {args.compare}")
        return 1
    
    # Compare gradients
    success = compare_gradients(
        reference_file=args.reference,
        compare_file=args.compare,
        device=args.device,
        mean_rel_diff_threshold=args.mean_rel_diff_threshold,
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())


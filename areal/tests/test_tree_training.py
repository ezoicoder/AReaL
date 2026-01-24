import os
import time
from contextlib import contextmanager
from importlib.metadata import version as get_version
from typing import Optional, Callable, Any, Tuple

import pytest
import torch
import torch.distributed as dist

from areal.api.alloc_mode import AllocationMode
from areal.api.cli_args import (
    FSDPEngineConfig,
    MegatronEngineConfig,
    MicroBatchSpec,
    OptimizerConfig,
    TrainEngineConfig,
)
from areal.api.io_struct import FinetuneSpec
from areal.engine.fsdp_engine import FSDPEngine
from areal.engine.megatron_engine import MegatronEngine
from areal.models.tree_attn.tree import build_packed_tree_batch
from areal.platforms import current_platform
from areal.tests.utils import get_model_path
from areal.utils import logging
from areal.engine.ppo.actor import grpo_loss_fn

logger = logging.getLogger("MegatronEngine Test")

MODEL_PATH = get_model_path(
    "/data/tree/models/Qwen3-8B", "Qwen/Qwen3-8B"
)

# Path to real tree training data
TREE_DATA_PATH = "/data/tree/tree-data/tau2-16k-small/call2_rank0.pt"


# =============================================================================
# Memory tracking utilities
# =============================================================================

def reset_peak_memory():
    """Reset CUDA peak memory statistics."""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

def get_memory_stats(stage_name: str) -> dict:
    """Get current CUDA memory statistics.
    
    Args:
        stage_name: Name of the current stage (for logging)
        
    Returns:
        Dictionary with memory stats in GB
    """
    torch.cuda.synchronize()
    stats = {
        "stage": stage_name,
        "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
        "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
        "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
    }
    return stats

def log_memory_stats(stats: dict, logger_instance):
    """Log memory statistics in a formatted way.
    
    Args:
        stats: Dictionary with memory stats from get_memory_stats
        logger_instance: Logger to use for output
    """
    logger_instance.info(f"\n{'='*60}")
    logger_instance.info(f"Memory Stats - {stats['stage']}")
    logger_instance.info(f"  Allocated: {stats['allocated_gb']:.2f} GB")
    logger_instance.info(f"  Reserved:  {stats['reserved_gb']:.2f} GB")
    logger_instance.info(f"  Peak:      {stats['peak_allocated_gb']:.2f} GB")
    logger_instance.info(f"{'='*60}\n")

# =============================================================================
# Engine setup helpers and context managers
# =============================================================================

@contextmanager
def setup_engine(
    engine_class,
    experiment_name: str,
    master_port: str,
    max_tokens_per_mb: int = 16384,
    enable_tree_training: bool = False,
    enable_tree_stack_training: bool = False,
    gradient_checkpointing: bool = False,
    model_path: str = MODEL_PATH,
    disable_optimizer: bool = True,
):
    """Context manager to setup and teardown an engine (FSDP or Megatron).
    
    Args:
        engine_class: FSDPEngine or MegatronEngine
        experiment_name: Name for the experiment
        master_port: Port for distributed communication
        max_tokens_per_mb: Maximum tokens per microbatch
        enable_tree_training: Whether to enable tree training mode
        enable_tree_stack_training: Whether to enable tree attention training mode
        gradient_checkpointing: Whether to enable gradient checkpointing
        model_path: Path to the model
        disable_optimizer: Whether to disable optimizer (for gradient-only tests)
        
    Yields:
        Initialized engine instance
    """
    os.environ.update(
        {
            "WORLD_SIZE": "1",
            "RANK": "0",
            "LOCAL_RANK": "0",
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": master_port,
        }
    )
    
    # Determine which engine config to use
    if engine_class == FSDPEngine:
        engine_specific_config = {"fsdp": FSDPEngineConfig()}
    elif engine_class == MegatronEngine:
        engine_specific_config = {"megatron": MegatronEngineConfig(use_deterministic_algorithms=True)}
    else:
        raise ValueError(f"Unknown engine class: {engine_class}")
    
    config = TrainEngineConfig(
        experiment_name=experiment_name,
        trial_name="test",
        path=model_path,
        mb_spec=MicroBatchSpec(max_tokens_per_mb=max_tokens_per_mb),
        optimizer=None if disable_optimizer else OptimizerConfig(),
        enable_tree_training=enable_tree_training,
        enable_tree_stack_training=enable_tree_stack_training,
        gradient_checkpointing=gradient_checkpointing,
        disable_optimizer=disable_optimizer,
        **engine_specific_config,
    )
    
    alloc_mode = AllocationMode.from_str("d1p1t1")
    ft_spec = FinetuneSpec(total_train_epochs=1, dataset_size=128, train_batch_size=8)
    
    engine = engine_class(config)
    engine.create_process_group(alloc_mode.train)
    engine.initialize(addr=None, ft_spec=ft_spec, parallel_strategy=alloc_mode.train)
    
    try:
        yield engine
    finally:
        engine.destroy()
        assert not dist.is_initialized()


def run_forward_pass(
    engine,
    input_data: dict[str, torch.Tensor],
    aggregate_fn: Optional[Callable] = None,
) -> Tuple[torch.Tensor, float]:
    """Run forward pass and return logprobs with timing.
    
    Args:
        engine: Engine instance (FSDP or Megatron)
        input_data: Input data dictionary
        aggregate_fn: Optional aggregation function for output
        
    Returns:
        Tuple of (log probabilities tensor, elapsed time in seconds)
    """
    engine.eval()
    
    # Synchronize before timing
    torch.cuda.synchronize()
    start_time = time.time()
    
    if isinstance(engine, FSDPEngine):
        result = engine.forward_batch(input_=input_data, aggregate_fn=aggregate_fn)
    elif isinstance(engine, MegatronEngine):
        result = engine.forward(input_=input_data, aggregate_fn=aggregate_fn)
    else:
        raise ValueError(f"Unknown engine type: {type(engine)}")
    
    # Synchronize after computation
    torch.cuda.synchronize()
    elapsed_time = time.time() - start_time
    
    return result, elapsed_time


def run_train_batch(
    engine,
    input_data: dict[str, torch.Tensor],
    loss_fn: Callable,
    loss_weight_fn: Callable,
) -> Tuple[Any, float]:
    """Run training batch (forward + backward) with timing.
    
    Args:
        engine: Engine instance (FSDP or Megatron)
        input_data: Input data dictionary
        loss_fn: Loss function
        loss_weight_fn: Loss weight function
        
    Returns:
        Tuple of (training result, elapsed time in seconds)
    """
    engine.train()
    
    # Synchronize before timing
    torch.cuda.synchronize()
    start_time = time.time()
    
    result = engine.train_batch(
        input_data,
        loss_fn=loss_fn,
        loss_weight_fn=loss_weight_fn,
    )
    
    # Synchronize after computation
    torch.cuda.synchronize()
    elapsed_time = time.time() - start_time
    
    return result, elapsed_time


# =============================================================================
# Common loss functions and weight functions
# =============================================================================

# def loss_fn(logprobs, entropy, input_data):
#     """Default loss function: -mean(logprobs)"""
#     # print(f"[Debug] logprobs shape: {logprobs.shape[0]}")
#     # assert logprobs.shape[0] == input_data["cu_seqlens"][-1], "logprobs shape and cu_seqlens shape do not match"
    
#     res = -logprobs.sum()
#     print(f"[Debug] logprobs: {logprobs}")
#     print(f"[Debug] temporary loss: {res.item()}")
#     return res

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

# def loss_fn(logprobs, entropy, input_data):
#     return entropy.sum() / input_data["loss_mask"].count_nonzero()

# def loss_fn(logprobs, entropy, input_data):
#     # Create a mask with 1s everywhere except for positions x-1 for x in cu_seqlens, which are 0
#     cu_seqlens = input_data["cu_seqlens"]
#     mask = torch.ones_like(logprobs, dtype=logprobs.dtype)
#     for x in cu_seqlens:
#         if x.item() > 0 and (x.item() - 1) < mask.numel():
#             mask[x.item() - 1] = 0
#     mask = mask.detach()  # Only mask is detached, will not affect logprobs autograd
#     logprobs = logprobs * mask
#     return logprobs.sum() / input_data["loss_mask"].count_nonzero()

def loss_weight_fn(input_data):
    """Default weight function based on attention_mask sum"""
    return input_data["loss_mask"].count_nonzero()



# =============================================================================
# Assertion helpers
# =============================================================================

def _assert_logprobs_close(
    logprob_tree: torch.Tensor,
    logprob_baseline: torch.Tensor,
    logger_instance,
    rtol: float = 0.2,
    atol: float = 0.2,
) -> None:
    """Assert that tree and baseline logprobs are close with detailed error reporting.
    
    Args:
        logprob_tree: Log probabilities from tree training
        logprob_baseline: Log probabilities from baseline
        logger_instance: Logger for error messages
        rtol: Relative tolerance
        atol: Absolute tolerance
    """
    is_close = torch.isclose(logprob_tree, logprob_baseline, rtol=rtol, atol=atol)
    if not is_close.all():
        mismatched_mask = ~is_close
        num_mismatched = mismatched_mask.sum().item()
        total_elements = mismatched_mask.numel()
        mismatch_percentage = 100.0 * num_mismatched / total_elements

        mismatched_indices = torch.nonzero(mismatched_mask, as_tuple=False)
        num_to_show = min(10, num_mismatched)
        
        logger_instance.error(
            f"Assertion failed: {num_mismatched}/{total_elements} elements mismatched ({mismatch_percentage:.2f}%)"
        )
        logger_instance.error(f"First {num_to_show} mismatched positions and values:")
        
        for i in range(num_to_show):
            idx = tuple(mismatched_indices[i].tolist())
            tree_val = logprob_tree[idx].item()
            baseline_val = logprob_baseline[idx].item()
            abs_diff = abs(tree_val - baseline_val)
            rel_diff = abs_diff / (abs(baseline_val) + 1e-8)
            logger_instance.error(
                f"  Position {idx}: tree={tree_val:.6f}, baseline={baseline_val:.6f}, "
                f"abs_diff={abs_diff:.6f}, rel_diff={rel_diff:.6f}"
            )

        abs_diff_all = (logprob_tree - logprob_baseline).abs()
        logger_instance.error(
            f"Overall abs diff: max={abs_diff_all.max().item():.6f}, "
            f"mean={abs_diff_all.mean().item():.6f}, median={abs_diff_all.median().item():.6f}"
        )

    assert is_close.all(), (
        f"logprob_tree and logprob_baseline differ: "
        f"{(~is_close).sum().item()}/{is_close.numel()} elements mismatched "
        f"({100.0 * (~is_close).sum().item() / is_close.numel():.2f}%)"
    )


@pytest.fixture(scope="module")
def real_tree_input(prefix_len):
    """Load real tree training data from saved file.

    Returns the full input_data from saved file (or prefix if prefix_len != -1).
    Additionally, prints out the sequence lengths for inspection.
    
    Args:
        prefix_len: Number of sequences to keep. If -1, keep all sequences.
    """
    import os
    if not os.path.exists(TREE_DATA_PATH):
        pytest.skip(f"Tree data file not found: {TREE_DATA_PATH}")

    data = torch.load(TREE_DATA_PATH)
    if "input_data" not in data:
        pytest.skip(f"No input_data found in {TREE_DATA_PATH}")

    input_data = data["input_data"]

    device = current_platform.device_type
    device_obj = device if isinstance(device, torch.device) else torch.device(device)

    # Output sequence lengths for debugging
    if "attention_mask" in input_data:
        attn_mask = input_data["attention_mask"]
        if attn_mask is not None:
            seq_lens = attn_mask.sum(dim=1).cpu().tolist()
            total_sequences = attn_mask.shape[0]
            print(f"[real_tree_input] Loaded {total_sequences} sequences.")
            print(f"[real_tree_input] Sequence lengths: min={min(seq_lens)}, max={max(seq_lens)}, mean={sum(seq_lens)/len(seq_lens):.2f}")
            # Optionally, print first few lengths for clarity
            print(f"[real_tree_input] First 10 sequence lengths: {seq_lens[:10]}")
            
            # Apply prefix_len filter if specified
            if prefix_len != -1:
                print(f"[real_tree_input] Applying prefix_len={prefix_len}, keeping first {prefix_len} sequences")

    # Load all fields present in input_data and apply prefix_len filter
    result = {}
    for field, value in input_data.items():
        if isinstance(value, torch.Tensor):
            # Apply prefix_len filter if not -1
            if prefix_len != -1 and value.size(0) >= prefix_len:
                result[field] = value[:prefix_len].to(device_obj)
            else:
                result[field] = value.to(device_obj)
        else:
            result[field] = value

    return result

def _collect_gradients(engine) -> dict[str, torch.Tensor]:
    grads = {}
    for model in engine.model:
        for name, param in model.named_parameters():
            # Megatron stores gradients in main_grad attribute
            if hasattr(param, "main_grad") and param.main_grad is not None:
                grads[name] = param.main_grad.clone()
            elif param.grad is not None:
                grads[name] = param.grad.clone()
    return grads


def _collect_parameters(engine) -> dict[str, torch.Tensor]:
    params = {}
    for model in engine.model:
        for name, param in model.named_parameters():
            params[name] = param.data.clone()
    return params


def _check_nan_params(params: dict[str, torch.Tensor], label: str) -> list[str]:
    nan_params = []
    for name, param in params.items():
        if torch.isnan(param).any():
            nan_count = torch.isnan(param).sum().item()
            total_count = param.numel()
            nan_params.append(name)
            print(f"  {name}: {nan_count}/{total_count} NaN values")
    if nan_params:
        print(f"\n⚠ NaN parameters in {label} ({len(nan_params)}):")
    return nan_params


def _compare_and_assert_gradients(
    baseline_grads: dict[str, torch.Tensor],
    tree_grads: dict[str, torch.Tensor],
    baseline_params: dict[str, torch.Tensor],
    tree_params: dict[str, torch.Tensor],
    logger_instance,
    max_mismatch_prints: int = 5,
    mean_rel_diff_threshold: float = 0.3,
) -> None:
    """Compare gradients between baseline and tree training engines and assert they match.
    
    All tensors are expected to be CPU tensors (detached).
    
    Args:
        baseline_grads: Gradients from baseline engine (CPU tensors)
        tree_grads: Gradients from tree training engine (CPU tensors)
        baseline_params: Parameters from baseline engine (CPU tensors)
        tree_params: Parameters from tree training engine (CPU tensors)
        logger_instance: Logger instance for logging messages
        max_mismatch_prints: Maximum number of detailed mismatch logs to print
        mean_rel_diff_threshold: Threshold for mean relative difference to trigger mismatch
    
    Raises:
        AssertionError: If gradients don't match or contain NaN/zero values
    """
    # ========== Compare gradients ==========
    baseline_keys = set(baseline_grads.keys())
    tree_keys = set(tree_grads.keys())

    # Check for missing keys
    only_in_baseline = baseline_keys - tree_keys
    only_in_tree = tree_keys - baseline_keys

    if only_in_baseline:
        logger_instance.warning(f"Gradients only in baseline: {only_in_baseline}")
    if only_in_tree:
        logger_instance.warning(f"Gradients only in tree training: {only_in_tree}")

    common_keys = baseline_keys & tree_keys
    logger_instance.info(f"Comparing {len(common_keys)} common gradient tensors on CPU")

    # Check for NaN and zero gradients
    nan_in_baseline = []
    nan_in_tree = []
    zero_in_baseline = []
    zero_in_tree = []

    for name in sorted(common_keys):
        if torch.isnan(baseline_grads[name]).any():
            nan_in_baseline.append(name)
        if torch.isnan(tree_grads[name]).any():
            nan_in_tree.append(name)
        if (baseline_grads[name] == 0).all():
            zero_in_baseline.append(name)
        if (tree_grads[name] == 0).all():
            zero_in_tree.append(name)

    if nan_in_baseline:
        logger_instance.info(f"\n⚠ NaN gradients in BASELINE ({len(nan_in_baseline)}):")
        for name in nan_in_baseline:
            nan_count = torch.isnan(baseline_grads[name]).sum().item()
            total_count = baseline_grads[name].numel()
            logger_instance.info(f"  {name}: {nan_count}/{total_count} NaN values")

    if nan_in_tree:
        logger_instance.info(f"\n⚠ NaN gradients in TREE TRAINING ({len(nan_in_tree)}):")
        for name in nan_in_tree:
            nan_count = torch.isnan(tree_grads[name]).sum().item()
            total_count = tree_grads[name].numel()
            logger_instance.info(f"  {name}: {nan_count}/{total_count} NaN values")

    # Check for NaN in updated parameters (if provided)
    nan_params_baseline = _check_nan_params(baseline_params, "BASELINE FSDP PARAMS") if baseline_params else []
    nan_params_tree = _check_nan_params(tree_params, "TREE TRAINING FSDP PARAMS") if tree_params else []

    mismatched_params = []
    max_diff_overall = 0.0
    mismatch_print_count = 0  # Counter for printed mismatches

    for name in sorted(common_keys):
        baseline_grad = baseline_grads[name]
        tree_grad = tree_grads[name]

        if baseline_grad.shape != tree_grad.shape:
            mismatched_params.append(
                (name, f"shape mismatch: {baseline_grad.shape} vs {tree_grad.shape}")
            )
            continue

        # All operations on CPU tensors now
        diff = (baseline_grad - tree_grad).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        # Compute relative difference: |a - b| / max(|a|, |b|)
        abs_max = torch.maximum(baseline_grad.abs(), tree_grad.abs())
        rel_diff = torch.where(abs_max > 0, diff / abs_max, torch.zeros_like(diff))
        max_rel_diff = rel_diff.max().item()
        mean_rel_diff = rel_diff.mean().item()

        # Check if gradients are close:
        # 1. Mean relative difference <= threshold
        # 2. Number of elements with rel_diff > 0.1 is less than 10% of total elements
        num_large_diff = (rel_diff > 0.1).sum().item()
        total_elements = rel_diff.numel()
        large_diff_ratio = num_large_diff / total_elements

        if mean_rel_diff > mean_rel_diff_threshold:
            mismatched_params.append(
                (name, f"max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}, max_rel_diff={max_rel_diff:.6e}, mean_rel_diff={mean_rel_diff:.6e}, large_diff_ratio={large_diff_ratio:.4f}")
            )
            
            # Only print detailed info for the first few mismatches
            if mismatch_print_count < max_mismatch_prints:
                # Find the position with max relative difference
                max_rel_diff_idx = rel_diff.argmax()
                max_rel_diff_pos = torch.unravel_index(max_rel_diff_idx, baseline_grad.shape)
                baseline_at_max = baseline_grad[max_rel_diff_pos].item()
                tree_at_max = tree_grad[max_rel_diff_pos].item()
                
                logger_instance.info(
                    f"Gradient mismatch for {name}: "
                    f"Shape: {baseline_grad.shape}, "
                    f"Baseline grad mean: {baseline_grad.float().mean().item():.6e}, "
                    f"Tree grad mean: {tree_grad.float().mean().item():.6e}, "
                    f"Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}, "
                    f"Max rel diff: {max_rel_diff:.6e}, Mean rel diff: {mean_rel_diff:.6e}, "
                    f"Large diff elements: {num_large_diff}/{total_elements} ({large_diff_ratio:.2%}), "
                    f"Max rel diff at position {max_rel_diff_pos}: baseline={baseline_at_max:.6e}, tree={tree_at_max:.6e}"
                )
                
                mismatch_print_count += 1

    assert len(only_in_baseline) == 0, (
        f"Gradients missing in tree training: {only_in_baseline}"
    )
    assert len(only_in_tree) == 0, f"Gradients missing in baseline: {only_in_tree}"
    assert len(nan_in_baseline) == 0, f"NaN gradients in baseline: {nan_in_baseline}"
    assert len(nan_in_tree) == 0, f"NaN gradients in tree training: {nan_in_tree}"
    assert len(nan_params_baseline) == 0, (
        f"NaN parameters in baseline: {nan_params_baseline}"
    )
    assert len(nan_params_tree) == 0, (
        f"NaN parameters in tree training: {nan_params_tree}"
    )
    assert len(mismatched_params) == 0, (
        f"Gradient mismatches found ({len(mismatched_params)}/{len(common_keys)} params): {mismatched_params}"
    )

# =============================================================================
# Tests for n_mbs and n_mbs_divisor in tree packing
# =============================================================================


def _create_test_input(
    batch_size: int,
    seq_lengths: list[int],
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """Create test input data with specified sequence lengths.

    Args:
        batch_size: Number of sequences.
        seq_lengths: List of sequence lengths for each sequence.
        device: Device for tensors.

    Returns:
        Dictionary with 'input_ids' and 'attention_mask' tensors.
    """
    assert len(seq_lengths) == batch_size
    max_len = max(seq_lengths)

    input_ids = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=device)

    for i, length in enumerate(seq_lengths):
        # Use unique tokens for each sequence to avoid sharing
        input_ids[i, :length] = torch.arange(
            i * 1000, i * 1000 + length, dtype=torch.long, device=device
        )
        attention_mask[i, :length] = True

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


def _create_shared_prefix_input(
    batch_size: int,
    prefix_length: int,
    suffix_lengths: list[int],
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """Create test input where all sequences share a common prefix.

    Args:
        batch_size: Number of sequences.
        prefix_length: Length of the shared prefix.
        suffix_lengths: List of suffix lengths for each sequence.
        device: Device for tensors.

    Returns:
        Dictionary with 'input_ids' and 'attention_mask' tensors.
    """
    assert len(suffix_lengths) == batch_size
    seq_lengths = [prefix_length + s for s in suffix_lengths]
    max_len = max(seq_lengths)

    input_ids = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=device)

    # Shared prefix tokens
    prefix_tokens = torch.arange(1, prefix_length + 1, dtype=torch.long, device=device)

    for i, (length, suffix_len) in enumerate(zip(seq_lengths, suffix_lengths)):
        # Shared prefix
        input_ids[i, :prefix_length] = prefix_tokens
        # Unique suffix for each sequence
        if suffix_len > 0:
            input_ids[i, prefix_length:length] = torch.arange(
                1000 + i * 100, 1000 + i * 100 + suffix_len, dtype=torch.long, device=device
            )
        attention_mask[i, :length] = True

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


def test_build_packed_tree_batch_n_mbs_minimum():
    """Test that n_mbs enforces minimum number of trees."""
    # Create input with 8 sequences that would naturally pack into fewer trees
    # Each sequence has unique tokens to avoid prefix sharing
    data = _create_test_input(
        batch_size=8,
        seq_lengths=[50, 50, 50, 50, 50, 50, 50, 50],
    )

    # With large max_tokens_per_mb, all sequences would fit in 1 tree
    # But n_mbs=4 should force at least 4 trees
    mb_spec = MicroBatchSpec(
        max_tokens_per_mb=10240,  # 80 * 128
        n_mbs=4,
        n_mbs_divisor=1,
    )

    result = build_packed_tree_batch(data, mb_spec, pad_to_maximum=True)

    assert len(result) >= 4, (
        f"Expected at least 4 trees (n_mbs=4), got {len(result)}"
    )


def test_build_packed_tree_batch_n_mbs_divisor():
    """Test that n_mbs_divisor ensures tree count is divisible."""
    # Create input with 5 sequences that can be grouped together
    # Each sequence has unique tokens to avoid prefix sharing
    data = _create_test_input(
        batch_size=5,
        seq_lengths=[100, 100, 100, 100, 100],
    )

    # With max_tokens_per_mb=512, sequences can be grouped (up to 5 per tree)
    # This would naturally create 1 tree with all 5 sequences
    # n_mbs_divisor=2 should force splitting to get an even number (2 trees)
    mb_spec = MicroBatchSpec(
        max_tokens_per_mb=512,  # 4 * 128
        n_mbs=1,
        n_mbs_divisor=2,
    )

    result = build_packed_tree_batch(data, mb_spec, pad_to_maximum=True)

    assert len(result) % 2 == 0, (
        f"Expected tree count divisible by 2 (n_mbs_divisor=2), got {len(result)}"
    )


def test_build_packed_tree_batch_n_mbs_and_divisor_combined():
    """Test that n_mbs and n_mbs_divisor work together correctly."""
    # Create input with 6 sequences
    data = _create_test_input(
        batch_size=6,
        seq_lengths=[80, 80, 80, 80, 80, 80],
    )

    # n_mbs=5 (minimum 5 trees), n_mbs_divisor=3 (must be divisible by 3)
    # Result should be 6 trees (next multiple of 3 >= 5)
    mb_spec = MicroBatchSpec(
        max_tokens_per_mb=128,  # 1 * 128
        n_mbs=5,
        n_mbs_divisor=3,
    )

    result = build_packed_tree_batch(data, mb_spec, pad_to_maximum=True)

    assert len(result) >= 5, (
        f"Expected at least 5 trees (n_mbs=5), got {len(result)}"
    )
    assert len(result) % 3 == 0, (
        f"Expected tree count divisible by 3 (n_mbs_divisor=3), got {len(result)}"
    )


def test_build_packed_tree_batch_default_values():
    """Test that default n_mbs=1 and n_mbs_divisor=1 work correctly."""
    # Create input that would naturally pack into 1 tree
    data = _create_shared_prefix_input(
        batch_size=4,
        prefix_length=50,
        suffix_lengths=[10, 10, 10, 10],
    )

    mb_spec = MicroBatchSpec(
        max_tokens_per_mb=10240,  # 80 * 128
        # n_mbs and n_mbs_divisor default to 1
    )

    result = build_packed_tree_batch(data, mb_spec, pad_to_maximum=True)

    # With shared prefix, all sequences should pack into 1 tree
    assert len(result) >= 1, f"Expected at least 1 tree, got {len(result)}"


def test_build_packed_tree_batch_cannot_split_raises_error():
    """Test that RuntimeError is raised when trees cannot be split to meet requirements."""
    # Create input with only 2 sequences - can only split to 2 trees max
    data = _create_test_input(
        batch_size=2,
        seq_lengths=[50, 50],
    )

    # Request 4 trees, but only 2 sequences available
    # This should raise RuntimeError since we can't create 4 trees from 2 sequences
    mb_spec = MicroBatchSpec(
        max_tokens_per_mb=128,  # 1 * 128
        n_mbs=4,
        n_mbs_divisor=1,
    )

    with pytest.raises(RuntimeError, match="Cannot split trees to meet n_mbs"):
        build_packed_tree_batch(data, mb_spec, pad_to_maximum=True)


def test_build_packed_tree_batch_cannot_split_divisor_raises_error():
    """Test that RuntimeError is raised when n_mbs_divisor cannot be satisfied."""
    # Create input with 3 sequences, each getting its own tree
    data = _create_test_input(
        batch_size=3,
        seq_lengths=[100, 100, 100],
    )

    # With max_tokens_per_mb=128, each sequence gets its own tree (3 trees)
    # n_mbs_divisor=2 requires even number, but 3 trees can't be split (1 seq each)
    # This should raise RuntimeError
    mb_spec = MicroBatchSpec(
        max_tokens_per_mb=128,  # 1 * 128
        n_mbs=1,
        n_mbs_divisor=2,
    )

    with pytest.raises(RuntimeError, match="Cannot split trees to meet"):
        build_packed_tree_batch(data, mb_spec, pad_to_maximum=True)


def test_build_packed_tree_batch_max_tokens_still_respected():
    """Test that max_tokens_per_mb is still respected when splitting."""
    # Create input with sequences that exceed max_tokens_per_mb individually
    data = _create_test_input(
        batch_size=4,
        seq_lengths=[100, 100, 100, 100],
    )

    # max_tokens_per_mb=128 means at most ~1 sequence per tree
    mb_spec = MicroBatchSpec(
        max_tokens_per_mb=128,  # 1 * 128
        n_mbs=2,
        n_mbs_divisor=1,
    )

    result = build_packed_tree_batch(data, mb_spec, pad_to_maximum=True)

    # Each tree should respect max_tokens_per_mb
    for i, mb in enumerate(result.mbs):
        if "trie_node" in mb:
            tree_tokens = mb["trie_node"].num_tokens
            assert tree_tokens <= 128, (
                f"Tree {i} has {tree_tokens} tokens, exceeds max_tokens_per_mb=128"
            )


# =============================================================================
# Multiprocessing test for dp_group synchronization
# =============================================================================


def _dp_group_worker(
    rank: int,
    world_size: int,
    backend: str,
    result_queue,
    data_per_rank: list[dict[str, torch.Tensor]],
    max_tokens_per_mb: int,
):
    """Worker function for distributed dp_group test.

    Each rank runs build_packed_tree_batch with different input data
    and validates that the number of trees is synchronized across ranks.
    """
    import torch.multiprocessing as mp

    try:
        # Set environment variables for distributed
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank)

        # Initialize process group
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
        )

        # Set device
        device = f"cuda:{rank}"
        torch.cuda.set_device(device)

        # Get data for this rank and move to GPU
        data = {
            k: v.to(device) for k, v in data_per_rank[rank].items()
        }

        # Create mb_spec
        mb_spec = MicroBatchSpec(
            max_tokens_per_mb=max_tokens_per_mb,
            n_mbs=1,
            n_mbs_divisor=1,
        )

        # Get the default process group as dp_group
        dp_group = dist.distributed_c10d._get_default_group()

        # Run build_packed_tree_batch with dp_group
        result = build_packed_tree_batch(
            data,
            mb_spec,
            pad_to_maximum=True,
            dp_group=dp_group,
        )

        num_trees = len(result)

        # All-gather to verify all ranks have same number of trees
        local_count = torch.tensor([num_trees], dtype=torch.int64, device=device)
        all_counts = [
            torch.zeros(1, dtype=torch.int64, device=device)
            for _ in range(world_size)
        ]
        dist.all_gather(all_counts, local_count)

        all_tree_counts = [c.item() for c in all_counts]

        # Put result in queue
        result_queue.put({
            "rank": rank,
            "num_trees": num_trees,
            "all_tree_counts": all_tree_counts,
            "success": True,
            "error": None,
        })

    except Exception as e:
        import traceback
        result_queue.put({
            "rank": rank,
            "num_trees": -1,
            "all_tree_counts": [],
            "success": False,
            "error": f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}",
        })

    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Requires at least 2 GPUs"
)
def test_build_packed_tree_batch_dp_group_sync():
    """Test that dp_group synchronizes tree count across ranks.

    This test spawns 2 processes (one per GPU) with different input data:
    - Rank 0: 2 sequences that fit in 1 tree
    - Rank 1: 4 sequences that require 2 trees

    With dp_group synchronization, both ranks should produce 2 trees.
    """
    import torch.multiprocessing as mp

    world_size = 2
    backend = "nccl"
    max_tokens_per_mb = 256  # 2 * 128

    # Create different data for each rank on CPU (will be moved to GPU in worker)
    # Rank 0: 2 sequences, total ~100 tokens -> fits in 1 tree
    data_rank0 = _create_test_input(
        batch_size=2,
        seq_lengths=[50, 50],
        device="cpu",
    )

    # Rank 1: 4 sequences, total ~400 tokens -> needs 2 trees (256 max per tree)
    data_rank1 = _create_test_input(
        batch_size=4,
        seq_lengths=[100, 100, 100, 100],
        device="cpu",
    )

    data_per_rank = [data_rank0, data_rank1]

    # Use spawn context for CUDA
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()

    processes = []
    for rank in range(world_size):
        p = ctx.Process(
            target=_dp_group_worker,
            args=(rank, world_size, backend, result_queue, data_per_rank, max_tokens_per_mb),
        )
        p.start()
        processes.append(p)

    # Collect results
    results = []
    for _ in range(world_size):
        results.append(result_queue.get(timeout=60))

    # Wait for processes to finish
    for p in processes:
        p.join(timeout=30)
        if p.is_alive():
            p.terminate()
            p.join()

    # Sort results by rank
    results.sort(key=lambda r: r["rank"])

    # Check for errors
    for r in results:
        if not r["success"]:
            pytest.fail(f"Rank {r['rank']} failed: {r['error']}")

    # Verify all ranks have the same number of trees
    tree_counts = [r["num_trees"] for r in results]
    assert len(set(tree_counts)) == 1, (
        f"Tree counts should be identical across ranks, got {tree_counts}"
    )

    # Verify the synchronized count is the maximum (rank 1 needed 2 trees)
    assert tree_counts[0] >= 2, (
        f"Expected at least 2 trees after sync, got {tree_counts[0]}"
    )

    # Verify all_tree_counts are consistent
    for r in results:
        assert r["all_tree_counts"] == tree_counts, (
            f"Rank {r['rank']} all_tree_counts mismatch: {r['all_tree_counts']} vs {tree_counts}"
        )


# =============================================================================
# FSDP Engine Tree Training Tests
# =============================================================================

fsdp_logger = logging.getLogger("FSDPEngine Test")


@pytest.fixture
def fsdp_engine(max_tokens_per_mb):
    """Fixture for baseline FSDP engine."""
    fsdp_logger.info(f"torch version={torch.__version__}")
    fsdp_logger.info(f"Using max_tokens_per_mb={max_tokens_per_mb}")
    
    with setup_engine(
        FSDPEngine,
        experiment_name="test_baseline",
        master_port="7780",
        max_tokens_per_mb=max_tokens_per_mb,
    ) as engine:
        fsdp_logger.info(f"FSDP Model initialized: {engine.model}")
        yield engine


def _collect_fsdp_gradients(engine: FSDPEngine) -> dict[str, torch.Tensor]:
    """Collect gradients from FSDP engine and immediately offload to CPU.
    
    Handles both regular tensors and DTensors (FSDP2) and DDP-wrapped models.
    
    Args:
        engine: FSDP engine
        
    Returns:
        Dictionary of CPU tensors (detached, local tensors extracted from DTensor).
        Parameter names have DDP's 'module.' prefix removed for consistency.
    """
    grads = {}
    dtensor_count = 0
    for name, param in engine.model.named_parameters():
        if param.grad is not None:
            # Remove DDP's 'module.' prefix for consistency with FSDP2
            clean_name = name.replace("module.", "", 1) if name.startswith("module.") else name
            
            # Check if this is a DTensor (FSDP2)
            if hasattr(param.grad, '_local_tensor'):
                # Extract local tensor from DTensor
                grads[clean_name] = param.grad._local_tensor.detach().cpu()
                dtensor_count += 1
            else:
                # Regular tensor (DDP or standard)
                grads[clean_name] = param.grad.detach().cpu()
    
    if dtensor_count > 0:
        fsdp_logger.debug(f"Collected {dtensor_count} DTensor gradients (extracted to local tensors)")
    
    return grads


def _collect_fsdp_parameters(engine: FSDPEngine) -> dict[str, torch.Tensor]:
    """Collect parameters from FSDP engine and immediately offload to CPU.
    
    Handles both regular tensors and DTensors (FSDP2) and DDP-wrapped models.
    
    Args:
        engine: FSDP engine
        
    Returns:
        Dictionary of CPU tensors (detached, local tensors extracted from DTensor).
        Parameter names have DDP's 'module.' prefix removed for consistency.
    """
    params = {}
    dtensor_count = 0
    for name, param in engine.model.named_parameters():
        # Remove DDP's 'module.' prefix for consistency with FSDP2
        clean_name = name.replace("module.", "", 1) if name.startswith("module.") else name
        
        # Check if this is a DTensor (FSDP2)
        if hasattr(param.data, '_local_tensor'):
            # Extract local tensor from DTensor
            params[clean_name] = param.data._local_tensor.detach().cpu()
            dtensor_count += 1
        else:
            # Regular tensor (DDP or standard)
            params[clean_name] = param.data.detach().cpu()
    
    if dtensor_count > 0:
        fsdp_logger.debug(f"Collected {dtensor_count} DTensor parameters (extracted to local tensors)")
    
    return params


### Never use gradient checkpointing for tree stack training
def test_fsdp_flex_forward(fsdp_engine, real_tree_input, max_tokens_per_mb):
    """Test FSDP tree training forward pass produces correct logprobs."""
    # Run baseline forward pass
    logprob_baseline, baseline_time = run_forward_pass(
        fsdp_engine,
        real_tree_input,
        aggregate_fn=lambda xs: torch.cat(xs, dim=-1),
    )
    fsdp_logger.info(f"Baseline forward_batch time: {baseline_time:.4f}s")
    print("logprob_baseline shape:", logprob_baseline.shape)

    # Run tree training forward pass
    with setup_engine(
        FSDPEngine,
        experiment_name="test_tree",
        master_port="7781",
        max_tokens_per_mb=max_tokens_per_mb,
        enable_tree_training=True,
    ) as tree_engine:
        logprob_tree, tree_time = run_forward_pass(tree_engine, real_tree_input)
        fsdp_logger.info(f"Tree training forward_batch time: {tree_time:.4f}s")
        
        speedup = baseline_time / tree_time
        fsdp_logger.info(f"Speedup (baseline/tree): {speedup:.2f}x")
        print("logprob_tree shape:", logprob_tree.shape)

        # Compare results
        _assert_logprobs_close(logprob_tree, logprob_baseline, fsdp_logger)

def test_fsdp_stack_forward(fsdp_engine, real_tree_input, max_tokens_per_mb):
    """Test FSDP tree attention training forward pass produces correct logprobs."""
    # Run baseline forward pass
    logprob_baseline, baseline_time = run_forward_pass(
        fsdp_engine,
        real_tree_input,
        aggregate_fn=lambda xs: torch.cat(xs, dim=-1),
    )
    fsdp_logger.info(f"Baseline forward_batch time: {baseline_time:.4f}s")
    print("logprob_baseline shape:", logprob_baseline.shape)

    # Run tree attention training forward pass
    with setup_engine(
        FSDPEngine,
        experiment_name="test_tree_stack",
        master_port="7781",
        max_tokens_per_mb=max_tokens_per_mb,
        enable_tree_stack_training=True,
    ) as tree_engine:
        logprob_tree, tree_time = run_forward_pass(tree_engine, real_tree_input)
        fsdp_logger.info(f"Tree attention forward_batch time: {tree_time:.4f}s")

        speedup = baseline_time / tree_time
        fsdp_logger.info(f"Speedup (baseline/tree_stack): {speedup:.2f}x")
        print("logprob_tree shape:", logprob_tree.shape)

        # Compare results
        _assert_logprobs_close(logprob_tree, logprob_baseline, fsdp_logger)

def test_fsdp_flex_backward(real_tree_input, max_tokens_per_mb, is_gradient_checkpointing):
    """Test FSDP tree training forward-backward pass produces correct gradients."""
    # Run baseline training FIRST
    reset_peak_memory()
    with setup_engine(
        FSDPEngine,
        experiment_name="test_baseline",
        master_port="7782",
        max_tokens_per_mb=max_tokens_per_mb,
        gradient_checkpointing=is_gradient_checkpointing,
    ) as baseline_engine:
        _, baseline_time = run_train_batch(baseline_engine, real_tree_input, loss_fn, loss_weight_fn)
        fsdp_logger.info(f"Baseline train_batch time: {baseline_time:.4f}s")
        
        # Get and log memory stats for baseline
        baseline_mem_stats = get_memory_stats("Baseline Training")
        log_memory_stats(baseline_mem_stats, fsdp_logger)

        # Collect gradients and params, automatically offloaded to CPU
        baseline_grads = _collect_fsdp_gradients(baseline_engine)
        baseline_params = _collect_fsdp_parameters(baseline_engine)
        
        fsdp_logger.info(f"[Baseline] Collected {len(baseline_grads)} gradients (detached, on CPU)")
        
        # Check NaN in baseline params immediately
        nan_params_baseline = _check_nan_params(baseline_params, "BASELINE FSDP PARAMS")
        assert len(nan_params_baseline) == 0, f"NaN parameters in baseline: {nan_params_baseline}"
        del baseline_params  # Free CPU memory

    # Force garbage collection and clear CUDA cache to free all GPU memory
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    # Report memory status after cleanup
    mem_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
    mem_reserved = torch.cuda.memory_reserved() / 1024**3
    mem_max = torch.cuda.max_memory_allocated() / 1024**3
    fsdp_logger.info(f"[Cleanup] After GC and cache clear - Allocated: {mem_allocated:.2f} GB, Reserved: {mem_reserved:.2f} GB, Peak: {mem_max:.2f} GB")
    
    time.sleep(1)  # Wait to observe memory release

    # Run flatten tree training SECOND
    reset_peak_memory()
    with setup_engine(
        FSDPEngine,
        experiment_name="test_tree",
        master_port="7783",
        max_tokens_per_mb=max_tokens_per_mb,
        gradient_checkpointing=is_gradient_checkpointing,
        enable_tree_training=True,
    ) as tree_engine:
        _, tree_time = run_train_batch(tree_engine, real_tree_input, loss_fn, loss_weight_fn)
        fsdp_logger.info(f"Tree training train_batch time: {tree_time:.4f}s")
        
        speedup = baseline_time / tree_time
        fsdp_logger.info(f"Speedup (baseline/tree): {speedup:.2f}x")
        
        # Get and log memory stats for tree training
        tree_mem_stats = get_memory_stats("Flex Tree Training")
        log_memory_stats(tree_mem_stats, fsdp_logger)

        # Collect gradients and params, automatically offloaded to CPU
        tree_grads = _collect_fsdp_gradients(tree_engine)
        tree_params = _collect_fsdp_parameters(tree_engine)
        
        fsdp_logger.info(f"[Tree] Collected {len(tree_grads)} gradients (detached, on CPU)")
        
        # Check NaN in tree params immediately
        nan_params_tree = _check_nan_params(tree_params, "TREE TRAINING FSDP PARAMS")
        assert len(nan_params_tree) == 0, f"NaN parameters in tree training: {nan_params_tree}"
        del tree_params  # Free CPU memory
    
    # Log comparison of memory usage
    mem_savings = baseline_mem_stats['peak_allocated_gb'] - tree_mem_stats['peak_allocated_gb']
    mem_ratio = tree_mem_stats['peak_allocated_gb'] / baseline_mem_stats['peak_allocated_gb']
    fsdp_logger.info(f"\n{'='*60}")
    fsdp_logger.info(f"Memory Comparison (Baseline vs Flex Tree)")
    fsdp_logger.info(f"  Memory savings: {mem_savings:.2f} GB")
    fsdp_logger.info(f"  Memory ratio:   {mem_ratio:.2%}")
    fsdp_logger.info(f"{'='*60}\n")

    # Compare gradients directly on CPU (params already checked and freed)

    fsdp_logger.info(f"[Comparison] Comparing gradients on CPU...")
    _compare_and_assert_gradients(
        baseline_grads=baseline_grads,
        tree_grads=tree_grads,
        baseline_params={},  # Already checked and freed
        tree_params={},  # Already checked and freed
        logger_instance=fsdp_logger,
    )

def test_fsdp_stack_backward(real_tree_input, max_tokens_per_mb, is_gradient_checkpointing):
    """Test FSDP tree attention training forward-backward pass produces correct gradients."""
    # Run baseline training FIRST
    reset_peak_memory()
    with setup_engine(
        FSDPEngine,
        experiment_name="test_baseline",
        master_port="7782",
        gradient_checkpointing=is_gradient_checkpointing,
        max_tokens_per_mb=max_tokens_per_mb,
    ) as baseline_engine:
        _, baseline_time = run_train_batch(baseline_engine, real_tree_input, loss_fn, loss_weight_fn)
        fsdp_logger.info(f"Baseline train_batch time: {baseline_time:.4f}s")
        
        # Get and log memory stats for baseline
        baseline_mem_stats = get_memory_stats("Baseline Training")
        log_memory_stats(baseline_mem_stats, fsdp_logger)

        # Collect gradients and params, automatically offloaded to CPU
        baseline_grads = _collect_fsdp_gradients(baseline_engine)
        baseline_params = _collect_fsdp_parameters(baseline_engine)
        
        fsdp_logger.info(f"[Baseline] Collected {len(baseline_grads)} gradients (detached, on CPU)")
        
        # Check NaN in baseline params immediately
        nan_params_baseline = _check_nan_params(baseline_params, "BASELINE FSDP PARAMS")
        assert len(nan_params_baseline) == 0, f"NaN parameters in baseline: {nan_params_baseline}"
        del baseline_params  # Free CPU memory

    # Force garbage collection and clear CUDA cache to free all GPU memory
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    # Report memory status after cleanup
    mem_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
    mem_reserved = torch.cuda.memory_reserved() / 1024**3
    mem_max = torch.cuda.max_memory_allocated() / 1024**3
    fsdp_logger.info(f"[Cleanup] After GC and cache clear - Allocated: {mem_allocated:.2f} GB, Reserved: {mem_reserved:.2f} GB, Peak: {mem_max:.2f} GB")
    
    time.sleep(1)  # Wait to observe memory release

    # Run tree stack training SECOND
    reset_peak_memory()
    with setup_engine(
        FSDPEngine,
        experiment_name="test_tree_stack",
        master_port="7783",
        max_tokens_per_mb=max_tokens_per_mb,
        enable_tree_stack_training=True,
    ) as tree_engine:
        _, tree_time = run_train_batch(tree_engine, real_tree_input, loss_fn, loss_weight_fn)
        fsdp_logger.info(f"Tree stack train_batch time: {tree_time:.4f}s")
        
        speedup = baseline_time / tree_time
        fsdp_logger.info(f"Speedup (baseline/tree_stack): {speedup:.2f}x")
        
        # Get and log memory stats for tree stack training
        tree_mem_stats = get_memory_stats("Tree Stack Training")
        log_memory_stats(tree_mem_stats, fsdp_logger)

        # Collect gradients and params, automatically offloaded to CPU
        tree_grads = _collect_fsdp_gradients(tree_engine)
        tree_params = _collect_fsdp_parameters(tree_engine)
        
        fsdp_logger.info(f"[TreeStack] Collected {len(tree_grads)} gradients (detached, on CPU)")
        
        # Check NaN in tree params immediately
        nan_params_tree = _check_nan_params(tree_params, "TREE STACK FSDP PARAMS")
        assert len(nan_params_tree) == 0, f"NaN parameters in tree stack: {nan_params_tree}"
        del tree_params  # Free CPU memory
    
    # Log comparison of memory usage
    mem_savings = baseline_mem_stats['peak_allocated_gb'] - tree_mem_stats['peak_allocated_gb']
    mem_ratio = tree_mem_stats['peak_allocated_gb'] / baseline_mem_stats['peak_allocated_gb']
    fsdp_logger.info(f"\n{'='*60}")
    fsdp_logger.info(f"Memory Comparison (Baseline vs Tree Stack)")
    fsdp_logger.info(f"  Memory savings: {mem_savings:.2f} GB")
    fsdp_logger.info(f"  Memory ratio:   {mem_ratio:.2%}")
    fsdp_logger.info(f"{'='*60}\n")

    # Compare gradients directly on CPU (params already checked and freed)
    fsdp_logger.info(f"[Comparison] Comparing gradients on CPU...")
    _compare_and_assert_gradients(
        baseline_grads=baseline_grads,
        tree_grads=tree_grads,
        baseline_params={},  # Already checked and freed
        tree_params={},  # Already checked and freed
        logger_instance=fsdp_logger,
    )

def test_flex(real_tree_input, max_tokens_per_mb, is_gradient_checkpointing):
    """Test flex (flatten tree) training and record execution time.
    
    This test runs flex tree training mode and records the training time.
    """
    reset_peak_memory()
    with setup_engine(
        FSDPEngine,
        experiment_name="test_flex",
        master_port="7784",
        max_tokens_per_mb=max_tokens_per_mb,
        gradient_checkpointing=is_gradient_checkpointing,
        enable_tree_training=True,
    ) as flex_engine:
        _, flex_time = run_train_batch(flex_engine, real_tree_input, loss_fn, loss_weight_fn)
        
        # Get and log memory stats
        flex_mem_stats = get_memory_stats("Flex Tree Training")
        
        fsdp_logger.info(f"\n{'='*60}")
        fsdp_logger.info(f"Flex tree training time: {flex_time:.4f}s")
        fsdp_logger.info(f"Peak memory usage: {flex_mem_stats['peak_allocated_gb']:.2f} GB")
        fsdp_logger.info(f"{'='*60}\n")
        
        log_memory_stats(flex_mem_stats, fsdp_logger)


def test_stack(real_tree_input, max_tokens_per_mb):
    """Test stack (tree attention) training and record execution time.
    
    This test runs stack tree training mode and records the training time.
    Note: Stack training does not support gradient checkpointing.
    """
    reset_peak_memory()
    with setup_engine(
        FSDPEngine,
        experiment_name="test_stack",
        master_port="7785",
        max_tokens_per_mb=max_tokens_per_mb,
        enable_tree_stack_training=True,
    ) as stack_engine:
        _, stack_time = run_train_batch(stack_engine, real_tree_input, loss_fn, loss_weight_fn)
        
        # Get and log memory stats
        stack_mem_stats = get_memory_stats("Tree Stack Training")
        
        fsdp_logger.info(f"\n{'='*60}")
        fsdp_logger.info(f"Stack tree training time: {stack_time:.4f}s")
        fsdp_logger.info(f"Peak memory usage: {stack_mem_stats['peak_allocated_gb']:.2f} GB")
        fsdp_logger.info(f"{'='*60}\n")
        
        log_memory_stats(stack_mem_stats, fsdp_logger)


def test_baseline(real_tree_input, max_tokens_per_mb, is_gradient_checkpointing):
    """Test baseline (standard) training and record execution time.
    
    This test runs baseline training mode without any tree optimizations
    and records the training time for comparison purposes.
    """
    reset_peak_memory()
    with setup_engine(
        FSDPEngine,
        experiment_name="test_baseline",
        master_port="7786",
        max_tokens_per_mb=max_tokens_per_mb,
        gradient_checkpointing=is_gradient_checkpointing,
        enable_tree_training=False,
        enable_tree_stack_training=False,
    ) as baseline_engine:
        _, baseline_time = run_train_batch(baseline_engine, real_tree_input, loss_fn, loss_weight_fn)
        
        # Get and log memory stats
        baseline_mem_stats = get_memory_stats("Baseline Training")
        
        fsdp_logger.info(f"\n{'='*60}")
        fsdp_logger.info(f"Baseline training time: {baseline_time:.4f}s")
        fsdp_logger.info(f"Peak memory usage: {baseline_mem_stats['peak_allocated_gb']:.2f} GB")
        fsdp_logger.info(f"{'='*60}\n")
        
        log_memory_stats(baseline_mem_stats, fsdp_logger)

"""
A100(80G):
AREAL_FLEX_ATTENTION_BLOCK_SIZE=64 python -m pytest areal/tests/test_tree_training.py::test_fsdp_flex_backward -v -s --max-tokens-per-mb 16384

python -m pytest areal/tests/test_tree_training.py::test_fsdp_stack_backward -v -s --max-tokens-per-mb 16384

AREAL_FLEX_ATTENTION_BLOCK_SIZE=64 python -m pytest areal/tests/test_tree_training.py::test_fsdp_flex_forward -v -s --max-tokens-per-mb 16384

python -m pytest areal/tests/test_tree_training.py::test_fsdp_stack_forward -v -s --max-tokens-per-mb 16384

AREAL_FLEX_ATTENTION_BLOCK_SIZE=64 python -m pytest areal/tests/test_tree_training.py::test_flex -v -s --max-tokens-per-mb 16384

python -m pytest areal/tests/test_tree_training.py::test_stack -v -s --max-tokens-per-mb 16384

python -m pytest areal/tests/test_tree_training.py::test_baseline -v -s --max-tokens-per-mb 16384 --prefix-len 10

AREAL_FLEX_ATTENTION_BLOCK_SIZE=64 python -m pytest areal/tests/test_tree_training.py::test_fsdp_flex_backward -v -s --max-tokens-per-mb 39936 --prefix-len 10

AREAL_FLEX_ATTENTION_BLOCK_SIZE=64 python -m pytest areal/tests/test_tree_training.py::test_fsdp_flex_backward -v -s --max-tokens-per-mb 16384 --prefix-len 10
"""
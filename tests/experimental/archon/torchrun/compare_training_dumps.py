"""Offline comparison tool for two ArchonEngine training-test dumps.

Given two directories produced by :mod:`run_archon_training_test`, this script:

1. Loads global ``stats.jsonl`` and builds a per-step view, then performs a
   strict loss alignment check.
2. Preferentially loads ``diff.pt`` from each dump and compares parameter update
   signatures. If ``diff.pt`` is absent, falls back to legacy ``params.pt``
   (and optionally ``params_initial.pt``) tensor diffs.

The tool is launched as a plain Python script -- no distributed setup required.

Example::

    python tests/experimental/archon/torchrun/compare_training_dumps.py \\
        --dump-a /tmp/run_a --dump-b /tmp/run_b \\
        --loss-rtol 1e-6 --loss-atol 1e-6

Exit code is non-zero when the loss alignment check fails; parameter / diff-file
comparison is informational only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------


def _stats_file(dump_dir: str) -> str:
    path = os.path.join(dump_dir, "stats.jsonl")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Missing stats file: {path}. Did the training run finish?"
        )
    return path


def _load_stats(dump_dir: str) -> dict[int, list[dict[str, Any]]]:
    """Return ``{step -> list[records]}`` for one dump."""
    by_step: dict[int, list[dict[str, Any]]] = {}
    path = _stats_file(dump_dir)
    with open(path) as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            step = int(rec["step"])
            by_step.setdefault(step, []).append(rec)
    return by_step


def _rank_head_loss(records: list[dict[str, Any]]) -> float:
    """Pick one valid loss for a step.

    Current dumps store one global record per step; we read the first valid one.
    """
    for r in records:
        loss = r.get("loss")
        if loss is None:
            continue
        if isinstance(loss, float) and math.isnan(loss):
            continue
        return float(loss)
    return float("nan")


# -----------------------------------------------------------------------------
# Loss comparison
# -----------------------------------------------------------------------------


@dataclass
class LossDiff:
    step: int
    loss_a: float
    loss_b: float
    abs_gap: float
    rel_gap: float
    aligned: bool


def _compare_losses(
    stats_a: dict[int, list[dict[str, Any]]],
    stats_b: dict[int, list[dict[str, Any]]],
    *,
    atol: float,
    rtol: float,
) -> list[LossDiff]:
    steps_a = set(stats_a.keys())
    steps_b = set(stats_b.keys())
    shared = sorted(steps_a & steps_b)
    if steps_a != steps_b:
        print(
            f"[warn] step sets differ: only_a={sorted(steps_a - steps_b)[:10]} "
            f"only_b={sorted(steps_b - steps_a)[:10]}"
        )

    diffs: list[LossDiff] = []
    for step in shared:
        la = _rank_head_loss(stats_a[step])
        lb = _rank_head_loss(stats_b[step])
        gap = abs(la - lb)
        rel = gap / max(abs(lb), 1e-12)
        aligned = gap <= (atol + rtol * abs(lb))
        diffs.append(
            LossDiff(
                step=step,
                loss_a=la,
                loss_b=lb,
                abs_gap=gap,
                rel_gap=rel,
                aligned=aligned,
            )
        )
    return diffs


# -----------------------------------------------------------------------------
# Parameter comparison
# -----------------------------------------------------------------------------


@dataclass
class ParamDiff:
    name: str
    shape_match: bool
    max_diff: float
    mean_diff: float
    l2_diff: float
    rel_l2_diff: float


@dataclass
class ParamUpdateStat:
    name: str
    numel: float
    mean_abs_update: float
    max_abs_update: float
    l2_update: float
    rel_l2_update: float


@dataclass
class DiffFileGap:
    name: str
    numel_match: bool
    mean_abs_gap: float
    max_abs_gap: float
    l2_gap: float
    rel_l2_gap: float


def _load_diff_signatures(path: str) -> dict[str, ParamUpdateStat]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict payload in {path}, got {type(payload)}")
    params = payload.get("params")
    if not isinstance(params, dict):
        raise ValueError(f"Expected 'params' dict in {path}, got {type(params)}")

    out: dict[str, ParamUpdateStat] = {}
    for name, item in params.items():
        if not isinstance(item, dict):
            raise ValueError(
                f"Expected metrics dict for parameter '{name}' in {path}, got {type(item)}"
            )
        out[name] = ParamUpdateStat(
            name=str(name),
            numel=float(item.get("numel", 0.0)),
            mean_abs_update=float(item.get("mean_abs_update", 0.0)),
            max_abs_update=float(item.get("max_abs_update", 0.0)),
            l2_update=float(item.get("l2_update", 0.0)),
            rel_l2_update=float(item.get("rel_l2_update", 0.0)),
        )
    return out


def _compare_diff_signatures(
    stats_a: dict[str, ParamUpdateStat],
    stats_b: dict[str, ParamUpdateStat],
) -> tuple[list[DiffFileGap], list[str], list[str]]:
    names_a = set(stats_a.keys())
    names_b = set(stats_b.keys())
    shared = sorted(names_a & names_b)
    only_a = sorted(names_a - names_b)
    only_b = sorted(names_b - names_a)

    gaps: list[DiffFileGap] = []
    for name in shared:
        a = stats_a[name]
        b = stats_b[name]
        gaps.append(
            DiffFileGap(
                name=name,
                numel_match=int(round(a.numel)) == int(round(b.numel)),
                mean_abs_gap=abs(a.mean_abs_update - b.mean_abs_update),
                max_abs_gap=abs(a.max_abs_update - b.max_abs_update),
                l2_gap=abs(a.l2_update - b.l2_update),
                rel_l2_gap=abs(a.rel_l2_update - b.rel_l2_update),
            )
        )
    return gaps, only_a, only_b


def _summarize_diff_gaps(gaps: list[DiffFileGap]) -> dict[str, float]:
    if not gaps:
        return {
            "num_params": 0.0,
            "numel_mismatch": 0.0,
            "max_abs_gap_max": 0.0,
            "mean_abs_gap_mean": 0.0,
            "l2_gap_mean": 0.0,
            "rel_l2_gap_mean": 0.0,
        }
    return {
        "num_params": float(len(gaps)),
        "numel_mismatch": float(sum(0 if g.numel_match else 1 for g in gaps)),
        "max_abs_gap_max": float(max(g.max_abs_gap for g in gaps)),
        "mean_abs_gap_mean": float(sum(g.mean_abs_gap for g in gaps) / len(gaps)),
        "l2_gap_mean": float(sum(g.l2_gap for g in gaps) / len(gaps)),
        "rel_l2_gap_mean": float(sum(g.rel_l2_gap for g in gaps) / len(gaps)),
    }


def _load_state_dict(path: str) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError(f"Expected dict state_dict in {path}, got {type(state)}")
    return {k: v.detach().float() for k, v in state.items()}


def _compare_state_dicts(
    state_a: dict[str, torch.Tensor],
    state_b: dict[str, torch.Tensor],
) -> tuple[list[ParamDiff], list[str], list[str]]:
    names_a = set(state_a.keys())
    names_b = set(state_b.keys())
    shared = sorted(names_a & names_b)
    only_a = sorted(names_a - names_b)
    only_b = sorted(names_b - names_a)

    diffs: list[ParamDiff] = []
    for name in shared:
        a = state_a[name]
        b = state_b[name]
        shape_ok = a.shape == b.shape
        if not shape_ok:
            diffs.append(
                ParamDiff(
                    name=name,
                    shape_match=False,
                    max_diff=float("inf"),
                    mean_diff=float("inf"),
                    l2_diff=float("inf"),
                    rel_l2_diff=float("inf"),
                )
            )
            continue
        delta = a - b
        l2_a = float(a.norm().item())
        l2_delta = float(delta.norm().item())
        diffs.append(
            ParamDiff(
                name=name,
                shape_match=True,
                max_diff=float(delta.abs().max().item()),
                mean_diff=float(delta.abs().mean().item()),
                l2_diff=l2_delta,
                rel_l2_diff=l2_delta / max(l2_a, 1e-12),
            )
        )
    return diffs, only_a, only_b


def _summarize_param_diffs(diffs: list[ParamDiff]) -> dict[str, float]:
    if not diffs:
        return {
            "num_params": 0,
            "global_max_diff": 0.0,
            "global_mean_diff": 0.0,
            "global_l2_diff": 0.0,
            "global_rel_l2_diff": 0.0,
        }
    matched = [d for d in diffs if d.shape_match]
    if not matched:
        return {
            "num_params": len(diffs),
            "global_max_diff": float("inf"),
            "global_mean_diff": float("inf"),
            "global_l2_diff": float("inf"),
            "global_rel_l2_diff": float("inf"),
        }
    max_diff = max(d.max_diff for d in matched)
    total_l2 = math.sqrt(sum(d.l2_diff**2 for d in matched))
    mean_diff = sum(d.mean_diff for d in matched) / len(matched)
    rel_l2 = sum(d.rel_l2_diff for d in matched) / len(matched)
    return {
        "num_params": len(diffs),
        "global_max_diff": float(max_diff),
        "global_mean_diff": float(mean_diff),
        "global_l2_diff": float(total_l2),
        "global_rel_l2_diff": float(rel_l2),
    }


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------


def _print_loss_report(
    diffs: list[LossDiff],
    atol: float,
    rtol: float,
) -> bool:
    print("\n=== Per-step loss comparison (strict) ===")
    print(f"atol={atol:.3e} rtol={rtol:.3e}")
    print(
        f"{'step':>4} {'loss_a':>14} {'loss_b':>14} "
        f"{'abs_gap':>12} {'rel_gap':>12} {'status':>8}"
    )
    ok = True
    for d in diffs:
        status = "OK" if d.aligned else "MISMATCH"
        ok = ok and d.aligned
        print(
            f"{d.step:>4d} {d.loss_a:>14.6f} {d.loss_b:>14.6f} "
            f"{d.abs_gap:>12.3e} {d.rel_gap:>12.3e} {status:>8}"
        )
    print(f"Loss alignment overall: {'PASS' if ok else 'FAIL'}")
    return ok


def _print_param_report(
    label: str,
    diffs: list[ParamDiff],
    only_a: list[str],
    only_b: list[str],
    top_k: int = 10,
) -> None:
    summary = _summarize_param_diffs(diffs)
    print(f"\n=== {label} parameter comparison (informational) ===")
    if only_a or only_b:
        print(
            f"[warn] parameter key mismatch: only_a={only_a[:5]}{'...' if len(only_a) > 5 else ''} "
            f"only_b={only_b[:5]}{'...' if len(only_b) > 5 else ''}"
        )
    for k, v in summary.items():
        print(f"  {k}: {v}")
    worst = sorted(diffs, key=lambda d: d.max_diff, reverse=True)[:top_k]
    print(f"  top-{len(worst)} tensors by max_diff:")
    for d in worst:
        print(
            f"    {d.name[:80]:<80} max={d.max_diff:.3e} "
            f"mean={d.mean_diff:.3e} rel_l2={d.rel_l2_diff:.3e} "
            f"shape_match={d.shape_match}"
        )


def _print_diff_file_report(
    label: str,
    gaps: list[DiffFileGap],
    only_a: list[str],
    only_b: list[str],
    top_k: int = 10,
) -> None:
    summary = _summarize_diff_gaps(gaps)
    print(f"\n=== {label} diff.pt comparison (informational) ===")
    if only_a or only_b:
        print(
            f"[warn] parameter key mismatch: only_a={only_a[:5]}{'...' if len(only_a) > 5 else ''} "
            f"only_b={only_b[:5]}{'...' if len(only_b) > 5 else ''}"
        )
    for k, v in summary.items():
        print(f"  {k}: {v}")
    worst = sorted(gaps, key=lambda g: g.max_abs_gap, reverse=True)[:top_k]
    print(f"  top-{len(worst)} tensors by max_abs_gap:")
    for g in worst:
        print(
            f"    {g.name[:80]:<80} max_abs_gap={g.max_abs_gap:.3e} "
            f"mean_abs_gap={g.mean_abs_gap:.3e} rel_l2_gap={g.rel_l2_gap:.3e} "
            f"numel_match={g.numel_match}"
        )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def run_comparison(
    dump_a: str,
    dump_b: str,
    *,
    loss_atol: float,
    loss_rtol: float,
    compare_initial: bool,
) -> bool:
    print(f"[compare] dump_a={dump_a}")
    print(f"[compare] dump_b={dump_b}")

    stats_a = _load_stats(dump_a)
    stats_b = _load_stats(dump_b)
    loss_diffs = _compare_losses(stats_a, stats_b, atol=loss_atol, rtol=loss_rtol)
    loss_ok = _print_loss_report(loss_diffs, loss_atol, loss_rtol)

    diff_a = Path(dump_a) / "diff.pt"
    diff_b = Path(dump_b) / "diff.pt"
    if diff_a.exists() and diff_b.exists():
        sig_a = _load_diff_signatures(str(diff_a))
        sig_b = _load_diff_signatures(str(diff_b))
        gaps, only_a, only_b = _compare_diff_signatures(sig_a, sig_b)
        _print_diff_file_report("Final", gaps, only_a, only_b)
    else:
        final_a = Path(dump_a) / "params.pt"
        final_b = Path(dump_b) / "params.pt"
        if final_a.exists() and final_b.exists():
            print(
                "\n[info] diff.pt not found in both dumps; "
                "falling back to legacy params.pt comparison."
            )
            state_a = _load_state_dict(str(final_a))
            state_b = _load_state_dict(str(final_b))
            diffs, only_a, only_b = _compare_state_dicts(state_a, state_b)
            _print_param_report("Final", diffs, only_a, only_b)
        else:
            print(
                f"\n[info] skipping final param comparison "
                f"(diff exists: a={diff_a.exists()}, b={diff_b.exists()}; "
                f"params exists: a={final_a.exists()}, b={final_b.exists()})"
            )

    if compare_initial:
        init_a = Path(dump_a) / "params_initial.pt"
        init_b = Path(dump_b) / "params_initial.pt"
        if init_a.exists() and init_b.exists():
            state_a = _load_state_dict(str(init_a))
            state_b = _load_state_dict(str(init_b))
            diffs, only_a, only_b = _compare_state_dicts(state_a, state_b)
            _print_param_report("Initial", diffs, only_a, only_b)

    return loss_ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare two ArchonEngine training-test dump_dirs."
    )
    parser.add_argument("--dump-a", type=str, required=True, help="First dump dir.")
    parser.add_argument("--dump-b", type=str, required=True, help="Second dump dir.")
    parser.add_argument(
        "--loss-atol",
        type=float,
        default=1e-6,
        help="Absolute tolerance for per-step loss alignment (default 1e-6).",
    )
    parser.add_argument(
        "--loss-rtol",
        type=float,
        default=1e-6,
        help="Relative tolerance for per-step loss alignment (default 1e-6).",
    )
    parser.add_argument(
        "--compare-initial",
        action="store_true",
        help="Also compare legacy params_initial.pt if present in both dumps.",
    )
    args = parser.parse_args(argv)

    ok = run_comparison(
        dump_a=args.dump_a,
        dump_b=args.dump_b,
        loss_atol=args.loss_atol,
        loss_rtol=args.loss_rtol,
        compare_initial=args.compare_initial,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

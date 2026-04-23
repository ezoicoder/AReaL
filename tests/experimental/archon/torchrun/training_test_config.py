"""Configuration types + YAML/CLI loader for ArchonEngine training tests.

The runner expects a YAML file with at least these sections:

```yaml
engine:              # Standard AReaL TrainEngineConfig fields (see cli_args.py).
  experiment_name: archon_train_test
  # trial_name is optional in this test loader (defaults to "trial0").
  path: /path/to/model
  dtype: bfloat16
  mb_spec:
    max_tokens_per_mb: 5596
  optimizer:          # Ignored when test_config.disable_optimizer=true.
    type: adam
    lr: 1e-5
    ...
  tree_training_mode: dta
  partition_mode: seqlen

parallel: archon:d2 # Or use a mapping with ParallelStrategy fields.

test_config:         # Test-only knobs, see ``TestOnlyConfig``.
  step: 4
  data_dir: /path/to/data_dir
  disable_optimizer: false
  fileroot: /storage/openpsi/experiments
  save_diff: true
```

OmegaConf-style dotlist overrides are supported on the CLI, eg::

    torchrun --nproc_per_node=2 run_archon_training_test.py \
        --config config.yaml \
        test_config.step=4 test_config.disable_optimizer=true
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import re
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union

from omegaconf import DictConfig, OmegaConf

from areal.api.alloc_mode import AllocationMode, ParallelStrategy
from areal.api.cli_args import TrainEngineConfig


@dataclass
class TestOnlyConfig:
    """Test-only settings not inherited from AReaL configs."""

    step: int = -1
    data_dir: str = ""
    disable_optimizer: bool = False
    fileroot: str = "/storage/openpsi/experiments"
    prompt_ratio: float = 0.3
    save_diff: bool = True
    save_params: bool = False
    save_initial_params: bool = False
    seed: int = 42

    def __post_init__(self) -> None:
        if self.step is None or int(self.step) < 0:
            raise ValueError(
                f"test_config.step must be a non-negative integer, got {self.step}."
            )
        if not self.data_dir:
            raise ValueError(
                "test_config.data_dir is required and must be a non-empty path."
            )


@dataclass
class TestParallelConfig:
    """Subset of ParallelStrategy fields exposed to the test YAML."""

    data_parallel_size: int = 1
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    context_parallel_size: int = 1
    expert_parallel_size: int = 1
    expert_tensor_parallel_size: int = 1

    def to_parallel_strategy(self) -> ParallelStrategy:
        return ParallelStrategy(
            tensor_parallel_size=self.tensor_parallel_size,
            pipeline_parallel_size=self.pipeline_parallel_size,
            data_parallel_size=self.data_parallel_size,
            context_parallel_size=self.context_parallel_size,
            expert_parallel_size=self.expert_parallel_size,
            expert_tensor_parallel_size=self.expert_tensor_parallel_size,
        )

    def to_compact_tag(self) -> str:
        """Compact path-friendly tag, e.g. ``d8t2c2``."""
        parts = [f"d{int(self.data_parallel_size)}"]
        if int(self.pipeline_parallel_size) > 1:
            parts.append(f"p{int(self.pipeline_parallel_size)}")
        if int(self.tensor_parallel_size) > 1:
            parts.append(f"t{int(self.tensor_parallel_size)}")
        if int(self.context_parallel_size) > 1:
            parts.append(f"c{int(self.context_parallel_size)}")
        if int(self.expert_parallel_size) > 1:
            parts.append(f"e{int(self.expert_parallel_size)}")
        if int(self.expert_tensor_parallel_size) > 1:
            parts.append(f"et{int(self.expert_tensor_parallel_size)}")
        return "".join(parts)


@dataclass
class ArchonTrainingTestConfig:
    """Top-level container combining AReaL engine config + test knobs."""

    engine: TrainEngineConfig = field(default_factory=TrainEngineConfig)
    parallel: TestParallelConfig = field(default_factory=TestParallelConfig)
    test_config: TestOnlyConfig = field(default_factory=TestOnlyConfig)

    @staticmethod
    def _safe_token(value: str, *, fallback: str) -> str:
        s = (value or "").strip()
        if not s:
            s = fallback
        s = s.replace(os.sep, "_")
        if os.altsep:
            s = s.replace(os.altsep, "_")
        s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
        s = s.strip("._-")
        return s or fallback

    @staticmethod
    def _expand_path(path: str) -> str:
        return os.path.expanduser(os.path.expandvars(path))

    def resolve_dump_dir(self) -> str:
        """Pick a compact dump_dir rooted by experiment name."""
        exp = self._safe_token(
            str(self.engine.experiment_name or "archon_train_test"),
            fallback="archon_train_test",
        )
        tree_mode = self._safe_token(
            str(getattr(self.engine, "tree_training_mode", "unknown") or "unknown"),
            fallback="unknown",
        )
        parallel_tag = self._safe_token(self.parallel.to_compact_tag(), fallback="d1")
        model_name = self._safe_token(
            Path(str(self.engine.path or "")).name, fallback="model"
        )
        leaf = f"{tree_mode}_{parallel_tag}_{model_name}"

        if self.test_config.fileroot:
            base = Path(self._expand_path(self.test_config.fileroot)) / exp
        else:
            base = Path.cwd() / exp
        return str(base / leaf)


def _merge_yaml_and_overrides(
    yaml_path: str,
    overrides: list[str],
) -> DictConfig:
    yaml_cfg = OmegaConf.load(yaml_path)
    if not isinstance(yaml_cfg, DictConfig):
        raise ValueError(
            f"Top-level YAML at {yaml_path} must be a mapping, got {type(yaml_cfg)}."
        )
    override_cfg = OmegaConf.from_dotlist(list(overrides))
    return OmegaConf.merge(yaml_cfg, override_cfg)


def _as_dict(section: Any) -> dict[str, Any]:
    """Resolve an OmegaConf node into a plain ``dict``."""
    if section is None:
        return {}
    if isinstance(section, DictConfig):
        return OmegaConf.to_container(section, resolve=True)  # type: ignore[return-value]
    if isinstance(section, dict):
        return dict(section)
    raise TypeError(f"Expected mapping-like config section, got {type(section)}")


def _coerce_value(tp: Any, value: Any) -> Any:
    """Best-effort coercion of ``value`` to the dataclass field type ``tp``.

    Handles ``Optional[X]``, nested dataclasses, and ``list[DataClass]`` /
    ``tuple[DataClass, ...]``. Other annotations (``Literal``, ``int``, ``str``,
    ``dict``, ...) pass through unchanged so OmegaConf primitives continue to
    work.
    """
    if value is None:
        return None

    origin = typing.get_origin(tp)
    args = typing.get_args(tp)

    if origin is Union or origin is types.UnionType:
        # Try the non-None variants in order; first one that accepts the value
        # wins. Primitives pass through since ``_coerce_value`` is a no-op for
        # non-dataclass leaf types.
        non_none = [a for a in args if a is not type(None)]
        for alt in non_none:
            if dataclasses.is_dataclass(alt) and isinstance(value, dict):
                return _build_dataclass(alt, value)
        return value

    if dataclasses.is_dataclass(tp):
        if isinstance(value, dict):
            return _build_dataclass(tp, value)
        return value

    if origin in (list, tuple) and args:
        inner = args[0]
        if dataclasses.is_dataclass(inner) and isinstance(value, (list, tuple)):
            coerced = [_coerce_value(inner, v) for v in value]
            return tuple(coerced) if origin is tuple else coerced

    return value


def _build_dataclass(cls: type, data: dict[str, Any]) -> Any:
    """Instantiate ``cls`` from ``data``, recursively coercing nested fields.

    Fields not present in ``data`` fall back to their dataclass defaults so
    partial YAML sections are allowed. Unknown keys raise ``TypeError`` to
    surface typos early.
    """
    assert dataclasses.is_dataclass(cls), f"{cls} is not a dataclass"
    hints = typing.get_type_hints(cls)
    init_kwargs: dict[str, Any] = {}
    known_names = {f.name for f in dataclasses.fields(cls) if f.init}
    for key, value in data.items():
        if key not in known_names:
            raise TypeError(
                f"Unknown field '{key}' for {cls.__name__}; "
                f"valid fields: {sorted(known_names)[:30]}..."
            )
    for f in dataclasses.fields(cls):
        if not f.init:
            continue
        if f.name not in data:
            continue
        tp = hints.get(f.name, f.type)
        init_kwargs[f.name] = _coerce_value(tp, data[f.name])
    return cls(**init_kwargs)


def _build_engine_config(section: Any) -> TrainEngineConfig:
    """Build a :class:`TrainEngineConfig` from a YAML section dict."""
    data = _as_dict(section)
    # Keep test YAML concise: allow omitting experiment/trial names.
    data.setdefault("experiment_name", "archon_train_test")
    data.setdefault("trial_name", "trial0")
    return _build_dataclass(TrainEngineConfig, data)


def _parallel_strategy_to_test_config(strategy: ParallelStrategy) -> TestParallelConfig:
    """Convert ``ParallelStrategy`` to ``TestParallelConfig``."""
    return TestParallelConfig(
        data_parallel_size=int(strategy.data_parallel_size),
        tensor_parallel_size=int(strategy.tensor_parallel_size),
        pipeline_parallel_size=int(strategy.pipeline_parallel_size),
        context_parallel_size=int(strategy.context_parallel_size),
        expert_parallel_size=int(strategy.expert_parallel_size),
        expert_tensor_parallel_size=int(strategy.expert_tensor_parallel_size),
    )


def _build_parallel_config(section: Any) -> TestParallelConfig:
    """Build ``TestParallelConfig`` from mapping or allocation-mode string.

    Supported forms:
    - Mapping (legacy):
        parallel:
          data_parallel_size: 2
          tensor_parallel_size: 1
          ...
    - String (reuses AllocationMode parser):
        parallel: archon:d8
        parallel: sglang:d16+archon:d8
    """
    if section is None:
        return TestParallelConfig()

    if isinstance(section, str):
        mode = AllocationMode.from_str(section)
        train_allocs = [
            a for a in mode.allocations if a.backend in ("fsdp", "megatron", "archon")
        ]
        if len(train_allocs) != 1:
            raise ValueError(
                "parallel string must resolve to exactly one training allocation "
                f"(got {len(train_allocs)}): {section}"
            )
        alloc = train_allocs[0]
        if alloc.backend != "archon":
            raise ValueError(
                "Only archon backend is supported by this test runner. "
                f"Got training backend '{alloc.backend}' from parallel='{section}'."
            )
        if alloc.parallel is None:
            raise ValueError(
                f"Resolved archon allocation has no parallel strategy: {section}"
            )
        return _parallel_strategy_to_test_config(alloc.parallel)

    return TestParallelConfig(**_as_dict(section))


def load_training_test_config(
    argv: list[str] | None = None,
) -> tuple[ArchonTrainingTestConfig, str]:
    """Parse CLI and return a resolved ``ArchonTrainingTestConfig``."""
    parser = argparse.ArgumentParser(
        description="Run ArchonEngine training-side test under torchrun."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML config file.",
    )
    args, overrides = parser.parse_known_args(argv)
    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    merged = _merge_yaml_and_overrides(str(config_path), overrides)

    engine_cfg = _build_engine_config(merged.get("engine") if merged else None)
    parallel_cfg = _build_parallel_config(merged.get("parallel") if merged else None)
    test_cfg = TestOnlyConfig(**_as_dict(merged.get("test_config") if merged else None))

    cfg = ArchonTrainingTestConfig(
        engine=engine_cfg,
        parallel=parallel_cfg,
        test_config=test_cfg,
    )
    return cfg, str(config_path)


def ensure_dump_dir(cfg: ArchonTrainingTestConfig, rank: int) -> str:
    """Create (on rank 0) and return the resolved dump_dir."""
    dump_dir = cfg.resolve_dump_dir()
    if rank == 0:
        os.makedirs(dump_dir, exist_ok=True)
    return dump_dir

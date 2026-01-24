# Tree Training Tests 说明文档

## 概述

测试树形训练优化技术（Flex 和 Stack 模式），**单卡环境**（`WORLD_SIZE=1`）。

### 数据和模型配置

在 `test_tree_training.py` 中可配置：

```python
# 模型路径（优先使用本地路径，不存在则使用 HuggingFace）
MODEL_PATH = get_model_path(
    "/data/tree/models/Qwen3-8B", "Qwen/Qwen3-8B"
)

# 测试数据路径
TREE_DATA_PATH = "/data/tree/tree-data/tau2-16k-small/call2_rank0.pt"
```

### Loss 函数配置

当前使用 `grpo_loss_fn` 的 partial 包装版本（第 242-258 行）：

```python
loss_fn = partial(
    grpo_loss_fn,
    eps_clip=0.2,
    importance_sampling_level="token",
    prox_logp_method="recompute",
    # ... 其他参数
)
```

**重要注意事项**：

1. **Logprobs 填充差异**：
   - **Baseline/Flex 模式**：logprobs 最后一位可能非 0（正常计算）
   - **Stack 模式**：logprobs 最后一位为 0（填充位）
   - Loss 函数需要正确处理这种差异

2. **Microbatch 线性可加性要求**：
   
   Loss 函数必须满足以下等式，确保 microbatch 拆分不影响梯度：
   
   ```
   loss_fn(a) * loss_weight_fn(a) + loss_fn(b) * loss_weight_fn(b) 
   = loss_fn(cat(a,b)) * loss_weight_fn(cat(a,b))
   ```
   
   这要求 loss 对序列具有线性可加性。如果自定义 loss_fn 和 loss_weight_fn，务必验证此性质。

## 测试模块

| 测试函数 | 模式 | 功能 | 输出内容 |
|---------|------|------|---------|
| `test_fsdp_flex_forward` | Flex | 前向传播测试 | 时间、logprobs 正确性 |
| `test_fsdp_stack_forward` | Stack | 前向传播测试 | 时间、logprobs 正确性 |
| `test_fsdp_flex_backward` | Flex | 反向传播+梯度验证 | 时间、加速比、显存对比、梯度正确性 |
| `test_fsdp_stack_backward` | Stack | 反向传播+梯度验证 | 时间、加速比、显存对比、梯度正确性 |
| `test_flex` | Flex | 独立性能测试 | 时间、峰值显存 |
| `test_stack` | Stack | 独立性能测试 | 时间、峰值显存 |
| `test_baseline` | Baseline | 独立性能测试 | 时间、峰值显存 |

## 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--max-tokens-per-mb` | int | 16384 | 每个 microbatch 的最大 token 数（A100 80GB 推荐 16384，显存小可降至 8192/4096）|
| `--prefix-len` | int | -1 | 限制测试序列数量（-1 = 全部序列） |
| `--disable-gradient-checkpointing` | flag | False | 禁用梯度检查点（默认启用，仅 Flex 和 Baseline 支持） |

**环境变量**: `AREAL_FLEX_ATTENTION_BLOCK_SIZE=64`（Flex 模式需要）

## 使用方法

### 基本命令（建议加 `-v -s`）

```bash
python -m pytest areal/tests/test_tree_training.py::<test_name> -v -s [options]
```

### 示例

```bash
# Flex 反向传播测试（默认启用梯度检查点）
AREAL_FLEX_ATTENTION_BLOCK_SIZE=64 python -m pytest areal/tests/test_tree_training.py::test_fsdp_flex_backward -v -s --max-tokens-per-mb 16384

# Stack 反向传播测试
python -m pytest areal/tests/test_tree_training.py::test_fsdp_stack_backward -v -s --max-tokens-per-mb 16384
```

## 重要说明

### 梯度检查点

- **默认启用**: Flex 和 Baseline 模式
- **禁用方式**: 添加 `--disable-gradient-checkpointing` 参数
- **Stack 模式**: 不支持梯度检查点

## 相关文件

- 测试代码: `areal/tests/test_tree_training.py`
  - 配置数据路径: `TREE_DATA_PATH`（第 35 行）
  - 配置模型路径: `MODEL_PATH`（第 30-32 行）
  - 配置 loss 函数: `loss_fn`（第 242-258 行）
- 参数配置: `areal/tests/conftest.py`
- FSDP Engine: `areal/engine/fsdp_engine.py`
- Tree 实现: `areal/models/tree_attn/`

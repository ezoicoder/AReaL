# tree-attn
**gradient_checkpoint 和 enable_tree_attn_training 不能同时存在**

在 2xA100(40G) 配置上，可用如下方式启动三种不同的训练模式：

```bash
cd AReaL

# 启动 baseline
bash examples/multi_turn_math/baseline.sh

# 启动 tree_flex
bash examples/multi_turn_math/tree_flex.sh

# 启动 tree_attn
bash examples/multi_turn_math/tree_attn.sh
```

每个脚本分别对应不同的训练配置和训练模式。

# Training a Multi-Turn GSM8K Math Agent in AReaL

Files in this folder presents an example that train a multi-turn GSM8K math agent from
Qwen/Qwen2.5-1.5B-Instruct, using `ArealOpenAI` APIs and its `concat` mode to organize
training data and discount reward.

# To run the example

```bash
python3 -m areal.launcher.ray examples/multi_turn_math/gsm8k_rl_mt.py \
    --config examples/multi_turn_math/gsm8k_grpo_mt.yaml \
    experiment_name=gsm8k-grpo-multiturn trial_name=trial0
```

only the following config are added compared to the original `gsm8k_grpo.yaml` config:

```yaml
export_style: concat
agent_run_args:
  max_turns: 2
```

## Reward Curve

<img align="center" alt="reward curve" src="reward_curve.png" width="100%">

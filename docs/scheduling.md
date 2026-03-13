# AReaL 调度机制分析

## 支持的调度方式

支持 **Slurm**、**Ray**、**Local** 三种，各有两层：

| 层级 | 作用 |
|---|---|
| Launcher 层（`areal/launcher/`） | 整个训练 job 的提交入口，启动推理和训练进程 |
| Scheduler 层（`areal/scheduler/`） | 单控制器模式，按 worker 角色细粒度提交 |

---

## 资源配置

### allocation_mode

决定 GPU 如何在推理和训练之间分配：

```
sglang:d4p1t1+d4p1t1
         ↑          ↑
    推理侧(4路DP) 训练侧(4路DP)
```

- `+`：decoupled，推理和训练各占独立节点
- `|`：colocated，共享同一批节点

### SchedulingSpec

控制每种角色的资源预算，推理（`rollout.scheduling_spec`）和训练（`actor.scheduling_spec`）可以**独立配置**：

```yaml
actor:
  scheduling_spec:
    - gpu: 1
      cpu: 4
      mem: 32
      image: /path/to/trainer.sif

rollout:
  scheduling_spec:
    - gpu: 1
      cpu: 8        # 推理需要更多 CPU
      mem: 64
      image: /path/to/inference.sif
      nodelist: gpu-inference-[01-04]   # 指定节点
```

`nodelist`、`exclude`、`srun_additional_args`（可加 `--partition=xxx`）均可独立设置。

---

## Slurm 模式实现细节

### 两个独立的 sbatch job

推理（`llm_server`）和训练（`trainer`）是**两个完全独立的 Slurm job**，各自有独立的 Job ID，独立排队。训练 job 等待推理 job 通过 name_resolve 注册地址后才提交。

### sbatch script 结构

每个 job 生成一个 `.sh` 文件，结构如下：

```
sbatch script
├── 探测 head node IP（小 srun 调 hostname）
├── 探测空闲端口
└── N 条 srun & （后台并行）
    └── bg_pids 收集 PID，while 循环监控
        └── 任意一条失败 → exit 1 → trap 杀掉所有后台进程
```

### task 粒度

```
1 task = 1 节点 = 该节点全部 GPU（ntasks_per_node 固定为 1）
```

- **训练侧**：task 内跑 `torchrun --nnodes=N --nproc-per-node=8 --node-rank=i`，torchrun fork 出 8 个 GPU 子进程。`--nnodes=N` 是告诉 PyTorch 分布式全局有几个节点，`srun --nodes=1` 只是把这条命令放到指定节点上。
- **推理侧**：task 内跑 SGLang/vLLM server，内部用 tensor parallelism 管理全部 GPU。

### GPU 资源规则

- `--gres=gpu:N` 是 per-node 分配，在 job 的资源池中**独占**，并发 step 之间有竞争（不够就等待，不报错）
- 单个 srun 内多个 task 共享 gres，要独占需用 `--gpus-per-task`
- `CUDA_VISIBLE_DEVICES` 由 CUDA 驱动层强制，进程无法访问列表外的 GPU

---

## Ray 模式实现细节

### 前提：Ray cluster 已存在

Ray 不分配物理资源，需要先有一个 Ray cluster（例如通过 SkyPilot 在云上拉起）：

```bash
# node0
ray start --head --port=6379
python3 -m areal.launcher.ray ...

# node1+
ray start --address $head_ip:6379
```

`ray.init()` 只是连接已有集群，不申请资源。

### Placement Group

每个角色（`llm_server`、`trainer`）各创建一个 Placement Group，`strategy="PACK"` 把 task 压到同一批节点：

```python
placement_group = ray.util.placement_group(bundles=[...] * nodes, strategy="PACK")
```

### task 粒度（与 Slurm 的关键差异）

- **推理侧**：1 task = 1 节点（`gpus_per_task=n_gpus_per_node`），与 Slurm 相同
- **训练侧**：1 task = **1 个 GPU 进程**（`gpus_per_task=1`），`LOCAL_RANK` 永远是 0

Ray 把 torchrun 的工作自己实现了：Placement Group 负责节点亲和性，`torch_env_hook` 注入 `RANK`/`WORLD_SIZE`/`MASTER_ADDR`/`MASTER_PORT`。

### 资源隔离对比

| 维度 | Slurm | Ray |
|---|---|---|
| GPU | `CUDA_VISIBLE_DEVICES`（驱动强制，等价） | `CUDA_VISIBLE_DEVICES`（驱动强制，等价） |
| CPU | cgroups cpuset，物理绑核 | 逻辑计数，不绑核 |
| 内存 | cgroups 硬限制，超出 OOM kill | 逻辑计数，超出不管 |
| 小数 GPU | 不支持（最小单位 1，或 MIG 硬件切分） | 支持（仅调度逻辑，无隔离） |

---

## Local 模式

单机，`subprocess.Popen`，GPU 轮询分配 `CUDA_VISIBLE_DEVICES`，训练用 `torchrun --nnodes 1`，无资源隔离。

# catan-rl-llm

> 训练 Qwen3-8B 玩《卡坦岛》（Settlers of Catan）的完整实验仓库。

---

## 1. 一句话状态

**当前最强方案：Hybrid Agent（工具 + VF guardrail）= 100% (6/6) 胜率 vs WeightedRandom**。完整实验历程、失败教训、决策依据见 [`PROJECT_SUMMARY.md`](./PROJECT_SUMMARY.md)。

| 方案 | 胜率 | guardrail | 备注 |
|---|---|---|---|
| **Hybrid Agent（tools + VF）** | **100% (6/6)** | VF | 当前最强 |
| Hybrid Agent（tools only） | 66.7% (2/3) | 无 | 工具够战略，缺战术精修 |
| VF-Distill v2（独立模型） | 40% (8/20) | 无 | 文本观察不够 |
| AB-SFT（纯模仿） | 25% (5/20) | 无 | 模仿只学格式不学策略 |
| RL-Guard（RL 模型打分） | 0–67% 不稳定 | RL | RL 模型预测 outcome 不预测 action 质量 |
| Hybrid Agent（tools + RL guard） | 0% (0/3) | RL | RL guard 是有害的 |

---

## 2. 项目目标

以 Qwen3-8B 为基座、围绕 catanatron-gym 引擎，研究 SFT / RL / 蒸馏 / 工具调用 / Guardrail 等不同训练范式在多人博弈场景下的有效性。

最终目标层级：

| 层级 | 描述 | 当前状态 |
|---|---|---|
| L0 | 正确格式输出合法动作（≥ 95%） | ✅ 100% |
| L1 | 击败 WeightedRandom 基线（4 人局 > 25%） | ✅ 最高 100% |
| L2 | 击败 VictoryPointPlayer（最强内置） | ❌ 未达成 |
| L3 | 强到可以作为研究对象 | ❌ 未达成 |

---

## 3. 时间线（短版）

每个方法 3–5 行要点；完整论证见 [`PROJECT_SUMMARY.md`](./PROJECT_SUMMARY.md) 第 2 章。

| # | 方法 | 日期 | 关键结果 |
|---|---|---|---|
| 1 | AB-SFT（纯模仿） | 2026-08-07 | loss 0.044、token acc 98%、WR **25%**（基线齐平） |
| 2 | VF-Guard（发现） | 2026-08-07 | LLM + 手写 VF 打分，WR **90%**（9/10） |
| 3 | VF-Distill v2 | 2026-08-07 | override-only 蒸馏，WR **40%**（8/20） |
| 4 | RL-Guard | 2026-08-07 | 30 特征 RL 模型打分，WR **0–67%** 不稳定 |
| 5 | Hybrid Agent v1 | 2026-08-07 | 工具 + VF，WR **100%** (3/3) |
| 6 | GRPO / VF rollout | 2026-08-08 | 三组数据全失败，WR **0–20%** |
| 7 | AESL entropy 早停 | 2026-08-08 | 假说被拒绝，峰值点 **0%** WR |
| 8 | RL 模型修复 | 2026-08-08 | 30→72 特征 + VF 残差，WR **44% vs WeightedRandom** |
| 9 | Hybrid Agent 消融 | 2026-08-08 | tools+VF 100%、tools+RL 0%、tools only 66.7% |
| 10 | Option C 课程 | 2026-08-08 | outcome label 噪声，WR **12–38%** |

---

## 4. 快速上手

### 4.1 环境

```bash
# 硬件：NVIDIA RTX 4090 D 24GB（AutoDL 容器）
# Python ≥ 3.10，PyTorch ≥ 2.1
cd catan-rl-llm
pip install -r requirements.txt
# 或 poetry / pip install -e .
```

主要依赖：`torch`、`transformers`、`trl`、`peft`、`catanatron-gym`、`gymnasium`、`vllm`。

### 4.2 数据生成

```bash
# 第一次 SFT 数据（VictoryPointPlayer × WeightedRandom 100 局）
python scripts/generate_sft_data.py --num_games 100 --output data/sft/ --seed 42

# AB-SFT 数据（VictoryPointPlayer × WeightedRandom 300 局，更大规模）
python scripts/generate_ab_sft_data.py --num_games 300 --output data/ab_sft/

# VF 蒸馏数据（VF-Guard 覆盖 LLM 的决策）
python scripts/generate_vf_distill_data_v2.py

# GRPO rollout 数据（bot-vs-bot）
python scripts/generate_grpo_data.py --num_games 100 --output data/grpo/iter1/
```

### 4.3 训练

```bash
# SFT（从 Qwen3-8B 基座）
python scripts/train_sft.py \
    --data data/sft/ \
    --output checkpoints/sft/

# AB-SFT
python scripts/train_sft_best.py \
    --data data/ab_sft/ \
    --output checkpoints/ab_sft/

# VF 蒸馏
python scripts/train_vf_distill_v2.py

# AESL entropy 监控训练
python scripts/train_aesl.py

# 价值网络（30 特征旧版 / 72 特征 VF 残差版）
python scripts/train_value_network.py
```

### 4.4 评估

```bash
# AB-SFT 模型
python scripts/eval_ab_sft.py --model checkpoints/ab_sft/checkpoint-200/

# VF-Guard
python scripts/eval_vf_guard.py --games 10

# RL-Guard
python scripts/eval_rl_guard.py --games 10

# Hybrid Agent（最强方案）
python scripts/eval_agent_hybrid.py --games 6

# Hybrid Agent 消融（3 配置）
python scripts/eval_hybrid_ablation.py

# 最终统一评估
python scripts/run_final_eval.py
```

---

## 5. 项目结构

```
catan-rl-llm/
├── README.md              # 本文件（用法 + 当前最强方案）
├── PROJECT_SUMMARY.md     # 完整实验档案（决策依据 + 失败教训）
├── pyproject.toml
├── requirements.txt
├── .env                   # HF_TOKEN / WANDB_API_KEY 占位
│
├── configs/               # YAML：default / sft / grpo / eval
├── src/catan_rl/          # 代码包
│   ├── agent/             # LlamaGym Agent 基类 + Qwen 实现 + 观察/解析/提示
│   ├── env/               # Catanatron 适配、状态序列化、奖励、模拟器
│   ├── rl/                # 特征工程、值网络（30/72 维）、minimax
│   ├── data/              # SFT / GRPO 数据集定义、rollout
│   ├── training/          # 训练入口
│   └── eval/              # 对战 arena、指标、可视化
│
├── scripts/               # 30+ 命令行入口（数据生成 / 训练 / 评估）
├── data/                  # 11 类训练数据子目录
├── checkpoints/           # 14 个模型/目录
├── experiments/           # 4 份阶段性搭建文档
├── notebooks/             # 11 份实验结果（按时间编号）
└── results/               # 最终评估 JSON、arena 复盘、plots
```

---

## 6. 实验文档索引

| 文档 | 用途 |
|---|---|
| [**PROJECT_SUMMARY.md**](./PROJECT_SUMMARY.md) | **完整实验档案**（决策记录 + 失败教训 + 9 章详解）。从「为什么这个方法」到「下一步做什么」都在这里。 |
| [`notebooks/00-original-log-2026-08-06.md`](./notebooks/00-original-log-2026-08-06.md) | 原始实验日志：环境搭建、SFT、GRPO 试训 |
| [`notebooks/01-session-2026-08-08.md`](./notebooks/01-session-2026-08-08.md) | 8 月 8 日全天 session：VF-Guard、RL 修复、Hybrid Agent 评估、Option C |
| [`notebooks/ab-sft-results.md`](./notebooks/ab-sft-results.md) | AB-SFT 25% WR；纯模仿不够 |
| [`notebooks/vf-guard-discovery.md`](./notebooks/vf-guard-discovery.md) | VF-Guard 90% WR；转折点 |
| [`notebooks/option-a-v2-results.md`](./notebooks/option-a-v2-results.md) | VF 蒸馏 v2 40% WR；三处修复 |
| [`notebooks/rl-guard-results.md`](./notebooks/rl-guard-results.md) | RL-Guard 0–67% 不稳定 |
| [`notebooks/hybrid-agent-results.md`](./notebooks/hybrid-agent-results.md) | Hybrid Agent v1 100% (3/3) WR |
| [`notebooks/grpo-results.md`](./notebooks/grpo-results.md) | GRPO / VF rollout 0–20% WR |
| [`notebooks/aesl-experiment-results.md`](./notebooks/aesl-experiment-results.md) | AESL entropy 假说被拒绝 |
| [`notebooks/rl-model-fixed.md`](./notebooks/rl-model-fixed.md) | RL 模型修复：72 特征 + VF 残差 |
| [`notebooks/option-c-curriculum-results.md`](./notebooks/option-c-curriculum-results.md) | Option C 课程 12–38% WR |
| [`notebooks/final-results-2026-08-08.md`](./notebooks/final-results-2026-08-08.md) | 完整管线结果 + Hybrid Agent 消融 |
| [`notebooks/MEMORY.md`](./notebooks/MEMORY.md) | 实验结果导航索引 |
| [`experiments/01_phase1_setup.md`](./experiments/01_phase1_setup.md) | 阶段一：环境搭建 |
| [`experiments/02_phase2_agent.md`](./experiments/02_phase2_agent.md) | 阶段二：Agent 实现 |
| [`experiments/03_phase3_sft.md`](./experiments/03_phase3_sft.md) | 阶段三：SFT 训练 |
| [`experiments/04_phase4_rl.md`](./experiments/04_phase4_rl.md) | 阶段四：GRPO 基础设施 |

---

## 7. 写作风格与文档维护约定

- 中文为主，专有名词与代码标识符保留英文。
- 陈述句为主，不用「令人」「不仅」「更重要的是」这类修饰词堆砌。
- 每个结论后跟证据（实验 notebook 链接或文件路径）。
- 新方法完成后同步更新 `PROJECT_SUMMARY.md` 的 §2、§3、§6。
- 新增 notebook 时同步更新 `PROJECT_SUMMARY.md` 的 §9.3 与本文件的 §6。

---

*最后更新：2026-08-10*
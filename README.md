# Catan RL + LLM：用 GRPO 把 Qwen3-8B 训练成卡坦岛 AI

> **项目仓库**：[`github.com/TFC1224/catan-rl-llm`](https://github.com/TFC1224/catan-rl-llm)
> **更新时间**：2026-08-10
> **整体背景**：[`PROJECT_SUMMARY.md`](../PROJECT_SUMMARY.md) — 描述整个 `llm-rl-catan` 工作区（两个子项目并行）

本项目用 **LlamaGym** 风格的 Agent 模式与 **TRL** 的 **GRPO**（Group Relative Policy Optimization）算法，对 **Qwen3-8B-Instruct** 做监督微调（SFT）和强化学习（RL），目标是让它能像人类玩家一样做出有竞争力的《卡坦岛》（Settlers of Catan）决策。

| 项目 | 内容 |
|---|---|
| 基础模型 | Qwen3-8B-Instruct |
| 微调方式 | 4-bit QLoRA（r=16, α=32, dropout=0.05） |
| 训练范式 | SFT 冷启动 → GRPO 强化学习；当前在 SimSFT 路线收尾，准备进入 AESL 重启 |
| 游戏引擎 | Catanatron v3.2.1 + catanatron-gym v4.0.0 |
| Agent 模式 | LlamaGym：`get_system_prompt` / `format_observation` / `extract_action` |
| 训练框架 | TRL v1.9.2（GRPOTrainer） |
| 硬件 | NVIDIA RTX 4090 D 24 GB（AutoDL 临时容器） |

---

## 目录

1. [项目目标](#1-项目目标)
2. [总体架构](#2-总体架构)
3. [当前状态与关键结果](#3-当前状态与关键结果)
4. [项目结构](#4-项目结构)
5. [Phase 详解：五阶段进展](#5-phase-详解五阶段进展)
6. [关键发现与教训](#6-关键发现与教训)
7. [下一步路线（AESL + GRPO 重启）](#7-下一步路线aesl--grpo-重启)
8. [快速上手](#8-快速上手)
9. [配置与依赖](#9-配置与依赖)
10. [评估方法与可信度](#10-评估方法与可信度)
11. [实验文档索引](#11-实验文档索引)
12. [引用资源](#12-引用资源)

---

## 1. 项目目标

让 **LLM 在《卡坦岛》中学会多目标决策**。

《卡坦岛》相比围棋、象棋有以下独特挑战：

- **多人博弈（4 人）**：奖励信号不仅取决于自己，还要看其他三人的动作和交易；
- **稀疏且随机的奖励**：最终胜利点 10 分，但单局长度 80–120 轮，骰子运气影响极大；
- **状态空间巨大**：地图、资源、交易、建造、道路规划多线交织；
- **动作合法性约束**：每一步只能选当前合法的动作（约 290 维动作空间中被动态过滤）。

本项目的核心科学问题是：**在如此稀疏、随机的奖励下，GRPO 能否通过端到端微调让 LLM 学到强策略？** 回答是：在原始 SFT 冷启动下**不能**，因为冷启动把模型训到过度确定性，GRPO 没有组内多样性可用。这一发现正是 AESL 论文的实验佐证。

---

## 2. 总体架构

```
LLM Agent (Qwen3-8B)  ←→  Catanatron Gym Environment
        │                        │
        │  format_observation()   │  env.get_valid_actions()
        │  extract_action()       │  env.step(action_index)
        ▼                        ▼
   GRPOTrainer (TRL)    ←  Reward via game simulation
```

### Agent 循环（LlamaGym 风格）

```python
# src/catan_rl/agent/qwen_agent.py
class CatanAgent(ABC):
    def get_system_prompt(self) -> str: ...
    def format_observation(self, state, valid_actions) -> str: ...
    def extract_action(self, completion: str, valid_actions) -> int: ...

# 训练时
prompt = system + observation(obs)
completions = llm.generate(prompt, n=K)                # K=4 for GRPO
rewards = [simulate_and_score(c) for c in completions] # 模拟到游戏结束
loss = grpo_loss(completions, rewards)                 # group-relative advantage
```

### 关键技术决策

| 决策 | 选择 | 原因 |
|---|---|---|
| 量化 | 4-bit QLoRA | 24 GB VRAM 装得下 8B + LoRA + 训练状态 |
| 注意力实现 | `sdpa` | 目标硬件不支持 `flash_attention_2` |
| 训练目标 | SFT → GRPO → SimSFT | 标准范式，但需要轻量冷启动 |
| 实验追踪 | `report_to=none` | 沙盒环境未登录 wandb |
| Reward | 模拟游戏终局得分（+1 / -1 / 0），无效动作 -0.5 | 端到端，无人工奖励工程 |

---

## 3. 当前状态与关键结果

> **一句话总结**：完成了 SFT 冷启动（100% 动作合法），但 GRPO 因 SFT 后熵塌陷而失败；改用 SimSFT 路线（44.3% 状态找到更优动作），目前评估受阻于评估脚本输出缓冲；下一步采用 AESL 风格轻量冷启动重置 SFT，再重启 GRPO。

### 3.1 各 Phase 完成度

| Phase | 内容 | 状态 |
|---|---|---|
| **Phase 1** | 环境搭建 | ✅ 完成（18/18 检查通过） |
| **Phase 2** | Agent 实现（5 模块） | ✅ 完成 |
| **Phase 3** | SFT 冷启动（18.5k 数据） | ✅ 完成（100% 动作合法） |
| **Phase 4** | GRPO 强化学习 | ❌ 结构性失败（熵塌陷） |
| **Phase 4b** | SimSFT（模拟引导 SFT） | 🔄 训练完成，评估遇阻 |
| **Phase 5** | AESL 轻量冷启动 | ⏳ 路线已设计，待实施 |

### 3.2 关键数字一览

| 指标 | 数值 | 出处 |
|---|---|---|
| SFT 训练记录 | 18,502（val 2,056） | Phase 3 |
| SFT 训练时长 | ~1h 45m（3 epochs / 564 steps） | Phase 3 |
| SFT 最终 loss | train 0.0887 / eval 0.02186 | Phase 3 |
| SFT 动作合法性 | **100%（1018/1018）** | Phase 3 |
| SFT Token 准确率 | 99.08% | Phase 3 |
| SFT 2-player 胜率（vs WeightedRandom） | 2/6 = 33.3% | Phase 3 |
| GRPO 模型熵（K=4 组内） | **0.01 – 0.06**（极度确定性） | Phase 4 |
| GRPO Advantage | 趋近 0（无学习信号） | Phase 4 |
| GRPO KL 散度 | 1e-9 ~ 1e-4（策略几乎不变） | Phase 4 |
| SimSFT 输入记录 | 1,000（来自 GRPO iter1 rollout） | Phase 4b |
| SimSFT 可优化状态 | 212 条 | Phase 4b |
| SimSFT 找到更优动作比例 | **44.3%** | Phase 4b |
| SimSFT 平均 reward 提升 | **+0.146** | Phase 4b |
| SimSFT 训练时长 | ~4 分钟 | Phase 4b |
| SimSFT 最终 loss | train 0.036 / eval 0.034 | Phase 4b |

### 3.3 失败 vs 成功的明确边界

- ✅ **格式学习完全成功**：模型能稳定输出合法 JSON 动作；
- ❌ **策略学习失败**：2 人局 33% 胜率（与 WeightedRandom 接近），GRPO 无法改进策略；
- ⚠️ **评估可信度低**：MINI 地图 + 6VP 下，连 VictoryPointPlayer 都很难赢（100 轮后 VP 仅 2-3），说明**评测环境本身有问题**，不是单纯模型弱。

---

## 4. 项目结构

```
catan-rl-llm/
├── README.md                       # 本文档
├── PROGRESS.md                     # 详细进展报告（Phase 1–4b）
├── log.md                          # 实验过程日志（gitignored，不上传）
├── pyproject.toml                  # 项目配置
├── requirements.txt                # Python 依赖
├── .env / .gitignore
│
├── checkpoints/                    # 模型权重（gitignored）
│   ├── sft/                        # Phase 3 SFT LoRA（174 MB）
│   ├── ab_sft/                     # AlphaBeta 蒸馏 SFT
│   ├── simsft/                     # Phase 4b SimSFT 精调
│   └── aesl/                       # Phase 5（待生成）
│
├── configs/                        # YAML 配置（数据/训练/评估）
├── data/                           # 生成的数据集（gitignored）
│   ├── sft/                        # 20,558 条 SFT 记录
│   ├── grpo/iter1_train/           # 1,000 条 GRPO rollout
│   └── simsft/iter1/               # 212 条 refined
│
├── experiments/                    # 各 Phase 的详细文档
│   ├── 01_phase1_setup.md
│   ├── 02_phase2_agent.md
│   ├── 03_phase3_sft.md
│   └── 04_phase4_rl.md
│
├── notebooks/                      # Jupyter 实验记录
├── results/                        # 评估输出（JSON / 图表）
│
├── scripts/                        # CLI 入口脚本
│   ├── 数据生成
│   │   ├── generate_sft_data.py        # 100 局专家对局 → SFT 数据
│   │   ├── generate_grpo_data.py       # rollout 收集
│   │   ├── generate_vn_sft_data.py     # 价值网络蒸馏 SFT
│   │   ├── generate_ab_sft_data.py     # AlphaBeta 蒸馏 SFT
│   │   └── generate_vf_distill_data_v2.py
│   ├── 训练
│   │   ├── train_sft.py                # 标准 SFT
│   │   ├── train_sft_best.py           # 当前最优 SFT 流程
│   │   ├── train_grpo.py               # GRPO（Phase 4）
│   │   ├── train_aesl.py               # Phase 5 AESL（待验证）
│   │   ├── train_simsft.py             # Phase 4b SimSFT
│   │   ├── train_value_network.py      # 价值网络训练
│   │   ├── train_vf_distill.py         # 价值网络蒸馏 v1
│   │   └── train_vf_distill_v2.py      # 价值网络蒸馏 v2
│   ├── 评估
│   │   ├── evaluate.py                 # 通用评估器
│   │   ├── eval_grpo.py                # GRPO 模型评估
│   │   ├── eval_grpo_sft.py            # GRPO+SFT 联合
│   │   ├── eval_simsft.py              # SimSFT 模型评估
│   │   ├── eval_ab_sft.py              # AlphaBeta SFT 评估
│   │   ├── eval_aesl_checkpoints.py    # AESL checkpoint 评估（Phase 5）
│   │   ├── eval_qwen_agent.py          # 纯 Qwen3-8B zero-shot
│   │   ├── eval_agent_hybrid.py        # Hybrid Agent（多工具）
│   │   ├── eval_hybrid_ablation.py     # Hybrid 消融
│   │   ├── eval_vf_guard.py            # 价值网络 guardrail
│   │   └── eval_rl_guard.py            # RL guardrail
│   ├── 工具
│   │   ├── download_model.py           # 下载 Qwen3-8B
│   │   ├── prepare_grpo_data.py        # 准备 GRPO 训练集
│   │   ├── rollout.py                  # 游戏回放采集
│   │   ├── simsft.py                   # SimSFT 数据生成核心
│   │   ├── run_final_eval.py           # 最终评估入口
│   │   ├── setup_env.sh                # 环境初始化
│   │   └── test_imports.py             # 依赖冒烟测试
│
└── src/catan_rl/                   # 核心代码库
    ├── __init__.py
    ├── agent/                      # Agent 实现（LlamaGym 模式）
    │   ├── base.py                 #   抽象 CatanAgent
    │   ├── qwen_agent.py           #   Qwen3-8B concrete
    │   ├── observation.py          #   7 段结构化观察
    │   ├── action_parser.py        #   5 级动作解析回退
    │   └── prompts.py              #   3 套系统提示词
    ├── env/                        # Catanatron 包装
    │   ├── catan_env.py            #   环境工厂
    │   ├── game_state.py           #   状态序列化
    │   ├── simulator.py            #   并行游戏模拟
    │   └── reward.py               #   奖励函数
    ├── data/                       # 数据管线
    │   ├── rollout.py              #   游戏回放采集
    │   ├── sft_dataset.py          #   SFT 数据集
    │   ├── grpo_dataset.py         #   GRPO 数据集
    │   └── preprocessing.py        #   Chat template 应用
    ├── training/                   # 训练编排
    │   ├── train_sft.py
    │   ├── train_grpo.py
    │   └── utils.py                #   模型加载、LoRA 配置
    ├── eval/                       # 评估
    │   ├── arena.py                #   锦标赛 runner
    │   ├── metrics.py              #   胜率、ELO、统计
    │   └── visualize.py            #   Matplotlib 图表
    └── rl/                         # RL 辅助
        ├── value.py                #   价值函数
        ├── value_network.py        #   神经网络版价值函数
        ├── features.py             #   状态特征提取
        ├── tree_search_utils.py    #   minimax 树搜索
        └── minimax.py              #   alpha-beta 搜索
```

---

## 5. Phase 详解：五阶段进展

### Phase 1 — 环境搭建 ✅

详见 [`experiments/01_phase1_setup.md`](experiments/01_phase1_setup.md)。

- 18/18 环境检查通过：`torch 2.1.2`、`transformers 5.14.1`、`trl 1.9.2`、`catanatron 3.2.1`
- CUDA 12.1 / Driver 13.2 / GPU 23.5 GB
- `flash_attention_2` 不可用 → 改用 `sdpa`
- `wandb` 未登录 → `report_to=none`

### Phase 2 — Agent 实现 ✅

详见 [`experiments/02_phase2_agent.md`](experiments/02_phase2_agent.md)。

5 个核心模块：

- `agent/base.py` — 抽象 `CatanAgent`（继承 LlamaGym 范式）
- `agent/prompts.py` — 3 套 system prompt（详细版 / 标准版 / 精简版）
- `agent/observation.py` — 7 段结构化观察（己方 / 地图 / 资源 / 建筑 / 道路 / 交易 / 对手）
- `agent/action_parser.py` — 5 级解析回退（正则 → JSON → 索引匹配 → 模糊 → 默认）
- `agent/qwen_agent.py` — Qwen3-8B 落地实现

**4 个关键 API 适配**（踩坑记录）：

1. `state.current_color` 是**方法**不是属性 → 必须 `state.current_color()`
2. `Game` **没有** `.players` 属性 → 用 `state.current_color()` 判断
3. P0（BLUE）必须传 `Color.BLUE` **枚举**而不是字符串
4. `env.get_valid_actions()` 返回**动作空间索引**（0–289），模型输出的是**顺序索引** → 需要做映射

### Phase 3 — SFT 冷启动 ✅

详见 [`experiments/03_phase3_sft.md`](experiments/03_phase3_sft.md)。

| 维度 | 值 |
|---|---|
| 数据 | 100 局 VictoryPointPlayer vs WeightedRandom，MINI 地图 6 VP |
| 记录数 | 20,558（train 18,502 / val 2,056） |
| 训练时长 | 6,307 秒 ≈ 1h 45m |
| Epochs / Steps | 3 / 564 |
| 最终 loss | train 0.0887 / eval 0.02186 |
| Token 准确率 | 99.08% |
| **动作合法性** | **100%**（1018/1018） |
| 胜率（vs WeightedRandom，2 人局） | 2/6 = 33.3% |

**9 个关键 bug 修复**（详见 `experiments/03_phase3_sft.md`），包括 `state.current_color()` 方法化、`Color.BLUE` 枚举、TRL v1.9.2 API 迁移、FlashAttention2 → sdpa 等。

**关键结论**：

> 0% 胜率但 100% 合法性 → **模型学会了"格式"，没有学会"策略"**。
> 2 人局 MINI 6VP 资源太少，游戏常常 100 轮后 VP = 2-3，**评估环境本身有问题**。

### Phase 4 — GRPO 强化学习 ❌（结构性失败）

详见 [`experiments/04_phase4_rl.md`](experiments/04_phase4_rl.md)。

GRPO 基础设施全部打通（reward 函数、模拟器、rollout、dataset），但完整训练暴露**结构性问题**：

| 指标 | 观测值 | 含义 |
|---|---|---|
| Entropy | **0.01 – 0.06** | 模型极度确定性，K=4 生成几乎相同 |
| Advantage | 趋近 0 | 组内样本雷同 → 无学习信号 |
| Reward | -0.94 ~ -0.44 | 被无效动作 / 低分拖累 |
| KL 散度 | 1e-9 ~ 1e-4 | 策略几乎不变 |

**根因**：SFT 把模型训到熵 0.01–0.06，每组 K=4 次生成几乎完全相同 → advantage ≈ 0 → GRPO 无梯度可学。**这是调参无法修复的结构性问题**。

**5 个 GRPO 关键 bug 修复**：

1. TRL `GRPOTrainer` 的 reward 函数签名是 `List[str]`，不是 `List[List[Dict]]`
2. Qwen3 thinking mode → 必须 `enable_thinking=False`
3. Prompt 必须含完整 chat template
4. TRL v1.9.2 无 `max_prompt_length` 参数
5. `per_device_train_batch_size` 必须被 `num_generations` 整除

### Phase 4b — SimSFT（模拟引导 SFT）🔄

详见 `PROGRESS.md` 第 3 节。

GRPO 失败后，退而求其次：**绕开"模型生成多样性"依赖**。

**思路**：

1. 对每个游戏状态**枚举所有合法动作**（而非模型生成）
2. 用 VictoryPointPlayer 模拟每个动作 5 次 → 选最优
3. 以**模拟最优动作**为新 SFT 目标
4. 在已有 SFT checkpoint 上继续精调

| 指标 | 值 |
|---|---|
| 输入记录 | 1,000（来自 GRPO iter1 的 rollout） |
| 可优化状态 | 212 条（排除只有 1 个合法动作的状态） |
| 找到更好动作的比例 | **44.3%** |
| 平均 reward 提升 | **+0.146** |
| 训练时长 | ~4 分钟 |
| 最终 loss | train 0.036 / eval 0.034 |

**评估遇阻**：

- 2-player 评估：两个模型都是 0% 胜率、100% 合法性
- 3-player 评估：被输出缓冲问题阻塞
- **根因**：MINI 6VP 下连专家都难赢，骰子运气对结果影响 > 动作选择
- **SimSFT 仅 212 条数据，量太小不足以改变模型行为**

---

## 6. 关键发现与教训

> **核心结论**：SFT 冷启动把模型"训得太死"是 RL 失败的根源。
> 这一现象与 ICLR 2026 AESL 论文描述的 **distribution forgetting / diversity forgetting** 高度吻合。

### 6.1 五大主要发现

| # | 发现 | 证据 |
|---|---|---|
| 1 | **SFT 后熵塌陷摧毁 RL** | entropy 0.01–0.06 → advantage ≈ 0 |
| 2 | **Catan 奖励极其稀疏且随机** | 100 轮后 VP 仅 2-3，骰子主导 |
| 3 | **MINI 6VP 评估失真** | 连 VictoryPointPlayer 都难赢 |
| 4 | **格式与策略是两件事** | 100% 合法性 vs 33% 胜率 |
| 5 | **SimSFT 数据生成有效** | 44.3% 状态找到更优动作（+0.146） |

### 6.2 五大工程教训

| # | 教训 | 适用 |
|---|---|---|
| 1 | TRL reward 函数签名与官方文档不一致 | GRPO 接入 |
| 2 | Qwen3 必须显式禁用 thinking mode | Qwen 系列 |
| 3 | `state.current_color` 是方法不是属性 | Catanatron |
| 4 | `Game` 没有 `.players` 属性 | Catanatron |
| 5 | 输出缓冲问题影响所有 LLM 评估 | 所有 eval 脚本 |

### 6.3 与 Catanatron-main 子项目的对照

整个工作区还有 [`Catanatron-main`](../Catanatron-main/) 子项目走**另一条路线**：

| 维度 | catan-rl-llm（这里） | Catanatron-main（兄弟项目） |
|---|---|---|
| 路线 | LLM 端到端微调 | RL value network + Agentic tool-use |
| 最终成果 | 100% 动作合法，但策略弱 | Hybrid Agent **100% WR**（6 局） |
| 主要失败 | GRPO 熵塌陷 | Win/loss 标签噪声 |
| 主要成功 | 完整训练管线 | 价值网络作 guardrail |

**两个项目的共同结论**：

- Catan 4 人局奖励信号极其稀疏（无论端到端 LLM 还是 value network）
- 端到端训练必须配合**多样性保护**（AESL）或**显式子模块**（Hybrid Agent）
- 评估指标单一胜率不够，需要 VP 差距、合法性、决策方差等组合

详见 [`PROJECT_SUMMARY.md`](../PROJECT_SUMMARY.md) 第 4 节"跨项目关键教训"。

---

## 7. 下一步路线（AESL + GRPO 重启）

> 论文：**Getting Your LLMs Ready for Reinforcement Learning with Lightweight SFT**（AESL, ICLR 2026）
> 核心洞察：**评估分数 ≠ RL 后性能**；冷启动 SFT 应关注输出**多样性**而非评估分数。

### 7.1 AESL 与本项目问题的对应

| AESL 论文发现 | 本项目现状 | 启示 |
|---|---|---|
| 冷启动 SFT 过度训练 → **distribution forgetting** | SFT 训满 3 epochs，entropy 掉到 0.01–0.06 | 冷启动**不应**训到评估分数最高点 |
| **多样性指标**（entropy / self-BLEU）比评估分数更能预测 RL 潜力 | entropy 塌陷 → GRPO advantage=0 | 应在**多样性峰值**处停止冷启动 |
| AESL 自适应加权损失保护 base 分布 | 模型失去探索能力 → RL 无法学习 | 用 **AESL 损失**取代标准 CE 损失 |
| 轻量级冷启动（1k–6k 数据）足够 | 当前用了 18.5k 全量数据 | 可能**过拟合了数据集分布** |

### 7.2 三阶段路线

```
Phase 5（下一步）：AESL 冷启动 SFT
    ├── 实现 aesl_loss.py（自适应加权 + entropy 监控）
    ├── 轻量数据（3k）+ t_scaling∈[3,5] 训练
    ├── 在多样性峰值处早停（entropy > 0.3）
    └── 验证：GRPO 的 K=4 组内样本有差异

Phase 6：GRPO 重试
    ├── 用 AESL 冷启动 checkpoint 替代原 SFT checkpoint
    ├── 修复评估脚本输出缓冲（python -u + 显式 flush）
    └── 用 BASE 地图 + 10VP 评估（胜率 > 0% 即算成功）

Phase 7：SimSFT 规模化
    ├── 基于 AESL 模型重新生成 rollout（数千条）
    ├── 只保留 improvement > 0 的记录做加权训练
    └── 与 GRPO 结果对比
```

### 7.3 AESL 损失函数（计划）

```
L = -Σ_t  w_t · log π_θ(s*_t | q, s_<t)
w_t = 1 - sigmoid( logit_t / t_scaling · prefix_avg_log_prob_t )
t_scaling ∈ [3, 5]   # 论文最优区间
```

### 7.4 风险与对策

| 风险 | 对策 |
|---|---|
| AESL 损失实现细节与论文有出入 | 严格按论文公式，t_scaling 扫 {3,5,7} |
| Catan reward 信号过于随机 | 增加 rollouts + 用 VP 差距做 dense reward |
| MINI 地图评估失真 | 评估迁移到 BASE 地图 + 10VP |
| 评估缓冲问题反复阻塞 | 全部 eval 脚本加 `python -u` + 显式 `flush=True` |
| TRL API drift | 锁版本 `trl==1.9.2` |

---

## 8. 快速上手

> ⚠️ **评估说明**：当前 Phase 4b 的评估存在输出缓冲问题，建议直接看 `experiments/*.md` 中已记录的指标，而不是重跑 2-player 评估。

### 8.1 环境准备

```bash
# Python ≥ 3.10
python -m venv .venv && source .venv/bin/activate    # Linux/Mac
python -m venv .venv && .venv\Scripts\activate       # Windows

pip install -r requirements.txt
# 或
pip install -e .
```

下载 Qwen3-8B（首次需要）：

```bash
python scripts/download_model.py
```

冒烟测试：

```bash
python scripts/test_imports.py
```

### 8.2 数据生成

```bash
# Phase 3 SFT 数据（100 局专家对局 → 20k 记录）
python scripts/generate_sft_data.py \
    --num-games 100 \
    --map MINI \
    --vp 6 \
    --output data/sft/

# Phase 4 GRPO rollout（20 局 bot vs bot → 1k 记录）
python scripts/generate_grpo_data.py \
    --num-games 20 \
    --output data/grpo/iter1_train/

# Phase 4b SimSFT 数据（212 条 refined）
python scripts/simsft.py \
    --input data/grpo/iter1_train/ \
    --output data/simsft/iter1/
```

### 8.3 训练

```bash
# Phase 3 SFT（已跑过，~1h45m）
python scripts/train_sft.py --config configs/sft.yaml

# Phase 4b SimSFT（已跑过，~4 分钟）
python scripts/train_simsft.py --config configs/simsft.yaml

# Phase 5 AESL（待实施）
python scripts/train_aesl.py --config configs/aesl.yaml
```

### 8.4 评估

```bash
# 推荐用 python -u + 显式 flush 解决输出缓冲
python -u scripts/evaluate.py \
    --checkpoint checkpoints/sft/ \
    --opponent WeightedRandomPlayer \
    --num-games 6 \
    --map MINI \
    --vp 6 2>&1 | tee results/sft_eval.log
```

### 8.5 GRPO（Phase 5 之后才推荐重跑）

```bash
# 准备 GRPO 数据
python scripts/prepare_grpo_data.py \
    --checkpoint checkpoints/aesl/ \
    --output data/grpo/iter2/

# 跑 GRPO
python scripts/train_grpo.py --config configs/grpo_aesl.yaml
```

---

## 9. 配置与依赖

### 9.1 核心依赖（`pyproject.toml`）

| 包 | 版本 | 用途 |
|---|---|---|
| `torch` | ≥ 2.1.0 | 训练 |
| `transformers` | ≥ 4.45.0 | Qwen3-8B 模型 |
| `trl` | ≥ 0.12.0（实测 1.9.2） | GRPOTrainer |
| `peft` | ≥ 0.12.0 | QLoRA |
| `bitsandbytes` | ≥ 0.43.0 | 4-bit 量化 |
| `accelerate` | ≥ 0.28.0 | 分布式 |
| `catanatron-gym` | ≥ 4.0.0 | 游戏引擎 |
| `gymnasium` | ≥ 0.29.0 | RL 接口 |
| `vllm` | ≥ 0.5.0 | 快速推理（可选） |
| `wandb` | ≥ 0.16.0 | 实验追踪（沙盒中禁用） |
| `datasets` | ≥ 3.0.0 | 数据加载 |
| `pyyaml` | ≥ 6.0 | 配置 |
| `matplotlib` / `seaborn` | — | 可视化 |

### 9.2 硬件要求

| 模式 | 显存 | 推荐 |
|---|---|---|
| SFT 训练（4-bit QLoRA） | ≥ 20 GB | RTX 4090 / A5000 |
| GRPO 训练（K=4） | ≥ 24 GB | RTX 4090 D / A6000 |
| 推理（Qwen3-8B 4-bit） | ~10 GB | RTX 3080+ |

---

## 10. 评估方法与可信度

### 10.1 当前评估的局限性

| 问题 | 影响 | 缓解 |
|---|---|---|
| **MINI 6VP 资源过少** | 100 轮后 VP 仅 2-3，无法区分策略 | 改用 BASE 地图 + 10VP |
| **单局方差极大** | 骰子运气主导 50% 胜率波动 | 每配置 ≥ 10 局，最好 30+ 局 |
| **2-player 配置** | 没有交易维度 | 优先 4-player |
| **评估输出缓冲** | 日志丢失，难以排查 | `python -u` + `flush=True` |
| **奖励 1/-1/0 太稀疏** | 单一指标难以反映策略改进 | 加 VP 差距、合法性、决策熵 |

### 10.2 推荐的评估协议（Phase 5/6 之后）

```yaml
评估协议:
  地图: BASE
  VP: 10
  玩家数: 4
  对手类型:
    - WeightedRandomPlayer (baseline)
    - VictoryPointPlayer (medium)
    - AlphaBetaPlayer (strong)
  每配置局数: ≥ 30
  指标:
    - 胜率
    - 平均 VP 差距（vs 对手平均）
    - 动作合法性
    - 决策熵（多样性）
    - 交易成功率（4 人局专属）
```

---

## 11. 实验文档索引

每个 Phase 都有独立的详细记录：

| Phase | 文档 | 主要内容 |
|---|---|---|
| 1 | [`experiments/01_phase1_setup.md`](experiments/01_phase1_setup.md) | 环境搭建、版本兼容性 |
| 2 | [`experiments/02_phase2_agent.md`](experiments/02_phase2_agent.md) | Agent 设计、API 适配 |
| 3 | [`experiments/03_phase3_sft.md`](experiments/03_phase3_sft.md) | SFT 训练、9 个 bug 修复 |
| 4 | [`experiments/04_phase4_rl.md`](experiments/04_phase4_rl.md) | GRPO 失败根因、5 个修复 |

更高级别进展：

| 文档 | 内容 |
|---|---|
| [`PROGRESS.md`](PROGRESS.md) | 项目进展报告（Phase 1–4b） |
| 顶层 [`PROJECT_SUMMARY.md`](../PROJECT_SUMMARY.md) | 整个工作区文档（两个子项目并行） |

---

## 12. 引用资源

### 论文

- **AESL**: *Getting Your LLMs Ready for Reinforcement Learning with Lightweight SFT* (ICLR 2026) — 冷启动 SFT 的多样性保护
- **GRPO**: *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models* — Group Relative Policy Optimization
- **LlamaGym**: *LlamaGym: A Framework for Abbreviated Reinforcement Learning Research* — Agent 抽象模式

### 项目

- [Catanatron](https://github.com/bcollazo/catanatron) — 卡坦岛 Python 引擎
- [LlamaGym](https://github.com/KhoomeiK/LlamaGym) — LLM + Gym RL 框架
- [TRL](https://github.com/huggingface/trl) — Transformer RL
- [Qwen3](https://huggingface.co/Qwen/Qwen3-8B-Instruct) — 基础模型

### 兄弟项目

- [`Catanatron-main`](../Catanatron-main/) — RL value network + Hybrid Agent 路线
- 顶层 [`PROJECT_SUMMARY.md`](../PROJECT_SUMMARY.md) — 合并文档

---

## 维护说明

- **生成时间**：2026-08-10
- **数据来源**：[`PROJECT_SUMMARY.md`](../PROJECT_SUMMARY.md)、[`PROGRESS.md`](PROGRESS.md)、`experiments/01~04_phase*.md`、`pyproject.toml`
- **下次更新时机**：Phase 5 AESL 训练完成 / Phase 6 GRPO 重启 / Phase 7 SimSFT 规模化
- **写作原则**：完整梳理当前状态、明确标注成败、给出可执行的下一步

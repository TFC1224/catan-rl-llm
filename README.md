# catan-rl-llm

训练语言模型玩《卡坦岛》（Settlers of Catan）的实验仓库，使用 catanatron 引擎与 Qwen3-8B 基座，尝试 SFT、蒸馏、RL、工具调用、guardrail 等方法在多人博弈中的效果。

更详细的实验档案见 [`PROJECT_SUMMARY.md`](./PROJECT_SUMMARY.md)。

---

## 当前进度

工作自 2026-08-06 开始，到 2026-08-10 共四天。期间共实施 10 种方法、3 次模型修复、1 次系统消融。按强度排序的主要结论：

- Hybrid Agent（工具 + 价值函数 guardrail）在一组 6 局评估中取得对 WeightedRandom 的 6/6 胜率，局长约 90 回合。这一结果未在更大样本上复现。
- VF-Guard（LLM + 手写价值函数，无工具）在一组 10 局评估中取得 9/10 胜率，作为本项目的「实用上限」基准。
- 纯模仿（SFT）训练收敛（loss 0.04、token 准确率 99%）但胜率与随机基线无差异（25%）。
- 任何依赖「游戏最终胜负」作为训练信号的 RL 方法（GRPO、outcome label 课程）胜率均低于或接近随机基线。
- 30 维特征的轻量 RL 模型用作动作打分时表现不稳定（0% – 67%），替换为 72 维特征 + 价值函数残差训练后稳定到 44%（vs WeightedRandom）。

这些数字的样本量都偏小（3–20 局/配置），区分方法优劣时需保留显著性的怀疑。30 局以上的复现评估列入下一步。

---

## 研究问题

下面六个问题驱动了项目的主要工作。每一节给出动机、做法与结果。

### RQ1：纯模仿学习能否让 LLM 学会玩卡坦？

**动机**：卡坦是部分可观测的多人博弈，决策依赖对手建模与长程规划。监督微调在很多领域有效，但在博弈中可能只学到「合法动作」而学不到「好动作」。

**方法**：用 catanatron 内置的 VictoryPointPlayer（最强内置 bot）生成 300 局共 18502 条决策，训练 Qwen3-8B 的 LoRA 适配器（r=16，α=32），3 个 epoch。

**结果**：训练收敛（最终 loss 0.089，token 准确率 99%）。评估 20 局，胜率 5/20 = 25%，与 WeightedRandom 在 4 人局中的随机基线齐平。模型合法动作率 100%，但战略层面无可见提升。

**结论**：纯模仿在卡坦上仅学到「格式正确」，不能学到「策略有效」。这是 RQ2 开始的动机。

### RQ2：是否需要强化学习？

**动机**：SFT 失败的常见解释是模仿只复制表面、不内化决策逻辑。GRPO 是当前 LLM RL 的主流方法之一。

**方法**：基于 SFT LoRA 启动 GRPO 训练，使用游戏终局胜负作为奖励，4 人局对战 WeightedRandom。

**结果**：训练过程出现两类持续性问题。一是 KL 散度几乎为零、奖励在 −1 到 −0.4 间震荡，模型未发生有效策略更新；二是在更高温度（1.0 以上）下模型输出中文幻觉文本（已通过温度上限与超长输出惩罚缓解）。即使在基础设施正常的情况下，单纯把胜负作为奖励信号在本任务上未见明显改善。

**结论**：用胜负作为 RL 奖励在本任务上不奏效，这与 RQ5 中的 outcome label 失败是同一现象的不同侧面。

### RQ3：手写价值函数能否替代 RL？

**动机**：卡坦已有一个成熟的价值函数（catanatron 仓库中的 `contender_fn`，13 个特征的线性加权）。它能在单人前瞻搜索下达到接近上限的表现。如果它本身已经够好，RL 的目标应该是「让 LLM 学会它」，而不是「绕开它」。

**方法**：VF-Guard = LLM 输出动作类型 + `contender_fn` 对所有合法动作打分，挑选最高分。LLM 的提议在约半数非平凡决策中被覆盖，绝大多数是「同类型内的具体位置」（哪条边、哪个节点）。

**结果**：4 人局对 WeightedRandom，10 局中胜 9 局。这一表现与纯 `contender_fn` 在 3 人局上的 90% 胜率相当，意味着组合 LLM 的战略意图与价值函数的战术打分已经接近天花板。

**意义**：VF-Guard 给出了本项目的实用上限——任何纯 LLM 方案若想突破它，需要的不是更聪明的算法，而是文本观察无法承载的空间信息。这把研究问题从「如何训练 LLM」转向「如何把价值函数桥接到 LLM 上」。

### RQ4：LLM 能否学会价值函数的偏好？

**动机**：VF-Guard 每步都要做一次价值函数打分（约 50% 的非平凡决策被覆盖）。如果能把这些覆盖模式蒸馏进 LLM，推理时就不必每次都调用价值函数。

**方法**：从 VF-Guard 100 局游戏中收集 1022 条决策，其中 439 条是 VF 覆盖 LLM 的情况。从 SFT LoRA 续训，学习率 1e-4，2 个 epoch。

**结果**：训练收敛（loss 0.073，准确率 97.6%）。评估 20 局，胜率 8/20 = 40%，比 SFT（25%）高，比 VF-Guard（90%）低 50 个百分点。

**分析**：20 局的差距集中于「同类型内的位置选择」。文本观察把候选位置编码成「可建节点列表第 N 项」，无法承载几何相邻关系，LLM 在这种细粒度区分上始终输给显式计算。RQ5 转向另一条路：让 LLM 直接访问工具而非内化工具的偏好。

### RQ5：工具调用是否能补足文本观察的盲区？

**动机**：SFT 学到战略、价值函数补足战术——但 SFT 是与价值函数分离的两段式拼接。如果让 LLM 在决策前看到价值函数的输出（以工具结果形式），战略和战术是否能在 LLM 内部被更自然地结合？

**方法**：Hybrid Agent 在 LLM 决策之前调用四个工具（`analyze_position` 评估胜率与产能、`check_threats` 评估对手威胁、`get_best_move` 给定目标下的候选动作、`simulate_outcome` 单步推演），将结果附加到观察文本。LLM 输出动作类型，最后仍由价值函数做最终 guardrail（即覆盖 LLM 决策）。

**结果**：消融三组：

| 配置 | 胜率 | 样本 |
|---|---|---|
| 工具 + 价值函数 guard | 6/6 | 6 局 |
| 工具 + RL 模型 guard | 0/3 | 3 局 |
| 仅工具，无 guard | 2/3 | 3 局 |

工具本身已把胜率从 25% 提升到约 67%，价值函数 guardrail 进一步把 67% 推到 100%。RL 模型作为 guard 时胜率反而降到 0%，原因是 RL 模型训练目标是「预测游戏终局胜负」而非「评估动作质量」，30 维特征下 47% 的决策输出完全相同的分数（详见 RQ6）。

**结论**：在卡坦这种文本难以编码空间信息的任务上，把外部计算以工具结果形式注入 LLM 比让 LLM 内化价值函数更直接。RL guard 的失败提示：不是所有能打分的模型都适合作为战术裁判。

### RQ6：RL 模型本身的失败是特征不够还是训练目标不对？

**动机**：30 维特征的 RL 模型在 Hybrid Agent 中不仅无用而且有害。一个候选解释是特征不足（同类动作的特征向量完全相同），另一个是训练目标（预测胜负）与使用方式（评分动作）不匹配。

**方法**：两件事并行做。

一是把特征从 30 维扩到 72 维，加入 per-resource 产能、对手明细、港口访问、建造能力标志等「会随具体动作变化」的特征。

二是把训练目标从「预测胜负（sigmoid + BCE）」改为「预测价值函数残差」（线性输出 + MSE，标签 = `(VF − VP·3e14) / 1e8`，范围约 [−1, 2]）。价值函数残差关注「同一胜局水平下，质量差异在哪里」。

**结果**：在 AlphaBeta 数据上训练 500 步后，对 Random 胜率 69%（原 25%），对 WeightedRandom 胜率 44%（原 0% – 25%），同类动作间的分数方差（action spread）从约 0.018 升到 0.064，flat 决策比例从 47% 降到 3.1%。

**结论**：在本任务上，特征工程的收益（15 倍 flat 决策下降）远大于算法层面的改动。但即使是修复后的模型，在「同类型内位置选择」上仍不能区分（比如所有修路动作的特征几乎一致），最终仍需价值函数 guardrail 兜底。

---

## 讨论

下面五条不是某一次实验的结果，而是多次失败后的反思。

**1. 4 人博弈的胜负是噪声标签。** 4 人局中 75% 的状态来自输家。输家可能打得很差也可能打得很好但牌运差。把胜负当 state quality 训练，会让模型学到「平均胜率 25% + 高置信度」，对动作选择没有帮助。这一点在 Option C 课程实验中表现最明显：训练指标漂亮（相关系数 0.83、flat 决策 0%），实际胜率 14%。

**2. 价值函数残差是好训练信号，胜负不是。** 残差范围 [−1, 2]、线性输出、干净梯度，是「同胜局水平下状态质量的连续度量」。它直接来自现在这个状态，不依赖未来 100 步的随机性。

**3. 价值函数是好的裁判、坏的教师。** 作为推理时的最终裁判，VF-Guard 和 Hybrid Agent 都能稳定接近上限。作为训练数据的来源（直接蒸馏 VF 的覆盖决策、或把 VF 评分当作标签），所有方案都在 0% – 40%。原因是 VF 在不同状态下的「最优动作」之间没有一致的策略语境。

**4. 给模型看得见的特征比给它更聪明的算法更重要。** 30 维特征升到 72 维，flat 决策下降 15 倍。这比同时间内任何 RL 算法改进都大。

**5. 训练指标会骗人。** loss、token 准确率、价值函数相关系数、动作方差，这些指标都可能与对战胜率脱钩。唯一可信的指标是对战胜率本身，且需要在多局、对多对手的条件下取平均。

---

## 复现

环境：Python ≥ 3.10，PyTorch ≥ 2.1，NVIDIA RTX 4090 D 24GB（AutoDL 容器）。

```bash
pip install -r requirements.txt
```

主要依赖：`torch`、`transformers`、`trl`、`peft`、`catanatron-gym`、`gymnasium`、`vllm`。

数据生成：

```bash
# SFT 数据（VictoryPointPlayer × WeightedRandom 100 局，~18.5K 决策）
python scripts/generate_sft_data.py --num_games 100 --output data/sft/ --seed 42

# AB-SFT 数据（300 局，更大规模）
python scripts/generate_ab_sft_data.py --num_games 300 --output data/ab_sft/

# VF 蒸馏数据（VF-Guard 覆盖 LLM 的决策）
python scripts/generate_vf_distill_data_v2.py
```

训练：

```bash
# AB-SFT
python scripts/train_sft_best.py --data data/ab_sft/ --output checkpoints/ab_sft/

# VF 蒸馏 v2
python scripts/train_vf_distill_v2.py

# AESL entropy 监控训练
python scripts/train_aesl.py

# 价值网络（72 维 VF 残差版本）
python scripts/train_value_network.py
```

评估：

```bash
# AB-SFT
python scripts/eval_ab_sft.py --model checkpoints/ab_sft/checkpoint-200/

# VF-Guard（实用上限基准）
python scripts/eval_vf_guard.py --games 10

# RL-Guard
python scripts/eval_rl_guard.py --games 10

# Hybrid Agent（消融与最终评估）
python scripts/eval_hybrid_ablation.py
python scripts/eval_agent_hybrid.py --games 6

# 统一最终评估
python scripts/run_final_eval.py
```

---

## 仓库结构

```
catan-rl-llm/
├── README.md              # 本文件
├── PROJECT_SUMMARY.md     # 实验档案：背景、时间线、失败路径、组件细节、评估协议与下一步
├── pyproject.toml
├── requirements.txt
├── configs/               # YAML 配置：default / sft / grpo / eval
├── src/catan_rl/          # 代码包
│   ├── agent/             # LlamaGym Agent 基类、Qwen 实现、观察格式化、动作解析
│   ├── env/               # Catanatron 适配、状态序列化、奖励、模拟器
│   ├── rl/                # 特征工程、值网络、minimax
│   ├── data/              # 数据集定义与 rollout
│   ├── training/          # 训练入口
│   └── eval/              # 对战 arena、指标、可视化
├── scripts/               # 30+ 命令行入口
├── data/                  # 训练数据（11 个子目录，按方法分）
├── checkpoints/           # 模型权重（14 个目录）
├── experiments/           # 阶段搭建文档
├── notebooks/             # 实验结果（按日期与方法编号）
└── results/               # 最终评估数据与可视化
```

---

## 文档索引

### 实验结果（按时间排序）

- [`notebooks/00-original-log-2026-08-06.md`](./notebooks/00-original-log-2026-08-06.md)：原始实验日志，含环境搭建、SFT、GRPO 试训全过程
- [`notebooks/ab-sft-results.md`](./notebooks/ab-sft-results.md)：RQ1，纯模仿 25% 胜率
- [`notebooks/vf-guard-discovery.md`](./notebooks/vf-guard-discovery.md)：RQ3，VF-Guard 9/10
- [`notebooks/option-a-v2-results.md`](./notebooks/option-a-v2-results.md)：RQ4，VF 蒸馏 40% 胜率
- [`notebooks/rl-guard-results.md`](./notebooks/rl-guard-results.md)：RL-Guard 不稳定
- [`notebooks/hybrid-agent-results.md`](./notebooks/hybrid-agent-results.md)：RQ5 初版 100% (3/3)
- [`notebooks/01-session-2026-08-08.md`](./notebooks/01-session-2026-08-08.md)：8 月 8 日全天 session 记录
- [`notebooks/grpo-results.md`](./notebooks/grpo-results.md)：RQ2，GRPO/VF rollout 失败
- [`notebooks/aesl-experiment-results.md`](./notebooks/aesl-experiment-results.md)：entropy 早停假说被拒
- [`notebooks/rl-model-fixed.md`](./notebooks/rl-model-fixed.md)：RQ6，72 特征 + VF 残差
- [`notebooks/option-c-curriculum-results.md`](./notebooks/option-c-curriculum-results.md)：outcome label 课程 12–38%
- [`notebooks/final-results-2026-08-08.md`](./notebooks/final-results-2026-08-08.md)：Hybrid Agent 消融与最终评估
- [`notebooks/MEMORY.md`](./notebooks/MEMORY.md)：实验结果导航

### 阶段搭建文档

- [`experiments/01_phase1_setup.md`](./experiments/01_phase1_setup.md)：环境与依赖
- [`experiments/02_phase2_agent.md`](./experiments/02_phase2_agent.md)：Agent 实现
- [`experiments/03_phase3_sft.md`](./experiments/03_phase3_sft.md)：第一次 SFT
- [`experiments/04_phase4_rl.md`](./experiments/04_phase4_rl.md)：GRPO 基础设施

### 综合档案

- [`PROJECT_SUMMARY.md`](./PROJECT_SUMMARY.md)：实验档案，包含背景、时间线、失败路径、组件细节、评估协议与下一步。

---

## 局限与下一步

样本量是最大的局限。Hybrid Agent 的 100% (6/6) 与 VF-Guard 的 9/10 都不能区分「方法真的有效」与「恰好赢了几局」。下一步优先级最高的事：30 局以上的复现评估、对 VictoryPointPlayer 的对比、以及把 100% 的声明建立在更大样本上。

其他待办：修复 72 维 RL 模型在修路动作上的盲区；把端到端 Qwen 推理替换掉 Ollama 中间层（已发现可节省约 50% 推理延迟）；修复 Option C 课程中已发现的公式 bug 并重跑 VF 残差版本。

更多细节见 [`PROJECT_SUMMARY.md`](./PROJECT_SUMMARY.md)。

---

*最后更新：2026-08-10*
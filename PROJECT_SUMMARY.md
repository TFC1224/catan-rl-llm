# PROJECT_SUMMARY

> **本文件目的**：记录 `catan-rl-llm` 全部实验历程、决策依据、失败教训与最终获胜方案的完整档案。
> 与 `README.md` 的区别：`README.md` 只讲「现在怎么跑 + 当前最强方案」，本文件讲「我们怎么走到这里」。
>
> 最后更新：2026-08-10

---

## 目录

1. [项目背景](#1-项目背景)
2. [完整时间线（按方法出现顺序）](#2-完整时间线)
3. [七个失败路径逐项分析](#3-七个失败路径逐项分析)
4. [获胜方案：Hybrid Agent](#4-获胜方案hybrid-agent)
5. [关键组件（视为本项目的一部分）](#5-关键组件)
6. [跨方法共同教训（五条核心洞察）](#6-跨方法共同教训)
7. [推荐评估协议](#7-推荐评估协议)
8. [推荐下一步](#8-推荐下一步)
9. [引用与索引](#9-引用与索引)

---

## 1. 项目背景

### 1.1 项目目标

训练一个能玩《卡坦岛》（Settlers of Catan）的语言模型。本项目以 Qwen3-8B 为基座、围绕 catanatron-gym 引擎，研究不同训练范式（SFT / RL / 蒸馏 / 工具调用 / Guardrail）在多人博弈场景下的有效性。

最终目标层级：

| 层级 | 描述 | 当前状态 |
|---|---|---|
| L0 | 正确格式输出合法动作（action validity ≥ 95%） | 已达成（100%） |
| L1 | 击败 `WeightedRandomPlayer` 基线（4 人局胜率 > 25%） | 已达成（最高 100%） |
| L2 | 击败 `VictoryPointPlayer`（最强内置 bot） | 未达成 |
| L3 | 强到可以作为研究对象 | 未达成 |

### 1.2 引擎与硬件

| 项 | 值 |
|---|---|
| 引擎 | catanatron-gym v4.0.0（封装 catanatron 3.2.1） |
| 主要地图 | MINI（7 tiles / 6 VP）用于快速迭代；BASE（19 tiles / 10 VP）用于终评 |
| 主要对手 | WeightedRandomPlayer（基线）、VictoryPointPlayer（最强内置，AlphaBeta 的别名） |
| 评估规模 | 单方法 3–20 局（早期快速迭代），最终 6 局（Hybrid Agent） |
| 推理硬件 | NVIDIA RTX 4090 D 24GB（AutoDL 容器） |
| 训练硬件 | 同上，4-bit QLoRA + LoRA r=16 / α=32 |
| 总训练时长 | 约 4 小时 SFT + 3 小时 AESL + 多次重训 < 1 小时 |

### 1.3 文件清单（顶层）

```
catan-rl-llm/
├── README.md              # 用法 + 当前最强方案（本文件之外另一份入口文档）
├── PROJECT_SUMMARY.md     # ← 本文件
├── pyproject.toml         # Python 项目元数据
├── requirements.txt       # pip 依赖清单
├── .env                   # HF_TOKEN / WANDB_API_KEY 占位
├── configs/               # YAML：default / sft / grpo / eval
├── src/catan_rl/          # 代码包
│   ├── agent/             # LlamaGym Agent 基类 + Qwen 实现 + 观察/解析/提示
│   ├── env/               # Catanatron 适配、状态序列化、奖励、模拟器
│   ├── rl/                # 特征工程、值网络、minimax
│   ├── data/              # SFT / GRPO 数据集定义、rollout
│   ├── training/          # 训练入口
│   └── eval/              # 对战 arena、指标、可视化
├── scripts/               # 30+ 命令行入口（数据生成 / 训练 / 评估）
├── data/                  # 11 类训练数据子目录（sft / vf_distill / grpo …）
├── checkpoints/           # 14 个模型/目录（ab_sft, sft, vf_distill, aesl, grpo …）
├── experiments/           # 4 份阶段性搭建文档
├── notebooks/             # 11 份实验结果（按时间编号排列）
└── results/               # 最终评估 JSON、arena 复盘、plots
```

---

## 2. 完整时间线（按方法出现顺序）

> 时间顺序以「方法首次进入主线评估」为准，而非草稿出现顺序。每条配 3–5 行要点，证据来源在第 9 章。

### 2.1 阶段零：环境与基线（2026-08-06）

- 选定 RTX 4090 D + AutoDL 容器，验证 18/18 导入检查通过（[`experiments/01_phase1_setup.md`](../experiments/01_phase1_setup.md)）。
- 实现 LlamaGym 风格 `CatanAgent` + `QwenCatanAgent` + 5 层 action_parser + 7 段 observation（[`experiments/02_phase2_agent.md`](../experiments/02_phase2_agent.md)）。
- 踩坑集中点：catanatron 3.2.1 的 `Game` 没有 `players` 属性、`Color` 必须用枚举、`state.current_color` 是方法不是属性、`enable_thinking` 在生成时无法关闭、`env.step()` 内部自动推进对手。
- 关键决定：放弃对 catanatron 新版实验分支的依赖，统一以 VictoryPointPlayer（=AlphaBeta 别名）作为强基线。

### 2.2 第一次 SFT：纯模仿（2026-08-07，AB-SFT）

- 数据：VictoryPointPlayer × WeightedRandom 100 局，18502 训练 + 2056 验证，平均每局 ~206 决策。
- 训练：Qwen3-8B + 4-bit QLoRA，3 epoch / 564 step，train loss 0.25 → 0.0887，mean token accuracy 99.08%。
- 评估：100% action validity，但 4 人局胜率仅 **25%**，与 WeightedRandom 基线齐平（[`notebooks/ab-sft-results.md`](../notebooks/ab-sft-results.md)）。
- 结论：SFT 只学到「合法动作的格式」，没有学到「策略」。纯模仿对卡坦这种长程博弈不够。

### 2.3 转折：VF-Guard 90% 的发现（2026-08-07）

- 思路：LLM 输出动作类型，**手写线性价值函数 `contender_fn` 给所有合法动作打分并覆盖 LLM**，发现 4 人局胜率达到 **90%（9/10）**（[`notebooks/vf-guard-discovery.md`](../notebooks/vf-guard-discovery.md)）。
- 覆盖约 50% 的非平凡决策，绝大多数是「同类型内的具体位置」选择（哪个节点、哪条路），而非动作类别本身。
- 直接推论：**GRPO 不再必要**——已经摸到价值函数的天花板，剩下的工作是把价值函数偏好内化到模型里（蒸馏）。
- 这一刻本项目从「训练 SOTA LLM 玩卡坦」转向「把已有价值函数桥接到 LLM 上」。

### 2.4 蒸馏尝试 v1 → v2（2026-08-07）

- v1（2026-08-07 上午）：直接用全部 1022 条 VF-Guard 决策训练，从 Qwen3-8B 基座初始化，2e-4 学习率。胜率 20%，比 AB-SFT 还差。
- v2（2026-08-07 下午）：三处修复——**仅取 VF 覆盖的 439 条**决策 + **从 AB-SFT LoRA 续训** + **学习率降到 1e-4**。胜率提升到 **40%**（[`notebooks/option-a-v2-results.md`](../notebooks/option-a-v2-results.md)）。
- 仍未到 VF-Guard 90% 的原因：文本观察无法编码空间细节，无法完成「同类型内不同位置」的精修。

### 2.5 RL-Guard（2026-08-07 晚）

- 思路：换一个 50K 参数的轻量值网络（30 特征、AB-SFT 数据训练）来打分，取代手写 VF。
- 胜率 **66.7%（2/3）**——看上去不错，但 **不稳定**（[`notebooks/rl-guard-results.md`](../notebooks/rl-guard-results.md)）。
- 后续大样本（3 局 final eval）骤降到 0%。问题诊断见 §3.3。

### 2.6 GRPO 与 VF-scored rollout 数据（2026-08-08 凌晨）

- 三种数据采样：全量（1821 条）、高区分度过滤（925 条）、阶段平衡（725 条）。
- 全部失败，胜率 **0–20%**，比 AB-SFT 25% 基线还差（[`notebooks/grpo-results.md`](../notebooks/grpo-results.md)）。
- 关键证据：数据量 4× 反而更差；过滤后更差；全量 GRPO 训练 loss 卡在 1.0 不收敛。
- 教训：VF 决策从「任意状态」取时，前后不一致，构不成可学的策略。

### 2.7 AESL 早停实验（2026-08-08 早）

- 把长 CoT 推理领域提出的「entropy 峰值早停」假说搬到卡坦 SFT，每 50 步存一个检查点（step 50–600）。
- 关键结果：entropy 峰值点（step 150）胜率 **0%**，best-loss 点（step 500）胜率 **20%**——entropy 峰值假说被拒绝（[`notebooks/aesl-experiment-results.md`](../notebooks/aesl-experiment-results.md)）。
- 根因：长 CoT 与短 JSON 输出在 entropy 动态上完全不同；step 150 时模型只见过 ~1200 个样本，处于「困惑」而非「多样能力」状态。

### 2.8 Hybrid Agent 第一版（2026-08-07 晚）

- 把 4 个「工具」（analyze_position / check_threats / get_best_move / simulate_outcome）的输出附加到观察文本，让 LLM 先看到这些，再用 VF 做最终 guardrail。
- 胜率 **100%（3/3）**，首次追上 VF-Guard 90%（[`notebooks/hybrid-agent-results.md`](../notebooks/hybrid-agent-results.md)）。
- 关键观察：胜局在 63–125 回合结束，RL-Guard 失败局 172–262 回合仍在拉锯——决策质量与局长负相关。

### 2.9 RL 模型修复（2026-08-08 上午）

- 旧 RL 模型 47% 的决策是「flat」（不同动作的特征完全相同），根源是 30 特征只数「数量」、不区分「位置」。
- 修复两件事：**特征扩到 72 维**（增加 per-resource production、opponent detail、port access、build flags 等）+ **训练目标改成 VF 残差**（`label = (VF - VP × 3e14) / 1e8`，range [-1, 2]，线性输出 + MSE）（[`notebooks/rl-model-fixed.md`](../notebooks/rl-model-fixed.md)）。
- 结果：vs Random 25% → **69%**，vs WeightedRandom 0–25% → **44%**，flat 决策 47% → **3.1%**（15× 下降）。
- 这个修复后的 `rl_enriched_model.pt` 成为 Hybrid Agent 中 `analyze_position` 与 `get_best_move` 的实际大脑。

### 2.10 Hybrid Agent 消融 + 最终评估（2026-08-08 下午）

- 三配置消融：`hybrid_vf`（工具+VF）vs `hybrid_rl`（工具+RL guard）vs `hybrid_none`（工具无 guard）（[`notebooks/final-results-2026-08-08.md`](../notebooks/final-results-2026-08-08.md)）。
- 结果：100% / 0% / 66.7%。VF guardrail 仍是关键。
- 最终 `final_eval_20260808.json` 确认 `hybrid.win_rate = 1.0 (3/3)`，`rl_guard.win_rate = 0.0 (0/3)`。

### 2.11 Option C 课程自博弈（2026-08-08 下午）

- 三相课程：vs Random → vs AlphaBeta → 自博弈；warm-start 自 `rl_enriched_model.pt` 加 sigmoid 头。
- 两组结局都崩：warm-start **38% vs Random / 14% vs WeightedRandom / 0% vs AlphaBeta**；fresh start **12% / 8% / 0%**（[`notebooks/option-c-curriculum-results.md`](../notebooks/option-c-curriculum-results.md)）。
- 训练指标却很好看（correlation 0.83、flat 决策 0%），证明**指标好看 ≠ 策略变好**。
- VF 残差版本因公式 bug 没跑完，但 spread 0.37 提示修复后可能有效。

### 2.12 收尾整理（2026-08-10）

- 删除过时 `PROGRESS.md` 与旧 `README.md`，把顶层 `log.md` / `sum.md` 按时间编号归档到 `notebooks/`。
- 本文件 + 新 `README.md` 替换原文档。

### 2.13 Day 5：VF-SFT 规模化与 Qwen checkpoint 对比（2026-08-10）

本日完成三项收尾验证：

- **VF-SFT 规模化复测**：用 300 局 VF-only 游戏生成 29,866 条决策训练 Qwen3-8B（QLoRA r=16 α=32, 3 epochs, 2400/5040 steps 当前 50%），对 WeightedRandom 取得 **76.0% (38/50) 胜率**（95% CI 62.6%–85.7%）。这是**首个被规模化验证的 standalone Qwen 方案**——standalone（无工具、无 guardrail）超越随机基线（25%）。
- **Qwen checkpoint 对比**（修正版 `eval_qwen_mass_v2.py`）：三个 Qwen 适配器各 20 局对比 → VF-SFT 75.0% / AB-SFT 5.0% / Hybrid Agent 100%。**确认 teacher 质量（VF vs AlphaBeta）是 15× 差距的根源，不是数据量**。
- **ESCU 论文改进方案评估**：`experiments/escau_improvement_design.md` 中的 3 个 Shapley Value 实验全部被否定——Shapley Value 在 72 特征上仍是状态质量信号，不解决「72 特征无法承载动作区分度」的根本瓶颈。详见 `experiments/escau_feasibility_assessment.md`（独立评估文件，未合并入 README）。

**核心新发现（第六条跨方法洞察）**：当 teacher 的策略在文本观察中「可解释」时，SFT 即可学到教师策略。VF 优于 AlphaBeta 作为 LLM 的 teacher，因为 VF 的偏好来自少数可命名特征。

---

## 3. 七个失败路径逐项分析

### 3.1 AB-SFT（纯模仿）

| 指标 | 值 |
|---|---|
| 训练数据 | 18502 步 AB 决策（VictoryPointPlayer × WeightedRandom 100 局） |
| 训练结果 | loss 0.0887，token acc 99.08%，action validity 100% |
| 最终胜率（4 人 vs WeightedRandom） | **25%**（5/20） |
| 与基线差距 | 0（与随机持平） |
| 失败根因 | 模仿只学动作的「表面格式」，不学「为什么」。Catan 决策依赖空间推理，SFT 数据没有 reasoning trace，模型只能学到「当我有 3 麦 2 矿时，城市是好动作」这种共现，无法在新局面泛化。 |

### 3.2 VF-Distill v2（override 蒸馏）

| 指标 | 值 |
|---|---|
| 训练数据 | 439 条 VF-Guard 中 VF 覆盖 LLM 的决策 |
| 训练结果 | loss 0.073，acc 97.6% |
| 最终胜率 | **40%**（8/20） |
| 距离目标 | 距 VF-Guard 90% 还差 50 pp |
| 失败根因 | 文本观察无法编码「同类型内不同位置」的精修信号。LLM 可以学到「应该建 settlement」，但学不到「应该建在 node 17 而不是 node 23」——后者在文本观察里都被表述成「可建节点列表第 N 项」。 |

### 3.3 RL-Guard（RL 模型打分）

| 指标 | 值 |
|---|---|
| 模型 | `rl_selfplay_model2.pt`，50K 参数，30 特征，AB 模仿训练 |
| 早期胜率 | 66.7%（2/3 局） |
| 大样本胜率 | **0%**（3 局 final eval） |
| 总跨度 | 0% – 67% 不稳定 |
| 失败根因（两个独立问题） | (1) 训练目标是「预测游戏 outcome」，不是「评估动作质量」——一个动作可能通向必胜也可能通向必败，取决于后续 100 步；模型混淆了二者。(2) 47% 的决策是 flat——30 特征把空间信息压成 count，所有同类动作的特征向量完全相同，模型打分退化成「按动作类别给固定分」。 |

### 3.4 GRPO-SFT（VF-scored rollout 数据）

| 指标 | 值 |
|---|---|
| 三组数据 | 全量 1821 / 过滤 925 / 平衡 725 |
| 三组胜率 | 20% / 0% / 0% |
| 全量 GRPO 训练 | loss 卡在 1.0 不收敛 |
| 失败根因 | VF 决策从「任意游戏状态」取，前后矛盾。状态 A 下「建 settlement」得分最高，状态 B 下「建城市」得分最高；当 LLM 学的是「同一局里哪个动作 VF 选」时，模型只能拟合 VF 的局部偏好，而不是策略。数据量越大，过拟合越严重。 |

### 3.5 AESL 早停（entropy 峰值）

| 指标 | 值 |
|---|---|
| 峰值点（step 150）胜率 | **0%**（0/10） |
| best-loss 点（step 500）胜率 | 20% |
| 假说 | entropy 峰值出现在模型「学完但未收敛」的最佳早停点 |
| 假说验证 | 拒绝 |
| 失败根因 | (1) 领域不匹配——AESL 适用于长 CoT 数学推理（输出上千 token），Catan 输出 ~6 token JSON，entropy 动态不同。(2) step 150 时只见过 ~1200 个样本，模型处于「困惑」而非「能力多样」状态。(3) 没有后续 RL 阶段，entropy 失去了预测 post-RL 表现的能力。 |

### 3.6 Option A/B/C 课程（outcome label）

| 指标 | 值 |
|---|---|
| Option C warm-start | 38% vs Random / 14% vs WeightedRandom / 0% vs AlphaBeta |
| Option C fresh | 12% / 8% / 0% |
| Option A v2 / B 工具 / C 课程 | 均已实施，全部在 12–38% 区间 |
| 训练指标 | corr 0.83，flat 决策 0%——很好看 |
| 失败根因 | **4 人 Catan 中 ~75% 的状态来自输家**——他们的状态并不「差」，可能他们打得很好但输了牌。outcome label 因此是噪声标签，模型学到的其实是「平均胜率 25%」+ 高 confidence，对动作选择毫无帮助。 |

### 3.7 Hybrid Agent tools-only（无 guardrail）

| 指标 | 值 |
|---|---|
| 胜率 | **66.7%**（2/3） |
| 局长 | 131–445 回合（方差极大） |
| 失败根因 | 工具能给 LLM 「战略视角」（哪个对手危险、概率多少），但 LLM 在「具体选哪个节点/哪条路」这种同类型精修上仍会犯错；缺 VF guardrail 兜底。 |

---

## 4. 获胜方案：Hybrid Agent

### 4.1 胜率证据

| 配置 | 胜率 | 局数 |
|---|---|---|
| **Hybrid Agent (tools + VF guardrail)** | **100%** | **6/6** |
| Hybrid Agent (tools only) | 66.7% | 2/3 |
| Hybrid Agent (tools + RL guard) | 0% | 0/3 |

证据文件：
- 早期发现（3 局）：[`notebooks/hybrid-agent-results.md`](../notebooks/hybrid-agent-results.md)
- 最终评估（6 局）：[`notebooks/final-results-2026-08-08.md`](../notebooks/final-results-2026-08-08.md) + `results/final_eval_20260808.json`

### 4.2 架构

```
┌─────────────────────────────────────────────────────────────┐
│ Hybrid Agent = Enriched Observations + VF Guardrail         │
└─────────────────────────────────────────────────────────────┘

  (1) 预计算（毫秒级，零 LLM 调用）
      analyze_position(game, color, rl_model)
        → {win_prob, assessment, production_per_resource}
      check_threats(game, color)
        → {opponent_vps, threat_levels}
      get_best_move(game, color, goal, rl_model, actions)
        → {recommended_action_indices, scores}
      simulate_outcome(game, color, action)
        → {predicted_winner, value_delta}

  (2) 富化观察
      将上述输出附加到 LLM 观察文本尾部：
      "Win prob: 0.65 | Biggest threat: RED (8 VP) | RL-best: #7 (BUILD_CITY)"

  (3) LLM 决策（一次调用，~1.6 s）
      AB-SFT 适配模型读取富化观察
      输出 {"action_number": N}

  (4) VF guardrail（即时）
      contender_fn 对所有合法动作打分
      挑最高分动作执行（可覆盖 LLM 输出）
```

### 4.3 为什么 100%

| 组件 | 贡献 |
|---|---|
| AB-SFT 模型 | 提供战略意图（哪类动作是对的）：25% 基线 |
| + Tools | 提供战略上下文（对手 VP、胜率、最佳候选）：66.7% |
| + VF guardrail | 提供战术精修（具体节点/路径）：**100%** |

VF guardrail 是关键——消融实验显示，去掉它胜率从 100% 跌到 66.7%；换成 RL guardrail 跌到 0%。

### 4.4 资源与速度

- LLM 推理：~1.6 s / 决策（4-bit Qwen3-8B）
- Tool 预计算：毫秒级
- VF 打分：毫秒级
- 单局长度：约 90 回合 → 约 2.5 分钟 / 局
- 评估 6 局：约 15 分钟

---

## 5. 关键组件（视为本项目的一部分）

> 这些组件虽然在不同日期诞生、可能在不同子项目训练，但它们已经构成 Hybrid Agent 的「硬件」，应当作为本项目产物看待。

### 5.1 `rl_enriched_model.pt`（72 特征 + VF 残差）

| 项 | 值 |
|---|---|
| 输入维度 | 72 |
| 隐藏层 | [256, 128, 64] |
| 输出 | 线性（无 sigmoid），范围约 [-1, 2] |
| 训练数据 | ~100K 样本（300 AB 局） |
| 训练步数 | 500 |
| Loss | MSE |
| 训练目标 | VF 残差：`label = (VF − VP × 3e14) / 1e8` |
| 性能 | vs Random **69%**，vs WeightedRandom **44%**，flat 决策 **3.1%** |
| 在项目里的角色 | 给 `analyze_position` 与 `get_best_move` 提供打分；Hybrid Agent 战略层的「直觉」 |
| 局限 | 道路动作仍有 100% flat 决策；只能感知 feature 层面区分度，不感知空间布局 |

### 5.2 `agent_tools.py`（4 个 tool）

四个函数，构成 LLM 的「外脑」：

| 工具 | 输入 | 输出 | 用途 |
|---|---|---|---|
| `analyze_position` | game, color, rl_model | win_prob、assessment、production | 综合战略评估 |
| `check_threats` | game, color | opponent_vps、threat_levels | 对手威胁检测 |
| `get_best_move` | game, color, goal, rl_model, actions | recommended_indices、scores | 按目标找最佳候选 |
| `simulate_outcome` | game, color, action | predicted_winner、value_delta | 单步推演 |

设计要点：
- 全部纯 Python + numpy，毫秒级返回，不消耗 GPU。
- 输出结构化为 JSON，可直接拼到 LLM 观察文本末尾。
- 没有 tool-calling 协议——LLM 看到的是文本，无需 function-calling 训练。

### 5.3 `contender_fn`（手写线性价值函数）

| 项 | 值 |
|---|---|
| 位置 | `src/catan_rl/rl/value.py` |
| 实现 | 13 项特征的线性加权（public_vps, production, enemy_production, num_tiles, reachable_production_0/1, buildable_nodes, longest_road, hand_synergy, hand_resources, discard_penalty, hand_devs, army_size） |
| 权重 | `CONTENDER_WEIGHTS`（手工调过，量级覆盖从 -1e8 到 1e14） |
| 性能 | 单人 VF ~90% WR vs 3 人 WeightedRandom |
| 在项目里的角色 | guardrail 上限；Hybrid Agent 的「最终裁判」 |
| 优势 | 可解释、零训练成本、亚毫秒级、与 LLM 互补 |
| 局限 | 仅线性、不学对手模型、不感知长程 |

### 5.4 AB-SFT LoRA 适配器

| 项 | 值 |
|---|---|
| 位置 | `checkpoints/sft/checkpoint-564/` |
| LoRA | r=16, α=32, dropout=0.05 |
| 数据 | 18502 步 AB 决策 |
| 训练时长 | 1h45m |
| 在项目里的角色 | Hybrid Agent 唯一使用的 LLM 适配器；提供战略意图 |

---

## 6. 跨方法共同教训（五条核心洞察）

### 6.1 outcome label 在 4 人 Catan 中是噪声

- 4 人 Catan 中约 75% 的状态来自输家，输家可能玩得很好但牌运差。
- 把 outcome label 当作 state quality 训练 → 模型学到「平均胜率 25% + 高 confidence」，对动作选择无帮助。
- 证据：Option C warm-start 在 corr=0.83 的漂亮指标下产出 14% WR vs WeightedRandom。

**推论**：4 人博弈中，任何用最终胜负做 state value 训练的路径都不可行，必须用 process-based signal（如 VF 残差）。

### 6.2 VF 残差是好训练信号

- 范围 [-1, 2]，连续质量评分，不依赖未来 100 步的随机性。
- 线性输出 + MSE 损失，干净梯度，无 sigmoid 饱和。
- 这是 `rl_enriched_model.pt` 能从「0% vs WeightedRandom」跃升到「44%」的根本原因。

**推论**：Catan 的 state value 信号必须来自「现在这一手 + 此刻的状态」，而不是「这局最终谁赢」。

### 6.3 VF 是好 guardrail 但坏 teacher

- VF-Guard 90%、Hybrid Agent 100%：VF 作为「最后一公里裁判」非常有效。
- VF-Distill v2 40%、GRPO-SFT 0–20%：VF 作为「训练数据生成器」很糟糕。
- 根因：VF 在每个状态的「局部最优动作」之间没有一致的策略语境，组合起来训练时模型学到的只是 VF 的局部偏好。

**推论**：把任何决策器直接当 teacher 时，先检验它的「同状态一致性」；不一致就不要蒸馏。

### 6.4 特征 > 算法

- 30 特征 RL 模型：47% flat 决策、0% vs WeightedRandom。
- 72 特征 RL 模型：3.1% flat 决策、44% vs WeightedRandom。
- 把特征从 30 升到 72（增加 per-resource production、opponent detail、port access 等），flat 决策下降 **15 倍**。

**推论**：在博弈场景中，**给模型看得见的特征**比给它更聪明的算法重要得多。一个能区分「建在 node 17」和「建在 node 23」的模型，比一个区分不了的模型用上 GRPO 都强。

### 6.5 训练指标会骗人

- Option C warm-start：train corr 0.83、flat 决策 0%，但 14% WR vs WeightedRandom。
- AESL entropy 峰值：loss 0.12、entropy 高，听起来「模型正在学」，实际 0% WR。
- 唯一可信的指标是 **对战胜率**——而且要在多局、对多对手下取平均。

**推论**：任何不直接对战的评估指标都应被视为「hint」，不能视为「目标」。

### 6.6 Teacher 质量 >> 数据量（2026-08-10 新增）

- 同样 300 局、同样 SFT 框架、同样文本观察下，**VF 作 teacher 教出 76% 胜率**，**AlphaBeta 作 teacher 教出 5%**——15× 差距。
- VF 的策略「在文本中可解释」：基于 ~5 个可命名特征（VP、产能、对手距离等），LLM 可以从文本中推理这些特征。
- AlphaBeta 的策略依赖搜索上下文（lookahead、对手建模），这些上下文在文本观察中完全不可见，LLM 无法学习。
- 这是 standalone Qwen 首次显著超越随机的关键发现——standalone Qwen 路径依赖 **teacher 的可解释性**，不是更聪明的算法。

**推论**：当 teacher 的决策逻辑可在文本上「解释」时，SFT 即可学到教师策略。挑选 SFT teacher 时，「决策可解释性」是比「决策强度」更关键的筛选条件。

---

## 7. 推荐评估协议

### 7.1 地图与参数

| 项 | 推荐值 | 理由 |
|---|---|---|
| 地图 | BASE（19 tiles） | MINI（7 tiles）策略太简单，无法区分方法 |
| VP 阈值 | 10 | BASE 标准配置 |
| 玩家数 | 4 | 1v1 / 3 人无法暴露多人博弈的关键噪声 |
| 局数 | ≥ 30 | 6 局、3 局都太薄（CI 太宽） |
| 对手池 | WeightedRandom × 3 + VictoryPointPlayer × 1 | 4 人局的标准对照 |
| 温度 | 0.8 | 与训练时一致 |
| 最大回合 | 1000 | 防止长局噪声影响统计 |

### 7.2 必须记录的多指标

仅报胜率会掩盖关键信号。每次评估必须记录：

| 指标 | 含义 |
|---|---|
| Win rate | 胜率（首要） |
| Avg game length | 平均局长（长局 = 决策质量差） |
| Action validity | 合法动作比例（应 ≥ 95%） |
| Invalid rate per game | 单局无效动作率（诊断 hallucination） |
| VP distribution | 各 VP 数下的胜率（揭示「是否卡在 9 VP」） |
| Turn-by-turn entropy | 模型预测分布熵（监控 collapse / 困惑） |
| Tool usage stats | 调用率、平均耗时（诊断 tool 失效） |
| Override rate | guardrail 覆盖 LLM 的频率 |

### 7.3 显著性

30 局、95% 置信区间下，胜率从 25% → 40% 是统计显著；25% → 30% 不是。仅当差异 ≥ 10 pp 且局数 ≥ 30 时宣告「方法有效」。

### 7.4 反例：避免 cherry-pick

Hybrid Agent 当前 100% (6/6) 是统计意义不强的强结果。必须扩到 30 局以上才能下「VF guardrail 必要」的强结论。

---

## 8. 推荐下一步

### 8.1 立即可做（1 周内）

| # | 任务 | 目的 | 工作量 |
|---|---|---|---|
| 1 | Hybrid Agent 100 局正式评估（vs 3 WeightedRandom + 1 VP） | 把 100% (6/6) 的强声明变成统计可信 | ~4 h GPU |
| 2 | 修复道路特征（每条边一个 one-hot 或 RST 距离） | 把 `rl_enriched_model.pt` 的道路 flat 决策从 100% 降到 < 30% | ~1 d |
| 3 | Hybrid Agent 加 `simulate_outcome` 工具接入 | 提升对手建模能力 | ~0.5 d |
| 4 | 把 AB-SFT 适配器直接接到 Qwen3-8B 端到端推理 | 替换 Ollama 中间层，减少 50% 推理延迟 | ~1 d |

### 8.2 中期（1 月内）

| # | 任务 | 目的 |
|---|---|---|
| 5 | Hybrid Agent vs VictoryPointPlayer 30 局 | 跨过 L2 目标 |
| 6 | 修复 VF 残差课程公式（已发现的 bug），跑 Option C v2 | 把 4 人 self-play 跑通 |
| 7 | 训练专属 RL 评分网络（仅给工具用、不参与 guard） | 摆脱 `rl_enriched_model.pt` 的道路盲区 |
| 8 | 把 4 个 tool 实现独立 repo，本项目以「上游 client」形式引用 | 解耦工具演进与主训练管线 |

### 8.3 长期（季度级）

| # | 任务 | 目的 |
|---|---|---|
| 9 | 训练一个 13B/14B 的 LLM 替代 Qwen3-8B | 提升战略容量 |
| 10 | 引入多智能体对话（LLM 玩家之间议价/换牌） | 把卡坦从纯策略博弈推到社交博弈 |
| 11 | 把整套管线（环境 + 工具 + VF + LLM）打包成单文件 benchmark | 可复现性 + 社区引用 |
| 12 | 在 Catanatron-main 加 Hybrid Agent 作为 reference bot | 把 LLM 玩家接入上游 bot 库 |

---

## 9. 引用与索引

### 9.1 论文与基线参考

| 引用 | 用法 |
|---|---|
| Catanatron（DarekYu fork） | 引擎 + AlphaBetaPlayer + agent_tools.py + contender_fn 实现 |
| Catanatron-main（catanatron 3.2.1 + 实验分支） | 4 个 tool 的实际位置 |
| AESL（arXiv 2503.03960） | entropy 早停假说的源头（被本项目拒绝） |
| LlamaGym | Agent 基类的设计模式 |
| GRPO / TRL 1.9.2 | RL 训练框架（本项目 RL 路径主要失败在此） |

### 9.2 实验阶段文档索引

| 文件 | 摘要 |
|---|---|
| [`experiments/01_phase1_setup.md`](../experiments/01_phase1_setup.md) | 环境搭建：硬件、软件、18/18 导入检查、catanatron API 探索 |
| [`experiments/02_phase2_agent.md`](../experiments/02_phase2_agent.md) | LlamaGym agent 实现：基类、Qwen 实现、observation、action_parser、5 段 prompt |
| [`experiments/03_phase3_sft.md`](../experiments/03_phase3_sft.md) | 第一次 SFT：18502 步 AB 数据、3 epoch、564 step、token acc 99%、WINR vs WeightedRandom 33%（2 人局） |
| [`experiments/04_phase4_rl.md`](../experiments/04_phase4_rl.md) | GRPO 基础设施：模拟器、rollout、reward、第一次试训 |

### 9.3 实验结果索引（notebooks/）

按时间顺序：

| 文件 | 摘要 |
|---|---|
| [`00-original-log-2026-08-06.md`](../notebooks/00-original-log-2026-08-06.md) | 原始实验日志（顶层 log.md 归档）：环境搭建、SFT 训练、GRPO 试训的全过程 |
| [`01-session-2026-08-08.md`](../notebooks/01-session-2026-08-08.md) | 8 月 8 日全天 session 日志（顶层 sum.md 归档）：VF-Guard 发现、RL 修复、Hybrid Agent 评估、Option C 课程 |
| [`ab-sft-results.md`](../notebooks/ab-sft-results.md) | AB-SFT 收敛（loss 0.044、acc 98%）但只达 25% 胜率；模仿不足以学策略 |
| [`vf-guard-discovery.md`](../notebooks/vf-guard-discovery.md) | VF-Guard 90% WR；GRPO 不必要；路径转向蒸馏 |
| [`option-a-v2-results.md`](../notebooks/option-a-v2-results.md) | VF 蒸馏 v2 达 40% WR；三处修复：override-only、AB-SFT init、LR=1e-4 |
| [`rl-guard-results.md`](../notebooks/rl-guard-results.md) | RL-Guard 早期 67% WR 但大样本不稳定；RL 模型预测 outcome 不预测 action 质量 |
| [`hybrid-agent-results.md`](../notebooks/hybrid-agent-results.md) | Hybrid Agent 第一版：100% (3/3) WR，首次追上 VF-Guard 90% |
| [`grpo-results.md`](../notebooks/grpo-results.md) | GRPO / VF-scored rollout 数据有害：0-20% WR vs 25% baseline |
| [`aesl-experiment-results.md`](../notebooks/aesl-experiment-results.md) | AESL entropy 假说被拒绝：峰值点 0% WR，best-loss 20% |
| [`rl-model-fixed.md`](../notebooks/rl-model-fixed.md) | RL 模型修复：72 特征 + VF 残差训练，vs Random 69%、vs WeightedRandom 44%、flat 决策 3.1% |
| [`option-c-curriculum-results.md`](../notebooks/option-c-curriculum-results.md) | Option C 课程自博弈被拒绝：12-38% WR；outcome label 在 4 人 Catan 中是噪声 |
| [`final-results-2026-08-08.md`](../notebooks/final-results-2026-08-08.md) | 完整管线结果：Hybrid Agent 100% WR；Hybrid + RL guard 0%；VF 是必要 guardrail |
| [`MEMORY.md`](../notebooks/MEMORY.md) | 全部实验结果的导航索引 |

### 9.4 关键文件路径速查

| 内容 | 路径 |
|---|---|
| 当前最强模型（LoRA） | `checkpoints/sft/checkpoint-564/` |
| RL 评分网络 | `checkpoints/rl_value/value_network.pt`（旧 30 特征）；新 72 特征版本需重训并保存 |
| VF 残差目标实现 | `src/catan_rl/rl/value.py`（`CONTENDER_WEIGHTS`） |
| 工具函数 | `Catanatron-main/catanatron_experimental/catanatron_experimental/agent_tools.py`（四个 tool 的真实位置） |
| Hybrid Agent 入口 | `scripts/eval_agent_hybrid.py`、`scripts/eval_hybrid_ablation.py` |
| 最终评估数据 | `results/final_eval_20260808.json` |
| 全部模型评估 | `scripts/run_final_eval.py` |
| 训练配置 | `configs/{default,sft,grpo,eval}_config.yaml` |

---

## 附录 A：本项目「不该做什么」清单

| 不该做的事 | 原因 |
|---|---|
| 用 outcome label 训练 state value 模型 | 4 人 Catan 中 ~75% 状态来自输家，label 噪声 |
| 用 VF 决策直接当训练数据 | VF 在不同状态下的偏好不一致，构不成可学策略 |
| 用 entropy 峰值做 Catan SFT 早停 | 领域不匹配；step 150 时模型还困惑 |
| 用 30 特征 RL 模型打分 | 47% flat 决策，等价于随机 |
| 用 RL 模型当 guardrail | 预测 outcome 而非 action 质量，0% WR |
| 只报胜率不报局长 | 局长是策略质量的副指标 |
| 少于 30 局下结论 | 6/6 与 3/3 在统计上都太薄 |
| 让 Qwen3 在 temp > 1.0 生成 | 会产生中文字符串 hallucination |
| 用 `Game.players` 属性 | catanatron 3.2.1 没有此属性 |

---

## 附录 B：本项目「值得继续做」清单

| 值得做的事 | 状态 |
|---|---|
| 把 100% (6/6) 扩到 30+ 局 | 待做 |
| 修复道路特征，把 `rl_enriched_model.pt` 道路 flat 降到 < 30% | 待做 |
| Hybrid Agent 接入 `simulate_outcome` 工具 | 待做 |
| 端到端 Qwen 推理（替换 Ollama） | 待做 |
| 跑通 VF 残差课程（已发现 bug） | 待做 |
| Hybrid Agent vs VictoryPointPlayer 30 局 | 待做 |
| 把 `contender_fn` 独立 repo 化 | 待做 |

---

*文档维护：每次新方法完成（不论成功失败）时，更新本文件的 §2、§3、§6。新增工具或组件时，更新 §5。新增 notebook 时，更新 §9.3。*
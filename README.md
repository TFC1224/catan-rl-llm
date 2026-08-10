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

## 实验结果一览

下面是四天工作中评估过的所有训练方法与变体，按最终对战胜率降序排列。每个条目的训练数据来源、关键超参、评估局数都标在表里。

| 方法 | 训练数据 | 设置 | 评估 | 对手 | 胜率 |
|---|---|---|---|---|---|
| Hybrid Agent（tools + VF） | AB-SFT 加权 (18,945 决策) + 4 个工具调用 | 推理时 VF guardrail | 6 局 | WeightedRandom | 100% (6/6) |
| VF-Guard | 无训练，AB-SFT 推理 + 推理时 VF 打分 | 推理时打分 | 10 局 | WeightedRandom | 90% (9/10) |
| Hybrid Agent（tools only） | AB-SFT + 4 工具 | 无 guardrail | 3 局 | WeightedRandom | 66.7% (2/3) |
| RL-Guard | 30 维特征 + outcome label | 全连接 MLP | 3 局 | WeightedRandom | 67% (2/3)（不稳定） |
| RL Model Fixed | 72 维特征 + VF 残差 | 全连接 MLP，linear + MSE | 20 局 | WeightedRandom / Random | 44% / 69% |
| VF-Distill v2 | 439 例 VF 覆盖 LLM 的决策 | LoRA r=16, 2 epochs, LR=1e-4 | 20 局 | WeightedRandom | 40% (8/20) |
| AB-SFT | 18,945 决策 / 300 局 AlphaBeta 自博弈 | QLoRA r=16 α=32, 3 epochs | 20 局 | Random | 25% (5/20) |
| GRPO-SFT-All | 1,821 例 VF-best rollout | LoRA, 2 epochs | 5 局 | WeightedRandom | 20% (1/5) |
| AESL Best-Loss | 同 AB-SFT + entropy 监控 | step=500 | 10 局 | WeightedRandom | 20% (2/10) |
| Option C Curriculum | outcome label curriculum | 1000 episodes | — | WeightedRandom / Random | 14% / 38% |
| GRPO-SFT-Filtered | 925 例 high-discrimination | LoRA, 2 epochs | 5 局 | WeightedRandom | 0% (0/5) |
| GRPO-SFT-Balanced | 725 例 phase-balanced | LoRA, 2 epochs | 5 局 | WeightedRandom | 0% (0/5) |
| AESL Entropy-Peak | 同 AB-SFT + entropy 监控 | step=150 | 10 局 | WeightedRandom | 0% (0/10) |
| Hybrid Agent（tools + RL） | AB-SFT + 4 工具 | RL guardrail | 3 局 | WeightedRandom | 0% (0/3) |

注意：除 VF-Guard（10 局）外，所有评估样本都在 20 局以下。6/6、9/10、8/20 这类数字在统计上与基线难以显著区分。30 局以上的复现评估列入下一步。

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

## 实验细节

下面对每个方法给出**做法**（数据规模、超参、推理时结构）、**结果**（胜率与样本量）、**失败原因**（若适用）。这一节是 RQ1–RQ6 与实验结果一览表的展开版。

### Hybrid Agent（tools + VF guardrail）— 100% (6/6)

**做法**：在 LLM 决策前调用四个工具，把结果追加到观察文本：

- `analyze_position` — RL 模型 (`rl_enriched_model.pt`) 输出当前胜率与产能评级
- `check_threats` — 扫描对手，输出每个对手的 VP、紧急度（≥8 VP / ≥6 VP）
- `get_best_move` — 给定目标（如「扩展最长道路」「建城市」）用 RL 模型给所有合法动作打分
- `simulate_outcome` — 单步推演候选动作的预期效果

LLM 仍输出 `{"action_number": N}`（无需重训，AB-SFT 权重直接复用）。最后由 `contender_fn` 给所有合法动作打分，挑最高分作为最终动作。

**结果**：6 局全胜。局长 57–127 回合，明显短于 RL-Guard 的 172–262 回合。胜率与 VF-Guard 同档（90%），但路由更长。

**消融**：去掉 VF guardrail 留 4 工具 → 2/3（66.7%）；去掉工具保留 VF guardrail 退化为 VF-Guard（90%）。两边都贡献正向收益，guardrail 贡献更大。

### VF-Guard（LLM + 手写 VF）— 90% (9/10)

**做法**：AB-SFT 权重不动，推理时做三件事：LLM 输出一个动作提议；`contender_fn`（catanatron 内置的 13 特征线性价值函数）对所有合法动作打分；如果 VF 的最高分动作不是 LLM 提议的动作，就覆盖。整局每步约 2 分钟（其中 LLM 调用是瓶颈）。

**结果**：10 局中胜 9 局。约 50% 的非平凡决策被 VF 覆盖，绝大多数是「同类型内的位置选择」（哪条边、哪个节点），LLM 负责动作类型选择、VF 负责位置精细化。

**意义**：这是本项目的实用上限。要想突破它，必须让 LLM 拿到文本观察无法承载的空间信息——这正是 RQ5（Hybrid Agent）的动机。

### RL Model Fixed（72 维特征 + VF 残差）— 44% vs WeightedRandom / 69% vs Random

**做法**：

特征从 30 维扩到 72 维，新增以下几类：

- 产能 per-resource（5 维）：随具体动作（强盗位置、定居点位置）变化
- 强盗上下文（3 维）：强盗所在玩家与地块值
- 港口访问（4 维）：当前拥有港口类型
- 对手明细（每个对手 18 维 ×3 = 54 维中的核心）：VP、knight 数、城市数、产能
- 强盗资源 one-hot（5 维）
- 路/扩展（4 维）：拥有地块数、平均产能
- 可建造标志（4 维）：city / settlement / road / dev card 是否可建
- 牌手距离（1 维）：到下一个城市/定居点完成距离

训练目标从「预测胜负（sigmoid + BCE）」改为「预测 VF 残差」：

```
label = (VF - VP * 3e14) / 1e8   # 范围 [-1, 2]
loss  = MSE, output linear (no sigmoid)
```

直接预测 VF 会被 VP 项淹没（6 个数量级），残差关注的是「同胜局水平下质量差异」。

数据：300 局 AlphaBeta 自博弈 ~100K 样本；网络 `[256, 128, 64]` 全连接；500 步训练。

**结果**：对 Random 胜率 25% → 69%；对 WeightedRandom 胜率 0–25% → 44%；action spread 0.018 → 0.064；flat 决策 47% → 3.1%。

**残留盲区**：所有 BUILD_ROAD 动作的特征向量仍然几乎一致（`roads_placed` 与 `my_road_len` 同），模型在修路决策上仍输出相同分数，需要 VF guardrail 兜底。

### VF-Distill v2（蒸馏 VF-Guard 覆盖决策）— 40% (8/20)

**做法**：跑 100 局 VF-Guard 游戏，记录 LLM 每步的提议与 VF 是否覆盖。1022 条决策里，VF 覆盖 LLM 的有 439 条。从 AB-SFT LoRA 续训（不重头），仅在这 439 条覆盖决策上 SFT：LoRA r=16，2 epochs，batch_size=16，LR=1e-4（比 AB-SFT 的 2e-4 小），max_length=1024。训练 9.5 分钟。

**结果**：loss 0.073，token 准确率 97.6%。8/20 胜率（40%），比 AB-SFT（25%）高 15 个百分点，比 VF-Guard（90%）低 50 个百分点。

**失败原因**：20 局的差距集中在「同类型内的位置选择」。文本观察把候选位置编码成「可建节点列表第 N 项」，无法承载几何相邻关系——LLM 在这种细粒度区分上始终输给显式计算。

**v1 → v2 的三处修复**：v1 仅 20% 胜率（2/10），原因是同时跑全 1022 条数据（覆盖 + 未覆盖混训）、从 base Qwen 启动（不续 AB-SFT）、LR=2e-4。三个一起改才到 40%。

### RL-Guard（30 维 RL 模型作 guardrail）— 67% (2/3) / 0% (3/3)

**做法**：用 `rl_selfplay_model2.pt`（30 维特征，sigmoid + BCELoss 训练，outcome label）替代 VF-Guard 中的 `contender_fn`，给所有合法动作打分。

**结果**：原始 3 局评估 2/3 (67%)；Hybrid 消融中重测 0/3。

**失败原因**：

- 30 维特征对同类动作（修路、强盗移动）输出几乎相同的分数——47% 的决策 flat（spread < 0.001）
- 训练目标是「预测胜负」，而不是「评估动作质量」——「正确动作」不一定是「胜局」，4 人局中 ~75% 状态来自输家
- 模型学会「平均 25% 胜率 + 高置信度」，对动作选择无帮助

后续 72 特征 + VF 残差的 `rl_enriched_model.pt` 即为对此失败模式的修复（见上）。

### AB-SFT（纯模仿）— 25% (5/20)

**做法**：用 AlphaBetaPlayer（DarekYu fork，catanatron 内置最强 bot）生成 300 局自博弈 18,945 条决策，98% AB 胜率。Qwen3-8B + QLoRA（r=16, α=32），3 epochs。推理时直接用训练好的 LoRA，每步采样温度 0.1。

**结果**：训练收敛，loss 1.627 → 0.044，token 准确率 68% → 98%。评估 20 局对 RandomPlayer，胜率 5/20 = 25%，与 4 人局加权随机基线齐平。合法动作率 100%。

**失败原因**：模仿学习只能复制「合法动作到动作序号」的表面映射，不能内化 AlphaBeta 选择该动作的几何/博弈理由。面对训练集外的棋盘布局与对手策略，映射就崩了。

### GRPO / VF-rollout SFT（3 组）— 0% – 20%

**做法**：把 GRPO 思路简化为「用 VF 在 rollouts 上打分的最佳动作作为 SFT 数据」，做三个过滤版本：

- GRPO-SFT-All：1,821 例 VF-best（所有 rollout）
- GRPO-SFT-Filtered：925 例 VF-best 且 VF 分数高离散度（hard examples）
- GRPO-SFT-Balanced：725 例 VF-best 且阶段均衡

都从 AB-SFT 续训，2 epochs。原始 GRPO 完整实现（带 KL 与策略梯度）也曾试，loss 在 1.0 附近不下降。

**结果**：All 1/5 (20%)；Filtered 0/5 (0%)；Balanced 0/5 (0%)。

**失败原因**：

1. VF 在任意状态上的「最优动作」不形成连贯策略。同类动作在不同上下文里可能被 VF 标「最优」也可能不被，模型学不到一致模式。
2. 数据量增加 4 倍反而更差（1,821 例 → 20% vs 439 例 → 40%），说明质量比数量重要。
3. 过滤让结果更差：high-discrimination 子集是 VF 分数差异最显著的 hard examples，但模型从中学不到规律。
4. VF-Distill v2 之所以有效是因为它训练「VF 覆盖 LLM 的一致模式」（一个有规则的决策边界），而不是「VF 的所有最优选择」。

### AESL Entropy Early-Stopping（2 组）— 20% / 0%

**做法**：在 5,000 例 AB-SFT 子集（从 17,050 例中采样）上 1 epoch 训练，每步记录熵与 loss，训练 step=150（entropy 峰值）与 step=500（best loss）两个 checkpoint 都评估。

**结果**：Best-Loss (step=500) 2/10 (20%)，Entropy-Peak (step=150) 0/10 (0%)。

**失败原因**：AESL 假设来自数学推理（长 CoT 输出数千 token），卡坦是 ~6 token 的 JSON 输出（`{"action_number": 5}`），熵动力学根本不同。step=150 的「高峰」反映的是「还没学会」，不是「多样性强」。且没有 RL 阶段（GRPO 在本任务失败），熵对后训练性能的预测能力失效。

### Option C Curriculum（outcome label 课程）— 14% / 38% / 0%

**做法**：三组自博弈课程实验：

- warm-start 从 `rl_enriched_model.pt` 出发 → sigmoid + BCELoss 训 1000 episodes
- fresh-start（base 模型）→ sigmoid + BCELoss 训 500 episodes
- VF 残差课程 — 公式有 bug 未完成

**结果**：warm-start vs Random 38% / vs WeightedRandom 14% / vs AlphaBeta 0%；fresh-start vs Random 12% / vs WeightedRandom 8% / vs AlphaBeta 0%；VF 残差实验因 loss=9e13 在早期停止。

**失败原因**：

- 训练指标完美（correlation 0.83、action spread 0.074、flat 决策 0%），但模型对动作打分与胜率无关——只是学会了「4 人局平均胜率 ~25% + 高置信度」
- VF 残差实验公式 bug：「`VP * VP_WEIGHT` 写成了 `VP_WEIGHT` 单项」（6 个数量级的常量），导致 loss 爆炸。修复后公式应为 `(VF - vp * VP_WEIGHT) / 1e8`
- 4 人 Catan 中 ~75% 状态来自输家，win/loss label 本身是噪声
- warm-start 从 VF 残差（[-1, 2] 线性输出）切到 sigmoid（[0, 1]）初始化错位，进一步放大噪声

### Hybrid Agent（tools + RL guard）— 0% (0/3)

**做法**：同 Hybrid Agent（tools + VF），但用 RL Model Fixed 后的 `rl_enriched_model.pt` 替代 `contender_fn` 作 guardrail。

**结果**：3 局全败。局长 155–350 回合，慢速输掉。

**失败原因**：尽管 RL 模型修复后整体胜率 44%，但修路决策上仍 flat（同 RQ6 残留盲区）。当 RL guard 强制改 LLM 的提议时，往往改成另一个「同样不优」的路段，加速劣势累积。

### Hybrid Agent（tools only，无 guard）— 67% (2/3)

**做法**：4 工具照常调用，LLM 提议即最终动作。

**结果**：2/3 胜率，局长 131–445 回合（比 tools+VF 长，比 tools+RL 短）。

**意义**：工具已经把胜率从 25% 推到 67%（提升 42 个百分点），证明「外部计算注入观察」有效；guardrail 再贡献 33 个百分点（67% → 100%）。

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
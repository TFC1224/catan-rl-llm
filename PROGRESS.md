# 项目进展报告：Catan RL + LLM

> 更新时间：2026-08-07
> 项目目标：用 LlamaGym + GRPO 微调 Qwen3-8B，使其成为有竞争力的《卡坦岛》AI 玩家
> 当前阶段：Phase 4（GRPO RL）遇阻 → 提出并验证 SimSFT 方案 → 完成 SimSFT 训练与初步评估

---

## 1. 项目概述

| 项目 | 内容 |
|---|---|
| 基础模型 | Qwen3-8B（4-bit QLoRA 微调） |
| 训练范式 | SFT 冷启动 → GRPO 强化学习 |
| 游戏引擎 | Catanatron v3.2.1 + catanatron-gym v4.0.0 |
| Agent 模式 | LlamaGym（get_system_prompt / format_observation / extract_action） |
| 硬件 | RTX 4090 D 24GB / AutoDL |
| 仓库 | `github.com/TFC1224/catan-rl-llm` |

架构示意：

```
LLM Agent (Qwen3-8B)  ←→  Catanatron Gym Environment
        │                        │
        │  format_observation()   │  env.get_valid_actions()
        │  extract_action()       │  env.step(action_index)
        v                        v
   GRPOTrainer (TRL)    ←  Reward via game simulation
```

---

## 2. 已完成工作

### Phase 1：环境搭建 ✅

- 全部 18/18 环境检查通过（torch 2.1.2、transformers 5.14.1、trl 1.9.2、catanatron 3.2.1）
- 关键版本约束：`flash_attention_2` 不可用 → 使用 `sdpa`；wandb 未登录 → `report_to=none`

### Phase 2：Agent 实现 ✅

- 5 个核心模块：`base.py`（抽象 Agent）、`prompts.py`（3 套系统提示词）、`observation.py`（7 段结构化观察）、`action_parser.py`（5 级动作解析回退）、`qwen_agent.py`
- 关键 API 适配：
  - `state.current_color` 是方法不是属性（`current_color()`）
  - Catanatron 的 `Game` 没有 `.players` 属性
  - P0（BLUE）必须用 `Color.BLUE` 枚举而非字符串
  - `env.get_valid_actions()` 返回**动作空间索引**（0–289），模型输出的是**顺序索引**，需做映射

### Phase 3：SFT 冷启动 ✅

**数据**：100 局 VictoryPointPlayer（专家）vs WeightedRandomPlayer，MINI 地图 6VP，共 20,558 条记录（train 18,502 / val 2,056）

**训练**：3 epochs / 564 steps / 约 1h45m

| 指标 | 值 |
|---|---|
| 最终 train loss | 0.0887 |
| 最终 eval loss | 0.02186 |
| Token 准确率 | 99.08% |
| 动作合法性 | **100%**（1018/1018） |
| 胜率（vs WeightedRandom） | 2/6 = 33.3% |

**结论**：SFT 模型学会输出合法 JSON 动作格式，但策略上并不强。

### Phase 4：GRPO 强化学习 ❌（根本性失败）

- 修复了 5 个关键 bug（reward 函数签名、Qwen3 thinking 模式、prompt 格式、TRL API 迁移等）
- 单步测试通过（loss 6e-05，entropy 0.18）
- 但完整训练暴露**根本性问题**：

| 指标 | 观测值 | 问题 |
|---|---|---|
| Entropy | **0.01–0.06** | 模型极度确定性，K=4 个生成几乎相同 |
| Advantage | 趋近 0 | 组内样本雷同 → 无学习信号 |
| Reward | -0.94 ~ -0.44 | 被无效动作/低分拖累 |
| KL | 1e-9 ~ 1e-4 | 策略几乎不动 |

**根因**：SFT 后的模型 entropy 极低（0.01–0.06），每组 K=4 次生成几乎完全相同 → advantage≈0 → GRPO 没有梯度可以学习。**这是调参无法修复的结构性问题。**

---

## 3. SimSFT：模拟引导的 SFT（当前方案）✅

### 3.1 思路

GRPO 依赖模型生成的多样性，但 SFT 模型太确定性。SimSFT 绕开这个依赖：

1. 对每个游戏状态**枚举所有合法动作**（而非模型生成）
2. 用 VictoryPointPlayer 模拟每个动作的结果（5 次取平均）
3. 选出**模拟最优动作**作为新的 SFT 训练目标
4. 在已有 SFT checkpoint 上继续精调

### 3.2 数据生成结果

| 指标 | 值 |
|---|---|
| 输入记录 | 1,000（来自 GRPO iter1 的 rollout.jsonl） |
| 可优化状态 | 212 条（排除只有 1 个合法动作的状态） |
| 找到更好动作的比例 | **44.3%** |
| 平均 reward 提升 | **+0.146** |
| train / val 划分 | 190 / 22 |

### 3.3 训练结果

| 指标 | 值 |
|---|---|
| 训练时间 | ~4 分钟 |
| 最终 loss | 0.036 |
| eval loss | 0.034 |

### 3.4 评估结果（遇阻）

- 2-player 评估：两个模型都是 0% 胜率、100% 动作合法性
- 3-player 评估：被输出缓冲问题阻塞，反复 kill 后仍未拿到干净结果

**评估遇阻的根因**：
- MINI 地图 + 6VP 下，连 VictoryPointPlayer（参考"强"bot）都几乎赢不了 —— 100 轮后 VP 仅 2–3 分
- **Catan 的 reward 信号极其稀疏且随机**：骰子运气对结果的影响大于动作选择
- 2-player 配置下资源生产过慢，游戏进程极慢
- SimSFT 仅用 212 条数据做 refinement，量太小，不足以改变模型行为

---

## 4. 核心发现与教训

### 4.1 主要成果

1. **动作合法性 100%** — 模型已学会输出正确格式的 JSON 动作
2. **SimSFT 数据有效** — 44.3% 的状态找到了 reward 更高的动作（平均 +0.146）
3. **完整训练管线打通** — 数据生成 → SFT → GRPO/SimSFT → 评估，全部脚本可用

### 4.2 核心问题

| # | 问题 | 严重度 |
|---|---|---|
| 1 | **SFT 后模型 entropy 过低（0.01–0.06）**，导致 GRPO 无学习信号 | 🔴 致命 |
| 2 | **Catan 的 reward 信号稀疏、随机**，模拟评估方差大 | 🔴 根本 |
| 3 | SimSFT 数据量太小（212 条）不足以改变模型行为 | 🟡 制约 |
| 4 | MINI 地图 + 6VP 太苛刻，连强 bot 都难赢 | 🟡 评估 |
| 5 | 评估输出缓冲问题浪费时间 | 🟢 工程 |

### 4.3 关键教训

> **SFT 冷启动阶段把模型"训得太死"是 RL 失败的根源。**
> 模型在 SFT 后变得极度确定性（entropy→0），这摧毁了后续 RL 所需的探索能力。
> 这一现象与 ICLR 2026 AESL 论文描述的 **distribution forgetting / diversity forgetting** 高度吻合（见第 5 节）。

---

## 5. 后续展望：基于 AESL 的冷启动 SFT 改进

> 论文：**Getting Your LLMs Ready for Reinforcement Learning with Lightweight SFT**（AESL, ICLR 2026）
> 核心洞察：**评估分数 ≠ RL 后性能**；冷启动 SFT 应关注输出**多样性**而非评估分数。

### 5.1 AESL 与当前项目问题的对应关系

| AESL 论文发现 | 本项目现状 | 启示 |
|---|---|---|
| 冷启动 SFT 过度训练 → **distribution forgetting** | SFT 训满 3 epochs，entropy 掉到 0.01–0.06 | 冷启动**不应**训到评估分数最高点 |
| **多样性指标**（entropy / self-BLEU）比评估分数更能预测 RL 潜力 | entropy 塌陷 → GRPO advantage=0 | 应在**多样性峰值**处停止冷启动 |
| AESL 自适应加权损失保护 base 分布 | 模型失去探索能力 → RL 无法学习 | 用 **AESL 损失**取代标准 CE 损失 |
| 轻量级冷启动（1k–6k 数据）就足够 | 当前用了 18.5k 全量数据 | 可能**过拟合了数据集分布** |

### 5.2 具体改进路线

**路线 A：SFT 冷启动阶段改用 AESL 损失（推荐）**

1. 用 AESL 损失函数替换标准 CE 损失：
   ```
   L = -Σ_t  w_t · log πθ(s*_t | q, s_<t)
   w_t = 1 - sigmoid( logit_t / t_scaling · prefix_avg_log_prob_t )
   ```
2. 训练过程中监控 **entropy / self-BLEU**，在多样性峰值处停止（而非 eval loss 最低点）
3. `t_scaling` 取 3–5（论文最优区间）
4. 用轻量子采样数据（1k–6k 而非全量 18.5k）

**预期收益**：冷启动后的模型保持足够多样性 → GRPO 的 K=4 组内样本有差异 → advantage 不再为 0 → RL 可学习。

**路线 B：SimSFT 数据规模化**

- 用 AESL 冷启动模型重新生成 GRPO rollout → 扩大 SimSFT 数据集（212 条 → 数千条）
- 只保留 improvement > 0 的记录做加权训练

**路线 C：解决评估信号问题**

- 用 BASE 地图 + 10VP（标准规则），避免 MINI 地图资源过少导致的评估失真
- 增加模拟 rollouts 数量（3–5 次）降低方差
- 增加评估游戏局数，降低 Catan 随机性影响

### 5.3 分阶段行动计划

```
Phase 5（下一步）: AESL 冷启动 SFT
    ├── 实现 aesl_loss.py（自适应加权 + entropy 监控）
    ├── 轻量数据（3k）+ t_scaling∈[3,5] 训练
    ├── 在多样性峰值处早停
    └── 验证：entropy 是否恢复到健康水平（>0.3）

Phase 6: GRPO 重试
    ├── 用 AESL 冷启动 checkpoint 替代原 SFT checkpoint
    ├── 修复评估脚本输出缓冲问题
    └── 用 BASE 地图 + 10VP 评估（胜率 > 0% 即算成功）

Phase 7: SimSFT 规模化
    ├── 基于 AESL 模型生成更大 SimSFT 数据集
    └── 与 GRPO 结果对比
```

### 5.4 风险与对策

| 风险 | 对策 |
|---|---|
| AESL 损失实现细节与论文有出入 | 严格按论文公式，t_scaling 扫 {3,5,7} |
| Catan reward 信号过于随机 | 增加 rollouts + 用 VP 差距做 dense reward |
| MINI 地图评估失真 | 评估迁移到 BASE 地图 + 10VP |

---

## 6. 待办清单

- [x] Phase 1 环境搭建
- [x] Phase 2 Agent 实现
- [x] Phase 3 SFT 冷启动
- [x] Phase 4 GRPO 基础设施 + 诊断（GRPO 本身失败）
- [x] SimSFT 数据生成（212 条，44.3% 改进）
- [x] SimSFT 训练（loss 0.036）
- [ ] SimSFT 完整评估（修复缓冲问题后重跑）
- [ ] Phase 5 AESL 冷启动 SFT（见 5.3）
- [ ] Phase 6 GRPO 重试（BASE 地图 + 10VP）
- [ ] Phase 7 SimSFT 规模化

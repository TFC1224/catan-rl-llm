# catan-rl-llm

用 LLM 玩《卡坦岛》：以 Qwen3-8B 为基座，通过 SFT（训练基本策略） + GRPO 强化学习训练一个卡坦岛 AI 玩家。

## 目录

- [背景与目标](#背景与目标)
- [当前进展](#当前进展)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [配置](#配置)
- [硬件要求](#硬件要求)
- [参考资料](#参考资料)

## 背景与目标

现有 LLM 游戏智能体的研究多局限于棋类或简单回合制游戏。《卡坦岛》兼具资源管理、路径规划与多人博弈等复杂要素，是更具挑战性的 RL 测试环境。本项目基于 LlamaGym 构建 LLM-Agent 交互框架，采用 TRL 实现 GRPO 训练，沿"先 SFT 学会基本策略，再 RL 在线优化"的技术路线，探索 LLM 在复杂策略游戏中的决策能力边界。

## 当前进展

| 阶段 | 状态 | 说明 |
|---|---|---|
| Phase 1 环境搭建 | ✅ 完成 | Catanatron + Qwen3-8B + TRL 环境验证通过 |
| Phase 2 Agent 实现 | ✅ 完成 | 观察格式化、动作解析、3 套系统提示词 |
| Phase 3 SFT 基础策略训练 | ✅ 完成 | 100% 动作合法性；MINI 地图胜率 33% |
| Phase 4 GRPO RL | ❌ 受阻 | SFT 后模型 entropy 过低，组内生成雷同，学习信号少 |
| SimSFT 模拟引导精调 | 🔄 实验 | 数据生成 + 训练已完成，评估未完成 |

**核心问题**：SFT 训练导致模型输出过于确定性，学习效率低。细节见 [PROGRESS.md](PROGRESS.md)。

**未来展望**：结束当前SimSFT后，按 AESL（ICLR 2026）论文思路改进冷启动 SFT——监控输出多样性并在熵峰处早停，观察是否提升后续训练效果。

## 项目结构

```
configs/               # YAML 配置文件
src/catan_rl/
  agent/               # LlamaGym 风格 agent
  env/                 # Catanatron 环境封装、游戏模拟、奖励函数
  data/                # 数据采集与数据集构建
  training/            # SFT / GRPO 训练脚本
  eval/                # 对局评估
scripts/               # CLI 入口
experiments/           # 各阶段实验记录
```

## 快速开始

### 1. 环境搭建

```bash
bash scripts/setup_env.sh            # 安装依赖
python scripts/download_model.py     # 下载 Qwen3-8B
python scripts/test_imports.py       # 验证环境（18/18 项）
```

### 2. SFT 预训练

```bash
# 从专家 bot（VictoryPointPlayer）对局生成训练数据
python scripts/generate_sft_data.py --num_games 500 --output data/sft/

# 训练模型模仿专家动作
python scripts/train_sft.py --data data/sft/ --output checkpoints/sft/
```

### 3. GRPO 强化学习

```bash
# 采集对局数据
python scripts/rollout.py --model checkpoints/sft/ --output data/grpo/iter1/ --num_games 100

# 训练
python scripts/train_grpo.py --lora checkpoints/sft/ --data data/grpo/iter1/ --output checkpoints/grpo/iter1/
```

### 4. 评估

```bash
python scripts/evaluate.py --model checkpoints/grpo/iter1/ --games 100 --output results/
```

> 注意：Phase 4 的 GRPO 训练目前不可用（见[当前进展](#当前进展)）。`--lora` 指向的 checkpoint 需在修复冷启动后重新训练。

## 配置

超参数集中在 `configs/`：

- `default.yaml` — 模型、LoRA、环境、生成参数
- `sft_config.yaml` — SFT 数据生成与训练参数
- `grpo_config.yaml` — GRPO 训练参数与课程
- `eval_config.yaml` — 评估设置

## 硬件要求

| 组件 | 显存 |
|---|---|
| Qwen3-8B（4-bit） | ~6 GB |
| LoRA（r=16） | ~0.1 GB |
| KV cache | ~2 GB |
| **合计** | **12+ GB**（推荐） |

已在 RTX 4090 D 24GB 上验证。

## 参考资料

- [Catanatron](https://github.com/bcollazo/catanatron) — 卡坦岛游戏引擎
- [LlamaGym](https://github.com/KhoomeiK/LlamaGym) — LLM + Gym agent 模式
- [TRL](https://github.com/huggingface/trl) — GRPO 训练库
- [AESL: Getting Your LLMs Ready for Reinforcement Learning with Lightweight SFT](https://github.com/LXXXXR/AESL)（ICLR 2026）— 冷启动 SFT 改进路线依据

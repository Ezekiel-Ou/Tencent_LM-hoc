# hok_semi PPO算法完全指南（面向初学者）

> 本文档详细解析 hok_semi 的PPO算法实现，帮助你理解神经网络是如何被训练优化的。

---

## 目录

1. [强化学习基础概念](#一强化学习基础概念)
2. [PPO算法核心思想](#二ppo算法核心思想)
3. [Loss函数详解](#三loss函数详解)
4. [Dual-Clip机制](#四dual-clip机制)
5. [Advantage归一化](#五advantage归一化)
6. [合法动作掩码](#六合法动作掩码)
7. [训练流程](#七训练流程)

---

## 一、强化学习基础概念

### 1.1 什么是强化学习？

**类比**：训练宠物狗
- **Agent（智能体）** = 狗
- **Environment（环境）** = 整个世界
- **State（状态）** = 当前看到的情况（如"前面有球"）
- **Action（动作）** = 可以做的事（如"跑过去"、"坐下"）
- **Reward（奖励）** = 做对了给零食，做错了不给

**目标**：让Agent学会在每种状态下选择最好的动作，以获得最多的累积奖励。

### 1.2 策略（Policy）和价值（Value）

| 概念 | 符号 | 含义 | 类比 |
|------|------|------|------|
| **策略** | π(a\|s) | "在状态s下，选择动作a的概率" | 狗的条件反射：看到球→跑过去 |
| **价值** | V(s) | "状态s有多好"（预期未来奖励） | 狗的预期：跑过去能捡到球，很开心 |

**神经网络的作用**：
- **策略网络**：输入状态，输出每个动作的概率
- **价值网络**：输入状态，输出一个数值（状态价值）

### 1.3 为什么需要PPO？

**问题**：直接优化策略很容易"步子迈得太大"，导致：
- 策略突然变差（"训崩了"）
- 样本利用率低（需要大量数据才能学好）

**PPO的解决思路**：
- **小步多次更新**：每次只更新一点点
- **限制更新幅度**：防止策略突变
- **重用样本**：一个样本可以被用来更新多次

---

## 二、PPO算法核心思想

### 2.1 策略更新公式

```
新策略 = 旧策略 + 小步调整
```

**具体实现**：
```python
# 计算新旧策略的概率比
ratio = π_new(action|state) / π_old(action|state)

# 限制ratio的范围
clipped_ratio = clip(ratio, 1-ε, 1+ε)  # ε=0.2

# 取最小值（防止更新过大）
loss = -min(ratio * A, clipped_ratio * A)
```

**为什么取最小值？**
- 如果ratio > 1+ε（更新太大）→ 用clipped_ratio限制
- 如果ratio < 1-ε（更新太小）→ 用clipped_ratio限制
- 如果在范围内 → 用ratio正常更新

### 2.2 hok_semi的PPO改进：Dual-Clip

标准PPO只限制了ratio的变化范围，hok_semi增加了**额外的下界保护**：

```python
# 标准PPO clip
surr1 = ratio * advantage
surr2 = clip(ratio, 1-0.2, 1+0.2) * advantage
clipped = min(surr1, surr2)

# Dual-Clip（hok_semi增加）
# 当advantage < 0时，额外限制不能过度惩罚
dual_clipped = max(clipped, 3.0 * advantage) if advantage < 0 else clipped
```

**为什么要加下界？**

假设当前动作比预期差（advantage < 0）：
- 标准PPO：可能会过度惩罚这个动作（ratio变得极小）
- Dual-Clip：限制惩罚程度，防止策略变得过于保守

**类比**：
- 标准PPO："这个动作不好，以后永远别做了"
- Dual-Clip："这个动作不太好，但别太绝对，下次再看看"

---

## 三、Loss函数详解

### 3.1 总Loss组成

```python
total_loss = value_loss + policy_loss + beta * entropy_loss
```

| 组件 | 作用 | 权重 |
|------|------|------|
| **value_loss** | 让价值函数预测更准确 | 0.5（默认） |
| **policy_loss** | 让策略选择更好的动作 | 1.0（默认） |
| **entropy_loss** | 鼓励探索，防止策略太保守 | beta=0.01 |

### 3.2 Value Loss（价值损失）

**目标**：让价值网络能准确预测"当前状态有多好"

```python
# 预测价值
predicted_value = value_mlp(public_hidden)  # (B, 1)

# 实际回报（从样本中获取）
actual_return = reward  # (B,)

# MSE损失
value_loss = 0.5 * MSE(predicted_value, actual_return)
```

**为什么用MSE？**
- 价值是一个连续数值（如"这个状态值100分"）
- MSE适合回归问题（预测连续值）

**可选：Value Clipping**
```python
# 与PPO的policy clip对称
old_value = reward - advantage
value_clipped = old_value + clip(predicted_value - old_value, -0.2, 0.2)
value_loss = max(MSE(predicted_value, reward), MSE(value_clipped, reward))
```

hok_semi默认**关闭**value clipping（`USE_VALUE_CLIP=False`）。

### 3.3 Policy Loss（策略损失）

**目标**：让策略网络选择能带来高回报的动作

```python
# 1. 计算动作概率
action_probs = softmax(logits, dim=-1)  # (B, action_dim)

# 2. 取出实际执行动作的概率
action_prob = action_probs[actual_action]  # (B,)

# 3. 计算新旧策略的比值
ratio = action_prob_new / action_prob_old  # (B,)

# 4. 计算Advantage（优势函数）
advantage = actual_return - predicted_value  # (B,)

# 5. PPO clip
surr1 = ratio * advantage
surr2 = clip(ratio, 1-0.2, 1+0.2) * advantage
policy_loss = -min(surr1, surr2)
```

**为什么是负号？**
- PyTorch默认做梯度下降（最小化loss）
- 但我们想**最大化**回报
- 所以加负号，把"最大化回报"变成"最小化负回报"

### 3.4 Entropy Loss（熵损失）

**目标**：防止策略变得太"保守"

**什么是熵？**
- 熵衡量"不确定性"
- 高熵：策略很随机（各动作概率差不多）
- 低熵：策略很确定（总是选同一个动作）

**为什么需要高熵？**
- 训练初期：需要探索，尝试不同动作
- 如果熵太低：策略会陷入局部最优（"我只学会了一种打法"）

```python
# 计算策略的熵
entropy = -sum(action_probs * log(action_probs), dim=-1)

# 加入loss（注意负号）
entropy_loss = -entropy  # 最大化熵 = 最小化负熵

# 总loss中加入entropy项
total_loss = value_loss + policy_loss + beta * entropy_loss
```

**beta的作用**：
- beta大（如0.1）：鼓励更多探索
- beta小（如0.001）：允许策略更确定
- hok_semi：beta_start=0.01

---

## 四、Dual-Clip机制详解

### 4.1 标准PPO的问题

标准PPO通过clip限制了ratio的范围：
```python
clip(ratio, 0.8, 1.2)  # ratio只能在0.8~1.2之间
```

但在极端情况下：
- 某个动作的实际回报远低于预期（advantage = -50）
- 如果ratio很大（如5.0，表示新策略概率是旧策略的5倍）
- 标准PPO可能会过度惩罚这个动作
- 导致策略崩溃

### 4.2 Dual-Clip的解决方案

当advantage < 0且ratio很大时：
```python
# 标准PPO
clipped = min(ratio * A, clip(ratio, 0.8, 1.2) * A)
# 如果ratio=5.0, A=-10：clipped = min(-50, -12) = -50（过于惩罚）

# Dual-Clip保护
dual_clipped = max(clipped, 3.0 * A)
# max(-50, -30) = -30（限制了惩罚程度）
```

**效果**：
- 防止极端情况下的策略崩溃
- 是一个"安全网"机制
- 在正常情况下通常不会触发

### 4.3 hok_semi的参数

```python
CLIP_PARAM = 0.2        # 标准clip范围：[0.8, 1.2]
DUAL_CLIP_PARAM = 3.0   # Dual-Clip下界保护
```

---

## 五、Advantage归一化

### 5.1 什么是Advantage？

```python
advantage = actual_return - predicted_value
```

**含义**：
- actual_return：实际获得的回报
- predicted_value：价值网络预测的回报
- advantage > 0：实际比预期好（"这个动作超预期"）
- advantage < 0：实际比预期差（"这个动作没达到预期"）

### 5.2 为什么要归一化？

**问题**：不同batch的advantage量级可能差异很大：
- Batch 1：advantage在[-100, 100]之间
- Batch 2：advantage在[-1, 1]之间

如果直接用于训练，会导致：
- 梯度大小不稳定
- 学习率难以调
- 训练震荡

**解决方案**：对每个batch的advantage做归一化

```python
if Config.USE_ADVANTAGE_NORM:
    valid_adv = advantage[frame_is_train > 0]
    if valid_adv.numel() > 1:
        advantage = (advantage - valid_adv.mean()) / (valid_adv.std() + 1e-8)
```

**效果**：
- 所有batch的advantage都近似服从标准正态分布
- 梯度大小稳定
- 训练更平滑

---

## 六、合法动作掩码

### 6.1 为什么需要掩码？

MOBA游戏中，很多动作在特定状态下是**非法的**：
- 技能CD中 → 不能释放该技能
- 没蓝了 → 不能释放耗蓝技能
- 被沉默了 → 不能释放任何技能

如果让模型在非法动作上浪费探索，会：
- 降低样本效率
- 导致奇怪的行为（如一直尝试放CD中的技能）

### 6.2 掩码的实现

```python
# 从观测中获取合法动作掩码
legal_action_flag_list = torch.split(feature_legal_action, LABEL_SIZE_LIST, dim=1)

# 对每个动作头分别处理
for task_index in range(len(IS_REINFORCE_TASK_LIST)):
    # 获取该动作头的合法掩码
    legal_mask = legal_action_flag_list[task_index]
    
    # 非法动作的logits压到极小值
    legal_action_flag_list_max_mask = (1 - legal_mask) * boundary  # boundary=1e20
    label_logits_subtract_max = torch.clamp(
        label_result[task_index] - torch.max(
            label_result[task_index] - legal_action_flag_list_max_mask, 
            dim=1, keepdim=True
        ).values,
        -boundary, 1
    )
    
    # 在掩码后的分布上计算概率
    label_exp_logits = legal_mask * (torch.exp(label_logits_subtract_max) + MIN_POLICY)
    label_sum_exp_logits = torch.clamp(label_exp_logits.sum(1, keepdim=True), min=epsilon)
    label_probability = label_exp_logits / label_sum_exp_logits
```

**核心逻辑**：
1. 合法动作：保留原始logits
2. 非法动作：logits减去一个极大值（如1e20），使其概率接近0
3. 重新计算softmax概率

### 6.3 数值稳定性处理

```python
MIN_POLICY = 0.00001  # 防止概率为0
```

**为什么需要MIN_POLICY？**
- 如果某个合法动作的概率为0，log(0)会返回-inf
- 导致梯度爆炸
- 加一个很小的偏移量（1e-5）保证数值稳定

---

## 七、训练流程

### 7.1 数据流

```
环境交互
    ├── Actor（采样进程）
    │   ├── 观察状态 → 预测动作 → 执行动作
    │   └── 收集样本：(state, action, reward, next_state)
    │
    └── Learner（训练进程）
        ├── 从Memory Pool获取样本
        ├── 计算Loss（value + policy + entropy）
        ├── 反向传播更新网络参数
        └── 定期保存模型
```

### 7.2 样本格式

```python
DATA_SPLIT_SHAPE = [
    FEATURE_DIM + LEGAL_ACTION_DIM,  # 3910 + 85 = 3995
    1,  # reward
    1,  # advantage
    1, 1, 1, 1, 1, 1,  # 各种标志位
    12, 16, 16, 16, 16, 9,  # 6个动作头
    1, 1, 1, 1, 1, 1, 1,  # 概率、权重等
    LSTM_UNIT_SIZE,  # 512
    LSTM_UNIT_SIZE,  # 512
]
```

### 7.3 关键超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| GAMMA | 0.995 | 折扣因子（未来奖励的衰减） |
| LAMDA | 0.95 | GAE参数（平衡偏差和方差） |
| CLIP_PARAM | 0.2 | PPO clip范围 |
| DUAL_CLIP_PARAM | 3.0 | Dual-Clip保护系数 |
| BETA_START | 0.01 | 熵系数初始值 |
| LR_START | 1e-5 | 学习率初始值 |
| BATCH_SIZE | - | 通常1024-2048 |
| GRAD_CLIP | 0.5 | 梯度裁剪范围 |

### 7.4 GAE（广义优势估计）

```python
# GAE公式
A_t = δ_t + (γλ)δ_{t+1} + (γλ)^2 δ_{t+2} + ...
其中 δ_t = r_t + γV(s_{t+1}) - V(s_t)
```

**参数含义**：
- γ (gamma)：折扣因子，0.995表示看重长期回报
- λ (lambda)：平衡偏差和方差
  - λ=0：低偏差，高方差（只看一步）
  - λ=1：高偏差，低方差（看很多步）
  - λ=0.95：平衡点

**在hok_semi中的实现**：
- 在 `definition.py` 的 `FrameCollector` 中计算
- 反向遍历时间步，累积计算advantage

---

## 总结

hok_semi 的PPO算法是一个**稳定、高效**的实现：

1. **标准PPO clip**：限制策略更新幅度，防止训崩
2. **Dual-Clip**：额外的安全保护，防止极端情况
3. **Advantage归一化**：稳定训练过程
4. **合法动作掩码**：提高样本效率
5. **Entropy正则化**：鼓励探索，防止局部最优

**关键认知**：PPO的核心不是"让策略变好"，而是"在不让策略变坏的前提下，让策略变好"。所有的clip、归一化、掩码都是为了这个目的。

---

*下一步建议：阅读 `hok_semi_奖励设计详解.md`，了解如何设计奖励函数引导策略学习。*

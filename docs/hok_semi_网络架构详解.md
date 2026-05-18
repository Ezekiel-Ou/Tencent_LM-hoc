# hok_semi 网络架构完全指南（面向初学者）

> 本文档详细解析 hok_semi 的网络模型设计，帮助你理解 3910 维输入如何经过神经网络生成动作决策。

---

## 目录

1. [网络架构整体概览](#一网络架构整体概览)
2. [输入层：特征分割](#二输入层特征分割)
3. [分组编码器（核心创新）](#三分组编码器核心创新)
4. [MLP + LSTM 融合层](#四mlp--lstm-融合层)
5. [Target Attention 机制](#五target-attention-机制)
6. [动作头与价值头](#六动作头与价值头)
7. [版本演进与教训](#七版本演进与教训)

---

## 一、网络架构整体概览

### 1.1 为什么需要复杂的网络？

MOBA游戏的决策是**多层次的**：
- **战术层**：躲避技能、补刀、换血
- **战略层**：推塔时机、回城决策、装备选择
- **目标层**：先打谁？（敌方英雄/小兵/防御塔）

单一的网络结构无法同时处理这么多层次，因此需要**模块化设计**。

### 1.2 hok_semi 的网络结构图

```
输入特征 (3910维)
    ↓
特征分割（按单位类型切分）
    ↓
分组编码器（每个类型有专属MLP）
    ├── 英雄编码器 → 128维
    ├── 小兵编码器 → 32维（4个max-pooling）
    ├── 河蟹编码器 → 32维
    ├── 防御塔编码器 → 32维
    └── 子弹编码器 → 32维（9个max-pooling）
    ↓
拼接层（所有编码结果拼接）
    ↓
并行处理
    ├── MLP分支（当前帧信息）
    │   └── [480→512→512] → 512维
    └── LSTM分支（时序信息）
        └── LSTM(512 hidden, 16 steps) → 512维
    ↓
融合层（拼接+MLP）
    └── [512+512→512] → 512维公共特征
    ↓
输出层
    ├── 动作头 × 5（button/move/skill方向）
    ├── Target Attention（选择攻击目标）
    └── 价值头（评估当前状态价值）
```

### 1.3 核心设计思想

| 设计 | 解决的问题 | 类比 |
|------|-----------|------|
| **分组编码** | 不同类型单位特征语义不同 | 就像人眼看东西：看人脸和看路标用不同的大脑区域 |
| **MLP+LSTM融合** | 既要即时反应又要记忆 | 开车时既要看当前路况，又要记住导航路线 |
| **Target Attention** | "打谁"和"怎么打"解耦 | 打架时先决定"打谁"，再决定"用什么招式" |
| **Max-Pooling** | 处理变长实体序列 | 看到一群敌人时，只关注最危险的那个 |

---

## 二、输入层：特征分割

### 2.1 为什么要分割？

3910维向量是一个**扁平数组**，但不同部分的语义完全不同：
- 第0~346维：我方英雄信息
- 第347~693维：敌方英雄信息
- 第694~1293维：我方小兵信息
- ...

如果不分割直接输入一个大MLP，模型需要自己学会"前347维是英雄，后面是小兵"，这非常浪费训练资源。

### 2.2 分割代码详解

```python
def forward(self, feature_vec, lstm_hidden_init, lstm_cell_init):
    # 第一步：按大类型分割
    feature_vec_split_list = feature_vec.split(
        [
            self.all_hero_feature_dim,      # 694 (347×2)
            self.all_soldier_feature_dim,   # 1400 (175×8)
            self.single_river_crab_feature_dim,  # 172
            self.all_organ_feature_dim,     # 344 (172×2)
            self.all_bullet_feature_dim,    # 1300 (130×10)
        ],
        dim=1,
    )
    
    # 第二步：细分每个类型
    hero_vec_list = feature_vec_split_list[0].split(
        [347, 347], dim=1  # 我方英雄, 敌方英雄
    )
    soldier_vec_list = feature_vec_split_list[1].split(
        [175*4, 175*4], dim=1  # 我方小兵, 敌方小兵
    )
    # ... 类似处理其他类型
```

### 2.3 分割的重要性

**如果没有正确分割**，模型可能：
- 把英雄的技能CD当成小兵的类型
- 把子弹的位置当成防御塔的血量
- 导致训练永远无法收敛

**验证方法**：`conf.py` 中的 `DimConfig` 定义了分割点，模型中用 `assert` 验证：
```python
assert len(x_all_units) == Args.DIM_ALL_UNITS
```

---

## 三、分组编码器（核心创新）

### 3.1 共享编码层

在分组之前，所有单位共享两个基础编码器：

```python
# 位置编码器：所有实体的位置信息共享
self.position_mlp = MLP([Args.DIM_DISTANCE, 128, 64], "position_mlp")
# 输入：125维位置编码（87相对+38全局）
# 输出：64维压缩位置特征

# 单位属性编码器：非位置属性共享
self.unit_no_pos_mlp = MLP([Args.DIM_UNIT - Args.DIM_DISTANCE, 64, 32], "unit_no_pos_mlp")
# 输入：42维（167通用-125位置=42，如血量、印记等）
# 输出：32维压缩属性特征
```

**为什么要共享？**
- "距离我1000单位"对英雄和小兵含义相同
- 共享减少参数量，防止过拟合
- 让模型学会通用的空间感知能力

### 3.2 分组编码器详解

#### 英雄编码器（最重要）

```python
# 输入维度：347(英雄专属) + (64+32-167)(共享编码delta) = 276
fc_hero_dim_list = [276, 512, 256, 128]

self.hero_mlp = MLP([276, 512, 256], "hero_mlp", non_linearity_last=True)
# 输入：276维（处理后的英雄特征）
# 输出：256维

self.hero_frd_fc = make_fc_layer(256, 128)  # 我方英雄 → 128维
self.hero_emy_fc = make_fc_layer(256, 128)  # 敌方英雄 → 128维
# 敌我分别过不同的FC层，因为"我方英雄"和"敌方英雄"的决策意义不同
```

**为什么敌我分开？**
- 我方英雄：我需要保护他、配合他
- 敌方英雄：我需要攻击他、躲避他
- 分开处理让模型更容易学习这种区别

#### 小兵编码器

```python
fc_soldier_dim_list = [175 + delta, 128, 64, 32]

self.soldier_mlp = MLP([..., 128, 64], "soldier_mlp", non_linearity_last=True)
self.soldier_frd_fc = make_fc_layer(64, 32)  # 我方小兵
self.soldier_emy_fc = make_fc_layer(64, 32)  # 敌方小兵
```

**Max-Pooling处理**：
```python
# 4个小兵各自过MLP后，取最大值
soldier_frd_concat_result = torch.cat(soldier_frd_result_list, dim=1) \
    .reshape(-1, 4, 32) \
    .max(dim=1).values
```

**为什么用Max-Pooling？**
- 小兵数量不固定（有时2个，有时4个）
- Max-Pooling保证输出维度固定（永远是32维）
- 取"最显著"的特征（如最近的小兵、最危险的小兵）

#### 防御塔编码器

```python
fc_organ_dim_list = [172 + delta, 128, 64, 32]

self.organ_mlp = MLP([..., 128, 64], "organ_mlp", non_linearity_last=True)
self.organ_frd_fc = make_fc_layer(64, 32)  # 我方防御塔
self.organ_emy_fc = make_fc_layer(64, 32)  # 敌方防御塔
```

#### 子弹编码器

```python
fc_bullet_list = [130 + pos_delta, 64, 64, 32]

self.bullet_mlp = MLP([..., 64, 64], "bullet_mlp", non_linearity_last=True)
self.bullet_hero_fc = make_fc_layer(64, 32)  # 英雄子弹
self.bullet_organ_fc = make_fc_layer(64, 32)  # 防御塔子弹
```

### 3.3 编码器输出汇总

| 实体类型 | 数量 | 单实体输出 | 处理后输出 | 说明 |
|---------|------|-----------|-----------|------|
| 我方英雄 | 1 | 128 | 128 | 直接输出 |
| 敌方英雄 | 1 | 128 | 128 | 直接输出 |
| 我方小兵 | 4 | 32 | 32 | max-pooling后 |
| 敌方小兵 | 4 | 32 | 32 | max-pooling后 |
| 河蟹 | 1 | 32 | 32 | 直接输出 |
| 我方防御塔 | 1 | 32 | 32 | 直接输出 |
| 敌方防御塔 | 1 | 32 | 32 | 直接输出 |
| 英雄子弹 | 9 | 32 | 32 | max-pooling后 |
| 防御塔子弹 | 1 | 32 | 32 | 直接输出 |

**拼接后总维度**：
```
128 + 128 + 32 + 32 + 32 + 32 + 32 + 32 + 32 = 480维
```

---

## 四、MLP + LSTM 融合层

### 4.1 为什么需要时序信息？

MOBA游戏中，**当前帧的信息是不够的**：
- 敌方英雄在往哪个方向移动？（趋势）
- 技能CD还有多久？（时间）
- 敌方刚才放了什么技能？（连招）

LSTM（长短期记忆网络）可以记住过去16帧的信息，帮助模型做**有预谋的决策**。

### 4.2 MLP分支（当前帧）

```python
self.concat_mlp_other = MLP([480, 512, 512], "concat_other_mlp")
# 输入：480维拼接特征
# 输出：512维
# 作用：提取当前帧的即时特征
```

**MLP的优点**：
- 计算快
- 能捕捉当前帧的复杂模式
- 适合"凭直觉"的快速反应

### 4.3 LSTM分支（时序）

```python
self.concat_mlp = MLP([480, 512], "concat_mlp", non_linearity_last=True)
# 先将480维降到512维，作为LSTM输入

self.lstm = nn.LSTM(
    input_size=512,
    hidden_size=512,
    num_layers=1,
    batch_first=True,
)
# 输入：(batch, seq_len=16, 512)
# 输出：(batch, seq_len=16, 512)
```

**LSTM的作用**：
- 记住过去16步（约16×6=96帧，约3秒）的历史
- 捕捉敌方移动趋势
- 预判技能释放时机

### 4.4 融合层

```python
# LSTM输出和MLP输出拼接
lstm_outputs = lstm_outputs.reshape(-1, 512)  # (B*L, 512)
public_mlp_result = self.concat_mlp_other(concat_result)  # (B*L, 512)

# 融合
public_hidden = self.lstm_and_linear_mlp(
    torch.cat([lstm_outputs, public_mlp_result], dim=-1)
)
# 输入：512+512=1024维
# 输出：512维
```

**融合的意义**：
- MLP说："现在敌方英雄在我左前方，血量很低"
- LSTM说："他过去3秒一直在往左走，可能要逃跑"
- 融合后："应该追击！"

### 4.5 为什么是512维？

- **太小**（如128维）：信息容量不够，复杂策略学不到
- **太大**（如2048维）：参数量爆炸，容易过拟合
- **512维**：在表达能力和计算效率之间的平衡点

champion.md的经验："FC:LSTM=3:1"，即全连接层占3份，LSTM占1份。

---

## 五、Target Attention 机制

### 5.1 为什么需要 Target Attention？

MOBA游戏中，一个核心决策是：**"我打谁？"**
- 敌方英雄（高风险高回报）
- 敌方小兵（补刀经济）
- 敌方防御塔（推塔胜利）
- 河蟹（额外经济）

Target Attention 让模型**先选择目标，再决定怎么打**。

### 5.2 Target Embedding 的构建

```python
# 从各实体的编码输出中提取"target"部分
target_embed_list = []

# 敌方英雄 → target embed
hero_emy_fc_out = self.hero_emy_fc(hero_emy_mlp_out)
_, target_part = hero_emy_fc_out.split([96, 32], dim=1)
target_embed_list.append(target_part)  # 32维

# 自己 → target embed（给自己加血/护盾）
hero_frd_fc_out = self.hero_frd_fc(hero_frd_mlp_out)
_, target_part = hero_frd_fc_out.split([96, 32], dim=1)
target_embed_list.append(target_part)  # 32维

# 敌方小兵 → target embed
for soldier in soldier_emy:
    soldier_emy_fc_out = self.soldier_emy_fc(soldier_emy_mlp_out)
    target_embed_list.append(soldier_emy_fc_out)  # 32维

# 敌方防御塔 → target embed
organ_emy_fc_out = self.organ_emy_fc(organ_emy_mlp_out)
target_embed_list.append(organ_emy_fc_out)  # 32维

# 插入固定嵌入
target_embed_list.insert(0, 0.1 * ones)  # "none"目标
target_embed_list.append(0.1 * ones)      # 占位
```

**Target顺序**（必须固定，与动作协议对齐）：
```python
TARGET_ORDER = [
    "none",           # 0: 无目标
    "enemy_hero",     # 1: 敌方英雄
    "self_hero",      # 2: 自己
    "enemy_soldier_1",# 3-6: 敌方小兵
    "enemy_soldier_2",
    "enemy_soldier_3",
    "enemy_soldier_4",
    "enemy_tower",    # 7: 敌方防御塔
    "monster",        # 8: 河蟹
]
```

### 5.3 Attention 计算

```python
# Query: 当前状态"想要什么样的目标"
lstm_target_embed_result = self.lstm_tar_embed_mlp(public_hidden)
# (B, 512) → (B, 32)

# Key: 每个候选目标的特征
target_query = self.target_embed_mlp(target_embedding)
# (B, 9, 32) → (B, 9, 32)

# Softmax归一化
target_query = nn.functional.softmax(target_query, dim=-1)

# Attention: 计算每个目标的得分
target_logits = torch.matmul(target_query, lstm_target_embed_result)
# (B, 9, 32) @ (B, 32, 1) = (B, 9, 1)
# reshape → (B, 9)
```

**直观理解**：
- Query 问："我现在想要一个血量低、距离近的目标"
- Key 回答："敌方英雄血量30%且距离500 → 匹配度高"
- Attention 输出："选敌方英雄！"

### 5.4 为什么用Attention而不是直接分类？

| 方式 | 缺点 | Attention优点 |
|------|------|--------------|
| 直接9分类 | 目标特征和决策特征混在一起 | 解耦：先编码目标特征，再做决策 |
| 手工规则 | 无法适应复杂情况 | 自动学习"什么样的目标值得选" |
| 独立网络 | 参数量大 | 共享backbone，轻量高效 |

---

## 六、动作头与价值头

### 6.1 动作空间设计

```python
LABEL_SIZE_LIST = [12, 16, 16, 16, 16, 9]
```

| 动作头 | 维度 | 含义 | 例子 |
|--------|------|------|------|
| button | 12 | 按哪个按键 | 普攻/技能1/移动/回城... |
| skill_x | 16 | 技能X方向 | 16等分地图 |
| skill_z | 16 | 技能Z方向 | 16等分地图 |
| move_x | 16 | 移动X方向 | 16等分地图 |
| move_z | 16 | 移动Z方向 | 16等分地图 |
| target | 9 | 选择目标 | none/敌方英雄/小兵/塔... |

**总动作维度**：12 + 16 + 16 + 16 + 16 + 9 = 85

### 6.2 动作头的实现

```python
# 前5个动作头（button + 4个方向）
self.label_mlp = ModuleDict({
    f"hero_label{label_index}_mlp": MLP(
        [512, LABEL_SIZE_LIST[label_index]],
        f"hero_label{label_index}_mlp",
    )
    for label_index in range(5)  # 0~4
})

# 第6个动作头（target）由Attention机制生成
```

### 6.3 价值头

```python
self.value_mlp = MLP([512, 64, 1], "hero_value_mlp")
# 输入：512维公共特征
# 输出：1维状态价值
```

**价值函数的作用**：
- 评估"当前状态有多好"
- 用于计算Advantage（优势函数）
- 指导策略更新方向

### 6.4 Button Logit Bias（冷启动技巧）

```python
BUTTON_LOGIT_BIAS = [-2.0, -1.0, 1.2, 0.5, 0.2, 0.2, 0.2, 0.0, 0.0, -1.5, -0.5, -0.2]
# 索引：  0      1     2    3    4    5    6    7    8    9     10    11
# 动作： invalid noop  move attack skill1 skill2 skill3 recover summoner recall skill4 equipment
```

**作用**：在训练初期给某些动作更高的先验概率：
- `move` (1.2)：鼓励移动，防止站桩
- `attack` (0.5)：鼓励普攻
- `noop` (-1.0)：抑制无操作
- `recall` (-1.5)：抑制过早回城

---

## 七、版本演进与教训

### 7.1 hok_semi 的网络版本

| 版本 | 结构 | 参数量 | 结果 |
|------|------|--------|------|
| model_origin.py | 纯MLP，无LSTM | ~1MB | 基线可用 |
| model.py | MLP+LSTM融合，单头 | 2.9MB | ✅ 最终采用 |
| model_multi_head.py | 共享backbone + 三英雄多头 | ~10MB | 训练慢，回退 |
| model_multi.py | 三独立模型 | 50MB | ❌ 训练失败 |

### 7.2 关键教训

1. **不要过度复杂化**：简单的MLP+LSTM融合效果最好
2. **单模型足够**：英雄差异通过hero_id输入让模型自己学习
3. **参数量要克制**：50MB的模型训练不稳定，2.9MB的模型收敛更好
4. **共享backbone**：位置编码、单位属性编码共享，减少冗余

### 7.3 与绝悟论文的对比

| 设计 | 绝悟论文（5v5） | hok_semi（1v1） | 原因 |
|------|----------------|----------------|------|
| CNN小地图 | ✅ 有 | ❌ 删除 | 开悟平台不提供图像 |
| 单位向量 | ✅ MLP | ✅ 分组MLP | 核心设计保留 |
| LSTM | ✅ 有 | ✅ 有 | 时序信息重要 |
| Target Attention | ✅ 有 | ✅ 有 | 目标选择关键 |
| 队友通信 | ✅ 有 | ❌ 删除 | 1v1无队友 |
| 多英雄head | ✅ 有 | ❌ 删除 | 单模型足够 |

---

## 总结

hok_semi 的网络架构是一个**经过验证的、稳定的、高效的设计**：

1. **分组Encoder**：让不同类型单位的信息被正确处理
2. **MLP+LSTM融合**：兼顾即时反应和时序记忆
3. **Target Attention**：解耦"选目标"和"怎么打"
4. **单模型路径**：参数量少，训练稳定，效果不差

**关键认知**：在强化学习中，**网络的"复杂度"不等于"性能"**。一个简单但训练充分的模型，往往比复杂但训练不足的模型更强。

---

*下一步建议：阅读 `hok_semi_PPO算法详解.md`，了解网络如何被训练优化。*

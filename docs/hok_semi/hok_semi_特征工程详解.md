# hok_semi 特征工程完全指南（面向初学者）

> 本文档详细解析 hok_semi 高分代码的特征工程实现，帮助你从零理解 3910 维观测向量是如何构建的。

---

## 目录

1. [特征工程整体思路](#一特征工程整体思路)
2. [位置编码系统（核心创新）](#二位置编码系统核心创新)
3. [单位特征编码详解](#三单位特征编码详解)
4. [完整特征拼接流程](#四完整特征拼接流程)
5. [维度计算验证](#五维度计算验证)
6. [调试与验证方法](#六调试与验证方法)

---

## 一、特征工程整体思路

### 1.1 为什么要做特征工程？

强化学习中，Agent（智能体）无法直接"看懂"游戏画面。特征工程的作用是将**原始游戏状态**（如英雄坐标、血量、技能CD等）转换为**神经网络可以理解**的固定维度数值向量。

**类比**：就像教人类下棋时需要告诉棋手"当前棋盘局势"，而不是让他自己看模糊的棋盘照片。

### 1.2 hok_semi 的设计理念

hok_semi 的特征工程遵循三个核心原则：

| 原则 | 含义 | 代码体现 |
|------|------|---------|
| **结构化分组** | 按单位类型（英雄/小兵/塔）分别编码 | `process_hero`, `process_soldier`, `process_sub_tower` |
| **位置双层编码** | 同时编码精细相对位置和粗粒度全局位置 | `process_position` 中的 `x_rpos` + `x_wpos` |
| **离散化连续值** | 将血量、CD等连续值转为one-hot编码 | `x_hp`, `x_skill_cd` 等 |

### 1.3 整体流程图

```
原始游戏状态 (state_dict)
    ├── frame_no: 当前帧号
    ├── hero_states: [我方英雄, 敌方英雄]
    ├── npc_states: [小兵×N, 防御塔×2, 河蟹×1]
    ├── bullets: [敌方子弹×N]
    └── cakes: [血包×2]
         ↓ 解包 (unpack_state_dict.py)
Info对象 (Python对象)
    ├── hero_our: HeroInfo
    ├── hero_enemy: HeroInfo
    ├── soldiers_our: [SoldierInfo×4]
    ├── soldiers_enemy: [SoldierInfo×4]
    ├── river_crab: ActorInfo
    ├── organ_our.sub_tower: ActorInfo
    ├── organ_enemy.sub_tower: ActorInfo
    ├── bullets_enemy: [BulletInfo×10]
    ├── cake_our: CakeInfo
    └── cake_enemy: CakeInfo
         ↓ 编码 (obs_builder.py)
观测向量 (3910维浮点数数组)
    ├── 单位个体特征 (2610维)
    │   ├── 我方英雄 (347维)
    │   ├── 敌方英雄 (347维)
    │   ├── 我方小兵 (175维×4 = 700维)
    │   ├── 敌方小兵 (175维×4 = 700维)
    │   ├── 河蟹 (172维)
    │   ├── 我方防御塔 (172维)
    │   └── 敌方防御塔 (172维)
    └── 飞行物特征 (1300维)
        └── 敌方子弹 (130维×10)
```

---

## 二、位置编码系统（核心创新）

### 2.1 为什么位置编码如此重要？

在MOBA游戏中，**位置信息**决定了：
- 能不能打到敌人（攻击距离）
- 会不会被敌人打到（敌方技能范围）
- 该不该逃跑（敌方人数、我方血量）
- 怎么推塔（距离敌方防御塔多远）

### 2.2 双层位置编码的设计

hok_semi 使用**两层不同粒度**的位置编码：

#### 第一层：相对位置编码（精细，用于近距离战斗）

**作用**：精确描述"目标相对于我"的位置

**参数**：
- 网格单位大小：`600` 游戏单位
- 最大范围：`24600`（即 ±12300）
- 视野内网格：`41×41`（中心英雄左右各20格）
- 视野外padding：左右各加1格 → 变成 `43×43`

**编码方式**：one-hot编码

```python
# 计算目标相对于当前英雄的网格索引
rpos = [
    int(clip(fix((target_x - hero_x) / 600), -21, 21) + 21)
    for target_x, hero_x in zip(target_pos, hero_pos)
]
# 结果：rpos[0] 在 [0, 42] 之间，rpos[1] 在 [0, 42] 之间

# one-hot编码
x_rpos = [0] * 87  # 43*2 + 1
x_rpos[rpos[0]] = 1.0           # x方向位置
x_rpos[rpos[1] + 43] = 1.0      # z方向位置
x_rpos[-1] = distance / 12000   # 归一化距离
```

**为什么是43维？**
- 中心点是第21维（索引21）
- 向左20格：索引 1~20
- 向右20格：索引 22~41
- 左溢出（超出视野）：索引 0
- 右溢出（超出视野）：索引 42

**维度计算**：
```
x_rpos维度 = 43(x方向one-hot) + 43(z方向one-hot) + 1(距离) = 87维
```

#### 第二层：全局位置编码（粗粒度，用于战略决策）

**作用**：描述"目标在整张地图上的大致位置"

**参数**：
- 网格单位大小：`5000` 游戏单位
- 最大范围：`90000`（即 ±45000）
- 全局网格：`18×18`

**编码方式**：one-hot + 局部比例

```python
# 计算目标在全局地图中的网格索引
wpos = [
    int(clip(floor((target_x + 45000) / 5000), 0, 17))
    for target_x in target_pos
]

# 计算在网格内的局部比例（-1到1之间）
wratio = [
    clip((target_x + 45000) / 5000 - wpos[i], -1, 1)
    for i, target_x in enumerate(target_pos)
]

# one-hot编码 + 比例
x_wpos = [0] * 38  # 18*2 + 2
x_wpos[wpos[0]] = 1.0
x_wpos[wpos[1] + 18] = 1.0
x_wpos[-2] = wratio[0]
_x_wpos[-1] = wratio[1]
```

**维度计算**：
```
x_wpos维度 = 18(x方向one-hot) + 18(z方向one-hot) + 2(局部比例) = 38维
```

### 2.3 两层编码的对比

| 特性 | 相对位置编码 | 全局位置编码 |
|------|-------------|-------------|
| **用途** | 精确战斗（躲避技能、普攻距离） | 战略规划（推进、撤退、支援） |
| **中心点** | 当前英雄 | 地图中心(0,0) |
| **网格大小** | 600（精细） | 5000（粗粒度） |
| **总范围** | ±12300 | ±45000 |
| **维度** | 87 | 38 |
| **one-hot精度** | 43档 | 18档 |

### 2.4 为什么不用原始坐标？

**问题**：原始坐标是连续值（如 x=12345, z=-6789），直接输入神经网络会有问题：
1. **量纲不一致**：坐标范围（±10000）和血量（0~2400）量级不同
2. **语义不明确**：x=10000 对于不同位置的英雄含义不同
3. **无法处理"不可见"**：视野外的敌人坐标未知

**解决方案**：
- **相对坐标**：以当前英雄为中心，语义统一（"在左前方600单位"对所有英雄含义相同）
- **离散化**：将连续坐标转为one-hot，消除量纲问题
- **不可见处理**：视野外用padding索引（0或42）表示

---

## 三、单位特征编码详解

### 3.1 通用单位信息（167维）

所有单位（英雄、小兵、防御塔、河蟹）都包含以下通用信息：

#### 位置信息（87 + 38 = 125维）
- 相对位置编码：`x_rpos`（87维）
- 全局位置编码：`x_wpos`（38维）

#### 血量信息（27维）

```python
# 连续值 + 离散one-hot
x_hp = [0] * 27
x_hp[0] = current_hp / max_hp  # 当前血量比例（连续值）
# 离散化血量（向上取整）
hp_bucket = min(ceil(current_hp / 100), 25)
x_hp[1 + hp_bucket] = 1.0       # one-hot编码
```

**为什么是25+1=26档？**
- 血量范围 0~2400
- 每100一档：0, 100, 200, ..., 2400
- 超过2400的统一到第25档
- 加上比例值，共27维

#### 印记信息（15维）

印记（Mark）是MOBA中的特殊机制，如：
- 后羿的普攻叠层（3层）
- 李元芳的一技能标记（4层）

```python
x_mark = [0] * 15
for mark_id, max_layer in MARK_ID_LAYERS.items():
    if mark_id in buff_marks:
        layer = min(buff_marks[mark_id], max_layer)
        x_mark[current_idx + layer] = 1.0
    current_idx += max_layer + 1
# 未知印记
if unknown_marks:
    x_mark[-1] = 1.0
```

**通用单位信息总维度**：
```
DIM_UNIT = 87(相对位置) + 38(全局位置) + 27(血量) + 15(印记) = 167维
```

### 3.2 英雄专属信息（180维）

在通用信息（167维）基础上，英雄还有额外信息：

| 特征 | 维度 | 说明 |
|------|------|------|
| 英雄类型 | 1 | -1/0/1标识三种英雄 |
| 行为状态 | 9 | 死亡/待机/移动/普攻/复活/技能1/2/3/其他 |
| 法力值 | 10 | 当前/最大 + 离散8档 |
| 一技能CD | 14 | 当前/最大 + 离散12档 |
| 二技能CD | 14 | 同上 |
| 三技能CD | 14 | 同上 |
| 闪现CD | 14 | 同上 |
| 回复CD | 14 | 同上 |
| 等级 | 15 | 1~15级one-hot |
| 金币获取 | 18 | 离散增量 + 总量比例 |
| 草丛状态 | 1 | 是否在草丛中 |
| 塔关系 | 2 | 是否在塔范围/是否为塔目标 |
| Buff状态 | 54 | 各种buff的one-hot |

**英雄总维度**：
```
DIM_HERO = 167(通用) + 180(专属) = 347维
```

### 3.3 小兵专属信息（8维）

在通用信息（167维）基础上：

| 特征 | 维度 | 说明 |
|------|------|------|
| 行为状态 | 3 | 死亡/攻击路径/其他 |
| 类型 | 3 | 近战/远程/炮车 |
| 塔关系 | 2 | 是否在塔范围/是否为塔目标 |

**小兵总维度**：
```
DIM_SOLDIER = 167(通用) + 8(专属) = 175维
```

**注意**：最多取4个小兵（按距离排序）

### 3.4 防御塔专属信息（5维）

| 特征 | 维度 | 说明 |
|------|------|------|
| 攻击目标 | 3 | 无/英雄/小兵 |
| 血包状态 | 1 | 是否有血包 |
| 血包倒计时 | 1 | 距离下次生成的时间比例 |

**防御塔总维度**：
```
DIM_ORGAN = 167(通用) + 5(专属) = 172维
```

### 3.5 河蟹专属信息（5维）

| 特征 | 维度 | 说明 |
|------|------|------|
| 行为状态 | 5 | 死亡/自动/复活/出生/其他 |

**河蟹总维度**：
```
DIM_RIVER_CRAB = 167(通用) + 5(专属) = 172维
```

### 3.6 子弹专属信息（5维）

子弹只有**位置信息**和**来源技能**，没有血量等属性：

| 特征 | 维度 | 说明 |
|------|------|------|
| 来源技能 | 5 | 普攻/技能1/2/3/其他 |
| 位置编码 | 125 | 与单位相同的位置编码 |

**子弹总维度**：
```
DIM_BULLET = 5(来源) + 125(位置) = 130维
```

---

## 四、完整特征拼接流程

### 4.1 拼接顺序

```python
# 单位个体（2610维）
x_hero_our = process_hero(hero_our)           # 347维
x_hero_enemy = process_hero(hero_enemy)       # 347维
x_soldier_our = process_soldier(soldiers_our) # 175×4 = 700维
x_soldier_enemy = process_soldier(...)        # 175×4 = 700维
x_river_crab = process_river_crab()           # 172维
x_tower_our = process_sub_tower(tower_our)    # 172维
x_tower_enemy = process_sub_tower(tower_enemy)# 172维

# 飞行物（1300维）
x_bullets = process_bullets()                 # 130×10 = 1300维

# 总维度
DIM_ALL = 2610 + 1300 = 3910维
```

### 4.2 为什么是这个顺序？

顺序决定了模型在 `forward` 中如何 `split` 特征向量。必须严格保持一致：

```python
# model.py 中的 split 逻辑
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
```

**如果改变拼接顺序，模型会读错特征，导致训练失败！**

---

## 五、维度计算验证

### 5.1 手动验证计算

```python
# 单位个体
DIM_HERO = 347
DIM_SOLDIER = 175
DIM_RIVER_CRAB = 172
DIM_ORGAN = 172

DIM_ALL_UNITS = 2*DIM_HERO + 8*DIM_SOLDIER + DIM_RIVER_CRAB + 2*DIM_ORGAN
              = 2*347 + 8*175 + 172 + 2*172
              = 694 + 1400 + 172 + 344
              = 2610

# 飞行物
DIM_BULLET = 130
DIM_BULLETS = 10 * DIM_BULLET = 1300

# 总维度
DIM_ALL = DIM_ALL_UNITS + DIM_BULLETS = 2610 + 1300 = 3910
```

### 5.2 代码中的验证

```python
# conf/conf.py
assert Args.DIM_ALL == 3910
assert Args.DIM_HERO == 347
assert Args.DIM_SOLDIER == 175
```

---

## 六、调试与验证方法

### 6.1 打印维度检查

```python
obs_builder = ObsBuilder()
obs, masks = obs_builder.build_observation(info, need_mask=True)

print(f"观测维度: {len(obs)}")  # 应该输出 3910
print(f"期望维度: {Args.DIM_ALL}")
assert len(obs) == Args.DIM_ALL
```

### 6.2 固定值调试法

```python
# 将某类特征固定为特定值，检查模型是否正确划分
for x0, value in zip(
    [x_hero_our, x_hero_enemy, x_soldier_our, x_soldier_enemy, 
     x_river_crab, x_tower_our, x_tower_enemy],
    [0, 1, 2, 3, 4, 5, 6]
):
    for i in range(len(x0)):
        x0[i] = value
```

如果模型划分正确，每个子网络（hero_mlp, soldier_mlp等）应该输出对应固定值的特征。

### 6.3 检查位置编码

```python
# 测试位置编码是否正确
hero_pos = (0, 0)
target_pos = (600*20, 0)  # 在x方向20格处

obs_builder = ObsBuilder()
obs_builder.pos = hero_pos
x_pos = obs_builder.process_position(target_pos)

# 检查one-hot位置
import numpy as np
x_pos = np.array(x_pos)
print(f"相对位置激活索引: {np.argwhere(x_pos[:43]).reshape(-1)}")
# 应该输出 [41]（索引41 = 第42维 = 右方20格+1格padding）
```

---

## 总结

hok_semi 的特征工程是一个**精心设计的系统**：

1. **双层位置编码**同时提供精细战斗感知和宏观战略感知
2. **结构化分组**让不同类型单位的信息互不干扰
3. **离散化连续值**消除了量纲问题，便于神经网络学习
4. **严格的维度控制**确保了模型输入的一致性

理解这套特征工程的关键是：**想象你自己在玩这个游戏，你需要哪些信息才能做出好的决策？** hok_semi 的设计就是将这些信息以最有利于神经网络学习的方式编码出来。

---

*下一步建议：阅读 `hok_semi_网络架构详解.md`，了解这3910维向量如何被神经网络处理。*

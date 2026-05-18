# hok_semi 奖励设计完全指南（面向初学者）

> 本文档详细解析 hok_semi 的奖励函数设计，帮助你理解如何通过奖励引导AI学习正确的行为。

---

## 目录

1. [奖励设计的重要性](#一奖励设计的重要性)
2. [8项基础奖励详解](#二8项基础奖励详解)
3. [零和奖励机制](#三零和奖励机制)
4. [时间衰减机制](#四时间衰减机制)
5. [前进奖励（Forward Reward）](#五前进奖励forward-reward)
6. [奖励权重调参](#六奖励权重调参)
7. [常见陷阱与解决方案](#七常见陷阱与解决方案)

---

## 一、奖励设计的重要性

### 1.1 奖励是强化学习的"指南针"

**类比**：训练宠物狗
- 如果"坐下"给零食，"握手"不给 → 狗只会坐下
- 如果"坐下"给1个零食，"握手"给10个零食 → 狗更爱握手

**在MOBA中**：
- 如果只有"推塔胜利"奖励 → AI可能永远学不会（太稀疏）
- 如果"击杀"给高奖励 → AI可能只追杀，不推塔
- 如果"经济"给高奖励 → AI可能只刷兵，不打团

### 1.2 好的奖励设计标准

| 标准 | 说明 | 例子 |
|------|------|------|
| **对齐目标** | 奖励应与最终胜利条件一致 | 推塔是胜利条件，所以tower_hp奖励权重高 |
| **足够密集** | 不能只有稀疏奖励 | 每帧都有hp、经济等奖励 |
| **可区分** | 不同行为应有不同奖励 | 补刀给钱奖励，击杀给击杀奖励 |
| **不冲突** | 奖励之间不应矛盾 | 不能同时鼓励" aggressive"和"passive" |

---

## 二、8项基础奖励详解

### 2.1 奖励总览

| 奖励项 | 类型 | 计算方式 | hok_semi权重 | 设计意图 |
|--------|------|---------|-------------|---------|
| **hp_point** | dense | 血量比例开四次方后差分 | 2.0 | 换血策略 |
| **tower_hp_point** | dense | 塔血比例差分 | 10.0 | 推塔核心 |
| **money** | dense | 金币总量差分 | 0.004 | 经济发育 |
| **exp** | dense | 经验总量差分 | 0.004 | 等级提升 |
| **ep_rate** | dense | 法力比例差分（仅正增长） | 0.75 | 技能续航 |
| **death** | sparse | 死亡次数差分 | -1.0 | 惩罚死亡 |
| **kill** | sparse | 击杀次数差分 | -0.6 | 鼓励击杀 |
| **last_hit** | sparse | 补刀次数 | 0.5 | 补刀经济 |
| **forward** | dense | 前进进度 | 0.01 | 反站桩 |

### 2.2 hp_point（换血奖励）

**计算方式**：
```python
# 当前血量比例
hp_rate = current_hp / max_hp
# 开四次方（非线性变换）
hp_point = hp_rate ** 0.25
# 奖励 = 当前帧 - 上一帧
reward = hp_point_current - hp_point_last
```

**为什么要开四次方？**

```
血量比例 → 开方后
0.0 → 0.0
0.2 → 0.67
0.5 → 0.84
0.8 → 0.95
1.0 → 1.0
```

**效果**：
- 血量高时（如80%→100%）：变化小，奖励小
- 血量低时（如20%→0%）：变化大，惩罚大
- **设计意图**：鼓励"血量高时换血，血量低时珍惜生命"

**类比**：满血时你愿意和敌人换血（反正血多），残血时你会逃跑（保命重要）。

### 2.3 tower_hp_point（推塔奖励）

**计算方式**：
```python
# 塔血比例
tower_hp_rate = tower_hp / tower_max_hp
# 奖励 = 当前帧 - 上一帧（零和）
reward = (my_tower_hp_rate_current - my_tower_hp_rate_last) - \
         (enemy_tower_hp_rate_current - enemy_tower_hp_rate_last)
```

**为什么权重最高（10.0）？**
- **推塔是胜利条件**
- 高权重确保AI始终将推塔作为首要目标
- 其他所有行为最终都是为了推塔服务

**注意**：hok_semi中权重是10.0，但作者最终总结建议改为5.0（agent_diy已修正）。

### 2.4 money（经济奖励）

**计算方式**：
```python
# 当前总金币
money = total_money
# 奖励 = 当前帧 - 上一帧（零和）
reward = (my_money_current - my_money_last) - (enemy_money_current - enemy_money_last)
```

**为什么用总量差分而不是增量？**
- 总量差分 = 当前金币 - 上次金币 = 这段时间获得的金币
- 直接反映了经济发育情况

**权重0.004的含义**：
- 获得1000金币 → 奖励 = 1000 × 0.004 = 4.0
- 与hp_point（权重2.0）相比，经济奖励相对较小
- **设计意图**：经济重要，但不是唯一目标

### 2.5 exp（经验奖励）

**计算方式**：
```python
# 经验总和（累积每级经验）
exp_sum = sum(max_exp_of_each_level[:current_level-1]) + current_exp
# 15级后关闭（经验不再增加）
if level >= 15:
    exp_reward = 0
```

**为什么要累积每级经验？**
- 升级需要越来越多的经验
- 直接用当前经验会导致"后期升级奖励太小"
- 累积经验确保每级升级的奖励大致相同

### 2.6 ep_rate（法力奖励）

**计算方式**：
```python
# 法力比例
ep_rate = current_ep / max_ep
# 仅当正增长时给奖励
if ep_rate_current > ep_rate_last:
    reward = ep_rate_current - ep_rate_last
else:
    reward = 0
```

**为什么只给正增长？**
- 法力减少是正常使用技能的代价
- 不应该惩罚放技能
- 但法力回复（自然回复或蓝buff）值得奖励

**权重0.75**：适度鼓励技能使用，但不像推塔那么重要。

### 2.7 death（死亡惩罚）

**计算方式**：
```python
# 每次死亡计数
death_count = current_death_count
# 奖励 = 当前帧 - 上一帧（零和）
reward = (my_death_current - my_death_last) - (enemy_death_current - enemy_death_last)
# 权重 -1.0
weighted_reward = reward * (-1.0)
```

**效果**：
- 我方死亡 → 奖励 = 1 × (-1.0) = -1.0（惩罚）
- 敌方死亡 → 奖励 = (-1) × (-1.0) = +1.0（奖励）

### 2.8 kill（击杀奖励）

**计算方式**：与death类似

**注意**：hok_semi中kill权重是**-0.6**（负值），这与直觉相反！

**作者解释**：
- 可能是代码bug或特殊设计
- agent_diy已修正为**+0.6**

### 2.9 last_hit（补刀奖励）

**计算方式**：
```python
# 检测小兵死亡事件
for dead_action in frame_action.dead_actions:
    if dead_action.death is soldier:
        if dead_action.killer is my_hero:
            reward += 1.0
        elif dead_action.killer is enemy_hero:
            reward -= 1.0
```

**为什么重要？**
- 补刀是经济的主要来源
- 但补刀需要精确的时机把握
- 专门奖励鼓励AI练习补刀

**已知问题**：
- hok_semi注释"当前的dead_action不完整 (bug)"
- 可能导致补刀统计不准确

### 2.10 forward（前进奖励）

**计算方式**：
```python
def calculate_forward(main_hero, main_tower, enemy_tower):
    # 距离敌方塔的距离
    dist_to_enemy_tower = distance(hero_pos, enemy_tower_pos)
    # 两塔之间的总距离
    dist_between_towers = distance(main_tower_pos, enemy_tower_pos)
    # 前进进度（0=在家，1=在敌方塔）
    progress = 1.0 - dist_to_enemy_tower / dist_between_towers
    return clip(progress, -1.0, 1.0)
```

**为什么需要forward奖励？**
- **防止"站桩"**：有些AI学会在安全位置刷兵但不敢推进
- **鼓励上线**：让AI从泉水走到前线

**时间限制**：
```python
REMOVE_FORWARD_AFTER = 1000  # 1000帧后关闭
```

**原因**：
- 前期：需要forward鼓励AI上线
- 后期：AI已经学会推进，不需要额外奖励

---

## 三、零和奖励机制

### 3.1 什么是零和？

**定义**：我方收益 = 敌方损失，总和为0。

**计算公式**：
```python
# 大多数奖励项
reward = (my_current - my_last) - (enemy_current - enemy_last)
```

**例子**（hp_point）：
- 我方血量从80%→70%（损失10%）
- 敌方血量从80%→60%（损失20%）
- 零和reward = (-10%) - (-20%) = +10%

**含义**：虽然我也损失了血量，但敌方损失更多，所以奖励为正。

### 3.2 为什么用零和？

**1v1是零和博弈**：
- 我的胜利 = 敌方的失败
- 我的推塔 = 敌方的塔被毁
- 我的经济优势 = 敌方的经济劣势

**零和奖励的优势**：
- **竞争性更强**：AI不仅要自己好，还要让敌方差
- **避免"刷分"**：不能只顾自己发育，不管敌方
- **与胜利条件对齐**：最终目标是击败敌方

### 3.3 非零和奖励项

以下奖励**不采用零和**：

| 奖励项 | 原因 |
|--------|------|
| **ep_rate** | 法力回复是自身属性，与敌方无关 |
| **forward** | 自身位置，与敌方位置无关 |
| **last_hit** | 基于事件计数，直接统计 |

---

## 四、时间衰减机制

### 4.1 公式

```python
time_scale = 0.6 ^ (frame_no / TIME_SCALE_ARG)
# TIME_SCALE_ARG = 8000
```

### 4.2 衰减曲线

| 帧数 | 时间（秒） | 衰减系数 | 含义 |
|------|-----------|---------|------|
| 0 | 0 | 1.0 | 全权重 |
| 8000 | ~267 | 0.6 | 60%权重 |
| 16000 | ~533 | 0.36 | 36%权重 |
| 20000 | ~667 | 0.22 | 22%权重 |

### 4.3 为什么需要时间衰减？

**前期（0~8000帧）**：
- 奖励权重高
- AI积极探索、发育
- 学习基础操作

**后期（8000~20000帧）**：
- 奖励权重降低
- AI更关注终局目标（推塔）
- 减少非核心行为的干扰

**类比**：
- 前期："先学会走路、吃饭、睡觉"
- 后期："现在要准备高考了，其他事先放一放"

### 4.4 哪些奖励不衰减？

```python
REWARD_WITHOUT_TIME_SCALE = {}  # 默认空，所有奖励都衰减
```

可以配置某些奖励不衰减（如death、kill等稀疏奖励），但hok_semi默认全部衰减。

---

## 五、前进奖励（Forward Reward）

### 5.1 详细计算

```python
def calculate_forward(main_hero, main_tower, enemy_tower):
    if main_hero is None or main_tower is None or enemy_tower is None:
        return 0.0
    
    # 当前血量比例
    hp_rate = current_hp / max_hp
    if hp_rate <= 0.0:
        return 0.0  # 死亡时不给forward奖励
    
    # 计算距离
    dist_hero_to_enemy_tower = distance(hero_pos, enemy_tower_pos)
    dist_between_towers = distance(main_tower_pos, enemy_tower_pos)
    
    # 前进进度（0=在家，1=在敌方塔）
    progress = 1.0 - dist_hero_to_enemy_tower / dist_between_towers
    
    return clip(progress, -1.0, 1.0)
```

### 5.2 Forward奖励的曲线

```
位置 → Forward值
我方泉水 → -1.0（惩罚在家待着）
我方塔下 → 0.0
地图中间 → 0.5
敌方塔下 → 1.0（奖励推进）
超过敌方塔 → 1.0（最大值限制）
```

### 5.3 为什么后期关闭？

```python
REMOVE_FORWARD_AFTER = 1000  # 约33秒后关闭
```

**原因**：
- 前期：AI需要学会"走出泉水"
- 后期：AI已经学会上线，不需要额外奖励
- 防止后期为了forward奖励而无意义地深入敌方腹地

---

## 六、奖励权重调参

### 6.1 hok_semi的权重设置

```python
REWARD_WEIGHT_DICT = {
    "hp_point": 2.0,         # 换血
    "tower_hp_point": 10.0,  # 推塔（最高）
    "money": 0.004,          # 经济
    "exp": 0.004,            # 经验
    "ep_rate": 0.75,         # 法力
    "death": -1.0,           # 死亡（惩罚）
    "kill": -0.6,            # 击杀（注意：负值！）
    "last_hit": 0.5,         # 补刀
    "forward": 0.01,         # 前进
}
```

### 6.2 权重调参原则

| 原则 | 说明 |
|------|------|
| **推塔最高** | 推塔是胜利条件，权重必须最高 |
| **生存次之** | 活着才能推塔，hp和death权重适中 |
| **经济基础** | money和exp提供长期收益，但权重不宜过高 |
| **探索鼓励** | forward帮助初期探索，后期可关闭 |

### 6.3 作者的经验教训

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| AI不推塔 | tower_hp权重太低 | 提高tower_hp权重 |
| AI只刷兵 | money权重太高 | 降低money权重 |
| AI站桩 | 没有forward奖励 | 添加forward奖励 |
| AI太激进 | death惩罚不够 | 加大death惩罚 |

---

## 七、常见陷阱与解决方案

### 7.1 奖励 hacking

**现象**：AI找到了奖励函数的"漏洞"
- 例：forward奖励导致AI一直在地图中间晃悠
- 例：hp奖励导致AI永远不和敌人换血

**解决方案**：
- 多种奖励组合，避免单一奖励主导
- 定期观察AI行为，发现异常及时调整
- 使用零和奖励，防止"刷分"

### 7.2 奖励稀疏

**现象**：AI长时间得不到奖励，学习缓慢
- 例：只有推塔胜利才给奖励

**解决方案**：
- 添加密集奖励（hp、money、exp）
- 添加中间目标奖励（补刀、击杀）
- 使用时间衰减，前期奖励密集

### 7.3 奖励冲突

**现象**：两个奖励互相矛盾
- 例：forward鼓励推进，但hp鼓励保守

**解决方案**：
-  carefully设计权重比例
-  使用条件奖励（如高血量时才给forward）
-  观察AI行为，调整冲突奖励

### 7.4 hok_semi的已知问题

| 问题 | 说明 | 状态 |
|------|------|------|
| kill为负值 | 可能是bug | agent_diy已修正为+0.6 |
| tower_hp过高 | 10.0可能过大 | agent_diy已改为5.0 |
| last_hit统计不全 | dead_action不完整 | agent_diy已增加诊断 |
| money/exp权重 | 作者最终建议除以2 | agent_diy用0.004，v2可调 |

---

## 总结

奖励设计是强化学习的**灵魂**。好的奖励函数能让AI快速学会正确的行为，坏的奖励函数会让AI学"歪"。

hok_semi 的奖励设计是一个**经过验证的方案**：
- **8项基础奖励**覆盖了MOBA的核心要素
- **零和机制**增强了竞争性
- **时间衰减**让AI从探索转向专注
- **forward奖励**解决了"站桩"问题

**关键认知**：奖励不是越多越好，而是要对齐最终目标（推塔胜利）。每一项奖励都应有明确的目的，且与其他奖励协调一致。

---

*下一步建议：阅读 `hok_semi_赛题对齐与差异分析.md`，了解如何将hok_semi适配到本次赛题。*

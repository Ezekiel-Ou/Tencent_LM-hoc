# 特征工程详细设计

本文档用于冻结 `agent_diy` 的特征工程 schema。目标是在开始训练前一次性设计完整特征维度，后续训练期间不再修改 `FEATURE_DIM`、`DATA_SPLIT_SHAPE`、`SAMPLE_DIM` 和模型输入切分。

设计边界：

- 只使用官方数据协议、当前 debug 日志已确认字段、以及 112/133 英雄技能的固定机制知识。
- 不把策略规则写进特征。例如不会写“鲁班二技能远距离必须释放”，只编码该技能是长距离/全图方向弹道。
- 所有变长实体使用固定数量槽位、mask、unknown/hash bucket、clip 兜底。
- 对缺失字段不虚造值：缺失时使用 `exist=0` 或字段 mask，并将数值填 0。
- 位置特征学习 `hok_semi` 的离散分桶思路，但坐标轴重新映射为长条地图坐标。

当前维度状态：

```text
DIM_POSITION       = 180  # 已按重映射坐标确认
DIM_GLOBAL         = 60   # 第 2 节当前设计
DIM_UNIT_CORE      = 206
DIM_HERO           = 693
DIM_SOLDIER        = 224
DIM_MONSTER        = 222
DIM_TOWER          = 248
DIM_BULLET         = 211
DIM_TARGET         = 30
FEATURE_DIM        = 4833  # 按当前所有块计入 target candidate
```

注意：当前 `FEATURE_DIM=4833` 是按本文所有已确认子块计算的冻结候选；实现时必须用 shape assert 再次校验 `DATA_SPLIT_SHAPE`、`SAMPLE_DIM` 和模型输入切分。

## 1. 位置特征设计

### 1.1 坐标轴重映射

地图不是轴对齐正方形，而是接近对角线方向的长条形。后续所有位置、距离、方向、塔距、血包距、子弹位置、目标候选距离都先映射到本项目坐标：

```python
lane = (x + z) / sqrt(2)
width = (x - z) / sqrt(2)
```

视角统一规则：

- 我方基地固定在 `lane < 0`，敌方基地固定在 `lane > 0`。
- 红方视角必须镜像。实现时可以先对原始 `x,z` 取反再投影，也可以投影后同时对 `lane,width` 取反；全项目必须只采用一种实现。
- 不再把原始 `x/z` 直接作为模型输入。

已冻结地图尺度：

| 区域 | lane 半长 | width 全宽 | width 半宽 | 用途 |
| --- | ---: | ---: | ---: | --- |
| 全局长轴 | `45000` | - | - | 基地到基地的主进攻轴 |
| 常规区域 | - | `14000` | `7000` | 主兵线、塔区、大部分粗略位置 |
| 中心精细区 | `15000` | `20000` | `10000` | 两个血包之间的核心交战区 |

注意：`14000` 和 `20000` 都是全宽，不是半宽。连续归一化时使用半宽 `7000/10000`。

中心区判定：

```python
is_center_core = abs(lane) <= 15000
width_half = 10000 if is_center_core else 7000
```

异常坐标兜底：

- `location` 缺失：位置特征全 0，由上层实体 `exist/visible` 表达不可用。
- `abs(x) > 60000` 或 `abs(z) > 60000`：视为协议异常或哨兵坐标，位置特征全 0，不参与距离排序。
- 投影后的连续值统一 clip 到 `[-1, 1]`。
- 超出 `lane=45000` 或当前 `width_half` 的点不扩大尺度，进入边界/兜底桶，避免训练中尺度漂移。

### 1.2 `DIM_POSITION = 180`

每个可定位对象使用同一套位置特征。该块复用 `hok_semi` 的“离散 one-hot + 连续 ratio/残差 + 边界桶”思想，但分桶轴完全换成本项目的 `lane/width`。

| 子块 | 维度 | 设计 |
| --- | ---: | --- |
| 中心精细 lane one-hot | 63 | `lane` 在 `[-15000, 15000]` 上以 `500` 为单位分桶，并保留边界/兜底桶 |
| 中心精细 width one-hot | 83 | `width` 在 `[-10000, 10000]` 上以 `250` 为单位分桶，并保留边界/兜底桶 |
| 全局 lane one-hot | 21 | `lane` 在 `[-45000, 45000]` 上以 `5000` 为单位分桶，并保留边界/兜底桶 |
| 全局 width one-hot | 7 | `width` 在 `[-7000, 7000]` 上以 `3500` 为单位分桶，并保留边界/兜底桶 |
| continuous/residual | 6 | `center_lane_norm`, `center_width_norm`, `global_lane_norm`, `global_width_norm`, `center_dist_ratio`, `global_dist_ratio` |
| 合计 | 180 | 全项目统一使用的位置编码 |

设计理由：

- 中心精细分桶服务血包争夺、中路交战、外塔附近横向走位和短距离技能命中。
- 全局分桶服务推塔、回补、全图/长距离技能和前后压制节奏。
- 鲁班二技能这类长距离/全图技能不要求中心窗口覆盖全图；它依赖全局 `lane/width`、目标候选距离、技能 range class 和合法动作共同表达。
- 坐标兜底不在特征尺度里吸收异常值，异常点统一 mask/clip，避免一次协议异常污染归一化范围。

## 2. 全局特征：`DIM_GLOBAL = 60`

全局特征只放对整帧决策有用、与具体实体槽位无关或需要全局对称比较的信息。塔坐标、血包坐标、单位位置、子弹位置等仍进入后续实体/资源锚点特征，不在全局块里重复编码。

血包不放在全局块。原因是血包是固定地图锚点，是否可争夺取决于我方/敌方英雄到血包的重映射距离、塔区压力和血包是否存在；这些应在后续资源锚点或塔区资源特征中按固定锚点编码，而不是压缩成一个全局符号。

| 子块 | 维度 | 字段来源 | 设计 |
| --- | ---: | --- | --- |
| 英雄 ID | 4 | self/enemy `config_id` | self: `112/133` 2 维，enemy: `112/133` 2 维；不设 unknown |
| 阵营/镜像 | 2 | `player_camp`, mirror flag | `self_camp_is_1`, `is_mirrored` |
| 时间节奏 | 10 | `frame_no` | `frame_ratio` 1，阶段 one-hot 4，关键节点倒计时 ratio 3，timeout pressure 1，炮车期 flag 1 |
| 目标塔血 | 6 | tower/crystal `hp/max_hp` | 我塔血比、敌塔血比、塔血差、我水晶血比、敌水晶血比、水晶血差 |
| 英雄血量与塔压 | 10 | hero hp、tower range/target、remapped distance | 双方 hp 比与差值、双方是否在敌塔范围、双方是否被塔锁定、双方到进攻塔距离 ratio、塔压差 |
| 等级/经济 | 12 | hero `level`, `exp`, `moneyCnt/money` | 双方等级、等级差、双方经验 ratio、双方总经济、经济差、双方帧经济、双方经济增量 |
| legal action 粗摘要 | 16 | `legal_action`, `sub_action_mask` | 主按钮 12 维可用 mask，目标类可用摘要 4 维 |

全局块的维度顺序固定如下，后续实现必须按这个顺序写入：

| 索引范围 | 维度 | 名称 | 具体含义 |
| --- | ---: | --- | --- |
| `0:2` | 2 | self hero id | `[self_is_112, self_is_133]` |
| `2:4` | 2 | enemy hero id | `[enemy_is_112, enemy_is_133]` |
| `4:6` | 2 | camp/mirror | `[self_camp_is_1, is_mirrored]` |
| `6:16` | 10 | time | 见 2.1 |
| `16:22` | 6 | tower/crystal hp | `[self_tower_hp, enemy_tower_hp, tower_hp_diff, self_crystal_hp, enemy_crystal_hp, crystal_hp_diff]`，均为 clipped ratio |
| `22:32` | 10 | hero pressure | `[self_hp, enemy_hp, hp_diff, self_in_enemy_tower_range, enemy_in_self_tower_range, self_targeted_by_tower, enemy_targeted_by_tower, self_to_enemy_tower_dist, enemy_to_self_tower_dist, tower_pressure_diff]` |
| `32:44` | 12 | level/economy | `[self_level, enemy_level, level_diff, self_exp, enemy_exp, exp_diff, self_money_total, enemy_money_total, money_diff, self_money_frame, enemy_money_frame, money_delta_diff]` |
| `44:60` | 16 | legal summary | 主按钮 12 维 + 目标类 4 维 |

数值规范：

- `*_hp` 使用 `hp / max_hp`，缺失时填 0。
- `*_diff` 使用 `self - enemy`，再 clip 到 `[-1, 1]`。
- `level` 使用 `level / 15`，`level_diff` 使用 `(self_level - enemy_level) / 15`。
- `exp` 如果能拿到当前等级进度则用等级内进度；否则用 `exp / max_observed_exp` 并 clip。
- `money_total` 使用 `money_total / 15000`，`money_frame` 使用 `money_frame / 2000`，`money_delta_diff` 使用 `(self_delta - enemy_delta) / 500`，均 clip 到 `[-1, 1]`。
- `*_to_*_tower_dist` 使用重映射后的欧氏距离，除以 `45000` 后 clip 到 `[0, 1]`。
- `tower_pressure_diff = enemy_in_self_tower_range - self_in_enemy_tower_range`，范围 `[-1, 1]`。
- 目标类 4 维 legal 摘要为 `[enemy_hero_target_available, self_target_available, any_enemy_soldier_available, enemy_tower_or_monster_available]`。

### 2.1 时间节奏 10 维

时间节点使用 `hok_semi` 思路和当前 debug 观察到的关键帧作为量级参考：

```text
river crab / 初期资源节点：920
cake / 血包节点：1778
cannon / 炮车节奏节点：6254
```

阶段 one-hot：

| 阶段 | frame 范围 | 目的 |
| --- | --- | --- |
| 抢线期 | `[0, 920)` | 兵线和一级对拼 |
| 初期资源期 | `[920, 1778)` | 河道/血包前后的节奏 |
| 塔压过渡期 | `[1778, 6254)` | 外塔消耗、血包循环、等级经济差扩大 |
| 炮车/后期 | `[6254, end]` | 炮车推进和终局塔血压力 |

关键节点倒计时 ratio 使用 `(target_frame - frame_no) / target_frame` 后 clip 到 `[-1, 1]`。`frame_ratio` 使用配置中的最大局帧数；缺失时暂按 `20000` 兜底。

### 2.2 经济与召唤师技能可观测性

最新 debug 已确认经济字段可按 `hero + camp` 获取：

- `level`
- `exp`
- `moneyCnt / money_total`
- `money / money_frame`
- `money_delta`

因此全局可以放心做双方经济差、经验差和近期经济增量，不需要只依赖击杀或补刀事件推断。

召唤师技能也已确认能从 `skill_state.slot_states` 获取。由于英雄特征的 6 个技能槽已经包含 `SLOT_SKILL_5`，全局块不再重复放 24 维召唤师摘要；这里只保留候选集合和实现约束：

- `config_id`
- `cooldown`
- `cooldown_max`
- `usable`
- `level`

正式特征只白名单六个候选：

| ID | 技能 | tag |
| ---: | --- | --- |
| 80102 | 治疗术 | heal |
| 80105 | 干扰 | tower_interference |
| 80107 | 净化 | cleanse |
| 80110 | 狂暴 | combat_buff |
| 80115 | 闪现 | mobility, instant_reposition |
| 80109 | 疾跑 | mobility, speed_buff |

static tags 是数值类型先验，不是策略规则，也不是自然语言语义注入。模型不会“知道闪现两个字”，但会同时看到 `mobility/instant_reposition`、CD、可用状态、距离、塔压和后续奖励，从轨迹里学习这个技能槽的效果。后续训练采样也只在这六个召唤师技能中选择；正式比赛技能选择再根据训练效果决定。

## 3. 通用单位核心：`DIM_UNIT_CORE = 206`

英雄、小兵、野怪、防御塔共享：

| 子块 | 维度 | 字段来源 | 价值 |
| --- | ---: | --- | --- |
| exist / visible | 2 | 是否存在、`camp_visible` 或位置是否有效 | 区分未观测与真实 0 |
| camp relation | 3 | `camp` | self/enemy/neutral_or_unknown |
| hp state | 16 | `hp`, `max_hp` | 生存状态、斩杀线、推塔/补刀价值 |
| attack target type | 5 | `attack_target` + runtime_id 映射 | 谁正在攻击谁 |
| position | 180 | `location` | 局部和全局位置 |
| 合计 | 206 | - | 通用单位核心当前冻结设计 |

### 3.1 exist / visible 2 维

```text
[exist, visible]
```

- `exist=1`：该槽位有实体对象；padding 槽为 0。
- `visible=1`：位置有效且不是异常坐标；如果实体存在但坐标缺失或异常，`exist=1, visible=0`。

### 3.2 camp relation 3 维

协议字段为：

```text
camp: int，所属阵营，蓝方=1，红方=2
```

特征不直接把蓝/红当作战术含义，而是结合当前智能体阵营转换成相对关系：

```text
[is_self_camp, is_enemy_camp, is_neutral_or_unknown]
```

转换规则：

- `camp == self_camp`：`[1, 0, 0]`
- `camp in {1, 2} and camp != self_camp`：`[0, 1, 0]`
- 野怪、无阵营、缺失或未知：`[0, 0, 1]`

蓝红原始阵营只在全局 `self_camp_is_1/is_mirrored` 中保留；单位特征只表达相对敌我关系。

### 3.3 hp state 16 维

不再照搬 `hok_semi` 的 27 维血量桶。当前 debug 已确认的量级：

- 英雄最大血量约 `6000+`
- 防御塔最大血量 `12000`
- 水晶最大血量 `6000`
- 泉水最大血量 `9000`
- 小兵/野怪血量低于塔，但随类型变化

因此通用单位血量使用“比例 + 绝对量级 + 低血线”的组合，保证英雄、小兵、野怪、塔都可用：

| 索引 | 维度 | 名称 | 具体含义 |
| --- | ---: | --- | --- |
| `0` | 1 | hp_ratio | `hp / max_hp`，缺失或 `max_hp<=0` 时为 0 |
| `1` | 1 | max_hp_ratio | `max_hp / 12000`，clip 到 `[0, 1]` |
| `2` | 1 | alive | `hp > 0` |
| `3` | 1 | hp_known | `hp` 和 `max_hp` 都有效 |
| `4:10` | 6 | hp_ratio_bucket | one-hot: `0`, `(0,0.2]`, `(0.2,0.4]`, `(0.4,0.6]`, `(0.6,0.8]`, `(0.8,1]` |
| `10:15` | 5 | hp_abs_bucket | one-hot: `<=500`, `<=1500`, `<=3000`, `<=6000`, `>6000` |
| `15` | 1 | low_hp_flag | `0 < hp_ratio <= 0.25` |

数值规范：

- 所有 ratio 均 clip 到 `[0, 1]`。
- `hp_abs_bucket` 使用当前 `hp`，不是 `max_hp`；`>6000` 主要覆盖后期英雄、泉水、防御塔等高血量对象。
- 对 `exist=0` 的 padding 槽，16 维全部填 0。
- 对存在但血量字段缺失的实体，`hp_known=0`，其他血量数值填 0。

### 3.4 attack target type 5 维

`attack_target` 通过当前帧 runtime id 映射到实体类型：

```text
[none_or_unknown, target_self_hero, target_enemy_hero, target_soldier, target_organ_or_monster]
```

如果协议字段缺失、target runtime id 不在当前帧索引中，进入 `none_or_unknown`。

## 4. 英雄特征：`DIM_HERO = 693`，共 2 个

顺序：

1. self hero
2. enemy hero

| 子块 | 维度 | 字段来源 | 价值 |
| --- | ---: | --- | --- |
| unit core | 206 | Hero 通用字段 | 位置、血量、阵营关系、攻击目标 |
| hero id / relation | 4 | `config_id`, 槽位顺序 | 112/133 条件化、自身/敌方槽位 |
| behavior one-hot | 9 | `behav_mode` | 稳定编码行为枚举，不依赖语义猜测 |
| ep state | 10 | `ep`, `max_ep` | 技能资源状态，按最大 `1500` 缩放 |
| combat attributes | 16 | `attack_range`, `phy_atk`, `phy_def`, `mov_spd`, `atk_spd` | 只保留攻击、防御、移速、攻速、攻击范围等关键对拼属性 |
| skill slots | 240 | `skill_state.slot_states` | 6 个槽位，每槽 40 维，包含恢复技能和召唤师技能 |
| level | 15 | `level` | `1..15` one-hot |
| money | 18 | `moneyCnt/money`，上一帧差分 | 延续 `hok_semi` 的经济分桶思路 |
| grass | 1 | `is_in_grass` | 视野/伏击 |
| tower relation | 6 | 英雄到敌/我塔距离，塔 `attack_target` | 塔下风险 |
| hero buff skills | 136 | `buff_state.buff_skills` | 112/133 被动和技能状态主要来源 |
| hit/take-hurt/recent events | 24 | `hit_target_info`, `take_hurt_infos`, `frame_action` | 最近战斗反馈 |
| retreat / recall context | 8 | hp、敌方距离、base anchor、button 9 legal | 残血回城、回撤、吃血包前的安全判断 |

### 4.1 hero id / relation 4 维

```text
[is_112, is_133, is_self_slot, is_enemy_slot]
```

- `is_112/is_133` 来自 `config_id`。
- `is_self_slot/is_enemy_slot` 来自固定槽位顺序，而不是再重复使用 `camp`。
- unknown 不单独设维度；本任务英雄固定为 `112/133`。

### 4.2 behavior one-hot 9 维

只编码日志中已观测到的行为枚举，不强行解释语义：

```text
[behav=0, behav=1, behav=2, behav=4, behav=9, behav=10, behav=23, behav=27, other]
```

如果后续日志出现新值，进入 `other`，不扩维。

### 4.3 ep state 10 维

法力/能量按最大 `1500` 建模：

| 索引 | 维度 | 名称 | 具体含义 |
| --- | ---: | --- | --- |
| `0` | 1 | ep_ratio | `ep / 1500`，clip 到 `[0,1]` |
| `1` | 1 | max_ep_ratio | `max_ep / 1500`，clip 到 `[0,1]` |
| `2` | 1 | ep_known | `ep` 或 `max_ep` 字段有效 |
| `3` | 1 | uses_ep | `max_ep > 0` |
| `4:10` | 6 | ep bucket | one-hot: `0`, `(0,0.2]`, `(0.2,0.4]`, `(0.4,0.6]`, `(0.6,0.8]`, `(0.8,1]` |

如果英雄实际不消耗蓝，`uses_ep=0`，其他 ratio 仍按协议字段填 0。

### 4.4 combat attributes 16 维

只保留对 1v1 决策最直接的属性：

| 索引 | 维度 | 名称 | 归一化 |
| --- | ---: | --- | --- |
| `0` | 1 | attack_range | `attack_range / 12000` |
| `1` | 1 | phy_atk | `phy_atk / 1000` |
| `2` | 1 | phy_def | `phy_def / 1000` |
| `3` | 1 | mov_spd | `mov_spd / 1000` |
| `4` | 1 | atk_spd | `atk_spd / 10000` |
| `5` | 1 | hp_recover | `hp_recover / 500` |
| `6` | 1 | attack_range_known | 字段是否有效 |
| `7` | 1 | combat_attr_known | 关键属性是否至少有一个有效 |
| `8:12` | 4 | attack_range bucket | `<=1000`, `<=3000`, `<=8800`, `>8800` |
| `12:16` | 4 | phy_atk bucket | `<=150`, `<=300`, `<=600`, `>600` |

法攻、法防、暴击、吸血、冷却缩减等字段不进入当前主英雄特征；若后续确认变化显著，可占用 reserved 或进入专项特征，但不作为初始核心。

### 4.5 技能槽：每槽 `40`，共 `6 * 40 = 240`

固定槽位：

1. 普攻/基础攻击槽：`SLOT_SKILL_0`
2. 一技能：`SLOT_SKILL_1`
3. 二技能：`SLOT_SKILL_2`
4. 三技能：`SLOT_SKILL_3`
5. 恢复技能：`SLOT_SKILL_4`
6. 召唤师技能：`SLOT_SKILL_5`

`SLOT_SKILL_6` 不作为固定主槽，进入 unknown/other 统计或 recent event，避免把系统槽写死。

每槽 40 维：

| 子块 | 维度 | 字段来源 | 具体含义 |
| --- | ---: | --- | --- |
| slot identity | 6 | `slot_type` | `[attack, skill1, skill2, skill3, recover, summoner]` |
| config identity | 7 | `configId` | `[is_112_related, is_133_related, is_recover_config, is_summoner_known, is_summoner_80115, is_summoner_80109, config_other]` |
| level / usable / CD | 12 | `level`, `usable`, `cooldown`, `cooldown_max` | 当前是否学会、是否可用、CD 状态 |
| usage feedback | 5 | `usedTimes`, `hitHeroTimes`, `succUsedInFrame`, 最近帧缓存 | 是否刚释放、命中、被打断 |
| next / combo | 3 | `nextConfigID`, `comboEffectTime` | 多段技能或组合状态 |
| static type tags | 7 | 固定技能表 | 技能类型先验，不是策略规则 |

level / usable / CD 12 维：

```text
level_ratio            = level / 15
usable                 = bool(usable)
cooldown_ratio         = cooldown / max(cooldown_max, 1)
cooldown_max_ratio     = cooldown_max / 120000
cd_known               = cooldown 字段有效
learned_or_available   = level > 0 or usable
cd_bucket 6维          = 0, <=1s, <=3s, <=6s, <=15s, >15s
```

usage feedback 5 维：

```text
used_this_frame        = succUsedInFrame > 0
used_times_ratio       = clip(usedTimes / 20, 0, 1)
hit_hero_times_ratio   = clip(hitHeroTimes / 20, 0, 1)
recent_hit_flag        = 最近若干帧 hit_target_info 命中过英雄/塔/小兵
recent_interrupted     = 最近使用恢复技能或血包后受到攻击，且恢复未形成有效 hp 增量
```

`recent_interrupted` 专门覆盖恢复技能和血包被攻击打断的问题。恢复技能在本英雄技能槽中编码；血包属于后续资源锚点特征，但两者共享最近受击/恢复尝试缓存。

next / combo 3 维：

```text
has_next_config        = nextConfigID > 0
combo_active           = comboEffectTime > 0
combo_time_ratio       = clip(comboEffectTime / 10000, 0, 1)
```

鲁班释放技能后立即扫射，不单靠一个“策略规则”表达：技能槽里用 `passive_attack_trigger`、`combo/next`、`succUsedInFrame`，英雄状态里再用 `buff_skills` 捕捉强化普攻/被动相关 ID。

static type tags 7 维：

```text
[damage, mobility, recover_or_heal, cleanse_or_defense, tower_interference, long_or_global, passive_attack_trigger]
```

该 metadata 是数值类型先验，不是策略规则，也不是自然语言语义注入。模型不会“知道闪现两个字”，但会同时看到 `mobility`、CD、可用状态、距离、塔压和后续奖励，从轨迹里学习该槽位效果。

### 4.6 技能 metadata

鲁班七号 `112`：

| 槽位 | static type tags |
| --- | --- |
| 普攻 | `damage` |
| 一技能 | `damage`, `passive_attack_trigger` |
| 二技能 | `damage`, `long_or_global`, `passive_attack_trigger` |
| 三技能 | `damage`, `passive_attack_trigger` |
| 恢复技能 | `recover_or_heal` |

狄仁杰 `133`：

| 槽位 | static type tags |
| --- | --- |
| 普攻 | `damage` |
| 一技能 | `damage` |
| 二技能 | `cleanse_or_defense` |
| 三技能 | `damage` |
| 恢复技能 | `recover_or_heal` |

召唤师技能只保留当前 1v1 训练候选集合：

| ID | 技能 | static type tags |
| ---: | --- | --- |
| 80102 | 治疗术 | `recover_or_heal` |
| 80105 | 干扰 | `tower_interference` |
| 80107 | 净化 | `cleanse_or_defense` |
| 80109 | 疾跑 | `mobility` |
| 80110 | 狂暴 | `damage` |
| 80115 | 闪现 | `mobility` |

### 4.7 其他英雄子块

level 15 维：

```text
level 1..15 one-hot
```

money 18 维：延续 `hok_semi` 的经济设计思路，使用总经济、当前帧经济、上一帧差分和对应 bucket；具体分母按当前全局节中的 `15000/2000/500` 规则，英雄自身只编码本槽位经济，敌我差已在 global 编码。

tower relation 6 维：

```text
dist_to_enemy_tower_ratio
dist_to_self_tower_ratio
in_enemy_tower_range
in_self_tower_range
targeted_by_enemy_tower
targeted_by_self_tower
```

塔攻击范围缺失时使用已确认 `8800`。距离使用重映射后的 `lane/width` 欧氏距离。

hit/take-hurt/recent events 24 维：编码最近若干帧是否命中敌方英雄/小兵/塔、是否受到英雄/塔/小兵伤害、最近伤害来源 slot、最近承伤量 ratio、最近命中量 ratio，以及恢复/血包尝试是否被打断。

retreat / recall context 8 维：

```text
low_hp_recall_need        = hp_ratio <= 0.30
critical_hp_flag          = hp_ratio <= 0.15
enemy_near_threat         = enemy_dist <= 8800
recent_damage_flag        = 最近若干帧受到英雄/塔/小兵伤害
recall_button_legal       = legal_action[button=9]
dist_to_self_base_ratio   = dist(self_hero, self_base_anchor) / 45000
base_direction_lane_sign  = self_base_lane - self_lane 的符号/归一化
safe_recall_context       = low_hp 且 enemy_near_threat=0 且 recent_damage_flag=0
```

该块不写“残血必须回城”的策略规则，只给模型回城动作所需的可观测上下文。`button=9` 已由 debug/action 约定为 recall；如果平台 legal mask 禁止回城，`recall_button_legal=0`，模型仍受合法动作约束。

## 5. 英雄 buff skills：`DIM_HERO_BUFF = 136`

当前 debug 反复显示 `buff_marks` 为空，而 `buff_skills` 有大量 `112xxx/133xxx`，因此 `buff_skills` 是主状态入口。

| 子块 | 维度 | 设计 | 价值 |
| --- | ---: | --- | --- |
| explicit whitelist | 96 | 已观测 ID + 当前配置候选 | 捕获常见状态 |
| 112 family | 8 | `1120xx/1121xx/1122xx/1123xx/1128xx/1129xx/other/unknown` | 鲁班被动和技能状态 |
| 133 family | 8 | `1330xx/1331xx/1332xx/1333xx/1339xx/other/unknown/reserved` | 狄仁杰被动和技能状态 |
| common/system family | 8 | `100xx/110xx/900xx/911xx/914xx/500xxx/other/unknown` | 回复、净化、通用状态 |
| unknown hash | 8 | `hash(buff_id) % 8` | 新 ID 不丢失 |
| count/delta | 8 | buff 数量、hero-specific 数量、出现/消失计数 | 技能后摇、鲁班扫射、恢复/净化/狂暴等短暂状态变化 |

explicit whitelist 初始来源：

- 当前 debug 日志：`10000, 10010, 10014, 11001, 11002, 11010, 50000, 90015, 90019, 90110, 500009, 911260, 911290, 912330, 912350, 914110, 914210, 914211, 914230, 914232, 919900, 167602, 131956`
- 鲁班：`112000, 112001, 112010, 112015, 112020, 112025, 112030, 112035, 112040, 112041, 112042, 112043, 112044, 112045, 112046, 112047, 112048, 112100, 112200, 112201, 112210, 112300, 112301, 112320, 112890, 112910, 112990, 112991`
- 狄仁杰：`133000, 133001, 133010, 133011, 133020, 133090, 133100, 133200, 133250, 133260, 133300, 133310, 133320, 133330, 133350, 133390, 133950, 133951`
- 当前 `agent_diy.conf.conf` 已有 common/112/133 候选

若 ID 超过 96 个，不扩维：按“日志已出现 + 英雄相关 + 通用系统”的优先级进入 whitelist，其余走 family/hash。

## 6. 小兵特征：`DIM_SOLDIER = 224`，共 6 个

顺序：

- 我方小兵 3 个
- 敌方小兵 3 个

排序：

- 敌方小兵优先对齐动作 target slot `3-5`；第 6 个动作 target 槽若存在更多敌方小兵，只通过 target candidate 的 legal/mask 表达，不再单独进入小兵实体池。缺失目标顺序时按“lane 进攻方向 + 距离我方英雄”兜底。
- 我方小兵不进入动作 target 头，按“lane 进攻方向 + 距离我方英雄”排序：优先保留交战前线和即将进塔/出塔的小兵。
- 不采用单纯“距离我方英雄”作为最终排序，因为它会在补刀和推塔场景下把身后的低价值小兵排到前面。

| 子块 | 维度 | 字段来源 | 价值 |
| --- | ---: | --- | --- |
| unit core | 206 | NPC 通用字段 | 位置、血量、攻击目标 |
| soldier type/config | 4 | `config_id`, `(actor_type=1, sub_type=11)` | 近战/远程/炮车/unknown |
| behavior | 4 | `behav_mode` | 死亡、行进、攻击、unknown |
| tower relation | 4 | 到敌/我塔距离、塔锁定 | 塔下兵线判断 |
| target slot relation | 4 | target index | 与动作目标对齐 |
| last-hit/income context | 2 | `kill_income`, hp bucket | 补刀收益 |

当前维度校验：`206 + 4 + 4 + 4 + 4 + 2 = 224`。

注意：当前项目必须使用已确认映射，小兵是 `(actor_type=1, sub_type=11)`，不能沿用旧代码里的 `sub_type=1` 假设。

## 7. 野怪特征：`DIM_MONSTER = 222`

同一时间只编码 1 只最相关野怪。优先级：

1. 动作 target slot 对应的野怪。
2. 距离我方英雄最近的可见野怪。
3. 若无可见野怪，则整块填 0。

| 子块 | 维度 | 字段来源 | 价值 |
| --- | ---: | --- | --- |
| unit core | 206 | NPC 通用字段 | 位置、血量、攻击目标 |
| monster type/config | 4 | `config_id`, `(actor_type=1, sub_type=0)` | 河道野怪/其他野怪/unknown/hash |
| behavior | 4 | `behav_mode` | 已观测行为/死亡/unknown |
| target slot relation | 2 | target index 8 | 是否对应动作目标、是否可被当前动作 target |
| income/value | 6 | `kill_income`, `dead_action.income_info`, hp | 野怪含金量、是否值得补刀/争夺 |

当前维度校验：`206 + 4 + 4 + 2 + 6 = 222`。

income/value 6 维：

```text
kill_income_ratio       = clip(kill_income / 300, 0, 1)
income_money_ratio      = clip(dead_action.income_info.money / 300, 0, 1)
income_exp_ratio        = clip(dead_action.income_info.exp / 300, 0, 1)
low_hp_last_hit_flag    = 0 < hp_ratio <= 0.25
valuable_monster_flag   = income_money_ratio > 0 or income_exp_ratio > 0
contested_flag          = self_dist_to_monster <= 12000 and enemy_dist_to_monster <= 12000
```

不额外设计 `distance context`。野怪距离已由 `unit core.position` 提供，目标候选特征也会再次编码目标距离；在野怪块中重复双方英雄/塔/中心距离收益不高。

## 8. 防御塔特征：`DIM_TOWER = 248`，共 2 个

顺序：

1. self tower
2. enemy tower

| 子块 | 维度 | 字段来源 | 价值 |
| --- | ---: | --- | --- |
| unit core | 206 | NPC 通用字段 | 位置、血量、攻击目标 |
| fixed tower anchor | 12 | 已确认外塔坐标，重映射后编码 | 即使塔实体暂不可见，也能稳定提供塔区几何 |
| organ type | 4 | sub_type `21/23/24/unknown` | 塔/水晶/泉水识别 |
| range/sight | 6 | `attack_range`, `sight_area` | 塔压制范围 |
| aggro context | 12 | `attack_target` runtime_id + 最近攻击事件 | 是否锁英雄/小兵，以及敌塔下攻击敌方英雄的仇恨风险 |
| cake state | 6 | `cakes` + 固定血包坐标 | 塔下补给 |
| tower pressure/progress | 8 | tower hp、我方英雄距离 | 推塔进度 |

当前维度校验：`206 + 12 + 4 + 6 + 12 + 6 + 8 = 248`。

### 8.1 fixed tower anchor 12 维

外塔坐标已确认，不再依赖 debug 继续观测。进入特征前必须先做第 1 节的 `lane/width` 重映射和视角镜像。

固定锚点：

```text
self_outer_tower_lane  ≈ -13000
self_outer_tower_width ≈ 0
enemy_outer_tower_lane ≈  13000
enemy_outer_tower_width≈ 0
```

如果实现保留原始协议坐标，应先把已确认外塔原始坐标转换成上述视角统一后的 `lane/width`，最终特征只使用 `lane/width`。

每个塔槽的 12 维：

```text
anchor_lane_norm          = anchor_lane / 45000
anchor_width_norm         = anchor_width / 7000
self_hero_to_anchor_dist  = dist(self_hero, anchor) / 45000
enemy_hero_to_anchor_dist = dist(enemy_hero, anchor) / 45000
anchor_is_enemy_side      = 1 if this slot is enemy tower else 0
anchor_is_self_side       = 1 if this slot is self tower else 0
hero_between_towers       = self_hero_lane between self/enemy tower lanes
enemy_between_towers      = enemy_hero_lane between self/enemy tower lanes
lane_forward_ratio        = clipped progress from self tower to enemy tower
width_abs_ratio           = abs(self_hero_width - anchor_width) / 10000
enemy_width_abs_ratio     = abs(enemy_hero_width - anchor_width) / 10000
anchor_valid              = 1
```

### 8.2 aggro context 12 维

塔仇恨不仅来自 `attack_target`，还取决于塔下攻击规则。关键规则：

```text
在敌方防御塔攻击范围内攻击敌方英雄，会触发敌方塔锁定我方英雄。
```

该规则不直接写成策略动作，而是编码成风险上下文：

```text
target_none_or_unknown
target_self_hero
target_enemy_hero
target_self_soldier
target_enemy_soldier
target_other
self_hero_in_this_tower_range
enemy_hero_in_this_tower_range
self_hero_targeted_by_this_tower
enemy_hero_targeted_by_this_tower
self_recent_hit_enemy_hero_under_enemy_tower
enemy_recent_hit_self_hero_under_self_tower
```

其中：

- `*_in_this_tower_range` 使用塔攻击范围，运行时字段优先，缺失时使用 `8800`。
- `*_targeted_by_this_tower` 来自 `attack_target == hero.runtime_id`。
- `self_recent_hit_enemy_hero_under_enemy_tower` 来自最近帧 `hit_target_info/real_cmd/take_hurt_infos` 缓存：我方英雄最近攻击敌方英雄，且我方英雄处于敌塔 `8800` 范围内。
- 对称地，`enemy_recent_hit_self_hero_under_self_tower` 表达敌方在我塔下攻击我方英雄的风险。

防御塔攻击范围：

- 运行时字段优先。
- 缺失时使用已确认 `8800`。
- 不使用水晶/泉水范围替代塔范围。

### 8.3 cake state 6 维

血包作为塔区资源锚点处理，不进入全局块，也不进入 target 头。它通过固定位置和 `cakes` 是否存在影响移动、回撤和续航决策。

固定锚点使用重映射后的统一视角：

```text
self_cake_lane  ≈ -15000
self_cake_width ≈ 0
enemy_cake_lane ≈  15000
enemy_cake_width≈ 0
```

每个塔槽的 6 维：

```text
cake_exists              = 当前 cakes 中该侧血包是否存在
self_hero_to_cake_dist   = dist(self_hero, cake_anchor) / 45000
enemy_hero_to_cake_dist  = dist(enemy_hero, cake_anchor) / 45000
cake_need_by_hp          = self_hp_ratio <= 0.50
cake_safe_to_take        = enemy_hero_to_cake_dist > self_hero_to_cake_dist and not self_in_enemy_tower_range
cake_respawn_ratio       = 若刚被吃掉，用 75s 刷新周期估计；未知填 0
```

吃血包不是离散 target 动作，模型通过移动头走向血包锚点；`cake_exists`、距离、残血需求和塔区风险共同提供学习信号。受到攻击导致恢复/血包被打断的信息进入英雄 `recent_interrupted`。

### 8.4 tower pressure/progress 8 维

```text
tower_hp_ratio
tower_low_hp_flag
self_hero_in_attack_range
enemy_hero_in_attack_range
self_can_attack_tower_range
self_is_targeted_by_tower
soldier_or_other_targeted_by_tower
push_value_context
```

`self_can_attack_tower_range` 使用我方英雄 `attack_range` 与敌塔距离估计；`push_value_context` 综合敌塔血量低、我方小兵在塔下、我方未被塔锁定、敌方英雄不近身等上下文。它不是“必须推塔”的规则，而是让模型能区分安全磨塔、带兵进塔和裸进塔被打的风险。

## 9. 子弹特征：`DIM_BULLET = 211`，共 5 个

子弹协议字段已确认可采集：

```text
runtime_id, camp, source_actor, slot_type, skill_id, location
```

debug 代码已经统计 `bullet_slot_types_seen`、`bullet_skill_ids_seen`、`bullet_source_kind_counts`、`bullet_source_actor_samples`、`bullet_slot_to_skill_mapping`、`bullet_camp_slot_counts`，并记录 bullet 坐标样本。当前日志能证明子弹信息存在，且能通过 `source_actor -> runtime_actor_index` 追溯来源是 hero / tower / soldier / monster。

重要限制：

- bullet `skill_id` 不能作为主要语义来源。`hok_semi` 注释中也说明该字段基本无用，我们当前日志里也主要看到 `skill_id=0`。
- 当前多轮 debug 中 `bullet_skill_ids=[0]`，`bullet_slot_to_skill` 也全部映射到 0，因此正式特征不为 `skill_id` 分配独立语义维度；仅保留一个 `skill_id_nonzero_or_unknown` 兜底位。
- 当前日志中 `slot_type` 主要为 `1/2/3`，没有稳定看到普攻 bullet，但仍保留 `slot_type=0` 和 `tower/basic_or_valid` 兜底，避免平台版本或未覆盖对局出现普攻/塔弹时直接丢失信息。
- 鲁班扫射不单独作为特殊子弹类型硬编码。它应主要通过“敌方英雄来源 + 普攻/技能槽位 + 多个连续英雄子弹 + 位置/速度逼近 + 敌方英雄 buff/技能释放状态”共同表达。其他可躲技能只要在协议里表现为 bullet，也走同一套弹道威胁建模。
- 我方子弹不进入主子弹池。自己的释放、命中和扫射状态已经在英雄技能槽、buff skills、hit/take/recent 事件、目标候选和 reward 中编码；子弹池优先服务规避敌方威胁。

排序优先级：

1. 敌方英雄/unknown 威胁子弹 4 个：按威胁分排序，而不是只按距离。
2. 敌方防御塔子弹 1 个：取距离我方英雄最近的塔弹。

敌方英雄子弹威胁分：

```text
threat = 3 * approaching_self
       + 2 * source_is_enemy_hero_or_unknown
       + 2 * near_self
       + 1 * high_speed
       + 1 * recent_enemy_hero_burst_context
```

其中 `recent_enemy_hero_burst_context` 不从 bullet 自身虚构语义，而来自敌方英雄块中的英雄 ID、最近技能释放、`slot_type`、`succUsedInFrame`、buff skills 变化、以及连续 bullet 数量。鲁班扫射、狄仁杰连续普攻或技能弹道都走同一套上下文。若 `source_actor` 暂时无法映射到单位类型，但 `camp` 属于敌方且子弹接近我方英雄，仍作为 unknown 威胁子弹保留，避免丢掉视野边缘或来源索引缺失的弹道。

单个 bullet 211 维：

| 子块 | 维度 | 字段来源 | 具体含义 |
| --- | ---: | --- | --- |
| exist | 1 | bullet 是否存在 | 空槽 mask |
| source relation | 3 | bullet `camp` | enemy / self / unknown；实际排序只保留 enemy，self 作为兜底 |
| source kind | 5 | `source_actor` 映射 | hero / tower / soldier / monster / unknown |
| source hero id | 2 | source actor 对应 hero config | 112 / 133 |
| slot_type | 5 | bullet `slot_type` | `0/1/2/3/other_or_unknown`；当前主要观测到 `1/2/3`，但保留普攻和兜底 |
| skill_id fallback | 1 | bullet `skill_id` | `skill_id_nonzero_or_unknown`；已确认 `0` 不提供语义 |
| center lane one-hot | 63 | bullet `location` | `delta_lane = bullet_lane - self_lane`，使用第 1 节中心精细区域，范围 `[-15000, 15000]`，单位 500，含越界兜底 |
| center width one-hot | 83 | bullet `location` | `delta_width = bullet_width - self_width`，使用第 1 节中心精细区域，范围 `[-10000, 10000]`，单位 250，含越界兜底 |
| distance continuous | 4 | bullet/self/enemy/tower remapped distance | 到我方英雄、敌方英雄、我方塔、敌方塔的 ratio |
| direction sign | 6 | `delta_lane/delta_width` | bullet 相对我方英雄的前后/左右方向 one-hot |
| trajectory | 18 | `runtime_id` 上一帧位置缓存 | 速度是否已知、速度 ratio、`v_lane/v_width` 符号、是否朝我方逼近、最近点距离 ratio、预计接近时间 bucket |
| source context | 12 | source actor + hero recent cache | 来源英雄到我距离、来源英雄是否 112/133、来源是否刚释放技能/普攻、敌方连续弹道上下文、来源是否不可见 |
| hit risk | 8 | bullet 位置 + trajectory + 近身距离近似 | 近身距离 bucket、横向擦身风险、是否已越过我方、是否可能接近我方下一小段移动区间 |

维度校验：

```text
1 + 3 + 5 + 2 + 5 + 1 + 63 + 83 + 4 + 6 + 18 + 12 + 8 = 211
DIM_BULLETS = 211 * 5 = 1055
```

子弹位置仍然使用第 1 节的 `lane/width` 重映射；location 只采用中心精细区域，不使用全局位置桶。原因是子弹特征服务短时间规避，关键是相对我方英雄的局部方向、距离、速度和逼近风险，而不是全图坐标。坐标异常时整颗 bullet 的位置/trajectory 子块置 0，只保留 exist/source/slot 信息。

鲁班扫射能否学会躲避：

- 如果扫射在协议中表现为连续的敌方 hero bullet，模型可以通过 bullet 局部位置、速度、逼近方向和来源 hero=112 学到走位规避。
- 如果扫射 bullet 出现得很晚或只在命中附近出现，单靠 bullet 不足以提前躲避；此时依赖敌方鲁班的技能释放状态、buff skills 变化、连续弹道上下文和双方距离方向来提供预判信号。
- 因此最终实现必须同时保留 hero buff/skill/recent 事件和 bullet trajectory，不能只做 bullet one-hot。

## 10. 目标候选特征：`DIM_TARGET = 30`，共 9 个

严格对应动作 target 头：

```text
0 None
1 enemy hero
2 self hero
3 enemy soldier 1
4 enemy soldier 2
5 enemy soldier 3
6 enemy soldier 4
7 enemy tower
8 monster
```

| 子块 | 维度 | 字段来源 | 价值 |
| --- | ---: | --- | --- |
| exist/legal | 2 | entity 存在、`legal_action` target mask | 避免非法目标 |
| target type | 5 | 固定 target slot | hero/self/soldier/tower/monster |
| hp/alive | 3 | target hp | 击杀/补刀/推塔 |
| distance/direction | 6 | position projection | 技能和普攻目标选择 |
| in-range by skill class | 6 | 技能 metadata + 距离 | 区分普攻、近中远技能 |
| tower danger/value | 4 | tower relation | 塔下目标风险 |
| entity hash/hint | 4 | runtime/config hash | 稳定但不扩维 |

这个块值得设计，因为动作空间最后一头就是 9 个 target。如果没有 target 候选特征，模型只能从单位池隐式学习目标排列，样本效率差。

注意：小兵实体池只保留 3 个敌方小兵，但 target 头仍有 4 个敌方小兵槽 `3-6`。因此第 4 个敌方小兵目标必须在 target candidate 中保留 `exist/legal/hp/distance` 等摘要特征；它不进入 soldier entity pool，但仍可作为动作目标被模型识别。

`in-range by skill class` 使用静态技能射程估计，而不是只做 coarse class。它不是策略规则，只表达“当前目标大致是否落在该按钮可能命中的范围内”。6 维固定为：

```text
basic_attack_in_range
skill1_in_range
skill2_in_range
skill3_in_range
summoner_in_effective_range
any_main_action_in_range
```

射程估计规则：

- 普攻使用运行时 `attack_range`，缺失时按英雄日志量级兜底。
- 鲁班二技能按全图/长轴可达处理，但仍保留方向和目标距离，让模型自己学习命中条件。
- 闪现/疾跑/净化/狂暴/治疗/干扰只表达“对该目标/场景是否可能相关”，不写具体策略动作。
- 不确定技能统一进入 coarse fallback：近程 `3000`、中程 `8800`、长程 `45000`，并保留 `unknown` 到 source/skill 特征中。

## 11. 历史/差分状态

`FeatureProcess` 内部维护上一帧缓存，但输出仍是固定维度，不引入 RNN 外的变长序列。

缓存项：

- self/enemy hero hp delta
- self/enemy tower hp delta
- money delta
- 最近技能成功释放 flag
- 最近命中 hero/tower/soldier flag
- 最近受到塔伤/英雄伤 flag
- buff appeared/disappeared count
- bullet `runtime_id` 上一帧位置和最近出现帧

这些差分分别进入 hero/global/tower/buff 子块，不单独新增大块维度。

bullet 轨迹缓存策略：

- key 使用 bullet `runtime_id`。
- 每帧记录重映射后的 `lane/width`、frame_no、source_actor、camp、slot_type。
- 如果同一 `runtime_id` 在上一帧或最近 2 帧出现，则计算速度和逼近风险；超过 2 帧未出现则不再用于速度估计。
- 缓存保留上限 8 帧，用于处理平台帧跳或日志间隔；超过 8 帧清理。
- 坐标异常或 camp/source 明显冲突时，只保留当前帧静态子弹特征，不计算 trajectory。

## 12. 字段缺失与安全兜底

每个字段都按以下规则处理：

- 数值字段缺失：填 0，并通过 exist/visible/unknown 位表达不可用。
- ID 字段未知：explicit 未命中时进入 family 或 hash bucket。
- 坐标异常：`abs(x)>60000` 或 `abs(z)>60000` 时位置全 0，clip 计数只用于 debug。
- `buff_marks` 为空且已确认不作为特征入口：不设计 mark 块；英雄状态主要依赖 `buff_skills`。
- `cooldown_max=0`：CD ratio 填 0，CD bucket 由 `usable/level` 辅助判断。
- target 不存在：target 块只有 `exist=0`，其余 0。

## 13. 模型切分建议

维度切分建议：

```python
DIM_POSITION = 180
DIM_GLOBAL = 60
DIM_HERO = 693
DIM_SOLDIER = 224
SOLDIER_SLOTS = 6
DIM_MONSTER = 222
DIM_TOWER = 248
DIM_BULLET = 211
DIM_TARGET = 30
FEATURE_DIM = 4833

# 60 + 2*693 + 6*224 + 222 + 2*248 + 5*211 + 9*30 = 4833
```

模型侧可以保持 `hok_semi` 风格：

- position MLP 处理 `DIM_POSITION=180`。
- unit MLP 处理 `DIM_UNIT_CORE - DIM_POSITION`。
- hero/soldier/tower/monster/bullet 分支编码。
- target embedding 使用 `DIM_TARGET=30` 的 target 候选特征，而不是只从单位隐藏向量中拼接。
- PPO action head 仍保持 `[12, 16, 16, 16, 16, 9]`。

## 14. 已冻结实现决策

### 14.1 explicit buff whitelist 96 个 ID 排序

排序原则：

1. 通用系统/召唤师/恢复类 buff，保证所有英雄共享状态位置稳定。
2. 鲁班 `112` 相关 buff，按 ID 升序。
3. 狄仁杰 `133` 相关 buff，按 ID 升序。
4. 当前日志出现但无法归类的系统 ID，按 ID 升序补齐。
5. 如果不足 96，剩余位固定为 reserved；如果超过 96，未进入白名单的 ID 走 family/hash。

冻结候选来源为第 5 节列出的 debug ID 与 `agent_diy.conf.conf` 当前候选。实现时生成常量 `BUFF_WHITELIST_96`，并在启动时 assert 长度为 96。

### 14.2 小兵排序

敌方小兵实体池只保留 3 个槽，优先对齐动作 target slot `3-5`。第 4 个敌方小兵 target slot `6` 只通过 target candidate 摘要表达。如果协议/动作 mask 没有直接目标顺序，则按以下 key 排序：

```python
(
    -frontline_score,        # 越接近双方交战前线越靠前
    lane_progress_to_enemy,  # 越靠近敌塔/我方进攻方向越靠前
    distance_to_self_hero,   # 再按我方英雄距离
    hp_ratio,                # 最后让低血量更靠前，利于补刀
)
```

我方小兵不对应 target 头，按 `frontline_score + lane_progress + distance_to_self_hero` 排序，优先保留正在交战、进塔或挡塔伤的小兵。

### 14.3 子弹轨迹缓存

采用第 11 节的 `runtime_id` 缓存策略：

- 最近 2 帧内同一 bullet 可计算速度。
- 缓存最多保留 8 帧。
- 超过 2 帧未出现不再计算 trajectory，但 8 帧内可用于去重和来源连续性。
- 坐标异常只保留静态 bullet 特征。

### 14.4 target candidate 射程

使用静态技能射程估计，不只做 coarse class。原因是 target 头直接决定动作目标，如果没有 per target 的射程提示，模型必须从英雄/目标池隐式拼出“这个按钮打不打得到这个目标”，样本效率较差。

静态射程只作为特征，不作为合法动作过滤；最终动作仍必须服从 `legal_action/sub_action_mask`。

### 14.5 模型结构

保留 `hok_semi` 的多分支编码和 max-pool 思路，但对 target candidate 增加独立 embedding，并在策略主干中拼接 target summary。这样不会破坏 PPO/action head 结构，又能提升目标选择效率。

## 15. 实现规范附录

本节面向后续代码重构 agent。实现时必须按本文 schema 写入，不允许边实现边改维度。

### 15.1 顶层特征顺序和切片

顶层顺序固定为：

```text
global
self_hero
enemy_hero
self_soldier_0..2
enemy_soldier_0..2
monster
self_tower
enemy_tower
bullet_0..4
target_0..8
```

维度公式：

```text
DIM_GLOBAL   = 60
DIM_HERO     = 693
DIM_SOLDIER  = 224
SOLDIER_SLOTS = 6
DIM_MONSTER  = 222
DIM_TOWER    = 248
DIM_BULLET   = 211
DIM_TARGET   = 30

FEATURE_DIM = 60 + 2*693 + 6*224 + 222 + 2*248 + 5*211 + 9*30 = 4833
```

实现必须在每个子函数末尾 assert：

```python
assert len(x_global) == 60
assert len(x_hero) == 693
assert len(x_soldier) == 224
assert len(x_monster) == 222
assert len(x_tower) == 248
assert len(x_bullet) == 211
assert len(x_target) == 30
assert len(feature) == 4833
```

`Config.DATA_SPLIT_SHAPE`、`Config.SAMPLE_DIM`、模型输入维度和样本序列化长度必须与 `FEATURE_DIM=4833` 同步。

### 15.2 常量表

实现时集中放在 `agent_diy` 的配置文件中，不要散落在 feature 代码里。

英雄 ID：

```python
HERO_IDS = [112, 133]
```

召唤师技能候选：

```python
SUMMONER_CANDIDATES = [80102, 80105, 80107, 80109, 80110, 80115]
```

语义标签：

```text
80102: recover_or_heal
80105: tower_interference
80107: cleanse_or_defense
80109: mobility
80110: damage/combat_buff
80115: mobility/instant_reposition
```

固定锚点，均为视角统一后的 `lane/width`：

```text
self_base_lane   ≈ -40000
enemy_base_lane  ≈  40000
self_tower_lane  ≈ -13000
enemy_tower_lane ≈  13000
self_cake_lane   ≈ -15000
enemy_cake_lane  ≈  15000
all_anchor_width ≈ 0
```

固定范围：

```python
TOWER_ATTACK_RANGE_FALLBACK = 8800
CAKE_RESPAWN_SECONDS = 75
```

目标顺序必须与动作 target 头一致：

```text
0 none
1 enemy_hero
2 self_hero
3 enemy_soldier_0
4 enemy_soldier_1
5 enemy_soldier_2
6 enemy_soldier_3
7 enemy_tower
8 monster
```

`BUFF_WHITELIST_96`：

- 按第 14.1 节排序生成。
- 长度必须 assert 为 96。
- 未进入白名单的 buff 进入 family/hash，不扩维。

### 15.3 坐标和异常兜底

所有位置都先执行：

```python
lane = (x + z) / sqrt(2)
width = (x - z) / sqrt(2)
```

然后按当前阵营做镜像，使我方基地始终在 `lane < 0`，敌方基地始终在 `lane > 0`。

异常规则：

- `location is None`：位置子块全 0。
- `abs(x) > 60000 or abs(z) > 60000`：位置子块全 0，`visible=0`。
- 所有 ratio clip 到合法范围。
- 不再把原始 `x/z` 输入模型。

### 15.4 历史缓存

`FeatureProcess` 需要维护以下跨帧缓存，缓存不进入样本序列，只用于生成固定维度特征：

```text
last_self_money
last_enemy_money
last_self_hp
last_enemy_hp
last_self_tower_hp
last_enemy_tower_hp
last_self_buff_set
last_enemy_buff_set
recent_hit_events
recent_take_hurt_events
recent_skill_success
recent_recover_or_cake_attempt
cake_next_spawn_frame_by_side
bullet_cache_by_runtime_id
```

bullet cache：

- key: `runtime_id`
- value: `lane, width, frame_no, source_actor, camp, slot_type`
- 最近 2 帧内出现才计算速度/逼近风险。
- 最多保留 8 帧，超过清理。
- 坐标异常时不更新 trajectory。

recent event 窗口建议：

```text
短窗口：最近 3 帧，用于技能释放、受击打断、子弹速度。
中窗口：最近 15 帧，用于最近命中/承伤/塔锁定。
长窗口：最近 75 秒对应帧数，用于血包刷新估计。
```

### 15.5 reserved 维度

本次按训练前冻结 schema 处理，实体/全局/子弹/target 中不再保留纯固定 0 的 reserved 维度。

规则：

- 不为“以后可能要用”额外占空维度。
- 若后续确实新增语义，视为 schema 变更，应重新训练。
- buff family 内的 `reserved` 只是分类桶名称，不代表额外固定 0 维度。

### 15.6 禁止改动项

以下内容实现时不得改动：

```text
FEATURE_DIM = 4833
LABEL_SIZE_LIST = [12, 16, 16, 16, 16, 9]
target order = [none, enemy hero, self hero, enemy soldier 0..3, enemy tower, monster]
heroes = 112 / 133
camp convention = blue 1, red 2
actor mapping:
  hero    = (0, 0)
  monster = (1, 0)
  soldier = (1, 11)
  tower   = (2, 21)
  crystal = (2, 23)
  spring  = (2, 24)
```

不得把策略规则写进特征。例如：

- 不写“残血必须回城”。
- 不写“有兵线必须推塔”。
- 不写“鲁班二技能必须远距离释放”。

只能编码可观测状态、距离、合法动作、静态技能类型和历史反馈。

### 15.7 最小验证清单

实现完成后必须做静态/shape 验证：

```text
1. 每个子块长度 assert 通过。
2. 最终 feature 长度为 4833。
3. DATA_SPLIT_SHAPE 与模型 feature split 完全一致。
4. SAMPLE_DIM 与样本序列化长度一致。
5. legal_action 仍按 [12,16,16,16,16,9] 切分。
6. target candidate 9 个槽与 action target 头一致。
7. 红蓝双方镜像后，同一战术位置的 lane/width 符号一致。
8. 缺失字段、padding、异常坐标不会抛异常。
9. buff whitelist 长度为 96。
10. bullet unknown enemy threat 不被直接丢弃。
```

当前会话约束：用户会自行运行平台 debug / `train_test.py`；Codex 不主动运行 `python train_test.py`。

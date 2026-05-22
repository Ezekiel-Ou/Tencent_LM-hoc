# 腾讯开悟 hok1v1 开发计划

> 主文档：`开悟强化学习系统化调研报告.md`  
> 开发入口：`AGENTS.md`  
> 核心策略：官方代码包接口为边界，先将 `kaiwu_taichu/hok_semi` 改造成符合本次赛题的自有基线，并沉淀到 `agent_diy`，再继续优化。

## 0. 总体路线

本项目采用“高分基线适配 + 官方协议对齐 + 小步验证”的路线。

| 优先级 | 方向 | 目标 |
| --- | --- | --- |
| P0 | 适配 `hok_semi` 为 `agent_diy` 基线 | 拥有可训练、可评估、可保存加载的高分基线改写版本 |
| P0 | 修正 PPO 基础稳定性 | dual-clip、value loss、advantage normalization、shape assert |
| P0 | 对齐英雄、特征和奖励协议 | 将半决赛英雄/状态字段映射为本次鲁班/狄仁杰 hok1v1 |
| P1 | 保留结构化 encoder 与 target attention | 以 `hok_semi` 的建模思想作为初始上限 |
| P1 | 设计训练课程 | common_ai 预热、自对弈、历史模型池、PFSP |
| P2 | 增强项 | 动作约束损失、情景奖励、value normalization、Module Soup |

## 1. 阶段一：将 `hok_semi` 适配为本赛题基线

目标：把 `kaiwu_taichu/hok_semi` 的高分实现改写为符合本次官方 hok1v1 赛题的 `agent_diy` 基线，不破坏 `agent_ppo` 官方 baseline。

### 1.1 任务

| 编号 | 任务 | 文件范围 | 验收 |
| --- | --- | --- | --- |
| 1.1.1 | 准备 `hok_semi` 代码快照 | 外部仓库到本地参考目录或手动迁移源 | 能稳定查看其 `agent_ppo/conf/feature/model/workflow` |
| 1.1.2 | 将 `hok_semi` 主体改写到 `agent_diy` | `agent_diy/**` | `agent_diy` 具备 `predict/exploit/learn/save/load/init_config` |
| 1.1.3 | 替换英雄和阵容假设 | `agent_diy/conf/`、`workflow/`、`feature/` | 只使用鲁班七号 `112` 和狄仁杰 `133` |
| 1.1.4 | 对齐动作协议 | `agent_diy/agent.py`、`model/`、`feature/definition.py` | 保持 `[12,16,16,16,16,9]` 和 target 顺序 |
| 1.1.5 | 保持 `conf/algo_conf_hok1v1.toml` 中 `diy` 入口可用 | `conf/algo_conf_hok1v1.toml` | `diy` 指向 `agent_diy` |
| 1.1.6 | 跑通静态/shape smoke checks | `agent_diy/**` | `py_compile`、`Config.validate()`、forward/sample-shape probe 不报错 |
| 1.1.7 | 添加关键 shape assert | `agent_diy/feature/definition.py`、`agent_diy/model/model.py` | `SAMPLE_DIM`、feature dim、legal action dim 可核验 |

### 1.2 注意事项

- 适配时先保持 `hok_semi` 的主体工程结构，优先替换赛题不匹配处。
- `agent_ppo` 只作为参考，不删除、不重命名。
- 每次完成一个协议映射后立即运行静态/shape smoke checks，避免一次性大改导致不可定位。

### 1.3 `hok_semi` 与本次赛题差异映射

| 维度 | `hok_semi` | 本次赛题 | 适配动作 |
| --- | --- | --- | --- |
| 英雄池 | 后羿 `169`、李元芳 `173`、虞姬 `174` | 鲁班七号 `112`、狄仁杰 `133` | 删除旧英雄分支，重建 112/133 技能、buff、被动、CD 特征 |
| 阵容轮换 | 三英雄组合 | 两英雄红蓝轮转，覆盖同英雄和异英雄 | 改为 `[112, 133]` round-robin，并覆盖 `112-112`、`112-133`、`133-112`、`133-133` |
| 状态维度 | README 描述约 `3910` 维，离散/连续拆分 | 当前官方 baseline 仅 10 维，目标需自定义 | 保留结构化 schema 思想，但重新计算所有维度 |
| 英雄专属特征 | 后羿/李元芳/虞姬 mark、技能状态 | 鲁班扫射被动、狄仁杰被动层数/控制解除/大招眩晕等 | 重写 hero-specific extractor |
| 地图实体 | 英雄、小兵、塔、河蟹、飞行物、血包等 | 文档明确英雄、小兵、塔；协议也有 monster、bullets、cakes | 可保留通用实体框架，不稳定实体置零加 mask |
| 召唤师技能 | 需按其半决赛策略核对 | 本赛题开放 10 个召唤师技能，`init_config` 赛前选择 | 改成 112/133 对阵规则或可学习表 |
| 动作空间 | 层次化动作，预期与开悟 1v1 接近 | `[12,16,16,16,16,9]` | 保留 head 结构，强校验 target 顺序 |
| reward | 多项 reward 管理 | 需要塔血、血量、击杀、死亡、经济、经验、补刀、forward | 复用 reward manager 结构，重设权重和字段来源 |
| 模型结构 | encoder + MLP/LSTM + target attention + value norm | baseline MLP，需升级 | 作为初始基线结构保留，先简化到可跑通 |
| workflow | 半决赛配置和模型池逻辑 | 本地官方 `common_ai/selfplay/model_id` | 对齐本地 `train_env_conf.toml` 和 `kaiwu.json` |

## 2. 阶段二：修正 PPO 与 baseline 明显缺陷

目标：让基础 PPO 稳定、可诊断、没有已知硬 bug。

### 2.1 任务

| 编号 | 任务 | 文件范围 | 验收 |
| --- | --- | --- | --- |
| 2.1.1 | 修正 value loss 目标 | `agent_diy/model/model.py` | value loss 使用 `reward_sum/value target`，不再重复覆盖 |
| 2.1.2 | 实现标准 dual-clip PPO | `agent_diy/model/model.py` 或 loss helper | 仅 `advantage < 0` 时应用 dual-clip 下界 |
| 2.1.3 | 添加 advantage normalization | `agent_diy/model/model.py` 或 `algorithm.py` | 每 batch 内 advantage 标准化，可开关 |
| 2.1.4 | 添加 value clipping | `agent_diy/model/model.py` | critic 更新幅度受控，可开关 |
| 2.1.5 | 修正学习率默认值 | `agent_diy/conf/conf.py` | 起始 LR 优先使用 `1e-4` 级别 |
| 2.1.6 | 记录 PPO 关键指标 | `monitor_builder.py`、loss 返回值 | 可看到 total/value/policy/entropy，后续扩展 ratio/KL |

### 2.2 验收

- 静态/shape checks 通过。
- loss 不出现明显 NaN/Inf。
- PPO ratio、entropy、value loss 有可观测输出或可调试入口。

## 3. 阶段三：基础特征与奖励扩展

目标：保留 `hok_semi` 的结构化观测思想，但将字段全部映射到本次官方 hok1v1 数据协议。

### 3.1 特征任务

| 编号 | 任务 | 内容 | 验收 |
| --- | --- | --- | --- |
| 3.1.1 | 设计 feature schema v1 | hero/self/enemy/tower/soldier/global/summoner | 文档化每组维度和归一化方式 |
| 3.1.2 | 扩展英雄特征 | HP、level、money、exp、位置、攻击范围、技能 CD、buff、可见性 | 能区分自身和敌方英雄 |
| 3.1.3 | 扩展防御塔特征 | 双方塔血、位置、存活、距离 | 塔血 reward 与观测一致 |
| 3.1.4 | 扩展小兵特征 | 最近 N 个小兵 HP、camp、位置、距离、攻击目标 | 缺失单位用 0 + mask |
| 3.1.5 | 召唤师技能特征 | 我方已选技能、敌方英雄组合 | 支持 `init_config` 决策 |
| 3.1.6 | 镜像处理 | 红蓝双方坐标统一到同一视角 | 所有位置字段一致变换 |

### 3.2 奖励任务

| 编号 | reward | 目的 | 首版建议 |
| --- | --- | --- | --- |
| 3.2.1 | `tower_hp_point` | 对齐胜负目标 | 保持最高权重 |
| 3.2.2 | `hp_point` | 学习换血收益 | 零和差分 |
| 3.2.3 | `kill` / `dead` | 学习击杀和避死 | 稀疏项，权重谨慎 |
| 3.2.4 | `money` / `exp` | 学习发育 | 小权重，归一化 |
| 3.2.5 | `last_hit` | 提升补刀 | 若协议可稳定识别则加入 |
| 3.2.6 | `ep_rate` | 控制资源消耗 | 视英雄资源字段可用性决定 |
| 3.2.7 | `forward` | 防止过度保守 | 低权重，避免送塔 |

### 3.3 验收

- reward dict 中每个子项都可打印或监控。
- reward 总和与各子项权重一致。
- feature dim、`DATA_SPLIT_SHAPE`、`SAMPLE_DIM` 同步更新。
- 静态/shape checks 通过。

## 4. 阶段四：稳定 `hok_semi` 网络结构适配版

目标：在 `agent_diy` 中稳定结构化 encoder + 时序分支 + target attention，先做到可训练，再做结构精简或增强。

### 4.1 任务

| 编号 | 任务 | 说明 | 验收 |
| --- | --- | --- | --- |
| 4.1.1 | 分组 encoder | hero、tower、soldier、global 分别编码 | 每组输入维度明确 |
| 4.1.2 | MLP 当前帧分支 | 捕捉当前态势 | 输出固定 hidden dim |
| 4.1.3 | GRU/LSTM 时序分支 | 接入 `LSTM_TIME_STEPS=16` 样本 | forward 中真实调用 recurrent 层 |
| 4.1.4 | 分支融合 | concat 或 gated fusion | action/value 共用 backbone |
| 4.1.5 | target attention | 用候选目标 embedding 生成 target logits | target head 不再只是普通 MLP 分类 |
| 4.1.6 | 多 action head | 保持 `[12,16,16,16,16,9]` | mask 后采样合法 |

### 4.2 迁移原则

- 英雄特定字段必须替换为鲁班/狄仁杰字段。
- 先实现最小 target attention，再做 target-first concat 等增强。
- 任何网络结构改变都必须同步更新 sample shape 和推理状态返回。

## 5. 阶段五：召唤师技能策略

目标：把 `init_config` 从随机选择改成可解释的对阵策略。

### 5.1 首版规则

| 对阵 | 推荐方向 | 原因 |
| --- | --- | --- |
| 鲁班 vs 鲁班 | 闪现/治疗 | 提高生存和拉扯 |
| 鲁班 vs 狄仁杰 | 净化/闪现 | 应对控制和追击 |
| 狄仁杰 vs 鲁班 | 晕眩/闪现 | 增强开打和留人 |
| 狄仁杰 vs 狄仁杰 | 终结/闪现 | 强化收割或容错 |

注意：本赛题不应假设只出现鲁班打狄仁杰。官方配置允许蓝方和红方分别选择 `112/133`，当前 baseline 的 `lineup_iterator_roundrobin_camp_heroes([112, 133])` 会生成四种组合，因此训练和评估代码必须把同英雄对局作为常规路径处理。

### 5.2 后续学习化

- 维护 `(my_hero, enemy_hero) -> summoner_skill` 胜率表。
- 使用 epsilon-greedy 或 softmax 按评估结果更新。
- 将已选召唤师技能加入 global feature。

## 6. 阶段六：训练课程与自对弈

目标：从可打 common_ai 到具备自对弈鲁棒性。

### 6.1 训练阶段

| 阶段 | 对手 | 目标 | 切换条件 |
| --- | --- | --- | --- |
| Warmup | `common_ai` | 学会基础走位、补刀、压塔 | common_ai 胜率稳定升高 |
| Basic self-play | `selfplay/latest` | 避免只适配规则 AI | self-play 不崩、对局不过度保守 |
| Historical pool | 历史 checkpoint | 抑制遗忘和退化 | 对历史模型胜率稳定 |
| PFSP | 难打历史模型高采样 | 补短板 | 有胜率矩阵和采样权重 |

### 6.2 工程任务

| 编号 | 任务 | 文件范围 | 验收 |
| --- | --- | --- | --- |
| 6.2.1 | 保留 common_ai 训练配置 | `train_env_conf.toml` | 可一键切换 |
| 6.2.2 | 历史模型池记录 | workflow 或辅助模块 | checkpoint ID、胜率、采样权重可查 |
| 6.2.3 | PFSP 采样 | workflow | `p ∝ (1 - winrate)^2` 或可配置变体 |
| 6.2.4 | 评估对手矩阵 | `kaiwu.json`、监控 | 至少支持 3 个对手模型 ID |

## 7. 阶段七：增强项与冲刺策略

这些不是第一阶段阻塞项，基础链路稳定后再接入。

| 增强项 | 价值 | 风险 | 验收 |
| --- | --- | --- | --- |
| 动作约束损失 | 抑制站桩、无效回城、乱放技能 | 误伤合理等待动作 | 动作分布改善，胜率不下降 |
| 情景奖励 | 放大关键稀有样本 | reward hacking | 触发次数可监控，回放行为合理 |
| Value normalization / PopArt | 降低 reward 尺度敏感性 | 实现复杂 | A/B 对比更稳 |
| 课程式 reward 切换 | 不同阶段优化不同目标 | 切换点震荡 | 切换后胜率不崩 |
| Module Soup | 融合互补模型 | 结构必须一致 | 融合后对多对手不掉点 |

## 8. 代码审计清单

每次大改 feature、model、loss、workflow 后执行：

| 审计项 | 要求 |
| --- | --- |
| 本地测试 | `py_compile`、`Config.validate()` 和相关 shape/helper checks 通过；不运行 `python train_test.py` |
| 样本拆分 | `Config.DATA_SPLIT_SHAPE` 与实际 sample 拆分一致 |
| 样本维度 | `Config.SAMPLE_DIM` 与打包后的样本长度一致 |
| 动作维度 | `LABEL_SIZE_LIST` 未被破坏 |
| 动作合法性 | `legal_action` 和 `sub_action_mask` 同时生效 |
| 目标顺序 | target head 的候选目标顺序与环境动作协议一致 |
| 镜像处理 | 红蓝镜像对所有 location 字段一致 |
| 推理接口 | `predict` 训练采样和 `exploit` 评估贪心逻辑都可用 |
| 赛前配置 | `init_config` 返回合法召唤师技能 |
| 奖励方向 | reward 子项没有明显符号错误 |
| 模型加载 | 两个 camp 的模型加载逻辑符合当前训练/评估模式 |
| 保存频率 | 模型保存频率不触发平台限制 |

## 9. 立即执行顺序

建议后续 Codex CLI 第一轮开发按以下顺序执行：

1. 准备 `kaiwu_taichu/hok_semi` 代码快照，重点保留 `conf/feature/model/algorithm/workflow`。
2. 将其主体改写进 `agent_diy`，先解决 import、入口类和官方 wrapper 兼容。
3. 替换英雄池、阵容轮换和 `init_config`：只支持鲁班 `112` / 狄仁杰 `133`。
4. 对齐动作空间和 target 顺序：保持 `[12,16,16,16,16,9]`。
5. 重写 hero-specific feature extractor，并重新计算 feature dim / `DATA_SPLIT_SHAPE` / `SAMPLE_DIM`。
6. 复用 reward manager 结构，映射为本赛题 8 项基础 reward。
7. 跑通静态/shape checks，不运行 `python train_test.py`。
8. 稳定 PPO loss、GRU/LSTM、target attention 和 value head。
9. 建立 common_ai warmup 配置和基础 self-play 配置。
10. 再进入 PFSP、动作约束损失、情景奖励和 Module Soup。

## 10. 当前未处理事项

- 暂不删除 `调研报告.md`。它已降级为历史备忘录，待开发进入稳定阶段后可移动到 archive 或删除。
- `kaiwu_taichu/hok_semi` 代码快照已引入 `external/kaiwu_taichu_2025/`。后续开发继续以该目录为只读参考源，改动沉淀到 `agent_diy`。

## 11. 2026-05-07 阶段一执行记录

### 11.0 当前验证约束

- 自 2026-05-15 起，Codex 不再在任何步骤下运行 `python train_test.py`。该命令依赖官方 Kaiwu 运行环境，历史上本地只会失败于缺少 `kaiwudrl`，不能提供有效项目验证信号。
- 后续 Codex 验证统一使用静态/轻量检查：`python -m py_compile ...`、`Config.validate()`、模型 forward/sample-shape probe、reward/action helper 定向检查。平台协议 smoke test 由官方环境或用户侧执行。

### 11.1 已完成

| 阶段项 | 状态 | 说明 |
| --- | --- | --- |
| 1.1.1 准备 `hok_semi` 快照 | 已完成 | 已将 `wty-yy/kaiwu_taichu` 的 `2025` 分支 sparse clone 到 `external/kaiwu_taichu_2025/`，可查看 `hok_semi/code/agent_ppo/{conf,feature,model,algorithm,workflow}`。 |
| 1.1.2 改写主体到 `agent_diy` | 初版完成 | `agent_diy` 已具备独立 `predict/exploit/learn/save/load/init_config/workflow` 链路，不依赖 `agent_ppo` 代码。 |
| 1.1.3 替换英雄和阵容假设 | 初版完成 | `agent_diy.conf.GameConfig.HERO_IDS=[112,133]`，workflow 使用 round-robin 覆盖 `112-112`、`112-133`、`133-112`、`133-133`。 |
| 1.1.4 对齐动作协议 | 初版完成 | 保持 `LABEL_SIZE_LIST=[12,16,16,16,16,9]`；推理侧校验 raw legal action 184 维，训练样本侧校验压缩 legal action 85 维。 |
| 1.1.5 保持 `diy` 入口 | 已确认 | `conf/algo_conf_hok1v1.toml` 中 `diy` 仍指向 `agent_diy`。 |
| 1.1.7 添加 shape assert | 高维版完成 | `FEATURE_DIM=3910`，`SAMPLE_DIM=66544`，feature/sample/legal action/model logits 均有显式校验。 |

### 11.2 官方文档冲突修正

- `hok_semi` README 的目标小兵排序备注偏向 runtime id；本赛题官方文档描述 target 为“最近的四个小兵”，因此 `agent_diy` 首版特征按距离选择最近 4 个小兵，并保持 target head 顺序为 `None, enemy hero, self, soldier1-4, tower, monster`。
- `hok_semi` 默认 reward 中 `kill` 权重为负值；本赛题官方文档将 `kill` 作为击杀正向奖励项，`agent_diy` 首版使用 `kill=+0.6`、`death=-1.0`。

### 11.3 验证记录

- 已运行 `python -m py_compile` 覆盖 `agent_diy` 核心文件，通过。
- 已运行模型前向 shape 自测：`FEATURE_DIM=3910`、logits `(1,85)`、value `(1,1)`、LSTM state `(1,1,512)`，通过。
- 已运行结构化特征样例自测：输出长度 `3910`，通过。
- 已运行一次假样本 `Algorithm.learn` 自测：样本维度 `66544`，loss 正常返回，通过。
- 历史上曾尝试运行 `python train_test.py`，但本机缺少官方运行包，失败于 `ModuleNotFoundError: No module named 'kaiwudrl'`，未进入 `agent_diy` 代码路径。自 2026-05-15 起不再由 Codex 运行该命令。

### 11.4 2026-05-07 smoke test 修复记录

- 修复 `_legal_sample` 使用 `np.random.multinomial` 时概率和因浮点精度略大于 1 触发 `ValueError` 的问题：采样改为 `np.random.choice`，并在合法动作 mask 内做 64-bit 严格归一化。
- 修复 workflow 传给 `env.step` 的动作可能被包装为 `[None]` 或 `[[a1,...,a6]]` 的问题：`agent.action_process` 和 workflow 入 env 前都强制规整为 6 个 int 的扁平动作，异常动作回退到 `NONE_ACTION`。

### 11.5 2026-05-07 hok_semi 高维迁移修订

- 作废 `FEATURE_DIM=304 / SAMPLE_DIM=8848` 的轻量 baseline 锚点，改为对齐 `hok_semi` 的结构化高维观测：`FEATURE_DIM=3910`、`SAMPLE_DIM=66544`。
- `agent_diy/conf/conf.py` 已重算并锁定 `DIM_HERO=347`、`DIM_SOLDIER=175`、`DIM_RIVER_CRAB=172`、`DIM_ORGAN=172`、`DIM_BULLET=130`，动作空间仍保持 `[12,16,16,16,16,9]`。
- `agent_diy/feature/feature_process.py` 已迁移为 `self hero / enemy hero / soldiers / river crab / towers / bullets` 的 hok_semi 顺序，并按官方 112/133 协议处理扁平 dict、红蓝镜像、最近 4 小兵、敌方英雄/防御塔子弹。
- `agent_diy/model/model.py` 已迁移为 position encoder、unit encoder、hero/soldier/organ/bullet 分组 encoder、MLP+LSTM 融合、target attention 和 value head。
- `agent_diy/feature/reward_process.py` 增加首帧差分保护；workflow 和 monitor 已补充 reward 子项、ratio、approx_kl、clip_fraction 监控。
- 本轮按用户要求没有运行 `python train_test.py`；已运行非训练检查：`python -m py_compile`、模型 forward shape、结构化特征样例、假样本 `Algorithm.learn`。

### 11.6 2026-05-07 common_ai warmup 反站桩修订

- 现象：训练早期 `money_per_frame` 长时间不变，说明智能体没有稳定进入兵线接触区；这不是单纯训练时间不足，而是冷启动探索没有产生可学习的上线轨迹。
- `forward` reward 从“仅在己方塔后且满血时给负值”修订为“己方塔到敌方塔方向的进度值”，并将早期作用窗口从 `1000` 帧延长到 `3000` 帧，权重从 `0.01` 提到 `0.03`。
- `agent_diy/model/model.py` 对 button head 增加固定 logits 先验：提高移动、普攻和基础技能探索，压低无动作和回城；不改变 `FEATURE_DIM`、`SAMPLE_DIM`、`LABEL_SIZE_LIST` 或 checkpoint tensor shape。
- `agent_diy/model/model.py` 的 LSTM reshape 改为按 hidden batch 自动推导序列长度，兼容训练侧 16 帧序列和推理侧单帧输入。
- `agent_diy/conf/train_env_conf.toml` 当前默认对手已是 `common_ai`，继续用于 v1 cold-start warmup；等 `money/exp/last_hit` 开始有非零变化、塔血和帧数曲线正常后，再切 self-play。

### 11.7 2026-05-08 last_hit 诊断修订

- 现象：9h common_ai warmup 中 `reward_last_hit` 仍显示为 0；该项只依赖 `frame_action.dead_action`，而 `hok_semi` 参考已标注死亡事件并不完整。
- `agent_diy/feature/reward_process.py` 已兼容 `frame_action/frameAction`、`dead_action/deadAction`、`runtime_id/runtimeId`、`sub_type/subType`，并兼容 nested `actor_state`。
- 新增 last_hit 诊断监控：`last_hit_dead_action_count`、`last_hit_soldier_dead_count`、`last_hit_main_count`、`last_hit_enemy_count`。若前两项长期为 0，说明平台帧事件没有给到可用小兵死亡事件；若前两项非 0 但 main/enemy 为 0，则继续查 killer runtime 对齐。
- 本修订不改变特征维度、样本维度、动作空间或模型 tensor shape；只影响 reward 解析和监控。

### 11.8 2026-05-08 召唤师技能探索与动作监控

- 官方文档明确 `init_config` 用于对局初始化前选择召唤师技能；v1 将训练期召唤师技能从固定 matchup 规则改为按英雄候选列表轮换探索，评估期仍使用固定规则，避免评估噪声。
- 训练期候选：鲁班 `112` 使用闪现、治疗、狂暴、净化、干扰；狄仁杰 `133` 使用闪现、治疗、狂暴、晕眩、终结、干扰。候选列表集中在 `GameConfig.SUMMONER_SKILL_CANDIDATES_BY_HERO`，可热启动调整。
- 新增动作与技能监控：无动作、移动、普攻、技能、恢复、召唤师、回城、装备技能动作次数；1/2/3 技能实际释放、恢复实际释放、召唤师技能实际释放。
- 新增召唤师技能选择 one-hot 监控：`selected_summoner_80102/80103/80104/80105/80107/80108/80109/80110/80115/80121`。
- 官方协议中可见 `equip_state`、`buy_equip`、`sell_equip` 字段，但平台 aisrv 对 agent 输出做格式校验，当前 hok1v1 训练接口只接受双方各 6 个整数的动作列表：`[[a1,a2,a3,a4,a5,a6],[b1,b2,b3,b4,b5,b6]]`。2026-05-08 实测 raw `buy_equip` dict 被拒绝为 `Invalid actions format`，因此 v1 不支持通过 agent action 自定义购买装备。
- 已移除 `buy_equip` 探针相关代码、配置项和监控项，避免误配置再次导致正式训练崩溃。
- 为避免平台监控配置合并失败，`monitor_builder.py` 已改为少量聚合 panel：`loss`、`ppo`、`reward_items`、`last_hit`、`action_debug`、`action_target`、`summoner_choice`。

### 11.9 2026-05-08 self-play 切换

- `agent_diy/conf/train_env_conf.toml` 已将 `opponent_agent` 从 `common_ai` 切到 `selfplay`，`eval_opponent_type` 和 `eval_opponent_types` 继续保留 `common_ai`，用于观察 self-play 后是否遗忘 common_ai 基础能力。
- `Config.INIT_LEARNING_RATE_START` 从 `1e-4` 下调到 `3e-5`。原因是当前 checkpoint 只热启动模型权重，optimizer/scheduler 不恢复；self-play 起步用 fresh Adam 时不宜继续用 common_ai 冷启动阶段的较高学习率。
- 本轮不改 `TARGET_LR=1e-5`、`TARGET_STEP=5000`、`BETA_START=0.01`、`CLIP_PARAM=0.2`、`DUAL_CLIP_PARAM=3.0`、`USE_ADVANTAGE_NORM=True`、`USE_VALUE_CLIP=False`、`TARGET_EMBED_DIM=32`。其中 `TARGET_EMBED_DIM` 属于模型结构维度，不能在 v1 热启动链路中随意改变。

### 11.10 2026-05-08 对线行为与目标选择诊断

- 现象：10h common_ai warmup 已能推塔和击杀 common_ai，但录像表现更像“原地普攻小兵、敌方英雄进入射程才顺带交互”，不能直接判断为训练时间不足。
- 官方 env 指标中已有 `hurt_to_hero`、`hurt_by_hero`、`kill/death`、`money_per_frame`、`tower_hp`、`frame`；其中 `hurt_to_hero` 是判断是否主动打人的首要指标。
- v1 新增 `action_target` 监控面板：`target_none_count`、`target_enemy_hero_count`、`target_self_hero_count`、`target_enemy_soldier_count`、`target_enemy_tower_count`、`target_monster_count`、`attack_target_enemy_hero_count`、`attack_target_enemy_soldier_count`、`skill_target_enemy_hero_count`。
- 判断标准：如果 `hurt_to_hero`、`target_enemy_hero_count`、`attack_target_enemy_hero_count`、`skill_target_enemy_hero_count` 长期偏低，而 `target_enemy_soldier_count` 与 `attack_target_enemy_soldier_count` 偏高，说明策略已陷入“清兵推线”局部最优，应优先改 reward/目标选择训练信号，而不是单纯加训练时长。
- 本轮只补诊断，不改变 `FEATURE_DIM`、`SAMPLE_DIM`、`LABEL_SIZE_LIST`、模型结构或 checkpoint tensor shape，可从 v1 checkpoint 继续训练。

### 11.11 2026-05-08 buy_equip 探针结论

- 训练日志报错：`Invalid actions format [[{'command_type':'COMMAND_TYPE_BuyEquip','buy_equip':{'equipId':1110,'obj_id':0}}], [[...6 ints...]]], expected [[a1,a2,a3,a4,a5,a6], [b1,b2,b3,b4,b5,b6]]`。
- 结论：当前比赛训练接口暴露的是 6 维离散动作协议，不接受 raw `CmdPkg` 字典作为 agent action；官方底层数据协议存在 `buy_equip` 字段不等于本赛题 agent 可直接下发购买命令。
- 处理：已移除 `buy_equip` 探针的 workflow 注入逻辑、`train_env_conf.toml` 配置项、`GameConfig` 常量和监控指标；后续不再从 agent action 层尝试购买装备。
- 装备 ID 表：本地官方文档只说明 `EquipSlot.configId` 对应装备配置表，没有提供完整配置表。后续最多只能从观测 `equip_state.equips[].configId` 反向记录“平台自动/默认出装实际出现过的装备 ID”，但在当前 6 维 action 接口下不能用这些 ID 自定义购买。

### 11.12 2026-05-08 官方动作协议复核

- 复核官方 `环境详述` 与 `数据协议`：agent action 必须保持 `[12,16,16,16,16,9]` 顺序的 6 个离散 id；`legal_action` 用于屏蔽非法动作，`sub_action_mask` 用于按 button 过滤无意义子动作训练。
- `agent_diy` 当前推理侧支持平台原始 `184` 维 legal action，并按已采样 button 压缩 target mask；训练样本侧保存 `85` 维压缩 legal action，与 `DATA_SPLIT_SHAPE` 一致。
- 清理 `agent_diy/conf/train_env_conf.toml` 中静态 `select_skill=80115`。召唤师技能不再写死在配置文件，而是在每局 `env.reset()` 前由 `Agent.init_config()` 返回并由 workflow 注入 `select_skill/summoner_skill_id`。
- 补齐动作监控口径：官方 button `10` 是 `Skill 4`，现已纳入 `action_skill_count` 与 `skill_target_enemy_hero_count` 统计；不改变模型结构、动作空间、样本维度或 checkpoint shape。
- 本轮按用户要求未运行 `python train_test.py`；已运行静态检查：`python -m py_compile ...` 和 `Config.validate()`，确认 `FEATURE_DIM=3910`、`LEGAL_ACTION_DIM=85`、`RAW_LEGAL_ACTION_DIM=184`、`SAMPLE_DIM=66544`。

### 11.13 2026-05-08 监控面板降噪

- 平台图表中单 panel 曲线过多，影响判断训练健康度；`monitor_builder.py` 已回退为少线面板。
- 保留面板：`reward`、`loss_core`、`ppo_health`、`reward_objective`、`reward_combat`、`last_hit`、`action_buttons`、`action_targets`。
- 移除展示面板：`summoner_choice`。训练侧仍会上报 `selected_summoner_*`，但默认不再画 10 条 one-hot 线。
- `last_hit` 保留为诊断项，不作为判断兵线能力的唯一指标。兵线学习主看 `money_per_frame`、塔血、frame、`target_enemy_soldier_count` 和平台录像；若平台 `dead_action` 不完整，`reward_last_hit` 可能长期失真。
- 本修订只改变监控显示，不改变 reward、模型结构、动作空间、样本维度或 checkpoint 兼容性。

### 11.14 2026-05-08 mixed opponent 训练配置

- 更正：平台历史模型对战应走官方自定义模型 ID 路径，不使用本地 `warmup` 别名；`15525` 预训练起点由平台界面操作，不写入 `preload_model`。
- `conf/configure_app.toml` 已恢复 `preload_model=false`，避免平台 preload 校验 `agent_diy/ckpt` 失败。
- `kaiwu.json` 已配置 `model_pool=[254424]`。
- `agent_diy/conf/train_env_conf.toml` 已改为 `opponent_agent="mixed"`，训练对手按权重轮换：
  - `selfplay`: `0.4`
  - `common_ai`: `0.3`
-  - custom model id `254424`: `0.3`
- `254424` 会直接写入 `usr_conf["episode"]["opponent_agent"]` 并通过官方 `load_opponent_agent(id="254424")` 加载；该侧不采样，避免对手模型样本进入 learner。
- 本修订不改变模型结构、动作空间、特征维度或样本维度。

### 11.15 2026-05-09 18h 训练判读与监控再降噪

- 本轮 18h mixed/self-play 训练从监控看属于训练链路健康：样本生产/消费线性增长，进程存活正常，CPU/内存/GPU 指标无资源泄漏或明显卡死；PPO `approx_kl` 与 `clip_fraction` 长期贴近低位，没有策略更新爆炸迹象。
- 训练尚不能判定为最终合格：胜率曲线仍有波动，评估胜率没有形成压倒性平台期；对线指标显示已有清兵、推塔和英雄交互，但还需要平台评估胜率、录像行为和多对手矩阵确认。
- `agent_diy/conf/monitor_builder.py` 已进一步简化为核心可读面板：`reward`、`total_loss`、`value_loss`、`ppo_kl`、`ppo_clip`、`objective_reward`、`target_focus`。
- `last_hit`、召唤师技能 one-hot、大量动作细项不再默认绘图；workflow 仍会上报这些字段，需要专项排查时可临时加回。
- 本修订只改变监控展示，不改变 reward、模型结构、动作空间、特征维度、样本维度或 checkpoint 兼容性。
### 11.16 2026-05-11 feature debug 坐标与实体锚点

- 新增实测记录文档：`docs/feature_debug_findings.md`。
- 当前 debug 结论：地图应按对角线长条形重映射，特征中使用 `lane=(x+z)/sqrt(2)`、`width=(x-z)/sqrt(2)`。
- 坐标策略已冻结为首版工程方案：全局长轴半长 `LANE_HALF_RANGE=45000`；常规区域宽半径 `NORMAL_WIDTH_HALF_RANGE=14000`；中心核心区 `CENTER_LANE_HALF_RANGE=15000`、`CENTER_WIDTH_HALF_RANGE=20000`；`COORD_CLIP_ABS=60000` 用于异常原始坐标兜底。
- 已确认实体映射：英雄 `(0,0)`、野怪 `(1,0)`、小兵 `(1,11)`、防御塔 `(2,21)`、水晶 `(2,23)`、泉水 `(2,24)`。
- 已确认锚点：出生点/泉水约在 `±40000` 对角线端，外塔约在 `±13000`，血包坐标约为 `[15340,15100]` 与 `[-15220,-15120]`。
- 后续重构 `agent_diy` 特征时，塔、血包、泉水坐标应以这些锚点和投影分区归一化为基准；子弹、野怪、小兵、英雄等其他位置统一走同一套投影和 clip 兜底，不再继续阻塞在坐标常量观测上。

### 11.17 2026-05-12 4833 特征与模型重构

- 按 `docs/feature_engineering_design.md` 将 `agent_diy/feature/feature_process.py` 重构为 4833 维冻结 schema：`global + 2 hero + 6 soldier + monster + 2 tower + 5 bullet + 9 target`，统一使用 `lane/width` 坐标重映射和红方镜像。
- 新特征实现覆盖通用单位核心、英雄技能槽、buff whitelist/family/hash、经济差分、塔/血包锚点、敌方威胁子弹轨迹缓存、target candidate 9 槽，并在各子块保留长度断言。
- 发现设计文档中 `DIM_TOWER=248` 与塔子项表格求和 `254` 不一致；实现以冻结维度 `248` 为准，压缩塔的 `range/aggro/pressure` 子块，保留塔血、塔锁定、塔下风险、血包和推塔上下文。
- 按 `docs/model修改版本.md` 将 `agent_diy/model/model.py` 重构为 4833 schema 的分组 encoder：global、hero、soldier、monster、tower、bullet、target candidate 独立编码，同类实体批量 MLP，soldier/bullet/target 使用 max+mean pooling。
- target head 已改为标准 scaled dot-product attention；legal action mask 常量从 `1e20` 降为 `1e8`；主干 `concat_mlp`、`concat_mlp_other`、`lstm_and_linear_mlp` 启用 LayerNorm；`algorithm.py` 增加 entropy beta 衰减。
- 保持 `LABEL_SIZE_LIST=[12,16,16,16,16,9]`、`LSTM_TIME_STEPS=16`、`LSTM_UNIT_SIZE=512`、压缩 legal action 85 维和 raw legal action 184 维不变。
- 本轮按设计文档约束未运行 `python train_test.py`，交由用户在官方 Kaiwu 环境中执行；已完成非训练验证：`py_compile`、模型 forward shape、最小观测特征长度、假样本 `Algorithm.learn`。验证结果：`FEATURE_DIM=4833`、logits `(1,85)`、value `(1,1)`、LSTM state `(1,1,512)`、`SAMPLE_DIM=81312`。

### 11.18 2026-05-12 reward v1.1 小幅修订

- `forward` reward 从绝对站位值改为早期推进进度差分，只在前 `1000` 帧生效，避免长期奖励无意义前压；差分后单步信号较小，权重从 `0.03` 调整为 `0.2`。
- `tower_hp_point` 权重从 `5.0` 小幅提高到 `6.0`，继续把胜负目标压在推塔上。
- `ep_rate` 权重从 `0.75` 降到 `0.25`，降低保留能量/蓝量对技能释放探索的抑制风险。
- 本轮不改变特征维度、模型结构、动作空间、样本维度或 checkpoint tensor shape；属于可从当前模型继续训练的小幅 reward 调整。
- 复核 `agent_diy` 投训链路时修齐召唤师技能白名单：`SUMMONER_SKILL_IDS` 覆盖 `80102/80103/80104/80105/80107/80108/80109/80110/80115/80121`，并将 133 训练候选改为包含 `80103` 与 `80108`，避免训练候选与评估默认技能不一致。
- `hero exp` 归一化从固定 `/2000` 改为按当前等级查 `LEVEL_MAX_EXP`，与 reward 经验累计共用同一张表，保持“等级内进度”语义。
- 保留 `mov_spd / 10000`：现有日志中英雄/单位移动速度常见为 `3088~5000`，若按文档草案 `/1000` 会大面积饱和到 1；当前分母更符合实测数值尺度。
- 修正狄仁杰 `133` 三技能 metadata：移除 `long_or_global` tag，仅保留 `damage`，避免与鲁班二技能的长距离/全图先验混淆。
- 恢复/血包打断逻辑不扩维升级：恢复尝试只认恢复槽或治疗术 `80102`，血包尝试用“上一帧血包存在、当前消失、己方更接近血包锚点”启发式记录；`recent_interrupted` 改为尝试后受击且 HP 未形成明显净增才置 1。

### 11.19 2026-05-15 red-camp action mirror 待验证

- 曾尝试在 `agent_diy` 对红方 `legal_action` 与环境动作执行 `i -> 15-i` 镜像，但复核 `hok_semi` 与 `external/test0508_15525/agent_ppo` 后没有找到正式代码对动作 head 做同类在线反镜像；现已回退该改动。
- 当前保守方案：特征侧继续把红方位置统一镜像到蓝方视角；动作侧保持官方 6 维离散协议原样，`action_process` 不再改写 `move_x/move_z/skill_x/skill_z`，样本构造继续使用环境原始 `observation["legal_action"]`。
- `i -> 15-i` 只是在 16 桶低高端严格对称时成立；现有注释把技能/目标中心写作 `(8,8)`，而 `15-8=7`，因此不能静态确认它符合官方动作桶定义。
- 下一步需要平台实测验证：分别在蓝/红方固定输出 `button=4`、`target=1`、`skill_x/skill_z` 为 `8/8`、`0/8`、`15/8` 等动作，记录实际 `dir_skill/pos_skill` 方向、敌方命中与录像偏移，再决定是否引入动作镜像或改为中心保持的自定义映射。
- 验证：本项不能用本地静态检查定论；需要平台实测。Codex 不运行 `python train_test.py`。

### 11.20 2026-05-15 鲁班 1 技能瞄准辅助

- 基于录像观察，鲁班 2 技能与狄仁杰 3 技能没有同类明显偏移，因此不再按全局动作镜像问题处理鲁班 1 技能偏移；当前修订只针对鲁班 1 技能。
- `agent_diy/agent.py` 增加鲁班 1 技能瞄准辅助：仅当策略已经选择 `button=4`、己方英雄为 `112`、敌方英雄在 `8800` 范围内且 `legal_action` 允许时，将 `skill_x/skill_z` 修正为中心桶 `8/8`，并将 `target` 修正为 `enemy_hero`。该规则不强行选择释放 1 技能。
- 瞄准辅助在 `update_status()` 前修改 `act_data`，使训练样本记录的动作与实际执行动作一致；不改变特征维度、动作空间、模型结构或样本维度。
- 技能命中奖励细分已在 11.22 落地；本项只保留鲁班 1 瞄准辅助本身。
- `agent_diy/conf/monitor_builder.py` 新增 `luban_skill1_aim_assist_count`，用于区分“释放了 1 技能”和“瞄准辅助实际生效”。

### 11.21 2026-05-15 reward v1.2 首批落地

- 按本轮决策调整基础权重：`kill` 从 `0.6` 提到 `0.8`，`forward` 从 `0.2` 降到 `0.1`；恢复打断奖励后续再议。
- `cake_pickup` 改成三段事件奖励：拾取前 `hp_rate < 0.9` 给 `0.3`，`hp_rate < 0.5` 给 `0.5`，`hp_rate < 0.2` 给 `1.0`；近满血拾取不再奖励。
- 新增终局稀疏奖励 `win`：胜局 `+3.0`，败局 `-3.0`，由 workflow 在最后一帧注入。选 `3.0` 而非 `5.0`，避免初始 warmup 期过度压过塔血差分。
- 新增 per-key time scale：`kill/death/tower_hp_point/cleanse_success/win/berserk_timing/no_op_streak_penalty` 不衰减，其余沿用 `TIME_SCALE_ARG=8000`；`forward` 继续保留 1000 帧硬截断。
- 新增 `berserk_timing`：召唤师技能 `80110` 使用后 20 帧内进入 `9000` 距离交战给 `+0.5`，否则给 `-0.8` 空放惩罚。
- 新增 `no_op_streak_penalty`：连续 5 帧 `button=0/1` 后，每帧注入 `-0.1`。
- 本轮不改变特征维度、动作空间、模型结构或样本维度，但 reward 分布改变，建议按 warmup 方式接续训练并观察 critic/value loss。

### 11.22 2026-05-15 技能命中奖励细分

- 用英雄+技能槽细分奖励替代通用 `skill_hit_enemy_hero`：鲁班 1 命中敌方小兵 `0.10`，鲁班 1/2/3 命中敌方英雄均为 `0.25`，狄仁杰 1 命中敌方英雄 `0.10`，狄仁杰 3 命中敌方英雄 `0.40`。
- 狄仁杰 2 技能不再进入命中奖励映射；其收益只由现有 `cleanse_success` 表达，避免把二技能当输出技能鼓励。
- 英雄命中判定仍基于 `take_hurt_infos`：攻击者 runtime 对齐、`skillSlot` 对齐、`hurtValue > 0`。鲁班/狄仁杰 3 技能继续保留一次施放周期内只记一次的防刷逻辑。
- 鲁班 1 命中小兵不能依赖 `take_hurt_infos`，因为官方协议中该字段属于 Hero；NPC/小兵没有承伤列表。现由 workflow 在执行动作前记录鲁班 1 的目标/落点坐标；reward 侧只在鲁班 1 成功释放后的 12 帧内，对“动作目标小兵”或“距离该落点 1800 以内”的敌方小兵 HP 下降计奖。这样可以排除无关扫射/普攻导致的小兵掉血。
- 狄仁杰 3 技能新增追击奖励 `direnjie_skill3_followup_damage=0.10`：大招命中后 30 帧（约 1 秒）内，只要我方继续对敌方英雄造成任意正伤害，即额外给一次小奖励。
- 监控新增 `skill_hit_detail` 面板，显示各英雄技能命中奖励；不改变特征维度、动作空间、模型结构或样本维度。

### 11.23 2026-05-15 death 动态倍率

- `death` 基础权重仍保持 `-1.0`，不新增 reward key；在 `reward_process.py` 结算权重时按局势乘动态倍率。
- 炮车帧 `6254` 前死亡倍率 `0.7`，降低早期探索/换血阶段过度保守风险；炮车帧后倍率 `1.3`，强化中后期死亡会丢线权和塔血的代价。
- 如果敌塔血量低于 `25%`，death 倍率回落到 `1.0`，避免终局可换命推塔时模型过度怕死。
- 本修订不改变 reward key、特征维度、动作空间、模型结构或样本维度。

### 11.24 2026-05-15 tower-push 情景奖励

- 新增 `minion_tower_push`：当己方小兵进入敌方塔攻击范围，且敌方英雄死亡或离敌塔超过 `12000` 时，每 `10` 帧评估一次。若正在打塔、敌塔掉血，或己方英雄在攻击范围内且敌塔正在打己方小兵，给 `+0.08`；若还在向敌塔靠近，给 `+0.04`；否则给 `-0.08`，用于抑制兵线进塔后的无意义游走。
- 新增 `enemy_dead_enemy_cake`：炮车帧 `6254` 前，敌方英雄死亡、敌塔范围内至少有 `2` 个己方小兵且敌方血包存在时，每 `10` 帧给小额入侵血包引导。靠近敌方血包给 `+0.04`，已经接近血包 `1500` 范围内给 `+0.06`，不加惩罚。
- 两个新 reward 均不走时间衰减，并加入 `new_reward_signals` 监控；不改变特征维度、动作空间、模型结构或样本维度。
### 11.25 2026-05-15 rule override audit

- Fixed hard-rule sample mismatch: all post-model hard action overrides now set `rule_override_active`, and `build_frame()` marks those frames `is_train=False` so PPO does not learn from action/prob pairs that were not actually executed.
- Fixed auto-cleanse bookkeeping: `cleanse_override_count` now resets per episode and is exported to the rule intervention monitor panel.
- Tightened force-home gating: own tower must be present and at least 40% HP; hero distance uses `collider.location` fallback; hard force-home movement is generated through `legal_action`; own cake state must have been observed and must not respawn within 150 frames.
- Validation: `python -m py_compile agent_diy\agent.py agent_diy\feature\definition.py agent_diy\conf\monitor_builder.py agent_diy\workflow\train_workflow.py` passed; `Config.validate()` passed; targeted rule probes for force-home far/near, raw target legality, and auto-cleanse override passed.

### 11.26 2026-05-15 force-home walk rule

- Replaced the unusable recall-button assumption with a two-phase movement rule: low HP triggers forced walking toward own base; after HP reaches 80%, the rule forces walking back toward the own first-tower coordinate and exits near that point.
- Enemy visibility rule changed as requested: the rule may start when the enemy hero is not visible, or when the visible enemy hero is farther than 9000 raw-coordinate distance.
- The forced action is now `button=2` with legal `move_x/move_z` only; irrelevant skill/target heads are not allowed to block move-rule legality.
- Correction: tower and own-cake gates remain required. Tower/cake ownership and fallback anchors use the same red-mirrored `lane/width` projection as `FeatureProcess`; movement targets are then unprojected back to raw X/Z for the environment move heads.
- Validation: `python -m py_compile ...` passed; `Config.validate()` passed; targeted probes covered blue/red retreat direction, near-enemy non-trigger, 80% HP return transition, tower-near exit, and raw legal-action move legality.

### 11.27 2026-05-15 force-home audit follow-up

- Force-home tower gate now requires own tower HP rate `>= 0.40`.
- Fixed camp matching in `agent_diy/agent.py` rule entity selection to use normalized camp keys, so `1` and `PLAYERCAMP_1` do not diverge.
- Renamed the hard-rule entry from `_maybe_force_recall` to `_maybe_force_home`; forced walking now only sets `rule_override_active`, leaving the legacy `recall_override_active` flag false.
- Removed stale recall-button feature semantics: retreat context now records `button=2` move legality instead of `button=9` recall legality; design notes were updated accordingly.
- Hardened coordinate and slot compatibility in feature/reward/workflow helpers: location reads now prefer `collider.location`, and `skill_state/skillState`, `slot_states/slotStates`, `slot_type/slotType` are accepted where relevant.
- Validation: `python -m py_compile ...` passed; `Config.validate()` passed; targeted probes covered tower HP 39/40/41%, mixed camp encodings, camelCase skill slot fields, and force-home move output.

### 11.28 2026-05-15 rule monitor split

- Split rule-intervention monitoring into per-rule counters: `force_home_override_count`, `force_home_start_count`, `force_home_retreat_count`, `force_home_return_count`, `cleanse_override_count`, and `luban_skill1_aim_assist_count`.
- `rule_override_count` now includes post-model hard overrides from force-home and auto-cleanse; force-home phase counters distinguish walking to base from returning to first tower.
- Validation: `python -m py_compile agent_diy\agent.py agent_diy\workflow\train_workflow.py agent_diy\conf\monitor_builder.py` passed; `Config.validate()` passed; a targeted force-home counter probe covered start/retreat/return increments.

### 11.29 2026-05-15 recover reward and buff whitelist

- Reworked `recover_skill_low_hp` to follow `docs/相关报告/v1.2.md`: low-HP slot 4, summoner heal `80102`, or recover start buff `10000` now opens a pending recover window; reward is paid only after `15` frames if HP increased by at least `200`.
- Added recover monitoring counters: `recover_attempt_count`, `recover_success_count`, and `recover_interrupted_count`; the monitor builder now exposes them in a dedicated `recover_debug` panel.
- Updated `BUFF_WHITELIST_96` composition without changing feature dimensions: all 22 missing observed IDs from v1.2 are included, 18 lowest-priority old hero candidates are removed, and the retained fallback IDs are `11111`, `911220`, `914250`, `112110`, `133030`.
- Validation: `python -m py_compile agent_diy\conf\conf.py agent_diy\feature\reward_process.py agent_diy\feature\feature_process.py agent_diy\conf\monitor_builder.py` passed; `Config.validate()` passed; whitelist probe confirmed 96 IDs, no missing required IDs, and no removed low-priority IDs; targeted reward probe covered interrupted, `80102`, and `10000/10010` recover paths.

### 11.30 2026-05-15 runtime log cleanup

- Removed high-frequency training-loop logs from `agent_diy`: per-episode `env_config`, full `usr_conf`, `init_config`, `training_metrics`, and disabled `OBS_DEBUG` observation dump calls/code.
- Removed repeated `load_model(latest)` skip logs and model-pool opponent delegation logs from `agent_diy/agent.py`; kept the concise episode-end summary and real warning/error paths.
- Validation: `python -m py_compile agent_diy\agent.py agent_diy\workflow\train_workflow.py agent_diy\workflow\env_conf_manager.py` passed; `Config.validate()` passed; `rg` confirms the removed debug strings no longer exist in active training code.

### 11.31 2026-05-15 third-run opponent pool

- Updated training opponents to `mixed`: `selfplay` 55%, model `266606` 30%, model `265636` 15%.
- Updated eval opponents to uniform random among `common_ai`, `266606`, `265636`, and `263170`.
- Updated `kaiwu.json` model pool to `[266606, 265636, 263170]`.
- Validation: parsed `agent_diy/conf/train_env_conf.toml` and `kaiwu.json` with Python and asserted the configured pools/weights match the third-run plan.

### 11.32 2026-05-15 pre-training review fixes

- Fixed a hard-rule training-signal mismatch: Luban skill-1 aim assist now marks `rule_override_active` and increments `rule_override_count`, so frames whose sub-actions were rewritten are excluded from PPO gradients like force-home and auto-cleanse frames.
- Moved rule-override flag reset to the start of `predict`/`exploit`, preserving any hard-rule flag set during post-model action rewriting.
- Aligned `eval_interval` semantics with `train_env_conf.toml`: `eval_interval = 20` now means one eval every 20 episodes, not every 21; `random_eval_start` is sampled in `[0, 19]`.
- Validation: full `agent_diy` py_compile passed; `Config.validate()` passed; opponent-pool parse assertions passed; model forward shape probe returned logits `(1,85)`, value `(1,1)`, and LSTM states `(1,1,512)`; targeted probes covered Luban aim-assist override marking and eval interval parsing.

### 11.33 2026-05-16 v2.1 Di Renjie/Cleanse/Berserk landing

- Implemented the 133v133 reverse skill-2 legal-action mask from `docs/相关报告/v2.1.md`: if the enemy 133 is alive, level >= 4, and has not used slot 3 within 300 frames, button 5 is masked before policy inference. The mask is skipped immediately after observing enemy slot-3 `succUsedInFrame > 0`.
- Kept the positive auto-cleanse override for true ult-hit windows, and added counters for skill-2 blocking, total skill-2 commands, outside-window skill-2 commands, and cleanse rate.
- Changed `cleanse_success` to `cleanse_main - skill2_misuse`; enemy cleanse of our ult is now exported as `enemy_cleansed_us_count` only.
- Added `berserk_no_damage_penalty`: after summoner 80110, if no damage to an enemy hero, soldier, or tower is observed within 60 frames, the event value is `-1.0` with no time decay.
- Changed `direnjie_skill3_followup_damage` from a single post-ult credit to per-frame credit capped at 8 events within the 30-frame follow-up window.
- Validation: `python -m py_compile agent_diy\agent.py agent_diy\conf\conf.py agent_diy\conf\monitor_builder.py agent_diy\feature\reward_process.py agent_diy\workflow\train_workflow.py` passed; `Config.validate()` passed with `FEATURE_DIM=4833`, `LEGAL_ACTION_DIM=85`, `RAW_LEGAL_ACTION_DIM=184`, `SAMPLE_DIM=81312`; model forward returned logits `(1,85)`, value `(1,1)`, and LSTM states `(1,1,512)`; targeted probes covered skill-2 mask/unmask, berserk no-damage penalty, and 8-credit Di Renjie follow-up cap.
### 11.34 2026-05-16 force-home cake gate tightening

- Changed force-home continuation semantics: own-cake unavailable remains a start gate, but an active `retreat` phase no longer exits when own cake respawns. Active retreat still requires own tower context and legal movement.
- Reviewed red/blue force-home coordinate mapping against official docs and current feature projection: official `camp=1` is blue, `camp=2` is red; feature projection mirrors camp 2 to camp 1 view; force-home base target unprojects `SELF_BASE_ANCHOR` to raw negative coordinates for blue and raw positive coordinates for red when `hero_camp` is correct.
- Residual risk: force-home uses `self.hero_camp` from `reset()`, while `FeatureProcess` can resync from per-frame `observation.player_camp`. If platform observations ever drift from reset camp, force-home and feature projection can desync; this needs a follow-up hardening pass.
- Validation: `python -m py_compile agent_diy\agent.py` passed; `Config.validate()` passed; targeted probe confirmed retreat persists after own-cake respawn and blue/red base targets produce opposite raw move directions.

### 11.35 2026-05-16 force-home waypoints and camp hardening

- Hardened force-home camp resolution: each frame now prefers `observation.player_camp`, cross-checks it against the observed player hero camp when `player_id` is available, updates `self.hero_camp` only on a valid camp, and disables force-home on camp conflicts instead of guessing.
- Replaced direct base-line retreat with own-perspective `lane/width` waypoints: `retreat_entry=(-20000,3500)`, `retreat_spring=(-40000,3500)`, and `return=(-15000,3500)`. These anchors are unprojected through the current camp, so red side mirrors automatically.
- The entry waypoint uses a `3000` radius to enter `retreat_spring`; the spring stage does not exit by distance while HP is below `80%`, preventing an early stop outside fountain. The configured spring arrival radius is capped at `1000` for future checks.
- Validation: `python -m py_compile agent_diy\agent.py agent_diy\conf\conf.py` passed; `Config.validate()` passed; targeted probe covered start->entry, entry->spring, low-HP spring hold, recovered return, blue/red mirrored waypoint raw coordinates, and camp-conflict disable.

### 11.36 2026-05-17 force-home legal direction fallback

- Fixed hard-rule direction legalization: direction heads now choose the legal bucket nearest to the preferred direction instead of falling back to the first legal bucket. This avoids return actions such as `[15,15]` being rewritten to `[0,0]` when only a nearby bucket like `14` is legal.
- Validation: `python -m py_compile agent_diy\agent.py` passed; `Config.validate()` passed; targeted probe confirmed preferred direction `15` with legal buckets `[0,14]` selects `14`, while target-head fallback remains first-legal.

### 11.37 2026-05-17 force-home start gates by cannon phase

- Reworked force-home start gates: before cannon frame `6254`, force-home starts only when HP `<35%`, own cake is absent, and enemy hero is invisible or farther than `8800` in own-perspective coordinates. After cannon frame, force-home starts only when HP `<20%`, own tower HP is `>35%`, own cake is absent, and the same enemy safety gate holds.
- Active force-home phases no longer re-check start gates such as cake, tower HP, or enemy distance. Retreat switches to return at HP `>=90%`; return exits once own-perspective lane reaches `-15000`.
- Hardened HP reads for `max_hp/maxHp/maxHP` and visibility reads for `camp_visible/campVisible`.
- Validation: `python -m py_compile agent_diy\agent.py agent_diy\conf\conf.py` passed; `Config.validate()` passed; targeted probes covered pre/post-cannon HP thresholds, cake gate, enemy near/invisible gates, tower HP 35/36%, active retreat gate bypass, 90% return transition, and lane `-15000` return exit.

### 11.38 2026-05-17 force-home opening path cache

- Added a force-home path cache from the first `600` frames. It stores sparse own-perspective `lane/width` points from base toward lane, caps the cache at `10` points, and requires at least `5` valid points spanning base side to lane `-15000`.
- Force-home now uses the cached path when valid: retreat targets walk backward along the opening path; after HP reaches `90%`, return targets walk forward along the same path until own-perspective lane reaches `-15000`. The previous `(-20000,3500) -> (-40000,3500)` retreat and `(-15000,3500)` return anchors remain as fallback when the cache is incomplete.
- Preserved the `-15000` crossing point even when it is closer than the normal `2500` sparse-sampling distance, so the path does not become invalid after stepping from around `-16000` to `-14500`.
- Rechecked HP semantics against the official protocol and local feature code: force-home uses observed `hp / max_hp` (also accepting `maxHp/maxHP`); the `12000` constant is only a normalization scale / tower reference, not the hero HP denominator.
- Validation: `python -m py_compile agent_diy\agent.py agent_diy\conf\conf.py` passed; `Config.validate()` passed; targeted probes covered path recording and cap, mandatory `-15000` crossing retention, path-based retreat/return transitions, HP ratio with `max_hp/maxHp`, red-side raw mirroring, and red-side enemy-distance checks in own-perspective coordinates.

### 11.39 2026-05-17 distance/HP audit

- Audited the `8800` safety threshold: force-home computes enemy safety as Euclidean distance in own-perspective `lane/width`, i.e. `sqrt((self_lane-enemy_lane)^2 + (self_width-enemy_width)^2) > 8800`. The value is the same gameplay-unit threshold already used for tower/medium-range fallback contexts, while official docs expose `attack_range/sight_area` fields but do not provide a fixed global range.
- Confirmed most feature distances already use the remapped own-perspective coordinates from `FeatureProcess._project_location`; reward distances that stay in raw `x/z` are only magnitude/progress checks where Euclidean distance is invariant under the lane/width rotation and red-side mirror.
- Hardened HP ratio reads in `feature_process.py` and `reward_process.py` to accept `hp/HP` and `max_hp/maxHp/maxHP`, matching the force-home compatibility without changing feature dimensions.
- Replaced the `safe_last_hit` reward's hard-coded own-tower `8800` check with `_tower_attack_range(main_tower)`, so it uses runtime tower `attack_range` when present and falls back to `8800` only when missing.
- Replaced the remaining direct HP liveness reads in force-home/skill-2 blocking and reward helpers with the shared HP helpers, keeping official lowercase `hp/max_hp` behavior unchanged while accepting `HP/maxHP` variants.
- Validation: `python -m py_compile agent_diy\agent.py agent_diy\conf\conf.py agent_diy\feature\feature_process.py agent_diy\feature\reward_process.py agent_diy\workflow\train_workflow.py agent_diy\model\model.py agent_diy\algorithm\algorithm.py` passed; `Config.validate()` passed; targeted probes covered `maxHp`-only feature/reward/force-home HP reads, blue/red force-home enemy-distance checks around the `8800` threshold, dynamic tower range for `safe_last_hit`, and force-home `retreat_entry`/`retreat_spring` switching to `return` at exactly `90%` HP.

### 11.40 2026-05-17 force-home trigger monitor

- Added `force_home_trigger_count` as a dedicated monitor counter for the force-home rule. It increments exactly when the start gates pass and the state machine enters force-home, independent of whether the first forced move is legal on that frame.
- Exported `force_home_trigger_count` through `train_workflow.py` and added it to the `rule_intervention` monitor panel next to the existing `force_home_start/retreat/return/override` counters.
- Validation: `python -m py_compile agent_diy\agent.py agent_diy\workflow\train_workflow.py agent_diy\conf\monitor_builder.py` passed; `Config.validate()` passed; targeted probe confirmed the trigger counter increments once on rule start and does not increment again during active retreat frames.

### 11.41 2026-05-17 money-frame feature fix

- Fixed `agent_diy` money-frame features to use non-negative per-frame deltas from `money_cnt/moneyCnt` via `_money_total()` history instead of falling back to current spendable `money`. This preserves the existing 12 affected feature slots and their scales/buckets while restoring the intended frame-income semantics.
- Validation: `python -m py_compile agent_diy\feature\feature_process.py` passed; `Config.validate()` passed with `FEATURE_DIM=4833` and `SAMPLE_DIM=81312`; targeted probe confirmed first-frame income is `0`, `800->850` income is `50`, and negative/current-money fallback semantics no longer drive the frame-income path.

### 11.42 2026-05-17 PPO ratio epsilon symmetry

- Fixed PPO ratio log-prob stability in `agent_diy/model/model.py`: selected old-policy probability is now summed before adding `LOG_EPSILON`, matching the current-policy path. This removes the previous `old_selected_prob + action_head_size * epsilon` denominator bias.
- Validation: `python -m py_compile agent_diy\model\model.py` passed; `Config.validate()` passed with `FEATURE_DIM=4833` and `SAMPLE_DIM=81312`; targeted ratio probe confirmed symmetric `(p + eps) / (old_p + eps)` behavior.

### 11.43 2026-05-17 force-home cache-only path and red direction

- Removed the hard-coded force-home coordinate fallback. Force-home now starts only when the recorded opening path cache is valid; if the cache is missing or becomes invalid during an active phase, the rule clears and returns control to the policy.
- Removed the previous `(-20000,3500)`, `(-40000,3500)`, and `(-15000,3500)` fallback anchors and their unused waypoint radii from config.
- Fixed forced move direction generation to compute direction buckets from red/blue mirrored own-perspective raw coordinates. The cached path still uses own-perspective `lane/width`, while action heads now receive the same movement bins for equivalent blue/red own-perspective situations.
- Validation: `python -m py_compile agent_diy\agent.py agent_diy\conf\conf.py agent_diy\workflow\train_workflow.py agent_diy\conf\monitor_builder.py` passed; `Config.validate()` passed; targeted probe confirmed no-path force-home does not trigger/count, valid cached path starts retreat, HP `>=90%` enters return, and blue/red equivalent own-perspective geometry produces identical move direction buckets.

### 11.44 2026-05-17 force-home anti-stuck path following

- Increased the force-home opening path cache from `10` to `24` points and lowered sparse sampling from `2500` to `1200` gameplay units, so early fountain/tower-side bends are less likely to be collapsed into long straight segments.
- Changed path compression from uniform downsampling to importance-based retention: first/last points, strong turns, and the `lane=-15000` return-exit crossing are prioritized when the cache exceeds the point cap.
- Added path-following anti-stuck behavior. Retreat/return now skip waypoints already within the waypoint radius, and if distance to the current waypoint fails to improve by `300` units for `45` consecutive rule frames, the rule skips to the next waypoint in the travel direction.
- Validation: `python -m py_compile agent_diy\agent.py agent_diy\conf\conf.py agent_diy\workflow\train_workflow.py agent_diy\conf\monitor_builder.py` passed; `Config.validate()` passed; targeted probe confirmed turn-aware compression keeps bend/crossing points, waypoint-radius lookahead advances the path index, no-progress detection skips a stuck waypoint, no-path force-home still does not trigger, and blue/red equivalent own-perspective movement still produces identical direction buckets.

### 11.45 2026-05-17 force-home path validity lane

- Split the cached-path validity threshold from the return-rule exit threshold. A path now needs to reach own-perspective `lane >= -13000` to be considered valid, while active return still exits at `lane >= -15000`.
- Validation: `python -m py_compile agent_diy\agent.py agent_diy\conf\conf.py` passed; `Config.validate()` passed; targeted probe confirmed a path ending at `-14000` is invalid and a path ending at `-13000` is valid.

### 11.46 2026-05-17 Di Renjie delayed cleanse

- Delayed the 133v133 auto-cleanse override by one frame. Enemy ult hit frames now record a pending cleanse and mask skill 2 only for policy inference on that frame; the hard override emits skill 2 on the next frame after rechecking cooldown and legal action.
- Validation: `python -m py_compile agent_diy\agent.py` passed; `Config.validate()` passed; targeted probe confirmed ult-hit frame only sets pending/masks policy skill 2, and frame `+1` emits `[5, 15, 15, 15, 15, 2]`.

### 11.47 2026-05-17 force-home v3.0 gates

- Implemented `docs/相关报告/v3.0回城规则.md` for `agent_diy`: force-home is disabled from frame `6000`, early HP trigger is now `<20%`, return starts at `>=80%`, enemy safe range is `10000`, and own tower HP gate is `>40%`.
- Added v3.0 hard start gates for recovery priority and lane state: own cake present, slot-4 recover available, heal summoner `80102` ready, recent recover/cake cooldown, enemy minions under own tower, and enemy-dead lane minions all prevent force-home start.
- Added shallow-retreat exits for own cake / recovery availability before lane `-20000`, plus state tracking for enemy death count, protocol `revive_time`, `frame_action.dead_action`, recover success, and own-cake pickup proximity.
- Validation: `python -m py_compile agent_diy\agent.py agent_diy\conf\conf.py` passed; `Config.validate()` passed; targeted force-home helper probes covered tower-area minion counting, enemy-dead lane minion gate, recover cooldown escape, `hp/dead_cnt/revive_time/dead_action` enemy death paths, and heal summoner readiness.

### 11.48 2026-05-18 Di Renjie ult follow-up window

- Changed `DI_RENJIE_SKILL3_FOLLOWUP_WINDOW` from `30` to `150` frames. The follow-up reward still pays per detected damage frame, keeps weight `0.10`, and remains capped at `8` events per ult-hit window.

### 11.49 2026-05-18 Di Renjie early follow-up window

- Added `DI_RENJIE_SKILL3_FOLLOWUP_WINDOW_EARLY = 60`. Di Renjie ult hits before `CANNON_FRAME` now use a 60-frame follow-up reward window, while later ult hits keep the 150-frame window. Weight `0.10` and cap `8` are unchanged.

### 11.50 2026-05-18 opening unstuck assist

- Added a short opening anti-stuck assist for frames `180-300`: if the hero's own-perspective movement between consecutive policy frames is `<=30` units and the assist is not cooling down, the executed action is overridden to a legal move in positive `lane` direction with unchanged `width`.
- The forward target is built in own-perspective `lane/width` and unprojected through the existing red/blue mirror path, so equivalent blue/red states produce the same own-forward move bins without making red side walk backward.
- Exported `opening_unstuck_count` through workflow monitor data and the `rule_intervention` panel.
- Validation: `python -m py_compile agent_diy\agent.py agent_diy\conf\conf.py agent_diy\conf\monitor_builder.py agent_diy\workflow\train_workflow.py` passed; `Config.validate()` passed; targeted probe confirmed blue and red opening stuck states both emit `[2, 15, 15, 8, 8, 0]` in own-perspective action bins while raw forward targets are mirrored.

### 11.51 2026-05-20 forward reward window

- Changed `REMOVE_FORWARD_AFTER` from `1000` to `500`, so the geometric forward-progress reward is limited to the earlier opening phase.

### 11.52 2026-05-20 Di Renjie skill-2 unmask window

- Changed the 133v133 reverse skill-2 mask so enemy Di Renjie ult cast opens a policy skill-2 window only from frame `+60` through `+420`; before `+60`, after `+420`, or before any tracked enemy ult cast, skill 2 remains masked by the reverse rule.
- The true ult-hit path is unchanged: `take_hurt_infos` must show the enemy runtime as attacker with `skillSlot == 3`; the hit frame masks policy skill 2 and the next frame hard-overrides to `[5, 15, 15, 15, 15, 2]` if legal and available.
- Validation: `python -m py_compile agent_diy\agent.py agent_diy\conf\conf.py` passed; `Config.validate()` passed; targeted probe covered `last_cast + 59/60/420/421` mask boundaries and enemy-ult-hit detection.

### 11.53 2026-05-20 Di Renjie ult-hit detection hardening

- Hardened the auto-cleanse ult-hit detector to mirror the skill-hit reward evidence stack: `take_hurt_infos/takeHurtInfos` with string-normalized attacker runtime and slot 3, enemy `hit_target_info/hitTargetInfo` targeting our runtime with slot 3, and enemy slot-3 `hitHeroTimes` delta as a final fallback.
- Validation: `python -m py_compile agent_diy\agent.py agent_diy\conf\conf.py` passed; `Config.validate()` passed; targeted probe covered mixed string/int runtime hurt events, `hitTargetInfo`, `hitHeroTimes` increment-only detection, and wrong-slot/wrong-target negatives.

### 11.54 2026-05-20 duel summoner training and reward

- Implemented `docs/相关报告/v3.1召唤师技能训练方案.md`: main training candidates are now `80110` 狂暴 and `80121` 弱化, treated as one hero-trade summoner category.
- Added workflow skill forcing by episode context: self-play uses the same skill on both sides and alternates through the duel-summoner list; historical/common-ai training samples both sides independently from the same list; train-time eval cycles through all skill-pair combinations.
- Replaced the old `berserk_timing` / `berserk_no_damage_penalty` reward keys with `duel_summoner_timing`: a cast opens a 90-frame window, rewards `+0.8` only when distance reaches `<=8500`, hero damage reaches the 20% max-HP scale, and at least 5 hero damage interactions occur; light poke is neutral and obvious far empty use is `-0.4`.
- Validation: `python -m py_compile agent_diy\agent.py agent_diy\conf\conf.py agent_diy\conf\monitor_builder.py agent_diy\feature\reward_process.py agent_diy\workflow\train_workflow.py` passed; `Config.validate()` passed; targeted probes covered self-play same-skill alternation, history/eval skill selection, real duel reward `+0.8`, poke reward `0`, and weak/far empty use `-0.4`.

### 11.55 2026-05-20 opponent pool update

- Updated this training round to `mixed` opponents with only `selfplay` and model `271476`, weighted `0.55 / 0.45`.

- Updated eval opponents to `common_ai`, `271476`, and `267822`; updated `kaiwu.json` model pool to `[271476, 267822]`.
- Replaced coarse eval summoner monitors with matchup/skill monitors `eval_m{my}_o{opp}_s{skill}_{count,win}` covering 112/133 matchups and 80110/80121; each win/count ratio is the train-time eval win rate for that matchup and skill.
- Added `SUMMONER_SKILL_MATCHUP_WINRATE` and made formal eval/exam `init_config()` select the higher-winrate duel summoner for `(my_hero, opponent_hero)`, falling back to default 80110 on ties or missing data.

### 11.56 2026-05-20 monitor panel cleanup

- Reorganized `agent_diy/conf/monitor_builder.py` into diagnostic panels: PPO health, match result, objective/economy, combat reward, skill-hit reward, action buttons, skill usage, target selection, rule intervention, summoner/recover, and per-matchup eval panels.
- Removed non-displayed monitor-only values from workflow aggregation: episode count, broad summoner-skill one-hot output, and last-hit debug counters that are not used by reward computation, model inputs, or training samples.
- Validation: `python -m py_compile agent_diy\conf\monitor_builder.py agent_diy\conf\conf.py agent_diy\workflow\train_workflow.py agent_diy\algorithm\algorithm.py` passed; `Config.validate()` passed; a stubbed `MonitorConfigBuilder` probe confirmed the panel config builds with 14 panels and 101 displayed metrics, with retained producer metrics covered by the display set.

### 11.57 2026-05-20 Di Renjie scheduled cleanse fallback

- Changed the 133v133 skill-2 unmask window to enemy ult cast frame `+12` through `+420`, replacing the previous `+60` start.
- Added a one-shot scheduled cleanse fallback: for each tracked enemy Di Renjie ult cast, the first observed frame at or after `+12` attempts to hard-overwrite skill 2 if it is available and legal. If skill 2 is unavailable at that first eligible frame, the fallback is consumed for that ult and the policy can still use skill 2 naturally during the unmasked window.
- Guarded force-home from overriding a cleanse hard action on the same frame.
- Validation: `python -m py_compile agent_diy\agent.py agent_diy\conf\conf.py` passed; `Config.validate()` passed; targeted probe covered `+11/+12/+420/+421` mask boundaries, one-shot `+12` fallback success, unavailable-at-`+12` no-retry behavior, and hardened ult-hit detection.

### 11.58 2026-05-22 force-home economy and post-kill lane clear

- Replaced the frame-6000 force-home shutdown with an economy gate: if the controlled hero has `moneyCnt/money_cnt/money >= 2900`, force-home clears any active phase and does not start.
- Added a post-kill lane-clear branch: after confirmed enemy hero death, if lane `[-13000, 13000]` has no living enemy minions and HP is `<40%`, force-home can start without checking own cake or slot-4 recover availability. This branch is windowed by `FORCE_HOME_POST_KILL_WINDOW_FRAMES=900`.
- Changed force-home return transition from `80%` HP to `70%` HP.
- Removed force-home's special-case treatment of heal summoner `80102`; the final candidate will not carry or train this skill, so recovery-priority checks now only consider slot-4 recovery and recovery buffs.
- Validation: `python -m py_compile agent_diy\agent.py agent_diy\conf\conf.py` passed; `Config.validate()` passed; targeted probes covered post-kill HP 39/41%, lane uncleared, money 2900 disable, ordinary low-HP cake gate, enemy death tracking, clearing post-kill force-home state, and ignored heal-summoner readiness.

### 11.59 2026-05-22 late-game own-tower minion defense reward

- Added two late-game defensive scenario rewards in `agent_diy`: `enemy_minion_tower_front` penalizes living enemy minions in `(R, 1.2R]` around our tower, and `enemy_minion_under_own_tower` penalizes living enemy minions inside our tower range `R`.
- Both rewards start from `CANNON_FRAME`, evaluate every `10` frames, require our hero and tower to be alive, cap counts at `3`, and use tower-HP multipliers `x1.5` below `50%` and `x2.0` below `25%`.
- Added defense pressure monitors for front count, under-tower count, multiplier, and weighted reward values.
- Validation: `python -m py_compile agent_diy\conf\conf.py agent_diy\feature\reward_process.py agent_diy\conf\monitor_builder.py` passed; `Config.validate()` passed; targeted reward probe covered pre-cannon no-op, `1.2R` front boundary, tower-range split, 10-frame interval gating, count caps, and low-tower HP multiplier.

### 11.60 2026-05-22 summoner defaults and 133v112 sampling

- Updated formal eval/exam summoner defaults from the matchup winrate table: `112v112`, `112v133`, and `133v133` now prefer `80121`; `133v112` keeps `80110`.
- Training still samples both duel summoners `80110/80121`; train-time eval monitor panels still enumerate both skills through workflow forced-skill logic.
- Added configurable lineup sampling weights and set `133v112` to `2x` while keeping the other three matchups at `1x`.
- Validation: `python -m py_compile agent_diy\conf\conf.py agent_diy\feature\definition.py agent_diy\workflow\train_workflow.py agent_diy\agent.py` passed; `Config.validate()` passed; a stubbed iterator probe confirmed 50 draws produce `133v112:20` and each other matchup `10`, and default skill selection resolves to `{112v112:80121, 112v133:80121, 133v112:80110, 133v133:80121}`.

### 11.61 2026-05-22 opponent pool for next run

- Updated this training round to `mixed` opponents with `selfplay` and model `276301`, weighted `0.60 / 0.40`.
- Updated eval opponents to `common_ai`, `276301`, and `272245`; updated `kaiwu.json` model pool to `[276301, 272245]`.
- Validation: parsed `agent_diy/conf/train_env_conf.toml` and `kaiwu.json`, confirming train opponents, weights, eval opponents, and model pool match the intended IDs.

### 11.62 2026-05-22 grass field compatibility

- Hardened the hero grass-state feature to read both `is_in_grass` and `isInGrass` without changing the frozen feature layout.
- Validation: `python -m py_compile agent_diy\feature\feature_process.py` passed; `Config.validate()` passed with `FEATURE_DIM=4833` and `SAMPLE_DIM=81312`; targeted probe confirmed snake-case and camel-case grass fields both set hero offset `518`.

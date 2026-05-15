# `model.py` 优化建议（平台稳定版）

  

> 本文档列出对 [`model.py`](./model.py) 的优化建议，所有修改按 **对训练平台接口的影响** 分类，仅推荐 **不改 sample 格式 / 不改 action 接口 / 不改 LSTM 状态形状 / 不改特征维度** 的修改。

  

---

  

## 🚫 不要触碰的接口（平台依赖）

  

以下内容由平台 sample buffer 和数据通路依赖，**严禁修改**：

  

| 接口 | 位置 | 为何不能改 |

|------|------|-----------|

| `LABEL_SIZE_LIST = [12,16,16,16,16,9]` | `conf/conf.py` | 决定 sample 中 label / legal_action / old_prob 的形状 |

| `SERI_VEC_SPLIT_SHAPE` | `conf/conf.py` | 输入特征维度，影响 obs_builder 和平台数据通路 |

| `LSTM_UNIT_SIZE = 512` 及 LSTM `(h, c)` 形状 | `conf/conf.py`, `model.py` | LSTM 状态以 `(h, c)` 二元组形式存于每个 sample |

| `LSTM_TIME_STEPS = 16` | `conf/conf.py` | 影响 sample 拼接长度 |

| `DATA_SPLIT_SHAPE` | `conf/conf.py` | sample 序列化格式 |

  

由此衍生 **不能做** 的优化：

- ❌ LSTM → GRU（状态形状变化）

- ❌ Target 维度 9 → 7（动作空间变化）

- ❌ Value loss clipping（需要 sample 中加 `v_old`）

- ❌ Auxiliary losses（需要新增 sample 字段）

- ❌ 改 `LSTM_TIME_STEPS`

  

---

  

## ✅ 推荐修改清单

  

### 🟢 第一阶段：可热替换、马上见效（< 50 行）

  

这些修改**数学上等价**或**仅改 loss 内部**，**可以加载已有 checkpoint 继续训练**。

  

---

  

#### #1. 修正 Target Attention 为标准 scaled dot-product

  

**位置：** [`model.py:378-388`](./model.py#L378-L388)

  

**问题：**

- `softmax(q, dim=-1)` 在 query 特征维度上 softmax 不是标准 attention

- 缺少 `/√d_k` 缩放因子

- 几何意义不清晰

  

**修改：**

  

```python

# 原代码（删除这几行）：

# lstm_tar_embed_result = self.lstm_tar_embed_mlp(reshape_lstm_outputs_result)

# reshape_label_result = lstm_tar_embed_result.reshape(-1, self.target_embed_dim, 1)

# ulti_tar_embedding = self.target_embed_mlp(tar_embedding)

# ulti_tar_embedding = nn.functional.softmax(ulti_tar_embedding, dim=-1)

# label_result = torch.matmul(ulti_tar_embedding, reshape_label_result)

  

# 新代码：

import math # 文件顶部加

# Query: 9 个候选目标的特征

q = self.target_embed_mlp(tar_embedding) # (B*L, 9, 32)

# Key: 来自 LSTM 的全局意图

k = self.lstm_tar_embed_mlp(reshape_lstm_outputs_result) # (B*L, 32)

k = k.unsqueeze(2) # (B*L, 32, 1)

# 标准 scaled dot-product attention

label_result = torch.matmul(q, k) / math.sqrt(self.target_embed_dim) # (B*L, 9, 1)

target_output_dim = int(np.prod(label_result.shape[1:]))

reshape_label_result = label_result.reshape(-1, target_output_dim) # (B*L, 9)

result_list.append(reshape_label_result)

```

  

**安全性：**

- ✅ 输出 shape 仍为 `(B*L, 9)`

- ✅ 下游 legal_action mask 不变

- ✅ sample 格式不变

- ⚠️ 因网络结构改变，建议从头训练（虽然可以热替换，但 attention 行为变化大）

  

**预期收益：** ⭐⭐⭐⭐⭐ target 选择质量提升

  

---

  

#### #2. 修复 PPO Clip 多余的 clamp

  

**位置：** [`model.py:513-520`](./model.py#L513-L520)

  

**问题：** `clip_ratio = ratio.clamp(0.0, 3.0)` 是非标准的额外 clip，与下面的 PPO clip 重复。

  

**修改（方案 A — 简单去除）：**

  

```python

ratio = torch.exp(final_log_p)

# 删除这行: clip_ratio = ratio.clamp(0.0, 3.0)

  

surr1 = ratio * advantage # 改用原始 ratio

surr2 = ratio.clamp(1.0 - self.clip_param, 1.0 + self.clip_param) * advantage

temp_policy_loss = -torch.sum(

torch.minimum(surr1, surr2) * (weight_list[task_index].float()) * 1

) / torch.maximum(torch.sum((weight_list[task_index].float()) * 1), torch.tensor(1.0))

```

  

**修改（方案 B — Dual-clip PPO，绝悟原论文方案）：**

  

```python

ratio = torch.exp(final_log_p)

surr1 = ratio * advantage

surr2 = ratio.clamp(1.0 - self.clip_param, 1.0 + self.clip_param) * advantage

surr3 = 3.0 * advantage # dual clip 下界

  

# adv < 0 时再加一层下界，防止 ratio 过大造成大幅负更新

clipped = torch.where(

advantage < 0,

torch.max(torch.min(surr1, surr2), surr3),

torch.min(surr1, surr2)

)

temp_policy_loss = -torch.sum(

clipped * (weight_list[task_index].float()) * 1

) / torch.maximum(torch.sum((weight_list[task_index].float()) * 1), torch.tensor(1.0))

```

  

**安全性：** ✅ 纯 loss 计算，sample 与接口完全不变。可以热替换。

  

**预期收益：** ⭐⭐⭐⭐ 训练稳定性提升

  

---

  

#### #3. 修复 legal_action mask 数值常量

  

**位置：** [`model.py:480, 483`](./model.py#L480)

  

**问题：** `boundary = 10**20` 数值过大，FP16 / AMP 下会溢出，FP32 下也容易在 `exp` 后产生 inf。

  

**修改：**

  

```python

boundary = torch.pow(torch.tensor(10.0), torch.tensor(8.0)) # 1e20 → 1e8

```

  

**安全性：** ✅ 纯数值常量，完全安全。可以热替换。

  

**预期收益：** ⭐⭐⭐ 避免训练偶发 NaN

  

---

  

#### #4. 向量化同类实例的 MLP 调用

  

**位置：** [`model.py:248-325`](./model.py#L248-L325)

  

**问题：** 4 个小兵 / 9 个子弹 / 2 个英雄都是 Python `for` 循环逐个过 MLP，GPU 利用率低。

  

**修改思路：** 把同类多实例 stack 后批量过 MLP。例如敌方小兵（原 [model.py:279-287](./model.py#L279-L287)）：

  

```python

# 原代码：

# soldier_emy_result_list = []

# for index in range(len(_soldier_5_8)):

# soldier_emy_mlp_out = self.process_sub_feature(_soldier_5_8[index], self.soldier_mlp, True)

# soldier_emy_fc_out = self.soldier_emy_fc(soldier_emy_mlp_out)

# soldier_emy_result_list.append(soldier_emy_fc_out)

# tar_embed_list.append(soldier_emy_fc_out)

# soldier_emy_concat_result = torch.cat(soldier_emy_result_list, dim=1)

# reshape_emy_soldier = soldier_emy_concat_result.reshape(-1, 4, 32)

# soldier_emy_concat_result, _ = reshape_emy_soldier.max(dim=1)

  

# 新代码：

soldier_emy_stack = torch.stack(_soldier_5_8, dim=1) # (B*L, 4, dim)

B_, N_, D_ = soldier_emy_stack.shape

flat = soldier_emy_stack.reshape(B_ * N_, D_) # (B*L*4, dim)

out_flat = self.process_sub_feature(flat, self.soldier_mlp, True)

out_flat = self.soldier_emy_fc(out_flat) # (B*L*4, 32)

soldier_emy_result = out_flat.reshape(B_, N_, 32) # (B*L, 4, 32)

  

# 收集 tar_embed（保持原有顺序）

for i in range(N_):

tar_embed_list.append(soldier_emy_result[:, i, :]) # target 3,4,5,6

  

# 池化

soldier_emy_concat_result = soldier_emy_result.max(dim=1).values # (B*L, 32)

```

  

类似地可改 `_hero_emy`, `_hero_frd`, `_soldier_1_4`, `_organ_1`, `_organ_2`, `_bullet_1_9`, `_bullet_10`。

  

**安全性：**

- ✅ **数学等价**（同样的权重应用在同样的输入上）

- ✅ 接口完全不变，sample 不变

- ✅ **可以加载已有 checkpoint** —— 建议改完后用旧 ckpt 跑一次推理，断言新旧输出 `torch.allclose(old, new)` 验证等价

  

**预期收益：** ⭐⭐⭐⭐ 推理速度提升 20-40%

  

---

  

#### #5. Entropy 系数 decay

  

**位置：** Algorithm 中调用 `model.var_beta` 的地方（不在 `model.py`，请在 `algorithm/algorithm.py` 找）

  

**问题：** `var_beta` 固定，训练后期探索过度。

  

**修改：**

  

```python

# 在每个训练 step 更新 var_beta

beta_start = Config.BETA_START

beta_end = Config.BETA_END if hasattr(Config, 'BETA_END') else beta_start * 0.1

decay_rate = 0.9999 # 或按总步数线性衰减

self.model.var_beta = max(beta_end, beta_start * (decay_rate ** training_step))

```

  

**安全性：** ✅ 单个 scalar，平台完全不感知。

  

**预期收益：** ⭐⭐⭐ 探索/利用平衡，后期收敛更稳

---

  

#### #6. 主干通路加 LayerNorm

  

**位置：** [`model.py:597-632`](./model.py#L597-L632) 的 `MLP` 类

  

**问题：** 整个网络无任何 normalization，深层 MLP 训练时梯度尺度不稳。

  

**修改 1：** 扩展 `MLP` 类，加 `use_layer_norm` 参数：

  

```python

class MLP(nn.Module):

def __init__(

self,

fc_feat_dim_list: List[int],

name: str,

non_linearity: nn.Module = nn.ReLU,

non_linearity_last: bool = False,

use_layer_norm: bool = False, # 新增

):

super(MLP, self).__init__()

self.fc_layers = nn.Sequential()

for i in range(len(fc_feat_dim_list) - 1):

fc_layer = make_fc_layer(fc_feat_dim_list[i], fc_feat_dim_list[i + 1])

self.fc_layers.add_module(f"{name}_fc{i+1}", fc_layer)

is_last = (i + 1 == len(fc_feat_dim_list) - 1)

if not is_last or non_linearity_last:

if use_layer_norm:

self.fc_layers.add_module(

f"{name}_ln{i+1}",

nn.LayerNorm(fc_feat_dim_list[i + 1])

)

self.fc_layers.add_module(

f"{name}_non_linear{i+1}", non_linearity()

)

```

  

**修改 2：** 仅对**主干 3 处**启用 LayerNorm（保守做法），位置 [`model.py:126-130`](./model.py#L126-L130)：

  

```python

self.concat_mlp = MLP(fc_concat_dim_list, "concat_mlp",

non_linearity_last=True, use_layer_norm=True)

self.concate_mlp_other = MLP(fc_concat_other_mlp_list, "concat_other_mlp",

use_layer_norm=True)

self.lstm_and_linear_mlp = MLP(fc_lstm_and_linear_mlp, 'lstm_and_linear_mlp',

non_linearity_last=True, use_layer_norm=True)

```

  

**安全性：**

- ✅ 接口完全不变

- ⚠️ state_dict 多了 LayerNorm 的 weight/bias，**不能加载老 checkpoint**

- ✅ LayerNorm 在小 batch 下也稳定，不引入平台兼容问题

  

**注意：** 项目笔记说"删除了 layernorm 不确定的 tricks"，但只在主干 3 处加 LN 是非常保守的做法，与"全网加 BN"完全不同。**LayerNorm 对 RL 稳定性的帮助是社区共识**（OpenAI Five、AlphaStar 都用）。

  

**预期收益：** ⭐⭐⭐⭐ 训练稳定性显著提升

  

---

  

#### #7. 池化改进：max + mean 拼接

  

**位置：** [`model.py:275-276, 286-287, 316-317`](./model.py#L275)

  

**问题：** 4 兵 / 9 弹直接 max pool 丢失个体差异和数量信息。

  

**修改：**

  

```python

# 原: soldier_frd_concat_result, _ = reshape_frd_soldier.max(dim=1)

soldier_frd_max = reshape_frd_soldier.max(dim=1).values # (B*L, 32)

soldier_frd_mean = reshape_frd_soldier.mean(dim=1) # (B*L, 32)

soldier_frd_concat_result = torch.cat([soldier_frd_max, soldier_frd_mean], -1) # (B*L, 64)

```

  

**配套修改：** [`model.py:117-124`](./model.py#L117-L124) 的 `concat_dim` 计算需要相应增加（每个 max+mean 位置 +32 维）。

  

**安全性：**

- ✅ 接口不变

- ⚠️ 参数量略增，需重训

- 📌 **建议：作为第 2 阶段实验，不是关键修复**

  

**预期收益：** ⭐⭐ 不确定（保留分布信息可能有用）

  

---

  

## 推荐执行顺序

  

| 步骤 | 修改 | 工时 | 收益 | 加载老 ckpt |

|------|------|------|------|------------|

| 1 | #2 修 PPO clip | 0.5h | ⭐⭐⭐⭐ | ✅ 可热替换 |

| 2 | #3 修 mask 常量 | 5min | ⭐⭐⭐ | ✅ 可热替换 |

| 3 | #4 向量化 MLP（数学等价） | 2-3h | ⭐⭐⭐⭐ | ✅ 可热替换（先验证等价） |

| 4 | #1 修 Target Attention | 0.5h | ⭐⭐⭐⭐⭐ | ❌ 建议重训 |

| 5 | #5 entropy decay | 0.5h | ⭐⭐⭐ | ✅ 可热替换 |

| 6 | #6 主干加 LayerNorm | 1h | ⭐⭐⭐⭐ | ❌ 必须重训 |

| 7 | #7 池化改进 | 1h | ⭐⭐ | ❌ 必须重训 |

  

---

  

## 最小改动包（强烈推荐）

  

如果只想做 **绝对安全、立即见效** 的修改，做这 4 项：

  

1. ✅ #1 修复 Target Attention

2. ✅ #2 修复 PPO Clip

3. ✅ #3 修复 mask 数值常量

4. ✅ #4 向量化同类 MLP 调用

  

总改动 < 50 行，**只动 [`model.py`](./model.py) 一个文件**，不会触碰任何平台接口。

  

---

  

## 验证清单

  

每次修改后，按顺序验证：

  

1. **结构验证：** 运行 `python model.py` 末尾的 `test_model()`，确保 forward pass 不报错、输出 shape 正确

2. **数学等价验证（仅 #4 需要）：** 用旧 ckpt 加载，对同一输入跑新旧代码，断言：

```python

assert torch.allclose(old_output, new_output, atol=1e-5)

```

3. **训练验证：** 短期训练 1-2h，确认 loss 不发散、与基线趋势一致

4. **平台验证：** 提交平台跑一个 episode，确认无 sample 格式错误

  

---

  

## 参考

  

- 绝悟原论文：[Mastering Complex Control in MOBA Games with Deep RL (AAAI 2020)](https://aaai.org/ojs/index.php/AAAI/article/view/6144)

- Dual-clip PPO 出处：同上论文 Section 4.2

- LayerNorm 在 RL 中的应用：OpenAI Five、AlphaStar、DreamerV3 等
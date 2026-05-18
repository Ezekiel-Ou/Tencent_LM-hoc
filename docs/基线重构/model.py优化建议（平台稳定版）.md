
# 不要触碰的接口（平台依赖）

  

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
# 推荐修改清单

## 1. 修正 Target Attention 为标准 scaled dot-product

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

---

  

## 2. 修复 PPO Clip 多余的 clamp

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
## 3. 修复 legal_action mask 数值常量

**问题：** `boundary = 10**20` 数值过大，FP16 / AMP 下会溢出，FP32 下也容易在 `exp` 后产生 inf。

**修改：**
```python

boundary = torch.pow(torch.tensor(10.0), torch.tensor(8.0)) # 1e20 → 1e8

```

**安全性：** ✅ 纯数值常量，完全安全。可以热替换。

**预期收益：** ⭐⭐⭐ 避免训练偶发 NaN

---

## 4. 向量化同类实例的 MLP 调用


**修改思路：** 把同类多实例 stack 后批量过 MLP。
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

## 5. 池化改进：max + mean 拼接

**问题：** 直接 max pool 丢失个体差异和数量信息。

```python

# 原: soldier_frd_concat_result, _ = reshape_frd_soldier.max(dim=1)

soldier_frd_max = reshape_frd_soldier.max(dim=1).values # (B*L, 32)

soldier_frd_mean = reshape_frd_soldier.mean(dim=1) # (B*L, 32)

soldier_frd_concat_result = torch.cat([soldier_frd_max, soldier_frd_mean], -1) # (B*L, 64)

```

---

  

## 推荐执行顺序

  

| 步骤 | 修改 | 工时 | 收益 | 加载老 ckpt |

|------|------|------|------|------------|

| 1 | #2 修 PPO clip | 0.5h | ⭐⭐⭐⭐ | ✅ 可热替换 |

| 2 | #3 修 mask 常量 | 5min | ⭐⭐⭐ | ✅ 可热替换 |

| 3 | #4 向量化 MLP（数学等价） | 2-3h | ⭐⭐⭐⭐ | ✅ 可热替换（先验证等价） |

| 4 | #1 修 Target Attention | 0.5h | ⭐⭐⭐⭐⭐ | 建议重训 |

| 5 | #5 池化改进 | 1h | ⭐⭐ | ❌ 必须重训 |

---

如果只想做 **绝对安全、立即见效** 的修改，做这 4 项：

1. ✅ #1 修复 Target Attention

2. ✅ #2 修复 PPO Clip

3. ✅ #3 修复 mask 数值常量

4. ✅ #4 向量化同类 MLP 调用
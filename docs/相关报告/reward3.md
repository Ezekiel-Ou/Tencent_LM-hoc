# (9 封私信 / 37 条消息) 腾讯多智能体强化学习大赛奖励函数专题（三） - 知乎

# 腾讯多智能体强化学习大赛奖励函数专题（三）

[![克里斯朵弗](image)](https://www.zhihu.com/people/michael-scofild)

[克里斯朵弗](https://www.zhihu.com/people/michael-scofild)

[​](https://www.zhihu.com/question/48510028)

电子科技大学 计算机技术硕士

​关注他

[收录于 · 腾讯多智能体强化学习大赛](https://www.zhihu.com/column/c_1683282804258045952)

23 人赞同了该文章

[克里斯朵弗：腾讯多智能体强化学习大赛奖励函数专题（二）](https://zhuanlan.zhihu.com/p/657946316)*比较细致的讲了一下实现时序奖励的思路，专题二以帮助队友为例说明了如何在连续帧情况下去判别帮助队友的情景；以诸葛亮大招学习为例介绍如何结合英雄帧信息去做一些更复杂的情景判定*。

今天我们接着深入去探讨一下奖励工程设计的问题，这一次的主题是**[稀疏化](https://zhida.zhihu.com/search?content_id=234689561&content_type=Article&match_order=1&q=%E7%A8%80%E7%96%8F%E5%8C%96&zhida_source=entity)与团队化**。在[克里斯朵弗：腾讯多智能体强化学习大赛奖励函数专题（一）](https://zhuanlan.zhihu.com/p/655435009)中，我们提到[情景累加](https://zhida.zhihu.com/search?content_id=234689561&content_type=Article&match_order=1&q=%E6%83%85%E6%99%AF%E7%B4%AF%E5%8A%A0&zhida_source=entity)的概念（情景累加指的是当情景奖励生效后如何作用到现有的奖励函数体系之下），接着这个概念往下看，我们将提出情景累加的第三种方式！

```
# 方式一
if 情景a成立:
    1. 添加情景a奖励项作为自定义奖励项到hero_reawrd_detail_dict
    2. 参与后续的hero_reward_sum_decay计算
# 方式二
if 情景a成立:
    跳过奖励项累加零和等计算，直接将情景加到hero_reward_sum计算结果上
```

## 奖励函数设计的稀疏化

随着我们自定义添加的情景奖励越来越多，不管是方式一还是方式二都会额外累加越来越多的奖励，如果自定义情景奖励正值偏多会导致累积奖励reward sum估计的越来越大，我们实际上在训练样本日志中也发现了这种情况，估计的reward sum能发散至10倍以上，即使自定义情景奖励正负值设置的比较合理、平均，reward sum的方差也会变大，这些都会导致[价值网络](https://zhida.zhihu.com/search?content_id=234689561&content_type=Article&match_order=1&q=%E4%BB%B7%E5%80%BC%E7%BD%91%E7%BB%9C&zhida_source=entity)value network难以学习，估计不准。

另一方面，训练强化学习训练智能体，设计奖励函数的一个普适思路就是：**在前期用密集奖励加大智能体探索，后期用稀疏奖励帮助智能体学习高难度动作**。在王者荣耀奖励因子调参中，大家基本上也是前期提高money、exp这些密集奖励去加大英雄对获取资源动作的探索，后期减少这些奖励，提高kill assist来让英雄学会诱导击杀敌人。这里降低money、exp的好处考虑主要有两点：一是可以加大英雄的探索，让英雄不那么贪念眼前的兵线资源而放给残血逃跑的敌人，更容易学会场距离追杀敌人，二是保证累积奖励总体保持变化不大，提高部分奖励因子权重的同时衰减其他部分的奖励因子权重，有利于价值网络训练的稳定性。

基于上面两段的考虑，在[AI模型训练](https://zhida.zhihu.com/search?content_id=234689561&content_type=Article&match_order=1&q=AI%E6%A8%A1%E5%9E%8B%E8%AE%AD%E7%BB%83&zhida_source=entity)后期时，情景奖励判定生效时可以有第三种用法，**对默认的奖励因子（尤其是密集奖励）去做稀疏化和衰减化。**

```
# 方式三
if 情景a成立:
    1. 对已有的hero_reawrd_detail_dict奖励因子做权重衰减
    2. 参与后续的hero_reward_sum_decay计算
```

- 残血及时回家

英雄在血量很低时，如果money、exp密集奖励权重大，训练的英雄一直被一波波兵线吸引难以学会回家。如果把money、exp权重调小，hp相关权重调大，训练的英雄受了一些伤就喜欢直接回家。要想让英雄学会残血回家，一种方式就是对exp、money、hp进行精细的调节来达成这种效果，往往比较难。这里介绍一种判断情景奖励然后衰减的方式：

```
# 如果英雄的血量比低于0.2， 即状态很差衰减这种情景的money、exp
def goHomeInLowHp(hero_index, hero_reawrd_detail_dict):
    # 假定先将Hp，maxHp注入到了hero_reawrd_detail_dict中
    ratio = float(hero_reawrd_detail_dict[hero_index]["hp"]/hero_reawrd_detail_dict[hero_index]
    if (ratio <= 0.2):
        # 下面不完全正确，实际上需要将所有英雄对应的detail_reward都同步衰减而不单是衰减一个
        hero_reawrd_detail_dict[hero_index]["money"] = 0.1 * hero_reawrd_detail_dict[hero_index]["money"]
        hero_reawrd_detail_dict[hero_index]["exp"] = 0.1 * hero_reawrd_detail_dict[hero_index]["exp"]
```

## 奖励函数设计的团队化：

经验表明，最简单加强团队合作的方式是调高assist，而调高team\_spirit几乎没什么用。利用时序情景奖励也可以构造很多团队合作的情景。这里介绍两个我们的经过时间检验的团队化奖励的设计思路供参考：

- 集火行为

AI英雄行为的镜像性，各打各的

鼓励英雄在短时间内同时攻击同一个敌方英雄，而不是各自为战，达到1+1>2的效果。虽然我们玩家知道打架时要集火秒掉一个人来奠定巨大的优势，对于AI来说却没那么好学。由于王者荣耀AI是通过自我博弈的方式训练的，训练时两边的模型参数一模一样或者相近，这会导致AI英雄行为具有一定的**镜像性**，很容易出现敌方我方英雄各打各镜像英雄的情况。

这里我们想了这样一个办法来实现集火行为的学习：如果队友英雄攻击了一个敌方英雄会有一个标记，当前英雄后15帧内有攻击该英雄则判定生效。

```
def cooperative_hurt_to_hero(self, frd_rl_data_map):
    reversed_keys = list(frd_rl_data_map.keys())
    for frame_number in reversed_keys:
        // 如果队友造成了伤害，将对应受伤的敌方英雄hero_index记录下来
        emy_hero_index = frd_hurt_in_this_frame(hero_index, frame_number, frd_rl_data_map)
        if emy_hero_index != -1:
            flag = True
            # 用后15帧判断，这个值可自行设置
            for temp_frame_number in range(frame_number, frame_number + 15, 3):
                if hurt_kill_assist_special_in_this_frame(hero_index, emy_hero_index, temp_frame_number, frd_rl_data_map):
                    flag = False
                    break
            # 做标记到all_hero_reward中
            if flag:
                self.mark_time_reward(frd_rl_data_map, frame_number, hero_index, "cooperative_hurt", 1)
```

这里frd\_hurt\_in\_this\_frame用于判断这一帧是否队友造成了英雄伤害，逻辑和专题一类似省去了，hurt\_kill\_assist\_special\_in\_this\_frame用于判断当前帧选中英雄是否对指定敌方英雄造成伤害、击杀或者助攻

```
def hurt_kill_assist_special_in_this_frame(hero_index, emy_hero_index, temp_frame_number, frd_rl_data_map):
    flag = False
    hero_info = frd_rl_data_map[temp_frame_number][hero_index]
    if hero_info.all_hero_reward[hero_index]['killCnt'] > 0.0001 or \
        hero_info.all_hero_reward[hero_index]['assistCnt'] > 0.0001 or \
        hero_info.all_hero_reward[hero_index]['total_hurt_to_hero'] > 0.0001:
           if hero_info.all_hero_reward[emy_hero_index]['hp_rate_sqrt_sqrt'] < 0:
               flag = True
           return flag
```

判定生效后可以选择给予额外奖励或者增大对应hurt\_to\_hero的权重来鼓励英雄攻击标记英雄的行为（类似于上面讲的稀疏化，稀疏化是衰减奖励因子权重这里是增大奖励因子权重）。

- 稀疏奖励团队化

这一点在[lunlun：腾讯多智能体强化学习大赛冠军思路分享](https://zhuanlan.zhihu.com/p/654972230)提到过，原来的总奖励等于个人零和奖励配上系数加上团队零和奖励，实验发现通过直接调节系数team\_spirit难以加强英雄间的团队合作，原因可能是团队奖励包含的密集奖励出现的频率过高干扰了本来稀疏的合作场景的学习，因此将击杀、助攻、死亡的稀疏奖励单独抽出来作为团队奖励计算，**尽可能的减少了频繁的密集奖励对团队奖励的干扰**。

## 小结

这次在专题一和二的基础上进一步讲了引入情景奖励需要考虑的细节，以及如何朝着稀疏化和合作性方向去训练高水平王者荣耀AI。这次也算是奖励函数专项的完结篇，本着**授人以鱼不如授人以渔**的道理，我们并没有将所有的情景奖励设计方法和细节全部写到文中，本专题从一到三大概就是我们在奖励工程方向迭代的思路，是逐渐递进的，其中的思考也一并写在了文中，示例代码是基于开源代码的伪代码，用于描述大致思路，具体的细节还需要**结合实际**比赛代码去考虑，希望能对开悟比赛的小伙伴有所启发。

发布于 2023-10-04 23:52・上海

[强化学习 (Reinforcement Learning)](https://www.zhihu.com/topic/20039099)

[游戏AI](https://www.zhihu.com/topic/20190693)

[王者荣耀（游戏）](https://www.zhihu.com/topic/20034016)

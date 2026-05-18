# (9 封私信 / 37 条消息) 腾讯多智能体强化学习大赛奖励函数专题（二） - 知乎

# 腾讯多智能体强化学习大赛奖励函数专题（二）

[![克里斯朵弗](image)](https://www.zhihu.com/people/michael-scofild)

[克里斯朵弗](https://www.zhihu.com/people/michael-scofild)

[​](https://www.zhihu.com/question/48510028)

电子科技大学 计算机技术硕士

​关注他

[收录于 · 腾讯多智能体强化学习大赛](https://www.zhihu.com/column/c_1683282804258045952)

11 人赞同了该文章

[克里斯朵弗：腾讯多智能体强化学习大赛奖励函数专题（一）](https://zhuanlan.zhihu.com/p/655435009)讲了比较基础的王者AI奖励设计理念和方法，今天的内容将更加深入和灵活，尤其是将时序情景奖励讲的明明白白！

*我们先来回顾一下专题一的内容：瞬时情景奖励基于当前帧现有的奖励项的值的组合作为情景判断条件，比如判断是否暴君被我方英雄击杀通过这一帧3个英雄是否同时获得大量的金币和经验来判断。*

但情景往往是一个事件，这个事件从发生到结束很难通过一帧的数据来判断定义；事件是一个过程，如果能够**引入时间维度**去描述这个事件，是不是就方便很多了呢？

在介绍我们方法之前，让我们先看一下咱们王者荣耀AI训练样本的奖励是咋来的，下面源码部分引用自腾讯AI lab开源源码[https://github.dev/tencent-ailab/hok\_env](https://link.zhihu.com/?target=https%3A//github.dev/tencent-ailab/hok_env)。

## sample\_manager奖励计算逻辑

sample\_manager主要的作用是准备PPO算法训练的样本，然后发送到样本池中给RL端训练：

```
    def send_samples(self):
        self._calc_reward()
        self._format_data()
        self._send_game_data()
```

我们需要重点关注\_calc\_reward这个函数，这个函数会通过GAE算法（PPO算法中用来估计优势advantage值的算法）来计算累积奖励以及advantage：

```
        for i in range(self.num_agents):
            reversed_keys = list(self.rl_data_map[i].keys())
            reversed_keys.reverse()
            gae = [0.0, 0.0, 0.0]
            self.reward_manager.reset()
            LOG.info("i:%d reversed_keys_size:%d" % (i, len(reversed_keys)))
            index = 0
            for j in reversed_keys:
                index += 1
                rl_info = self.rl_data_map[i][j]
                for hero_idx in range(self.hero_num):
                    advantage, reward_sum = self.reward_manager.calc_advantage(
                        i,
                        hero_idx,
                        rl_info[hero_idx].reward,
                        rl_info[hero_idx].value,
                        rl_info[hero_idx].next_value,
                        rl_info[hero_idx].all_hero_reward,
                        rl_info[hero_idx],
                    )
                    rl_info[hero_idx].advantage = advantage
                    rl_info[hero_idx].reward_sum = reward_sum
```

其中self.num\_agents表示红蓝两个阵营，self.rl\_data\_map[i][j]表示阵营i的第j帧样本，样本信息包括官方内置计算的reward（我们用自定义的奖励不需要这个），value为这一帧的状态价值，next\_value为下一帧的状态价值，all\_hero\_reward作为字典存放奖励因子小项。上面的大概逻辑就是对红蓝每一个阵营，将rl信息以**倒序的方式排列遍历**，通过GAE算法计算advantage和累积奖励并存放到rl\_info中。倒序是为了方便算累积奖励，前面帧的奖励得后面帧的奖励算完了才好算。

## 时序情景奖励

在rl\_data\_map中如果**将前后帧拿到的all\_hero\_reward奖励小项用于情景判断**，这相当于有了时间的维度，这不就是引入时间维度去描述事件嘛！换句话说也就是时序情景奖励。

### 例子一：帮助队友时序版本

```
# 考虑时间维度的帮助队友：如果某一帧队友死亡，判断前15帧是否有击杀、助攻或者造成伤害，没有则给予惩罚
# frd_rl_data_map表示一方阵营的rl_data_map，即self.rl_data_map[阵营indx]
def friend_help_flag(hero_index, frd_rl_data_map):
    reversed_keys = list(frd_rl_data_map.keys())
    for frame_number in reversed_keys:
        if frd_die_in_this_frame(hero_index, frame_number, frd_rl_data_map):
            flag = True
            # 用前15帧判断，这个值可自行设置
            for temp_frame_number in range(frame_number - 15, frame_number + 1, 3):
                if hurt_kill_assist_in_this_frame(hero_index, temp_frame_number, frd_rl_data_map):
                    flag = False
                    break
            # 做标记到all_hero_reward中
            if flag:
                self.mark_time_reward(frd_rl_data_map, frame_number, hero_index, "frd_die_publish", -1)
```

这里frd\_die\_in\_this\_frame函数用于判断队友是否在这一帧死亡，逻辑和专题一类似省去了，hurt\_kill\_assist\_in\_this\_frame用于判断当前帧选中英雄是否有造成伤害、击杀或者助攻，直接读取相应的奖励小项值就行：

```
def hurt_kill_assist_in_this_frame(hero_index, temp_frame_number, frd_rl_data_map):
    flag = False
    hero_info = frd_rl_data_map[temp_frame_number][hero_index]
    if hero_info.all_hero_reward[hero_index]['killCnt'] > 0.0001 or \
        hero_info.all_hero_reward[hero_index]['assistCnt'] > 0.0001 or \
        hero_info.all_hero_reward[hero_index]['total_hurt_to_hero'] > 0.0001:
           flag = True
           return flag
```

值得注意的是这里我们时序情景判断完成后并没有直接返回flag而是标记到了all\_hero\_reward中，这是因为像专题一直接返回的方式在自定义计算reward时可以直接读取当前帧reward奖励小项获得。

如果这里直接返回flag的话我们自定义计算reward的函数（参考专题一）是不能读取前面帧reward小项的，除非将计算reward的函数整个改成包含前后帧奖励项的函数，这样也需要考虑到数据冗余和计算效率的问题。

这里mark\_time\_reward函数直接将flag以字典key-value的形式保存到当前帧的奖励小项中，在自定义计算reward函数中去读取该key-value计算。这是一种取巧和改动小的办法，小伙伴们也可以思考有没有效率更高的办法。

```
def mark_time_reward(frd_rl_data_map, frame_number, hero_index, key, value):
    # 将情景判断结果存到reward_detail奖励小项中用于自定义奖励计算，过程参考专题一
    rl_info = frd_rl_data_map[frame_number]
    # 保证友方英雄的奖励情景奖励小项内容一致
    for index in range(3):
        rl_info[index].all_hero_reward[hero_idx][key] = value
```

英雄间合作要求一段时间帧同时能对敌方英雄造成伤害，惩罚一个英雄单挑另一个英雄补兵类似的情况

### 例子二：诸葛亮大招释放

对于职业玩家来说玩诸葛亮用大招收割残血并连续击杀多名敌人入门级的操作，但对AI来说却很难学（大家自己训练王者荣耀英雄试一试懂得都懂），这里介绍一种结合帧英雄数据与时序奖励小项的情景判断设计方法来辅助诸葛亮大招的学习：

```
# 如果诸葛亮的大招的target是敌方英雄，并且在2s内敌方死了，给个奖励标记
# 如果诸葛亮的大招的目标是非英雄单位，给个惩罚标记
def zgl_skill3_flag(hero_index, frd_rl_data_map):
    reversed_keys = list(frd_rl_data_map.keys())
    for frame_number in reversed_keys:
       ....
```

这里难点是怎么判断用了大招？以及如果用了大招，释放对象是英雄还是小兵呢？这个时候就要利用上rl\_info的英雄动作信息，在动作信息中，我们可以读取获得英雄的action, legal\_action以及目标target，结合这些信息做情景判断，比如判断英雄释放的动作是否为大招：

```
heroInfo = frd_rl_data_map[frame_number][hero_index]
button = heroInfo.action[0]
// button值为6表示释放了大招
if (button == 6):
    pass
```

要判断诸葛亮的大招的target是敌方英雄，除button键要求是大招、读取target目标信息外，还要判断相应的动作是否合法，不合法游戏引擎不执行对应的操作，简单的判断逻辑为：

优化诸葛亮大招简易流程

注：如何获取细节的这些信息可以看比赛和开源官方文档，同时记得及时的**以日志的方式打印**出来看去对比。

## 小结

这次比较细致的讲了一下实现时序奖励的思路，重点说明了如何在连续帧情况下去判别帮助队友的情景，以及简要说明了结合如何英雄帧信息去做一些更复杂的情景判定。可以看一下我们在[世界大学生数智竞技邀请赛-AI赛道预选赛第四场：清华大学VS电子科技大学\_哔哩哔哩\_bilibili](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1MR4y1D7ZY/%3Fspm_id_from%3D333.1007.top_right_bar_window_history.content.click%26vd_source%3D5fbfa028789415de7fc9c5dbaf9f9ccf)诸葛亮逆风2打3连杀翻盘就部分得益于这部分设计（另一部分在总结中的辅助loss做法）。有了时序这个维度就可以发挥想象设计各种情景奖励，并且可以通过录像回放来实时查看效果，不断打磨自己设计的情景奖励程序！

但微操的好看与提升并不一定带来胜率的提升，**过分的追求微操可能会损失其他的能力**。我们决赛的模型打的不并精彩，诸葛亮大招仍满血常常发生，这是因为后期去掉了诸葛亮大招的情景判断，不去追求微操，专注于更有AI行为的**效率性和合作性**。在下一期我们会介绍这些以及奖励函数调参的一些碎碎念~敬请期待！

编辑于 2023-09-23 17:31・上海

[强化学习 (Reinforcement Learning)](https://www.zhihu.com/topic/20039099)

[王者荣耀（游戏）](https://www.zhihu.com/topic/20034016)

[游戏AI](https://www.zhihu.com/topic/20190693)

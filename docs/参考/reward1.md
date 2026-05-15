# (9 封私信 / 37 条消息) 腾讯多智能体强化学习大赛奖励函数专题（一） - 知乎

# 腾讯多智能体强化学习大赛奖励函数专题（一）

[![克里斯朵弗](image)](https://www.zhihu.com/people/michael-scofild)

[克里斯朵弗](https://www.zhihu.com/people/michael-scofild)

[​](https://www.zhihu.com/question/48510028)

电子科技大学 计算机技术硕士

​关注他

[收录于 · 腾讯多智能体强化学习大赛](https://www.zhihu.com/column/c_1683282804258045952)

33 人赞同了该文章

*对于训练[王者荣耀AI](https://zhida.zhihu.com/search?content_id=233780852&content_type=Article&match_order=1&q=%E7%8E%8B%E8%80%85%E8%8D%A3%E8%80%80AI&zhida_source=entity)的小伙伴来说，比赛前期将强化学习算法以及网络结构固定了，是不是经常感觉后面只能调调[PPO](https://zhida.zhihu.com/search?content_id=233780852&content_type=Article&match_order=1&q=PPO&zhida_source=entity)的系数，训练的学习率，或者奖励配置项系数了？大错特错！在中后期训练过程中，除了以上提到的以外，我们还做了包括训练算法的辅助损失函数、[动态调整训练模型池](https://zhida.zhihu.com/search?content_id=233780852&content_type=Article&match_order=1&q=%E5%8A%A8%E6%80%81%E8%B0%83%E6%95%B4%E8%AE%AD%E7%BB%83%E6%A8%A1%E5%9E%8B%E6%B1%A0&zhida_source=entity)、自定义奖励函数工程等。其中奖励函数工程就像是不断给AI设定新目标，逐渐迭代更新更优的策略。[腾讯多智能体强化学习大赛冠军思路分享](https://zhuanlan.zhihu.com/p/654972230)重点涵盖了我们所有的技术及经验，而今天我们将从基础入门概念到伪代码实现，分享我们在奖励函数工程上茴字的八种写法，保证浅显易懂，哈哈！*

那么现在就开始正式进入今天的专题--王者荣耀AI的奖励函数设计（以3V3为背景）！

### 个人英雄奖励的计算

如下图所示，在开悟平台的奖励函数配置文件中有一系列奖励小项：

| 奖励因子 | 说明 | 权重 |
| --- | --- | --- |
| hp\_rate\_sqrt\_sqrt | 血量比值开四次方 | 1 |
| money | 经济增长值 | 0.001 |
| killCnt | 击杀数 | 1 |
| ... | ... | ... |

首先我们要明确这些奖励因子的含义，知道他们是怎么来的，比如money代表的是该英雄当前帧的金钱总数减去上一帧的金钱总数。官方默认的每个英雄的奖励计算步骤包括加权求和、[零和博弈](https://zhida.zhihu.com/search?content_id=233780852&content_type=Article&match_order=1&q=%E9%9B%B6%E5%92%8C%E5%8D%9A%E5%BC%88&zhida_source=entity)、团队奖励以及奖励项时间衰减，可以通过奖励函数配置文件进行纯粹的调参，比如希望英雄打的更激进就调高hurt\_to\_hero, killCnt，更有效率的打钱就提高money等。根据我们经验而言，尽量少动团队奖励。假如调参发现AI模型能力止步不前怎么办？亦或是录像回放时发现了AI比较严重的问题不好直接调奖励项？这个时候从奖励函数方面我们会进行**自定义**，重点是[情景奖励](https://zhida.zhihu.com/search?content_id=233780852&content_type=Article&match_order=1&q=%E6%83%85%E6%99%AF%E5%A5%96%E5%8A%B1&zhida_source=entity)的设计。

继续训练迭代过程

要想引用我们这一套奖励函数设计方法，建议先自己从奖励项开始实现一遍官方的奖励计算过程，这样后面我们就可以基于函数方便的修改来添加情景奖励了。

具体来说，定义一个函数计算单个英雄的奖励：

```
# hero_reward_detail_dict存放的是当前帧所有英雄的乘了奖励因子系数的值
# 比如hero_reward_detail_dict[0]["money"]代表的是当前帧
def calc_final_reward(hero_index, hero_reawrd_detail_dict):
    # 计算当前帧考虑时间衰减的奖励之和
    decay_reward_list = calc_decay_sum(XXX)
    # 获取自身英雄的时间衰减奖励之和
    self_sum_reward = decay_reward_list[hero_index]
    # 时间衰减的奖励之和后计算零和奖励
    zero_sum_reward = calc_zero_sum_reward(XXX)
    # 零和奖励之后计算团队奖励
    return calc_team_reward(XXX)
```

写完本地跑一下日志验证看是不是跟官方的一样呢（浮点数计算可能有细微差别）。

### 自定义时间衰减

官方对奖励的时间衰减做法简单粗暴，计算所有奖励因子的加权和后对游戏的帧数做指数级衰减：

 hero\_reward =hero\_reward∗scaling\_discount frame-non  scalingtime \text { hero\_reward }=h e r o \\_r e w a r d \* s c a l i n g \\_d i s c o u n t^{\frac{\text { frame-non }}{\text { scalingtime }}}

我们可以给奖励因子子项设置不同的时间衰减系数，若考虑到KDA整局都很重要，可以将击杀死亡助攻整局游戏都不衰减，只衰减其他的因子；若考虑到后期应该积极推塔，那么对于塔相关的奖励因子可以衰减更少甚至不衰减；若考虑到前期英雄互相换血太厉害不注重自身血量健康，可以衰减前期阶段的部分奖励因子值：

```
# 提前将帧号放入到hero_reawrd_detail_dict中
def calc_decay_sum(hero_index, hero_reawrd_detail_dict):
    # hero_reward_detail_dict存放的是当前帧所有英雄的乘了奖励因子系数的值
    # 比如hero_reward_detail_dict[0]["money"]代表的是当前帧获得的金币数乘权重的结果
    reward_sum = 0
    # 考虑到前期英雄互相换血太厉害，额外衰减前期阶段的部分奖励因子值
    if int(hero_reawrd_detail_dict[0]["frame_no"] <= 900):
        # 1分钟900帧，前一分钟
        decay_total_hurt_to_hero_and_kill(XXX)
    for key in hero_reward_detail_dict[0]:
        # 如果考虑KDA一直很重要，那么可以当key是KDA时只累加不衰减
        XXX
```

### 情景奖励

情景奖励的设计又分为情景判定以及奖励累加两个部分，情景奖励判定什么时候情景奖励生效，而情景累加指的是当情景奖励生效后如何作用到现有的奖励函数体系之下。

对于情景累加，我们想到的有两种方式：

```
# 方式一
if 情景a成立:
    1. 添加情景a奖励项作为自定义奖励项到hero_reawrd_detail_dict
    2. 参与后续的hero_reward_sum_decay计算
# 方式二
if 情景a成立:
    跳过奖励项累加零和等计算，直接将情景加到hero_reward_sum计算结果上
```

两种方式我们混着用的，眨眼一看方式二粗暴的破坏了奖励设计的零和性，验证时并没有太大的负面效果，甚至效果比方式一更明显。

**瞬时情景奖励**

基于当前帧现有的奖励项的值的组合作为情景判断条件。下面举了两个例子。

- 击杀[暴君](https://zhida.zhihu.com/search?content_id=233780852&content_type=Article&match_order=1&q=%E6%9A%B4%E5%90%9B&zhida_source=entity)

在3V3地图中，暴君的击杀可以给队伍带来巨大的金币经验以及能力增益，但暴君位处于地图的最下方野区中，对抗路英雄很少到此处，怎么才能让英雄学会高效的抱团击杀暴君呢？

王者荣耀3v3长平攻防战地图

我们观察日志偶然发现了一个规律：一方英雄击杀暴君时，三个友方英雄都会获得200倍大小的相应金币和经验奖励！于是暴君的情景奖励判定就有了：

```
def kill_baojun_flag(hero_reawrd_detail_dict):
    # 默认击杀暴君
    flag = true
    for hero_index in range(3):
        # money_weight、exp_weight表示奖励因子相应的权重
        if hero_reawrd_detail_dict[i]["money"] <= 200 * money_weight \
        and hero_reawrd_detail_dict[i]["exp"] <= 200 * exp_weight:
            flag = false
            break
    return flag
```

就是这么简单！当然，实现暴君的情景判断方法不止一种，比如可以从帧信息中直接读取暴君的血量来判定。但需要注意的是帧信息是经过了mask处理的，如果在视野范围之外的阴影部分，是获取不到当前帧暴君的血量的，mask处理后的值是1。

- 帮助队友

对抗路2V2

在MOBA游戏中英雄之间的合作产生的化学反应会起到一加一大于二的作用，那么如何让AI基于瞬时情景奖励学会加强合作呢？我们构思这样一个场景，如果这一帧英雄的队友死了，该英雄却没有对敌方英雄造成伤害、击杀或者助攻，我们就认为他做的不够好给予惩罚；如果这一帧英雄的队友死了，该英雄却对敌方英雄造成伤害、击杀或者助攻就给予奖励：

```
# 帮助队友奖励判定
def friend_help_flag(hero_index, hero_reawrd_detail_dict):
    flag = false
    for i in range(3):
        # deadCnt用于判断友方英雄当前是否死亡
        if i != hero_index and hero_reawrd_detail_dict[i]["deadCnt"] < 0:
            if hero_reawrd_detail_dict[hero_index]["killCnt"] > 0 \
            or hero_reawrd_detail_dict[hero_index]["assistCnt"] > 0 \
            or hero_reawrd_detail_dict[hero_index]["hurt_to_hero"] > 0:
                flag = true
                return flag
    return flag

# 不帮助队友惩罚判定
def friend_punish_flag(hero_index, hero_reawrd_detail_dict):
    flag = false
    for i in range(3):
        # deadCnt用于判断友方英雄当前是否死亡
        if i != hero_index and hero_reawrd_detail_dict[i]["deadCnt"] < 0:
            if hero_reawrd_detail_dict[hero_index]["killCnt"] <= 0 \
            and hero_reawrd_detail_dict[hero_index]["assistCnt"] <= 0 \
            and hero_reawrd_detail_dict[hero_index]["hurt_to_hero"] <= 0:
                flag = true
                return flag
    return flag
```

### 小结

小伙伴可能会觉得今天讲的方法很naive，不太能经得起推敲，比如在帮助队友判断中会想：为什么非得在一帧队友死亡的时候正好能造成英雄伤害甚至正好击杀敌方英雄？说不定在前两帧造成的伤害，这里判定就漏掉了呢？这种方法虽然不完美但在实际训练中还是看得出效果的。

所以在下一节我们会将奖励工程专题完结并介绍另一大杀器：时序情景奖励，从此就可以大开脑洞做一些奇奇怪怪的情景判定了，尤其是我们在团队合作上的迭代思路。

分享不易，喜欢的大佬们别忘了关注点赞！

发布于 2023-09-12 01:04・上海

[强化学习 (Reinforcement Learning)](https://www.zhihu.com/topic/20039099)

[多智能体强化学习](https://www.zhihu.com/topic/21327203)

[王者荣耀（游戏）](https://www.zhihu.com/topic/20034016)

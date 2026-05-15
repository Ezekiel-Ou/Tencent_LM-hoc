#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Monitor panel configuration for agent_diy.
"""

from kaiwudrl.common.monitor.monitor_config_builder import MonitorConfigBuilder


def build_monitor():
    monitor = MonitorConfigBuilder()

    config_dict = (
        monitor.title("智能决策1v1-diy")
        .add_group(group_name="算法指标", group_name_en="algorithm")
        .add_panel(name="累积回报", name_en="reward", type="line")
        .add_metric(metrics_name="reward", expr="round(avg(reward{}), 0.01)")
        .end_panel()
        .add_panel(name="总损失", name_en="total_loss", type="line")
        .add_metric(metrics_name="total_loss", expr="round(avg(total_loss{}), 0.01)")
        .end_panel()
        .add_panel(name="价值损失", name_en="value_loss", type="line")
        .add_metric(metrics_name="value_loss", expr="round(avg(value_loss{}), 0.01)")
        .end_panel()
        .add_panel(name="策略损失", name_en="policy_loss", type="line")
        .add_metric(metrics_name="policy_loss", expr="round(avg(policy_loss{}), 0.01)")
        .end_panel()
        .add_panel(name="熵损失", name_en="entropy_loss", type="line")
        .add_metric(metrics_name="entropy_loss", expr="round(avg(entropy_loss{}), 0.01)")
        .end_panel()
        .add_panel(name="PPO平均ratio", name_en="mean_ratio", type="line")
        .add_metric(metrics_name="mean_ratio", expr="round(avg(mean_ratio{}), 0.0001)")
        .end_panel()
        .add_panel(name="PPO近似KL", name_en="approx_kl", type="line")
        .add_metric(metrics_name="approx_kl", expr="round(avg(approx_kl{}), 0.000001)")
        .end_panel()
        .add_panel(name="PPO裁剪比例", name_en="clip_fraction", type="line")
        .add_metric(metrics_name="clip_fraction", expr="round(avg(clip_fraction{}), 0.0001)")
        .end_panel()
        .add_panel(name="血量奖励", name_en="reward_hp_point", type="line")
        .add_metric(metrics_name="reward_hp_point", expr="round(avg(reward_hp_point{}), 0.0001)")
        .end_panel()
        .add_panel(name="塔血奖励", name_en="reward_tower_hp_point", type="line")
        .add_metric(metrics_name="reward_tower_hp_point", expr="round(avg(reward_tower_hp_point{}), 0.0001)")
        .end_panel()
        .add_panel(name="经济奖励", name_en="reward_money", type="line")
        .add_metric(metrics_name="reward_money", expr="round(avg(reward_money{}), 0.0001)")
        .end_panel()
        .add_panel(name="经验奖励", name_en="reward_exp", type="line")
        .add_metric(metrics_name="reward_exp", expr="round(avg(reward_exp{}), 0.0001)")
        .end_panel()
        .add_panel(name="能量奖励", name_en="reward_ep_rate", type="line")
        .add_metric(metrics_name="reward_ep_rate", expr="round(avg(reward_ep_rate{}), 0.0001)")
        .end_panel()
        .add_panel(name="击杀奖励", name_en="reward_kill", type="line")
        .add_metric(metrics_name="reward_kill", expr="round(avg(reward_kill{}), 0.0001)")
        .end_panel()
        .add_panel(name="死亡奖励", name_en="reward_death", type="line")
        .add_metric(metrics_name="reward_death", expr="round(avg(reward_death{}), 0.0001)")
        .end_panel()
        .add_panel(name="补刀奖励", name_en="reward_last_hit", type="line")
        .add_metric(metrics_name="reward_last_hit", expr="round(avg(reward_last_hit{}), 0.0001)")
        .end_panel()
        .add_panel(name="前压奖励", name_en="reward_forward", type="line")
        .add_metric(metrics_name="reward_forward", expr="round(avg(reward_forward{}), 0.0001)")
        .end_panel()
        .end_group()
        .build()
    )
    return config_dict

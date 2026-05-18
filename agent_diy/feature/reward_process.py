#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Reward manager adapted from hok_semi and remapped to the current hok1v1 dict protocol.
"""

import math

from agent_diy.conf.conf import GameConfig


TOWER_SUB_TYPES = {21, "21", "ACTOR_SUB_TOWER"}
SOLDIER_SUB_TYPES = {1, "1", "ACTOR_SUB_SOLDIER"}


class RewardStruct:
    def __init__(self, m_weight=0.0):
        self.cur_frame_value = 0.0
        self.last_frame_value = 0.0
        self.value = 0.0
        self.weight = m_weight


def init_calc_frame_map():
    return {key: RewardStruct(weight) for key, weight in GameConfig.REWARD_WEIGHT_DICT.items()}


def _get(obj, key, default=0):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _get_any(obj, keys, default=0):
    for key in keys:
        value = _get(obj, key, None)
        if value is not None:
            return value
    return default


def _safe_div(num, den, default=0.0):
    try:
        den = float(den)
        if den == 0:
            return default
        return float(num) / den
    except (TypeError, ValueError):
        return default


def _loc(obj):
    location = _get(obj or {}, "location", {}) or {}
    return float(_get(location, "x", 0)), float(_get(location, "z", 0))


def _camp_key(camp):
    if camp in (1, "1", "PLAYERCAMP_1", "blue_camp"):
        return 1
    if camp in (2, "2", "PLAYERCAMP_2", "red_camp"):
        return 2
    return camp


class GameRewardManager:
    def __init__(self, main_hero_runtime_id):
        self.main_hero_player_id = main_hero_runtime_id
        self.main_hero_camp = -1
        self.m_reward_value = {}
        self.m_cur_calc_frame_map = init_calc_frame_map()
        self.m_main_calc_frame_map = init_calc_frame_map()
        self.m_enemy_calc_frame_map = init_calc_frame_map()
        self.time_scale_arg = GameConfig.TIME_SCALE_ARG
        self.m_each_level_max_exp = {}
        self.has_last_frame = False
        self.m_last_hit_debug = self._empty_last_hit_debug()
        self.init_max_exp_of_each_hero()

    def init_max_exp_of_each_hero(self):
        self.m_each_level_max_exp = dict(GameConfig.LEVEL_MAX_EXP)

    def result(self, frame_data):
        self.frame_data_process(frame_data)
        self.get_reward(frame_data, self.m_reward_value)
        return self.m_reward_value

    def set_cur_calc_frame_vec(self, calc_frame_map, frame_data, camp):
        main_hero, enemy_hero = self._find_heroes_by_camp(frame_data, camp)
        main_tower, enemy_tower = self._find_towers_by_camp(frame_data, camp)

        for reward_name, reward_struct in calc_frame_map.items():
            reward_struct.last_frame_value = reward_struct.cur_frame_value
            if main_hero is None:
                reward_struct.cur_frame_value = 0.0
                continue

            if reward_name == "money":
                reward_struct.cur_frame_value = _get(main_hero, "money_cnt", _get(main_hero, "money", 0))
            elif reward_name == "hp_point":
                hp_rate = _safe_div(_get(main_hero, "hp", 0), _get(main_hero, "max_hp", 0))
                reward_struct.cur_frame_value = math.sqrt(math.sqrt(max(hp_rate, 0.0)))
            elif reward_name == "ep_rate":
                hp = _get(main_hero, "hp", 0)
                reward_struct.cur_frame_value = 0.0 if hp <= 0 else _safe_div(_get(main_hero, "ep", 0), _get(main_hero, "max_ep", 0))
            elif reward_name == "kill":
                reward_struct.cur_frame_value = _get(main_hero, "kill_cnt", 0)
            elif reward_name == "death":
                reward_struct.cur_frame_value = _get(main_hero, "dead_cnt", 0)
            elif reward_name == "tower_hp_point":
                reward_struct.cur_frame_value = _safe_div(_get(main_tower or {}, "hp", 0), _get(main_tower or {}, "max_hp", 0))
            elif reward_name == "last_hit":
                last_hit_value, last_hit_debug = self._count_last_hit(frame_data, main_hero, enemy_hero)
                reward_struct.cur_frame_value = last_hit_value
                if calc_frame_map is self.m_main_calc_frame_map:
                    self.m_last_hit_debug = last_hit_debug
            elif reward_name == "exp":
                reward_struct.cur_frame_value = self.calculate_exp_sum(main_hero)
            elif reward_name == "forward":
                reward_struct.cur_frame_value = self.calculate_forward(main_hero, main_tower, enemy_tower)

    def calculate_exp_sum(self, hero):
        exp_sum = 0.0
        level = int(_get(hero, "level", 1) or 1)
        for level_idx in range(1, level):
            exp_sum += self.m_each_level_max_exp.get(level_idx, 0)
        exp_sum += _get(hero, "exp", 0)
        return exp_sum

    def calculate_forward(self, main_hero, main_tower, enemy_tower):
        if main_hero is None or main_tower is None or enemy_tower is None:
            return 0.0
        hp_rate = _safe_div(_get(main_hero, "hp", 0), _get(main_hero, "max_hp", 0))
        if hp_rate <= 0.0:
            return 0.0
        main_tower_pos = _loc(main_tower)
        enemy_tower_pos = _loc(enemy_tower)
        hero_pos = _loc(main_hero)
        dist_hero2enemy = math.dist(hero_pos, enemy_tower_pos)
        dist_main2enemy = max(math.dist(main_tower_pos, enemy_tower_pos), 1.0)
        progress = 1.0 - dist_hero2enemy / dist_main2enemy
        return max(min(progress, 1.0), -1.0)

    def frame_data_process(self, frame_data):
        main_camp, enemy_camp = -1, -1

        for hero in frame_data.get("hero_states", []):
            if self._is_main_hero(hero):
                main_camp = _get(hero, "camp")
                self.main_hero_camp = main_camp
            else:
                enemy_camp = _get(hero, "camp")
        self.set_cur_calc_frame_vec(self.m_main_calc_frame_map, frame_data, main_camp)
        self.set_cur_calc_frame_vec(self.m_enemy_calc_frame_map, frame_data, enemy_camp)

    def get_reward(self, frame_data, reward_dict):
        reward_dict.clear()
        frame_no = frame_data.get("frame_no", frame_data.get("frameNo", 0))
        reward_sum = 0.0
        main_hero = self._find_main_hero(frame_data)

        for reward_name, reward_struct in self.m_cur_calc_frame_map.items():
            if reward_name == "ep_rate":
                reward_struct.cur_frame_value = self.m_main_calc_frame_map[reward_name].cur_frame_value
                reward_struct.last_frame_value = self.m_main_calc_frame_map[reward_name].last_frame_value
                reward_struct.value = (
                    reward_struct.cur_frame_value - reward_struct.last_frame_value
                    if reward_struct.last_frame_value > 0
                    else 0.0
                )
            elif reward_name == "exp" and main_hero is not None and _get(main_hero, "level", 1) >= 15:
                reward_struct.value = 0.0
            elif reward_name == "forward":
                reward_struct.value = (
                    self.m_main_calc_frame_map[reward_name].cur_frame_value
                    - self.m_main_calc_frame_map[reward_name].last_frame_value
                )
                if GameConfig.REMOVE_FORWARD_AFTER is not None and frame_no > GameConfig.REMOVE_FORWARD_AFTER:
                    reward_struct.value = 0.0
            elif reward_name == "last_hit":
                reward_struct.value = self.m_main_calc_frame_map[reward_name].cur_frame_value
            else:
                reward_struct.cur_frame_value = (
                    self.m_main_calc_frame_map[reward_name].cur_frame_value
                    - self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                )
                reward_struct.last_frame_value = (
                    self.m_main_calc_frame_map[reward_name].last_frame_value
                    - self.m_enemy_calc_frame_map[reward_name].last_frame_value
                )
                reward_struct.value = reward_struct.cur_frame_value - reward_struct.last_frame_value

            time_scale = 1.0
            if self.time_scale_arg > 0 and reward_name not in GameConfig.REWARD_WITHOUT_TIME_SCALE:
                time_scale = math.pow(0.6, float(frame_no) / self.time_scale_arg)

            if not self.has_last_frame:
                reward_struct.value = 0.0

            reward_dict[f"{reward_name}_origin"] = reward_struct.value
            reward_dict[f"{reward_name}_weight"] = reward_struct.value * reward_struct.weight * time_scale
            reward_sum += reward_dict[f"{reward_name}_weight"]

        reward_dict["reward_sum"] = reward_sum
        reward_dict.update(self.m_last_hit_debug)
        self.has_last_frame = True
        return reward_dict

    def _find_main_hero(self, frame_data):
        for hero in frame_data.get("hero_states", []):
            if self._is_main_hero(hero):
                return hero
        return None

    def _is_main_hero(self, hero):
        return (
            _get(hero, "runtime_id", None) == self.main_hero_player_id
            or _get(hero, "player_id", None) == self.main_hero_player_id
        )

    def _find_heroes_by_camp(self, frame_data, camp):
        main_hero, enemy_hero = None, None
        for hero in frame_data.get("hero_states", []):
            if _camp_key(_get(hero, "camp")) == _camp_key(camp):
                main_hero = hero
            else:
                enemy_hero = hero
        return main_hero, enemy_hero

    def _find_towers_by_camp(self, frame_data, camp):
        main_tower, enemy_tower = None, None
        for npc in frame_data.get("npc_states", []):
            if _get(npc, "sub_type") not in TOWER_SUB_TYPES:
                continue
            if _camp_key(_get(npc, "camp")) == _camp_key(camp):
                main_tower = npc
            else:
                enemy_tower = npc
        return main_tower, enemy_tower

    def _count_last_hit(self, frame_data, main_hero, enemy_hero):
        frame_action = _get_any(frame_data, ["frame_action", "frameAction"], {}) or {}
        dead_actions = _get_any(frame_action, ["dead_action", "deadAction"], [])
        if not isinstance(dead_actions, list):
            dead_actions = []
        main_runtime = self._actor_runtime(main_hero)
        enemy_runtime = self._actor_runtime(enemy_hero)
        debug = self._empty_last_hit_debug()
        debug["last_hit_dead_action_count"] = len(dead_actions)
        value = 0.0
        for dead_action in dead_actions:
            killer = _get_any(dead_action, ["killer", "killer_actor"], {}) if isinstance(dead_action, dict) else {}
            death = _get_any(dead_action, ["death", "dead"], {}) if isinstance(dead_action, dict) else {}
            if self._actor_sub_type(death) not in SOLDIER_SUB_TYPES:
                continue
            debug["last_hit_soldier_dead_count"] += 1
            killer_id = self._actor_runtime(killer)
            if killer_id is None:
                killer_id = self._last_hurt_runtime(death)
            if killer_id == main_runtime:
                value += 1.0
                debug["last_hit_main_count"] += 1
            elif killer_id == enemy_runtime:
                value -= 1.0
                debug["last_hit_enemy_count"] += 1
        return value, debug

    def _empty_last_hit_debug(self):
        return {
            "last_hit_dead_action_count": 0,
            "last_hit_soldier_dead_count": 0,
            "last_hit_main_count": 0,
            "last_hit_enemy_count": 0,
        }

    def _actor_runtime(self, actor):
        if actor is None:
            return None
        actor_state = _get_any(actor, ["actor_state", "actorState"], None)
        if actor_state is not None:
            runtime_id = _get_any(actor_state, ["runtime_id", "runtimeId"], None)
            if runtime_id is not None:
                return runtime_id
        return _get_any(actor, ["runtime_id", "runtimeId"], None)

    def _actor_sub_type(self, actor):
        if actor is None:
            return None
        actor_state = _get_any(actor, ["actor_state", "actorState"], None)
        if actor_state is not None:
            sub_type = _get_any(actor_state, ["sub_type", "subType"], None)
            if sub_type is not None:
                return sub_type
        return _get_any(actor, ["sub_type", "subType"], None)

    def _last_hurt_runtime(self, death):
        single_hurt_list = _get_any(death, ["single_hurt_list", "singleHurtList"], [])
        if not isinstance(single_hurt_list, list) or not single_hurt_list:
            return None
        last_hurt = max(
            single_hurt_list,
            key=lambda item: _get_any(item, ["frameNo", "frame_no", "frame"], -1),
        )
        return _get_any(last_hurt, ["runtime_id", "runtimeId"], None)

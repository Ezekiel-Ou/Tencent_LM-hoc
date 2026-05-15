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
        self.init_max_exp_of_each_hero()

    def init_max_exp_of_each_hero(self):
        self.m_each_level_max_exp = {
            1: 160,
            2: 298,
            3: 446,
            4: 524,
            5: 613,
            6: 713,
            7: 825,
            8: 950,
            9: 1088,
            10: 1240,
            11: 1406,
            12: 1585,
            13: 1778,
            14: 1984,
        }

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
                reward_struct.cur_frame_value = self._count_last_hit(frame_data, main_hero, enemy_hero)
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
                reward_struct.value = self.m_main_calc_frame_map[reward_name].cur_frame_value
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
        frame_action = frame_data.get("frame_action", {}) or {}
        dead_actions = frame_action.get("dead_action", []) if isinstance(frame_action, dict) else []
        main_runtime = _get(main_hero or {}, "runtime_id", _get(main_hero or {}, "player_id", None))
        enemy_runtime = _get(enemy_hero or {}, "runtime_id", _get(enemy_hero or {}, "player_id", None))
        value = 0.0
        for dead_action in dead_actions:
            killer = dead_action.get("killer", {}) if isinstance(dead_action, dict) else {}
            death = dead_action.get("death", {}) if isinstance(dead_action, dict) else {}
            killer_id = _get(killer, "runtime_id", None)
            if _get(death, "sub_type") not in SOLDIER_SUB_TYPES:
                continue
            if killer_id == main_runtime:
                value += 1.0
            elif killer_id == enemy_runtime:
                value -= 1.0
        return value

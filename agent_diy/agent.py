#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

import hashlib
import json
import os
import random

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR

try:
    from kaiwu_agent.agent.base_agent import (
        BaseAgent,
        predict_wrapper,
        exploit_wrapper,
        learn_wrapper,
        save_model_wrapper,
        load_model_wrapper,
        reset_wrapper,
        load_opponent_agent_wrapper,
    )
    from kaiwu_agent.utils.common_func import attached
except ModuleNotFoundError:
    from kaiwudrl.interface.agent import BaseAgent

    def _identity_wrapper(func):
        return func

    predict_wrapper = _identity_wrapper
    exploit_wrapper = _identity_wrapper
    learn_wrapper = _identity_wrapper
    save_model_wrapper = _identity_wrapper
    load_model_wrapper = _identity_wrapper
    reset_wrapper = _identity_wrapper
    load_opponent_agent_wrapper = _identity_wrapper
    attached = _identity_wrapper

from agent_diy.algorithm.algorithm import Algorithm
from agent_diy.conf.conf import Args, Config, GameConfig
from agent_diy.feature.definition import NONE_ACTION, ActData, ObsData
from agent_diy.feature.feature_process import FeatureProcess
from agent_diy.feature.reward_process import GameRewardManager
from agent_diy.model.model import Model


torch.set_num_threads(1)
torch.set_num_interop_threads(1)


SUMMONER_SKILL_MAP = {
    80102: "治疗",
    80109: "疾跑",
    80104: "惩击",
    80108: "终结",
    80110: "狂暴",
    80105: "干扰",
    80103: "晕眩",
    80107: "净化",
    80121: "弱化",
    80115: "闪现",
}


@attached
class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        self.cur_model_name = ""
        self.device = device
        self.model = Model().to(self.device)
        self.model = self.model.to(memory_format=torch.channels_last)

        self.lstm_unit_size = Config.LSTM_UNIT_SIZE
        self.lstm_hidden = np.zeros([self.lstm_unit_size], dtype=np.float32)
        self.lstm_cell = np.zeros([self.lstm_unit_size], dtype=np.float32)
        self.label_size_list = Config.LABEL_SIZE_LIST
        self.legal_action_size = Config.LEGAL_ACTION_SIZE_LIST
        self.seri_vec_split_shape = Config.SERI_VEC_SPLIT_SHAPE

        self.hero_camp = 0
        self.player_id = 0
        self.env_id = None
        self.summoner_skill_counter = {}

        self.train_step = 0
        self.lr = Config.INIT_LEARNING_RATE_START
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-8)
        self.parameters = [p for param_group in self.optimizer.param_groups for p in param_group["params"]]
        self.target_lr = Config.TARGET_LR
        self.target_step = Config.TARGET_STEP
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=self.lr_lambda)

        self.reward_manager = None
        self.feature_processes = None
        self.logger = logger
        self.monitor = monitor
        self.algorithm = Algorithm(self.model, self.optimizer, self.scheduler, self.device, self.logger, self.monitor)

        # Force-home rule state (used by _maybe_force_home).
        self.own_cake_exists = False
        self.own_cake_seen_once = False
        self.last_own_cake_disappear_frame = -10000
        self.next_own_cake_frame = None
        self.prev_dead_cnt = 0
        self.force_home_phase = None
        self.rule_override_active = False
        self.recall_override_active = False
        self.cleanse_override_active = False
        self.rule_override_count = 0
        self.force_home_override_count = 0
        self.force_home_start_count = 0
        self.force_home_retreat_count = 0
        self.force_home_return_count = 0
        self.luban_skill1_aim_assist_count = 0
        self.cleanse_override_count = 0

        super().__init__(agent_type, device, logger, monitor)

    def lr_lambda(self, step):
        if step > self.target_step:
            return self.target_lr / self.lr
        return 1.0 - ((1.0 - self.target_lr / self.lr) * step / self.target_step)

    def init_config(self, config_data):
        my_heroes = config_data.get("my_heroes", [])
        opponent_heroes = config_data.get("opponent_heroes", [])
        opponent_hero = opponent_heroes[0] if opponent_heroes else None
        is_eval = bool(config_data.get("is_eval", False))
        select_skills = {}
        for hero_id in my_heroes:
            if is_eval:
                select_skills[hero_id] = self._default_summoner_skill(hero_id, opponent_hero)
            else:
                select_skills[hero_id] = self._select_train_summoner_skill(hero_id, opponent_hero)
        return select_skills

    def _default_summoner_skill(self, my_hero, opponent_hero):
        # Eval / match always uses the configured default (currently 80110 狂暴).
        # Training cycle path also falls back here when candidates are exhausted.
        return GameConfig.DEFAULT_SUMMONER_SKILL

    def _select_train_summoner_skill(self, my_hero, opponent_hero):
        default_skill = self._default_summoner_skill(my_hero, opponent_hero)
        candidates = list(GameConfig.SUMMONER_SKILL_CANDIDATES_BY_HERO.get(my_hero, []))
        if default_skill not in candidates:
            candidates.insert(0, default_skill)
        candidates = [skill_id for skill_id in candidates if skill_id in GameConfig.SUMMONER_SKILL_IDS]
        if not candidates:
            return default_skill

        mode = getattr(GameConfig, "SUMMONER_SKILL_TRAIN_MODE", "cycle")
        if mode == "fixed":
            return default_skill
        if mode == "random":
            return random.choice(candidates)

        counter_key = (int(my_hero), int(opponent_hero or 0))
        counter = self.summoner_skill_counter.get(counter_key, 0)
        self.summoner_skill_counter[counter_key] = counter + 1
        return candidates[counter % len(candidates)]

    @reset_wrapper
    def reset(self, observation):
        self.hero_camp = observation.get("player_camp", observation.get("camp", 0))
        self.player_id = observation.get("player_id", 0)
        self.lstm_hidden = np.zeros([self.lstm_unit_size], dtype=np.float32)
        self.lstm_cell = np.zeros([self.lstm_unit_size], dtype=np.float32)
        self.reward_manager = GameRewardManager(self.player_id)
        self.feature_processes = FeatureProcess(self.hero_camp, logger=self.logger)
        # Reset force-home rule state per episode.
        self.own_cake_exists = False
        self.own_cake_seen_once = False
        self.last_own_cake_disappear_frame = -10000
        self.next_own_cake_frame = None
        self.prev_dead_cnt = 0
        self.force_home_phase = None
        self.rule_override_active = False
        self.recall_override_active = False
        self.cleanse_override_active = False
        self.rule_override_count = 0
        self.force_home_override_count = 0
        self.force_home_start_count = 0
        self.force_home_retreat_count = 0
        self.force_home_return_count = 0
        self.luban_skill1_aim_assist_count = 0
        self.cleanse_override_count = 0

    def _model_inference(self, list_obs_data):
        feature = [obs_data.feature for obs_data in list_obs_data]
        legal_action = [obs_data.legal_action for obs_data in list_obs_data]
        lstm_cell = [obs_data.lstm_cell for obs_data in list_obs_data]
        lstm_hidden = [obs_data.lstm_hidden for obs_data in list_obs_data]

        input_list = [np.array(feature), np.array(lstm_cell), np.array(lstm_hidden)]
        torch_inputs = [torch.from_numpy(nparr).to(torch.float32) for nparr in input_list]
        for i, data in enumerate(torch_inputs):
            torch_inputs[i] = data.reshape(-1).float().to(self.device)

        feature, lstm_cell, lstm_hidden = torch_inputs
        feature_vec = feature.reshape(-1, self.seri_vec_split_shape[0][0])
        lstm_hidden_state = lstm_hidden.reshape(-1, self.lstm_unit_size)
        lstm_cell_state = lstm_cell.reshape(-1, self.lstm_unit_size)

        format_inputs = [feature_vec, lstm_hidden_state, lstm_cell_state]

        self.model.set_eval_mode()
        with torch.no_grad():
            output_list = self.model(format_inputs, inference=True)

        np_output = [output.detach().cpu().numpy() for output in output_list]
        logits, value, next_lstm_cell, next_lstm_hidden = np_output[:4]

        next_lstm_cell = next_lstm_cell.squeeze(axis=0)
        next_lstm_hidden = next_lstm_hidden.squeeze(axis=0)

        list_act_data = []
        for i in range(len(legal_action)):
            prob, d_prob, action, d_action = self._sample_masked_action(logits[i], legal_action[i])
            list_act_data.append(
                ActData(
                    action=action,
                    d_action=d_action,
                    prob=prob,
                    d_prob=d_prob,
                    value=value[i],
                    lstm_cell=next_lstm_cell[i],
                    lstm_hidden=next_lstm_hidden[i],
                )
            )
        return list_act_data

    @predict_wrapper
    def predict(self, observation):
        self._reset_rule_override_flags()
        obs_data = self.observation_process(observation)
        act_data = self._model_inference([obs_data])[0]
        self._maybe_aim_luban_skill1(observation, act_data, is_stochastic=True)
        self.update_status(obs_data, act_data)
        action = self.action_process(observation, act_data, True)
        action = self._maybe_auto_cleanse(observation, action)
        return self._maybe_force_home(observation, action)

    @exploit_wrapper
    def exploit(self, observation):
        self._reset_rule_override_flags()
        obs_data = self.observation_process(observation)
        act_data = self._model_inference([obs_data])[0]
        self._maybe_aim_luban_skill1(observation, act_data, is_stochastic=False)
        self.update_status(obs_data, act_data)
        action = self.action_process(observation, act_data, False)
        action = self._maybe_auto_cleanse(observation, action)
        return self._maybe_force_home(observation, action)

    def observation_process(self, observation):
        if self.feature_processes is None:
            self.feature_processes = FeatureProcess(observation.get("player_camp", observation.get("camp", 0)))
        feature = self.feature_processes.process_feature(observation)
        if len(feature) != Config.FEATURE_DIM:
            raise ValueError(f"feature dim mismatch: {len(feature)} != {Config.FEATURE_DIM}")
        legal_action = observation["legal_action"]
        if len(legal_action) not in (Config.LEGAL_ACTION_DIM, Config.RAW_LEGAL_ACTION_DIM):
            raise ValueError(
                f"legal_action dim mismatch: {len(legal_action)} not in "
                f"({Config.LEGAL_ACTION_DIM}, {Config.RAW_LEGAL_ACTION_DIM})"
            )
        return ObsData(
            feature=feature,
            legal_action=legal_action,
            lstm_cell=self.lstm_cell,
            lstm_hidden=self.lstm_hidden,
        )

    def action_process(self, observation, act_data, is_stochastic):
        action = act_data.action if is_stochastic else act_data.d_action
        return self._normalize_action(action)

    def _maybe_aim_luban_skill1(self, observation, act_data, is_stochastic):
        if not getattr(GameConfig, "LUBAN_SKILL1_AIM_ASSIST", False):
            return
        attr = "action" if is_stochastic else "d_action"
        action = self._normalize_action(getattr(act_data, attr, None))
        # Only correct the sub-actions after the policy has already selected
        # Luban skill 1. The rule does not force skill usage.
        if len(action) != 6 or action[0] != 4:
            return

        frame_state = observation.get("frame_state", {}) or {}
        main_hero, enemy_hero, _ = self._find_my_hero_and_tower(frame_state)
        if self._hero_config_id(main_hero) != 112 or enemy_hero is None:
            return
        if self._distance_between_heroes(main_hero, enemy_hero) > GameConfig.LUBAN_SKILL1_AIM_RANGE:
            return

        center = int(GameConfig.LUBAN_SKILL1_AIM_CENTER)
        target = int(GameConfig.LUBAN_SKILL1_AIM_TARGET)
        if not self._is_luban_skill1_aim_legal(observation, center, target):
            return

        action[3] = center
        action[4] = center
        action[5] = target
        setattr(act_data, attr, action)
        self.rule_override_active = True
        self.rule_override_count += 1
        self.luban_skill1_aim_assist_count += 1

    def _is_luban_skill1_aim_legal(self, observation, center, target):
        legal_action = observation.get("legal_action", [])
        if legal_action is None:
            legal_action = []
        if hasattr(legal_action, "tolist"):
            legal_action = legal_action.tolist()
        if len(legal_action) not in (Config.LEGAL_ACTION_DIM, Config.RAW_LEGAL_ACTION_DIM):
            return False
        offsets = [0]
        for size in Config.LABEL_SIZE_LIST[:-1]:
            offsets.append(offsets[-1] + size)
        button = 4
        if int(legal_action[button] or 0) != 1:
            return False
        skill_x_offset = offsets[3]
        skill_z_offset = offsets[4]
        if not (0 <= center < Config.LABEL_SIZE_LIST[3]):
            return False
        if int(legal_action[skill_x_offset + center] or 0) != 1:
            return False
        if int(legal_action[skill_z_offset + center] or 0) != 1:
            return False
        target_offset = offsets[5]
        if len(legal_action) == Config.RAW_LEGAL_ACTION_DIM:
            target_offset += button * Config.LABEL_SIZE_LIST[-1]
        return 0 <= target < Config.LABEL_SIZE_LIST[-1] and int(legal_action[target_offset + target] or 0) == 1

    def _hero_config_id(self, hero):
        if not hero:
            return 0
        return int(hero.get("config_id", hero.get("configId", 0)) or 0)

    def _hero_location(self, hero):
        if not hero:
            return None
        collider = hero.get("collider", {}) or {}
        loc = collider.get("location", None) or hero.get("location", None)
        if not loc:
            return None
        return float(loc.get("x", 0) or 0), float(loc.get("z", 0) or 0)

    def _distance_between_heroes(self, hero_a, hero_b):
        loc_a = self._hero_location(hero_a)
        loc_b = self._hero_location(hero_b)
        if loc_a is None or loc_b is None:
            return float("inf")
        dx = loc_a[0] - loc_b[0]
        dz = loc_a[1] - loc_b[1]
        return (dx * dx + dz * dz) ** 0.5

    def _dist(self, a, b, default=1e9):
        if a is None or b is None:
            return default
        dx = float(a[0]) - float(b[0])
        dz = float(a[1]) - float(b[1])
        return (dx * dx + dz * dz) ** 0.5

    def _maybe_auto_cleanse(self, observation, action):
        frame_state = observation.get("frame_state", {}) or {}
        main_hero, enemy_hero, _ = self._find_my_hero_and_tower(frame_state)
        if main_hero is None or enemy_hero is None:
            return action

        if self._hero_config_id(main_hero) != 133 or self._hero_config_id(enemy_hero) != 133:
            return action

        main_runtime = self._actor_runtime(main_hero)
        enemy_runtime = self._actor_runtime(enemy_hero)
        hit_by_ult = False
        for hurt in main_hero.get("take_hurt_infos", []) or []:
            atker = hurt.get("atker") or hurt.get("attacker")
            slot = self._slot_idx(hurt.get("skillSlot")) if "skillSlot" in hurt else self._slot_idx(hurt.get("skill_slot"))
            if atker == enemy_runtime and slot == 3:
                hit_by_ult = True
                break

        if not hit_by_ult:
            return action

        if not self._is_skill2_available(main_hero):
            return action

        cleanse_action = self._legalized_rule_action(observation, [5, 15, 15, 15, 15, 2])
        if cleanse_action is None:
            return action

        self.rule_override_active = True
        self.cleanse_override_active = True
        self.rule_override_count += 1
        self.cleanse_override_count += 1
        return cleanse_action

    def _actor_runtime(self, actor):
        if actor is None:
            return None
        actor_state = actor.get("actor_state") or actor.get("actorState")
        if actor_state is not None:
            runtime_id = actor_state.get("runtime_id") or actor_state.get("runtimeId")
            if runtime_id is not None:
                return runtime_id
        return actor.get("runtime_id") or actor.get("runtimeId")

    def _is_skill2_available(self, main_hero):
        slots = self._slot_states(main_hero)
        for slot in slots:
            if self._slot_idx(self._get_any(slot, ["slot_type", "slotType"])) != 2:
                continue
            usable = bool(slot.get("usable", False))
            cooldown = float(slot.get("cooldown", 0) or 0)
            return usable and cooldown <= 0
        return False

    def _slot_states(self, hero):
        skill_state = self._get_any(hero or {}, ["skill_state", "skillState"], {}) or {}
        return self._get_any(skill_state, ["slot_states", "slotStates"], []) or []

    def _get_any(self, obj, keys, default=None):
        for key in keys:
            if isinstance(obj, dict):
                value = obj.get(key, None)
            else:
                value = getattr(obj, key, None)
            if value is not None:
                return value
        return default

    def _reset_rule_override_flags(self):
        self.rule_override_active = False
        self.recall_override_active = False
        self.cleanse_override_active = False

    def _legalized_rule_action(self, observation, preferred_action, active_heads=None):
        legal_action = observation.get("legal_action", [])
        if legal_action is None:
            legal_action = []
        if hasattr(legal_action, "tolist"):
            legal_action = legal_action.tolist()
        if len(legal_action) not in (Config.LEGAL_ACTION_DIM, Config.RAW_LEGAL_ACTION_DIM):
            return None

        preferred_action = self._normalize_action(preferred_action)
        button = preferred_action[0]
        if not (0 <= button < Config.LABEL_SIZE_LIST[0]):
            return None
        if int(legal_action[button] or 0) != 1:
            return None

        offsets = [0]
        for size in Config.LABEL_SIZE_LIST[:-1]:
            offsets.append(offsets[-1] + size)

        action = list(preferred_action)
        active_heads = set(range(1, len(Config.LABEL_SIZE_LIST))) if active_heads is None else set(active_heads)
        for head_idx in range(1, len(Config.LABEL_SIZE_LIST)):
            if head_idx not in active_heads:
                continue
            if head_idx == len(Config.LABEL_SIZE_LIST) - 1 and len(legal_action) == Config.RAW_LEGAL_ACTION_DIM:
                size = Config.LABEL_SIZE_LIST[-1]
                target_base = offsets[-1]
                mask = legal_action[target_base + button * size : target_base + (button + 1) * size]
            else:
                size = Config.LABEL_SIZE_LIST[head_idx]
                mask = legal_action[offsets[head_idx] : offsets[head_idx] + size]
            action[head_idx] = self._preferred_or_first_legal(action[head_idx], mask)
            if action[head_idx] is None:
                return None
        return [int(value) for value in action]

    def _preferred_or_first_legal(self, preferred_idx, mask):
        if hasattr(mask, "tolist"):
            mask = mask.tolist()
        legal_indices = [idx for idx, value in enumerate(mask) if int(value or 0) == 1]
        if not legal_indices:
            return None
        if 0 <= int(preferred_idx) < len(mask) and int(mask[int(preferred_idx)] or 0) == 1:
            return int(preferred_idx)
        return int(legal_indices[0])

    def _maybe_force_home(self, observation, action):
        # Force walking back to base when low HP and safe, then return to the
        # first-tower area after healing. The task has no usable recall button.
        self.recall_override_active = False

        frame_state = observation.get("frame_state", {}) or {}
        frame_no = frame_state.get("frame_no", frame_state.get("frameNo", 0)) or 0

        main_hero, enemy_hero, main_tower = self._find_my_hero_and_tower(frame_state)
        if main_hero is None:
            self.force_home_phase = None
            return action

        max_hp = float(main_hero.get("max_hp", 1) or 1)
        hp = float(main_hero.get("hp", 0) or 0)
        hp_rate = hp / max(max_hp, 1.0)
        if hp <= 0:
            self.force_home_phase = None
            return action

        dead_cnt = int(main_hero.get("dead_cnt", 0) or 0)
        if dead_cnt > self.prev_dead_cnt:
            self.prev_dead_cnt = dead_cnt
            self.force_home_phase = None
            return action
        self.prev_dead_cnt = dead_cnt

        if self.force_home_phase == "retreat" and hp_rate >= GameConfig.FORCE_HOME_HP_RECOVERED:
            self.force_home_phase = "return"
        if self.force_home_phase == "return" and hp_rate <= GameConfig.FORCE_HOME_HP_TRIGGER:
            self.force_home_phase = "retreat"

        if self.force_home_phase == "retreat":
            if not self._has_valid_home_anchor_context(frame_state, frame_no):
                self.force_home_phase = None
                return action
            return self._force_walk_to(observation, main_hero, self._force_home_base_target(), action, phase="retreat")

        if self.force_home_phase == "return":
            target = self._force_home_return_target(main_tower)
            if target is None:
                self.force_home_phase = None
                return action
            if self._distance_to_raw_target(main_hero, target) <= GameConfig.FORCE_HOME_RETURN_RADIUS:
                self.force_home_phase = None
                return action
            return self._force_walk_to(observation, main_hero, target, action, phase="return")

        if hp_rate > GameConfig.FORCE_HOME_HP_TRIGGER:
            return action

        if not self._enemy_invisible_or_far(main_hero, enemy_hero):
            return action
        if not self._has_valid_home_anchor_context(frame_state, frame_no):
            return action

        self.force_home_phase = "retreat"
        return self._force_walk_to(
            observation,
            main_hero,
            self._force_home_base_target(),
            action,
            phase="retreat",
            count_start=True,
        )

    def _has_valid_home_anchor_context(self, frame_state, frame_no):
        _, _, main_tower = self._find_my_hero_and_tower(frame_state)
        if main_tower is None:
            return False
        if not self._tower_hp_above(main_tower, GameConfig.FORCE_HOME_TOWER_HP_MIN):
            return False
        if not self._is_own_cake_unavailable_for_force_home(frame_state, frame_no):
            return False
        return True

    def _tower_hp_above(self, tower, threshold):
        hp = float(tower.get("hp", 0) or 0)
        max_hp = float(tower.get("max_hp", 1) or 1)
        return hp / max(max_hp, 1.0) >= float(threshold)

    def _force_walk_to(self, observation, main_hero, target, fallback_action, phase=None, count_start=False):
        move_action = self._move_action_towards(main_hero, target)
        if move_action is None:
            return fallback_action
        legal_move = self._legalized_rule_action(observation, move_action, active_heads=(1, 2))
        if legal_move is None:
            return fallback_action
        self.rule_override_active = True
        self.rule_override_count += 1
        self.force_home_override_count += 1
        if count_start:
            self.force_home_start_count += 1
        if phase == "retreat":
            self.force_home_retreat_count += 1
        elif phase == "return":
            self.force_home_return_count += 1
        return legal_move

    def _move_action_towards(self, main_hero, target):
        loc = self._hero_location(main_hero)
        if loc is None or target is None:
            return None
        dx = float(target[0]) - loc[0]
        dz = float(target[1]) - loc[1]
        return [
            2,
            self._direction_bucket(dx),
            self._direction_bucket(dz),
            8,
            8,
            0,
        ]

    def _direction_bucket(self, delta):
        deadzone = float(GameConfig.FORCE_HOME_DIRECTION_DEADZONE)
        if delta > deadzone:
            return 15
        if delta < -deadzone:
            return 0
        return 8

    def _enemy_invisible_or_far(self, main_hero, enemy_hero):
        if enemy_hero is None:
            return True
        main_loc = self._hero_location(main_hero)
        enemy_loc = self._hero_location(enemy_hero)
        if main_loc is None or enemy_loc is None:
            return True
        dx = main_loc[0] - enemy_loc[0]
        dz = main_loc[1] - enemy_loc[1]
        return (dx * dx + dz * dz) ** 0.5 > GameConfig.FORCE_HOME_ENEMY_SAFE_RANGE

    def _force_home_base_target(self):
        return self._unproject_own_perspective(Args.SELF_BASE_ANCHOR)

    def _force_home_return_target(self, main_tower):
        tower_loc = self._hero_location(main_tower)
        if tower_loc is not None:
            return tower_loc
        return self._unproject_own_perspective(Args.SELF_TOWER_ANCHOR)

    def _distance_to_raw_target(self, unit, target):
        loc = self._hero_location(unit)
        if loc is None or target is None:
            return float("inf")
        dx = loc[0] - float(target[0])
        dz = loc[1] - float(target[1])
        return (dx * dx + dz * dz) ** 0.5

    def _project_own_perspective(self, obj_or_loc):
        loc = obj_or_loc
        if obj_or_loc is not None and not isinstance(obj_or_loc, dict):
            return None
        if isinstance(obj_or_loc, dict) and "x" not in obj_or_loc:
            loc = (obj_or_loc.get("collider", {}) or {}).get("location", None) or obj_or_loc.get("location", None)
        if not loc:
            return None
        x = loc.get("x", None)
        z = loc.get("z", None)
        if x is None or z is None:
            return None
        x = float(x)
        z = float(z)
        if abs(x) > Args.RAW_COORD_ABS_LIMIT or abs(z) > Args.RAW_COORD_ABS_LIMIT:
            return None
        if self._camp_key(self.hero_camp) == 2:
            x, z = -x, -z
        lane = (x + z) / Args.SQRT2
        width = (x - z) / Args.SQRT2
        return lane, width

    def _unproject_own_perspective(self, projected):
        lane, width = float(projected[0]), float(projected[1])
        x = (lane + width) / Args.SQRT2
        z = (lane - width) / Args.SQRT2
        if self._camp_key(self.hero_camp) == 2:
            x, z = -x, -z
        return x, z

    def _camp_key(self, camp):
        if camp in (1, "1", "PLAYERCAMP_1", "blue_camp"):
            return 1
        if camp in (2, "2", "PLAYERCAMP_2", "red_camp"):
            return 2
        return camp

    def _find_my_hero_and_tower(self, frame_state):
        main_hero = None
        enemy_hero = None
        main_towers = []
        main_camp = self._camp_key(self.hero_camp)
        for hero in frame_state.get("hero_states", []) or []:
            if self._camp_key(hero.get("camp")) == main_camp:
                main_hero = hero
            else:
                enemy_hero = hero
        for npc in frame_state.get("npc_states", []) or []:
            sub_type = npc.get("sub_type", None)
            if sub_type not in (21, "21", "ACTOR_SUB_TOWER"):
                continue
            if self._camp_key(npc.get("camp")) == main_camp:
                main_towers.append(npc)
        main_tower = self._select_nearest_projected_anchor(main_towers, Args.SELF_TOWER_ANCHOR)
        return main_hero, enemy_hero, main_tower

    def _select_nearest_projected_anchor(self, units, anchor):
        if not units:
            return None
        return sorted(units, key=lambda unit: self._dist(self._project_own_perspective(unit), anchor))[0]

    def _update_own_cake_state(self, frame_state, frame_no):
        cakes = frame_state.get("cakes", []) or []
        found_own = False
        for cake in cakes:
            pos = self._project_own_perspective(cake)
            if pos is None:
                continue
            if self._dist(pos, Args.SELF_CAKE_ANCHOR) <= self._dist(pos, Args.ENEMY_CAKE_ANCHOR):
                found_own = True
                break
        if found_own:
            self.own_cake_seen_once = True
            self.next_own_cake_frame = None
        if self.own_cake_exists and not found_own:
            self.last_own_cake_disappear_frame = frame_no
            self.next_own_cake_frame = frame_no + 2250
        self.own_cake_exists = found_own

    def _is_own_cake_unavailable_for_force_home(self, frame_state, frame_no):
        self._update_own_cake_state(frame_state, frame_no)
        if self.own_cake_exists:
            return False
        if not self.own_cake_seen_once:
            return False
        if self.next_own_cake_frame is None:
            return False
        # "Within 5s" = 150 frames before respawn; do not force home during that window.
        return self.next_own_cake_frame - frame_no > 150

    def _is_recover_skill_available(self, main_hero):
        slots = self._slot_states(main_hero)
        for slot in slots:
            if self._slot_idx(self._get_any(slot, ["slot_type", "slotType"])) != 4:
                continue
            usable = bool(slot.get("usable", False))
            cooldown = float(slot.get("cooldown", 0) or 0)
            return usable and cooldown <= 0
        return False

    def _slot_idx(self, slot_type):
        if isinstance(slot_type, int):
            return slot_type
        text = str(slot_type) if slot_type is not None else ""
        if text.startswith("SLOT_SKILL_"):
            try:
                return int(text.split("_")[-1])
            except (TypeError, ValueError):
                return -1
        try:
            return int(slot_type) if slot_type is not None else -1
        except (TypeError, ValueError):
            return -1

    @learn_wrapper
    def learn(self, list_sample_data):
        return self.algorithm.learn(list_sample_data)

    @save_model_wrapper
    def save_model(self, path=None, id="1"):
        model_file_path = self._model_file_path(path, id, create_dir=True)
        checkpoint = self._build_checkpoint_payload()
        torch.save(checkpoint, model_file_path)
        self._save_checkpoint_metadata(model_file_path, checkpoint["checkpoint_meta"])
        self._log_info(f"save model {model_file_path} successfully")

    @load_model_wrapper
    def load_model(self, path=None, id="1"):
        model_file_path = self._model_file_path(path, id)
        if not os.path.exists(model_file_path):
            if str(id) == "latest":
                self._log_warning(f"latest model {model_file_path} not found, keep current parameters")
                return
            raise FileNotFoundError(f"model file {model_file_path} not found")

        if self.cur_model_name == model_file_path:
            return
        else:
            state_dict = self._load_checkpoint_state_dict(model_file_path)
            self.model.load_state_dict(state_dict)
            self.cur_model_name = model_file_path
            self._log_info(f"load model {model_file_path} successfully")

    @load_opponent_agent_wrapper
    def load_opponent_agent(self, id="1"):
        # The official framework wrapper resolves and loads model-pool opponents.
        pass

    def _model_file_path(self, path, id, create_dir=False):
        if path is None:
            path = os.path.join(os.getcwd(), "agent_diy", "ckpt")
        if create_dir:
            os.makedirs(path, exist_ok=True)
        return os.path.join(path, f"model.ckpt-{str(id)}.pkl")

    def _build_checkpoint_payload(self):
        state_dict = self.model.state_dict()
        meta = {
            "checkpoint_signature": Config.checkpoint_signature(),
            "checkpoint_signature_hash": Config.checkpoint_signature_hash(),
            "model_state_shape_hash": self._state_dict_shape_hash(state_dict),
        }
        return {
            "state_dict": state_dict,
            "checkpoint_meta": meta,
        }

    def _load_checkpoint_state_dict(self, model_file_path):
        checkpoint = self._torch_load(model_file_path)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint and "checkpoint_meta" in checkpoint:
            self._validate_checkpoint_meta(checkpoint["checkpoint_meta"], model_file_path)
            return checkpoint["state_dict"]

        meta = self._load_checkpoint_metadata(model_file_path)
        if meta is not None:
            self._validate_checkpoint_meta(meta, model_file_path)
        else:
            self._log_warning(
                f"checkpoint {model_file_path} has no protocol metadata; treating it as a legacy state_dict"
            )
        return checkpoint

    def _save_checkpoint_metadata(self, model_file_path, meta):
        meta_file_path = self._checkpoint_metadata_path(model_file_path)
        with open(meta_file_path, "w", encoding="utf-8") as file:
            json.dump(meta, file, sort_keys=True, indent=2)

    def _load_checkpoint_metadata(self, model_file_path):
        meta_file_path = self._checkpoint_metadata_path(model_file_path)
        if not os.path.exists(meta_file_path):
            return None
        with open(meta_file_path, "r", encoding="utf-8") as file:
            return json.load(file)

    def _checkpoint_metadata_path(self, model_file_path):
        return f"{model_file_path}.meta.json"

    def _validate_checkpoint_meta(self, meta, model_file_path):
        expected_signature_hash = Config.checkpoint_signature_hash()
        actual_signature_hash = meta.get("checkpoint_signature_hash")
        if actual_signature_hash != expected_signature_hash:
            raise ValueError(
                "checkpoint protocol mismatch for "
                f"{model_file_path}: {actual_signature_hash} != {expected_signature_hash}"
            )

        expected_state_hash = self._state_dict_shape_hash(self.model.state_dict())
        actual_state_hash = meta.get("model_state_shape_hash")
        if actual_state_hash and actual_state_hash != expected_state_hash:
            raise ValueError(
                "checkpoint model shape mismatch for "
                f"{model_file_path}: {actual_state_hash} != {expected_state_hash}"
            )

    def _state_dict_shape_hash(self, state_dict):
        shapes = {
            key: list(value.shape)
            for key, value in state_dict.items()
            if hasattr(value, "shape")
        }
        payload = json.dumps(shapes, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _torch_load(self, model_file_path):
        try:
            return torch.load(model_file_path, map_location=self.device, weights_only=True)
        except TypeError:
            return torch.load(model_file_path, map_location=self.device)

    def _log_info(self, message):
        if self.logger:
            self.logger.info(message)

    def _log_warning(self, message):
        if self.logger:
            self.logger.warning(message)

    def update_status(self, obs_data, act_data):
        self.obs_data = obs_data
        self.act_data = act_data
        self.lstm_cell = act_data.lstm_cell
        self.lstm_hidden = act_data.lstm_hidden

    def _sample_masked_action(self, logits, legal_action):
        prob_list = []
        d_prob_list = []
        action_list = []
        d_action_list = []
        label_split_size = [sum(self.label_size_list[: index + 1]) for index in range(len(self.label_size_list))]
        legal_action = np.asarray(legal_action, dtype=np.float32)
        legal_actions = np.split(legal_action, label_split_size[:-1])
        logits_split = np.split(logits, label_split_size[:-1])

        for index in range(0, len(self.label_size_list) - 1):
            probs = self._legal_soft_max(logits_split[index], legal_actions[index])
            prob_list += list(probs)
            d_prob_list += list(probs)
            action_list.append(self._legal_sample(probs, use_max=False))
            d_action_list.append(self._legal_sample(probs, use_max=True))

        index = len(self.label_size_list) - 1
        if legal_action.shape[0] == Config.RAW_LEGAL_ACTION_DIM:
            target_legal_action_o = np.reshape(
                legal_actions[index],
                [self.legal_action_size[0], self.legal_action_size[-1] // self.legal_action_size[0]],
            )
            one_hot_actions = np.eye(self.label_size_list[0])[action_list[0]].reshape([self.label_size_list[0], 1])
            target_legal_action = np.sum(target_legal_action_o * one_hot_actions, axis=0)
        else:
            target_legal_action_o = None
            target_legal_action = legal_actions[index]

        probs = self._legal_soft_max(logits_split[-1], target_legal_action)
        prob_list += list(probs)
        action_list.append(self._legal_sample(probs, use_max=False))

        if target_legal_action_o is not None:
            one_hot_actions = np.eye(self.label_size_list[0])[d_action_list[0]].reshape([self.label_size_list[0], 1])
            target_legal_action_d = np.sum(target_legal_action_o * one_hot_actions, axis=0)
        else:
            target_legal_action_d = target_legal_action
        probs = self._legal_soft_max(logits_split[-1], target_legal_action_d)
        d_prob_list += list(probs)
        d_action_list.append(self._legal_sample(probs, use_max=True))

        return [prob_list], [d_prob_list], action_list, d_action_list

    def _legal_soft_max(self, input_hidden, legal_action):
        legal_action = np.asarray(legal_action, dtype=np.float64).reshape(-1)
        legal_action = (legal_action > 0).astype(np.float64)
        if legal_action.sum() <= 0:
            legal_action = np.ones_like(legal_action, dtype=np.float64)

        logits = np.asarray(input_hidden, dtype=np.float64).reshape(-1)
        masked = np.where(legal_action > 0, logits, -1e20)
        masked = masked - np.max(masked)
        exp_logits = np.exp(np.clip(masked, -80.0, 80.0)) * legal_action
        probs = exp_logits / max(exp_logits.sum(), 1e-12)
        probs = np.where(legal_action > 0, probs, 0.0)
        return self._normalize_probs(probs, legal_action=legal_action)

    def _legal_sample(self, probs, legal_action=None, use_max=False):
        if legal_action is None:
            legal_action = np.asarray(probs, dtype=np.float64) > 0
        probs = self._normalize_probs(probs, legal_action=legal_action)
        if use_max:
            return int(np.argmax(probs))
        return int(np.random.choice(len(probs), p=probs))

    def _normalize_probs(self, probs, legal_action=None):
        probs = np.asarray(probs, dtype=np.float64).reshape(-1)
        if legal_action is not None:
            legal_action = np.asarray(legal_action, dtype=np.float64).reshape(-1)
            legal_mask = legal_action > 0
        else:
            legal_mask = np.ones_like(probs, dtype=bool)
        probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        probs = np.clip(probs, 0.0, 1.0)
        probs = np.where(legal_mask, probs, 0.0)
        total = probs.sum(dtype=np.float64)
        if total <= 0.0:
            probs = legal_mask.astype(np.float64)
            probs = probs / probs.sum(dtype=np.float64)
        else:
            probs = probs / total

        legal_indices = np.flatnonzero(legal_mask)
        repair_index = int(legal_indices[-1])
        other_indices = np.arange(len(probs)) != repair_index
        other_sum = probs[other_indices].sum(dtype=np.float64)
        if other_sum >= 1.0:
            probs[other_indices] = probs[other_indices] / (other_sum + 1e-12)
            probs[other_indices] *= 1.0 - 1e-12
            other_sum = probs[other_indices].sum(dtype=np.float64)
        probs[repair_index] = max(0.0, 1.0 - other_sum)
        probs = np.where(legal_mask, probs, 0.0)
        return probs

    def _normalize_action(self, action):
        if action is None:
            return list(NONE_ACTION)
        if isinstance(action, np.ndarray):
            action = action.tolist()
        if isinstance(action, tuple):
            action = list(action)
        if isinstance(action, list) and len(action) == 1:
            first = action[0]
            if first is None:
                return list(NONE_ACTION)
            if isinstance(first, np.ndarray):
                first = first.tolist()
            if isinstance(first, tuple):
                first = list(first)
            if isinstance(first, list):
                action = first
        if not isinstance(action, list) or len(action) != len(Config.LABEL_SIZE_LIST):
            if self.logger:
                self.logger.warning(f"invalid action {action}, fallback to NONE_ACTION")
            return list(NONE_ACTION)
        return [int(x) for x in action]

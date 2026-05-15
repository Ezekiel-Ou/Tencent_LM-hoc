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
from agent_diy.conf.conf import Config, GameConfig
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

        super().__init__(agent_type, device, logger, monitor)

    def lr_lambda(self, step):
        if step > self.target_step:
            return self.target_lr / self.lr
        return 1.0 - ((1.0 - self.target_lr / self.lr) * step / self.target_step)

    def init_config(self, config_data):
        my_heroes = config_data.get("my_heroes", [])
        opponent_heroes = config_data.get("opponent_heroes", [])
        opponent_hero = opponent_heroes[0] if opponent_heroes else None
        select_skills = {}
        for hero_id in my_heroes:
            select_skills[hero_id] = self._select_summoner_skill(hero_id, opponent_hero)
        return select_skills

    def _select_summoner_skill(self, my_hero, opponent_hero):
        if my_hero == 112 and opponent_hero == 112:
            return 80115
        if my_hero == 112 and opponent_hero == 133:
            return 80107
        if my_hero == 133 and opponent_hero == 112:
            return 80103
        if my_hero == 133 and opponent_hero == 133:
            return 80108
        return GameConfig.DEFAULT_SUMMONER_SKILL

    @reset_wrapper
    def reset(self, observation):
        self.hero_camp = observation.get("player_camp", observation.get("camp", 0))
        self.player_id = observation.get("player_id", 0)
        self.lstm_hidden = np.zeros([self.lstm_unit_size], dtype=np.float32)
        self.lstm_cell = np.zeros([self.lstm_unit_size], dtype=np.float32)
        self.reward_manager = GameRewardManager(self.player_id)
        self.feature_processes = FeatureProcess(self.hero_camp, logger=self.logger)

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
        obs_data = self.observation_process(observation)
        act_data = self._model_inference([obs_data])[0]
        self.update_status(obs_data, act_data)
        return self.action_process(observation, act_data, True)

    @exploit_wrapper
    def exploit(self, observation):
        obs_data = self.observation_process(observation)
        act_data = self._model_inference([obs_data])[0]
        self.update_status(obs_data, act_data)
        return self.action_process(observation, act_data, False)

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
            self._log_info(f"current model is {model_file_path}, so skip load model")
        else:
            state_dict = self._load_checkpoint_state_dict(model_file_path)
            self.model.load_state_dict(state_dict)
            self.cur_model_name = model_file_path
            self._log_info(f"load model {model_file_path} successfully")

    @load_opponent_agent_wrapper
    def load_opponent_agent(self, id="1"):
        # The official framework wrapper resolves and loads model-pool opponents.
        self._log_info(f"load_opponent_agent delegated to framework, id={id}")
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

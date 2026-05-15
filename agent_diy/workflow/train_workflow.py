#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Training workflow for agent_diy.
"""

import os
import time

from agent_diy.conf.conf import Config, GameConfig
from agent_diy.feature.definition import (
    FrameCollector,
    NONE_ACTION,
    build_frame,
    lineup_iterator_roundrobin_camp_heroes,
    sample_process,
)
from agent_diy.workflow.env_conf_manager import EnvConfManager
from common_python.utils.workflow_disaster_recovery import handle_disaster_recovery
from tools.metrics_utils import get_training_metrics
from tools.model_pool_utils import get_valid_model_pool


def _normalize_env_action(action):
    if action is None:
        return list(NONE_ACTION)
    if hasattr(action, "tolist"):
        action = action.tolist()
    if isinstance(action, tuple):
        action = list(action)
    if isinstance(action, list) and len(action) == 1:
        first = action[0]
        if first is None:
            return list(NONE_ACTION)
        if hasattr(first, "tolist"):
            first = first.tolist()
        if isinstance(first, tuple):
            first = list(first)
        if isinstance(first, list):
            action = first
    if not isinstance(action, list) or len(action) != 6:
        return list(NONE_ACTION)
    return [int(x) for x in action]


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    do_learns = [True, True]
    last_save_model_time = time.time()

    env_conf_manager = EnvConfManager(
        config_path="agent_diy/conf/train_env_conf.toml",
        logger=logger,
    )
    lineup_iterator = lineup_iterator_roundrobin_camp_heroes(GameConfig.HERO_IDS)

    episode_runner = EpisodeRunner(
        env=envs[0],
        agents=agents,
        logger=logger,
        monitor=monitor,
        env_conf_manager=env_conf_manager,
        lineup_iterator=lineup_iterator,
    )

    while True:
        for g_data in episode_runner.run_episodes():
            for index, (do_learn, agent) in enumerate(zip(do_learns, agents)):
                if do_learn and len(g_data[index]) > 0:
                    agent.send_sample_data(g_data[index])
            g_data.clear()

            now = time.time()
            if now - last_save_model_time > GameConfig.MODEL_SAVE_INTERVAL:
                agents[0].save_model()
                last_save_model_time = now


class EpisodeRunner:
    def __init__(self, env, agents, logger, monitor, env_conf_manager, lineup_iterator):
        self.env = env
        self.agents = agents
        self.logger = logger
        self.monitor = monitor
        self.env_conf_manager = env_conf_manager
        self.lineup_iterator = lineup_iterator
        self.agent_num = len(agents)
        self.episode_cnt = 0
        self.last_report_monitor_time = 0
        self.selected_summoner_skills = [None for _ in range(self.agent_num)]

    def _call_init_config(self, usr_conf, is_eval=False):
        blue_hero_ids, red_hero_ids = EnvConfManager.extract_hero_ids_from_usr_conf(usr_conf)
        camp_keys = ["blue_camp", "red_camp"]
        self.selected_summoner_skills = [None for _ in range(self.agent_num)]

        for agent_idx, agent in enumerate(self.agents):
            if agent_idx == 0:
                my_hero_ids = blue_hero_ids
                opponent_hero_ids = red_hero_ids
                camp_key = camp_keys[0]
            else:
                my_hero_ids = red_hero_ids
                opponent_hero_ids = blue_hero_ids
                camp_key = camp_keys[1]

            config_data = {
                "my_camp": camp_key,
                "my_heroes": my_hero_ids,
                "opponent_heroes": opponent_hero_ids,
                "is_eval": is_eval,
            }
            select_skills = agent.init_config(config_data)
            EnvConfManager.inject_select_skills(usr_conf, camp_key, select_skills)
            self.selected_summoner_skills[agent_idx] = self._first_selected_skill(select_skills, my_hero_ids)
            self.logger.info(f"Agent[{agent_idx}] init_config: camp={camp_key}, select_skills={select_skills}")

    def run_episodes(self):
        while True:
            training_metrics = get_training_metrics()
            if training_metrics:
                for key, value in training_metrics.items():
                    if key == "env":
                        for env_key, env_value in value.items():
                            self.logger.info(f"training_metrics {key} {env_key} is {env_value}")
                    else:
                        self.logger.info(f"training_metrics {key} is {value}")

            lineup = next(self.lineup_iterator)
            if sorted(lineup) and any(hero_id not in GameConfig.HERO_IDS for hero_id in lineup):
                raise ValueError(f"unsupported lineup {lineup}; expected heroes {GameConfig.HERO_IDS}")
            usr_conf, is_eval, monitor_side = self.env_conf_manager.update_config(lineup)
            self._call_init_config(usr_conf, is_eval=is_eval)

            env_obs = self.env.reset(usr_conf=usr_conf)
            if handle_disaster_recovery(env_obs, self.logger):
                break

            self._debug_observation(env_obs, prefix="_reset")

            observation = env_obs["observation"]
            self.reset_agents(observation)
            frame_collector = FrameCollector(self.agent_num)

            self.episode_cnt += 1
            reward_sum_list = [0] * self.agent_num
            is_train_test = os.environ.get("is_train_test", "False").lower() == "true"
            self.logger.info(f"Episode {self.episode_cnt} start, usr_conf is {usr_conf}")

            reward_item_sum_list = [
                {reward_name: 0.0 for reward_name in GameConfig.REWARD_WEIGHT_DICT}
                for _ in range(self.agent_num)
            ]
            reward_debug_sum_list = [
                {debug_name: 0.0 for debug_name in GameConfig.REWARD_DEBUG_KEY_LIST}
                for _ in range(self.agent_num)
            ]
            action_debug_sum_list = [
                {debug_name: 0.0 for debug_name in GameConfig.ACTION_DEBUG_KEY_LIST}
                for _ in range(self.agent_num)
            ]
            for i, (do_sample, agent) in enumerate(zip(self.do_samples, self.agents)):
                if do_sample:
                    reward = agent.reward_manager.result(observation[str(i)]["frame_state"])
                    observation[str(i)]["reward"] = reward
                    reward_sum_list[i] += reward["reward_sum"]
                    self._accumulate_reward_items(reward_item_sum_list[i], reward)
                    self._accumulate_reward_debug_items(reward_debug_sum_list[i], reward)
                    self._accumulate_skill_usage_items(action_debug_sum_list[i], observation[str(i)]["frame_state"], agent)

            while True:
                actions = [list(NONE_ACTION) for _ in range(self.agent_num)]

                for index, (do_predict, do_sample, agent) in enumerate(
                    zip(self.do_predicts, self.do_samples, self.agents)
                ):
                    if do_predict:
                        raw_action = agent.predict(observation[str(index)]) if not is_eval else agent.exploit(observation[str(index)])
                        actions[index] = _normalize_env_action(raw_action)
                        self._accumulate_action_debug_items(action_debug_sum_list[index], actions[index])

                        if not is_eval and do_sample:
                            frame = build_frame(agent, observation[str(index)])
                            frame_collector.save_frame(frame, agent_id=index)

                actions = [_normalize_env_action(action) for action in actions]
                env_reward, env_obs = self.env.step(actions)
                if handle_disaster_recovery(env_obs, self.logger):
                    break

                frame_no = env_obs["frame_no"]
                if frame_no % 100 == 0:
                    self._debug_observation(env_obs, prefix=f"_step_f{frame_no}")
                observation = env_obs["observation"]
                terminated = env_obs["terminated"]
                truncated = env_obs["truncated"]

                for i, (do_sample, agent) in enumerate(zip(self.do_samples, self.agents)):
                    if do_sample:
                        reward = agent.reward_manager.result(observation[str(i)]["frame_state"])
                        observation[str(i)]["reward"] = reward
                        reward_sum_list[i] += reward["reward_sum"]
                        self._accumulate_reward_items(reward_item_sum_list[i], reward)
                        self._accumulate_reward_debug_items(reward_debug_sum_list[i], reward)
                        self._accumulate_skill_usage_items(action_debug_sum_list[i], observation[str(i)]["frame_state"], agent)

                is_gameover = terminated or truncated or (is_train_test and frame_no >= 1000)
                if is_gameover:
                    self.logger.info(
                        f"episode_{self.episode_cnt} terminated in fno_{frame_no}, truncated:{truncated}, "
                        f"eval:{is_eval}, reward_sum:{reward_sum_list[monitor_side]}"
                    )
                    for i, (do_sample, agent) in enumerate(zip(self.do_samples, self.agents)):
                        if not is_eval and do_sample:
                            frame_collector.save_last_frame(
                                agent_id=i,
                                reward=observation[str(i)]["reward"]["reward_sum"],
                            )

                    now = time.time()
                    if now - self.last_report_monitor_time >= 60:
                        monitor_data = {"episode_cnt": self.episode_cnt}
                        if self.monitor:
                            if is_eval:
                                monitor_data["reward"] = round(reward_sum_list[monitor_side], 2)
                            for reward_name, reward_value in reward_item_sum_list[monitor_side].items():
                                monitor_data[f"reward_{reward_name}"] = round(reward_value, 4)
                            for debug_name, debug_value in reward_debug_sum_list[monitor_side].items():
                                monitor_data[debug_name] = round(debug_value, 4)
                            for debug_name, debug_value in action_debug_sum_list[monitor_side].items():
                                monitor_data[debug_name] = round(debug_value, 4)
                            selected_skill = self.selected_summoner_skills[monitor_side]
                            for skill_id in GameConfig.SUMMONER_SKILL_MONITOR_IDS:
                                monitor_data[f"selected_summoner_{skill_id}"] = 1.0 if selected_skill == skill_id else 0.0
                            monitor_data["rule_override_count"] = float(
                                getattr(self.agents[monitor_side], "rule_override_count", 0)
                            )
                            self.monitor.put_data({os.getpid(): monitor_data})
                            self.last_report_monitor_time = now

                    if len(frame_collector) > 0 and not is_eval:
                        yield sample_process(frame_collector)
                    break

    def reset_agents(self, observation):
        opponent_agent = self.env_conf_manager.get_opponent_agent()
        monitor_side = self.env_conf_manager.get_monitor_side()
        is_train_test = os.environ.get("is_train_test", "False").lower() == "true"

        self.do_predicts = [True, True]
        self.do_samples = [True, True]

        for i, agent in enumerate(self.agents):
            if i == monitor_side:
                agent.load_model(id="latest")
            else:
                if opponent_agent == "common_ai":
                    self.do_predicts[i] = False
                    self.do_samples[i] = False
                elif opponent_agent == "selfplay":
                    agent.load_model(id="latest")
                else:
                    eval_candidate_model = get_valid_model_pool(self.logger)
                    if int(opponent_agent) not in eval_candidate_model:
                        raise ValueError(f"opponent_agent model_id {opponent_agent} not in {eval_candidate_model}")
                    if is_train_test:
                        self.logger.info("Run train_test, cannot get opponent agent, so replace with latest model")
                        agent.load_model(id="latest")
                    else:
                        agent.load_opponent_agent(id=opponent_agent)
                    self.do_samples[i] = False
            agent.reset(observation[str(i)])

    def _debug_observation(self, env_obs, prefix=""):
        if os.environ.get("OBS_DEBUG", "False").lower() != "true":
            return
        try:
            obs = env_obs.get("observation", {})
            if not obs:
                self.logger.info(f"[OBS_DEBUG]{prefix} observation is empty")
                return

            for agent_id in ["0", "1"]:
                agent_obs = obs.get(agent_id)
                if not agent_obs:
                    continue

                frame_state = agent_obs.get("frame_state", {})
                frame_no = frame_state.get("frame_no", frame_state.get("frameNo", 0))

                heroes = frame_state.get("hero_states", [])
                npcs = frame_state.get("npc_states", [])
                bullets = frame_state.get("bullets", [])
                cakes = frame_state.get("cakes", [])

                hero_info_list = []
                for h in heroes:
                    loc = h.get("location", {})
                    hero_info_list.append({
                        "config_id": h.get("config_id", 0),
                        "runtime_id": h.get("runtime_id", 0),
                        "camp": h.get("camp", 0),
                        "hp": h.get("hp", 0),
                        "max_hp": h.get("max_hp", 0),
                        "ep": h.get("ep", 0),
                        "max_ep": h.get("max_ep", 0),
                        "x": loc.get("x", 0),
                        "z": loc.get("z", 0),
                        "level": h.get("level", 0),
                        "attack_range": h.get("attack_range", 0),
                        "sight_area": h.get("sight_area", 0),
                        "mov_spd": h.get("mov_spd", 0),
                        "atk_spd": h.get("atk_spd", 0),
                        "money": h.get("money", 0),
                        "money_cnt": h.get("money_cnt", 0),
                        "kill_cnt": h.get("kill_cnt", 0),
                        "dead_cnt": h.get("dead_cnt", 0),
                        "behav_mode": h.get("behav_mode", ""),
                        "is_in_grass": h.get("is_in_grass", False),
                    })

                npc_info_list = []
                for n in npcs:
                    loc = n.get("location", {})
                    npc_info_list.append({
                        "config_id": n.get("config_id", 0),
                        "runtime_id": n.get("runtime_id", 0),
                        "actor_type": n.get("actor_type", 0),
                        "camp": n.get("camp", 0),
                        "sub_type": n.get("sub_type", 0),
                        "behav_mode": n.get("behav_mode", 0),
                        "hp": n.get("hp", 0),
                        "max_hp": n.get("max_hp", 0),
                        "x": loc.get("x", 0),
                        "z": loc.get("z", 0),
                        "attack_range": n.get("attack_range", 0),
                        "attack_target": n.get("attack_target", 0),
                        "sight_area": n.get("sight_area", 0),
                        "mov_spd": n.get("mov_spd", 0),
                        "atk_spd": n.get("atk_spd", 0),
                        "phy_atk": n.get("phy_atk", 0),
                        "phy_def": n.get("phy_def", 0),
                    })

                bullet_count = len(bullets)
                cake_count = len(cakes)

                max_coord = 0
                if heroes:
                    for h in heroes:
                        loc = h.get("location", {})
                        max_coord = max(max_coord, abs(loc.get("x", 0)), abs(loc.get("z", 0)))

                hero_dist = 0
                if len(heroes) >= 2:
                    h0_loc = heroes[0].get("location", {})
                    h1_loc = heroes[1].get("location", {})
                    dx = h0_loc.get("x", 0) - h1_loc.get("x", 0)
                    dz = h0_loc.get("z", 0) - h1_loc.get("z", 0)
                    hero_dist = (dx*dx + dz*dz) ** 0.5

                self.logger.info(
                    f"[OBS_DEBUG]{prefix} agent={agent_id} frame={frame_no} "
                    f"heroes={len(heroes)} npcs={len(npcs)} bullets={bullet_count} cakes={cake_count} "
                    f"max_coord={max_coord:.0f} hero_dist={hero_dist:.0f}"
                )

                for idx, h_info in enumerate(hero_info_list):
                    self.logger.info(
                        f"[OBS_DEBUG_HERO]{prefix} agent={agent_id} idx={idx} "
                        f"config_id={h_info['config_id']} camp={h_info['camp']} "
                        f"pos=({h_info['x']:.0f},{h_info['z']:.0f}) "
                        f"hp={h_info['hp']}/{h_info['max_hp']} "
                        f"ep={h_info['ep']}/{h_info['max_ep']} "
                        f"lvl={h_info['level']} "
                        f"atk_range={h_info['attack_range']} "
                        f"sight={h_info['sight_area']} "
                        f"mov_spd={h_info['mov_spd']} "
                        f"atk_spd={h_info['atk_spd']} "
                        f"money={h_info['money']}({h_info['money_cnt']}) "
                        f"k/d={h_info['kill_cnt']}/{h_info['dead_cnt']} "
                        f"behav={h_info['behav_mode']} grass={h_info['is_in_grass']}"
                    )

                for idx, n_info in enumerate(npc_info_list[:12]):
                    self.logger.info(
                        f"[OBS_DEBUG_NPC]{prefix} agent={agent_id} idx={idx} "
                        f"config_id={n_info['config_id']} camp={n_info['camp']} "
                        f"actor_type={n_info['actor_type']} sub_type={n_info['sub_type']} "
                        f"pos=({n_info['x']:.0f},{n_info['z']:.0f}) "
                        f"hp={n_info['hp']}/{n_info['max_hp']} "
                        f"atk_range={n_info['attack_range']} sight={n_info['sight_area']} "
                        f"atk_target={n_info['attack_target']} "
                        f"mov_spd={n_info['mov_spd']} atk_spd={n_info['atk_spd']} "
                        f"phy_atk={n_info['phy_atk']} phy_def={n_info['phy_def']} "
                        f"behav={n_info['behav_mode']}"
                    )

        except Exception as e:
            self.logger.warning(f"[OBS_DEBUG]{prefix} error: {e}")

    def _accumulate_reward_items(self, target, reward):
        for reward_name in GameConfig.REWARD_WEIGHT_DICT:
            target[reward_name] += float(reward.get(f"{reward_name}_weight", 0.0))

    def _accumulate_reward_debug_items(self, target, reward):
        for debug_name in GameConfig.REWARD_DEBUG_KEY_LIST:
            target[debug_name] += float(reward.get(debug_name, 0.0))

    def _accumulate_action_debug_items(self, target, action):
        if not isinstance(action, list) or not action:
            return
        button = int(action[0])
        if len(action) >= len(Config.LABEL_SIZE_LIST):
            target_index = int(action[len(Config.LABEL_SIZE_LIST) - 1])
            if 0 <= target_index < len(Config.TARGET_ORDER):
                target_name = Config.TARGET_ORDER[target_index]
                if target_name == "none":
                    target["target_none_count"] += 1.0
                elif target_name == "enemy_hero":
                    target["target_enemy_hero_count"] += 1.0
                    if button == 3:
                        target["attack_target_enemy_hero_count"] += 1.0
                    elif button in (4, 5, 6, 10):
                        target["skill_target_enemy_hero_count"] += 1.0
                elif target_name == "self_hero":
                    target["target_self_hero_count"] += 1.0
                elif target_name.startswith("enemy_soldier"):
                    target["target_enemy_soldier_count"] += 1.0
                    if button == 3:
                        target["attack_target_enemy_soldier_count"] += 1.0
                elif target_name == "enemy_tower":
                    target["target_enemy_tower_count"] += 1.0
                elif target_name == "monster":
                    target["target_monster_count"] += 1.0
        if button in (0, 1):
            target["action_noop_count"] += 1.0
        elif button == 2:
            target["action_move_count"] += 1.0
        elif button == 3:
            target["action_attack_count"] += 1.0
        elif button in (4, 5, 6, 10):
            target["action_skill_count"] += 1.0
        elif button == 7:
            target["action_recover_count"] += 1.0
        elif button == 8:
            target["action_summoner_count"] += 1.0
        elif button == 9:
            target["action_recall_count"] += 1.0
        elif button == 11:
            target["action_equipment_count"] += 1.0

    def _accumulate_skill_usage_items(self, target, frame_state, agent):
        main_hero = self._find_agent_hero(frame_state, agent)
        if not main_hero:
            return
        slot_states = self._get(self._get(main_hero, "skill_state", {}) or {}, "slot_states", []) or []
        for slot in slot_states:
            used_times = float(self._get(slot, "succUsedInFrame", self._get(slot, "succ_used_in_frame", 0)) or 0)
            if used_times <= 0:
                continue
            slot_key = self._slot_type_key(self._get(slot, "slot_type", None))
            config_id = int(self._get(slot, "configId", self._get(slot, "config_id", 0)) or 0)
            if slot_key == "SLOT_SKILL_1":
                target["skill_1_used_count"] += used_times
            elif slot_key == "SLOT_SKILL_2":
                target["skill_2_used_count"] += used_times
            elif slot_key == "SLOT_SKILL_3":
                target["skill_3_used_count"] += used_times
            elif slot_key == "SLOT_SKILL_4":
                target["recover_used_count"] += used_times
            elif slot_key == "SLOT_SKILL_5" or config_id in GameConfig.SUMMONER_SKILL_IDS:
                target["summoner_skill_used_count"] += used_times

    def _find_agent_hero(self, frame_state, agent):
        player_id = getattr(agent, "player_id", None)
        hero_camp = getattr(agent, "hero_camp", None)
        for hero in frame_state.get("hero_states", []):
            if self._get(hero, "runtime_id", None) == player_id or self._get(hero, "player_id", None) == player_id:
                return hero
        for hero in frame_state.get("hero_states", []):
            if self._camp_key(self._get(hero, "camp", None)) == self._camp_key(hero_camp):
                return hero
        return None

    def _first_selected_skill(self, select_skills, hero_ids):
        if not isinstance(select_skills, dict):
            return int(select_skills) if select_skills is not None else None
        for hero_id in hero_ids:
            skill_id = select_skills.get(hero_id, select_skills.get(str(hero_id)))
            if skill_id is not None:
                return int(skill_id)
        return None

    def _slot_type_key(self, slot_type):
        if isinstance(slot_type, int):
            return f"SLOT_SKILL_{slot_type}"
        return str(slot_type)

    def _camp_key(self, camp):
        if camp in (1, "1", "PLAYERCAMP_1", "blue_camp"):
            return 1
        if camp in (2, "2", "PLAYERCAMP_2", "red_camp"):
            return 2
        return camp

    def _get(self, obj, key, default=0):
        if isinstance(key, (list, tuple)):
            for item in key:
                value = self._get(obj, item, None)
                if value is not None:
                    return value
            return default
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

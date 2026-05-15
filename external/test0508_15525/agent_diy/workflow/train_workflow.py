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

from agent_diy.conf.conf import GameConfig
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

    def _call_init_config(self, usr_conf):
        blue_hero_ids, red_hero_ids = EnvConfManager.extract_hero_ids_from_usr_conf(usr_conf)
        camp_keys = ["blue_camp", "red_camp"]

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
            }
            select_skills = agent.init_config(config_data)
            EnvConfManager.inject_select_skills(usr_conf, camp_key, select_skills)
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
            self._call_init_config(usr_conf)

            env_obs = self.env.reset(usr_conf=usr_conf)
            if handle_disaster_recovery(env_obs, self.logger):
                break

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
            for i, (do_sample, agent) in enumerate(zip(self.do_samples, self.agents)):
                if do_sample:
                    reward = agent.reward_manager.result(observation[str(i)]["frame_state"])
                    observation[str(i)]["reward"] = reward
                    reward_sum_list[i] += reward["reward_sum"]
                    self._accumulate_reward_items(reward_item_sum_list[i], reward)

            while True:
                actions = [list(NONE_ACTION) for _ in range(self.agent_num)]

                for index, (do_predict, do_sample, agent) in enumerate(
                    zip(self.do_predicts, self.do_samples, self.agents)
                ):
                    if do_predict:
                        raw_action = agent.predict(observation[str(index)]) if not is_eval else agent.exploit(observation[str(index)])
                        actions[index] = _normalize_env_action(raw_action)

                        if not is_eval and do_sample:
                            frame = build_frame(agent, observation[str(index)])
                            frame_collector.save_frame(frame, agent_id=index)

                actions = [_normalize_env_action(action) for action in actions]
                env_reward, env_obs = self.env.step(actions)
                if handle_disaster_recovery(env_obs, self.logger):
                    break

                frame_no = env_obs["frame_no"]
                observation = env_obs["observation"]
                terminated = env_obs["terminated"]
                truncated = env_obs["truncated"]

                for i, (do_sample, agent) in enumerate(zip(self.do_samples, self.agents)):
                    if do_sample:
                        reward = agent.reward_manager.result(observation[str(i)]["frame_state"])
                        observation[str(i)]["reward"] = reward
                        reward_sum_list[i] += reward["reward_sum"]
                        self._accumulate_reward_items(reward_item_sum_list[i], reward)

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

    def _accumulate_reward_items(self, target, reward):
        for reward_name in GameConfig.REWARD_WEIGHT_DICT:
            target[reward_name] += float(reward.get(f"{reward_name}_weight", 0.0))

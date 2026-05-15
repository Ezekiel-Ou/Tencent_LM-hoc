#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Agent DIY configuration for the hok1v1 112/133 task.

This version intentionally follows the high-dimensional hok_semi observation
layout while remapping hero-specific assumptions to Luban No.7 and Di Renjie.
"""

import hashlib
import json


class GameConfig:
    HERO_IDS = [112, 133]
    DEFAULT_SUMMONER_SKILL = 80115
    SUMMONER_SKILL_IDS = [80102, 80103, 80104, 80105, 80107, 80108, 80109, 80110, 80115, 80121]

    REWARD_WEIGHT_DICT = {
        "hp_point": 2.0,
        "tower_hp_point": 5.0,
        "money": 0.004,
        "exp": 0.004,
        "ep_rate": 0.75,
        "death": -1.0,
        "kill": 0.6,
        "last_hit": 0.5,
        "forward": 0.03,
    }
    REMOVE_FORWARD_AFTER = 3000
    TIME_SCALE_ARG = 8000
    REWARD_WITHOUT_TIME_SCALE = set()
    MODEL_SAVE_INTERVAL = 1800


class Args:
    RELATIVE_DISTANCE_UNIT_SIZE = 600
    RELATIVE_DISTANCE_MAX_SIZE = 24600
    DIM_RELATIVE_DISTANCE = (RELATIVE_DISTANCE_MAX_SIZE // RELATIVE_DISTANCE_UNIT_SIZE + 2) * 2 + 1
    WHOLE_DISTANCE_UNIT_SIZE = 5000
    WHOLE_DISTANCE_MAX_SIZE = int(9e4)
    DIM_WHOLE_DISTANCE = (WHOLE_DISTANCE_MAX_SIZE // WHOLE_DISTANCE_UNIT_SIZE) * 2 + 2
    DIM_DISTANCE = DIM_RELATIVE_DISTANCE + DIM_WHOLE_DISTANCE

    HP_UNIT_SIZE = 100
    HP_MAX_SIZE = 2400
    MARK_ID_LAYERS = {
        11200: 5,
        13300: 5,
        13310: 1,
    }
    DIM_MARK = sum([v + 1 for v in MARK_ID_LAYERS.values()]) + 1
    DIM_UNIT = int(DIM_DISTANCE + HP_MAX_SIZE // HP_UNIT_SIZE + 3 + DIM_MARK)

    HERO_CONFIG_ID = GameConfig.HERO_IDS
    HERO_BEHAVE = [
        "State_Dead",
        "State_Idle",
        "Direction_Move",
        "Normal_Attack",
        "State_Revive",
        "UseSkill_1",
        "UseSkill_2",
        "UseSkill_3",
    ]
    EP_UNIT_SIZE = 30
    EP_MAX_SIZE = 240
    CD_UNIT_SIZE = 1
    CD_MAX_SIZE = 10
    LEVEL_MAX = 15
    MONEY_UNIT_SIZE = 20
    MONEY_MAX_SIZE = 300

    COMMON_BUFFS = [
        90015,
        10000,
        10010,
        11001,
        11002,
        11010,
        11111,
        911220,
        911290,
        914110,
        914210,
        914211,
        914250,
    ]
    LUBAN_BUFFS = [
        112000,
        112010,
        112020,
        112030,
        112040,
        112100,
        112110,
        112120,
        112130,
        112200,
        112210,
        112220,
        112230,
        112300,
        112310,
        112320,
        112900,
        112910,
        112920,
        112930,
    ]
    DIRENJIE_BUFFS = [
        133000,
        133010,
        133020,
        133030,
        133040,
        133100,
        133110,
        133120,
        133130,
        133200,
        133210,
        133220,
        133230,
        133300,
        133310,
        133320,
        133900,
        133910,
        133920,
        133930,
    ]
    BUFFS = COMMON_BUFFS + LUBAN_BUFFS + DIRENJIE_BUFFS
    DIM_BUFF = len(BUFFS) + 1

    DIM_HERO = (
        DIM_UNIT
        + 1
        + len(HERO_BEHAVE)
        + 1
        + EP_MAX_SIZE // EP_UNIT_SIZE
        + 2
        + (CD_MAX_SIZE // CD_UNIT_SIZE + 4) * 5
        + LEVEL_MAX
        + MONEY_MAX_SIZE // MONEY_UNIT_SIZE
        + 3
        + 1
        + 2
        + DIM_BUFF
    )

    SOLDIER_MAX_NUM = 4
    SOLDIER_BEHAVE = ["State_Dead", "Attack_Path"]
    SOLDIER_CONFIG_ID = [[6801, 6804], [6800, 6803], [6802, 6805]]
    DIM_SOLDIER = DIM_UNIT + len(SOLDIER_BEHAVE) + 1 + len(SOLDIER_CONFIG_ID) + 2
    DIM_SOLDIERS = DIM_SOLDIER * SOLDIER_MAX_NUM

    RIVER_CRAB_BEHAVE = ["State_Dead", "State_Auto", "State_Revive", "State_Born"]
    DIM_RIVER_CRAB = DIM_UNIT + len(RIVER_CRAB_BEHAVE) + 1

    DIM_ORGAN = DIM_UNIT + 3 + 2
    DIM_ALL_UNITS = DIM_HERO * 2 + DIM_SOLDIERS * 2 + DIM_RIVER_CRAB + DIM_ORGAN * 2

    BULLET_MAX_NUM = 10
    BULLET_SLOT = ["SLOT_SKILL_0", "SLOT_SKILL_1", "SLOT_SKILL_2", "SLOT_SKILL_3", "SLOT_SKILL_VALID"]
    DIM_BULLET = len(BULLET_SLOT) + DIM_DISTANCE
    DIM_BULLETS = DIM_BULLET * BULLET_MAX_NUM

    DIM_ALL = DIM_ALL_UNITS + DIM_BULLETS


class DimConfig:
    DIM_OF_FEATURE = [Args.DIM_ALL]
    DIM_OF_HERO_FRD = [Args.DIM_HERO]
    DIM_OF_HERO_EMY = [Args.DIM_HERO]
    DIM_OF_SOLDIER_1_4 = [Args.DIM_SOLDIER] * Args.SOLDIER_MAX_NUM
    DIM_OF_SOLDIER_5_8 = [Args.DIM_SOLDIER] * Args.SOLDIER_MAX_NUM
    DIM_OF_RIVER_CRAB = [Args.DIM_RIVER_CRAB]
    DIM_OF_ORGAN_1 = [Args.DIM_ORGAN]
    DIM_OF_ORGAN_2 = [Args.DIM_ORGAN]
    DIM_OF_BULLET_1_9 = [Args.DIM_BULLET] * (Args.BULLET_MAX_NUM - 1)
    DIM_OF_BULLET_10 = [Args.DIM_BULLET]


class Config:
    NETWORK_NAME = "network"
    CHECKPOINT_PROTOCOL_VERSION = "agent_diy_hok1v1_hoksemi_dim_v1"
    TARGET_ORDER = [
        "none",
        "enemy_hero",
        "self_hero",
        "enemy_soldier_1",
        "enemy_soldier_2",
        "enemy_soldier_3",
        "enemy_soldier_4",
        "enemy_tower",
        "monster",
    ]
    LSTM_DROPOUT = 0
    LSTM_TIME_STEPS = 16
    LSTM_UNIT_SIZE = 512
    DIM_PUBLIC = 512
    FEATURE_DIM = Args.DIM_ALL

    LABEL_SIZE_LIST = [12, 16, 16, 16, 16, 9]
    # Button ids follow the official 12-way first action head:
    # 0/1=no-op, 2=move, 3=normal attack, 4-8=skills/summoner, 9=recall.
    # This fixed prior helps cold-start common_ai warmup leave the base without
    # changing the action protocol, feature shape, or checkpoint tensor shapes.
    BUTTON_LOGIT_BIAS = [-2.0, -1.0, 1.2, 0.5, 0.2, 0.2, 0.2, 0.0, 0.0, -1.5, -0.5, -0.2]
    LEGAL_ACTION_SIZE_LIST = LABEL_SIZE_LIST.copy()
    LEGAL_ACTION_SIZE_LIST[-1] = LEGAL_ACTION_SIZE_LIST[-1] * LEGAL_ACTION_SIZE_LIST[0]
    LEGAL_ACTION_DIM = sum(LABEL_SIZE_LIST)
    RAW_LEGAL_ACTION_DIM = sum(LEGAL_ACTION_SIZE_LIST)

    DATA_SPLIT_SHAPE = [
        FEATURE_DIM + LEGAL_ACTION_DIM,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        12,
        16,
        16,
        16,
        16,
        9,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        LSTM_UNIT_SIZE,
        LSTM_UNIT_SIZE,
    ]
    SERI_VEC_SPLIT_SHAPE = [(FEATURE_DIM,), (LEGAL_ACTION_DIM,)]

    INIT_LEARNING_RATE_START = 1e-4
    TARGET_LR = 1e-5
    TARGET_STEP = 5000
    BETA_START = 0.01
    LOG_EPSILON = 1e-6
    MIN_POLICY = 1e-5
    CLIP_PARAM = 0.2
    DUAL_CLIP_PARAM = 3.0
    VALUE_CLIP_PARAM = 0.2
    USE_ADVANTAGE_NORM = True
    USE_VALUE_CLIP = False
    TARGET_EMBED_DIM = 32

    IS_REINFORCE_TASK_LIST = [True, True, True, True, True, True]

    data_shapes = []

    GAMMA = 0.995
    LAMDA = 0.95

    USE_GRAD_CLIP = True
    GRAD_CLIP_RANGE = 0.5

    SAMPLE_DIM = sum(DATA_SPLIT_SHAPE[:-2]) * LSTM_TIME_STEPS + sum(DATA_SPLIT_SHAPE[-2:])

    @classmethod
    def validate(cls):
        assert cls.FEATURE_DIM == Args.DIM_ALL == 3910
        assert Args.DIM_HERO == 347
        assert Args.DIM_SOLDIER == 175
        assert Args.DIM_RIVER_CRAB == 172
        assert Args.DIM_ORGAN == 172
        assert Args.DIM_BULLET == 130
        assert cls.LABEL_SIZE_LIST == [12, 16, 16, 16, 16, 9]
        assert cls.LEGAL_ACTION_DIM == 85
        assert cls.RAW_LEGAL_ACTION_DIM == 184
        assert cls.DATA_SPLIT_SHAPE[0] == cls.FEATURE_DIM + cls.LEGAL_ACTION_DIM
        assert cls.SAMPLE_DIM == sum(shape[0] for shape in cls.data_shapes)
        assert len(cls.TARGET_ORDER) == cls.LABEL_SIZE_LIST[-1]
        assert len(cls.BUTTON_LOGIT_BIAS) == cls.LABEL_SIZE_LIST[0]

    @classmethod
    def checkpoint_signature(cls):
        return {
            "checkpoint_protocol_version": cls.CHECKPOINT_PROTOCOL_VERSION,
            "network_name": cls.NETWORK_NAME,
            "hero_ids": list(GameConfig.HERO_IDS),
            "feature_dim": cls.FEATURE_DIM,
            "label_size_list": list(cls.LABEL_SIZE_LIST),
            "legal_action_size_list": list(cls.LEGAL_ACTION_SIZE_LIST),
            "legal_action_dim": cls.LEGAL_ACTION_DIM,
            "raw_legal_action_dim": cls.RAW_LEGAL_ACTION_DIM,
            "data_split_shape": list(cls.DATA_SPLIT_SHAPE),
            "seri_vec_split_shape": [list(shape) for shape in cls.SERI_VEC_SPLIT_SHAPE],
            "lstm_time_steps": cls.LSTM_TIME_STEPS,
            "lstm_unit_size": cls.LSTM_UNIT_SIZE,
            "dim_public": cls.DIM_PUBLIC,
            "sample_dim": cls.SAMPLE_DIM,
            "target_order": list(cls.TARGET_ORDER),
            "model_class": "agent_diy.model.model.Model",
            "obs_layout": {
                "dim_hero": Args.DIM_HERO,
                "dim_soldier": Args.DIM_SOLDIER,
                "dim_river_crab": Args.DIM_RIVER_CRAB,
                "dim_organ": Args.DIM_ORGAN,
                "dim_bullet": Args.DIM_BULLET,
            },
        }

    @classmethod
    def checkpoint_signature_hash(cls):
        payload = json.dumps(cls.checkpoint_signature(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


Config.data_shapes = [[shape * Config.LSTM_TIME_STEPS] for shape in Config.DATA_SPLIT_SHAPE[:-2]] + [
    [Config.LSTM_UNIT_SIZE],
    [Config.LSTM_UNIT_SIZE],
]
Config.SAMPLE_DIM = sum(Config.DATA_SPLIT_SHAPE[:-2]) * Config.LSTM_TIME_STEPS + sum(Config.DATA_SPLIT_SHAPE[-2:])
Config.validate()

#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (C) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Frozen configuration for the agent_diy hok1v1 feature schema.

The feature layout is documented in docs/feature_engineering_design.md and is
kept in lockstep with the PPO sample serialization and model input split.
"""

import hashlib
import json


def _unique(values):
    seen = set()
    result = []
    for value in values:
        value = int(value)
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


class GameConfig:
    HERO_IDS = [112, 133]
    DEFAULT_SUMMONER_SKILL = 80110
    SUMMONER_SKILL_IDS = [80102, 80103, 80104, 80105, 80107, 80108, 80109, 80110, 80115, 80121]
    SUMMONER_SKILL_TRAIN_MODE = "cycle"
    SUMMONER_SKILL_CANDIDATES_BY_HERO = {
        112: [80110, 80115, 80102],
        133: [80110, 80115, 80102],
    }
    SUMMONER_SKILL_MONITOR_IDS = SUMMONER_SKILL_IDS.copy()
    LEVEL_MAX_EXP = {
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

    REWARD_WEIGHT_DICT = {
        "hp_point": 2.0,
        "tower_hp_point": 6.0,
        "money": 0.004,
        "exp": 0.004,
        "ep_rate": 0.25,
        "death": -1.0,
        "kill": 0.6,
        "last_hit": 0.5,
        "forward": 0.2,
        # Di Renjie (133) skill 2 dispels enemy 133 ultimate. Small weight, only
        # active in 133v133 mirror matchups.
        "cleanse_success": 0.15,
        # Skill (slot 1/2/3) hits on enemy hero. Slot 3 capped at 1 credit per
        # cast cycle to avoid Luban ultimate bullet-farm bias.
        "skill_hit_enemy_hero": 0.15,
        # Eating a health cake while not at full HP (teaches retreat-and-heal).
        "cake_pickup": 0.8,
        # Using slot 4 recover skill while low HP (teaches active healing).
        "recover_skill_low_hp": 0.3,
        # Last-hit bonus when farming under own tower (safe farming signal).
        "safe_last_hit": 0.15,
    }
    REMOVE_FORWARD_AFTER = 1000
    REWARD_DEBUG_KEY_LIST = [
        "last_hit_dead_action_count",
        "last_hit_soldier_dead_count",
        "last_hit_main_count",
        "last_hit_enemy_count",
    ]
    ACTION_DEBUG_KEY_LIST = [
        "action_noop_count",
        "action_move_count",
        "action_attack_count",
        "action_skill_count",
        "action_recover_count",
        "action_summoner_count",
        "action_recall_count",
        "action_equipment_count",
        "target_none_count",
        "target_enemy_hero_count",
        "target_self_hero_count",
        "target_enemy_soldier_count",
        "target_enemy_tower_count",
        "target_monster_count",
        "attack_target_enemy_hero_count",
        "attack_target_enemy_soldier_count",
        "skill_target_enemy_hero_count",
        "skill_1_used_count",
        "skill_2_used_count",
        "skill_3_used_count",
        "recover_used_count",
        "summoner_skill_used_count",
    ]
    TIME_SCALE_ARG = 8000
    REWARD_WITHOUT_TIME_SCALE = set()
    MODEL_SAVE_INTERVAL = 1800


class Args:
    # Remapped map geometry.
    SQRT2 = 2 ** 0.5
    RAW_COORD_ABS_LIMIT = 60000
    GLOBAL_LANE_HALF = 45000
    GLOBAL_WIDTH_HALF = 7000
    CENTER_LANE_HALF = 15000
    CENTER_WIDTH_HALF = 10000
    CENTER_LANE_UNIT = 500
    CENTER_WIDTH_UNIT = 250
    GLOBAL_LANE_UNIT = 5000
    GLOBAL_WIDTH_UNIT = 3500

    DIM_CENTER_LANE = 63
    DIM_CENTER_WIDTH = 83
    DIM_GLOBAL_LANE = 21
    DIM_GLOBAL_WIDTH = 7
    DIM_POSITION_CONT = 6
    DIM_POSITION = 180

    DIM_UNIT_CORE = 206
    DIM_GLOBAL = 60
    DIM_HERO = 693
    DIM_HERO_BUFF = 136
    DIM_SKILL_SLOT = 40
    HERO_SKILL_SLOT_NUM = 6
    DIM_SOLDIER = 224
    SOLDIER_PER_SIDE = 3
    SOLDIER_SLOTS = 6
    DIM_MONSTER = 222
    DIM_TOWER = 248
    TOWER_SLOTS = 2
    DIM_BULLET = 211
    BULLET_HERO_THREAT_SLOTS = 4
    BULLET_TOWER_SLOTS = 1
    BULLET_SLOTS = 5
    DIM_TARGET = 30
    TARGET_SLOTS = 9

    DIM_ALL = (
        DIM_GLOBAL
        + DIM_HERO * 2
        + DIM_SOLDIER * SOLDIER_SLOTS
        + DIM_MONSTER
        + DIM_TOWER * TOWER_SLOTS
        + DIM_BULLET * BULLET_SLOTS
        + DIM_TARGET * TARGET_SLOTS
    )

    HERO_CONFIG_ID = GameConfig.HERO_IDS
    HERO_BEHAVE = [0, 1, 2, 4, 9, 10, 23, 27]
    LEVEL_MAX = 15
    EP_MAX_SIZE = 1500
    HERO_HP_SCALE = 12000
    MONEY_TOTAL_SCALE = 15000
    MONEY_FRAME_SCALE = 2000
    MONEY_DELTA_SCALE = 500
    TOWER_ATTACK_RANGE_FALLBACK = 8800
    CAKE_RESPAWN_SECONDS = 75
    FRAME_MAX_FALLBACK = 20000

    SELF_BASE_ANCHOR = (-40000.0, 0.0)
    ENEMY_BASE_ANCHOR = (40000.0, 0.0)
    SELF_TOWER_ANCHOR = (-13000.0, 0.0)
    ENEMY_TOWER_ANCHOR = (13000.0, 0.0)
    SELF_CAKE_ANCHOR = (-15000.0, 0.0)
    ENEMY_CAKE_ANCHOR = (15000.0, 0.0)

    SUMMONER_CANDIDATES = GameConfig.SUMMONER_SKILL_IDS
    HERO_SKILL_SLOT_ORDER = [0, 1, 2, 3, 4, 5]
    SOLDIER_CONFIG_GROUPS = {
        "melee": {6801, 6804},
        "ranged": {6800, 6803},
        "cannon": {6802, 6805},
    }

    BUFF_SOURCE_IDS = _unique(
        [
            10000,
            10010,
            10014,
            11001,
            11002,
            11010,
            50000,
            90015,
            90019,
            90110,
            500009,
            911260,
            911290,
            912330,
            912350,
            914110,
            914210,
            914211,
            914230,
            914232,
            919900,
            167602,
            131956,
            112000,
            112001,
            112010,
            112015,
            112020,
            112025,
            112030,
            112035,
            112040,
            112041,
            112042,
            112043,
            112044,
            112045,
            112046,
            112047,
            112048,
            112100,
            112200,
            112201,
            112210,
            112300,
            112301,
            112320,
            112890,
            112910,
            112990,
            112991,
            133000,
            133001,
            133010,
            133011,
            133020,
            133090,
            133100,
            133200,
            133250,
            133260,
            133300,
            133310,
            133320,
            133330,
            133350,
            133390,
            133950,
            133951,
            # Historical hok_semi-style candidates retained only if room remains.
            11111,
            911220,
            914250,
            112110,
            112120,
            112130,
            112220,
            112230,
            112310,
            112920,
            112930,
            133030,
            133040,
            133110,
            133120,
            133130,
            133210,
            133220,
            133230,
            133900,
            133910,
            133920,
            133930,
        ]
    )
    BUFF_WHITELIST_96 = (BUFF_SOURCE_IDS + list(range(990000, 990000 + 96)))[:96]


class DimConfig:
    DIM_OF_FEATURE = [Args.DIM_ALL]
    DIM_OF_GLOBAL = [Args.DIM_GLOBAL]
    DIM_OF_HERO_SELF = [Args.DIM_HERO]
    DIM_OF_HERO_ENEMY = [Args.DIM_HERO]
    DIM_OF_SOLDIER_SELF = [Args.DIM_SOLDIER] * Args.SOLDIER_PER_SIDE
    DIM_OF_SOLDIER_ENEMY = [Args.DIM_SOLDIER] * Args.SOLDIER_PER_SIDE
    DIM_OF_MONSTER = [Args.DIM_MONSTER]
    DIM_OF_TOWER_SELF = [Args.DIM_TOWER]
    DIM_OF_TOWER_ENEMY = [Args.DIM_TOWER]
    DIM_OF_BULLET_HERO = [Args.DIM_BULLET] * Args.BULLET_HERO_THREAT_SLOTS
    DIM_OF_BULLET_TOWER = [Args.DIM_BULLET] * Args.BULLET_TOWER_SLOTS
    DIM_OF_TARGET = [Args.DIM_TARGET] * Args.TARGET_SLOTS


class Config:
    NETWORK_NAME = "network"
    CHECKPOINT_PROTOCOL_VERSION = "agent_diy_hok1v1_feature_4833_v1"
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

    INIT_LEARNING_RATE_START = 3e-5
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
        assert Args.DIM_POSITION == 180
        assert Args.DIM_UNIT_CORE == 206
        assert Args.DIM_GLOBAL == 60
        assert Args.DIM_HERO == 693
        assert Args.DIM_SOLDIER == 224
        assert Args.SOLDIER_SLOTS == 6
        assert Args.DIM_MONSTER == 222
        assert Args.DIM_TOWER == 248
        assert Args.DIM_BULLET == 211
        assert Args.BULLET_SLOTS == 5
        assert Args.DIM_TARGET == 30
        assert cls.FEATURE_DIM == Args.DIM_ALL == 4833
        assert len(Args.BUFF_WHITELIST_96) == 96
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
                "dim_global": Args.DIM_GLOBAL,
                "dim_hero": Args.DIM_HERO,
                "dim_soldier": Args.DIM_SOLDIER,
                "dim_monster": Args.DIM_MONSTER,
                "dim_tower": Args.DIM_TOWER,
                "dim_bullet": Args.DIM_BULLET,
                "dim_target": Args.DIM_TARGET,
                "soldier_slots": Args.SOLDIER_SLOTS,
                "bullet_slots": Args.BULLET_SLOTS,
                "target_slots": Args.TARGET_SLOTS,
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

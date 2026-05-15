#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
hok_semi-style structured observation builder for the current hok1v1 protocol.

Feature layout:
    self hero, enemy hero,
    self soldiers x4, enemy soldiers x4,
    neutral monster placeholder,
    self tower, enemy tower,
    enemy hero bullets x9, enemy tower bullet x1.
"""

import math

import numpy as np

from agent_diy.conf.conf import Args, Config, GameConfig


TOWER_SUB_TYPES = {21, "21", "ACTOR_SUB_TOWER"}
SOLDIER_SUB_TYPES = {1, "1", "ACTOR_SUB_SOLDIER"}
MONSTER_TYPES = {2, "2", "ACTOR_TYPE_MONSTER", "ACTOR_MONSTER"}
RIVER_CRAB_CONFIG_ID = 6827
UNSEEN_PADDING = 100000


def _clip(value, lo, hi):
    return max(lo, min(hi, value))


def _fix(value):
    return math.floor(value) if value > 0 else math.ceil(value)


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


class FeatureProcess:
    def __init__(self, camp, logger=None):
        self.logger = logger
        self.reset(camp)

    def reset(self, camp):
        self.main_camp = camp
        self.transform_camp2_to_camp1 = self._camp_key(camp) == 2
        self.pos = (0.0, 0.0)
        self.id2type = {}
        self.last_money = [None, None]
        self.next_cake_frame = [1778, 1778]

    def process_feature(self, observation):
        frame_state = observation["frame_state"]
        player_id = observation.get("player_id")
        player_camp = observation.get("player_camp", observation.get("camp", self.main_camp))
        if player_camp is not None and self._camp_key(player_camp) != self._camp_key(self.main_camp):
            self.reset(player_camp)

        self.n_frame = frame_state.get("frame_no", frame_state.get("frameNo", 0))
        heroes = list(frame_state.get("hero_states", []))
        npcs = list(frame_state.get("npc_states", []))
        bullets = list(frame_state.get("bullets", []))
        cakes = list(frame_state.get("cakes", []))

        self_hero, enemy_hero = self._split_heroes(heroes, player_id)
        self.pos = self._position(self_hero)

        self_tower, enemy_tower = self._split_towers(npcs)
        self_cake, enemy_cake = self._split_cakes(cakes, self_tower, enemy_tower)
        self_soldiers, enemy_soldiers = self._split_soldiers(npcs)
        neutral_monster = self._select_neutral_monster(npcs)
        self._build_id2type(
            heroes,
            self_soldiers + enemy_soldiers,
            [self_tower, enemy_tower],
            neutral_monster,
        )
        hero_bullets, tower_bullet = self._split_enemy_bullets(bullets, enemy_tower)

        feature = []
        feature.extend(self._process_hero(self_hero, is_enemy=False, opposed_tower=enemy_tower))
        feature.extend(self._process_hero(enemy_hero, is_enemy=True, opposed_tower=self_tower))
        feature.extend(self._process_soldiers(self_soldiers, opposed_tower=enemy_tower))
        feature.extend(self._process_soldiers(enemy_soldiers, opposed_tower=self_tower))
        feature.extend(self._process_river_crab(neutral_monster))
        feature.extend(self._process_tower(self_tower, self_cake, is_enemy=False))
        feature.extend(self._process_tower(enemy_tower, enemy_cake, is_enemy=True))
        feature.extend(self._process_bullets(hero_bullets, tower_bullet))

        if len(feature) != Config.FEATURE_DIM:
            raise ValueError(f"agent_diy feature dim mismatch: {len(feature)} != {Config.FEATURE_DIM}")
        return np.asarray(feature, dtype=np.float32).tolist()

    def _camp_key(self, camp):
        if camp in (1, "1", "PLAYERCAMP_1", "blue_camp"):
            return 1
        if camp in (2, "2", "PLAYERCAMP_2", "red_camp"):
            return 2
        return camp

    def _same_camp(self, camp_a, camp_b):
        return self._camp_key(camp_a) == self._camp_key(camp_b)

    def _loc_xy(self, obj):
        if obj is None:
            return None
        collider = _get(obj, "collider", None)
        if collider:
            loc = _get(collider, "location", {}) or {}
        else:
            loc = _get(obj, "location", {}) or {}
        if not loc:
            return None
        return float(_get(loc, "x", UNSEEN_PADDING)), float(_get(loc, "z", UNSEEN_PADDING))

    def _position(self, obj):
        pos = self._loc_xy(obj)
        if pos is None:
            return (UNSEEN_PADDING, UNSEEN_PADDING)
        x, z = pos
        if self.transform_camp2_to_camp1 and x != UNSEEN_PADDING:
            x, z = -x, -z
        return x, z

    def _runtime_id(self, obj):
        return _get_any(obj or {}, ["runtime_id", "player_id", "id"], None)

    def _build_id2type(self, heroes, soldiers, towers, neutral_monster):
        self.id2type = {}
        for hero in heroes:
            runtime_id = self._runtime_id(hero)
            if runtime_id is not None:
                self.id2type[runtime_id] = "hero"
        for soldier in soldiers:
            runtime_id = self._runtime_id(soldier)
            if runtime_id is not None:
                self.id2type[runtime_id] = "soldier"
        for tower in towers:
            runtime_id = self._runtime_id(tower)
            if runtime_id is not None:
                self.id2type[runtime_id] = "organ"
        runtime_id = self._runtime_id(neutral_monster)
        if runtime_id is not None:
            self.id2type[runtime_id] = "monster"

    def _split_heroes(self, heroes, player_id):
        self_hero, enemy_hero = None, None
        for hero in heroes:
            if player_id is not None and self._runtime_id(hero) == player_id:
                self_hero = hero
            elif self._same_camp(_get(hero, "camp"), self.main_camp):
                self_hero = hero
            else:
                enemy_hero = hero
        if enemy_hero is None:
            for hero in heroes:
                if hero is not self_hero:
                    enemy_hero = hero
                    break
        return self_hero, enemy_hero

    def _split_towers(self, npcs):
        self_tower, enemy_tower = None, None
        for npc in npcs:
            if _get(npc, "sub_type") not in TOWER_SUB_TYPES:
                continue
            if self._same_camp(_get(npc, "camp"), self.main_camp):
                self_tower = npc
            else:
                enemy_tower = npc
        return self_tower, enemy_tower

    def _split_soldiers(self, npcs):
        self_soldiers, enemy_soldiers = [], []
        for npc in npcs:
            if _get(npc, "sub_type") not in SOLDIER_SUB_TYPES:
                continue
            if self._same_camp(_get(npc, "camp"), self.main_camp):
                self_soldiers.append(npc)
            else:
                enemy_soldiers.append(npc)

        def sort_key(unit):
            return (math.dist(self._position(unit), self.pos), self._runtime_id(unit) or 0)

        self_soldiers = sorted(self_soldiers, key=sort_key)[: Args.SOLDIER_MAX_NUM]
        enemy_soldiers = sorted(enemy_soldiers, key=sort_key)[: Args.SOLDIER_MAX_NUM]
        return self_soldiers, enemy_soldiers

    def _select_neutral_monster(self, npcs):
        monsters = [
            npc
            for npc in npcs
            if (
                _get(npc, "config_id") == RIVER_CRAB_CONFIG_ID
                or (_get(npc, "actor_type") in MONSTER_TYPES and _get(npc, "sub_type") not in SOLDIER_SUB_TYPES)
            )
        ]
        if not monsters:
            return None
        return sorted(monsters, key=lambda unit: math.dist(self._position(unit), self.pos))[0]

    def _split_cakes(self, cakes, self_tower, enemy_tower):
        self_cake, enemy_cake = None, None
        tower_candidates = []
        if self_tower is not None:
            tower_candidates.append(("self", self._position(self_tower)))
        if enemy_tower is not None:
            tower_candidates.append(("enemy", self._position(enemy_tower)))
        for cake in cakes:
            if not tower_candidates:
                break
            cake_pos = self._position(cake)
            owner = min(tower_candidates, key=lambda item: math.dist(cake_pos, item[1]))[0]
            if owner == "self" and self_cake is None:
                self_cake = cake
            elif owner == "enemy" and enemy_cake is None:
                enemy_cake = cake
        return self_cake, enemy_cake

    def _split_enemy_bullets(self, bullets, enemy_tower):
        enemy_bullets = [bullet for bullet in bullets if not self._same_camp(_get(bullet, "camp"), self.main_camp)]
        hero_bullets, tower_bullets = [], []
        for bullet in enemy_bullets:
            source_type = self.id2type.get(_get(bullet, "source_actor"))
            if source_type == "organ":
                tower_bullets.append(bullet)
            elif source_type == "hero":
                hero_bullets.append(bullet)
        hero_bullets = sorted(hero_bullets, key=lambda bullet: math.dist(self._position(bullet), self.pos))
        tower_bullets = sorted(tower_bullets, key=lambda bullet: math.dist(self._position(bullet), self.pos))
        return hero_bullets[: Args.BULLET_MAX_NUM - 1], tower_bullets[0] if tower_bullets else None

    def _process_position(self, position):
        if position is None:
            return [0.0] * Args.DIM_DISTANCE
        x, z = position
        if x == UNSEEN_PADDING:
            return [0.0] * Args.DIM_DISTANCE

        unit_size, max_size = Args.RELATIVE_DISTANCE_UNIT_SIZE, Args.RELATIVE_DISTANCE_MAX_SIZE
        max_idx = (max_size / unit_size + 1) / 2
        rpos = [
            int(_clip(_fix((value - base) / unit_size), -max_idx, max_idx) + max_idx)
            for value, base in zip((x, z), self.pos)
        ]
        x_dis = _clip(math.dist((x, z), self.pos) / (max_size / 2), 0.0, 1.0)
        rpos_dim = int(2 * max_idx + 1)
        x_rpos = [0.0] * (rpos_dim * 2 + 1)
        x_rpos[rpos[0]] = 1.0
        x_rpos[rpos[1] + rpos_dim] = 1.0
        x_rpos[-1] = x_dis

        unit_size, max_size = Args.WHOLE_DISTANCE_UNIT_SIZE, Args.WHOLE_DISTANCE_MAX_SIZE
        wpos_dim = int(max_size / unit_size)
        wpos = [
            int(_clip(math.floor((value + max_size / 2) / unit_size), 0, wpos_dim - 1))
            for value in (x, z)
        ]
        x_wratio = [
            _clip((value + max_size / 2) / unit_size - index, -1.0, 1.0)
            for value, index in zip((x, z), wpos)
        ]
        x_wpos = [0.0] * (wpos_dim * 2 + 2)
        x_wpos[wpos[0]] = 1.0
        x_wpos[wpos[1] + wpos_dim] = 1.0
        x_wpos[-2] = x_wratio[0]
        x_wpos[-1] = x_wratio[1]
        return x_rpos + x_wpos

    def _process_unit(self, unit):
        if unit is None:
            return [0.0] * Args.DIM_UNIT
        hp = _get(unit, "hp", 0)
        max_hp = _get(unit, "max_hp", 0)
        hp_dim = int(Args.HP_MAX_SIZE / Args.HP_UNIT_SIZE) + 2
        x_hp = [0.0] * (1 + hp_dim)
        x_hp[0] = _safe_div(hp, max_hp)
        x_hp[1 + min(int(math.ceil(max(hp, 0) / Args.HP_UNIT_SIZE)), hp_dim - 1)] = 1.0

        x_mark = [0.0] * Args.DIM_MARK
        buff_marks = _get(_get(unit, "buff_state", {}) or {}, "buff_marks", []) or []
        now_feature_idx = 0
        used_marks = 0
        for mark_id, max_layer in Args.MARK_ID_LAYERS.items():
            mark_layer = None
            for mark in buff_marks:
                config_id = int(_get_any(mark, ["configId", "config_id"], 0) or 0)
                if config_id == mark_id:
                    mark_layer = int(_get(mark, "layer", 0) or 0)
                    break
            if mark_layer is not None:
                layer = max(min(mark_layer, max_layer), 0)
                x_mark[now_feature_idx + layer] = 1.0
                used_marks += 1
            now_feature_idx += max_layer + 1
        if used_marks < len(buff_marks):
            x_mark[-1] = 1.0

        return x_hp + x_mark + self._process_position(self._position(unit))

    def _process_skill(self, slot):
        cd_dim = int(Args.CD_MAX_SIZE / Args.CD_UNIT_SIZE) + 2
        x_cd = [0.0] * (2 + cd_dim)
        if slot is None:
            x_cd[-1] = 1.0
            return x_cd
        cd = _get(slot, "cooldown", 0)
        cd_max = _get(slot, "cooldown_max", 0)
        level = _get(slot, "level", 0)
        usable = bool(_get(slot, "usable", False))
        x_cd[0] = _safe_div(cd, cd_max)
        x_cd[1 + min(int(math.ceil(max(cd, 0) / Args.CD_UNIT_SIZE)), cd_dim - 1)] = 1.0
        if not usable and int(level or 0) == 0:
            x_cd[-1] = 1.0
        return x_cd

    def _skill_slots(self, hero):
        slot_states = _get(_get(hero or {}, "skill_state", {}) or {}, "slot_states", []) or []
        slots_by_type = {self._slot_type_key(_get(slot, "slot_type", None)): slot for slot in slot_states}
        ordered_slot_keys = ["SLOT_SKILL_1", "SLOT_SKILL_2", "SLOT_SKILL_3", "SLOT_SKILL_5", "SLOT_SKILL_4"]
        ordered_slots = [slots_by_type.get(slot_key) for slot_key in ordered_slot_keys]
        if any(slot is not None for slot in ordered_slots):
            return ordered_slots

        slot_states = sorted(
            slot_states,
            key=lambda slot: (
                str(_get(slot, "slot_type", "SLOT_SKILL_VALID")),
                _get_any(slot, ["configId", "config_id"], 0),
            ),
        )
        slot_states = list(slot_states)[:5]
        slot_states += [None] * (5 - len(slot_states))
        return slot_states

    def _slot_type_key(self, slot_type):
        if isinstance(slot_type, int):
            return f"SLOT_SKILL_{slot_type}"
        return str(slot_type)

    def _process_money(self, money, is_enemy):
        idx = int(is_enemy)
        if self.last_money[idx] is None:
            self.last_money[idx] = money
        delta = money - self.last_money[idx]
        self.last_money[idx] = money
        money_dim = int(Args.MONEY_MAX_SIZE / Args.MONEY_UNIT_SIZE) + 1
        x_money = [0.0] * (2 + money_dim)
        if 0 < delta < 20:
            x_money[-2] = 1.0
        else:
            x_money[max(min(int(math.floor(delta / Args.MONEY_UNIT_SIZE)), money_dim - 1), 0)] = 1.0
        x_money[-1] = min(float(money) / 10000.0, 1.0)
        return x_money

    def _process_tower_relation(self, unit, tower):
        if unit is None or tower is None:
            return [0.0, 0.0]
        in_range = math.dist(self._position(unit), self._position(tower)) <= _get(tower, "attack_range", 0)
        is_target = self._runtime_id(unit) == _get(tower, "attack_target", None)
        return [float(in_range), float(is_target)]

    def _process_hero(self, hero, is_enemy, opposed_tower):
        if hero is None:
            return [0.0] * Args.DIM_HERO

        config_ids = Args.HERO_CONFIG_ID
        config_id = _get(hero, "config_id", 0)
        if config_id in config_ids:
            hero_id_value = -1.0 if config_ids.index(config_id) == 0 else 1.0
        else:
            hero_id_value = 0.0
        x_hero_id = [hero_id_value]

        x_behave = [0.0] * (len(Args.HERO_BEHAVE) + 1)
        behave = _get(hero, "behav_mode", None)
        if _get(hero, "hp", 0) <= 0:
            idx = 0
        elif behave in Args.HERO_BEHAVE:
            idx = Args.HERO_BEHAVE.index(behave)
        else:
            idx = len(x_behave) - 1
        x_behave[idx] = 1.0

        ep_dim = int(Args.EP_MAX_SIZE / Args.EP_UNIT_SIZE) + 1
        x_ep = [0.0] * (1 + ep_dim)
        ep = _get(hero, "ep", 0)
        x_ep[0] = _safe_div(ep, _get(hero, "max_ep", 0))
        x_ep[1 + min(int(math.floor(max(ep, 0) / Args.EP_UNIT_SIZE)), ep_dim - 1)] = 1.0

        skill_features = []
        for slot in self._skill_slots(hero):
            skill_features.extend(self._process_skill(slot))

        x_level = [0.0] * Args.LEVEL_MAX
        level = max(min(int(_get(hero, "level", 1) or 1), Args.LEVEL_MAX), 1)
        x_level[level - 1] = 1.0

        money = _get(hero, "money_cnt", _get(hero, "money", 0))
        x_money = self._process_money(money, is_enemy)
        x_grass = [float(bool(_get(hero, "is_in_grass", False)))]
        x_organ = self._process_tower_relation(hero, opposed_tower)

        x_buff = [0.0] * Args.DIM_BUFF
        buff_skills = _get(_get(hero, "buff_state", {}) or {}, "buff_skills", []) or []
        for buff in buff_skills:
            buff_id = int(_get_any(buff, ["configId", "config_id"], 0) or 0)
            if buff_id in Args.BUFFS:
                x_buff[Args.BUFFS.index(buff_id)] = 1.0
            else:
                x_buff[-1] = 1.0

        values = (
            x_hero_id
            + x_behave
            + x_ep
            + skill_features
            + x_level
            + x_money
            + x_grass
            + x_organ
            + x_buff
            + self._process_unit(hero)
        )
        assert len(values) == Args.DIM_HERO, f"{len(values)=}, {Args.DIM_HERO=}"
        return values

    def _process_soldiers(self, soldiers, opposed_tower):
        values = []
        for soldier in soldiers[: Args.SOLDIER_MAX_NUM]:
            x_soldier = [0.0] * (Args.DIM_SOLDIER - Args.DIM_UNIT)
            behave = _get(soldier, "behav_mode", None)
            if _get(soldier, "hp", 0) <= 0:
                behave_idx = 0
            elif behave in Args.SOLDIER_BEHAVE:
                behave_idx = Args.SOLDIER_BEHAVE.index(behave)
            else:
                behave_idx = len(Args.SOLDIER_BEHAVE)
            x_soldier[behave_idx] = 1.0

            base_idx = len(Args.SOLDIER_BEHAVE) + 1
            config_id = _get(soldier, "config_id", 0)
            for type_idx, ids in enumerate(Args.SOLDIER_CONFIG_ID):
                if config_id in ids:
                    x_soldier[base_idx + type_idx] = 1.0
            base_idx += len(Args.SOLDIER_CONFIG_ID)

            relation = self._process_tower_relation(soldier, opposed_tower)
            x_soldier[base_idx] = relation[0]
            x_soldier[base_idx + 1] = relation[1]
            values.extend(x_soldier + self._process_unit(soldier))

        values.extend([0.0] * (Args.DIM_SOLDIERS - len(values)))
        return values

    def _process_river_crab(self, monster):
        if monster is None:
            return [0.0] * Args.DIM_RIVER_CRAB
        x_behave = [0.0] * (len(Args.RIVER_CRAB_BEHAVE) + 1)
        behave = _get(monster, "behav_mode", None)
        if _get(monster, "hp", 0) <= 0:
            idx = 0
        elif behave in Args.RIVER_CRAB_BEHAVE:
            idx = Args.RIVER_CRAB_BEHAVE.index(behave)
        else:
            idx = len(x_behave) - 1
        x_behave[idx] = 1.0
        return x_behave + self._process_unit(monster)

    def _process_tower(self, tower, cake, is_enemy):
        if tower is None:
            return [0.0] * Args.DIM_ORGAN
        x_target = [0.0] * 5
        attack_target = _get(tower, "attack_target", 0)
        if not attack_target:
            x_target[0] = 1.0
        elif self.id2type.get(attack_target) == "hero":
            x_target[1] = 1.0
        elif self.id2type.get(attack_target) == "soldier":
            x_target[2] = 1.0
        else:
            x_target[2] = 1.0
        x_target[3] = float(cake is not None)
        cake_idx = int(is_enemy)
        if cake is not None:
            self.next_cake_frame[cake_idx] = self.n_frame + 76 * 30
            x_target[4] = 0.0
        else:
            x_target[4] = min(max((self.next_cake_frame[cake_idx] - self.n_frame) / (75 * 30), 0.0), 1.0)
        return x_target + self._process_unit(tower)

    def _process_bullet(self, bullet):
        if bullet is None:
            return [0.0] * Args.DIM_BULLET
        x_slot = [0.0] * (Args.DIM_BULLET - Args.DIM_DISTANCE)
        slot_type = _get(bullet, "slot_type", "SLOT_SKILL_VALID")
        slot_key = f"SLOT_SKILL_{slot_type}" if isinstance(slot_type, int) and 0 <= slot_type <= 3 else str(slot_type)
        if slot_key in Args.BULLET_SLOT:
            x_slot[Args.BULLET_SLOT.index(slot_key)] = 1.0
        else:
            x_slot[-1] = 1.0
        return x_slot + self._process_position(self._position(bullet))

    def _process_bullets(self, hero_bullets, tower_bullet):
        values = []
        for bullet in hero_bullets[: Args.BULLET_MAX_NUM - 1]:
            values.extend(self._process_bullet(bullet))
        values.extend([0.0] * ((Args.BULLET_MAX_NUM - 1) * Args.DIM_BULLET - len(values)))
        values.extend(self._process_bullet(tower_bullet))
        assert len(values) == Args.DIM_BULLETS, f"{len(values)=}, {Args.DIM_BULLETS=}"
        return values

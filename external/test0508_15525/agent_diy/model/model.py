#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
hok_semi-style PPO policy/value model for agent_diy.
"""

from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.nn import ModuleDict

from agent_diy.conf.conf import Args, Config, DimConfig


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model_name = Config.NETWORK_NAME
        self.data_split_shape = Config.DATA_SPLIT_SHAPE
        self.lstm_time_steps = Config.LSTM_TIME_STEPS
        self.lstm_unit_size = Config.LSTM_UNIT_SIZE
        self.dim_public = Config.DIM_PUBLIC
        self.seri_vec_split_shape = Config.SERI_VEC_SPLIT_SHAPE
        self.m_var_beta = Config.BETA_START
        self.log_epsilon = Config.LOG_EPSILON
        self.label_size_list = Config.LABEL_SIZE_LIST
        self.is_reinforce_task_list = Config.IS_REINFORCE_TASK_LIST
        self.min_policy = Config.MIN_POLICY
        self.clip_param = Config.CLIP_PARAM
        self.dual_clip_param = Config.DUAL_CLIP_PARAM
        self.var_beta = self.m_var_beta
        self.target_embed_dim = Config.TARGET_EMBED_DIM
        self.cut_points = [value[0] for value in Config.data_shapes]
        self.legal_action_size = Config.LEGAL_ACTION_SIZE_LIST

        self.feature_dim = Config.SERI_VEC_SPLIT_SHAPE[0][0]
        self.legal_action_dim = np.sum(Config.LEGAL_ACTION_SIZE_LIST)
        self.lstm_hidden_dim = Config.LSTM_UNIT_SIZE

        self.single_hero_feature_dim = int(DimConfig.DIM_OF_HERO_EMY[0])
        self.single_soldier_feature_dim = int(DimConfig.DIM_OF_SOLDIER_1_4[0])
        self.single_organ_feature_dim = int(DimConfig.DIM_OF_ORGAN_1[0])
        self.single_river_crab_feature_dim = int(DimConfig.DIM_OF_RIVER_CRAB[0])
        self.single_bullet_feature_dim = int(DimConfig.DIM_OF_BULLET_1_9[0])

        self.all_hero_feature_dim = int(np.sum(DimConfig.DIM_OF_HERO_FRD)) + int(
            np.sum(DimConfig.DIM_OF_HERO_EMY)
        )
        self.all_soldier_feature_dim = int(np.sum(DimConfig.DIM_OF_SOLDIER_1_4)) + int(
            np.sum(DimConfig.DIM_OF_SOLDIER_5_8)
        )
        self.all_organ_feature_dim = int(np.sum(DimConfig.DIM_OF_ORGAN_1)) + int(
            np.sum(DimConfig.DIM_OF_ORGAN_2)
        )
        self.all_bullet_feature_dim = int(np.sum(DimConfig.DIM_OF_BULLET_1_9)) + int(
            np.sum(DimConfig.DIM_OF_BULLET_10)
        )

        self.position_mlp = MLP([Args.DIM_DISTANCE, 128, 64], "position_mlp")
        self.position_delta_dim = 64 - Args.DIM_DISTANCE
        self.unit_no_pos_mlp = MLP([Args.DIM_UNIT - Args.DIM_DISTANCE, 64, 32], "unit_no_pos_mlp")
        self.unit_delta_dim = 64 + 32 - Args.DIM_UNIT

        fc_hero_dim_list = [self.single_hero_feature_dim + self.unit_delta_dim, 512, 256, 128]
        self.hero_mlp = MLP(fc_hero_dim_list[:-1], "hero_mlp", non_linearity_last=True)
        self.hero_frd_fc = make_fc_layer(fc_hero_dim_list[-2], fc_hero_dim_list[-1])
        self.hero_emy_fc = make_fc_layer(fc_hero_dim_list[-2], fc_hero_dim_list[-1])

        fc_soldier_dim_list = [self.single_soldier_feature_dim + self.unit_delta_dim, 128, 64, 32]
        self.soldier_mlp = MLP(fc_soldier_dim_list[:-1], "soldier_mlp", non_linearity_last=True)
        self.soldier_frd_fc = make_fc_layer(fc_soldier_dim_list[-2], fc_soldier_dim_list[-1])
        self.soldier_emy_fc = make_fc_layer(fc_soldier_dim_list[-2], fc_soldier_dim_list[-1])

        fc_river_crab_list = [self.single_river_crab_feature_dim + self.unit_delta_dim, 128, 64, 32]
        self.river_crab_mlp = MLP(fc_river_crab_list, "river_crab_mlp")

        fc_organ_dim_list = [self.single_organ_feature_dim + self.unit_delta_dim, 128, 64, 32]
        self.organ_mlp = MLP(fc_organ_dim_list[:-1], "organ_mlp", non_linearity_last=True)
        self.organ_frd_fc = make_fc_layer(fc_organ_dim_list[-2], fc_organ_dim_list[-1])
        self.organ_emy_fc = make_fc_layer(fc_organ_dim_list[-2], fc_organ_dim_list[-1])

        fc_bullet_list = [self.single_bullet_feature_dim + self.position_delta_dim, 64, 64, 32]
        self.bullet_mlp = MLP(fc_bullet_list[:-1], "bullet_mlp", non_linearity_last=True)
        self.bullet_hero_fc = make_fc_layer(fc_bullet_list[-2], fc_bullet_list[-1])
        self.bullet_organ_fc = make_fc_layer(fc_bullet_list[-2], fc_bullet_list[-1])

        concat_dim = 128 * 2 + 32 * 2 + 32 + 32 * 2 + 32 * 2
        self.concat_mlp = MLP([concat_dim, self.lstm_unit_size], "concat_mlp", non_linearity_last=True)
        self.concat_mlp_other = MLP([concat_dim, 512, self.lstm_unit_size], "concat_other_mlp")
        self.lstm_and_linear_mlp = MLP(
            [self.lstm_unit_size * 2, self.dim_public],
            "lstm_and_linear_mlp",
            non_linearity_last=True,
        )

        self.lstm = nn.LSTM(
            input_size=self.lstm_unit_size,
            hidden_size=self.lstm_unit_size,
            num_layers=1,
            bias=True,
            batch_first=True,
            dropout=Config.LSTM_DROPOUT,
            bidirectional=False,
        )

        self.label_mlp = ModuleDict(
            {
                f"hero_label{label_index}_mlp": MLP(
                    [self.dim_public, self.label_size_list[label_index]],
                    f"hero_label{label_index}_mlp",
                )
                for label_index in range(len(self.label_size_list) - 1)
            }
        )
        self.lstm_tar_embed_mlp = make_fc_layer(self.dim_public, self.target_embed_dim)
        self.target_embed_mlp = make_fc_layer(32, self.target_embed_dim, use_bias=False)
        self.value_mlp = MLP([self.dim_public, 64, 1], "hero_value_mlp")

    def process_sub_feature(self, x, mlp, is_unit: bool):
        ret = [x]
        dim_suffix = Args.DIM_DISTANCE
        if is_unit:
            unit_no_pos = self.unit_no_pos_mlp(x[..., -Args.DIM_UNIT : -Args.DIM_DISTANCE])
            ret.append(unit_no_pos)
            dim_suffix = Args.DIM_UNIT
        pos = self.position_mlp(x[..., -Args.DIM_DISTANCE :])
        ret.append(pos)
        ret[0] = x[..., :-dim_suffix]
        return mlp(torch.cat(ret, dim=-1))

    def forward(self, data_list, inference=False):
        feature_vec, lstm_hidden_init, lstm_cell_init = data_list
        if feature_vec.shape[-1] != self.feature_dim:
            raise ValueError(f"model feature dim mismatch: {feature_vec.shape[-1]} != {self.feature_dim}")

        feature_vec_split_list = feature_vec.split(
            [
                self.all_hero_feature_dim,
                self.all_soldier_feature_dim,
                self.single_river_crab_feature_dim,
                self.all_organ_feature_dim,
                self.all_bullet_feature_dim,
            ],
            dim=1,
        )
        hero_vec_list = feature_vec_split_list[0].split(
            [int(np.sum(DimConfig.DIM_OF_HERO_FRD)), int(np.sum(DimConfig.DIM_OF_HERO_EMY))],
            dim=1,
        )
        soldier_vec_list = feature_vec_split_list[1].split(
            [int(np.sum(DimConfig.DIM_OF_SOLDIER_1_4)), int(np.sum(DimConfig.DIM_OF_SOLDIER_5_8))],
            dim=1,
        )
        river_crab_tensor = feature_vec_split_list[2]
        organ_vec_list = feature_vec_split_list[3].split(
            [int(np.sum(DimConfig.DIM_OF_ORGAN_1)), int(np.sum(DimConfig.DIM_OF_ORGAN_2))],
            dim=1,
        )
        bullet_vec_list = feature_vec_split_list[4].split(
            [int(np.sum(DimConfig.DIM_OF_BULLET_1_9)), int(np.sum(DimConfig.DIM_OF_BULLET_10))],
            dim=1,
        )

        hero_frd = hero_vec_list[0].split(DimConfig.DIM_OF_HERO_FRD, dim=1)
        hero_emy = hero_vec_list[1].split(DimConfig.DIM_OF_HERO_EMY, dim=1)
        soldier_1_4 = soldier_vec_list[0].split(DimConfig.DIM_OF_SOLDIER_1_4, dim=1)
        soldier_5_8 = soldier_vec_list[1].split(DimConfig.DIM_OF_SOLDIER_5_8, dim=1)
        organ_1 = organ_vec_list[0].split(DimConfig.DIM_OF_ORGAN_1, dim=1)
        organ_2 = organ_vec_list[1].split(DimConfig.DIM_OF_ORGAN_2, dim=1)
        bullet_1_9 = bullet_vec_list[0].split(DimConfig.DIM_OF_BULLET_1_9, dim=1)
        bullet_10 = bullet_vec_list[1].split(DimConfig.DIM_OF_BULLET_10, dim=1)

        target_embed_list = []

        hero_emy_result_list = []
        for hero_tensor in hero_emy:
            hero_emy_mlp_out = self.process_sub_feature(hero_tensor, self.hero_mlp, True)
            hero_emy_fc_out = self.hero_emy_fc(hero_emy_mlp_out)
            _, target_part = hero_emy_fc_out.split([96, 32], dim=1)
            target_embed_list.append(target_part)
            hero_emy_result_list.append(hero_emy_fc_out)
        hero_emy_concat_result = torch.cat(hero_emy_result_list, dim=1)

        hero_frd_result_list = []
        for hero_tensor in hero_frd:
            hero_frd_mlp_out = self.process_sub_feature(hero_tensor, self.hero_mlp, True)
            hero_frd_fc_out = self.hero_frd_fc(hero_frd_mlp_out)
            _, target_part = hero_frd_fc_out.split([96, 32], dim=1)
            target_embed_list.append(target_part)
            hero_frd_result_list.append(hero_frd_fc_out)
        hero_frd_concat_result = torch.cat(hero_frd_result_list, dim=1)

        soldier_frd_result_list = []
        for soldier_tensor in soldier_1_4:
            soldier_frd_mlp_out = self.process_sub_feature(soldier_tensor, self.soldier_mlp, True)
            soldier_frd_result_list.append(self.soldier_frd_fc(soldier_frd_mlp_out))
        soldier_frd_concat_result = torch.cat(soldier_frd_result_list, dim=1).reshape(-1, 4, 32).max(dim=1).values

        soldier_emy_result_list = []
        for soldier_tensor in soldier_5_8:
            soldier_emy_mlp_out = self.process_sub_feature(soldier_tensor, self.soldier_mlp, True)
            soldier_emy_fc_out = self.soldier_emy_fc(soldier_emy_mlp_out)
            soldier_emy_result_list.append(soldier_emy_fc_out)
            target_embed_list.append(soldier_emy_fc_out)
        soldier_emy_concat_result = torch.cat(soldier_emy_result_list, dim=1).reshape(-1, 4, 32).max(dim=1).values

        river_crab_result = self.process_sub_feature(river_crab_tensor, self.river_crab_mlp, True)

        organ_frd_result_list = []
        for organ_tensor in organ_1:
            organ_frd_mlp_out = self.process_sub_feature(organ_tensor, self.organ_mlp, True)
            organ_frd_result_list.append(self.organ_frd_fc(organ_frd_mlp_out))
        organ_frd_concat_result = torch.cat(organ_frd_result_list, dim=1)

        organ_emy_result_list = []
        for organ_tensor in organ_2:
            organ_emy_mlp_out = self.process_sub_feature(organ_tensor, self.organ_mlp, True)
            organ_emy_fc_out = self.organ_emy_fc(organ_emy_mlp_out)
            organ_emy_result_list.append(organ_emy_fc_out)
            target_embed_list.append(organ_emy_fc_out)
        organ_emy_concat_result = torch.cat(organ_emy_result_list, dim=1)

        bullet_hero_result_list = []
        for bullet_tensor in bullet_1_9:
            bullet_hero_mlp_out = self.process_sub_feature(bullet_tensor, self.bullet_mlp, False)
            bullet_hero_result_list.append(self.bullet_hero_fc(bullet_hero_mlp_out))
        bullet_hero_concat_result = torch.cat(bullet_hero_result_list, dim=1).reshape(-1, 9, 32).max(dim=1).values

        bullet_organ_result_list = []
        for bullet_tensor in bullet_10:
            bullet_organ_mlp_out = self.process_sub_feature(bullet_tensor, self.bullet_mlp, False)
            bullet_organ_result_list.append(self.bullet_organ_fc(bullet_organ_mlp_out))
        bullet_organ_concat_result = torch.cat(bullet_organ_result_list, dim=1)

        target_embed_0 = 0.1 * torch.ones_like(target_embed_list[-1], device=feature_vec.device)
        target_embed_list.insert(0, target_embed_0)
        target_embed_8 = 0.1 * torch.ones_like(target_embed_list[-1], device=feature_vec.device)
        target_embed_list.append(target_embed_8)
        if len(target_embed_list) != self.label_size_list[-1]:
            raise ValueError(f"target embedding count mismatch: {len(target_embed_list)} != {self.label_size_list[-1]}")
        target_embedding = torch.stack(target_embed_list, dim=1)

        concat_result = torch.cat(
            [
                hero_frd_concat_result,
                hero_emy_concat_result,
                soldier_frd_concat_result,
                soldier_emy_concat_result,
                river_crab_result,
                organ_frd_concat_result,
                organ_emy_concat_result,
                bullet_hero_concat_result,
                bullet_organ_concat_result,
            ],
            dim=1,
        )

        fc_public_result = self.concat_mlp(concat_result)
        batch_size = lstm_hidden_init.shape[0]
        total_steps = fc_public_result.shape[0]
        if batch_size <= 0 or total_steps % batch_size != 0:
            raise ValueError(f"invalid lstm batch/step shape: batch={batch_size}, total_steps={total_steps}")
        seq_len = total_steps // batch_size
        reshape_fc_public_result = fc_public_result.reshape(batch_size, seq_len, self.lstm_unit_size)
        lstm_initial_state = [lstm_hidden_init.unsqueeze(0), lstm_cell_init.unsqueeze(0)]
        lstm_outputs, state = self.lstm(reshape_fc_public_result, lstm_initial_state)
        lstm_hidden_output, lstm_cell_output = state[0], state[1]
        lstm_outputs = lstm_outputs.reshape(-1, self.lstm_unit_size)

        public_mlp_result = self.concat_mlp_other(concat_result)
        public_hidden = self.lstm_and_linear_mlp(torch.cat([lstm_outputs, public_mlp_result], dim=-1))
        public_hidden = public_hidden.reshape(-1, self.dim_public)

        result_list = []
        for label_index in range(len(self.label_size_list) - 1):
            label_logits = self.label_mlp[f"hero_label{label_index}_mlp"](public_hidden)
            if label_index == 0:
                button_bias = torch.as_tensor(
                    Config.BUTTON_LOGIT_BIAS,
                    dtype=label_logits.dtype,
                    device=label_logits.device,
                )
                label_logits = label_logits + button_bias
            result_list.append(label_logits)

        lstm_target_embed_result = self.lstm_tar_embed_mlp(public_hidden).reshape(-1, self.target_embed_dim, 1)
        target_query = self.target_embed_mlp(target_embedding)
        target_query = nn.functional.softmax(target_query, dim=-1)
        target_logits = torch.matmul(target_query, lstm_target_embed_result).reshape(-1, self.label_size_list[-1])
        result_list.append(target_logits)

        value_result = self.value_mlp(public_hidden)
        result_list.append(value_result)

        logits = torch.flatten(torch.cat(result_list[:-1], 1), start_dim=1)
        value = result_list[-1]
        if inference:
            return [logits, value, lstm_cell_output, lstm_hidden_output]
        return result_list

    def compute_loss(self, data_list, rst_list):
        seri_vec = data_list[0].reshape(-1, self.data_split_shape[0])
        usq_reward = data_list[1].reshape(-1, self.data_split_shape[1])
        usq_advantage = data_list[2].reshape(-1, self.data_split_shape[2])
        usq_is_train = data_list[-3].reshape(-1, self.data_split_shape[-3])

        usq_label_list = data_list[3 : 3 + len(self.label_size_list)]
        for shape_index in range(len(self.label_size_list)):
            usq_label_list[shape_index] = usq_label_list[shape_index].reshape(
                -1, self.data_split_shape[3 + shape_index]
            ).long()

        old_label_probability_list = data_list[3 + len(self.label_size_list) : 3 + 2 * len(self.label_size_list)]
        for shape_index in range(len(self.label_size_list)):
            old_label_probability_list[shape_index] = old_label_probability_list[shape_index].reshape(
                -1, self.data_split_shape[3 + len(self.label_size_list) + shape_index]
            )

        usq_weight_list = data_list[3 + 2 * len(self.label_size_list) : 3 + 3 * len(self.label_size_list)]
        for shape_index in range(len(self.label_size_list)):
            usq_weight_list[shape_index] = usq_weight_list[shape_index].reshape(
                -1,
                self.data_split_shape[3 + 2 * len(self.label_size_list) + shape_index],
            )

        reward = usq_reward.squeeze(dim=1)
        advantage = usq_advantage.squeeze(dim=1)
        frame_is_train = usq_is_train.squeeze(dim=1)
        raw_advantage = advantage.clone()
        if Config.USE_ADVANTAGE_NORM:
            valid_adv = advantage[frame_is_train > 0]
            if valid_adv.numel() > 1:
                advantage = (advantage - valid_adv.mean()) / (valid_adv.std(unbiased=False) + 1e-8)

        label_list = [ele.squeeze(dim=1) for ele in usq_label_list]
        weight_list = [weight.squeeze(dim=1) for weight in usq_weight_list]
        label_result = rst_list[:-1]
        value_result = rst_list[-1]

        _, split_feature_legal_action = torch.split(
            seri_vec,
            [np.prod(self.seri_vec_split_shape[0]), np.prod(self.seri_vec_split_shape[1])],
            dim=1,
        )
        feature_legal_action_shape = list(self.seri_vec_split_shape[1])
        feature_legal_action_shape.insert(0, -1)
        feature_legal_action = split_feature_legal_action.reshape(feature_legal_action_shape)
        legal_action_flag_list = torch.split(feature_legal_action, self.label_size_list, dim=1)

        value_result_squeezed = value_result.squeeze(dim=1)
        value_loss_unclipped = torch.square(reward - value_result_squeezed)
        if Config.USE_VALUE_CLIP:
            old_value = reward - raw_advantage
            value_clipped = old_value + torch.clamp(
                value_result_squeezed - old_value,
                -Config.VALUE_CLIP_PARAM,
                Config.VALUE_CLIP_PARAM,
            )
            value_loss_clipped = torch.square(reward - value_clipped)
            value_loss = torch.maximum(value_loss_unclipped, value_loss_clipped)
        else:
            value_loss = value_loss_unclipped
        value_train_weight = frame_is_train.float()
        self.value_cost = 0.5 * torch.sum(value_loss * value_train_weight) / torch.maximum(
            torch.sum(value_train_weight),
            torch.tensor(1.0, device=reward.device),
        )

        label_probability_list = []
        epsilon = 1e-5
        self.policy_cost = torch.tensor(0.0, device=reward.device)
        boundary = torch.tensor(1e20, device=reward.device)
        ratio_sum = torch.tensor(0.0, device=reward.device)
        kl_sum = torch.tensor(0.0, device=reward.device)
        clip_sum = torch.tensor(0.0, device=reward.device)
        ratio_weight_sum = torch.tensor(0.0, device=reward.device)

        for task_index in range(len(self.is_reinforce_task_list)):
            if not self.is_reinforce_task_list[task_index]:
                continue

            one_hot_actions = nn.functional.one_hot(label_list[task_index].long(), self.label_size_list[task_index])
            legal_mask = legal_action_flag_list[task_index]
            legal_action_flag_list_max_mask = (1 - legal_mask) * boundary
            label_logits_subtract_max = torch.clamp(
                label_result[task_index]
                - torch.max(label_result[task_index] - legal_action_flag_list_max_mask, dim=1, keepdim=True).values,
                -boundary,
                1,
            )
            label_exp_logits = legal_mask * (torch.exp(label_logits_subtract_max) + self.min_policy)
            label_sum_exp_logits = torch.clamp(label_exp_logits.sum(1, keepdim=True), min=epsilon)
            label_probability = label_exp_logits / label_sum_exp_logits
            label_probability_list.append(label_probability)

            policy_p = (one_hot_actions * label_probability).sum(1)
            policy_log_p = torch.log(policy_p + epsilon)
            old_policy_p = (one_hot_actions * old_label_probability_list[task_index] + epsilon).sum(1)
            old_policy_log_p = torch.log(old_policy_p)
            ratio = torch.exp(policy_log_p - old_policy_log_p)

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantage
            clipped = torch.minimum(surr1, surr2)
            dual_clipped = torch.where(advantage < 0, torch.maximum(clipped, self.dual_clip_param * advantage), clipped)
            train_weight = weight_list[task_index].float() * frame_is_train
            temp_policy_loss = -torch.sum(dual_clipped * train_weight) / torch.maximum(
                torch.sum(train_weight), torch.tensor(1.0, device=reward.device)
            )
            self.policy_cost = self.policy_cost + temp_policy_loss
            ratio_sum = ratio_sum + torch.sum(ratio.detach() * train_weight)
            kl_sum = kl_sum + torch.sum((old_policy_log_p - policy_log_p).detach() * train_weight)
            clip_sum = clip_sum + torch.sum((torch.abs(ratio.detach() - 1.0) > self.clip_param).float() * train_weight)
            ratio_weight_sum = ratio_weight_sum + torch.sum(train_weight)

        entropy_loss_list = []
        current_entropy_loss_index = 0
        for task_index in range(len(self.is_reinforce_task_list)):
            if self.is_reinforce_task_list[task_index]:
                temp_entropy_loss = -torch.sum(
                    label_probability_list[current_entropy_loss_index]
                    * legal_action_flag_list[task_index]
                    * torch.log(label_probability_list[current_entropy_loss_index] + epsilon),
                    dim=1,
                )
                train_weight = weight_list[task_index].float() * frame_is_train
                temp_entropy_loss = -torch.sum(temp_entropy_loss * train_weight) / torch.maximum(
                    torch.sum(train_weight), torch.tensor(1.0, device=reward.device)
                )
                entropy_loss_list.append(temp_entropy_loss)
                current_entropy_loss_index += 1
            else:
                entropy_loss_list.append(torch.tensor(0.0, device=reward.device))

        self.entropy_cost = torch.tensor(0.0, device=reward.device)
        for entropy_element in entropy_loss_list:
            self.entropy_cost = self.entropy_cost + entropy_element

        self.loss = self.value_cost + self.policy_cost + self.var_beta * self.entropy_cost
        ratio_den = torch.maximum(ratio_weight_sum, torch.tensor(1.0, device=reward.device))
        self.mean_ratio = ratio_sum / ratio_den
        self.approx_kl = kl_sum / ratio_den
        self.clip_fraction = clip_sum / ratio_den
        return self.loss, [self.loss, [self.value_cost, self.policy_cost, self.entropy_cost]]

    def set_train_mode(self):
        self.lstm_time_steps = Config.LSTM_TIME_STEPS
        self.train()

    def set_eval_mode(self):
        self.lstm_time_steps = 1
        self.eval()


def make_fc_layer(in_features: int, out_features: int, use_bias=True):
    fc_layer = nn.Linear(in_features, out_features, bias=use_bias)
    nn.init.orthogonal_(fc_layer.weight)
    if use_bias:
        nn.init.zeros_(fc_layer.bias)
    return fc_layer


class MLP(nn.Module):
    def __init__(
        self,
        fc_feat_dim_list: List[int],
        name: str,
        non_linearity: nn.Module = nn.ReLU,
        non_linearity_last: bool = False,
    ):
        super(MLP, self).__init__()
        self.fc_layers = nn.Sequential()
        for i in range(len(fc_feat_dim_list) - 1):
            fc_layer = make_fc_layer(fc_feat_dim_list[i], fc_feat_dim_list[i + 1])
            self.fc_layers.add_module(f"{name}_fc{i + 1}", fc_layer)
            if i + 1 < len(fc_feat_dim_list) - 1 or non_linearity_last:
                self.fc_layers.add_module(f"{name}_non_linear{i + 1}", non_linearity())

    def forward(self, data):
        return self.fc_layers(data)

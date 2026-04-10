# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause
# 
# # Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _set_reward_weight(env: ManagerBasedRLEnv, term_name: str, weight: float) -> None:
    """Mutate a reward term weight through the reward manager config."""
    reward_term_cfg = env.reward_manager.get_term_cfg(term_name)
    reward_term_cfg.weight = weight


def staged_command_lift_stairs(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> torch.Tensor:
    """Three-phase curriculum for command tracking, anti-drag foot lift, and stair walking.

    Phase 1:
        Learn stable command tracking with smaller command ranges and lighter gait shaping.
    Phase 2:
        Over-bias swing height / anti-drag rewards to break the low-clearance local optimum.
    Phase 3:
        Restore broader commands and moderate stair-oriented gait shaping for usable stair walking.
    """
    del env_ids  # curriculum is shared across all environments

    base_velocity_ranges = env.command_manager.get_term("base_velocity").cfg.ranges
    total_steps = int(env.common_step_counter)

    if total_steps == 0:
        env._airbotix_curriculum_initialized = True
        env._airbotix_stage = -1
        env._airbotix_stage_steps = (
            20_000_000,   # command tracking
            80_000_000,   # break anti-drag bias
        )

    stage = 0
    if total_steps >= env._airbotix_stage_steps[1]:
        stage = 2
    elif total_steps >= env._airbotix_stage_steps[0]:
        stage = 1

    if getattr(env, "_airbotix_stage", -1) != stage:
        env._airbotix_stage = stage

        if stage == 0:
            base_velocity_ranges.lin_vel_x = [-0.35, 0.45]
            base_velocity_ranges.lin_vel_y = [-0.20, 0.20]
            base_velocity_ranges.ang_vel_z = [-0.35, 0.35]

            _set_reward_weight(env, "track_lin_vel_xy_exp", 5.5)
            _set_reward_weight(env, "track_ang_vel_z_exp", 3.0)
            _set_reward_weight(env, "action_rate_l2", -0.003)
            _set_reward_weight(env, "feet_air_time", 4.0)
            _set_reward_weight(env, "feet_height_body", -0.4)
            _set_reward_weight(env, "feet_slide", -0.08)
            _set_reward_weight(env, "feet_gait", 0.5)
            _set_reward_weight(env, "stand_still", -0.1)
            env.reward_manager.get_term_cfg("feet_air_time").params["threshold"] = 0.40
            env.reward_manager.get_term_cfg("feet_height_body").params["target_height"] = -0.22

        elif stage == 1:
            base_velocity_ranges.lin_vel_x = [-0.50, 0.65]
            base_velocity_ranges.lin_vel_y = [-0.30, 0.30]
            base_velocity_ranges.ang_vel_z = [-0.45, 0.45]

            _set_reward_weight(env, "track_lin_vel_xy_exp", 5.0)
            _set_reward_weight(env, "track_ang_vel_z_exp", 3.0)
            _set_reward_weight(env, "action_rate_l2", -0.002)
            _set_reward_weight(env, "feet_air_time", 14.0)
            _set_reward_weight(env, "feet_height_body", -2.0)
            _set_reward_weight(env, "feet_slide", -0.15)
            _set_reward_weight(env, "feet_gait", 1.0)
            _set_reward_weight(env, "stand_still", -0.05)
            env.reward_manager.get_term_cfg("feet_air_time").params["threshold"] = 0.25
            env.reward_manager.get_term_cfg("feet_height_body").params["target_height"] = -0.14

        else:
            base_velocity_ranges.lin_vel_x = [-0.60, 0.80]
            base_velocity_ranges.lin_vel_y = [-0.40, 0.40]
            base_velocity_ranges.ang_vel_z = [-0.50, 0.50]

            _set_reward_weight(env, "track_lin_vel_xy_exp", 5.0)
            _set_reward_weight(env, "track_ang_vel_z_exp", 3.0)
            _set_reward_weight(env, "action_rate_l2", -0.003)
            _set_reward_weight(env, "feet_air_time", 9.0)
            _set_reward_weight(env, "feet_height_body", -1.0)
            _set_reward_weight(env, "feet_slide", -0.10)
            _set_reward_weight(env, "feet_gait", 0.8)
            _set_reward_weight(env, "stand_still", -0.08)
            env.reward_manager.get_term_cfg("feet_air_time").params["threshold"] = 0.35
            env.reward_manager.get_term_cfg("feet_height_body").params["target_height"] = -0.18

    return torch.tensor(float(stage), device=env.device)


def command_levels_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str,
    range_multiplier: Sequence[float] = (0.1, 1.0),
) -> None:
    """command_levels_vel"""
    base_velocity_ranges = env.command_manager.get_term("base_velocity").cfg.ranges
    # Get original velocity ranges (ONLY ON FIRST EPISODE)
    if env.common_step_counter == 0:
        env._original_vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device)
        env._original_vel_y = torch.tensor(base_velocity_ranges.lin_vel_y, device=env.device)
        env._initial_vel_x = env._original_vel_x * range_multiplier[0]
        env._final_vel_x = env._original_vel_x * range_multiplier[1]
        env._initial_vel_y = env._original_vel_y * range_multiplier[0]
        env._final_vel_y = env._original_vel_y * range_multiplier[1]

        # Initialize command ranges to initial values
        base_velocity_ranges.lin_vel_x = env._initial_vel_x.tolist()
        base_velocity_ranges.lin_vel_y = env._initial_vel_y.tolist()

    # avoid updating command curriculum at each step since the maximum command is common to all envs
    if env.common_step_counter % env.max_episode_length == 0:
        episode_sums = env.reward_manager._episode_sums[reward_term_name]
        reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
        delta_command = torch.tensor([-0.1, 0.1], device=env.device)

        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(episode_sums[env_ids]) / env.max_episode_length_s > 0.8 * reward_term_cfg.weight:
            new_vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device) + delta_command
            new_vel_y = torch.tensor(base_velocity_ranges.lin_vel_y, device=env.device) + delta_command

            # Clamp to ensure we don't exceed final ranges
            new_vel_x = torch.clamp(new_vel_x, min=env._final_vel_x[0], max=env._final_vel_x[1])
            new_vel_y = torch.clamp(new_vel_y, min=env._final_vel_y[0], max=env._final_vel_y[1])

            # Update ranges
            base_velocity_ranges.lin_vel_x = new_vel_x.tolist()
            base_velocity_ranges.lin_vel_y = new_vel_y.tolist()

    return torch.tensor(base_velocity_ranges.lin_vel_x[1], device=env.device)

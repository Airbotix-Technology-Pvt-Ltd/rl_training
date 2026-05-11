# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause
#
# # Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def joint_pos_rel_without_wheel(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    wheel_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.(Without the wheel joints)"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos_rel = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    joint_pos_rel[:, wheel_asset_cfg.joint_ids] = 0
    return joint_pos_rel


def phase(env: ManagerBasedRLEnv, cycle_time: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf") or env.episode_length_buf is None:
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    phase = env.episode_length_buf[:, None] * env.step_dt / cycle_time
    phase_tensor = torch.cat([torch.sin(2 * torch.pi * phase), torch.cos(2 * torch.pi * phase)], dim=-1)
    return phase_tensor


def terrain_height_diff_observation(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Compute terrain height difference between front and back of the robot.

    This observation helps the robot detect slopes and stairs by comparing
    the height of terrain in front of the robot to the terrain behind it.
    Returns: [height_diff_front_back, max_height_diff, terrain_class]
    """
    sensor: RayCaster = env.scene[sensor_cfg.name]
    ray_hits = sensor.data.ray_hits_w[..., 2]
    if torch.isnan(ray_hits).any() or torch.isinf(ray_hits).any() or ray_hits.numel() == 0:
        return torch.zeros(env.num_envs, 3, device=env.device)
    max_h = torch.max(ray_hits, dim=1)[0]
    min_h = torch.min(ray_hits, dim=1)[0]
    height_diff_front_back = max_h - min_h
    max_height_diff = height_diff_front_back
    terrain_class = torch.clamp(max_height_diff / 0.15, max=1.0)
    obs = torch.stack([height_diff_front_back, max_height_diff, terrain_class], dim=-1)
    return obs


def upcoming_terrain_slope(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Compute the upcoming terrain slope based on height scan.

    This observation estimates the slope of terrain ahead of the robot
    which is useful for stair climbing anticipation.
    Returns: [slope_x, slope_y, curvature]
    """
    sensor: RayCaster = env.scene[sensor_cfg.name]
    ray_hits = sensor.data.ray_hits_w[..., 2]
    if torch.isnan(ray_hits).any() or torch.isinf(ray_hits).any() or ray_hits.numel() == 0:
        return torch.zeros(env.num_envs, 3, device=env.device)
    max_h = torch.max(ray_hits, dim=1)[0]
    min_h = torch.min(ray_hits, dim=1)[0]
    slope_x = max_h - min_h
    slope_y = torch.zeros(env.num_envs, device=env.device)
    curvature = torch.var(ray_hits, dim=1)
    obs = torch.stack([slope_x, slope_y, curvature], dim=-1)
    return obs


def foot_placement_heuristic(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Heuristic observation for foot placement on stairs.

    Provides information about ideal foot placement locations based on
    terrain geometry.
    Returns: [step_detected, step_height, ideal_clearance]
    """
    sensor: RayCaster = env.scene[sensor_cfg.name]
    ray_hits = sensor.data.ray_hits_w[..., 2]
    if torch.isnan(ray_hits).any() or torch.isinf(ray_hits).any() or ray_hits.numel() == 0:
        return torch.zeros(env.num_envs, 3, device=env.device)
    max_height = torch.max(ray_hits, dim=1)[0]
    min_height = torch.min(ray_hits, dim=1)[0]
    height_range = max_height - min_height
    step_detected = (height_range > 0.05).float()
    step_height = torch.clamp(height_range, max=0.2)
    ideal_clearance = torch.clamp(step_height + 0.05, max=0.15)
    obs = torch.stack([step_detected, step_height, ideal_clearance], dim=-1)
    return obs

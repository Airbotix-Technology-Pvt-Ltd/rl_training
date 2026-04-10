from pathlib import Path

from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.sim import UsdFileCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

import rl_training.tasks.manager_based.locomotion.velocity.mdp as mdp
from .rough_env_cfg import DeeproboticsLite3RoughEnvCfg


@configclass
class DeeproboticsLite3AirbotixEnvCfg(DeeproboticsLite3RoughEnvCfg):
    custom_usd_path = Path(
        "/home/lite3/work/Lite3Robot/Lite3_sdk_deploy/src/isaac_bridge/isaacsim/demo3_training_stairs_ref.usda"
    )

    def __post_init__(self):
        super().__post_init__()

        # Keep only a handful of copies of the custom scene and separate them well
        # so the stair sets do not visually overlap in the simulator.
        self.scene.num_envs = 8
        self.scene.env_spacing = 18.0

        # Disable the height-scan observation path. We keep contact sensors enabled
        # because the locomotion rewards still use contact timing and force signals.
        if hasattr(self.observations.policy, "height_scan"):
            self.observations.policy.height_scan = None

        if hasattr(self.observations.critic, "height_scan"):
            self.observations.critic.height_scan = None

        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="plane",
        )

        if not self.custom_usd_path.is_file():
            raise FileNotFoundError(f"Custom USD scene not found: {self.custom_usd_path}")

        self.scene.custom_env = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Env",
            spawn=UsdFileCfg(usd_path=str(self.custom_usd_path)),
        )

        # Spawn the robot on the approach to the first stair flight instead of at the
        # origin, which is off to the side of the custom environment.
        self.scene.robot.init_state.pos = (1.35, 0.0, 0.42)

        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-0.2, 0.2),
                "y": (-0.15, 0.15),
                "z": (0.0, 0.05),
                "roll": (-0.05, 0.05),
                "pitch": (-0.05, 0.05),
                "yaw": (-0.15, 0.15),
            },
            "velocity_range": {
                "x": (-0.1, 0.1),
                "y": (-0.1, 0.1),
                "z": (-0.05, 0.05),
                "roll": (-0.05, 0.05),
                "pitch": (-0.05, 0.05),
                "yaw": (-0.1, 0.1),
            },
        }

        # Curriculum phase 1 starts with easier command magnitudes so the robot learns
        # to follow velocity commands before we push foot lift and stair traversal.
        self.commands.base_velocity.ranges.lin_vel_x = (-0.35, 0.45)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.2, 0.2)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.35, 0.35)
        self.commands.base_velocity.heading_command = True
        self.commands.base_velocity.rel_heading_envs = 0.6
        self.commands.base_velocity.rel_standing_envs = 0.05

        # Phase 1 defaults: prioritize command tracking and stable gait before we add
        # the stronger anti-drag / stair-specific shaping in later curriculum phases.
        self.rewards.track_lin_vel_xy_exp.weight = 5.5
        self.rewards.track_ang_vel_z_exp.weight = 3.0

        # Reduce competing penalties that were encouraging a conservative, low-motion gait.
        self.rewards.action_rate_l2.weight = -0.003
        self.rewards.flat_orientation_l2.weight = -1.5
        self.rewards.undesired_contacts.weight = -0.2
        self.rewards.contact_forces.weight = -0.04

        self.rewards.feet_air_time.weight = 4.0
        self.rewards.feet_air_time.params["threshold"] = 0.4
        self.rewards.feet_air_time_variance.weight = -4.0
        # Use body-frame swing height so stairs do not accidentally reward feet just
        # for being higher in the world while still in contact with a step.
        self.rewards.feet_height.weight = 0.0
        self.rewards.feet_height_body.weight = -0.4
        self.rewards.feet_height_body.params["target_height"] = -0.22
        self.rewards.feet_height_body.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_gait.weight = 0.5
        self.rewards.feet_slide.weight = -0.08
        self.rewards.stand_still.weight = -0.1

        self.curriculum.terrain_levels = None
        self.curriculum.command_levels = None
        self.curriculum.staged_locomotion = CurrTerm(func=mdp.staged_command_lift_stairs)

        # Remove terms that assume generated terrain or body-selection defaults that
        # are invalid for this custom USD scene.
        for term_name in ("terrain_out_of_bounds",):
            if hasattr(self.terminations, term_name):
                delattr(self.terminations, term_name)

        for term_name in ("base_height_l2", "body_lin_acc_l2"):
            if hasattr(self.rewards, term_name):
                delattr(self.rewards, term_name)

        # Subclasses do not inherit the parent class' zero-weight cleanup automatically.
        # Run it here so inactive reward terms do not get resolved during manager setup.
        self.disable_zero_weight_rewards()

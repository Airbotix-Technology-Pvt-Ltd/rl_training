from pathlib import Path

from isaaclab.assets import AssetBaseCfg
from isaaclab.sim import UsdFileCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

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

        self.curriculum.terrain_levels = None

        # Remove terms that assume generated terrain or body-selection defaults that
        # are invalid for this custom USD scene.
        for term_name in ("terrain_out_of_bounds",):
            if hasattr(self.terminations, term_name):
                delattr(self.terminations, term_name)

        for term_name in ("base_height_l2", "feet_height_body", "body_lin_acc_l2"):
            if hasattr(self.rewards, term_name):
                delattr(self.rewards, term_name)

        # Subclasses do not inherit the parent class' zero-weight cleanup automatically.
        # Run it here so inactive reward terms do not get resolved during manager setup.
        self.disable_zero_weight_rewards()

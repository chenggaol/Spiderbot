# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from SpiderBotTraining_1.robot.SpiderBotTraining_1 import MY_ROBOT_CFG

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
import math
from isaaclab.sensors import ContactSensor, ContactSensorCfg

@configclass
class SpiderbotSceneCfg(InteractiveSceneCfg):
    """Scene configuration — robot and sensors must live here."""

    robot: ArticulationCfg = MY_ROBOT_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )

    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*Lower",
        update_period=0.0,
        history_length=3,
        debug_vis=False,
    )


@configclass
class Spiderbottraining1EnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 10
    # - spaces definition
    action_space = 8 #one for each joint
    observation_space = 15 #gonna start off with position of each joint, in real life might need to be only imu data
    #gravity vector(3) + angular velocity(3) + previous actions(8) + heading direction(1)(converted from arrow keys)
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation,physx=sim_utils.PhysxCfg(
        gpu_max_rigid_patch_count=300298,  # exactly what the error asked for
    ))

    # scene
    scene: SpiderbotSceneCfg = SpiderbotSceneCfg(
        num_envs=4096,
        env_spacing=4.0,
        replicate_physics=True
    )

    

    # custom parameters/scales
    # - controllable joint
    joint_names = ["F_R_1","F_R_2","F_L_1","F_L_2","B_R_1","B_R_2","B_L_1","B_L_2"]

    # - action scale
    action_scale = 1  # [N]

    # - reward scales
    #positives
    rew_scale_alive = 0.05
    
    #moving right direction/heading alignment
    rew_scale_heading = 0.5
    #symetric movement
    #rew_scale_symmetric = 0.5

    #negatives
    #terminates when: base touches ground/direction certain degree off from desired direction/tipped over(pitch/roll angle too high)
    rew_scale_terminated = -2.0 
    #foot contact with ground(at least 2 on ground???)
    rew_scale_grounded = -0.1
    #actually moving in desired direction(0.5m/s ???, less reward for too fast and too slow)
    rew_scale_base_vel = -4.0
    
    # - reset states/conditions

    #torque greater then what servo can handle
    max_torque = 2.2 #Nm
    max_heading_deviation = math.radians(45)#radians
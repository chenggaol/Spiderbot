# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations


import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform,wrap_to_pi


from .spiderbottraining_1_env_cfg import Spiderbottraining1EnvCfg


class Spiderbottraining1Env(DirectRLEnv):
    cfg: Spiderbottraining1EnvCfg

    def __init__(self, cfg: Spiderbottraining1EnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._joint_dof_idx, _ = self.robot.find_joints(self.cfg.joint_names)

    def _setup_scene(self):
        # 1. add ground plane first
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        
        # 2. clone environments — this spawns everything defined in scene cfg
        self.scene.clone_environments(copy_from_source=False)
        
        # 3. filter collisions if needed
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        # 4. Get references after cloning
        self.robot= self.scene["robot"]
        self.contact_sensor = self.scene["contact_sensor"]

        # 5. lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # 6. initialise tensors
        self.actions= torch.zeros(self.num_envs, 8, device=self.device)
        self.target_vel_dir = torch.zeros(self.num_envs, device=self.device)
        self.heading_steps  = torch.zeros(self.num_envs, device=self.device)

    def _pre_physics_step(self, actions: torch.Tensor) -> None: #each step before you start physics?!
        self.actions = actions.clone()

        # tick timer and resample heading every 300 steps
        """ self.heading_steps += 1
        resample_ids = (self.heading_steps % 300 == 0).nonzero(as_tuple=False).flatten()
        if len(resample_ids) > 0:
            self.target_vel_dir[resample_ids] = (
                torch.rand(len(resample_ids), device=self.device) * 2 * torch.pi - torch.pi
            )
 """
    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self.actions * self.cfg.action_scale, joint_ids=self._joint_dof_idx)

    def _get_observations(self) -> dict:
        obs = torch.cat(
            (
                self.robot.data.projected_gravity_b,
                self.robot.data.root_ang_vel_b,
                self.actions, #previous joint positions
                self.target_vel_dir.unsqueeze(dim=1)
            ),
            dim=-1,
        )
        # debug — check which component has NaN
        if torch.any(torch.isnan(obs)):
            print("NaN in gravity:", torch.any(torch.isnan(self.robot.data.projected_gravity_b)))
            print("NaN in ang_vel:", torch.any(torch.isnan(self.robot.data.root_ang_vel_b)))
            print("NaN in actions:", torch.any(torch.isnan(self.actions)))
            print("NaN in heading:", torch.any(torch.isnan(self.target_vel_dir)))
        #clamp values between -10 and 10 to prevent vanish/explode grad
        obs = torch.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
        obs = torch.clamp(obs, -10.0, 10.0)
        observations = {"policy": obs}

        return observations

    def _get_rewards(self) -> torch.Tensor:
        rew_alive = self.cfg.rew_scale_alive*(1.0-self.reset_terminated.float())
        #reward for grounded
        foot_contact = self.contact_sensor.data.net_forces_w # (num_envs, 4, 3), last dimension is 3D force vector
        foot_contact_bool = torch.norm(foot_contact, dim=-1) > 1.0 # (num_envs, 4), only looks at z component of force and turns into bool if past 1N threshold
        num_feet_grounded = foot_contact_bool.sum(dim=1).float()

        foot_contact_reward = torch.where(
        num_feet_grounded >= 2,
        torch.zeros(self.num_envs, device=self.device),  # more feet down = more reward
        self.cfg.rew_scale_grounded * torch.ones(self.num_envs, device=self.device) #punish for less than 2 feet on 
        )

        #reward for symmetric gait, add later

        #reward for correct velocity direction
        vx = self.robot.data.root_lin_vel_w[:,0]
        vy = self.robot.data.root_lin_vel_w[:,1]
        speed = torch.sqrt(vx**2+vy**2)
        # add small epsilon to prevent atan2(0,0)
        # only matters when speed is near zero, negligible otherwise
        vx = torch.where(speed > 0.1, vx, torch.ones_like(vx))   # default to 1,0 when still
        vy = torch.where(speed > 0.1, vy, torch.zeros_like(vy))  # pointing world +X

        actual_vel_dir = torch.atan2(vy,vx)
        dir_error= wrap_to_pi(self.target_vel_dir - actual_vel_dir)
        dir_reward = (torch.pi - torch.abs(dir_error))* self.cfg.rew_scale_heading #could try exponential function

        #reward for base moving at correct speed
        #w = world frame, b = body/base_link frame
    
        velocity_reward = (0.5-speed)**2 * self.cfg.rew_scale_base_vel
        movement_reward = speed * 0.5

        total_reward = dir_reward + velocity_reward + foot_contact_reward + rew_alive + movement_reward
        #clamp reward as well to prevent exploding/vanish gradient
        return torch.clamp(total_reward, -5.0, 5.0)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        #robot off ground
        root_z = self.robot.data.root_pos_w[:,2] #,make sure more than 0.05
        fallen = root_z < 0.001

        #torque exceeds motor capability
        applied_torques = self.robot.data.applied_torque
        max_torque = torch.max(torch.abs(applied_torques), dim=1).values
        torque_exceeded = max_torque > 2.2

        terminate = fallen | torque_exceeded

        return terminate, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_pos += sample_uniform(
            -0.1, 0.1,
            joint_pos.shape,
            joint_pos.device,
        )
        joint_vel = self.robot.data.default_joint_vel[env_ids]

        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        
        self.target_vel_dir[env_ids] = torch.zeros_like(self.target_vel_dir[env_ids])
        
        """ num_envs_resetting = joint_pos.shape[0]
        self.target_vel_dir[env_ids] = (
            torch.rand(num_envs_resetting, device=self.device) * 2 * torch.pi - torch.pi
        )
        self.heading_steps[env_ids] = 0 """


""" @torch.jit.script
def compute_rewards(
    rew_scale_alive: float,
    rew_scale_terminated: float,
    rew_scale_pole_pos: float,
    rew_scale_cart_vel: float,
    rew_scale_pole_vel: float,
    pole_pos: torch.Tensor,
    pole_vel: torch.Tensor,
    cart_pos: torch.Tensor,
    cart_vel: torch.Tensor,
    reset_terminated: torch.Tensor,
):
    rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())
    rew_termination = rew_scale_terminated * reset_terminated.float()
    rew_pole_pos = rew_scale_pole_pos * torch.sum(torch.square(pole_pos).unsqueeze(dim=1), dim=-1)
    rew_cart_vel = rew_scale_cart_vel * torch.sum(torch.abs(cart_vel).unsqueeze(dim=1), dim=-1)
    rew_pole_vel = rew_scale_pole_vel * torch.sum(torch.abs(pole_vel).unsqueeze(dim=1), dim=-1)
    total_reward = rew_alive + rew_termination + rew_pole_pos + rew_cart_vel + rew_pole_vel
    return total_reward """

from __future__ import annotations

import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform, euler_xyz_from_quat

from .spiderbottraining_1_env_cfg import Spiderbottraining1EnvCfg


class Spiderbottraining1Env(DirectRLEnv):
    cfg: Spiderbottraining1EnvCfg

    def __init__(self, cfg: Spiderbottraining1EnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._joint_dof_idx, found_names = self.robot.find_joints(self.cfg.joint_names)
        print(f"[SPIDER] Joint indices: {self._joint_dof_idx}")
        print(f"[SPIDER] Joint names found: {found_names}")
        print(f"[SPIDER] All joints: {self.robot.data.joint_names}")

    def _setup_scene(self):
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        self.robot          = self.scene["robot"]
        self.contact_sensor = self.scene["contact_sensor"]

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        self.prev_actions = torch.zeros(self.num_envs, 8, device=self.device)
        self.actions      = torch.zeros(self.num_envs, 8, device=self.device)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.prev_actions = self.actions.clone()
        self.actions      = actions.clone()

    def _apply_action(self) -> None:
        targets = self.actions * self.cfg.action_scale

        if self.episode_length_buf[0] % 300 == 0:
            jp = self.robot.data.joint_pos[0, self._joint_dof_idx].cpu().numpy().round(3)
            vx = self.robot.data.root_lin_vel_w[0, 0].item()
            print(f"[SPIDER] targets[0]={targets[0].cpu().numpy().round(3)}")
            print(f"[SPIDER] joint_pos[0]={jp}")
            print(f"[SPIDER] vx[0]={vx:.4f}")

        self.robot.set_joint_effort_target(targets, joint_ids=self._joint_dof_idx)

    def _get_observations(self) -> dict:
        gravity   = self.robot.data.projected_gravity_b
        ang_vel   = self.robot.data.root_ang_vel_b
        joint_pos = self.robot.data.joint_pos[:, self._joint_dof_idx]
        prev_act  = self.prev_actions

        obs = torch.cat((gravity, ang_vel, joint_pos, prev_act), dim=-1)

        if torch.any(torch.isnan(obs)):
            print("[SPIDER] NaN in obs — gravity:", torch.any(torch.isnan(gravity)),
                  "ang_vel:", torch.any(torch.isnan(ang_vel)),
                  "joint_pos:", torch.any(torch.isnan(joint_pos)))
            nan_envs = torch.any(torch.isnan(obs), dim=1).nonzero(as_tuple=False).flatten()
            self._reset_idx(nan_envs.tolist())
            gravity   = self.robot.data.projected_gravity_b
            ang_vel   = self.robot.data.root_ang_vel_b
            joint_pos = self.robot.data.joint_pos[:, self._joint_dof_idx]
            prev_act  = self.prev_actions
            obs       = torch.cat((gravity, ang_vel, joint_pos, prev_act), dim=-1)

        obs = torch.nan_to_num(obs, nan=0.0, posinf=5.0, neginf=-5.0)
        obs = torch.clamp(obs, -5.0, 5.0)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        vx = self.robot.data.root_lin_vel_w[:, 0]
        speed = torch.norm(self.robot.data.root_lin_vel_w[:, :2], dim=1)

        # reward forward motion
        rew_forward = torch.clamp(vx, min=0.0) * self.cfg.rew_forward

        # heavy penalty for standing still
        rew_not_moving = torch.where(
            speed < 0.1, #increase so lower speed still penalized
            torch.full_like(speed, self.cfg.rew_not_moving),
            torch.zeros_like(speed),
        )

        # penalise tilt
        roll, pitch, _ = euler_xyz_from_quat(self.robot.data.root_quat_w)
        tilt            = torch.sqrt(roll**2 + pitch**2)
        rew_orientation = self.cfg.rew_orientation * tilt

        # reward joint motion to break zero equilibrium
        joint_vel        = self.robot.data.joint_vel[:, self._joint_dof_idx]
        rew_joint_motion = torch.sum(torch.abs(joint_vel), dim=1) * self.cfg.rew_joint_motion

        total = rew_forward + rew_not_moving + rew_orientation + rew_joint_motion
        return torch.clamp(total, -10.0, 10.0)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        roll, pitch, _ = euler_xyz_from_quat(self.robot.data.root_quat_w)
        fallen = (torch.abs(roll) > self.cfg.max_tilt) | (torch.abs(pitch) > self.cfg.max_tilt)

        applied_torques = self.robot.data.applied_torque
        max_torque      = torch.max(torch.abs(applied_torques), dim=1).values
        torque_exceeded = max_torque > self.cfg.max_torque

        return fallen | torque_exceeded, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_pos += sample_uniform(-0.02, 0.02, joint_pos.shape, joint_pos.device)
        joint_vel = self.robot.data.default_joint_vel[env_ids]

        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self.prev_actions[env_ids] = 0.0
        self.actions[env_ids]      = 0.0
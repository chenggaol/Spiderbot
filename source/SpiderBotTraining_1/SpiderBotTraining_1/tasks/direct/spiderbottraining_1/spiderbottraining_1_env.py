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
        self._joint_dof_idx, _ = self.robot.find_joints(self.cfg.joint_names)

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

        # track starting position each episode
        self.last_pos_y  = torch.zeros(self.num_envs, device=self.device)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.prev_actions = self.actions.clone()
        self.actions      = actions.clone()

    def _apply_action(self) -> None:
        target_pos  = self.actions * self.cfg.action_scale

        self.robot.set_joint_position_target(target_pos, joint_ids=self._joint_dof_idx)

    def _get_observations(self) -> dict:
        # all position/velocity values converted to cm/cm·s⁻¹
        # for stronger gradient signal on small robot

        # p_body in cm (3)
        p_body_cm = self.robot.data.root_pos_w * 100.0

        # v_body in cm/s (3)
        v_body_cms = self.robot.data.root_lin_vel_w * 100.0

        # orientation — roll/pitch/yaw in radians (3)
        roll, pitch, yaw = euler_xyz_from_quat(self.robot.data.root_quat_w)
        orientation = torch.stack([roll, pitch, yaw], dim=1)

        # angular velocity in rad/s (3)
        ang_vel = self.robot.data.root_ang_vel_b

        # joint positions in radians (8)
        joint_pos = self.robot.data.joint_pos[:, self._joint_dof_idx]

        # joint velocities in rad/s (8)
        joint_vel = self.robot.data.joint_vel[:, self._joint_dof_idx]

        # contact normal forces — z component per foot (4)
        contact_forces   = self.contact_sensor.data.net_forces_w
        contact_normal   = contact_forces[:, :, 2]

        # contact friction forces — xy magnitude per foot (4)
        contact_friction = torch.norm(contact_forces[:, :, :2], dim=-1)

        # previous actions (8)
        prev_act = self.prev_actions

        obs = torch.cat((
            p_body_cm,        # 3  — in cm
            v_body_cms,       # 3  — in cm/s
            orientation,      # 3  — radians
            ang_vel,          # 3  — rad/s
            joint_pos,        # 8  — radians
            joint_vel,        # 8  — rad/s
            contact_normal,   # 4  — Newtons
            contact_friction, # 4  — Newtons
            prev_act,         # 8  — normalized actions
        ), dim=-1)            # = 44 total

        # catch NaN before it enters rollout buffer
        if torch.any(torch.isnan(obs)):
            print("[SPIDER] NaN detected — resetting affected envs")
            nan_envs = torch.any(torch.isnan(obs), dim=1).nonzero(as_tuple=False).flatten()
            self._reset_idx(nan_envs.tolist())
            # re-fetch after reset
            p_body_cm        = self.robot.data.root_pos_w * 100.0
            v_body_cms       = self.robot.data.root_lin_vel_w * 100.0
            roll, pitch, yaw = euler_xyz_from_quat(self.robot.data.root_quat_w)
            orientation      = torch.stack([roll, pitch, yaw], dim=1)
            ang_vel          = self.robot.data.root_ang_vel_b
            joint_pos        = self.robot.data.joint_pos[:, self._joint_dof_idx]
            joint_vel        = self.robot.data.joint_vel[:, self._joint_dof_idx]
            contact_forces   = self.contact_sensor.data.net_forces_w
            contact_normal   = contact_forces[:, :, 2]
            contact_friction = torch.norm(contact_forces[:, :, :2], dim=-1)
            prev_act         = self.prev_actions
            obs = torch.cat((
                p_body_cm, v_body_cms, orientation, ang_vel,
                joint_pos, joint_vel, contact_normal, contact_friction, prev_act
            ), dim=-1)

        obs = torch.nan_to_num(obs, nan=0.0, posinf=50.0, neginf=-50.0)
        obs = torch.clamp(obs, -50.0, 50.0)  # wider clamp for cm values
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        # paper equation (13) — all scaled to cm units for stronger gradients
        # r = λvx*vx_cm + λΔt*(Ts/Tf) - λz*(pz_cm - pz_dh)² - λu*Σ(u²)

        #reward displacement not velocity
        # reward actual displacement from last step — not instantaneous velocity
        current_y = self.robot.data.root_pos_w[:, 0]
        displacement = current_y - self.last_pos_y
        self.last_pos_y = current_y.clone()
        r_displacement = torch.clamp(displacement, min=0.0) * 100

        # forward velocity in cm/s
        vy_cms = self.robot.data.root_lin_vel_w[:, 0] * 100.0
        #r_vy   = self.cfg.lambda_vy * vy_cms   # 3.0 * vx_cms
        r_vy = torch.where(vy_cms>5.0,self.cfg.lambda_vy * vy_cms, torch.zeros_like(vy_cms))
        # at 1 cm/s: r_vy = 3.0 — much stronger than before

        # alive reward — constant per step
        #r_alive = self.cfg.lambda_dt * (self.cfg.Ts / self.cfg.Tf)  # 0.002

        # height penalty in cm — squared deviation from desired height
        pz_cm    = self.robot.data.root_pos_w[:, 2] * 100.0
        r_height = -self.cfg.lambda_z * (pz_cm - self.cfg.pz_desired_cm) ** 2
        # at pz=3cm, desired=3cm: r_height = 0 (perfect)
        # at pz=1cm (tipping): r_height = -0.005 * 4 = -0.02

        # torque efficiency penalty
        torque_sq = torch.sum(self.prev_actions ** 2, dim=1)
        r_torque  = -self.cfg.lambda_u * torque_sq

        """  # flying penalty — triggers above 15cm
        r_flying = torch.where(
            pz_cm > 7.0,
            torch.full_like(r_vx, -5.0),
            torch.zeros_like(r_vx)
        )

        # tilt penalty
        roll, pitch, _ = euler_xyz_from_quat(self.robot.data.root_quat_w)
        tilt    = torch.sqrt(roll**2 + pitch**2)
        r_tilt  = -1.0 * tilt   # grows as robot tips over
 """    

        total = r_vy + r_height + r_torque #+ r_flying + r_tilt
        # debug print after total is computed
        if self.episode_length_buf[0] % 500 == 0:
            print(f"[REWARD] vy_cms={vy_cms[0]:.3f} r_vy={r_vy[0]:.3f} "
                f"r_disp={r_displacement[0]:.4f} "
                f"pz_cm={pz_cm[0]:.3f} r_height={r_height[0]:.4f} "
                f"r_torque={r_torque[0]:.4f} "
                f"total={total[0]:.3f}")

        return total

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # height termination — effectively disabled since origin on bottom face
        pz = self.robot.data.root_pos_w[:, 2]
        fallen_height = pz < self.cfg.min_height

        flying_height = pz > self.cfg.max_height

        # tilt termination — very loose early in training
        roll, pitch, yaw = euler_xyz_from_quat(self.robot.data.root_quat_w)
        fallen_tilt = (
            (torch.abs(roll)  > self.cfg.max_tilt) |
            (torch.abs(pitch) > self.cfg.max_tilt) |
            (torch.abs(yaw)   > self.cfg.max_tilt)
        )

        
        terminate = fallen_height | flying_height

        # debug — show what's terminating

        if torch.any(terminate):
            term_idx = terminate.nonzero(as_tuple=False).flatten()[0]
            print(f"[DONE] step={self.episode_length_buf[term_idx].item()} "
                f"pz={pz[term_idx]:.4f} "
                f"height={fallen_height[term_idx]} "
                f"height={flying_height[term_idx]} "
                f"tilt={fallen_tilt[term_idx]} "
                f"roll={roll[term_idx]:.2f} "
                f"pitch={pitch[term_idx]:.2f} "
                f"yaw={yaw[term_idx]:.2f}")
            print(f"Max length:{self.max_episode_length}")

        return terminate, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_pos += sample_uniform(-0.3, 0.3, joint_pos.shape, joint_pos.device)
        joint_vel  = self.robot.data.default_joint_vel[env_ids]

        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self.prev_actions[env_ids] = 0.0
        self.actions[env_ids]      = 0.0

        # reset the position of resetting environments what they are starting at
        self.last_pos_y[env_ids]  = self.robot.data.root_pos_w[env_ids, 1]
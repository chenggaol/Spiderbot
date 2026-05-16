from SpiderBotTraining_1.robot.SpiderBotTraining_1 import MY_ROBOT_CFG

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from isaaclab.sensors import ContactSensorCfg


@configclass
class SpiderbotSceneCfg(InteractiveSceneCfg):
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
    decimation       = 4        # 30Hz policy
    episode_length_s = 13.0     # ~400 policy steps per episode

    # observation space (all in cm/cm/s where applicable):
    # p_body(3) + v_body(3) + orientation(3) + ang_vel(3) +
    # joint_pos(8) + joint_vel(8) + contact_normal(4) + contact_friction(4) + prev_actions(8)
    # = 44
    action_space      = 8
    observation_space = 44
    state_space       = 0

    sim: SimulationCfg = SimulationCfg(
        dt=1/120,
        render_interval=decimation,
        physx=sim_utils.PhysxCfg(
            gpu_max_rigid_patch_count=500000,
        )
    )

    scene: SpiderbotSceneCfg = SpiderbotSceneCfg(
        num_envs=4096,
        env_spacing=4.0,
        replicate_physics=True,
    )

    joint_names  = ["F_R_1", "F_R_2", "F_L_1", "F_L_2", "B_R_1", "B_R_2", "B_L_1", "B_L_2"]
    action_scale = 1.5   # radians — PD controller converts to torques

    # PD controller gains
    kp = 1.0
    kd = 0.1
    max_torque = 5.0  # Nm clamp on PD output

    # reward weights from paper (equation 13) — scaled for cm units
    # r = λvx*vx_cm + λΔt*(Ts/Tf) - λz*(pz_cm - pz_dh_cm)² - λu*Σ(u²)
    lambda_vy  =  3.0    # forward velocity reward (vx in cm/s)
    lambda_dt  =  1.0    # alive reward weight
    lambda_z   =  0.005  # height penalty weight (reduced since pz in cm → larger values)
    lambda_u   =  0.01   # torque penalty weight (reduced — torques already in Nm)

    # paper timing values
    Ts = 0.02    # sampling time
    Tf = 10.0    # episode time for alive reward scaling

    # desired height in cm — robot stands at ~3cm
    pz_desired_cm = 1.25   # cm

    # termination — tilt was the problem so start very loose
    max_tilt   = 2.5      # ~143 degrees — essentially never terminates on tilt early in training
    max_height = 0.3
    min_height = 0.005   # effectively disabled — robot origin on bottom face near 0
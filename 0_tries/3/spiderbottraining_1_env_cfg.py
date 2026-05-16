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
    decimation = 20 #basically your frequency 1 = 120 hz, 4 = 30hz, 120 = 1hz
    episode_length_s = 30.0

    # 3 gravity + 3 ang_vel + 8 joint_pos + 8 prev_actions = 22
    action_space = 8
    observation_space = 22
    state_space  = 0

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
    action_scale = 0.3 # radians, max position target
    
    # PD controller gains
    kp = 10.0  # proportional gain
    kd = 1.0   # derivative gain (damping)

    rew_forward = 3.0
    rew_alive  =  0.0
    rew_orientation = -1.0
    rew_not_moving = -2.0
    rew_joint_motion =  0.0
    rew_not_grounded = -3.0 # punishment for all legs off ground
    rew_scale_torque_exceeded = -1.0
    rew_scale_heading = -0.3
    rew_scale_flying = -0.1
    
    velocity_sigma= 0.25 #allowed deviation 
    max_torque = 10 #maybe later add 2.2 as max servo torque
    max_tilt = 3.14  # 60 degrees
    #target speed = 3cm/sec
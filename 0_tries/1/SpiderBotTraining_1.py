import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

MY_ROBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="D:/05_Robots/SpiderBot/UpdatedURDF/Updated.usd",  # absolute path anywhere on disk
        activate_contact_sensors = True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=50.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=16,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.07), # x, y, z spawn position, a little off the ground so it doesnt start underground
        joint_pos={
            "F_R_1":0.0,
            "F_R_2":0.0,
            "F_L_1":0.0,
            "F_L_2":0.0,
            "B_R_1":0.0,
            "B_R_2":0.0,
            "B_L_1":0.0,
            "B_L_2":0.0,
        }, #KNOW THE DIFFERENT BETWEEN LINKS AND JOINTS, ONES ON GUI ARE PHYSICAL PARTS(ie L1_1,R1_1) THESE ARE RELATIONS BETWEEN THEM
        joint_vel={".*": 0.0},
    ),
    actuators = {"all_joints": ImplicitActuatorCfg(
        joint_names_expr=[".*"], #all joints,
        damping = 2.0,
        stiffness = 40,
    )},
)
from typing import Any, Tuple

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from mani_skill.agents.multi_agent import MultiAgent
from mani_skill.agents.robots.panda import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils.randomization.pose import random_quaternions
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table_rocobench import RocoTableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig


@register_env("rocobench_test", max_episode_steps=100)
class RocobenchTest(BaseEnv):
    """
    **Task Description:**
    A collaborative task where two robot arms need to work together to stack two cubes. One robot must pick up the green cube and place it on the target region, while the other robot picks up the blue cube and stacks it on top of the green cube.

    The cubes are initially positioned such that each robot can only reach one cube - the green cube is near the right robot and the blue cube is near the left robot. This requires coordination between the robots to complete the stacking task.

    **Randomizations:**
    - Both cubes have random rotations around their z-axis
    - The xy positions of both cubes on the table are randomized, while ensuring:
        - The cubes do not collide with each other
        - The green cube remains reachable by the right robot
        - The blue cube remains reachable by the left robot
    - The goal region is placed along the midline between the robots (y=0), with randomized x position

    **Success Conditions:**
    - The blue cube is stacked on top of the green cube (within half a cube size)
    - The green cube is placed on the red/white target region
    - Both cubes are released by the robots (not being grasped)

    """

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/TwoRobotStackCube-v1_rt.mp4"
    SUPPORTED_ROBOTS = [("panda_wristcam", "panda_wristcam")]
    agent: MultiAgent[Tuple[Panda, Panda]]

    goal_radius = 0.06

    def __init__(
        self,
        *args,
        robot_uids=("panda_wristcam", "panda_wristcam"),
        robot_init_qpos_noise=0.02,
        **kwargs
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25,
                max_rigid_patch_count=2**19,
                max_rigid_contact_count=2**21,
            )
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0, 0, 1], target=[-0.1, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    # controls where the camera is spawned for human rendering
    # first number is x perpendicular to long edge of table, y then z
    @property
    def _default_human_render_camera_configs(self):
        # pose = sapien_utils.look_at([1.4, 0.8, 0.75], [0.0, 0.1, 0.1]) # this perspective is good for demos
        pose = sapien_utils.look_at(eye=[0, 1.5, 1.5], target=[0, 0, 0])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        #print("Agents getting loaded with rocobenchtest")
        #print(options)
        super()._load_agent(
            options, [sapien.Pose(p=[1, -1, 0]), sapien.Pose(p=[0, 1, 0])]
        )

    def _load_scene(self, options: dict):
        self.cube_half_size = common.to_tensor([0.02] * 3, device=self.device)
        self.table_scene = RocoTableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.cubeA = actors.build_cube(
            self.scene,
            half_size=0.02,
            color=np.array([12, 42, 160, 255]) / 255,
            name="cubeA",
            initial_pose=sapien.Pose(p=[1, 0, 0.02]),
        )
        self.cubeB = actors.build_cube(
            self.scene,
            half_size=0.02,
            color=[0, 1, 0, 1],
            name="cubeB",
            initial_pose=sapien.Pose(p=[-1, 0, 0.02]),
        )
        self.goal_region = [actors.build_red_white_target(
            self.scene,
            radius=self.goal_radius,
            thickness=1e-5,
            name="goal_region_a",
            add_collision=False,
            body_type="kinematic",
            initial_pose=sapien.Pose(),
        ), actors.build_red_white_target(
            self.scene,
            radius=self.goal_radius,
            thickness=1e-5,
            name="goal_region_b",
            add_collision=False,
            body_type="kinematic",
            initial_pose=sapien.Pose(),
        )]

    # this is whats actually being run by gym.make i think
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            # the table scene initializes two robots. the first one self.agents[0] is on the left and the second one is on the right
            #print("initializting with options: ", options)
            #self._load_agent(options)
            torch.zeros((b, 3))
            torch.rand((b, 2)) * 0.2
            #cubeA is blue, cubeB is green
            cubeA_xyz = torch.zeros((b, 3))
            cubeA_xyz[:, 0] = torch.rand((b,)) * 0.1 - 0.05
            cubeA_xyz[:, 1] = 0.5 - torch.rand((b,)) * 0.1 + 0.05
            cubeB_xyz = torch.zeros((b, 3))
            cubeB_xyz[:, 0] = torch.rand((b,)) * 0.1 - 0.05
            cubeB_xyz[:, 1] = -0.5 + torch.rand((b,)) * 0.1 - 0.05
            cubeA_xyz[:, 2] = 0.02
            cubeB_xyz[:, 2] = 0.02

            qs = random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                lock_z=False,
            )
            self.cubeA.set_pose(Pose.create_from_pq(p=cubeA_xyz, q=qs))

            qs = random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                lock_z=False,
            )
            self.cubeB.set_pose(Pose.create_from_pq(p=cubeB_xyz, q=qs))
            
            target_region_a_xyz = torch.zeros((b, 3))
            target_region_a_xyz[:, 0] = torch.rand((b,)) * 0.1 + 0.15
            target_region_a_xyz[:, 1] = .5
            target_region_a_xyz[..., 2] = 1e-3
            self.goal_region[0].set_pose(
                Pose.create_from_pq(
                    p=target_region_a_xyz,
                    q=euler2quat(0, np.pi / 2, 0),
                )
            )

            target_region_b_xyz = torch.zeros((b, 3))
            target_region_b_xyz[:, 0] = torch.rand((b,)) * 0.1 - 0.15
            target_region_b_xyz[:, 1] = -0.5
            target_region_b_xyz[..., 2] = 1e-3
            self.goal_region[1].set_pose(
                Pose.create_from_pq(
                    p=target_region_b_xyz,
                    q=euler2quat(0, np.pi / 2, 0),
                )
            )
            

    # the robot that is next to goal region b, formerly left_agent
    @property
    def agent_b(self) -> Panda:
        return self.agent.agents[0]

    @property
    def agent_a(self) -> Panda:
        return self.agent.agents[1]

    def evaluate(self):
        pos_A = self.cubeA.pose.p
        pos_B = self.cubeB.pose.p
        cubeB_to_goalB_dist = torch.linalg.norm(
            self.cubeB.pose.p[:, :2] - self.goal_region[1].pose.p[..., :2], axis=1
        )
        cubeA_to_goalA_dist = torch.linalg.norm(
            self.cubeA.pose.p[:, :2] - self.goal_region[0].pose.p[..., :2], axis=1
        )
        cubeB_in_goal = cubeB_to_goalB_dist < self.goal_radius
        cubeA_in_goal = cubeA_to_goalA_dist < self.goal_radius
        is_agentB_grasping_cubeA = self.agent_b.is_grasping(self.cubeA)
        is_agentB_grasping_cubeB = self.agent_b.is_grasping(self.cubeB)
        is_agentA_grasping_cubeA = self.agent_a.is_grasping(self.cubeA)
        is_agentA_grasping_cubeB = self.agent_a.is_grasping(self.cubeB)
        success = (
            cubeB_in_goal * cubeA_in_goal
        )
        return {
            "is_agentB_grasping_cubeA": is_agentB_grasping_cubeA,
            "is_agentA_grasping_cubeB": is_agentA_grasping_cubeB,
            "is_agentB_grasping_cubeB": is_agentB_grasping_cubeB,
            "is_agentA_grasping_cubeA": is_agentA_grasping_cubeA,
            "is_cubeA_grasped": is_agentA_grasping_cubeA or is_agentB_grasping_cubeA,
            "is_cubeB_grasped": is_agentA_grasping_cubeB or is_agentB_grasping_cubeB,
            "cubeA_in_goal": cubeA_in_goal,
            "cubeB_in_goal": cubeB_in_goal,
            "success": success.bool(),
        }

    def _get_obs_extra(self, info: dict):
        obs = dict(
            arm_b_tcp=self.agent_b.tcp.pose.raw_pose,
            arm_a_tcp=self.agent_a.tcp.pose.raw_pose,
        )
        if "state" in self.obs_mode:
            obs.update(
                goal_region_pos=[self.goal_region[0].pose.p, self.goal_region[1].pose.p],
                cubeA_pose=self.cubeA.pose.raw_pose,
                cubeB_pose=self.cubeB.pose.raw_pose,
                arm_b_tcp_to_cubeA_pos=self.cubeA.pose.p
                - self.agent_b.tcp.pose.p,
                arm_a_tcp_to_cubeB_pos=self.cubeB.pose.p
                - self.agent_a.tcp.pose.p
            )
        return obs

    # info is ultimately going to be what you return in evaluate and elapsed steps
    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        # Stage 1: opposite agents reach and grasp
        # reward for the opposite robot reaching for its opposite cube
        cubeA_to_arm_b_tcp_dist = torch.linalg.norm(
            self.agent_b.tcp.pose.p - self.cubeA.pose.p, axis=1
        )

        cubeB_to_arm_a_tcp_dist = torch.linalg.norm(
            self.agent_a.tcp.pose.p - self.cubeB.pose.p, axis=1
        )

        reach_reward = (
            1
            - torch.tanh(5 * cubeA_to_arm_b_tcp_dist)
            + 1
            - torch.tanh(5 * cubeB_to_arm_a_tcp_dist)
        ) / 2

        cubeA_pos = self.cubeA.pose.p
        cubeB_pos = self.cubeB.pose.p
        # might have to turn the 3 into 2 im not sure
        reward = (reach_reward + info["is_agentA_grasping_cubeB"] + info["is_agentB_grasping_cubeA"]) / 3

        # pass condition for stage 1
        place_stage_reached = info["is_agentA_grasping_cubeB"] and info["is_agentA_grasping_cubeB"]

        # Stage 2: Place cubes in spot where they can be grabbed by their respective robot
        cubeB_to_arm_b_tcp_dist = torch.linalg.norm(
            self.agent_b.tcp.pose.p - self.cubeB.pose.p, axis=1
        )

        cubeA_to_arm_a_tcp_dist = torch.linalg.norm(
            self.agent_a.tcp.pose.p - self.cubeA.pose.p, axis=1
        )
        
        reach_stage_2_reward = (
            1
            - torch.tanh(5 * cubeA_to_arm_a_tcp_dist)
            + 1
            - torch.tanh(5 * cubeB_to_arm_b_tcp_dist)
        ) / 2
        stage_2_reward = reach_stage_2_reward + info["is_agentB_grasping_cubeB"] + info["is_agentA_grasping_cubeA"]

        # updates only those envs with place_stage_reached = true
        reward[place_stage_reached] = 2 + stage_2_reward[place_stage_reached] / 2

        # pass condition for stage 2
        cubes_grasped_by_right_agents = info["is_agentB_grasping_cubeB"] * info["is_agentA_grasping_cubeA"]

        # Stage 3: Place each cube in its respective target area

        cubeB_to_goalB_dist = torch.linalg.norm(
            cubeB_pos[:, :2] - self.goal_region[1].pose.p[..., :2], axis=1
        )

        cubeA_to_goalA_dist = torch.linalg.norm(
            cubeA_pos[:, :2] - self.goal_region[0].pose.p[..., :2], axis=1
        )

        reaching_for_goal_reward = (
            1
            - torch.tanh(5 * cubeA_to_goalA_dist)
            + 1
            - torch.tanh(5 * cubeB_to_goalB_dist)
        ) / 2

        stage_3_reward = reaching_for_goal_reward * info["cubeB_in_goal"] * info["cubeB_in_goal"]

        reward[cubes_grasped_by_right_agents] = (
            4 + stage_3_reward[cubes_grasped_by_right_agents]
        )

        cubes_in_goals = info["cubeB_in_goal"] * info["cubeB_in_goal"]
        # Stage 3: Place top cube while moving right arm away to give left arm space
        # place reward for top cube (cube A)
        
        
        # Stage 4: get both robots to stop grasping
        gripper_width = (self.agent_b.robot.get_qlimits()[0, -1, 1] * 2).to(
            self.device
        )  # NOTE: hard-coded with panda
        ungrasp_reward_b = (
            torch.sum(self.agent_b.robot.get_qpos()[:, -2:], axis=1) / gripper_width
        )
        ungrasp_reward_b[~info["is_cubeB_grasped"]] = 1.0
        ungrasp_reward_a = (
            torch.sum(self.agent_a.robot.get_qpos()[:, -2:], axis=1) / gripper_width
        )
        ungrasp_reward_a[~info["is_cubeA_grasped"]] = 1.0

        reward[cubes_in_goals] = (
            8 + (ungrasp_reward_a + ungrasp_reward_b)[cubes_in_goals] / 2
        )

        reward[info["success"]] = 10

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 10

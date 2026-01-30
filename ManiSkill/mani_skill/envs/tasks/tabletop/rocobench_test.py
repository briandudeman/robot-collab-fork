from typing import Any, Tuple

import numpy as np
import gymnasium as gym
import sapien
import torch
from transforms3d.euler import euler2quat

from mani_skill.agents.multi_agent import MultiAgent
from mani_skill.agents.robots.panda import PandaWristCam
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils.randomization.pose import random_quaternions
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table_rocobench.scene_builder import RocoTableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig


ACTION_SPACE="""
[Action Options]
1) PICK <obj> PATH <path>: only PICK if your gripper is empty;
2) PLACE <obj> PATH <path>: only if you have already PICKed the object, you can PLACE it, do NOT PLACE if another object is already in the position this object is going to be PLACEd in!

Each <path> must contain exactly four <coord>s that smoothly interpolate between start and goal, coordinates must be evenly distanced from each other.
The robot PATHs must efficiently reach target while avoiding collision avoid collision (e.g. move above the objects' heights).
The PATHs must do top-down pick or place: 
- move directly atop an object by height 0.2 before PICK: e.g. agentA's gripper is at (0, 0, 0.3), cubeB is at (-0.25, 0.39, 0.29): NAME agentA ACTION PICK cubeB PATH [(0, 0.1, 0.3),(0, 0.2, 0.49),(-0.1, 0.25, 0.49),(-0.25, 0.39, 0.49)]
- lift an object vertically up before moving it to PLACE: e.g. agentB's gripper is at (0.9, 0, 0.2), end_pos is at (0.35, 0.35, 0.43): NAME agentB ACTION PLACE cubeA end_pos PATH [(0.9,0.0,0.5), (0.5, 0, 0.5), (0.2, 0.1, 0.5),(0.35, 0.35, 0.5)]

[Action Output Instruction]
First output 'EXECUTE\n', then give exactly one ACTION per robot, each on a new line.
Example: 'EXECUTE\nNAME agentA ACTION PICK cubeA PATH <path>\nNAME agentB ACTION PLACE cubeB end_pos PATH <path>\n'
"""

@register_env("RocobenchTest", max_episode_steps=100)
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
    agent: MultiAgent[Tuple[PandaWristCam, PandaWristCam]]

    goal_radius = 0.06

    def __init__(
        self,
        *args,
        robot_uids=("panda_wristcam", "panda_wristcam"),
        robot_init_qpos_noise=0.02,
        **kwargs
    ):
        print("rocobench test being initialized")
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

    def get_action_prompt(self) -> str:
        return ACTION_SPACE

    # only works for this specific environment frank emika panda robots, super icky but whatever
    def cube_in_range(self, obs, agent:str, cube:str):
        if (agent == "agentA"):
            print(self.agentA.robot.pose.get_p())
            if (torch.cdist(obs["extra"][f"{cube}_pose"][0:1, 0:3], self.agentA.robot.pose.get_p()) < .82):
                return True
            return False
        elif (agent == "agentB"):
            if (torch.cdist(obs["extra"][f"{cube}_pose"][0:1, 0:3], self.agentB.robot.pose.get_p()) < .82):
                return True
            return False


    def describe_cube_states(self, obs):
        return f"""(Is agentA grasping cubeA: {obs["extra"]["is_agentA_grasping_cubeA"][0]})
                    (Is agentA grasping cubeB: {obs["extra"]["is_agentA_grasping_cubeB"][0]})
                    (Is agentB grasping cubeA: {obs["extra"]["is_agentB_grasping_cubeA"][0]})
                    (Is agentB grasping cubeB: {obs["extra"]["is_agentB_grasping_cubeB"][0]})
                    (Is cubeA grasped: {obs["extra"]["is_cubeA_grasped"][0]})
                    (Is cubeB grasped: {obs["extra"]["is_cubeB_grasped"][0]})
                    (Is cubeB in goal: {obs["extra"]["cubeB_in_goal"][0]})
                    (Is cubeA in goal: {obs["extra"]["cubeA_in_goal"][0]})
                        """

    def get_agent_prompt(self, obs, agent_name:str, include_response_instructions=True):
        
        agent_property = getattr(self, agent_name)

        target_locations = 'cubeB_goal_region_pos: (' + ", ".join([str(i) for i in obs['extra']['cubeB_goal_region_pos'].tolist()[0]]) + '), cubeA_goal_region_pos: (' + ", ".join([str(i) for i in obs['extra']['cubeB_goal_region_pos'].tolist()[0]]) + ")"

        cube_states = self.describe_cube_states(obs)

        graspable_list = []

        if (self.cube_in_range(obs, agent_name, "cubeA")):
            graspable_list.append("cubeA")
        
        if (self.cube_in_range(obs, agent_name, "cubeB")):
            graspable_list.append("cubeB")

        graspables = ", ".join(graspable_list)

        in_hand = ""
        if (obs["extra"][f"is_{agent_name}_grasping_cubeA"][0]):
            in_hand = ", grasping cubeA"
        elif ((obs["extra"][f"is_{agent_name}_grasping_cubeB"][0])):
            in_hand = ", grasping cubeB"
        else:
            in_hand = ", grasping nothing"
             

        agent_state = f'Your gripper: {(agent_property.robot.links_map["panda_hand_tcp"].pose.raw_pose)[:, 0:3].tolist()[0]}' + in_hand

        closest_cube = "cubeA"
        other_cube = "cubeB"

        closest_target = "targetB"
        other_target = "targetA"

        other_agent = "agentA"
        if (agent_name == "agentA"):
            other_agent = "agentB"

            closest_cube = "cubeB"
            other_cube = "cubeA"
            
            closest_target = "targetA"
            other_target = "targetB"


        agent_prompt = f"""
        `There are 2 cubes and 2 targets on the table. Each cube is close to the other cube's respective target, and each group of target-cube is infront of a robot arm.
        You are robot {agent_name} and on the other side of the table is {other_agent}, who you are collaborating with to move both cubes to their respective targets. The task is NOT done until all two cubes are sorted.
        Locations of the targets:
        {target_locations}
        At current round: 
        {cube_states}
        Your goal is to place {other_cube} on {closest_target}, but the only cube(s) in distance are/is {graspables}
        {agent_state}
        Never forget you are {agent_name}! Never forget you can only reach {graspables}!
        Think step-by-step about the task and others' response. Carefully check and correct them if they made a mistake. 
        Improve your plans if given [Environment Feedback].
        """
        if include_response_instructions:
            agent_prompt += f"""
        When you respond, tell others about your goal and all constraints. Respond very concisely but informatively, and do not repeat what others have said.
        Discuss with others to come up with the best plan, e.g. if your cube is out of your reach, ask others for help, and you can do the same for them. 
        Propose exactly one action for yourself at the **current** round, select from [Action Options].
        End your response by either: 1) output PROCEED, if the plans require further discussion, or 2) If everyone has made proposals and got approved, output EXECUTE and the final plan, must strictly follow [Action Output Instruction]!
        In the plan, at least one robot should be acting, you can't all WAIT.
        """
        # Example response #1:
        # [Reasons] I am {agent_name}, I must put blue_square on panel2, but I can't reach blue_square for now. Since Chad needs yellow_trapezoid, I propose to help Chad move it closer. What does everyone think?
        # [Proposal] PICK yellow_trapezoid PLACE panel3
        # [Decision] PROCEED
        # Example response #2:
        # [Reasons] I am Chad, My previous proposal was approved and no need for update. I approve the latest proposals from Alice and Bob.
        # [Proposal] WAIT 
        # [Decision] 
        # EXECUTE\nNAME Alice ACTION WAIT\nNAME Bob ACTION PICK blue_square PLACE panel3\nNAME Chad WAIT
                
        # if agent_name == "Alice":
        #     agent_prompt += f"You must put blue_square in panel2" #you can only reach panel2, panel1, panel3. But you can't reach panel5, panel7, or other bins."
        # elif agent_name == "Bob":
        #     agent_prompt += "You must put pink_polygon in panel4" # you can only reach panel4, panel3, panel5. But you can't reach panel1, panel7, or other bins."
        # elif agent_name == "Chad":
        #     agent_prompt += "You must put yellow_trapezoid in panel6" #you can only reach panel6, panel5, panel7. But you can't reach panel1, panel3, or other bins."
 
        return agent_prompt

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
            initial_pose=sapien.Pose(p=[-1, 0, 0.02]),
        )
        self.cubeB = actors.build_cube(
            self.scene,
            half_size=0.02,
            color=[0, 1, 0, 1],
            name="cubeB",
            initial_pose=sapien.Pose(p=[1, 0, 0.02]),
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
            cubeA_xyz[:, 1] = -0.5 - torch.rand((b,)) * 0.1 + 0.05
            cubeB_xyz = torch.zeros((b, 3))
            cubeB_xyz[:, 0] = torch.rand((b,)) * 0.1 - 0.05
            cubeB_xyz[:, 1] = 0.5 + torch.rand((b,)) * 0.1 - 0.05
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
    def agentB(self) -> Panda:
        return self.agent.agents[0]

    @property
    def agentA(self) -> Panda:
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
        is_agentB_grasping_cubeA = self.agentB.is_grasping(self.cubeA)
        is_agentB_grasping_cubeB = self.agentB.is_grasping(self.cubeB)
        is_agentA_grasping_cubeA = self.agentA.is_grasping(self.cubeA)
        is_agentA_grasping_cubeB = self.agentA.is_grasping(self.cubeB)
        success = (
            cubeB_in_goal * cubeA_in_goal
        )
        return {
            "is_agentB_grasping_cubeA": is_agentB_grasping_cubeA, # bad, redo later with Union[None, Actor]
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
        is_agentB_grasping_cubeA = self.agentB.is_grasping(self.cubeA)
        is_agentB_grasping_cubeB = self.agentB.is_grasping(self.cubeB)
        is_agentA_grasping_cubeA = self.agentA.is_grasping(self.cubeA)
        is_agentA_grasping_cubeB = self.agentA.is_grasping(self.cubeB)
        success = (
            cubeB_in_goal * cubeA_in_goal
        )
        
        obs = dict(
            arm_b_tcp=self.agentB.tcp.pose.raw_pose,
            arm_a_tcp=self.agentA.tcp.pose.raw_pose,
        )

        if "state" in self.obs_mode:
            obs.update(
                cubeA_goal_region_pos=self.goal_region[0].pose.p,
                cubeB_goal_region_pos=self.goal_region[1].pose.p,
                cubeA_pose=self.cubeA.pose.raw_pose,
                cubeB_pose=self.cubeB.pose.raw_pose,
                is_agentB_grasping_cubeA=is_agentB_grasping_cubeA,
                is_agentA_grasping_cubeB=is_agentA_grasping_cubeB,
                is_agentB_grasping_cubeB=is_agentB_grasping_cubeB,
                is_agentA_grasping_cubeA=is_agentA_grasping_cubeA,
                is_cubeA_grasped=is_agentA_grasping_cubeA or is_agentB_grasping_cubeA,
                is_cubeB_grasped=is_agentA_grasping_cubeB or is_agentB_grasping_cubeB,
                cubeA_in_goal=cubeA_in_goal,
                cubeB_in_goal=cubeB_in_goal,
                arm_b_tcp_to_cubeA_pos=self.cubeA.pose.p
                - self.agentB.tcp.pose.p,
                arm_a_tcp_to_cubeB_pos=self.cubeB.pose.p
                - self.agentA.tcp.pose.p
            )
        return obs

    # info is ultimately going to be what you return in evaluate and elapsed steps
    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        # Stage 1: opposite agents reach and grasp
        # reward for the opposite robot reaching for its opposite cube
        cubeA_to_arm_b_tcp_dist = torch.linalg.norm(
            self.agentB.tcp.pose.p - self.cubeA.pose.p, axis=1
        )

        cubeB_to_arm_a_tcp_dist = torch.linalg.norm(
            self.agentA.tcp.pose.p - self.cubeB.pose.p, axis=1
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
            self.agentB.tcp.pose.p - self.cubeB.pose.p, axis=1
        )

        cubeA_to_arm_a_tcp_dist = torch.linalg.norm(
            self.agentA.tcp.pose.p - self.cubeA.pose.p, axis=1
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
        gripper_width = (self.agentB.robot.get_qlimits()[0, -1, 1] * 2).to(
            self.device
        )  # NOTE: hard-coded with panda
        ungrasp_reward_b = (
            torch.sum(self.agentB.robot.get_qpos()[:, -2:], axis=1) / gripper_width
        )
        ungrasp_reward_b[~info["is_cubeB_grasped"]] = 1.0
        ungrasp_reward_a = (
            torch.sum(self.agentA.robot.get_qpos()[:, -2:], axis=1) / gripper_width
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


if __name__ == "__main__":
    gym.pprint_registry()
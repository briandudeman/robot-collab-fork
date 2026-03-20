import mplib
import gymnasium as gym
import numpy as np
import sapien
import torch
from mplib.pymp import ArticulatedModel, PlanningWorld
from transforms3d import quaternions

from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.agents.base_agent import BaseAgent
from mani_skill.agents.multi_agent import MultiAgent
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.tasks.tabletop.rocobench_test import RocobenchTest
from mani_skill.examples.motionplanning.base_motionplanner.motionplanner import BaseMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from mani_skill.utils.structs.pose import to_sapien_pose
from mani_skill.examples.motionplanning.base_motionplanner.utils import (
    compute_grasp_info_by_obb, get_actor_obb)


env_kwargs = dict(
    obs_mode="none",
    reward_mode=None,
    control_mode="pd_joint_pos",
    render_mode="human",
    sensor_configs=dict(shader_pack="default"),
    human_render_camera_configs=dict(shader_pack="default"),
    viewer_camera_configs=dict(shader_pack="default"),
    num_envs=1,
    sim_backend="auto",
    render_backend="cpu",
    enable_shadow=True,
    parallel_in_single_scene=False,
)

env = gym.make("RocobenchTest", **env_kwargs)
'''env = RecordEpisode(
    env,
    output_dir="ManiSkill/mani_skill/examples/motionplanning/panda/rocobenchtest_vids",
    save_trajectory=False, save_video=True,
    source_type="motionplanning",
    source_desc="official motion planning solution from ManiSkill contributors",
    video_fps=30,
    record_reward=False,
    save_on_reset=True
)'''
#viewer = env.render()
#viewer.paused = True
env.reset()

env.render()

class RocobenchTestPandaArmMotionPlanner(PandaArmMotionPlanningSolver):
    OPEN = 1
    CLOSED = -1
    
    
    def __init__(
        self,
        env: BaseEnv,
        debug: bool = False,
        vis: bool = True,
        base_pose: sapien.Pose = None,  # TODO mplib doesn't support robot base being anywhere but 0
        visualize_target_grasp_pose: bool = True,
        print_env_info: bool = True,
        joint_vel_limits=0.9,
        joint_acc_limits=0.9,
        agent_id=0
    ):
        #super().__init__(env, debug, vis, base_pose, visualize_target_grasp_pose, print_env_info, joint_vel_limits, joint_acc_limits, agent_id)
        self.env = env
        self.agent_id = agent_id
        self.base_env: BaseEnv = env.unwrapped
        self.env_agent: BaseAgent = self.base_env.agent
        if isinstance(self.env_agent, MultiAgent):
            self.env_agent = self.env_agent.agents[agent_id]
        self.robot = self.env_agent.robot
        self.joint_vel_limits = joint_vel_limits
        self.joint_acc_limits = joint_acc_limits

        self.base_pose = to_sapien_pose(base_pose)

        self.planner = self.setup_planner()
        self.control_mode = self.base_env.control_mode

        self.debug = debug
        self.vis = vis
        self.base_pose = base_pose
        self.gripper_state = self.OPEN
        self.visualize_target_grasp_pose = visualize_target_grasp_pose
        self.grasp_pose_visual = None
        if self.vis and self.visualize_target_grasp_pose:
            if "grasp_pose_visual" not in self.base_env.scene.actors:
                self.grasp_pose_visual = build_two_finger_gripper_grasp_pose_visual(
                    self.base_env.scene
                )
            else:
                self.grasp_pose_visual = self.base_env.scene.actors["grasp_pose_visual"]
            self.grasp_pose_visual.set_pose(self.env_agent.tcp_pose)
    
    
    def _update_grasp_visual(self, target: sapien.Pose) -> None:
        if self.grasp_pose_visual is not None:
            self.grasp_pose_visual.set_pose(target)


OPEN = 1
CLOSED = -1


def build_two_finger_gripper_grasp_pose_visual(scene: ManiSkillScene):
    builder = scene.create_actor_builder()
    grasp_pose_visual_width = 0.01
    grasp_width = 0.05

    builder.add_sphere_visual(
        pose=sapien.Pose(p=[0, 0, 0.0]),
        radius=grasp_pose_visual_width,
        material=sapien.render.RenderMaterial(base_color=[0.3, 0.4, 0.8, 0.7])
    )

    builder.add_box_visual(
        pose=sapien.Pose(p=[0, 0, -0.08]),
        half_size=[grasp_pose_visual_width, grasp_pose_visual_width, 0.02],
        material=sapien.render.RenderMaterial(base_color=[0, 1, 0, 0.7]),
    )
    builder.add_box_visual(
        pose=sapien.Pose(p=[0, 0, -0.05]),
        half_size=[grasp_pose_visual_width, grasp_width, grasp_pose_visual_width],
        material=sapien.render.RenderMaterial(base_color=[0, 1, 0, 0.7]),
    )
    builder.add_box_visual(
        pose=sapien.Pose(
            p=[
                0.03 - grasp_pose_visual_width * 3,
                grasp_width + grasp_pose_visual_width,
                0.03 - 0.05,
            ],
            q=quaternions.axangle2quat(np.array([0, 1, 0]), theta=np.pi / 2),
        ),
        half_size=[0.04, grasp_pose_visual_width, grasp_pose_visual_width],
        material=sapien.render.RenderMaterial(base_color=[0, 0, 1, 0.7]),
    )
    builder.add_box_visual(
        pose=sapien.Pose(
            p=[
                0.03 - grasp_pose_visual_width * 3,
                -grasp_width - grasp_pose_visual_width,
                0.03 - 0.05,
            ],
            q=quaternions.axangle2quat(np.array([0, 1, 0]), theta=np.pi / 2),
        ),
        half_size=[0.04, grasp_pose_visual_width, grasp_pose_visual_width],
        material=sapien.render.RenderMaterial(base_color=[1, 0, 0, 0.7]),
    )
    grasp_pose_visual = builder.build_kinematic(name="grasp_pose_visual")
    return grasp_pose_visual


def open_gripper(agent, planner, env, agent_id, other_agents_grip, t=6, gripper_state=None):
    if gripper_state is None:
        gripper_state = OPEN
    qpos = agent.robot.get_qpos()[0, : len(planner.joint_vel_limits)].cpu().numpy()
    other_robots_qpos = np.ndarray((len(env.agent.agents) - 1, qpos.shape[0]))
    other_agents = [specific_agent for specific_agent in env.agent.agents if specific_agent is not agent]
    for i in range(len(other_agents)):
        other_robots_qpos[i] = other_agents[i].robot.get_qpos()[0, : len(planner.joint_vel_limits)].cpu().numpy()
    for i in range(t):
        if other_robots_qpos.shape[1] < 8: # hardcoded number of joints, bad but whatever
            other_robots_qpos = np.hstack((other_robots_qpos, np.tile(other_agents_grip, (len(env.agent.agents) - 1, 1))))
        diff_len = other_robots_qpos.shape[1] - qpos.shape[0]
        temp = np.vstack((qpos.reshape((qpos.shape[0], 1)), np.tile(gripper_state, (diff_len, 1))))
        temp = temp.reshape((temp.shape[0], 1)).T
        action = np.insert(other_robots_qpos, agent_id, temp, axis=0) # ERROR: other qpos is same as temp
        obs, reward, terminated, truncated, info = env.step(action)
        env.unwrapped.render_human()
    return obs, reward, terminated, truncated, info, gripper_state


def close_gripper(agent, planner, env, agent_id, other_agents_grip, t=6, gripper_state=None):
    if gripper_state is None:
        gripper_state = CLOSED
    qpos = agent.robot.get_qpos()[0, : len(planner.joint_vel_limits)].cpu().numpy()
    other_robots_qpos = np.ndarray((len(env.agent.agents) - 1, qpos.shape[0]))
    other_agents = [specific_agent for specific_agent in env.agent.agents if specific_agent is not agent]
    for i in range(len(other_agents)):
        other_robots_qpos[i] = other_agents[i].robot.get_qpos()[0, : len(planner.joint_vel_limits)].cpu().numpy()
    for i in range(t):
        if other_robots_qpos.shape[1] < 8: # hardcoded number of joints, bad but whatever
            other_robots_qpos = np.hstack((other_robots_qpos, np.tile(other_agents_grip, (len(env.agent.agents) - 1, 1))))
        diff_len = other_robots_qpos.shape[1] - qpos.shape[0]
        temp = np.vstack((qpos.reshape((qpos.shape[0], 1)), np.tile(gripper_state, (diff_len, 1))))
        temp = temp.reshape((temp.shape[0], 1)).T
        action = np.insert(other_robots_qpos, agent_id, temp, axis=0) # ERROR: other qpos is same as temp
        obs, reward, terminated, truncated, info = env.step(action)
        env.unwrapped.render_human()
    return obs, reward, terminated, truncated, info, gripper_state



move_group = "panda_hand_tcp"
link_namesA = [link.get_name() for link in env.unwrapped.agentA.robot.get_links()]
joint_namesA = [joint.get_name() for joint in env.unwrapped.agentA.robot.get_active_joints()]
plannerA_base_pose = to_sapien_pose(env.unwrapped.agentA.robot.pose)

link_namesB = [link.get_name() for link in env.unwrapped.agentB.robot.get_links()]
joint_namesB = [joint.get_name() for joint in env.unwrapped.agentB.robot.get_active_joints()]
plannerB_base_pose = to_sapien_pose(env.unwrapped.agentB.robot.pose)


plannerA = mplib.Planner(
    urdf=env.unwrapped.agentA.urdf_path,
    srdf=env.unwrapped.agentA.urdf_path.replace(".urdf", ".srdf"),
    user_link_names=link_namesA,
    user_joint_names=joint_namesA,
    move_group=move_group)

plannerA_grip = OPEN

plannerB = mplib.Planner(
    urdf=env.unwrapped.agentB.urdf_path,
    srdf=env.unwrapped.agentB.urdf_path.replace(".urdf", ".srdf"),
    user_link_names=link_namesB,
    user_joint_names=joint_namesB,
    move_group=move_group)

plannerB_grip = OPEN


plannerA.set_base_pose(np.hstack([plannerA_base_pose.p, plannerA_base_pose.q]))
plannerA.joint_vel_limits = np.asarray(plannerA.joint_vel_limits) * 0.9
plannerA.joint_acc_limits = np.asarray(plannerA.joint_acc_limits) * 0.9


plannerB.set_base_pose(np.hstack([plannerB_base_pose.p, plannerB_base_pose.q]))
plannerB.joint_vel_limits = np.asarray(plannerB.joint_vel_limits) * 0.9
plannerB.joint_acc_limits = np.asarray(plannerB.joint_acc_limits) * 0.9



def plan(planner, agent, end_pose):
    result = planner.plan_screw(
        np.concatenate([end_pose.p, end_pose.q]),
        agent.robot.get_qpos().cpu().numpy()[0],
        time_step=env.unwrapped.control_timestep,
        use_point_cloud=False)
    if result["status"] != "Success":
        result = planner.plan_screw(
            np.concatenate([end_pose.p, end_pose.q]),
            agent.robot.get_qpos().cpu().numpy()[0],
            time_step=env.unwrapped.control_timestep,
            use_point_cloud=False)
    
    
    return result

def print_collisions(collisions):
        """Helper function to abstract away the printing of collisions"""
        if len(collisions) == 0:
            print("No collision")
            return
        for collision in collisions:
            print(
                f"{collision.link_name1} of entity {collision.object_name1} collides"
                f" with {collision.link_name2} of entity {collision.object_name2}"
            )
            
            
def pad_path(result1, result2):
    result1_len = result1["position"].shape[0]
    result2_len = result2["position"].shape[0]
    max_len = max(result1_len, result2_len)
    if result1_len < max_len:
        result1["position"] = np.vstack((result1["position"], np.tile(result1["position"][-1], (max_len - result1_len, 1))))
    elif result2_len < max_len:
        result2["position"] = np.vstack((result2["position"], np.tile(result2["position"][-1], (max_len - result2_len, 1))))
    return result1, result2

def follow_path_2_robot(result1, result2, result1_grip, result2_grip, refine_steps: int = 0):
    result1, result2 = pad_path(result1, result2)
    n_step = result1["position"].shape[0]
    print("result1 shape", result1["position"].shape)
    print("result2 shape", result2["position"].shape)
    for i in range(n_step + refine_steps):
        qpos1 = result1["position"][min(i, n_step - 1)]
        qpos2 = result2["position"][min(i, n_step - 1)]
        print("qpos1: ", qpos1.shape)
        print("qpos2: ", qpos2.shape)
        print("result1_grip: ", result1_grip)
        print("result2_grip: ", result2_grip)
        action = np.vstack((np.hstack([qpos1, result1_grip]), np.hstack([qpos2, result2_grip])))
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
    return obs, reward, terminated, truncated, info




move_group = "panda_hand_tcp"
agentB_link_names = [link.get_name() for link in env.unwrapped.agentB.robot.get_links()]
agentB_joint_names = [joint.get_name() for joint in env.unwrapped.agentB.robot.get_active_joints()]
agentB_articulation = ArticulatedModel(
            env.unwrapped.agentB.urdf_path,
            env.unwrapped.agentB.urdf_path.replace(".urdf", ".srdf"),
            [0, 0, -9.81],
            agentB_link_names,
            agentB_joint_names,
            False,
            False,
        )


FINGER_LENGTH = 0.025

# retrieves the object oriented bounding box (trimesh box object)
obbB = get_actor_obb(env.unwrapped.cubeB)
obbA = get_actor_obb(env.unwrapped.cubeA)

approaching = np.array([0, 0, -1])
# get transformation matrix of the tcp pose, is default batched and on torch
target_closingA = env.unwrapped.agentA.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
target_closingB = env.unwrapped.agentB.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
# we can build a simple grasp pose using this information for Panda
grasp_infoA = compute_grasp_info_by_obb(
    obbB,
    approaching=approaching,
    target_closing=target_closingA,
    depth=FINGER_LENGTH,
)
grasp_infoB = compute_grasp_info_by_obb(
    obbA,
    approaching=approaching,
    target_closing=target_closingB,
    depth=FINGER_LENGTH,
)

closingA, centerA = grasp_infoA["closing"], grasp_infoA["center"]
grasp_poseA = env.unwrapped.agentA.build_grasp_pose(approaching, closingA, env.unwrapped.cubeB.pose.sp.p)

closingB, centerB = grasp_infoB["closing"], grasp_infoB["center"]
grasp_poseB = env.unwrapped.agentB.build_grasp_pose(approaching, closingB, env.unwrapped.cubeA.pose.sp.p)


# -------------------------------------------------------------------------- #
# Reach
# -------------------------------------------------------------------------- #
reach_poseA = grasp_poseA * sapien.Pose([0, 0, -0.05])

poseA = to_sapien_pose(reach_poseA)

reach_poseB = grasp_poseB * sapien.Pose([0, 0, -0.05])

poseB = to_sapien_pose(reach_poseB)

middle_poseA = sapien.Pose(env.unwrapped.middle_goal.pose.sp.p + [0.1, 0, 0], grasp_poseA.q)
middle_poseB = sapien.Pose(env.unwrapped.middle_goal.pose.sp.p + [-0.1, 0, 0], grasp_poseB.q)

intermediate_poseA = sapien.Pose(env.unwrapped.middle_goal.pose.sp.p + [0, .2, .2], grasp_poseA.q)
intermediate_poseB = sapien.Pose(env.unwrapped.middle_goal.pose.sp.p + [0, -.2, .2], grasp_poseB.q)

goal_poseA = sapien.Pose(env.unwrapped.goal_region[0].pose.sp.p, grasp_poseA.q)
goal_poseB = sapien.Pose(env.unwrapped.goal_region[1].pose.sp.p, grasp_poseB.q)

resultA = plan(plannerA, env.unwrapped.agentA, poseA)
resultB = plan(plannerB, env.unwrapped.agentB, poseB)

print("resultA shape: ", resultA["position"].shape)
print("resultB shape: ", resultB["position"].shape)

#resultB = np.hstack((env.unwrapped.agentB.robot.pose.p, env.unwrapped.agentB.robot.pose.q))


follow_path_2_robot(resultB, resultA, plannerB_grip, plannerA_grip)

print("grasp pose a", np.hstack((grasp_poseA.p, grasp_poseA.q)))
print("grasp pose a", grasp_poseB)

grasp_resultA = plan(plannerA, env.unwrapped.agentA, grasp_poseA)
grasp_resultB = plan(plannerB, env.unwrapped.agentB, grasp_poseB)

print("grasp plan shape:", grasp_resultA["position"].shape)
print(grasp_resultB["position"].shape)

follow_path_2_robot(grasp_resultB, grasp_resultA, plannerB_grip, plannerA_grip)

_, _, _, _, _, plannerA_grip = close_gripper(env.agent.agents[1], plannerA, env, 1, plannerB_grip)
_, _, _, _, _, plannerB_grip = close_gripper(env.agent.agents[0], plannerB, env, 0, plannerA_grip)

up_resultA = plan(plannerA, env.unwrapped.agentA, poseA)
up_resultB = plan(plannerB, env.unwrapped.agentB, poseB)
follow_path_2_robot(up_resultB, up_resultA, plannerB_grip, plannerA_grip)

middle_resultA = plan(plannerA, env.unwrapped.agentA, middle_poseA)
middle_resultB = plan(plannerB, env.unwrapped.agentB, middle_poseB)
follow_path_2_robot(middle_resultB, middle_resultA, plannerB_grip, plannerA_grip)

_, _, _, _, _, plannerA_grip = open_gripper(env.agent.agents[1], plannerA, env, 1, plannerB_grip)
_, _, _, _, _, plannerB_grip = open_gripper(env.agent.agents[0], plannerB, env, 0, plannerA_grip)

intermediate_resultA = plan(plannerA, env.unwrapped.agentA, intermediate_poseA)
intermediate_resultB = plan(plannerB, env.unwrapped.agentB, intermediate_poseB)

follow_path_2_robot(intermediate_resultB, intermediate_resultA, plannerB_grip, plannerA_grip)

grasp_pose_oppA = env.agentA.build_grasp_pose(approaching, closingA, env.cubeA.pose.sp.p)
grasp_pose_oppB = env.agentB.build_grasp_pose(approaching, closingB, env.cubeB.pose.sp.p)

reach_pose_oppA = grasp_pose_oppA * sapien.Pose([0, 0, -0.05])
reach_pose_oppB = grasp_pose_oppB * sapien.Pose([0, 0, -0.05])



opp_reach_resultA = plan(plannerA, env.unwrapped.agentA, reach_pose_oppA)
opp_reach_resultB = plan(plannerB, env.unwrapped.agentB, reach_pose_oppB)

do_nothing_resultB = {"position": np.tile(env.unwrapped.agentB.robot.get_qpos()[:, :7].numpy(), (resultA["position"].shape[0], 1))}


follow_path_2_robot(do_nothing_resultB, opp_reach_resultA, plannerB_grip, plannerA_grip)

opp_grasp_resultA = plan(plannerA, env.unwrapped.agentA, grasp_pose_oppA)

follow_path_2_robot(do_nothing_resultB, opp_grasp_resultA, plannerB_grip, plannerA_grip)


_, _, _, _, _, plannerA_grip = close_gripper(env.agent.agents[1], plannerA, env, 1, plannerB_grip)

up_opp_reach_resultA = plan(plannerA, env.unwrapped.agentA, reach_pose_oppA)
follow_path_2_robot(do_nothing_resultB, up_opp_reach_resultA, plannerB_grip, plannerA_grip)

goal_resultA = plan(plannerA, env.unwrapped.agentA, goal_poseA)
follow_path_2_robot(opp_reach_resultB, goal_resultA, plannerB_grip, plannerA_grip)

# planner a completes
_, _, _, _, _, plannerA_grip = open_gripper(env.agent.agents[1], plannerA, env, 1, plannerB_grip)

do_nothing_resultA = {"position": np.tile(env.unwrapped.agentA.robot.get_qpos()[:, :7].numpy(), (resultB["position"].shape[0], 1))}

opp_grasp_resultB = plan(plannerB, env.unwrapped.agentB, grasp_pose_oppB)

follow_path_2_robot(opp_grasp_resultB, do_nothing_resultA, plannerB_grip, plannerA_grip)

_, _, _, _, _, plannerB_grip = close_gripper(env.agent.agents[0], plannerB, env, 0, plannerA_grip)

up_opp_reach_resultB = plan(plannerB, env.unwrapped.agentB, reach_pose_oppB)

follow_path_2_robot(up_opp_reach_resultB, do_nothing_resultA, plannerB_grip, plannerA_grip)


goal_resultB = plan(plannerB, env.unwrapped.agentB, goal_poseB)

follow_path_2_robot(goal_resultB, do_nothing_resultA, plannerB_grip, plannerA_grip)

_, _, _, _, _, plannerB_grip = open_gripper(env.agent.agents[0], plannerB, env, 0, plannerA_grip)

'''

# must be after plan, or it will give an error
plannerA.planning_world.add_articulation(plannerB.robot, "agentB")
plannerB.planning_world.add_articulation(plannerA.robot, "agentA")


env_collision_qpos = np.array(np.concatenate((pose.p, pose.q), axis=0), dtype=np.float64)
print_collisions(plannerA.planning_world.collide_full())
#obs, reward, terminated, truncated, info = follow_path_2_robot(resultA, resultB)
# planner.move_to_pose_with_screw(reach_pose)
'''
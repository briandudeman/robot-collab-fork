import mplib
import gymnasium as gym
import numpy as np
import sapien
import torch
from mplib.pymp import ArticulatedModel, PlanningWorld

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
env.reset() # could be the grasping info?
viewer = env.render()
#viewer.paused = True
env.render()

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

plannerB = mplib.Planner(
    urdf=env.unwrapped.agentB.urdf_path,
    srdf=env.unwrapped.agentB.urdf_path.replace(".urdf", ".srdf"),
    user_link_names=link_namesB,
    user_joint_names=joint_namesB,
    move_group=move_group)

plannerA.set_base_pose(np.hstack([plannerA_base_pose.p, plannerA_base_pose.q]))
plannerA.joint_vel_limits = np.asarray(plannerA.joint_vel_limits) * 0.9
plannerA.joint_acc_limits = np.asarray(plannerA.joint_acc_limits) * 0.9


plannerB.set_base_pose(np.hstack([plannerB_base_pose.p, plannerB_base_pose.q]))
plannerB.joint_vel_limits = np.asarray(plannerB.joint_vel_limits) * 0.9
plannerB.joint_acc_limits = np.asarray(plannerB.joint_acc_limits) * 0.9

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
print(plannerB.robot.get_user_link_names())
print(plannerB.robot.get_user_joint_names())
print(plannerB.robot.get_move_group_end_effectors())
print(plannerB.robot.get_move_group_qpos_dim())

print("agentB_articulation")
print(agentB_articulation.get_user_link_names())
print(agentB_articulation.get_user_joint_names())
print(agentB_articulation.get_move_group_end_effectors())
print(agentB_articulation.get_move_group_qpos_dim())

print("agent A")
print(plannerA.robot.get_user_link_names())
print(plannerA.robot.get_user_joint_names())
print(plannerA.robot.get_move_group_end_effectors())
print(plannerA.robot.get_move_group_qpos_dim())


FINGER_LENGTH = 0.025
env = env.unwrapped

# retrieves the object oriented bounding box (trimesh box object)
obbB = get_actor_obb(env.cubeB)
obbA = get_actor_obb(env.cubeA)

approaching = np.array([0, 0, -1])
# get transformation matrix of the tcp pose, is default batched and on torch
target_closingA = env.agentA.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
target_closingB = env.agentB.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
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

closingA, centeAr = grasp_infoA["closing"], grasp_infoA["center"]
grasp_poseA = env.agentA.build_grasp_pose(approaching, closingA, env.cubeB.pose.sp.p)

closingB, centerB = grasp_infoB["closing"], grasp_infoB["center"]
grasp_poseB = env.agentB.build_grasp_pose(approaching, closingB, env.cubeA.pose.sp.p)


# -------------------------------------------------------------------------- #
# Reach
# -------------------------------------------------------------------------- #
reach_pose = grasp_poseA * sapien.Pose([0, 0, -0.05])

pose = to_sapien_pose(reach_pose)

pose = sapien.Pose(env.middle_goal.pose.sp.p, grasp_poseA.q)

def plan(planner, agent, end_pose):
    print("planner", planner)
    print("agent ", agent.robot.get_qpos().cpu().numpy()[0])
    print("end pose p ", end_pose.p)
    print("end pose q ", end_pose.q)
    result = planner.plan_screw(
        np.concatenate([end_pose.p, end_pose.q]),
        agent.robot.get_qpos().cpu().numpy()[0],
        time_step=env.control_timestep,
        use_point_cloud=False)
    if result["status"] != "Success":
        result = planner.plan_screw(
            np.concatenate([end_pose.p, end_pose.q]),
            agent.robot.get_qpos().cpu().numpy()[0],
            time_step=env.control_timestep,
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

def follow_path(result, refine_steps: int = 0):
    n_step = result["position"].shape[0]
    for i in range(n_step + refine_steps):
        qpos = result["position"][min(i, n_step - 1)]
        action = np.hstack([qpos, 1.0])
        action = np.vstack((action, action))
        print(action)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
    return obs, reward, terminated, truncated, info




result = plan(plannerA, env.unwrapped.agentA, pose)

# must be after plan, or it will give an error
plannerA.planning_world.add_articulation(agentB_articulation, "agentB")
plannerB.planning_world.add_articulation(plannerA.robot, "agentA")


env_collision_qpos = np.array(np.concatenate((pose.p, pose.q), axis=0), dtype=np.float64)
print_collisions(plannerA.planning_world.collide_full())
obs, reward, terminated, truncated, info = follow_path(result)
# planner.move_to_pose_with_screw(reach_pose)

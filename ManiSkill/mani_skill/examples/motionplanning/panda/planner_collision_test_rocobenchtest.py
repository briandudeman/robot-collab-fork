import mplib
import gymnasium as gym
import numpy as np
import sapien
import torch

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
'''
planner = PandaArmMotionPlanningSolver(
        env,
        debug=False,
        vis=True,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=False,
        print_env_info=False,
    )
'''

move_group = "panda_hand_tcp"
link_names = [link.get_name() for link in env.unwrapped.agentA.robot.get_links()]
joint_names = [joint.get_name() for joint in env.unwrapped.agentA.robot.get_active_joints()]
planner_base_pose = to_sapien_pose(env.unwrapped.agentA.robot.pose)


print("move group: ", move_group)
print("link names: ", link_names)
print("joint names: ", joint_names)
print("planner_base_pose: ", planner_base_pose)
print("urdf path: ", env.unwrapped.agentA.urdf_path)
planner_base = mplib.Planner(
    urdf=env.unwrapped.agentA.urdf_path,
    srdf=env.unwrapped.agentA.urdf_path.replace(".urdf", ".srdf"),
    user_link_names=link_names,
    user_joint_names=joint_names,
    move_group=move_group)



planner_base.set_base_pose(np.hstack([planner_base_pose.p, planner_base_pose.q]))
planner_base.joint_vel_limits = np.asarray(planner_base.joint_vel_limits) * 0.9
planner_base.joint_acc_limits = np.asarray(planner_base.joint_acc_limits) * 0.9

FINGER_LENGTH = 0.025
env = env.unwrapped

# retrieves the object oriented bounding box (trimesh box object)
obb = get_actor_obb(env.cubeB)

approaching = np.array([0, 0, -1])
# get transformation matrix of the tcp pose, is default batched and on torch
target_closing = env.agentA.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
# we can build a simple grasp pose using this information for Panda
grasp_info = compute_grasp_info_by_obb(
    obb,
    approaching=approaching,
    target_closing=target_closing,
    depth=FINGER_LENGTH,
)
closing, center = grasp_info["closing"], grasp_info["center"]
grasp_pose = env.agentA.build_grasp_pose(approaching, closing, env.cubeB.pose.sp.p)

# -------------------------------------------------------------------------- #
# Reach
# -------------------------------------------------------------------------- #
reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])

pose = to_sapien_pose(reach_pose)

pose = to_sapien_pose(env.agentA.robot.pose)

result = planner_base.plan_screw(
    np.concatenate([torch.add(env.agentA.tcp_pose.p[0], .05), pose.q]),
    env.agentA.robot.get_qpos().cpu().numpy()[0],
    time_step=env.control_timestep,
    use_point_cloud=False)
if result["status"] != "Success":
    result = planner_base.plan_screw(
        np.concatenate([torch.add(env.agentA.tcp_pose.p[0], .05), pose.q]),
        env.agentA.robot.get_qpos().cpu().numpy()[0],
        time_step=env.control_timestep,
        use_point_cloud=False)

print(result)

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

obs, reward, terminated, truncated, info = follow_path(result)
#planner.move_to_pose_with_screw(reach_pose)



'''
move_group = "panda_hand_tcp"
link_names = [link.get_name() for link in env.unwrapped.agent.robot.get_links()]
joint_names = [joint.get_name() for joint in env.unwrapped.agent.robot.get_active_joints()]
planner_base_pose = to_sapien_pose(env.unwrapped.agent.robot.pose)

planner = mplib.Planner(
    urdf=env.unwrapped.agent.urdf_path,
    srdf=env.unwrapped.agent.urdf_path.replace(".urdf", ".srdf"),
    user_link_names=link_names,
    user_joint_names=joint_names,
    move_group=move_group)

planner.set_base_pose(np.hstack([planner_base_pose.p, planner_base_pose.q]))
planner.joint_vel_limits = np.asarray(planner.joint_vel_limits) * 0.9 # joint_vel_limits
planner.joint_acc_limits = np.asarray(planner.joint_acc_limits) * 0.9 # joint_acc_limits

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


env_collision_qpos = np.array(np.concatenate((env.cube.pose.p.numpy()[0], env.cube.pose.q.numpy()[0]), axis=0), dtype=np.float64)
print(env_collision_qpos)
result = planner.plan_screw(env_collision_qpos, env.unwrapped.agent.robot.get_qpos().cpu().numpy()[0])
print(result)


link_namesA = [link.get_name() for link in env.unwrapped.agentA.robot.get_links()]
link_namesB = [link.get_name() for link in env.unwrapped.agentB.robot.get_links()]

joint_namesA = [joint.get_name() for joint in env.unwrapped.agentA.robot.get_active_joints()]
joint_namesB = [joint.get_name() for joint in env.unwrapped.agentB.robot.get_active_joints()]


planner0_base_pose = to_sapien_pose(env.unwrapped.agentB.robot.pose)

planner0 = mplib.Planner(
    urdf=env.unwrapped.agentB.urdf_path,
    srdf=env.unwrapped.agentB.urdf_path.replace(".urdf", ".srdf"),
    user_link_names=link_namesB,
    user_joint_names=joint_namesB,
    move_group=move_group)

planner0.set_base_pose(np.hstack([planner0_base_pose.p, planner0_base_pose.q]))
planner0.joint_vel_limits = np.asarray(planner0.joint_vel_limits) * 0.9 # joint_vel_limits
planner0.joint_acc_limits = np.asarray(planner0.joint_acc_limits) * 0.9 # joint_acc_limits




planner1_base_pose = to_sapien_pose(env.unwrapped.agentA.robot.pose)

planner1 = mplib.Planner(
    urdf=env.unwrapped.agentA.urdf_path,
    srdf=env.unwrapped.agentA.urdf_path.replace(".urdf", ".srdf"),
    user_link_names=link_namesA,
    user_joint_names=joint_namesA,
    move_group=move_group)


planner1.set_base_pose(np.hstack([planner1_base_pose.p, planner1_base_pose.q]))
planner1.joint_vel_limits = np.asarray(planner1.joint_vel_limits) * 0.9 # joint_vel_limits
planner1.joint_acc_limits = np.asarray(planner1.joint_acc_limits) * 0.9 # joint_acc_limits



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


def follow_path(self, result, refine_steps: int = 0):
    n_step = result["position"].shape[0]
    for i in range(n_step + refine_steps):
        qpos = result["position"][min(i, n_step - 1)]
        if self.control_mode == "pd_joint_pos_vel":
            qvel = result["velocity"][min(i, n_step - 1)]
            action = np.hstack([qpos, qvel])
        else:
            action = np.hstack([qpos])
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.elapsed_steps += 1
        if self.print_env_info:
            print(
                f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
            )
        if self.vis:
            self.base_env.render_human()
    return obs, reward, terminated, truncated, info


env.reset()
viewer = env.render()
viewer.paused = True
env.render()


print(env.cubeB.pose.p.numpy())
print(env.cubeB.pose.q.numpy())
env_collision_qpos = np.array(np.concatenate((env.cubeB.pose.p.numpy()[0], env.cubeB.pose.q.numpy()[0]), axis=0), dtype=np.float64)
print(env_collision_qpos)
result = planner1.plan_screw(env_collision_qpos, env.unwrapped.agentA.robot.get_qpos().cpu().numpy()[0])
print("path: ", result, "\n")



print_collisions(planner1.check_for_env_collision(qpos=env_collision_qpos))

'''
import mplib
import gymnasium as gym
import numpy as np
import sapien
import openai
import os, json
import torch


from mplib.pymp import ArticulatedModel, PlanningWorld
from transforms3d import quaternions
from mani_skill.examples.motionplanning.base_motionplanner.utils import compute_grasp_info_by_obb, get_actor_obb
from mani_skill.envs.tasks.tabletop.rocobench_test import RocobenchTest
from mani_skill.utils.structs.pose import to_sapien_pose
from dm_control.utils.transformations import mat_to_quat, quat_to_euler, euler_to_quat 



OPEN = 1
CLOSED = -1

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
    for i in range(n_step + refine_steps):
        qpos1 = result1["position"][min(i, n_step - 1)]
        qpos2 = result2["position"][min(i, n_step - 1)]
        action = np.vstack((np.hstack([qpos1, result1_grip]), np.hstack([qpos2, result2_grip])))
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
    return obs, reward, terminated, truncated, info




assert os.path.exists("openai_key.json"), "Please put your OpenAI API key in a string in robot-collab/openai_key.json"
OPENAI_KEY = str(json.load(open("openai_key.json")))
openai.api_key = OPENAI_KEY


env_kwargs = dict(
    obs_mode="state_dict",
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

#viewer = env.render()
#viewer.paused = True
env.reset()

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


PATH_PLAN_INSTRUCTION="""
[Path Plan Instruction]
Each <pose> is a list [x,y,z,q1,q2,q3] for gripper location (first 3 numbers) and rotation in euler angles (4 - 6th numbers) of the gripper above the object, ready to initiate a grasping motion.
So each <pose> is the reach position and should be slightly above the target to allow for a safe and smooth picking-up motion, follow these steps to plan:
1) Decide target location and rotation(e.g. the position and rotation of an object you want to pick), and your current gripper location and rotation. All rotations should be in euler, and will most likely be related to the rotation of the target object
2) Return the <pose> representing the location and angle the gripper should be in to initiate grasp sequence of object.
"""


user_prompt = env.get_agent_prompt(env.unwrapped.get_obs(), "agentA")

response = openai.ChatCompletion.create(
                    model="gpt-4o-mini", 
                    messages=[
                        # {"role": "user", "content": ""},
                        {"role": "system", "content": PATH_PLAN_INSTRUCTION+user_prompt},                                    
                    ],
                    max_tokens=1024,
                    temperature=0.0,
                    )

# eval is an itty bitty security risk for now, but whatever
print("agent prompt: ", PATH_PLAN_INSTRUCTION+user_prompt)
print("agent response", response["choices"][0]["message"]["content"])

'''
agentA_pose = eval(response["choices"][0]["message"]["content"])
print("agent generated pose", agentA_pose)

print("regular pose", poseA)
print(euler_to_quat(agentA_pose[3:6]))
agentA_pose = sapien.Pose(p=np.array(agentA_pose[0:3], dtype=np.float32), q=np.array(euler_to_quat(agentA_pose[3:6]), dtype=np.float32))

resultA = plan(plannerA, env.unwrapped.agentA, agentA_pose)
print(resultA)
do_nothing_resultB = {"position": np.tile(env.unwrapped.agentB.robot.get_qpos()[:, :7].numpy(), (resultA["position"].shape[0], 1))}


print("cubeB position", env.unwrapped.cubeB.pose.get_p())
print(response["choices"][0]["message"]["content"])


follow_path_2_robot(do_nothing_resultB, resultA, plannerB_grip, plannerA_grip)
print(env.unwrapped.agentA.tcp.pose.get_p())
'''
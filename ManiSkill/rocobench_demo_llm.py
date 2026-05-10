import mplib
import gymnasium as gym
import numpy as np
import sapien
import openai
import os, json
import torch
import re


from mplib.pymp import ArticulatedModel, PlanningWorld
from transforms3d import quaternions
from mani_skill.agents.base_agent import BaseAgent
from mani_skill.examples.motionplanning.base_motionplanner.utils import compute_grasp_info_by_obb, get_actor_obb
from mani_skill.envs.tasks.tabletop.rocobench_test import RocobenchTest
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.structs.pose import to_sapien_pose
from mani_skill.examples.motionplanning.panda.planner_collision_test_rocobenchtest import plan, close_gripper, open_gripper, pad_path, follow_path_2_robot
from dm_control.utils.transformations import mat_to_quat, quat_to_euler, euler_to_quat 



OPEN = 1
CLOSED = -1



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

print("grasp pose", grasp_poseA.get_p(), grasp_poseA.get_q())
print("reach pose", reach_poseA.get_p(), reach_poseA.get_q())
reach_poseB = grasp_poseB * sapien.Pose([0, 0, -0.05])

poseB = to_sapien_pose(reach_poseB)


PATH_PLAN_INSTRUCTION="""
[Path Plan Instruction for Robot Arm Gripper]
Each <action> is one of 5 possible actions that will be executed, and will have arguments (preceded by their type) needed to execute each action in parentheses after that action:
 - <approach_target> (string: target_object)
	- this <action> represents moving the gripper hand to a position directly above target_object, such that it can effectively execute <grasp_target> without colliding with the environment. target_object should be the name or id of the object that the gripper is approaching.
 - <grasp_target> (string: target_object)
	- this <action> represents moving the gripper hand to a position where the object is directly between the gripper fingers. Because it is much closer to the target_object, it is only safe to execute this action if the gripper is directly above the target object.
 - <go_to> (float: x, float: y, float: z)
	- this <action> represents moving the gripper hand directly to a position either slightly above the table as to release the object, or somewhere in the air above the table. The numbers, x, y, and z represent the coordinates that the gripper hand should go to. These should all be numbers and not variables.
 - <open_gripper> (int: current_grip)
	- this <action> opens the gripper hand. If the gripper is currently holding an object, this should only happen once the gripper hand is very slightly above the table, as to not damage the object.
 - <close_gripper> (int: current_grip)
	- this <action> closes the gripper hand. If the gripper is currently holding an object, this will do nothing. This <action> should only be initialized once the gripper hand is in a position to grasp the target object, like one obtained by executing <grasp_target>.
At each query, you will only output one <action> which should be on its own line that starts with <Action> (EX: <Action> <approach_target> ("cubeA")), which will then be executed.
When planning to a position where both grippers may exist, as in a handoff, agentA will execute their <action> before agentB, then will leave the shared area so agentB can execute their <action>. Identify these shared areas before planning a path to them.
"""

def prompt_llm_and_plan(planner: mplib.Planner, agent_name: str, agent: BaseAgent, env: BaseEnv, base_prompt: str, planner_grip: int, other_planner_grip: int,  agent_action_history: str):
    
    result = None
    agent_prompt = env.get_agent_prompt(env.unwrapped.get_obs(), agent_name)

    response = openai.ChatCompletion.create(
                        model="gpt-4.1", 
                        messages=[
                            # {"role": "user", "content": ""},
                            {"role": "system", "content": base_prompt+agent_prompt+agent_action_history},                                    
                        ],
                        max_tokens=1024,
                        temperature=0.0,
                        )

    # eval is an itty bitty security risk for now, but whatever
    print("prompt: ", base_prompt+agent_prompt+agent_action_history)
    print("agent response", response["choices"][0]["message"]["content"])
    
    
    agent_generated_action = ""
    agent_generated_variables = ""

    for line in response["choices"][0]["message"]["content"].splitlines():
        if line[0:8] == "<Action>":
            agent_generated_action = re.search(r"\<(.*?)\>", line[8:]).group(1)
            agent_generated_variables = re.search(r"\(([^)]+)\)", line[8:]).group(1)
    print(agent_generated_action)
    print(agent_generated_variables)
    
    agent_action_history = agent_action_history + "<" + agent_generated_action + "> (" + agent_generated_variables + ") \n"
    
    
    match agent_generated_action:
        case "approach_target" | "grasp_target":
            target = agent_generated_variables.strip("\"")
            obb = get_actor_obb(getattr(env.unwrapped, target))
            target_closing = agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
            approaching = np.array([0, 0, -1])
            grasp_info = compute_grasp_info_by_obb(
                obb,
                approaching=approaching,
                target_closing=target_closing,
                depth=FINGER_LENGTH,
            )
            closing, center = grasp_info["closing"], grasp_info["center"]
            grasp_pose = agent.build_grasp_pose(approaching, closing, getattr(env.unwrapped, target).pose.sp.p)
            reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
            
            if agent_generated_action == "approach_target":
                pose = to_sapien_pose(reach_pose)
            else:
                pose = to_sapien_pose(grasp_pose)
                            
            result = plan(planner, agent, env, pose)            
        case "go_to":
            coord_strings = agent_generated_variables.split(",")
            coords = []
            for coord in coord_strings:
                coord = coord.replace(" ", "")
                coords.append(float(coord))
            agent_pose = sapien.Pose(p=np.array(coords, dtype=np.float32), q=np.array(agent.tcp.pose.get_q()[0]))
            result = plan(planner, agent, env, agent_pose)
            if result["status"] != "Success":
                result.pop("status")
                result_feedback = "\n[Environment Feedback]\n" 
                for k, v in result.items():
                    result_feedback = result_feedback + k + ": " + str(v) + "\n"
                
                result, planner_grip, agent_action_history = prompt_llm_and_plan(plannerA, "agentA", env.unwrapped.agentA, env.unwrapped, PATH_PLAN_INSTRUCTION + result_feedback, plannerA_grip, plannerB_grip, agentA_action_history)

        case "open_gripper":
            _, _, _, _, _, planner_grip = open_gripper(agent, planner, env, agent._agent_idx, other_planner_grip, OPEN, CLOSED)
        case "close_gripper":
            _, _, _, _, _, planner_grip = close_gripper(agent, planner, env, agent._agent_idx, other_planner_grip, OPEN, CLOSED)

        
    print("plan", result)
    
    '''
    print("agent generated pose", agent_generated_pose)
    print("agent generated pose type", type(agent_generated_pose))
    agent_pose = eval(agent_generated_pose)
    if isinstance(agent_pose, list) and len(agent_pose) == 6:
        agent_pose = sapien.Pose(p=np.array(agent_pose[3:], dtype=np.float32), q=np.array(euler_to_quat(agent_pose[0:3]), dtype=np.float32))
    else:
        raise Exception("Agent did not generate pose of length 6")
    
    result = plan(planner, agent, env, agent_pose)

    if result["status"] != "Success":
        result.pop("status")
        result_feedback = "\n[Environment Feedback]\n" 
        for k, v in result.items():
            result_feedback = result_feedback + k + ": " + str(v) + "\n"
        
        print("result feedback: ", result_feedback)
        
        response = openai.ChatCompletion.create(
                        model="gpt-4", 
                        messages=[
                            # {"role": "user", "content": ""},
                            {"role": "system", "content": base_prompt+agent_prompt+result_feedback},                                    
                        ],
                        max_tokens=1024,
                        temperature=0.0,
                        )
        
        agent_generated_pose = 0

        print("corrected response:", response["choices"][0]["message"]["content"])
        for line in response["choices"][0]["message"]["content"].splitlines():
            if line[0:7] == "<pose>:":
                agent_generated_pose = line[7:]
        
        agent_pose = eval(agent_generated_pose)
        if isinstance(agent_pose, list) and len(agent_pose) == 6:
            agent_pose = sapien.Pose(p=np.array(agent_pose[0:3], dtype=np.float32), q=np.array(euler_to_quat(agent_pose[3:]), dtype=np.float32))
        else:
            raise Exception("Agent did not generate pose of length 6")
        
        result = plan(planner, agent, env, agent_pose)
        
    
    if result["status"] != "Success":
        raise Exception("Agent failed to generate successful pose")
    '''
    return result, planner_grip, agent_action_history
    
agentA_action_history = "\n Previous Actions: \n"
agentB_action_history = "\n Previous Actions: \n"

for i in range(10):

    resultA, plannerA_grip, agentA_action_history = prompt_llm_and_plan(plannerA, "agentA", env.unwrapped.agentA, env.unwrapped, PATH_PLAN_INSTRUCTION, plannerA_grip, plannerB_grip, agentA_action_history)
    resultB, plannerB_grip, agentB_action_history = prompt_llm_and_plan(plannerB, "agentB", env.unwrapped.agentB, env.unwrapped, PATH_PLAN_INSTRUCTION, plannerB_grip, plannerA_grip, agentB_action_history)
    
    
    if resultA and resultB:
        follow_path_2_robot(env.unwrapped, resultB, resultA, plannerB_grip, plannerA_grip)
    elif (resultA and not resultB) or (not resultA and resultB):
        if not resultA:
            resultA = {"position": np.tile(env.unwrapped.agentA.robot.get_qpos()[:, :7].numpy(), (resultB["position"].shape[0], 1))}
        elif not resultB:
            resultB = {"position": np.tile(env.unwrapped.agentB.robot.get_qpos()[:, :7].numpy(), (resultA["position"].shape[0], 1))}
        
        follow_path_2_robot(env.unwrapped, resultB, resultA, plannerB_grip, plannerA_grip)


'''
#do_nothing_resultB = {"position": np.tile(env.unwrapped.agentB.robot.get_qpos()[:, :7].numpy(), (resultA["position"].shape[0], 1))}

follow_path_2_robot(env.unwrapped, resultB, resultA, plannerB_grip, plannerA_grip)

grasp_resultA = plan(plannerA, env.unwrapped.agentA, env.unwrapped, grasp_poseA)
grasp_resultB = plan(plannerB, env.unwrapped.agentB, env.unwrapped, grasp_poseB)

follow_path_2_robot(env.unwrapped, grasp_resultB, grasp_resultA, plannerB_grip, plannerA_grip)

_, _, _, _, _, plannerA_grip = close_gripper(env.agent.agents[1], plannerA, env, 1, plannerB_grip, OPEN, CLOSED)
_, _, _, _, _, plannerB_grip = close_gripper(env.agent.agents[0], plannerB, env, 1, plannerA_grip, OPEN, CLOSED)

for i in range(3):

    resultA = prompt_llm_and_plan(plannerA, "agentA", env.unwrapped.agentA, env.unwrapped, PATH_PLAN_INSTRUCTION)
    resultB = prompt_llm_and_plan(plannerB, "agentB", env.unwrapped.agentB, env.unwrapped, PATH_PLAN_INSTRUCTION)


    follow_path_2_robot(env.unwrapped, resultB, resultA, plannerB_grip, plannerA_grip)
'''
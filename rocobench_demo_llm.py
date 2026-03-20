import mplib
import gymnasium as gym
import numpy as np
import sapien
import openai
import os, json
import torch


from mplib.pymp import ArticulatedModel, PlanningWorld
from transforms3d import quaternions
from ManiSkill.mani_skill.envs.tasks.tabletop.rocobench_test import RocobenchTest


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

PATH_PLAN_INSTRUCTION="""
[Path Plan Instruction]
Each <coord> is a tuple (x,y,z) for gripper location, follow these steps to plan:
1) Decide target location (e.g. an object you want to pick), and your current gripper location.
2) Plan a list of <coord> that move smoothly from current gripper to the target location.
3) The <coord>s must be evenly spaced between start and target.
4) Each <coord> must not collide with other robots, and must stay away from table and objects.  
[How to Incoporate [Enviornment Feedback] to improve plan]
    If IK fails, propose more feasible step for the gripper to reach. 
    If detected collision, move robot so the gripper and the inhand object stay away from the collided objects. 
    If collision is detected at a Goal Step, choose a different action.
    To make a path more evenly spaced, make distance between pair-wise steps similar.
        e.g. given path [(0.1, 0.2, 0.3), (0.2, 0.2. 0.3), (0.3, 0.4. 0.7)], the distance between steps (0.1, 0.2, 0.3)-(0.2, 0.2. 0.3) is too low, and between (0.2, 0.2. 0.3)-(0.3, 0.4. 0.7) is too high. You can change the path to [(0.1, 0.2, 0.3), (0.15, 0.3. 0.5), (0.3, 0.4. 0.7)] 
    If a plan failed to execute, re-plan to choose more feasible steps in each PATH, or choose different actions.
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

print(response["choices"][0]["message"]["content"])

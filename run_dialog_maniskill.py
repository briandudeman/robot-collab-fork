import os
import pickle
import json
import numpy as np
import logging
from datetime import datetime
from glob import glob
from natsort import natsorted
from copy import deepcopy
import argparse
from dataclasses import dataclass
from typing import List, Annotated, Literal, Tuple, Dict, Union, Optional, Any
from collections import defaultdict
import matplotlib.pyplot as plt
import gymnasium as gym
import tyro

from ManiSkill.mani_skill.envs.tasks.tabletop import RocobenchTest
from ManiSkill.mani_skill.envs.sapien_env import BaseEnv
from ManiSkill.mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver
from ManiSkill.mani_skill.examples.motionplanning.base_motionplanner.utils import (
    compute_grasp_info_by_obb, get_actor_obb)

from rocobench.envs import SortOneBlockTask, CabinetTask, MoveRopeTask, SweepTask, MakeSandwichTask, PackGroceryTask, MujocoSimEnv, SimRobot, visualize_voxel_scene
from rocobench import PlannedPathPolicy, LLMPathPlan, MultiArmRRT
from prompting import LLMResponseParser, FeedbackManager, DialogPrompterM, SingleThreadPrompter, save_episode_html

# print out logging.info
logging.basicConfig(level=logging.INFO)
logging.root.setLevel(logging.INFO)

TASK_NAME_MAP = {
    "rocobench": RocobenchTest
}

class LLMRunner:
    def __init__(
        self,
        env: RocobenchTest,
        robots: Dict[str, SimRobot],
        max_runner_steps: int = 50,
        video_format: str = "mp4",
        num_runs: int = 1,
        verbose: bool =False,
        np_seed: int = 0,
        start_seed: int = 0,
        run_name: str = "run",
        data_dir: str = "data",
        overwrite: bool = False,
        llm_output_mode="action_only", # "action_only" or "action_and_path"
        llm_comm_mode="chat",
        llm_num_replans=1,
        give_env_feedback=True,
        direct_waypoints: int = 0,
        max_failed_waypoints: int = 0,
        debug_mode: bool = False,
        split_parsed_plans: bool = False,
        use_history: bool = False,
        use_feedback: bool = False,
        temperature: float = 0.0,
        llm_source: str = "gpt4",
        vis: bool = False,
        seed = None
        ):
        self.seed = seed
        self.env = env
        self.env.reset() # not migrated
        self.robots = robots
        self.robot_agent_names = list(robots.keys()) # ['Alice', etc.]
        self.data_dir = data_dir
        self.run_name = run_name
        run_dir = os.path.join(self.data_dir, self.run_name)
        os.makedirs(run_dir, exist_ok=overwrite)
        self.run_dir = run_dir
        self.verbose = verbose
        self.np_seed = np_seed
        self.start_seed = start_seed
        self.num_runs = num_runs
        self.overwrite = overwrite
        self.direct_waypoints = direct_waypoints
        self.max_failed_waypoints = max_failed_waypoints
        self.max_runner_steps = max_runner_steps
        self.give_env_feedback = give_env_feedback
        self.use_history = use_history
        self.use_feedback = use_feedback
        self.vis = vis
        
        self.llm_output_mode = llm_output_mode
        self.debug_mode = debug_mode # useful for debug


        self.llm_num_replans = llm_num_replans
        self.llm_comm_mode = llm_comm_mode
        self.response_keywords = ['NAME', 'ACTION']
        if llm_output_mode == "action_and_path":
            self.response_keywords.append('PATH')
        
        self.video_format = video_format
        self.split_parsed_plans = split_parsed_plans
        self.temperature = temperature
        self.parser = LLMResponseParser( # not migrated to maniskill
            self.env,
            llm_output_mode,
            self.env.robot_name_map,
            self.response_keywords,
            self.direct_waypoints,
            use_prepick=self.env.use_prepick,
            use_preplace=self.env.use_preplace # NOTE: should be custom defined in each task env
        )
        self.feedback_manager = FeedbackManager( # not migrated to maniskill
            env=self.env,
            planner=self.planner,
            llm_output_mode=self.llm_output_mode,
            robot_name_map=self.env.robot_name_map,
            step_std_threshold=self.env.waypoint_std_threshold,
            max_failed_waypoints=self.max_failed_waypoints,
        )
        if llm_comm_mode in ["plan", "chat"]:
            logging.warning(f'Using SingleThreadPrompter for {llm_comm_mode} mode')
            self.prompter = SingleThreadPrompter( # not migrated to maniskill, but i don't think we need to focus on this, migrate DialogPrompter first
                env=self.env,
                parser=self.parser,
                feedback_manager=self.feedback_manager,
                max_tokens=1024,
                debug_mode=self.debug_mode,
                use_waypoints=(self.llm_output_mode == "action_and_path"),
                use_history=self.use_history,
                num_replans=self.llm_num_replans,
                comm_mode=llm_comm_mode,
                temperature=self.temperature,
                llm_source=llm_source,
            )

        else:
            self.prompter = DialogPrompterM(
                env=self.env,
                parser=self.parser,
                feedback_manager=self.feedback_manager,
                max_tokens=512,
                debug_mode=self.debug_mode,
                robot_name_map=self.env.robot_name_map,
                max_calls_per_round=10,
                use_waypoints=(self.llm_output_mode == "action_and_path"),
                use_history=self.use_history,
                use_feedback=self.use_feedback,
                num_replans=self.llm_num_replans,
                temperature=self.temperature,
                llm_source=llm_source,
            )


    def display_plan(self, plan: LLMPathPlan, save_name = "vis_plan", save_dir = None):
        """ Display the plan in the open3d viewer """ 
        env = deepcopy(self.env) # all of this not migrated to maniskill, physics n such
        env.physics.data.qpos[:] = self.env.physics.data.qpos[:].copy()
        env.physics.forward()
        env.render_point_cloud = True
        obs = env.get_obs()
        path_ls = plan.path_3d_list
        if save_dir is not None:
            save_path = os.path.join(save_dir, f"{save_name}.jpg")
        visualize_voxel_scene(
            obs.scene,
            path_pts=path_ls,
            save_img=(save_dir is not None),
            img_path=save_path
            )
        

    def one_run(self, run_id: int = 0, start_step: int = 0, skip_reset = False, prev_llm_plans = [], prev_response = None, prev_actions = None):
        """ uses planner """
        self.env.seed(np_seed=run_id) # not migrated to maniskill
        if not skip_reset:
            self.env.reset(reload=True) # NOTE: need to do this to reset the model.eq_active vals
        env = self.env
        physics = env.physics
        success = False
        save_dir = os.path.join(self.run_dir, f"run_{run_id}")
        os.makedirs(save_dir, exist_ok=self.overwrite)

        done = False
        reward = 0
        obs = env.get_obs()
        for step in range(start_step, start_step + self.max_runner_steps):

            step_dir = os.path.join(save_dir, f"step_{step}")
            os.makedirs(step_dir, exist_ok=self.overwrite)
            prompt_path = os.path.join(step_dir, "prompts")
            os.makedirs(prompt_path, exist_ok=self.overwrite)

            sim_data = env.save_intermediate_state() # not migrated to maniskill
            data_fname = f"{step_dir}/env_init.pkl"
            with open(data_fname, "wb") as f:
                pickle.dump(sim_data, f)


            if step == start_step and len(prev_llm_plans) > 0:
                ready_to_execute = 1
                current_llm_plan = prev_llm_plans
                response = ""
                prompt_breakdown = dict()

            elif step == start_step and prev_actions is not None:
                ready_to_execute = 1
                current_llm_plan = prev_llm_plans
                response = ""
                prompt_breakdown = dict()

            else:
                ready_to_execute, current_llm_plan, response, prompt_breakdown = self.prompter.prompt_one_round(
                    obs,
                    save_path=prompt_path,
                    # prev_response=(prev_response['response'] if step == start_step and prev_response is not None else None)
                    )
                if not ready_to_execute or current_llm_plan is None:
                    print(f"Run {run_id}: Step {step} failed to get a plan from LLM. Move on to next step.")
                    continue

                for i, plan in enumerate(current_llm_plan):
                    save_fname = os.path.join(step_dir, f"llm_plan_{i}.pkl")
                    with open(save_fname, "wb") as f:
                        pickle.dump(plan, f)


            logging.info(f"Step: {step} LLM plan parsed, begin RRT planning ")
            # try execute this plan, if one of the plan failed, rewind the env to before the first plan was executed!
            rewind_env = False

            for i, plan in enumerate(current_llm_plan):
                print('tograsp:', plan.tograsp, 'inhand:', plan.inhand, plan.action_strs)
                """policy = PlannedPathPolicy( # not migrated to maniskill
                    physics=env.physics,
                    robots=self.robots,
                    path_plan=plan,
                    graspable_object_names=self.env.get_graspable_objects(),
                    allowed_collision_pairs=self.env.get_allowed_collision_pairs(),
                    plan_splitted=self.split_parsed_plans,
                    **self.policy_kwargs,
                )"""
                
                solver = PandaArmMotionPlanningSolver(
                    env=env,
                    vis=self.vis,
                    seed=self.seed,
                    debug=False,
                    visualize_target_grasp_pose=self.vis,
                    print_env_info=False,
                )
                
                # some equivalent of policy here

                num_sim_steps = 0
                if prev_actions is not None:
                    for sim_action in prev_actions:
                        # env.physics.model.eq_active[52:] = 0
                        # env.physics.forward() # DEBUG
                        obs, reward, done, info = env.step(sim_action, verbose=False) # not migrated to maniskill
                        num_sim_steps += 1
                else:
                    # breakpoint()
                    plan_success, reason = policy.plan(env) # not migrated to maniskill
                    logging.info(f"Stesp: {step} Plan success: {plan_success}, reason: {reason}")
                    if plan_success:
                        logging.info(f"Execute the plan for {len(policy.action_buffer)} steps")

                        plan_fname = os.path.join(step_dir, f"rrt_plan_{i}.pkl")
                        plans = policy.rrt_plan_results
                        with open(plan_fname, "wb") as f:
                            pickle.dump(plans, f)

                        actions_fname = f"{step_dir}/actions_{i}.pkl"
                        with open(actions_fname, "wb") as f:
                            pickle.dump(policy.action_buffer, f)

                        while not policy.plan_exhausted:
                            sim_action = policy.act(obs, env.physics) # not migrated to maniskill, includes policy
                            obs, reward, done, info = env.step(sim_action, verbose=False)
                            num_sim_steps += 1

                if num_sim_steps > 0:
                    vid_name = f"{step_dir}/execute.mp4"
                    env.export_render_to_video(vid_name, out_type=self.video_format,  fps=50) # not migrated to maniskill
                    print(f'Plans all executed! Video sample saved to {vid_name}')

                else:
                    print(f"Plan {i} failed to execute.")
                    rewind_env = True
                    break

            if rewind_env:
                print("Rewinding the environment to before the first plan was executed.")
                env.load_saved_state(sim_data) # not migrated to maniskill

            else:
                sim_data = env.save_intermediate_state() # not migrated to maniskill

            data_fname = f"{step_dir}/env_end.pkl"
            with open(data_fname, "wb") as f:
                pickle.dump(sim_data, f)

            self.prompter.post_execute_update(
                obs_desp="", # TODO
                execute_success=(not rewind_env),
                parsed_plan=current_llm_plan[0].get_action_desp()
            )

            if done:
                break

        success = reward > 0
        json.dump(
            dict(step=step, success=success),
            open(f"{save_dir}/steps{step}_success_{success}.json", "w"),
        )
        print("Run finished after {} timesteps".format(step))
        self.prompter.post_episode_update()
        save_episode_html(
            save_dir,
            html_fname=f"steps{step}_success_{success}",
            video_fname="execute.mp4",
            sender_keys=["Alice", "Bob", "Chad", "Dave", "Planner", "Feedback", "Action"],
            )
        print(f"Episode html saved to {save_dir}")


    def run(self, args):
        start_id = 0 if args.start_id == -1 else args.start_id
        if args.cont:
            logging.info("Continuing from previous run")
            load_run = glob(os.path.join(self.data_dir, args.load_run_name, f"run_{args.load_run_id}"))
            if len(load_run) == 0:
                raise ValueError(f"Cannot find run {args.load_run_id} in {args.load_run_name}")
                exit()
            load_run = load_run[0]
            # find the latest steps
            step_dirs = natsorted(
                glob(os.path.join(load_run, "step_*"))
            )
            if len(step_dirs) == 0:
                raise ValueError(f"Cannot find any steps in {load_run}")
                exit()
            latest_step = step_dirs[-1]
            env_init_fname = os.path.join(latest_step, "env_init.pkl")
            with open(env_init_fname, "rb") as f:
                saved_data = pickle.load(f)
                self.env.load_saved_state(saved_data) # not migrated to maniskill

            print(f"==== Loading back Run {args.load_run_id} ====")
            next_step = int(latest_step.split("/")[-1].split("_")[-1])
            prev_llm_plans = []
            prev_plans = natsorted(
                    glob(os.path.join(latest_step, "llm_plan_*pkl"))
                    )
            if len(prev_plans) > 0:
                prev_llm_plans = [pickle.load(open(fname, "rb")) for fname in prev_plans]

            prev_response = None
            prev_responses = natsorted(
                    glob(os.path.join(latest_step, "prompts", "*response.json"))
                    )
            if len(prev_responses) > 0:
                prev_response = json.load(open(prev_responses[-1], "rb"))

            prev_actions = None
            fname = os.path.join(latest_step, "actions.pkl")
            if os.path.exists(fname):
                prev_actions = pickle.load(open(fname, "rb"))

            self.one_run(
                args.load_run_id,
                start_step=next_step,
                skip_reset=True,
                prev_llm_plans=prev_llm_plans,
                prev_response=prev_response,
                prev_actions=prev_actions
                )
            start_id = args.load_run_id + 1
        existing_runs = glob(os.path.join(self.data_dir, args.run_name, "run_*"))
        if args.start_id == -1 and len(existing_runs) > 0:
            existing_run_ids = [int(run.split("_")[-1]) for run in existing_runs]
            start_id = max(existing_run_ids) + 1
        for run_id in range(start_id, start_id + self.num_runs):
            print(f"==== Run {run_id} starts ====")
            self.one_run(run_id)

@dataclass
class Args:
    llm_source: Annotated[Literal['gpt-3.5-turbo', 'gpt-4o-mini' 'gpt-4', 'gpt-3.5-turbo-16k'], tyro.conf.arg(aliases=["-llm"])] = "gpt-4"
    """The name of the llm model to use"""
    
    env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = "RocobenchTest"
    """The environment ID of the task you want to simulate"""

    obs_mode: Annotated[str, tyro.conf.arg(aliases=["-o"])] = "none"
    """Observation mode"""

    robot_uids: Annotated[Optional[str], tyro.conf.arg(aliases=["-r"])] = None
    """Robot UID(s) to use. Can be a comma separated list of UIDs or empty string to have no agents. If not given then defaults to the environments default robot"""

    sim_backend: Annotated[str, tyro.conf.arg(aliases=["-b"])] = "auto"
    """Which simulation backend to use. Can be 'auto', 'cpu', 'gpu'"""

    render_backend: Annotated[str, tyro.conf.arg(aliases=["-rb"])] = "gpu"
    """Which render backend to use. Can be 'gpu', 'cpu', 'none'"""

    reward_mode: Optional[str] = None
    """Reward mode"""

    num_envs: Annotated[int, tyro.conf.arg(aliases=["-n"])] = 1
    """Number of environments to run."""

    control_mode: Annotated[Optional[str], tyro.conf.arg(aliases=["-c"])] = "pd_joint_pos"
    """Control mode"""

    render_mode: str = "rgb_array"
    """Render mode"""

    shader: str = "default"
    """Change shader used for all cameras in the environment for rendering. Default is 'minimal' which is very fast. Can also be 'rt' for ray tracing and generating photo-realistic renders. Can also be 'rt-fast' for a faster but lower quality ray-traced renderer"""

    data_dir: str = "data"
    """Directory to save data on the run"""

    record_dir: Optional[str] = None
    """Directory to save recordings"""

    pause: Annotated[bool, tyro.conf.arg(aliases=["-p"])] = False
    """If using human render mode, auto pauses the simulation upon loading"""

    quiet: bool = False
    """Disable verbose output."""

    seed: Annotated[Optional[Union[int, list[int]]], tyro.conf.arg(aliases=["-s"])] = None
    """Seed(s) for random actions and simulator. Can be a single integer or a list of integers. Default is None (no seeds)"""

    tsteps: Annotated[int, tyro.conf.arg(aliases=["-tsteps"])] = 10
    """The number of times you can rerun the simulation."""
    
    num_runs: Annotated[int, tyro.conf.arg(aliases=["-nruns"])] = 1
    """The number of times the simulation will run, tsteps is nested inside of this."""
    
    run_name: Annotated[str, tyro.conf.arg(aliases=["-rn"])] = "test"
    """The name of the run, for logging purposes."""
    
    temperature: Annotated[int, tyro.conf.arg(aliases=["-temp"])] = 0
    """LLM argument for the randomness in selecting the next token."""
    
    start_id: Annotated[int, tyro.conf.arg(aliases=["-sid"])] = -1
    """The number/id of the starting run"""
    
    output_mode: Annotated[Literal["action_only", "action_and_path"], tyro.conf.arg(aliases=["-output_mode"])] = "action_only"
    """The output mode of the dialog prompter."""
    
    comm_mode: Annotated[Literal["chat", "plan", "dialog"], tyro.conf.arg(aliases=["-comm_mode"])] = "dialog"
    """The communication mode of the LLM model. dialog is used with DialogPrompterM"""
    
    direct_waypoints: Annotated[int, tyro.conf.arg(aliases=["-dw"])] = 5
    """Used by the response parser."""
    
    num_replans: Annotated[int, tyro.conf.arg(aliases=["-nr"])] = 5
    """Number of replans the LLM can do for 1 round."""
    
    cont: Annotated[bool, tyro.conf.arg(aliases=["-c"])] = False
    """Continuing from previous run, not implemented for now."""
    
    load_run_name: Annotated[str, tyro.conf.arg(aliases=["-lr"])] = "sort_task"
    """The name of the run to load if cont is True."""
    
    load_run_id: Annotated[int, tyro.conf.arg(aliases=["-ld"])] = 0
    """The id of the run to load if cont is True."""

    max_failed_waypoints: Annotated[int, tyro.conf.arg(aliases=["-max"])] = 1
    """The number of failed waypoints before termination. The feedback manager uses this."""

    debug_mode: Annotated[bool, tyro.conf.arg(aliases=["-i"])] = False
    """Enables or disables debug mode."""

    no_history: Annotated[bool, tyro.conf.arg(aliases=["-nh"])] = False
    """Enables or disables history for the system prompt for the LLM"""

    no_feedback: Annotated[bool, tyro.conf.arg(aliases=["-nf"])] = False
    """Enables or disables feedback for the system prompt for the LLM"""

    vis: Annotated[bool, tyro.conf.arg(aliases=["-vis"])] = False
    """whether or not to open a GUI to visualize the a motionplanning solution live"""




def main(args: Args):

    if args.render_mode == "none":
        args.render_mode = None
    np.set_printoptions(suppress=True, precision=3)
    verbose = not args.quiet
    if isinstance(args.seed, int):
        args.seed = [args.seed]
    if args.seed is not None:
        np.random.seed(args.seed[0])
    parallel_in_single_scene = args.render_mode == "human"
    if args.render_mode == "human" and args.obs_mode in ["sensor_data", "rgb", "rgbd", "depth", "point_cloud"]:
        print("Disabling parallel single scene/GUI render as observation mode is a visual one. Change observation mode to state or state_dict to see a parallel env render")
        parallel_in_single_scene = False
    if args.render_mode == "human" and args.num_envs == 1:
        parallel_in_single_scene = False

    env_kwargs = dict(
        obs_mode=args.obs_mode,
        reward_mode=args.reward_mode,
        control_mode=args.control_mode,
        render_mode=args.render_mode,
        sensor_configs=dict(shader_pack=args.shader),
        human_render_camera_configs=dict(shader_pack=args.shader),
        viewer_camera_configs=dict(shader_pack=args.shader),
        num_envs=args.num_envs,
        sim_backend=args.sim_backend,
        render_backend=args.render_backend,
        enable_shadow=True,
        parallel_in_single_scene=parallel_in_single_scene,
    )

    env : BaseEnv = gym.make(
        args.env_id,
        **env_kwargs
    )
    

    # save args into a json file
    args_dict = vars(args)
    args_dict["env"] = env.__class__.__name__
    timestamp = datetime.now().strftime("%Y%m_%H%M")
    fname = os.path.join(args.data_dir, args.run_name, f"args_{timestamp}.json")
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    json.dump(args_dict, open(fname, "w"), indent=2)
    
    runner = LLMRunner( # all args migrated, but class not migrated to maniskill
        env=env,
        data_dir=args.data_dir,
        max_runner_steps=args.tsteps,
        num_runs=args.num_runs,
        run_name=args.run_name,
        overwrite=True,
        llm_output_mode=args.output_mode, # "action_only" or "action_and_path"
        llm_comm_mode=args.comm_mode, # "chat" or "plan"
        llm_num_replans=args.num_replans,
        direct_waypoints=args.direct_waypoints,
        max_failed_waypoints=args.max_failed_waypoints,
        debug_mode=args.debug_mode,
        use_history=(not args.no_history),
        use_feedback=(not args.no_feedback),
        temperature=args.temperature,
        llm_source=args.llm_source,
        vis=args.vis,
        seed=args.seed
    )
    runner.run(args)



if __name__ == "__main__":
    parsed_args = tyro.cli(Args)
    logging.basicConfig(level=logging.INFO)
    main(parsed_args)

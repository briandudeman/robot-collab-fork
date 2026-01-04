import logging
import numpy as np
from time import time
from copy import deepcopy

from ManiSkill.mani_skill.envs.tasks.tabletop import RocobenchTest
from ManiSkill.mani_skill.agents import MultiAgent, BaseAgent
from Maniskill.mani_skill.agents.controllers import PDEEPoseController
from mani_skill.utils.structs import Pose

import matplotlib.pyplot as plt
from transforms3d import euler, quaternions
from typing import Callable, List, Optional, Tuple, Union, Dict, Set, Any, FrozenSet

from dm_control.utils.inverse_kinematics import qpos_from_site_pose
from dm_control.utils.transformations import mat_to_quat, quat_to_euler, euler_to_quat 

from rocobench.rrt import direct_path, smooth_path, birrt, NearJointsUniformSampler, CenterWaypointsUniformSampler
from rocobench.envs import SimRobot 


class ManiskillMultiArmRRT:
    def __init__(self, env: RocobenchTest, seed: int = 0):
        self.env = env
        self.multiagent = env.agent # should be a MultiAgent class 
        self.np_random = np.random.RandomState(seed)


    def forward_kinematics_all(): # actually moves the robots i think
        pass 

    # robot is the robot you want to solve ik for, uid
    def solve_ik(self, robot_uid, target_pos, target_quat): # should return an n by 6 tensor of the joints
        for agent in self.multiagent.agents:
            if agent.uid == robot_uid:
                robot_agent = self.multiagent.agents

        if (robot_agent != None):
            controller = robot_agent.controller.controllers[self.multiagent.control_mode] # should get the appropriate controller
            target_qpos = controller.kinematics.compute_ik(
                target_pose=Pose.create_from_pq(p = target_pos, q = target_quat),
                q0=controller.articulation.get_qpos()
            )  # (n, 6) tensor, where n is the number of joints, first 3 in other axis is for pos, last 3 is for euler rotation
            if target_qpos is None:
                raise ValueError("IK solution not found")

            return target_qpos
            # checking for collisions between arms
        
        else:
            print("robot_uid is not in list of robots, please check and rewrite")
        
        def reset_fn(physics): # should be implemented if it is determined that we need an element of randomness in computing IK
            pass

    # ee_poses is a dictionary of all the robots uids and their end poses, as maniskill Pose objects
    def inverse_kinematics_all(self, ee_poses: Dict[str, Pose], check_collisions=True):
        solved_iks = {}
        for robot_uid, robot_pose in ee_poses:
            solved_iks[robot_uid] = self.solve_ik(robot_uid, robot_pose.get_p(), robot_pose.get_q())


    def ee_l2_distance(self, 
        q1: np.ndarray, 
        q2: np.ndarray, 
        orientation_factor: float = 0.2
    ) -> float:
        pose1s = self.forward_kinematics_all(q1, return_ee_pose=True) # {robotA: Pose1, robotB: Pose1}
        pose2s = self.forward_kinematics_all(q2, return_ee_pose=True) # {robotA: Pose2, robotB: Pose2}
        assert pose1s is not None and pose2s is not None
        dist = 0

        # compute pair-wise distance between each robot's Pose1 and Pose2
        for robot_name in pose1s.keys():
            pose1 = pose1s[robot_name]
            pose2 = pose2s[robot_name]
            dist += pose1.distance(pose2, orientation_factor=orientation_factor)
        return dist
        

    def extend_ee_l2():
        pass

    def allow_collision_pairs():
        pass

    def get_collided_links():
        pass

    def check_relative_pose():
        pass

    def check_collision():
        pass

    
    def plan(self, 
        start_qpos: np.ndarray,  # can be either full length or just the desired qpos for the joints 
        goal_qpos: np.ndarray,
        init_samples: Optional[List[np.ndarray]] = None,
        skip_endpoint_collision_check: bool = False,
        skip_direct_path: bool = False,
        skip_smooth_path: bool = False,
        timeout: int = 200,
        check_relative_pose: bool = False
        ) -> Tuple[Optional[List[np.ndarray]], str]:
        
        if len(start_qpos) != len(goal_qpos):
            return None, "RRT failed: start and goal configs have different lengths."
        '''if len(start_qpos) != len(self.all_joint_idxs_in_qpos):
            start_qpos = start_qpos[self.all_joint_idxs_in_qpos]
        if len(goal_qpos) != len(self.all_joint_idxs_in_qpos):
            goal_qpos = goal_qpos[self.all_joint_idxs_in_qpos]
        '''
  
        def collision_fn(q: np.ndarray, show: bool = False):
            '''return self.check_collision(
                robot_qpos=q,
                physics=self.physics,
                allow_grasp=allow_grasp,           
                check_grasp_ids=check_grasp_ids,  
                check_relative_pose=check_relative_pose,
                show=show,
                # detect_grasp=False, TODO?
            )'''
            pass # needs to be implemented when i figure out how to do collisions in maniskill

        '''if not skip_endpoint_collision_check:
            if collision_fn(start_qpos, show=1):
                # print("RRT failed: start qpos in collision.")
                return None, f"ReasonCollisionAtStart_time0_iter0"
            elif collision_fn(goal_qpos, show=1): 
                # print("RRT failed: goal qpos in collision.")
                return None, "ReasonCollisionAtGoal_time0_iter0"
        ''' # uncomment when collision_fn is implemented

        paths, info = birrt(
                start_conf=start_qpos,
                goal_conf=goal_qpos,
                distance_fn=self.ee_l2_distance,
                sample_fn=CenterWaypointsUniformSampler(
                    bias=0.05,
                    start_conf=start_qpos,
                    goal_conf=goal_qpos,
                    numpy_random=self.np_random,
                    min_values=self.joint_minmax[:, 0],
                    max_values=self.joint_minmax[:, 1],
                    init_samples=init_samples,
                ),
                extend_fn=self.extend_ee_l2,
                collision_fn=collision_fn,
                iterations=800,
                smooth_iterations=200,
                timeout=timeout,
                greedy=True,
                np_random=self.np_random,
                smooth_extend_fn=self.extend_ee_l2,
                skip_direct_path=skip_direct_path,
                skip_smooth_path=skip_smooth_path, # enable to make sure it passes through the valid init_samples 
            )
        if paths is None:
            return None, f"RRT failed: {info}"
        return paths, f"RRT succeeded: {info}"
 
    def plan_splitted():
        def collision_fn(q: np.ndarray, show: bool = False):
            pass

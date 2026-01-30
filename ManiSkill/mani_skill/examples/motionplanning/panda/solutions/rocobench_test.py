import argparse
import gymnasium as gym
import numpy as np
import sapien
from transforms3d.euler import euler2quat

from mani_skill.envs.tasks import RocobenchTest
from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.base_motionplanner.utils import (
    compute_grasp_info_by_obb, get_actor_obb)
from mani_skill.utils.wrappers.record import RecordEpisode

def solve(env: RocobenchTest, seed=None, debug=False, vis=False):
    env.reset(seed=seed)
    assert env.unwrapped.control_mode in [
        "pd_joint_pos",
        "pd_joint_pos_vel",
    ], env.unwrapped.control_mode

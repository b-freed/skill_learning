import gym
from mujoco_py import GlfwContext
GlfwContext(offscreen=True)
import d4rl
import ipdb
import h5py
import time
from d4rl.kitchen.adept_envs.franka.kitchen_multitask_v0 import KitchenTaskRelaxV1
import numpy as np


env = gym.make('kitchen-partial-v0')
# env = gym.make('antmaze-large-diverse-v0')
dataset = env.get_dataset()
# import inspect
# inspect.getmro(env)
env.reset()
env.render()


# Assume environment starts with zero velocities
states = dataset['observations']
actions = dataset['actions']
terminals = dataset['terminals']
robot_states = states[:, :30]

qpos = robot_states[0]
qvel = np.zeros(env.model.nv)

# init_qpos = env.sim.model.key_qpos[0].copy()
# init_qvel = env.sim.model.key_qvel[0].copy()

# env.set_state(qpos, qvel)
env.sim.set_state(np.concatenate([qpos, qvel]))
env.sim.forward()
# env.sim.set_state(np.concatenate([init_qpos, init_qvel]))
# robot_states = states[:, :30]
for i in range(15000,20000):
    action = actions[i,:]
    # env.step(action)
    env.sim.set_state(np.concatenate([robot_states[i], qvel]))
    env.sim.forward()
    env.render()

    if terminals[i]:
        print('terminating!')
        qpos = robot_states[i]
        qvel = np.zeros(env.model.nv)
        env.reset()
        # init_qpos = env.sim.model.key_qpos[0].copy()
        # init_qvel = env.sim.model.key_qvel[0].copy()
        env.sim.set_state(np.concatenate([qpos, qvel]))
        # env.sim.set_state(np.concatenate([init_qpos, init_qvel]))
        env.sim.forward()
        time.sleep(0.1)
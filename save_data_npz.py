import gym
import d4rl
import numpy as np


env_name = 'antmaze-large-diverse-v0'
env = gym.make(env_name)
dataset = d4rl.qlearning_dataset(env)

np.savez('datasets/'+env_name,observations      =dataset['observations'],
                              actions           =dataset['actions'],
                              next_observations =dataset['next_observations'],
                              rewards           =dataset['rewards'],
                              terminals         =dataset['terminals'])
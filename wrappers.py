import gym
import numpy as np




class FrankaSliceWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def observation(self, obs):
        obs = obs[:30]
        return obs
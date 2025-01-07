import numpy as np
import gym 
import d4rl

env = gym.make('antmaze-medium-diverse-v0')
goals = []
for i in range(1000):
	# env.set_target()
	goal_state = np.array(env.target_goal)
	goals.append(goal_state)

goals = np.stack(goals)
print(np.mean(goals,axis=0))
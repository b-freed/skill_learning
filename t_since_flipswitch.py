import numpy as np
import d4rl
import gym
import matplotlib.pyplot as plt

env = gym.make('kitchen-partial-v0')
d = env.get_dataset()
states = d['observations']
terms = d['terminals']
dists = dists = np.linalg.norm(states[:,17:19] - np.array([-0.69, -0.05]),axis=-1)

t_since_ep_start = 0
t_flipswitch = []
min_t_flipswitch = 10**10
for t in range(states.shape[0]):
    if dists[t] <= .3:
        t_flipswitch.append(t_since_ep_start)
        if t_since_ep_start < min_t_flipswitch:
            min_t_flipswitch = t_since_ep_start
            print('!!!!!!!!!!!!! ',t)
            print('t_since_ep_start: ', t_since_ep_start)
        
    t_since_ep_start += 1
    if terms[t]:
        t_since_ep_start = 0


plt.plot(dists)
plt.plot(terms)
plt.show()

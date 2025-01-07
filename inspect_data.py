

import gym
import d4rl
import numpy as np
import matplotlib.pyplot as plt


# env = gym.make('antmaze-large-diverse-v0')
env = gym.make('kitchen-partial-v0')
d = env.get_dataset()
norms = np.linalg.norm(d['observations'][1:,:2]-d['observations'][:-1,:2],axis=-1)

plt.figure()
# plt.plot(norms)
plt.plot(d['timeouts'])
plt.hist(norms,bins=200)
plt.show()
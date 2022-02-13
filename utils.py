import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader
import torch.distributions.normal as Normal
from skill_model import SkillModel, SkillModelStateDependentPrior
import ipdb
import d4rl
import random
import gym
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from math import pi
from matplotlib import animation
from PIL import Image


# def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):

# 	#Mess with this to change frame size
# 	plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

# 	patch = plt.imshow(frames[0])
# 	plt.axis('off')

# 	def animate(i):
# 		patch.set_data(frames[i])

# 	anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
# 	anim.save(path, writer='imagemagick', fps=60)

def make_gif(frames,name):
    frames = [Image.fromarray(image) for image in frames]
    frame_one = frames[0]
    frame_one.save(name+'.gif', format="GIF", append_images=frames,
               save_all=True, duration=100, loop=0)
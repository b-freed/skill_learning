'''File where we will sample a set of waypionts, and plan a sequence of skills to have our pointmass travel through those waypoits'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader
import torch.distributions.normal as Normal
from skill_model import SkillModel
import matplotlib.pyplot as plt

# initialize skill sequence
# initialize optimizer for skill sequence
# determine waypoints


for e in epochs:
  # Optimize plan: compute expected cost according to the current sequence of skills, take GD step on cost to optimize skills
  
  # Test plan: deploy learned skills in actual environment.  Now we're going be selecting base-level actions conditioned on the current skill and state, and executign that action in the real environment

  # compute test and train plan cost, plot so we can see what they;re doing

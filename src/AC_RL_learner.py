import math
import random

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")
print(device)

class ActorCritic(nn.Module):
    def __init__(self,num_inputs,num_outputs,hidden_size,std=0.0):
        super(ActorCritic,self).__init__()

        self.critic=nn.Sequential(
            nn.Linear(num_inputs,hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,1)
        )

        self.actor=nn.Sequential(
            nn.Linear(num_inputs,hidden_size),
            nn.Relu(),
            nn.Linear(hidden_size,num_outputs)

        )

        def forward(self,x):
            value=self.critic(x)
            acts=self.actor(x)
            return acts,value

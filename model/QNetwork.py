
import numpy as np
import time
import itertools
import torch
from collections import Counter, defaultdict
import torch.nn as nn
from torch_geometric.utils import *
import tqdm
from collections import namedtuple
from copy import deepcopy
import random
import torch.nn.functional as F
from sklearn import preprocessing
from sklearn.metrics import f1_score
import os

class Transition(object):
    def __init__(self, state, action, batch=-1, reward=0):
        self.state = state
        self.action = action
        self.batch = batch
        self.reward = 0
    
    def __repr__(self):
        return f"{self.reward}"

class Normalizer(object):
    def __init__(self):
        self.mean = None
        self.std = None
        self.state_memory = []
        self.max_size = 1000
        self.length = 0

    def normalize(self, s):
        if self.length == 0:
            return s
        return (s - self.mean) / (self.std + 1e-8)

    def append(self, s):
        if len(self.state_memory) > self.max_size:
            self.state_memory.pop(0)
        self.state_memory.append(s)
        self.mean = np.mean(self.state_memory, axis=0)
        self.std = np.std(self.state_memory, axis=0)
        self.length = len(self.state_memory)

class Memory(object):
    def __init__(self, memory_size, batch_size):
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.fed_reward_index = 0
        self.memory = []
    
    def clear(self):
        self.memory = []
        self.fed_reward_index = 0
    
    @property
    def rewards(self):
        return [1 if x.reward else 0 for x in self.memory]

    def save(self, *args):
        if len(self.memory) == self.memory_size:
            return
        transition = Transition(*args)
        self.memory.append(transition)
    
    def fed_reward(self, reward):
        pass

    def sample(self):
        samples = random.sample(self.memory, self.batch_size)
        action = torch.cat([x.action for x in samples])
        state = torch.cat([x.state for x in samples])
        reward = torch.cat([x.reward for x in samples])
        return state, action, reward

class EstimatorNetwork(nn.Module):
    def __init__(self,
                 action_num,
                 state_shape,
                 mlp_layers,
                 device=torch.device('cpu')):
        super(EstimatorNetwork, self).__init__()

        # build the Q network
        layer_dims = [state_shape] + mlp_layers
        self.device = device
        fc = [nn.Flatten()]
        for i in range(len(layer_dims)-1):
            fc.append(nn.Linear(layer_dims[i], layer_dims[i+1], bias=True))
            fc.append(nn.Tanh())
        fc.append(nn.Linear(layer_dims[-1], action_num, bias=True))
        self.fc_layers = nn.Sequential(*fc)
        self.to(device)
        self.replay_memory_size = 0
        self.memory = []

    def forward(self, states):
        states = states.float().to(self.device)
        return self.fc_layers(states)
    
    def save(self, save_path):
        torch.save(self.state_dict(), save_path)
        return save_path
    
class Selector(object):
    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            if not k.startswith('_'):
                if not hasattr(self, k):
                    setattr(self, k, v)
        self._train = False
     
    def predict():
        pass

    def clear(self):
        self.memory.clear()

    def train():
        pass
    
    def add_memory():
        pass

    def is_full(self):
        return len(self.memory.memory) == self.replay_memory_size
        
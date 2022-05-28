
import numpy as np
import time
import itertools
import torch
from collections import Counter, defaultdict
import torch.nn as nn
from torch_geometric.utils import *
from QNetwork import Memory, EstimatorNetwork, Selector
import tqdm
from collections import namedtuple
from copy import deepcopy
import random
import torch.nn.functional as F
from sklearn import preprocessing
from sklearn.metrics import f1_score

class DepthMemory(Memory):
    def __init__(self, *args, **kwars):
        super(DepthMemory, self).__init__(*args, **kwars)
    
    def fed_reward(self, reward):
        for index, r in zip(range(self.fed_reward_index, len(self.memory)), reward):
            self.memory[index].reward = r.expand(self.memory[index].state.shape[0]).view(-1, 1)
        self.fed_reward_index = len(self.memory)

class DepthSelector(Selector):
    def __init__(self, *args, **kwargs):
        super(DepthSelector, self).__init__(*args, **kwargs)
        self.max_k_hop = self.action_num
        self.qnet = EstimatorNetwork(self.max_k_hop, self.state_shape, self.mlp_layers, self.device)
        self.qnet.eval()
        for p in self.qnet.parameters():
            if len(p.data.shape) > 1:
                nn.init.xavier_uniform_(p.data)
        self.memory = DepthMemory(self.replay_memory_size, self.batch_size)
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.optimizer = torch.optim.Adam(self.qnet.parameters(), lr=self.lr)

    
    def predict(self, node_list, graph_embedding, graph):
        neighbor_embedding_list = []
        for center_node in node_list:
            # one_hop_nodes = set(k_hop_subgraph(int(center_node), 1, graph.edge_index)[0].numpy()) - set([int(center_node)])
            one_hop_nodes = set(k_hop_subgraph(int(center_node), self.max_k_hop, graph.edge_index)[0].numpy()) - set([int(center_node)])
            if len(one_hop_nodes):
                neighbor_embedding_list.append(graph_embedding[list(one_hop_nodes)].mean(0))
            else:
                neighbor_embedding_list.append(graph_embedding[center_node])
        center_node_embedding = torch.cat((graph_embedding[node_list], torch.stack(neighbor_embedding_list)), dim=1)
        self.qnet.eval()
        with torch.no_grad():
            depth_prob = self.qnet(center_node_embedding)
            if self._train:
                self.memory.save(center_node_embedding, depth_prob)
            depth_prob = F.softmax(depth_prob, dim=1)
        assert len(torch.isnan(depth_prob).nonzero()) == 0
        return depth_prob
    
    def train(self, t):
        self.qnet.train()
        self.optimizer.zero_grad()
        state, action, reward = self.memory.sample()
        t.eval()
        with torch.no_grad():
            target_action = t.forward(state)
        a = torch.argmax(target_action, dim=-1)
        r = (torch.ones_like(reward)*-0.5).masked_fill_(reward, 0.5).squeeze()
        y = r + self.discount_factor * target_action.max(1)[0]
        q = self.qnet(state)
        Q = torch.gather(q, dim=-1, index=a.unsqueeze(-1)).squeeze(-1)
        loss = self.mse_loss(Q, y)
        self.optimizer.step()
        return loss.item()

        
    
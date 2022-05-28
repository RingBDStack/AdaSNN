
import numpy as np
import time
import itertools
import torch
from collections import Counter, defaultdict
import torch.nn as nn
from torch_geometric.utils import *
from torch_geometric.data import Batch
import tqdm
from QNetwork import Memory, EstimatorNetwork, Selector
from collections import namedtuple
from copy import deepcopy
import torch_geometric as pyg
import random
import torch.nn.functional as F
from sklearn import preprocessing
from sklearn.metrics import f1_score

class NeighborMemory(Memory):
    def __init__(self, *args, **kwargs):
        super(NeighborMemory, self).__init__(*args, **kwargs)

    def fed_reward(self, reward):
        for index in range(self.fed_reward_index, len(self.memory)):
            self.memory[index].reward = reward[self.memory[index].batch].expand(self.memory[index].state.shape[0]).view(-1, 1)
        self.fed_reward_index = len(self.memory)

class NeighborSelector(Selector):
    def __init__(self, *args, **kwargs):
        super(NeighborSelector, self).__init__(*args, **kwargs)
        self.qnet = EstimatorNetwork(self.action_num, 
                                    self.state_shape, 
                                    self.mlp_layers, 
                                    self.device)
        self.qnet.eval()
        for p in self.qnet.parameters():
            if len(p.data.shape) > 1:
                nn.init.xavier_uniform_(p.data)
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.optimizer = torch.optim.Adam(self.qnet.parameters(), lr=self.lr)
        self.memory = NeighborMemory(memory_size=self.replay_memory_size, batch_size=self.batch_size)

    def filter_adj(self, edge_index, filter_nodes):
        if len(filter_nodes) == 0:
            return edge_index
        num_node = int(edge_index.max()) + 1
        src, target = edge_index
        node_mask = torch.ones(num_node, dtype=bool).fill_(False)
        node_mask[filter_nodes] = True
        mask = ~(torch.index_select(node_mask, 0, target) | torch.index_select(node_mask, 0, src))
        return edge_index[:, mask]

    def predict(self, node_list, depth_list, graph_node_embedding, graph, batch):
        sub_data = []
        subgraph_index_list = []
        for center_node, depth in zip(node_list, depth_list):
            k_1_subgraph_node_list = [int(center_node)]
            k_1_embedding = graph_node_embedding[center_node]
            neighbors = [int(center_node)]
            filter_nodes = []
            for k in range(1, self.fixed_k_hop+depth+1):
                edge_index = self.filter_adj(graph.edge_index, filter_nodes)
                k_subgraph_node_list = k_hop_subgraph(int(center_node), k, graph.edge_index)[0].numpy()
                k_node_list = np.array(list((set(k_subgraph_node_list) - set(k_1_subgraph_node_list))))
                k_1_subgraph_node_list = k_subgraph_node_list
                k_embedding = graph_node_embedding[k_node_list]
                k_embedding = torch.cat((k_embedding, k_1_embedding.expand_as(k_embedding)), dim=-1)
                with torch.no_grad():
                    neighbor_prob = self.qnet(k_embedding)
                    if self._train:
                        self.add_memory(k_embedding, neighbor_prob, batch)
                    neighbor_prob = F.softmax(neighbor_prob, dim=1)
                k_node_list_selected = k_node_list[torch.bernoulli(neighbor_prob[:,0]).nonzero().view(-1).cpu().numpy()]
                neighbors.extend(k_node_list_selected.tolist())
                filter_nodes.extend(list(set(k_node_list) - set(k_node_list_selected.tolist())))
                k_1_embedding = graph_node_embedding[k_node_list_selected].mean(0) if len(k_node_list_selected) else k_1_embedding
            x = graph_node_embedding[neighbors] 
            edge_index = subgraph(neighbors, graph.edge_index, relabel_nodes=True)[0]
            sub_data.append(pyg.data.Data(x=x, edge_index=edge_index))
            subgraph_index_list.append(neighbors)
        return Batch.from_data_list(sub_data), subgraph_index_list
    
    def add_memory(self, embedding, results, batch):
        self.memory.save(embedding, results, batch)
        pass

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
        
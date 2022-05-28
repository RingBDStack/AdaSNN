import numpy as np
import time
import os
import itertools
import networkx as nx
import matplotlib.pyplot as plt
import torch
from collections import Counter, defaultdict
import torch.nn as nn
from torch_geometric.utils import *
from torch_geometric.data import Batch
import torch_geometric as pyg
import tqdm
from collections import namedtuple
from itertools import product
from copy import deepcopy
import random
import torch.nn.functional as F
from sklearn import preprocessing
from sklearn.metrics import f1_score
from neighbor_selector import NeighborSelector
from depth_selector import DepthSelector
import pickle

class AgentChain(object):
    def __init__(self, update_target_estimator_every, time_step, max_k_hop, epochs, visual=False, ablation_depth=0, wandb=None):
        self.update_target_estimator_every = update_target_estimator_every
        self.time_step = time_step
        self.max_k_hop = max_k_hop
        self.train_epoches = 0
        self.epochs = epochs
        self.visual = visual
        self.cnt, self.max = 0, 2000
        # agent act normal when ablation_depth is falsel; act randomly when ablation_depth == -1; act fixed when ablation_depth == 2
        self.ablation = ablation_depth
        self.wandb = wandb
    
    def bind_selector(self, depth, neighbor):
        self.depth_selector = depth
        self.neighbor_selector = neighbor

        self.target_depth_selector_net, self.target_neighbor_selector_net = self.snapshot()
        return self
    
    def load_best(self, depth, neighbor):
        self.depth_selector.qnet = depth
        self.neighbor_selector.qnet = neighbor
        return self

    def snapshot(self):
        return deepcopy(self.depth_selector.qnet), \
            deepcopy(self.neighbor_selector.qnet)

    def predict(self, graph_node_embedding, graph, batch):
        if self.ablation != 0:
            return self.predict_fake(graph_node_embedding, graph, batch, self.ablation)
        node_list = remove_isolated_nodes(graph.edge_index)[-1].nonzero().flatten().numpy().tolist()
        time_step = self.time_step if self.time_step < len(node_list) else len(node_list)
        np.random.seed(1234)
        self.candidate_node_list = np.random.choice(node_list, time_step, replace=False)
        depth_prob = self.depth_selector.predict(self.candidate_node_list, graph_node_embedding, graph)
        self.depth_list = [np.random.choice(list(range(1, self.max_k_hop+1)), p=node_depth.cpu().numpy()) for node_depth in depth_prob]
        sub_data, subgraph_index_list = self.neighbor_selector.predict(self.candidate_node_list, self.depth_list, graph_node_embedding, graph, batch)

        # for visual
        # if self.visual and self.cnt < self.max: 
        #     if not os.path.exists('./temp/save_results_edge_index'):
        #         os.makedirs('./temp/save_results_edge_index')
        #     with open(f'./temp/save_results_edge_index/{self.cnt}.pkl', 'wb') as f :
        #         s = {'graph':to_networkx(graph),
        #         'edge_index':graph.edge_index.numpy(), 
        #         'graph_label':int(graph.y),
        #         'node_label':graph.x.nonzero()[:,1].numpy(),
        #         'subgraph_list':subgraph_index_list,
        #         'kernel':self.candidate_node_list,
        #         'sketch_graph':to_networkx(pyg.data.Data(num_nodes=len(subgraph_index_list), edge_index=self.gen_sketch_graph(subgraph_index_list))),
        #         'depth':self.depth_list}
        #         pickle.dump(s, f)
        #     print(f'./temp/save_results_edge_index/{self.cnt}.pkl')
        #     self.cnt += 1

        return sub_data, self.gen_sketch_graph(subgraph_index_list)
    
    def predict_fake(self, graph_node_embedding, graph, batch, ablation_depth=2):
        node_list = remove_isolated_nodes(graph.edge_index)[-1].nonzero().flatten().numpy().tolist()
        time_step = self.time_step if self.time_step < len(node_list) else len(node_list)
        np.random.seed(1234)
        self.candidate_node_list = np.random.choice(node_list, time_step, replace=False)
        if ablation_depth == 2:
            self.depth_list = np.ones(len(self.candidate_node_list), dtype=np.int) * int(ablation_depth)
        elif ablation_depth == -1:
            self.depth_list = [np.random.choice(range(1, self.max_k_hop+1)) for _ in range(len(self.candidate_node_list))]
        sub_data, subgraph_index_list =  [], []
        for node, depth in zip(self.candidate_node_list, self.depth_list):
            node_index, edge_index, node_map, _ = k_hop_subgraph(int(node), int(depth), graph.edge_index, relabel_nodes=True)
            sub_data.append(pyg.data.Data(x=graph_node_embedding[node_index], edge_index=edge_index))
            subgraph_index_list.append(node_index.numpy().tolist())
        sub_data = Batch.from_data_list(sub_data)
        return sub_data, self.gen_sketch_graph(subgraph_index_list)

    def plot_sub_graph(self, sub_data, batch_index, subgraph_index_list):
        if not os.path.exists(f"./temp_sub/{batch_index}"):
            os.makedirs(f'./temp_sub/{batch_index}')

        for index, graph in enumerate(sub_data.to_data_list()):
            G = nx.Graph()
            for i,j in graph.edge_index.T:
                G.add_edge(int(i), int(j))
            nx.draw(G)
            plt.savefig(f"./temp_sub/{batch_index}/{index}.png")
            print(f"./temp_sub/{batch_index}/{index}.png saved!!!")
            plt.close()
            
    def gen_sketch_graph(self, subgraph_index_list, eps=1):
        sub_num = len(subgraph_index_list)
        adj = torch.zeros((sub_num, sub_num))
        for i, j in product(range(sub_num), range(sub_num)):
            if len(set(subgraph_index_list[i])&set(subgraph_index_list[j])) >= eps:
                adj[i, j] = adj[j, i] = 1
            if i > j:
                continue
        return dense_to_sparse(adj)[0]
    
    def fed_reward(self, reward):
        self.depth_selector.memory.fed_reward(reward)
        self.neighbor_selector.memory.fed_reward(reward)
    
    def train(self):
        for train_epochs in range(self.epochs):
            if train_epochs%self.update_target_estimator_every == 0:
                self.target_depth_selector_net, \
                self.target_neighbor_selector_net = self.snapshot()

            depth_loss = self.depth_selector.train(self.target_depth_selector_net)
            neighbor_loss = self.neighbor_selector.train(self.target_neighbor_selector_net)
        self.clear()
        return depth_loss, neighbor_loss
    
    def _eval(self):
        self.depth_selector._train = False
        self.neighbor_selector._train = False
    
    def _train(self):
        self.depth_selector._train = True
        self.neighbor_selector._train = True
    
    def is_full(self):
        return self.depth_selector.is_full() and self.neighbor_selector.is_full()
    
    def clear(self):
        self.neighbor_selector.clear()
        self.depth_selector.clear()

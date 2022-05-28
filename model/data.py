import os
import sys
DIR = os.path.dirname(os.path.abspath(__file__))
BASEDIR = os.path.dirname(DIR)
sys.path.append(BASEDIR)
sys.path.append(DIR)
import numpy as np
import tqdm
from tqdm import trange
from collections import namedtuple, Counter
from toolbox.data_loader import MUTAGData
# from toolbox.data_sampler import geo_dataset
import random
import torch
import pickle
from torch.nn import TransformerEncoderLayer, TransformerEncoder
from torch_geometric.nn import GCNConv, SAGEConv, GCN2Conv, GATConv, ECConv
from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T
from torch_geometric.utils import *
import torch.nn.functional as F
import torch.nn as nn
import networkx as nx
from scipy.sparse import csr_matrix
# import model.subgraph_utils
# from model.subgraph_utils import *
import math
from collections import defaultdict
from functools import partial
from copy import deepcopy

from sklearn.metrics import f1_score
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit

class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data

def load_subgraphs(data_dir):
    try:
        with open(f"{data_dir}_PPR_save.pth", 'rb') as f:
            k_hop_index = pickle.load(f)
    except FileNotFoundError:
        with open(f"{data_dir}_k_hop_save.pth", 'rb') as f:
            k_hop_index = pickle.load(f)
    
    if "DD" in data_dir:
        graphs = TUDataset("./data/raw", "DD")
    elif "NCI1" in data_dir:
        graphs = TUDataset("./data/raw", "NCI1")
    elif "PROTEINS" in data_dir:
        graphs = TUDataset("./data/raw", "PROTEINS")
    elif "IMDB-BINARY" in data_dir:
        graphs = TUDataset("./data/raw", "IMDB-BINARY")
    else:
        with open(f"{data_dir}_Graph.pth", 'rb') as f:
            graphs = pickle.load(f)

    if graphs.data.x is None:
        max_degree = 0
        degs = []
        for data in graphs:
            degs += [degree(data.edge_index[0], dtype=torch.long)]
            max_degree = max(max_degree, degs[-1].max().item())

        if max_degree < 1000:
            graphs.transform = T.OneHotDegree(max_degree)
        else:
            deg = torch.cat(degs, dim=0).to(torch.float)
            mean, std = deg.mean().item(), deg.std().item()
            graphs.transform = NormalizedDegree(mean, std)

    try:
        graph_num = graphs.graph_num
    except AttributeError:
        graph_num = len(graphs)

    if graphs[0].x == None:
        try:
            with open(f'{data_dir}_node_embedding.pth', 'rb') as f:
                graph_node_embedding = pickle.load(f)
        except FileNotFoundError:
            degree_list = torch.cat([degree(graphs[i].edge_index[0], graphs[i].num_nodes, torch.int) for i in range(graph_num)]).numpy().tolist()
            degree_set = list(set(degree_list))
            degree2index = {degree_set[i]:i for i in range(len(degree_set))}
            graph_node_embedding = [None for _ in range(graph_num)]
            for graph_index in range(graph_num):
                num_nodes, feature_size = graphs[graph_index].num_nodes, len(degree_set)
                graph_degree = degree(graphs[graph_index].edge_index[0], graphs[graph_index].num_nodes, torch.int).numpy().tolist()
                temp_fea = torch.zeros((num_nodes, feature_size))
                temp_fea[np.arange(num_nodes), [degree2index[d] for d in graph_degree]] = 1
                graphs[graph_index].x = temp_fea
                graph_node_embedding[graph_index] = temp_fea
            with open(f'{data_dir}_node_embedding.pth', 'wb') as f:
                pickle.dump(graph_node_embedding, f)

    try:
        with open(f'{data_dir}_k_hop_embedding.pth', 'rb') as f:
            k_hop_emb = pickle.load(f)
    except FileNotFoundError:
        k_hop_emb = [[] for _ in range(len(k_hop_index))]
        for graph_index in trange(len(k_hop_index)):
            for node_index in range(len(k_hop_index[graph_index])):
                if graphs[0].x == None:
                    k_hop_emb[graph_index].append([torch.stack([graph_node_embedding[graph_index][index] for index in neighbor]) if len(neighbor) else None\
                    for neighbor in k_hop_index[graph_index][node_index]])
                else:
                    k_hop_emb[graph_index].append([torch.stack([graphs[graph_index].x[int(index)] for index in neighbor]) if len(neighbor) else None\
                    for neighbor in k_hop_index[graph_index][node_index]])
        with open(f'{data_dir}_k_hop_embedding.pth', 'wb') as f:
            pickle.dump(k_hop_emb, f)

    shuffle = lambda x: np.random.shuffle(x)
    Node = namedtuple("node_state", ['node_embedding', 'node_index', 'neighbor', 'neighbor_embedding', 'graph_node_embedding', 'graph', 'graph_index', 'graph_label'])
    node_state_s = []
    node_state = []

    for graph_index in range(graph_num):
        graph = graphs[graph_index]
        for node_index in range(graphs[graph_index].num_nodes):
            try:
                node_embedding = graph.x[node_index] 
                node = Node(node_embedding=node_embedding, node_index=node_index, graph=graph, graph_node_embedding=graph.x, neighbor=k_hop_index[graph_index][node_index], neighbor_embedding=k_hop_emb[graph_index][node_index], graph_index=graph_index, graph_label=graph.y)
            except TypeError:
                node_embedding = torch.reshape(graph_node_embedding[graph_index][node_index], [1, -1])
                node = Node(node_embedding=node_embedding, node_index=node_index, graph=graph, graph_node_embedding=graph_node_embedding[graph_index], neighbor=k_hop_index[graph_index][node_index], neighbor_embedding=k_hop_emb[graph_index][node_index], graph_index=graph_index, graph_label=graph.y)
            node_state.append(node)
        shuffle(node_state)

    return node_state, graphs

def split(x, _train, _val, _test):
    train, val, test = len(x)*np.array([_train, _val, _test])/10
    _train, _val = int(np.floor(train)), int(np.floor(val))
    _test = len(x) - _train - _val
    split = lambda x: [x[0:_train], x[_train:_train+_val], x[_train+_val:_train+_val+_test]] # [train, val, test]
    return split(x)

def load_dataset(prefix='./data/preprocessed', dataset='MUTAG'):
    # load files
    data_dir =  f'{prefix}/{dataset}/{dataset}'
    node_state, graphs = load_subgraphs(data_dir)
    return node_state, graphs

def kfolds(node_state, k_fold, shuffle=True, random_state=12345):
    node_state_folds = KFold(k_fold, shuffle=shuffle, random_state=random_state).split(node_state)
    skf = StratifiedKFold(k_fold, shuffle=True, random_state=12345)

    test_indices, train_indices = [], []
    node_index2label = {node_state[_].graph_index: int(node_state[_].graph_label) for _ in range(len(node_state))}
    for _, idx in skf.split(torch.zeros(len(list(node_index2label.keys()))), list(node_index2label.values())):
        test_indices.append(torch.from_numpy(idx).to(torch.long))

    val_indices = [test_indices[i - 1] for i in range(k_fold)]

    train_indices = []
    for i in range(k_fold):
        train_mask = torch.ones(len(list(node_index2label.keys())), dtype=torch.bool)
        train_mask[test_indices[i]] = 0
        train_mask[val_indices[i]] = 0
        train_indices.append(train_mask.nonzero(as_tuple=False).view(-1))
        train_node, eval_node, test_node = [], [], []
        for node in tqdm.tqdm(node_state):
            if node.graph_index in list(train_indices[i]):
                train_node.append(node)
            elif node.graph_index in list(val_indices[i]):
                eval_node.append(node)
            elif node.graph_index in list(test_indices[i]):
                test_node.append(node)
        yield train_node, eval_node, test_node 

def graph_kfolds(node_state, k_fold, shuffle=True, random_state=2021):
    node_index2label = {node_state[_].graph_index: int(node_state[_].graph_label) for _ in range(len(node_state))}
    node_state_folds = StratifiedShuffleSplit(k_fold, random_state=random_state).split(list(node_index2label.keys()), y=list(node_index2label.values()))
    for train_graph_index, test_graph_index in node_state_folds:
        yield [node for node in node_state if node.graph_index in train_graph_index], [node for node in node_state if node.graph_index in test_graph_index]


if __name__ == "__main__":
    node_state = load_dataset(dataset='MUTAG')
    for train_states, test_states in kfolds(node_state, 10, shuffle=True, random_state=2021):
        pass


import os
import sys
BASEDIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASEDIR)
import numpy as np
import time
import random
import tqdm
from tqdm import trange
from collections import namedtuple, Counter
from toolbox.data_loader import MUTAGData
# from toolbox.data_sampler import geo_dataset
import random
import torch
import pickle
from torch.nn import TransformerEncoderLayer, TransformerEncoder, BCEWithLogitsLoss
from torch_geometric.nn import GCNConv, SAGEConv, GCN2Conv, GATConv, ECConv, global_mean_pool, GINConv
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
from torch_geometric.datasets import TUDataset
import torch_geometric.data as pyg_data
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
import os

class Net(torch.nn.Module):
    def __init__(self, max_layer, node_dim, hid_dim, out_dim, sub_num, sub_size, loss_type, sub_coeff, mi_coeff, device):
        super(Net, self).__init__()
        self.sub_coeff = sub_coeff
        self.mi_coeff = mi_coeff
        self.device = device
        self.conv1 = GINConv(
            Sequential(
                Linear(node_dim, hid_dim),
                ReLU(),
                Linear(hid_dim, hid_dim),
                ReLU(),
                BN(hid_dim),
            ), train_eps=False)
        self.conv2 = GINConv(
            Sequential(
                Linear(hid_dim, hid_dim),
                ReLU(),
                Linear(hid_dim, hid_dim),
                ReLU(),
                BN(hid_dim),
            ), train_eps=False)
        self.convs = torch.nn.ModuleList()
        for i in range(max_layer - 1):
            self.convs.append(
                GINConv(
                    Sequential(
                        Linear(hid_dim, hid_dim),
                        ReLU(),
                        Linear(hid_dim, hid_dim),
                        ReLU(),
                        BN(hid_dim),
                    ), train_eps=False))
        self.convs2 = torch.nn.ModuleList()
        for i in range(max_layer - 1):
            self.convs2.append(
                GINConv(
                    Sequential(
                        Linear(hid_dim, hid_dim),
                        ReLU(),
                        Linear(hid_dim, hid_dim),
                        ReLU(),
                        BN(hid_dim),
                    ), train_eps=False))
        self.lin1 = Linear(hid_dim*2, hid_dim)
        self.lin2 = Linear(hid_dim, out_dim)
        self.disc = Discriminator(hid_dim)
        self.b_xent = BCEWithLogitsLoss()
    
    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for conv in self.convs2:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
    
    def to(self, device):
        self.conv1.to(device)
        self.conv2.to(device)
        for conv in self.convs:
            conv.to(device)
        for conv in self.convs2:
            conv.to(device)
        self.lin1.to(device)
        self.lin2.to(device)
        self.disc.to(device)

    def forward(self, data, agent):
        # GNN
        x = self.conv1(data.x, data.edge_index)
        for conv in self.convs:
            x = conv(x, data.edge_index)

        # RL
        pool_x, sub_embedding, sub_labels = self.pool(x, data, agent)

        # loss_MI
        lbl = torch.cat([torch.ones_like(sub_labels), torch.zeros_like(sub_labels)], dim=0).float().to(self.device)
        logits = self.MI(pool_x, sub_embedding)
        loss_mi = self.b_xent(logits.view([1, -1]), lbl.view([1, -1]))

        # loss_sub
        sub_predict = F.log_softmax(self.lin2(torch.cat(sub_embedding, dim=0)), dim=-1)
        loss_sub = F.nll_loss(sub_predict, sub_labels.cuda())

        # loss_label 
        x = torch.cat((global_mean_pool(x, data.batch), pool_x), dim=-1)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        predicts = F.log_softmax(x, dim=-1)
        loss_label = F.nll_loss(predicts, data.y.view(-1).cuda())
        # using sub_predcit to reward RL learning
        # reward = sub_predict.max(1)[1].eq(sub_labels.cuda())
        # using graph_predict to reward RL learning
        reward = predicts.max(1)[1].eq(data.y.view(-1).cuda())
        loss = loss_label + self.sub_coeff * loss_sub + self.mi_coeff * loss_mi
        return predicts, reward, loss
    
    def MI(self, graph_embeddings, sub_embeddings):
        idx = torch.arange(graph_embeddings.shape[0]-1, -1, -1) 
        idx[len(idx)//2] = idx[len(idx)//2+1]
        shuffle_embeddings = torch.index_select(graph_embeddings, 0, idx.to(self.device))
        c_0_list, c_1_list = [], []
        for c_0, c_1, sub in zip(graph_embeddings, shuffle_embeddings, sub_embeddings):
            c_0_list.append(c_0.expand_as(sub))
            c_1_list.append(c_1.expand_as(sub))
        c_0, c_1, sub = torch.cat(c_0_list), torch.cat(c_1_list), torch.cat(sub_embeddings)
        return self.disc(sub, c_0, c_1)

    def pool(self, graph_node_embedding, data, agent):
        xs, labels = [], []
        for index, graph in enumerate(data.to_data_list()):
            sub_embedding = graph_node_embedding[(data.batch == index).nonzero().flatten()]
            sub_graph, edge_index = agent.predict(sub_embedding.cpu(), graph.cpu(), index)
            sub_graph, edge_index = sub_graph.to(self.device), edge_index.to(self.device)
            x = self.conv2(sub_graph.x.float(), sub_graph.edge_index)
            x = global_mean_pool(x, batch=sub_graph.batch)
            for conv in self.convs2:
                x = conv(x, edge_index)
            xs.append(x)
            labels.append(graph.y.expand(x.shape[0]))
        out = torch.stack([x.mean(0) for x in xs])
        return out, xs, torch.cat(labels)
    
    def pool_bak(self, big_graph_list, sketch_graph):
        xs = []
        for graph, edge_index in zip(big_graph_list, sketch_graph):
            x = self.conv2(graph.x.float(), graph.edge_index)
            x = global_mean_pool(x, batch=graph.batch)
            for conv in self.convs2:
                x = conv(x, edge_index)
            xs.append(x.mean(0))
        out = torch.stack(xs)
        return out

    def __repr__(self):
        return self.__class__.__name__

    def save(self, path):
        save_path = os.path.join(path, self.__class__.__name__+'.pt')
        torch.save(self.state_dict(), save_path)
        return save_path


class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        # c: 1, 512; h_pl: 1, 2708, 512; h_mi: 1, 2708, 512
        # c_x = torch.unsqueeze(c, 1)
        # c_x = c_x.expand_as(h_pl)

        c_x = c
        sc_1 = self.f_k(h_pl, c_x)
        sc_2 = self.f_k(h_mi, c_x)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 0)

        return logits
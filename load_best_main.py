import torch
import time
import argparse
import json
import gc
from tqdm import tqdm
from copy import deepcopy
import shutil
import numpy as np
import os
import random

from model.QLearning import QAgent
from model.data import load_dataset, kfolds, graph_kfolds
from model.Sugar import Net
from model.train_eval import learn, train, eval, test
from model.depth_selector import DepthSelector
from model.neighbor_selector import NeighborSelector
from model.agent_chain import AgentChain

from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.utils import *
from torch_geometric.loader import DataLoader, DenseDataLoader as DenseLoader
import torch_geometric.transforms as T

from torch.optim import Adam
import torch.nn.functional as F

from sklearn.model_selection import StratifiedKFold

from main import init_args, init_metric_saver, load_dataset, init_model

def test_GNN(args, 
            folds,
            graphs, 
            train_fold):

    for fold, (train_idx, test_idx,
               val_idx) in enumerate(zip(*k_fold(graphs, folds))):
        if fold != train_fold:
            continue
        # print(f"============================{fold}/{folds}==================================")
        train_dataset = graphs[train_idx]
        test_dataset = graphs[test_idx]
        val_dataset = graphs[val_idx]

        batch_size = args.batch_size
        if 'adj' in train_dataset[0]:
            train_loader = DenseLoader(train_dataset, batch_size, shuffle=True)
            eval_loader = DenseLoader(val_dataset, batch_size, shuffle=False)
            test_loader = DenseLoader(test_dataset, batch_size, shuffle=False)
        else:
            train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
            eval_loader = DataLoader(val_dataset, batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

        model, agent, optimizer = init_model(graphs, args)
        model.to(torch.device('cuda'))
        model.reset_parameters()

        best_depth_selector_net, \
        best_neighbor_slector_net = agent.snapshot()
        model, best_depth_selector_net, best_neighbor_slector_net = load_best_parameters(args.dataset, fold, model, best_depth_selector_net, best_neighbor_slector_net)
        agent.load_best(depth=best_depth_selector_net,
                        neighbor=best_neighbor_slector_net)
            
        test_acc, test_loss = test(test_loader, model, agent)
        print(f"test_acc:{test_acc} test_loss:{test_loss}")
    return test_acc

def load_best_args(dataset_name, fold):
    load_dir = os.path.join('./best_save/', dataset_name, "fold_{}".format(fold))
    all_dir_list = os.listdir(load_dir)
    load_dir = os.path.join(load_dir, all_dir_list[np.argmax([float(x) for x in all_dir_list])])
    print(load_dir)
    with open(load_dir + "/args.json", 'r') as f:
        args = json.load(f)
    return args

def load_best_parameters(dataset_name, fold, model, depth, neighbor):
    load_dir = os.path.join('./best_save/', dataset_name, "fold_{}".format(fold))
    all_dir_list = os.listdir(load_dir)
    load_dir = os.path.join(load_dir, all_dir_list[np.argmax([float(x) for x in all_dir_list])])
    model.load_state_dict(torch.load(load_dir + '/Net.pt'))
    depth.load_state_dict(torch.load(load_dir + '/depth.pt'))
    neighbor.load_state_dict(torch.load(load_dir + '/neighbor.pt'))
    return model, depth, neighbor


def k_fold(dataset, folds):
    skf = StratifiedKFold(folds, shuffle=True, random_state=12345)

    test_indices, train_indices = [], []
    for _, idx in skf.split(torch.zeros(len(dataset)), dataset.data.y):
        test_indices.append(torch.from_numpy(idx).to(torch.long))

    val_indices = [test_indices[i - 1] for i in range(folds)]

    for i in range(folds):
        train_mask = torch.ones(len(dataset), dtype=torch.bool)
        train_mask[test_indices[i]] = 0
        train_mask[val_indices[i]] = 0
        train_indices.append(train_mask.nonzero(as_tuple=False).view(-1))

    return train_indices, test_indices, val_indices

def meta_info(graphs):
    iso, size = [], []
    for index, graph in enumerate(graphs):
        if contains_isolated_nodes(graph.edge_index):
            iso.append(index)
        size.append(graph.x.shape[0])
    iso_np, size_np = np.array(iso), np.array(size)
    print(f"{graphs.name}: max_size:{size_np.max()} min_size:{size_np.min()} avg_size:{size_np.mean()} node_label:{graph.x.shape[-1]}")

def main(args=None, dataset="MUTAG", k_fold=10, fold=0):
    args = load_best_args(dataset, fold)
    if args is None:
        args = init_args()
    else:
        args = init_args(args)
                
    graphs = load_dataset(dataset_name=args.dataset)
    meta_info(graphs)
    return test_GNN(args, k_fold, graphs, fold)

if __name__ == "__main__":
    # dataset = "IMDB-BINARY"
    # dataset = "COLLAB"
    dataset = "PROTEINS"
    for fold in range(10):
        try:
            main(dataset=dataset, fold=fold)
        except FileNotFoundError:
            print(fold, 'error')
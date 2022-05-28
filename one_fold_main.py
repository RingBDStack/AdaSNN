import time
import argparse
import json
import gc
from tqdm import tqdm
from copy import deepcopy
import numpy as np
import os
import random

import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.utils import *
from torch_geometric.loader import DataLoader, DenseDataLoader as DenseLoader
import torch_geometric.transforms as T

from sklearn.model_selection import StratifiedKFold

from model.QLearning import QAgent
from model.data import load_dataset, kfolds, graph_kfolds
from model.Sugar import Net
from model.train_eval import learn, train, eval, test
from model.depth_selector import DepthSelector
from model.neighbor_selector import NeighborSelector
from torch.optim import Adam
from model.agent_chain import AgentChain
from toolbox.MetricSave import FoldMetricBase
from main import init_args, init_metric_saver, load_dataset, init_model, k_fold


def train_GNN(args, 
            folds,
            graphs, 
            metric_saver,
            train_fold):

    for fold, (train_idx, test_idx,
               val_idx) in enumerate(zip(*k_fold(graphs, folds))):
        if fold != train_fold:
            continue
        print(f"============================{fold}/{folds}==================================")
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
        # optimizer = Adam(model.parameters(), lr=args.lr, weight
        trange = tqdm(range(1, args.RL_episodes+1))
        best_acc = 0.0
        for r_episode in trange:
            learn_acc, learn_loss = learn(train_loader, model, agent, optimizer)
            eval_acc, eval_loss = eval(eval_loader, model, agent)
            if eval_acc > best_acc:
                best_depth_selector_net, \
                best_neighbor_slector_net = agent.snapshot()
                best_acc = eval_acc
                print('update RL !')
            print('learn_acc:', learn_acc, 'eval_acc:', eval_acc)
        
        agent.load_best(depth=best_depth_selector_net,
                        neighbor=best_neighbor_slector_net)
        test_acc, _ = test(test_loader, model, agent)
        del model, agent, optimizer
        gc.collect()
        print('==========the BEST test_acc on RL with test set is:', test_acc, '=============')

        model, agent, optimizer = init_model(graphs, args)
        agent.load_best(depth=best_depth_selector_net,
                        neighbor=best_neighbor_slector_net) 
        trange = tqdm(range(1, args.GNN_episodes+1))
        for i_episode in trange:
            train_acc, train_loss = train(train_loader, model, agent, optimizer)
            eval_acc, eval_loss = eval(eval_loader, model, agent)
            test_acc, test_loss = test(test_loader, model, agent)
            
            print(f"train_acc:{train_acc} train_loss:{train_loss} eval_acc:{eval_acc} eval_loss:{eval_loss}")
            if metric_saver.add_record(train_acc,
                                    train_loss,
                                    eval_acc,
                                    eval_loss,
                                    test_acc,
                                    test_loss) and args.save_RL:
                save(fold, args, metric_saver, best_neighbor_slector_net, best_depth_selector_net, model)
            if metric_saver.EarlyStopping():
                break
    return metric_saver.cur_saver.strict_best_acc
    # return metric_saver.cur_saver.best_acc
    # ans = metric_saver.save()

def save(fold, args, metric_saver, best_neighbor_slector_net, best_depth_selector_net, model):
    args.device = 0
    save_dir = os.path.join('./best_save/', args.dataset, "fold_{}".format(fold),"{:.5f}".format(metric_saver.cur_saver.strict_best_acc))
    metric_saver.cur_saver.save(save_dir, prefix='results: ')
    with open(save_dir + "/args.json", 'w') as f:
        json.dump(args.__dict__, f)
    best_depth_selector_net.save(save_dir+'/depth.pt')
    best_neighbor_slector_net.save(save_dir+'/neighbor.pt')
    model.save(save_dir)
    print(f"save to {save_dir}")

def meta_info(graphs):
    iso, size = [], []
    for index, graph in enumerate(graphs):
        if contains_isolated_nodes(graph.edge_index):
            iso.append(index)
        size.append(graph.x.shape[0])
    print(graphs)
    iso_np, size_np = np.array(iso), np.array(size)
    print(f"{graphs.name}: max_size:{size_np.max()} min_size:{size_np.min()} avg_size:{size_np.mean()} node_label:{graph.x.shape[-1]}")

def main(args=None, k_fold=10, fold=0):
    if args is None:
        args = init_args()
    else:
        args = init_args(args)
     
    graphs = load_dataset(dataset_name=args.dataset)
    meta_info(graphs)
    metric_saver = init_metric_saver(args.folds, 'test', args.comment, debug=args.tb)
    # train_RL(args, net, graphs, optimizer, agent)
    return train_GNN(args, k_fold, graphs, metric_saver, fold)

if __name__ == "__main__":
    main(fold=5)

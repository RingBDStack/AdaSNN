import time
import argparse
import json
import gc
from tqdm import tqdm
from copy import deepcopy
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
from toolbox.MetricSave import FoldMetricBase

import torch
import torch.nn.functional as F
from torch.optim import Adam

from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.utils import *
from torch_geometric.loader import DataLoader, DenseDataLoader as DenseLoader
import torch_geometric.transforms as T

from sklearn.model_selection import StratifiedKFold

class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data

def init_args(user_args=None):
    sub_num_dict = {
        "MUTAG": 18,
        "PTC_MR": 14,
        "PROTEINS": 40,
        "DD": 285,
        "NCI1": 30,
        "NCI109": 30,
        "ENZYNES": 33,
        "IMDB-BINARY": 20,
        "COLLAB": 75,
        "IMDB-MULTI": 20,
        "REDDIT-BINARY":20
    }
    parser = argparse.ArgumentParser(description='AdaSNN')

    parser.add_argument('--ablation_depth', type=int, default=0)
    parser.add_argument('--subgraph_num_delta', type=int, default=0)


    parser.add_argument('--record', type=int, default=1)
    parser.add_argument('--comment', type=str, default='debug')
    parser.add_argument('--tb', type=int, default=0, help="enable the tensorboard")

    parser.add_argument('--dataset', type=str, default="MUTAG")
    # parser.add_argument('--dataset', type=str, default="COLLAB")
    # parser.add_argument('--dataset', type=str, default="REDDIT-BINARY")
    # parser.add_argument('--dataset', type=str, default="PROTEINS")
    # parser.add_argument('--dataset', type=str, default="NCI109")
    # parser.add_argument('--dataset', type=str, default="NCI1")
    # parser.add_argument('--dataset', type=str, default="DD")
    # parser.add_argument('--dataset', type=str, default="IMDB-MULTI")
    # parser.add_argument('--dataset', type=str, default="IMDB-BINARY")
    # parser.add_argument('--dataset', type=str, default="REDDIT-MULTI-5K")


    parser.add_argument('--folds', type=int, default=10)
    parser.add_argument('--RL', type=int, default=0)
    parser.add_argument('--save_RL', type=int, default=1)
    parser.add_argument('--task', type=str, default='test')

    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=64)

    parser.add_argument('--GNN_episodes', type=int, default=100)
    parser.add_argument('--RL_episodes', type=int, default=50)
    parser.add_argument('--agent_episodes', type=int, default=150)

    parser.add_argument('--max_timesteps', type=int, default=5)
    parser.add_argument('--RL_lr', type=float, default=0.01)
    parser.add_argument('--RL_weight_decay', type=float, default=0.001)
    parser.add_argument('--RL_batch_size', type=int, default=64)

    # coeff
    parser.add_argument('--sub_coeff', type=float, default=0.2)
    parser.add_argument('--mi_coeff', type=float, default=0.5)

    # RL
    parser.add_argument('--replay_memory_size', type=int, default=100)
    parser.add_argument('--update_target_estimator_every', type=int, default=5)
    parser.add_argument('--mlp_layers', type=list, default=[64, 128, 256, 128, 64])
    parser.add_argument('--action_num', type=int, default=3)
    parser.add_argument('--fixed_k_hop', type=int, default=1)
    parser.add_argument('--discount_factor', type=float, default=0.95)
    parser.add_argument('--epsilon_start', type=float, default=1.)
    parser.add_argument('--epsilon_end', type=float, default=0.2)
    parser.add_argument('--epsilon_decay_steps', type=int, default=100)
    parser.add_argument('--norm_step', type=int, default=200)
    parser.add_argument('--hid_dim', type=int, default=128)

    args = parser.parse_args()
    if user_args is not None:
        for k, v in user_args.items():
            setattr(args, k, v)
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not hasattr(args, 'sub_num'):
        setattr(args, 'sub_num', sub_num_dict[args.dataset] + args.subgraph_num_delta)

    return args

def init_metric_saver(folds, dir_name, file_name, debug=False):
    metric_saver = FoldMetricBase(k_fold=folds, dir_name=dir_name, file_name=file_name, tb_server=debug)
    return metric_saver

# def balance(y, train_indices):
#     one_type, zero_type = [int(i) for i in train_indices if y[i]==1], [int(i) for i in train_indices if y[i]==0]
#     _min = min(len(one_type), len(zero_type))
#     balance_indices = one_type[:_min]+zero_type[:_min]
#     return torch.tensor(balance_indices)

def load_dataset(dataset_name="MUTAG"):
    graphs = TUDataset("./data/raw", dataset_name)
        
    if graphs.data.x is None:
        max_degree = 0
        degs = []
        for data in graphs:
            degs += [degree(data.edge_index[0], dtype=torch.long)]
            max_degree = max(max_degree, degs[-1].max().item())

        if max_degree < 2000:
            graphs.transform = T.OneHotDegree(max_degree)
        else:
            deg = torch.cat(degs, dim=0).to(torch.float)
            mean, std = deg.mean().item(), deg.std().item()
            graphs.transform = NormalizedDegree(mean, std)

    return graphs

def init_model(graphs, args, k_fold=10):
    net = Net(max_layer=2, 
                node_dim=graphs[0].x.shape[1], 
                hid_dim=args.hid_dim, 
                out_dim=graphs.num_classes, 
                sub_num=args.sub_num, 
                sub_size=15, 
                loss_type=0, 
                sub_coeff=args.sub_coeff,
                mi_coeff=args.mi_coeff,
                device=torch.device('cuda'))
    net.to(torch.device('cuda'))
    
    depth_selector = DepthSelector(action_num=args.action_num,
                                    fixed_k_hop=args.fixed_k_hop,
                                    lr=args.RL_lr,
                                    batch_size=args.RL_batch_size,
                                    state_shape=args.hid_dim*2,
                                    mlp_layers=args.mlp_layers,
                                    replay_memory_size=args.replay_memory_size,
                                    discount_factor=args.discount_factor,
                                    device=args.device)

    neighbor_selector = NeighborSelector(action_num=2, 
                                    fixed_k_hop=args.fixed_k_hop,
                                    lr=args.RL_lr, 
                                    batch_size=args.RL_batch_size,
                                    state_shape=args.hid_dim*2,
                                    mlp_layers=args.mlp_layers, 
                                    replay_memory_size=50*args.replay_memory_size,
                                    discount_factor=args.discount_factor,
                                    device=args.device)

    agent = AgentChain(update_target_estimator_every=args.update_target_estimator_every,
                     time_step=args.sub_num,
                     max_k_hop=args.action_num,
                     epochs=args.agent_episodes, 
                     ablation_depth=args.ablation_depth).bind_selector(depth=depth_selector, neighbor=neighbor_selector)
    # agent.visual = 1
    # agent.bind_selector(depth=depth_selector, neighbor=neighbor_selector)

    optimizer = torch.optim.Adam(net.parameters(),
                                          args.lr,
                                          weight_decay=args.weight_decay)
    return net, agent, optimizer

def train_GNN(args, 
            folds,
            graphs, 
            metric_saver):

    for fold, (train_idx, test_idx,
               val_idx) in enumerate(zip(*k_fold(graphs, folds))):
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

        model, agent, optimizer = init_model(graphs, args)
        agent.load_best(depth=best_depth_selector_net,
                        neighbor=best_neighbor_slector_net)
        trange = tqdm(range(1, args.GNN_episodes+1))
        for i_episode in trange:
            train_acc, train_loss = train(train_loader, model, agent, optimizer)
            eval_acc, eval_loss = eval(eval_loader, model, agent)
            test_acc, test_loss = test(test_loader, model, agent)
            
            print(f"train_acc:{train_acc} train_loss:{train_loss} eval_acc:{eval_acc} eval_loss:{eval_loss}")
            metric_saver.add_record(train_acc,
                                    train_loss,
                                    eval_acc,
                                    eval_loss,
                                    test_acc,
                                    test_loss)
        save(fold, args, metric_saver, best_neighbor_slector_net, best_depth_selector_net, model)
        metric_saver.next_fold()
    ans = metric_saver.save()

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

def k_fold(dataset, folds):
    skf = StratifiedKFold(folds, shuffle=True, random_state=12345)

    test_indices, train_indices = [], []
    for _, idx in skf.split(torch.zeros(len(dataset)), [int(x) for x in dataset.data.y]):
        test_indices.append(torch.from_numpy(idx).to(torch.long))

    val_indices = [test_indices[i - 1] for i in range(folds)]

    for i in range(folds):
        train_mask = torch.ones(len(dataset), dtype=torch.bool)
        train_mask[test_indices[i]] = 0
        train_mask[val_indices[i]] = 0
        # train_indices.append(balance(dataset.data.y, train_mask.nonzero(as_tuple=False).view(-1)))
        train_indices.append(train_mask.nonzero(as_tuple=False).view(-1))

    return train_indices, test_indices, val_indices

def meta_info(graphs):
    iso, size = [], []
    for index, graph in enumerate(graphs):
        if contains_isolated_nodes(graph.edge_index):
            iso.append(index)
        size.append(graph.x.shape[0])
    print(graphs)
    iso_np, size_np = np.array(iso), np.array(size)
    print(f"{graphs.name}: max_size:{size_np.max()} min_size:{size_np.min()} avg_size:{size_np.mean()} node_label:{graph.x.shape[-1]}")

def main(args=None, k_fold=10):
    if args is None:
        args = init_args()
    graphs = load_dataset(dataset_name=args.dataset)
    meta_info(graphs)
    metric_saver = init_metric_saver(args.folds, 'test', args.comment, debug=args.tb)
    train_GNN(args, k_fold, graphs, metric_saver)

if __name__ == "__main__":
    main()

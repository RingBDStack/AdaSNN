import os
import sys
BASEDIR = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
sys.path.append(BASEDIR)
from toolbox.data_loader import MUTAGData, NCI1Data, PTCData
import argparse
import torch_geometric
import os
import pickle
import numpy as np
import time
from multiprocessing import Process, Pool
from scipy.sparse import csr_matrix
# import numba as nb
# from numba import jit
import torch
import tqdm
from torch_geometric.utils import *
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import TUDataset
from torch_scatter import scatter_add
import threading
from functools import wraps
# class geo_dataset(TUDataset):
#     def __init__(self, *args, **kwargs):
#         super(geo_dataset, self).__init__(*args, **kwargs)
#         self.x = None
#         self.neighbor = None


def singlewrapper(sampler):
    """change output neighbor to [0], [1], [2] rather than [0], [0,1], [0,1,2]

    Arguments:
        sampler {func} -- the sampler

    Returns:
        [list] -- changed nbs
    """
    @wraps(sampler)
    def wrap_sampler(*args, **kwargs):
        nbs = sampler(*args, **kwargs)
        for node_index in range(len(nbs)):
            nb = nbs[node_index]
            single_nb = [nb[0]]
            for k_hop in range(1, len(nb)):
                single_nb.append(torch.from_numpy(np.array(list(set(nb[k_hop].numpy()) - set(nb[k_hop-1].numpy())))))
            nbs[node_index] = single_nb
        return nbs
    return wrap_sampler

@singlewrapper
def SAINT_sampler(graph, graph_index, saved_depth=5):
    adj = to_scipy_sparse_matrix(graph.edge_index).tocsr()
    nodes = np.arange(graph.num_nodes)

    _p_dist = np.array(
        [
            adj.data[
                adj.indptr[v] : adj.indptr[v + 1]
            ].sum()
            for v in nodes
        ],
        dtype=np.int64,
    )
    p_dist =  _p_dist/np.sum(_p_dist)

    node_num = len(nodes)
    sub_nodes = np.random.choice(np.arange(node_num), (b_szie), replace=False, p=p_dist)
    return sub_nodes

def node_neighbors(A, R, index, old):
    neighbors = A[index].argsort()[int(len(A[index]) - sum(A[index])):]  # using argsort to find all index that bigger that 0.0
    results = {int(key):float(val) for key, val in zip(neighbors, R[index][neighbors]) if key not in old.keys()}
    old.update(results)
    return  old

def sort_neighbors(neighbor_dict):
    neighbors_unsort = np.array(list(neighbor_dict.keys()))
    vals = np.array(list(neighbor_dict.values()))
    neighbors = neighbors_unsort[vals.argsort()[::-1]]
    return torch.from_numpy(neighbors)

def gen_PPR_hop(start_index, PPR_A, PI_PPR, saved_depth):
    # start_index = 0
    neighbors_dict = {}
    neighbors = [start_index]
    k_nb = []
    for depth in range(saved_depth):
        for index in neighbors:
            neighbors_dict = node_neighbors(PPR_A, PI_PPR, index, neighbors_dict)
            neighbors_dict.pop(start_index, None) # if start_index in neighbors, drop it
        neighbors = sort_neighbors(neighbors_dict)
        # k_nb.append([torch.tensor(start_index), *neighbors])
        k_nb.append(neighbors)
    return k_nb
    
@singlewrapper
def PPR_sampler(graph, graph_index, saved_depth=5):
    """using PPR mode to sample nodes

    Arguments:
        graph [torch_geometric.data] -- a graph of torch_geometric.data
        graph_index [int] -- the index of this graph in whole dataset

    Keyword Arguments:
        saved_depth {int} -- [description] (default: {5})

    Returns:
        [list] -- [node_index][saved_depth] = [neighbors]
    """
    edge_index = graph.edge_index
    num_nodes = graph.num_nodes

    degree_inv = degree(edge_index[0]).pow(-1)
    A = torch.squeeze(to_dense_adj(edge_index))

    alpha = 0.2
    PI_PPR = alpha * (torch.eye(num_nodes) - (1-alpha)*(degree_inv*A)).pow(-1)
    PI_PPR.masked_fill_(PI_PPR == float('inf'), 0)

    #--------------------------------------
    # using PI_PPR to generate neighbors
    PPR_A = PI_PPR.clone()
    PPR_A.masked_fill_(PPR_A!=0.00, 1)

    nbs = []
    for start_node in tqdm.tqdm(range(num_nodes), desc=f'subprocess on {graph_index} graph:'):
        nbs.append(gen_PPR_hop(start_node, PPR_A, PI_PPR, saved_depth))
        
    return nbs

@singlewrapper
def k_hop_sampler(graph, graph_index, saved_depth=5):
    """using k_hop mode to sample nodes

    Arguments:
        graph [torch_geometric.data] -- a graph of torch_geometric.data
        graph_index [int] -- the index of this graph in whole dataset

    Keyword Arguments:
        saved_depth {int} -- [description] (default: {5})

    Returns:
        [list] -- [node_index][saved_depth] = [neighbors]
    """
    num_nodes = graph.num_nodes
    nbs = []
    for start_index in tqdm.tqdm(range(num_nodes), desc=f'subprocess on {graph_index}:'):
        sub_nb = []
        for depth in range(saved_depth):
            neighbors, edge_index_connectivity, mapping, edge_mask = k_hop_subgraph(start_index, depth, graph.edge_index)
            sub_nb.append(neighbors)
        nbs.append(sub_nb)
    return nbs

def process_data(data, sampler, depth, num_worker, using_mp=False):
    """ using mutiprocessing to speed up subgraph generation

    Arguments:
        data {toch_geometric.data} -- A graph of dataset.
        sampler {function} -- 'k_hop', 'SAINT', 'PPR' or so on, 
        depth {int} -- specify how deep the subgraph saved
        num_worker {int} -- specify the total sum of subprocess, it should lower than cores of your cpus

    Returns:
        [type] -- [description]
    """
    t1 = time.time()
    # p = Pool(processes=num_worker)
    res = [None for _ in range(len(data))]
    if using_mp:
        for index in range(len(data)):
            p = Pool(processes=num_worker)
            process = p.apply_async(sampler, args=(data[index], index, depth))
            process.daemon = True
            res.append((index, process))
        p.close()
        p.join()
        # print(res[0].name)
        for index, process in res:
            # data[index].neighbor = process.get()
            res[index] = process.get()
            # data.data.neighbor[index] = process.get()
        # sub_nodes = [data[_[0]].sub_nodes = _[1].get() for _ in res]
    else:
        for index in range(len(data)):
                res[index] = sampler(data[index], index, depth)
                # data.data.neighbor[index] = sampler(data[index], index, depth)
                # data[index].neighbor = sampler(data[index], index, depth)
    t2 = time.time()
    T = "{:.2f} min, {:.2f} s".format((t2-t1)//60, (t2-t1)%60)
    print(f' {len(data)} of {data.name} graphs done, cost {T}')
    return res

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset Sampler")
    parser.add_argument('--dataset_parent_dir', type=str, default="./data/raw")
    # parser.add_argument('--dataset', type=str, default="REDDIT-BINARY")
    parser.add_argument('--dataset', type=str, default="MUTAG")
    # parser.add_argument('--dataset', type=str, default="PROTEINS")
    # parser.add_argument('--dataset', type=str, default="NCI109")
    # parser.add_argument('--dataset', type=str, default="NCI1")
    # parser.add_argument('--dataset', type=str, default="DD")
    # parser.add_argument('--dataset', type=str, default="IMDB-MULTI")
    # parser.add_argument('--dataset', type=str, default="REDDIT-MULTI-5K")
    # parser.add_argument('--dataset', type=str, default="PTC_MR")
    parser.add_argument('--num_worker', type=int, default=20)
    parser.add_argument('--sampler', type=str, choices=['k_hop', 'PPR', 'SAINT'], default='k_hop')
    parser.add_argument('--saved_depth', type=int,  default=10)
    parser.add_argument('--output_dir', type=str, default='./data/preprocessed')
    parser.add_argument('--using_mp', type=bool, default=False) # using Multi-processing to speed up, may cause system Error
    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, args.dataset)

    # check data dir
    if not os.path.exists(args.dataset_parent_dir):
        os.chdir('..')

    # ------------------------------------------------------------------------------------
    # chose dataset
    if args.dataset == "MUTAG":
        data = MUTAGData(args.dataset_parent_dir, args.dataset)
    elif args.dataset == "NCI1":
        data = NCI1Data(args.dataset_parent_dir, args.dataset)
    elif args.dataset == "PTC" or args.dataset == "PTC_MR":
        data = PTCData(args.dataset_parent_dir, args.dataset)
    elif args.dataset == "IMDB" or args.dataset == "IMDB-BINARY":
        # data = TUDataset(args.dataset_parent_dir, "IMDB-BINARY", use_node_attr=True)
        data = TUDataset(args.dataset_parent_dir, "IMDB-BINARY", use_node_attr=True)
    elif args.dataset == "REDDIT" or args.dataset == "REDDIT-BINARY":
        data = TUDataset(args.dataset_parent_dir, "REDDIT-BINARY")
    elif args.dataset == "PROTEINS":
        data = TUDataset(args.dataset_parent_dir, "PROTEINS")
    elif args.dataset == "NCI109":
        data = TUDataset(args.dataset_parent_dir, "NCI109")
    elif args.dataset == "DD":
        data = TUDataset(args.dataset_parent_dir, "DD")
    elif args.dataset == "IMDB-MULTI":
        data = TUDataset(args.dataset_parent_dir, "IMDB-MULTI")
    elif args.dataset == "REDDIT-MULTI-5K":
        data = TUDataset(args.dataset_parent_dir, args.dataset)

        # data = TUDataset(args.dataset_parent_dir, "REDDIT-BINARY")
        

    # -------------------------------------------------------------------------------------
    # chose sampler, using mutiprocessing to speed up
    if args.sampler == 'k_hop':
        k_hop_sampler(data[3], 10) # for test
        results = process_data(data, k_hop_sampler, args.saved_depth, args.num_worker, args.using_mp)
    elif args.sampler == 'PPR':
        PPR_sampler(data[3], 10) # for test
        results = process_data(data, PPR_sampler, args.saved_depth, args.num_worker, args.using_mp)
    elif args.sampler == 'SAINT':
        SAINT_sampler(data[3], 5) # for test
        results = process_data(data, SAINT_sampler, args.saved_depth, args.num_worker, args.using_mp)

    # ------------------------------------------------------------------------------------
    # save output
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(args.output_dir + f'/{args.dataset}_' + 'Graph.pth', 'wb') as f:
        pickle.dump(data, f)
    with open(args.output_dir + f'/{args.dataset}_' + args.sampler + f'_save.pth', 'wb') as f:
        pickle.dump(results, f)



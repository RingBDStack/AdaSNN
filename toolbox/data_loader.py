import numpy as np
from scipy import sparse
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
from torch_geometric.datasets import TUDataset
import torch
import os
import random
import argparse

class DataLoader(object):
    def __init__(self, data_parent_dir='./data/raw', dataset='MUTAG'):
        self.dataset = dataset
        self.data_dir = os.path.join(data_parent_dir, dataset) + '/'
        if not os.path.exists(self.data_dir):
            self.data_dir = self.data_dir+'raw/'
        assert os.path.exists(self.data_dir), f"can not find data in {self.data_dir}, check the position."


        self.raw_A, self.raw_A_str = self.load_A()
        self.raw_edge_labels, self.raw_edge_labels_str = self.load_edge_labels()
        self.raw_graph_indicator, self.raw_graph_indicator_str = self.load_graph_indicator()
        self.raw_graph_labels, self.raw_graph_labels_str = self.load_graph_labels()
        self.raw_node_labels, self.raw_node_labels_str = self.load_node_labels()

        self.node_num = len(self.raw_node_labels)

        self.edge_num = len(self.raw_A)

        self.node_type_num = len(set(self.raw_node_labels))
        self.edge_type_num = len(set(self.raw_edge_labels))
        self.graph_type_num = len(set(self.raw_graph_labels))

        self.map_node_type = {value: key for key, value in enumerate(set(self.raw_node_labels))}
        self.map_edge_type = {value: key for key, value in enumerate(set(self.raw_edge_labels))}
        self.map_graph_type = {value: key for key, value in enumerate(set(self.raw_graph_labels))}

    def load_A(self, filename='_A'):
        with open(self.txt(filename)) as f:
            raw_results = f.readlines()
        ###############################
        # results_str = [_.strip().split(',') for _ in raw_results]
        # results = np.array([[int(__) for __ in _.strip().split(',')] for _ in raw_results])
        ###############################
        try:
            results_str = [_.strip().split() for _ in raw_results]
            results = np.array([[int(__) for __ in _.strip().split()] for _ in raw_results])
        except ValueError:
            results_str = [_.strip().split(',') for _ in raw_results]
            results = np.array([[int(__) for __ in _.strip().split(',')] for _ in raw_results])

        return results, results_str
    
    def load_edge_labels(self, filename='_edge_labels'):
        if not os.path.exists(self.txt(filename)):
            print(f"{self.txt(filename)} not exitst!")
            return [0]*len(self.raw_A), ['0']*len(self.raw_A)

        with open(self.txt(filename)) as f:
            raw_results = f.readlines()
        results_str = [_.strip() for _ in raw_results]
        results = np.array([int(_.strip()) for _ in raw_results])
        return results, results_str
    
    def load_graph_indicator(self, filename='_graph_indicator'):
        return self.load_edge_labels(filename=filename)
        
    def load_graph_labels(self, filename='_graph_labels'):
        return self.load_edge_labels(filename=filename)

    def load_node_labels(self, filename='_node_labels'):
        return self.load_edge_labels(filename=filename)


    def txt(self, _name):
        data_path = self.data_dir + self.dataset + _name
        if not data_path.endswith('.txt'):
            data_path += '.txt'
        return data_path

    def load_npy(self, _dir):
        data_path = self.data_dir + _dir
        if not data_path.endwith('.txt'):
            pass
        return np.load(os.path.join(self.data_dir+_dir))

class MUTAGData(DataLoader):
    def __init__(self, data_parent_dir='./data/raw', dataset='MUTAG'):
        super(MUTAGData, self).__init__(data_parent_dir=data_parent_dir, dataset=dataset)

        self.node2graph, self.graph_num = self.FindNode2Graph()


        self.graph_edges, self.graph_edges_raw, self.graph_labels, \
        self.graph_edge_labels, self.graph_node_labels, self.node2pos = self.spiltData()

        self.data = self.wrapData()
        self.name = dataset

        
    def FindNode2Graph(self):
        """find which graph each node belongs to 
        !!! be careful of start index from 1 or 0
        return dict
        """
        node2graph = {}
        for index, graph_index in enumerate(self.raw_graph_indicator, start=1):
            node2graph[index] = graph_index - 1 # start at 1
        graph_num = max(node2graph.values())+1
        return node2graph, graph_num
    
    def spiltData(self):
        """split labels, edges, nodes to list (each graph)
        return list, ..., list
        """
        graph_edges = [[] for _ in range(self.graph_num)]
        graph_edges_raw = [[] for _ in range(self.graph_num)]
        graph_labels = [None for _ in range(self.graph_num)]
        graph_edge_labels = [[] for _ in range(self.graph_num)]
        graph_node_labels = [[] for _ in range(self.graph_num)]
        node2pos = {} # record each node'position in graph like node 135th is the 8th node of 7th graph, so we got {135:8}

        # node_label
        for index, node_label in enumerate(self.raw_node_labels, start=1):
            graph_node_labels[self.node2graph[index]].append(node_label)
            node2pos[index] = len(graph_node_labels[self.node2graph[index]])
        
        # graph_label
        for index, label in enumerate(self.raw_graph_labels):
            graph_labels[index] = label

        # edge_label 
        for edge, edge_label in zip(self.raw_A, self.raw_edge_labels):
            graph_index = self.node2graph[edge[0]]
            graph_edges[graph_index].append([node2pos[edge[0]], node2pos[edge[1]]])
            graph_edges_raw[graph_index].append([edge[0], edge[1]])
            graph_edge_labels[graph_index].append(edge_label)
        
        
        return graph_edges, graph_edges_raw, graph_labels, graph_edge_labels, graph_node_labels, node2pos
    
    def wrapData(self):
        data = []
        for graph_index in range(self.graph_num):
            x = self.gen_x(self.graph_node_labels[graph_index])
            edge_index = self.gen_edge_index(self.graph_edges[graph_index], len(self.graph_node_labels[graph_index]))
            edge_attr = self.gen_edge_attr(self.graph_edge_labels[graph_index])
            y = torch.from_numpy(np.array(self.map_graph_type[self.graph_labels[graph_index]]))
            Graph = Data(x=x, 
            edge_index=edge_index, 
            edge_attr=edge_attr, 
            y=y)
            data.append(Graph)
        return data
    
    def gen_x(self, graph_node_labels):
        node_num = len(graph_node_labels)
        x = np.zeros((node_num, self.node_type_num))
        for i in range(node_num):
            x[i][self.map_node_type[graph_node_labels[i]]] = 1
        return torch.from_numpy(x)
    
    def gen_edge_index(self, graph_edges, graph_node_num, dense=False):
        if dense:
            matrix = np.zeros((graph_node_num, graph_node_num))
            for edge in graph_edges:
                x, y = edge
                matrix[x-1, y-1] = 1
            edges = np.array(dense_to_sparse(torch.Tensor(matrix))[0])
        else:
            edges = (np.array(graph_edges)-1).T
        return torch.from_numpy(edges).long()
    
    def gen_edge_attr(self, graph_edge_labels):
        edge_num = len(graph_edge_labels)
        x = np.zeros((edge_num, self.edge_type_num))
        for i in range(edge_num):
            x[i][self.map_edge_type[graph_edge_labels[i]]] = 1
        return torch.from_numpy(x).long()
    
    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

class NCI1Data(MUTAGData):
    def __init__(self, data_parent_dir='./data/raw', dataset='NCI1'):
        super(NCI1Data, self).__init__(data_parent_dir=data_parent_dir, dataset=dataset)
    
class PTCData(MUTAGData):
    def __init__(self, data_parent_dir='./data/raw', dataset='PTC_MR'):
        super(PTCData, self).__init__(data_parent_dir=data_parent_dir, dataset=dataset)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset Sampler")
    parser.add_argument('--dataset', type=str, default="NCI1")
    args = parser.parse_args()
    # data_loader = DataLoader()
    if args.dataset == 'MUTAG':
        mutag = MUTAGData()
    elif args.dataset == 'NCI1':
        NCI1 = NCI1Data()
    pass
    
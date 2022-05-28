import numpy as np
from scipy import sparse
import os
import random

raw_prefix = '../data/raw/'
pre_prefix = '../data/preprocessed/'

def enhanced_save(filename, obj):
	file_path = os.path.dirname(filename)
	if not os.path.exists(file_path):
		os.makedirs(file_path)
		print(f'{file_path} created!')
	np.save(filename, obj)


def gen_graph_labels(dataname='MUTAG'):
	file_labels = open(raw_prefix + dataname + '/' + dataname + '_graph_labels.txt','r')
	labels = file_labels.readlines()
	file_labels.close()
	graph_labels = np.zeros(len(labels),dtype=int)
	for i, line in enumerate(labels):
		graph_labels[i] = int(line.strip().split(',')[0])
	graph_labels[graph_labels < 0] = 0
	enhanced_save(pre_prefix + dataname + '/' + dataname + '_graph_labels.npy', graph_labels)

def gen_graph_adjs_feats(dataname='MUTAG', central_node_num=5, sg_size=5, k_hop=3):
	graph_labels = np.load(pre_prefix + dataname + '/' + dataname + '_graph_labels.npy')
	g_num = len(graph_labels)

	file_edge = open(raw_prefix + dataname + '/' + dataname + '_A.txt', 'r')
	edges = file_edge.readlines() # 7442
	file_edge.close()
	file_indicator = open(raw_prefix + dataname + '/' + dataname + '_graph_indicator.txt', 'r')
	indicators = file_indicator.readlines() # 3371
	file_indicator.close()
	node_labels = open(raw_prefix + dataname + '/' + dataname + '_node_labels.txt', 'r') 
	nlabels = node_labels.readlines() # 3371
	node_labels.close()

	graph2nodes = [[] for i in range(g_num)]
	graph2node_num = {graph_id:0 for graph_id in range(g_num)}
	for i, line in enumerate(indicators):
		graph_id = int(line.strip().split(',')[0])
		graph2nodes[graph_id-1].append(i)
		graph2node_num[graph_id-1] += 1
	nodeid_old2new = [{old_id:new_id for new_id, old_id in zip(range(graph2node_num[i]),graph2nodes[i])} for i in range(g_num)]
	for i in range(1,g_num):
		graph2node_num[i] += graph2node_num[i-1]

	node2label_ = np.empty(len(nlabels),dtype=int)
	for i, line in enumerate(nlabels):
		node2label_[i] = int(line.strip()[0])
	feat_dim = len(set(node2label_))
	tmp_node2label = np.zeros((node2label_.shape[0], feat_dim))
	tmp_node2label[np.arange(node2label_.shape[0]), node2label_] = 1
	node2label = [[] for i in range(g_num)]
	node2label[0] = tmp_node2label[:graph2node_num[0]]
	for i in range(1,g_num):
		node2label[i] = tmp_node2label[graph2node_num[i-1]:graph2node_num[i]]

	graph2edges = [[[],[]] for i in range(g_num)]
	for i, line in enumerate(edges):
		# node_ids = [int(node_id)-1 for node_id in line.strip().split(', ')]
		node_ids = [int(node_id)-1 for node_id in line.strip().split('  ')]
		for j in range(len(graph2nodes)):
			if set(node_ids) < set(graph2nodes[j]):
				graph2edges[j][0].append(node_ids[0])
				graph2edges[j][1].append(node_ids[1])
				break

	# k_sub_adjs_sp = {i:{j:{k:[] for k in range(central_node_num)} for j in range(k_hop)} for i in range(g_num)}
	# k_sub_adjs_dense = np.empty((g_num, k_hop, central_node_num, sg_size, sg_size))
	# k_sub_feats = np.empty((g_num, k_hop, central_node_num, sg_size, feat_dim))
	# k_sub_sgnodes = np.empty((g_num, k_hop, central_node_num, sg_size))
	# k_sub_labels = np.empty((g_num, k_hop, central_node_num, 1))
	# for i in range(g_num):
	# 	for j in range(k_hop):
	# 		edges = [[nodeid_old2new[i][nodeid_old] for nodeid_old in graph2edges[i][m]] for m in [0,1]]
	# 		sp_adj = gen_k_spadj(j, edge2spadj(edges))
	# 		degrees = adj2degree(sp_adj.A)
	# 		central_nodes = degree2centres(degrees, central_node_num) # 非常厉害的算法
	# 		for n, centre in enumerate(central_nodes):
	# 			sg_nodes = gen_sg_nodes(centre, degrees[centre], sg_size, sp_adj)
	# 			sub_adj_sp, sub_adj_dense = gen_sub_adj(sp_adj, sg_nodes)
	# 			k_sub_adjs_sp[i][j][n] = sub_adj_sp
	# 			k_sub_adjs_dense[i][j][n] = sub_adj_dense
	# 			k_sub_feats[i][j][n] = node2label[i][sg_nodes]
	# 			k_sub_sgnodes[i][j][n] = np.array(sg_nodes) + graph2nodes[i][0]
	# 			k_sub_labels[i][j][n] = graph_labels[i]

	k_sub_adjs_sp = {i:{k:[] for k in range(central_node_num)} for i in range(g_num)}
	k_sub_adjs_dense = np.empty((g_num, central_node_num, sg_size, sg_size))
	k_sub_feats = np.empty((g_num, central_node_num, sg_size, feat_dim))
	k_sub_sgnodes = np.empty((g_num, central_node_num, sg_size))
	k_sub_labels = np.empty((g_num, central_node_num, 1))
	for i in range(g_num):
		edges = [[nodeid_old2new[i][nodeid_old] for nodeid_old in graph2edges[i][m]] for m in [0,1]]
		sp_adj = gen_k_spadj(0, edge2spadj(edges))
		degrees = adj2degree(sp_adj.A)
		central_nodes = degree2centres(degrees, central_node_num) # 非常厉害的算法
		for n, centre in enumerate(central_nodes):
			sg_nodes = gen_sg_nodes(centre, degrees[centre], sg_size, sp_adj)
			sub_adj_sp, sub_adj_dense = gen_sub_adj(sp_adj, sg_nodes)
			k_sub_adjs_sp[i][n] = sub_adj_sp
			k_sub_adjs_dense[i][n] = sub_adj_dense
			k_sub_feats[i][n] = node2label[i][sg_nodes]
			k_sub_sgnodes[i][n] = np.array(sg_nodes) + graph2nodes[i][0]
			k_sub_labels[i][n] = graph_labels[i]
	########################################################################################################
	save_folder = pre_prefix + dataname + '/'
	save_data_dict = {'k_sub_feats':'k_sgfeats', 'k_sub_adjs_sp':'k_sgadjs_sp', 'k_sub_adjs_dense':'k_sgadjs_dense', 'k_sub_sgnodes':'k_sgnodes', 'k_sub_labels':'k_sglabels'}
	save_data_list = ['graph2nodes', 'node2label', 'graph2edges']
	check(save_data_dict, save_data_list, locals().keys())	
	for k, v in locals().items():
		if k in save_data_dict.keys():
			saved_name = dataname + '_' + save_data_dict[k] + '.npy'
		elif k in save_data_list:
			saved_name = dataname + '_' + k + '.npy'
		else:
			continue
		enhanced_save(save_folder + saved_name, v)

def check(d, l, keys):
	for key in d.keys():
		assert key in keys, f"{key} in the saved dict is not exits, check spell"
	for key in l:
		assert key in keys, f"{key} in the saved list is not exits, check spell"

def edge2spadj(edges):
	edge_num = max(len(set(edges[0])), len(set(edges[1])))
	row = edges[0]
	col = edges[1]
	vals = np.ones(len(row))
	return sparse.csr_matrix((vals, (row, col)), shape=(edge_num, edge_num))

def adj2degree(adj):
	# adj = adj + np.eye(len(adj))
	adj[adj>0] = 1
	return np.sum(adj, axis=0, dtype=int)

def degree2centres(degrees, central_node_num):
	return np.argpartition(degrees, -central_node_num)[-central_node_num:]

def gen_k_spadj(k,k0_spadj):
	k_spadj = k0_spadj
	for i in range(k):
		k_spadj *= k0_spadj
	return k_spadj

def gen_sg_nodes(centre, degree, sg_size, sp_adj):
	sg_nodes = []
	if degree < sg_size:
		sg_nodes.extend(list(sp_adj.A[centre].nonzero()[0]))
		cand_list = list(set(range(len(sp_adj.A))) - set(sg_nodes))
		sg_nodes.extend(random.sample(cand_list, sg_size - degree))
	else:
		cand_list = list(sp_adj.A[centre].nonzero()[0])
		sg_nodes.extend(random.sample(cand_list, sg_size))
	return sg_nodes

def gen_sub_adj(sp_adj, sg_nodes):
	sub_adj_sp = [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]
	sg_nodes_ = {old_id:new_id for new_id, old_id in enumerate(sg_nodes)}
	for i in range(len(sp_adj.A)):
		for j in range(len(sp_adj.A)):
			if i in sg_nodes and j in sg_nodes and sp_adj.A[i,j] > 0.:
				sub_adj_sp[0].extend([sg_nodes_[i]]*int(sp_adj.A[i,j]))
				sub_adj_sp[1].extend([sg_nodes_[j]]*int(sp_adj.A[i,j]))
	sub_adj_dense = edge2spadj(sub_adj_sp)
	sub_adj_sp[0] = np.array(sub_adj_sp[0])
	sub_adj_sp[1] = np.array(sub_adj_sp[1])
	sub_adj_sp = np.vstack((np.expand_dims(sub_adj_sp[0], 0),np.expand_dims(sub_adj_sp[0], 0)))
	return sub_adj_sp, sub_adj_dense.A


def preprocess_entrance(dataname='MUTAG'):
	if not os.path.exists('../data') and os.path.exists('./data') and os.path.exists('./toolbox'):
		os.chdir('./toolbox')
	gen_graph_labels(dataname)
	gen_graph_adjs_feats(dataname,10,5,3)


if __name__ == '__main__':
	print('preprocess MUTAG...')
	preprocess_entrance('MUTAG')
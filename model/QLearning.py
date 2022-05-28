import numpy as np
import time
import itertools
import torch
from collections import Counter, defaultdict
import torch.nn as nn
from torch_geometric.utils import *
import tqdm
from collections import namedtuple
from copy import deepcopy
import random
import torch.nn.functional as F
from sklearn import preprocessing
from sklearn.metrics import f1_score

Transition = namedtuple('Transition', ['state', 'action_d', 'graph', 'action_b', 'reward', 'next_state', 'done'])

class Normalizer(object):
    def __init__(self):
        self.mean = None
        self.std = None
        self.state_memory = []
        self.max_size = 1000
        self.length = 0

    def normalize(self, s):
        if self.length == 0:
            return s
        return (s - self.mean) / (self.std + 1e-8)

    def append(self, s):
        if len(self.state_memory) > self.max_size:
            self.state_memory.pop(0)
        self.state_memory.append(s)
        self.mean = np.mean(self.state_memory, axis=0)
        self.std = np.std(self.state_memory, axis=0)
        self.length = len(self.state_memory)


class Memory(object):
    def __init__(self, memory_size, batch_size):
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memory = []

    def save(self, *args):
        if len(self.memory) == self.memory_size:
            self.memory.pop(0)
        transition = Transition(*args)
        if transition.done:
            self.memory.append(transition)

    def sample(self):
        samples = random.sample(self.memory, self.batch_size)
        # return map(np.array, zip(*samples))
        # Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])
        return map(list, zip(*samples))

class QAgent(object):
    def __init__(self,
                 replay_memory_size, replay_memory_init_size, update_target_estimator_every,
                 discount_factor, epsilon_start, epsilon_end, epsilon_decay_steps,
                 lr, batch_size,
                #  sg_num,
                 action_num,
                 norm_step,
                 mlp_layers,
                 state_shape,
                 device):
        self.replay_memory_size = replay_memory_size
        self.replay_memory_init_size = replay_memory_init_size
        self.update_target_estimator_every = update_target_estimator_every
        self.discount_factor = discount_factor
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

        # self.sg_num = sg_num
        self.state_shape = state_shape
        self.batch_size = batch_size
        self.action_num = action_num
        self.norm_step = norm_step
        self.device = device

        self.total_t = 0
        self.train_t = 0

        self.embedding_shape = [self.state_shape[0], self.state_shape[-1]*2]
        self.q_estimator = Estimator(action_num=action_num,
                                     lr=lr,
                                     state_shape=self.embedding_shape,
                                     mlp_layers=mlp_layers,
                                     device=device)
        self.target_estimator = Estimator(action_num=action_num,
                                          lr=lr,
                                          state_shape=self.embedding_shape,
                                          mlp_layers=mlp_layers,
                                          device=self.device)
        
        self.b_estimator = Estimator(action_num=2, 
                                     lr=lr, 
                                    #  state_shape=self.state_shape, 
                                     state_shape=self.embedding_shape, 
                                     mlp_layers=mlp_layers, 
                                     device=self.device)

        self.target_b_estimator = Estimator(action_num=2, 
                                     lr=lr, 
                                    #  state_shape=self.state_shape, 
                                     state_shape=self.embedding_shape, 
                                     mlp_layers=mlp_layers, 
                                     device=self.device)

        self.memory = Memory(replay_memory_size, batch_size)
        self.normalizer = Normalizer()


    def learn(self, env, total_timesteps):
        next_states = env.reset(env.train_nodes)# (7,256)

        trajectories = []
        for t in tqdm.tqdm(range(total_timesteps)):
            states = next_states 

            A = self.predict_batch(states)
            best_actions_depth = np.array([np.random.choice(np.arange(len(a)), p=a) for a in A])
            init_layer = 2
            graph = self.step_depth(best_actions_depth+init_layer, states)
      
            center_node = [int(torch.argmax(node.node_embedding).numpy()) for node in states] 
            center_node_results = np.argmax(A, 1)
            node_depth = defaultdict(list)
            for node, results in zip(center_node, center_node_results):
                node_depth[node].append(results)

            # b_RL
            sg_index_candidate, A, neighbor_embeddings = self.predict_batch_b(graph, states)
            best_actions_b, subgraph_index = self.randomchoice(sg_index_candidate, A)
            sg_adj, sg_x, sg_index_list, adj_len = self.gen_sg_adj_label(sg_index_candidate, states)

            next_states, rewards, dones, debug = env.step(sg_adj, sg_x, sg_index_list, states, best_actions_depth+init_layer)

            trajectories = zip(states, best_actions_depth, neighbor_embeddings, best_actions_b, rewards, next_states, dones)
            for each in trajectories:
                self.feed(each)

        loss = self.train()
        return loss, rewards, debug

    def step_depth(self, actions, states):
        if not isinstance(states, list):
            states = [states]
        sg_index = []
        NB = namedtuple("neighbors", ["index", "x"])
        for action, state in zip(actions, states):
            # action += 2
            neighbors = [NB(index=torch.from_numpy(np.array([state.node_index])), x=torch.reshape(state.node_embedding, [1, -1]))]
            for depth in range(action):
                indexes = state.neighbor[depth].long()
                embedding = state.neighbor_embedding[depth]
                if len(indexes) > 0:
                    neighbors.append(NB(index=indexes, x=embedding))
            sg_index.append(neighbors)
        return sg_index

    def randomchoice(self, sub_index, A):
        """
        """
        best_actions = []
        for graph_prob in A:
            graph_action = []
            for node_prob in graph_prob:
                graph_action.append(np.random.choice([0, 1], p=node_prob.numpy()))
            if len(graph_action) <= 4:
                graph_action = [1 for _ in range(len(graph_action))]
            else:
                while True: 
                    if Counter(graph_action)[1] > 4:
                        break
                    else:
                        graph_action = [np.random.choice([0, 1]) for _ in range(len(graph_action))]
            best_actions.append(graph_action)
        subgraph_index = [[index for index, p in zip(graph_index, action) if p == 1] for graph_index, action in zip(sub_index, best_actions)]
        return best_actions, subgraph_index

    def feed(self, ts):
        (state, best_actions_depth, graph, best_actions_b, reward, next_state, done) = tuple(ts)
        self.norm_step = -1
        if self.total_t < self.norm_step:
            self.feed_norm(state)
        else:
            self.feed_memory(state, best_actions_depth, graph, best_actions_b, reward, next_state, done)
        self.total_t += 1

    def print_time(self, str, first=False):
        if first:
            self.start_time = time.time()
            self.time = self.start_time
        else:
            # print(str, "{:.4f}, {:.4f}".format(time.time()-self.time, time.time()-self.start_time))
            self.time = time.time()

    def eval_step(self, states, init_layer=2, test=False, fixed_level=2):
        self.print_time("start:", first=True)
        A = self.predict_batch(states)
        self.print_time("0:")
        if test:
            best_actions_depth = np.ones((len(A)),dtype=np.int32) * fixed_level
        else:
            best_actions_depth = np.array([np.random.choice(np.arange(len(a)), p=a) for a in A])
        self.print_time("1:")
        graph = self.step_depth(best_actions_depth+init_layer, states)
        self.print_time("2:")

        # b_RL
        sg_index_candidate, A, neighbor_embeddings = self.predict_batch_b(graph, states)
        self.print_time("3:")
        best_actions_b, subgraph_index = self.randomchoice(sg_index_candidate, A)
        self.print_time("4:")
        if test:
            sg_adj, sg_index_list, sg_x, adj_len = self.gen_sg_adj_label(sg_index_candidate, states)
        else:
            sg_adj, sg_index_list, sg_x, adj_len = self.gen_sg_adj_label(subgraph_index, states)
        self.print_time("5:")
        return subgraph_index, sg_adj, sg_index_list, sg_x, best_actions_depth+init_layer

    def gen_sg_adj_label(self, sg_indexes, states):
        if not isinstance(states, list):
            states = [states]
        sg_adj = []
        sg_index_list = []
        sg_x = []
        adj_len = []
        for sg_index, state in zip(sg_indexes, states):
            index = [int(_.numpy()) for _ in sg_index]
            sg_index_list.append(index)
            sg_adj.append(subgraph(index, state.graph.edge_index, relabel_nodes=True)[0])
            adj_len.append(len(index))
            sg_x_list = [state.graph_node_embedding[i] for i in index]
            sg_x.append(torch.stack(sg_x_list))
        return sg_adj, sg_index_list, sg_x, adj_len

    def predict_batch(self, states):
        epsilon = self.epsilons[min(self.total_t, self.epsilon_decay_steps-1)]
        A = np.ones((len(states), self.action_num), dtype=float) * epsilon / self.action_num
        q_values = self.q_estimator.predict_nograd(states)
        best_action = np.argmax(q_values.cpu(), axis=1)
        for node_index, a in enumerate(best_action):
            A[node_index][a] += (1.0 - epsilon)
            A[node_index] = A[node_index] / sum(A[node_index])
        return A

    def predict_batch_b(self, graph, states):
        epsilon = self.epsilons[min(self.total_t, self.epsilon_decay_steps-1)]
        subgraph_index, probs, neighbor_embeddings = self.b_estimator.predict_b_nograd(graph, states)

        for graph_index, prob in enumerate(probs):
            for node_index, a in enumerate(prob):
                probs[graph_index][node_index][1] += epsilon
                probs[graph_index][node_index] = probs[graph_index][node_index] / sum(probs[graph_index][node_index])
        return subgraph_index, probs, neighbor_embeddings

    def train(self):
        state_batch, action_d_batch, graph_batch, action_b_batch, reward_batch, next_state_batch, done_batch = self.memory.sample()
        # depth-RL
        q_values_next = self.q_estimator.predict_nograd(next_state_batch)
        best_actions = np.argmax(q_values_next.cpu(), axis=1)
        q_values_next_target = self.target_estimator.predict_nograd(next_state_batch).cpu()

        DepthGetReward = lambda x: -0.2 if x else 0.5
        depth_reward = [DepthGetReward(reward) for reward in reward_batch]
        target_batch = torch.tensor(reward_batch) + self.discount_factor * q_values_next_target[np.arange(self.batch_size), best_actions]

        loss = self.q_estimator.update(state_batch, action_d_batch, target_batch)

        if self.train_t % self.update_target_estimator_every == 0:
            self.target_estimator = deepcopy(self.q_estimator)

        b_q_values_next = self.b_estimator.predict_b_train(graph_batch)
        b_q_values_next_target = self.target_b_estimator.predict_b_train(graph_batch)

        index = [key for key in range(len(graph_batch)) for depth in range(len(graph_batch[key])) for node in range(len(graph_batch[key][depth]))]
        b_reward_batch = np.array(reward_batch)[index]
        b_done_batch = np.array(done_batch)[index]

        NeighborGetReward = lambda x: -0.2 if x else 0.5
        neighbor_reward = [NeighborGetReward(reward) for reward in b_reward_batch]

        b_best_actions = torch.argmax(b_q_values_next, dim=1)

        target_b_batch = torch.tensor(neighbor_reward).to(self.device) + \
             self.discount_factor * b_q_values_next_target[np.arange(len(b_q_values_next_target)), b_best_actions]
        
        loss = self.b_estimator.update_b(graph_batch, action_b_batch, target_b_batch)

        if self.train_t % self.update_target_estimator_every == 0:
            self.target_b_estimator= deepcopy(self.b_estimator)
        self.train_t += 1
        return loss


    def feed_norm(self, state):
        self.normalizer.append(state)

    def feed_memory(self, *args):
        self.memory.save(*args)

class Estimator(object):
    def __init__(self,
                 action_num,
                 lr,
                 state_shape,
                 mlp_layers,
                 device):
        self.device = device
        qnet = EstimatorNetwork(action_num, state_shape, mlp_layers, device)
        qnet = qnet.to(device)
        self.qnet = qnet
        self.qnet.eval()
        for p in self.qnet.parameters():
            if len(p.data.shape) > 1:
                nn.init.xavier_uniform_(p.data)
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.optimizer = torch.optim.Adam(self.qnet.parameters(), lr=lr)

    def predict_nograd(self, states):
        if not isinstance(states, list):
            states = [states]
        with torch.no_grad():
            Q = []
            for state in states:
                node_embedding = state.node_embedding.view((1, -1))
                neighbor_embedding = torch.mean(state.neighbor_embedding[0], 0).view((1, -1))
                q_as = self.qnet(torch.cat((node_embedding, neighbor_embedding), 1))
                Q.append(q_as)
        return torch.cat(Q)

    def norm(self, feature):
        return feature

    def predict_b_nograd(self, state_k_graphs, states):
        NBs = []
        PRs = []
        EBs = []
        node2ans = defaultdict(list)
        with torch.no_grad():
            for graph, state in zip(state_k_graphs, states):
                NB_index = []
                prob = []
                eb = []
                previous_neighbor = state.node_embedding
                for neighbor in graph:
                    current_embedding = torch.cat((neighbor.x, self.norm(previous_neighbor).expand_as(neighbor.x)), -1)
                    eb.append(current_embedding)

                    _y = self.qnet(current_embedding)

                    previous_neighbor = torch.mean(neighbor.x, 0)

                    for node, ans in zip(torch.argmax(neighbor.x.cpu(), 1).numpy(), torch.argmax(_y.cpu(), 1).numpy()):
                        node2ans[node].append(ans)

                    y = F.softmax(_y, dim=1)
                    NB_index.extend([neighbor.index[i] for i in range(len(neighbor.index))])
                    prob.append(y)
                NBs.append(NB_index)
                PRs.append(prob)
                EBs.append(eb)
        return NBs, [torch.cat(_).cpu() for _ in PRs], EBs
        
    def predict_b_train(self, neighbor_embeddings):
        neighbor_embeddings = torch.cat([torch.cat([neighbor for neighbor in node]) for node in neighbor_embeddings])
        _y = self.qnet(neighbor_embeddings)
        return _y

    def update(self, _s, _a, _y):
        self.optimizer.zero_grad()
        self.qnet.train()
        # s = torch.from_numpy(s).float().to(self.device)
        s = torch.stack([torch.cat((i.node_embedding.view(1, -1), torch.mean(i.neighbor_embedding[1], 0).view(1, -1))) for i in _s]).float().to(self.device)
        a = torch.tensor(_a).long().to(self.device)
        y = _y.float().to(self.device)
        q_as = self.qnet(s)
        Q = torch.gather(q_as, dim=-1, index=a.unsqueeze(-1)).squeeze(-1)
        Q_loss = self.mse_loss(Q, y)
        Q_loss.backward()
        self.optimizer.step()
        Q_loss = Q_loss.item()
        self.qnet.eval()
        return Q_loss

    def update_b(self, _s, _a, _y):
        self.optimizer.zero_grad()
        self.qnet.train()
        s = torch.cat([i for batch in range(len(_s)) for i in _s[batch]]).float().to(self.device)
        a = torch.tensor([i for batch in range(len(_a)) for i in _a[batch]]).long().to(self.device)
        y = _y.float().to(self.device)
        q_as = self.qnet(s)
        Q = torch.gather(q_as, dim=-1, index=a.unsqueeze(-1)).squeeze(-1)
        Q_loss = self.mse_loss(Q, y)
        Q_loss.backward()
        self.optimizer.step()
        Q_loss = Q_loss.item()
        self.qnet.eval()
        return Q_loss

class EstimatorNetwork(nn.Module):
    def __init__(self,
                 action_num,
                 state_shape,
                 mlp_layers,
                 device=torch.device('cpu')):
        super(EstimatorNetwork, self).__init__()

        layer_dims = [state_shape[-1]] + mlp_layers
        self.device = device
        fc = [nn.Flatten()]
        for i in range(len(layer_dims)-1):
            fc.append(nn.Linear(layer_dims[i], layer_dims[i+1], bias=True))
            fc.append(nn.Tanh())
        fc.append(nn.Linear(layer_dims[-1], action_num, bias=True))
        self.fc_layers = nn.Sequential(*fc)

    def forward(self, states):
        states = states.float().to(self.device)
        return self.fc_layers(states)

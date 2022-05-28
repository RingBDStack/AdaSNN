import time

import torch
import torch.nn.functional as F
from torch import tensor
from torch.optim import Adam
from sklearn.model_selection import StratifiedKFold
from torch_geometric.loader import DataLoader, DenseDataLoader as DenseLoader
def learn(loader, model, agent, optimizer, recorder=None):
    correct, total_loss = 0, 0
    agent._train()
    model.train()
    for graph in loader:
        graph = graph.cuda()
        optimizer.zero_grad()
        predicts, reward, loss = model(graph, agent)
        # loss = F.nll_loss(predicts, graph.y.view(-1))
        loss.backward()
        correct_vector = predicts.max(1)[1].eq(graph.y.view(-1))

        # for visual
        # if recorder and reward is not None:
        #     r = (torch.ones_like(reward)*-0.5).masked_fill_(reward, 0.5).squeeze()
        #     recorder.append(r)

        agent.fed_reward(reward)
        if agent.is_full():
            depth_loss, neighbor_loss = agent.train()
            agent.clear()
        correct += correct_vector.sum().item()
        total_loss += loss.item() * num_graphs(graph)
        optimizer.step()
    train_acc = correct / len(loader.dataset)
    train_loss = total_loss / len(loader.dataset)
    return train_acc, train_loss

def train(loader, model, agent, optimizer):
    correct, total_loss = 0, 0
    agent._eval()
    model.train()
    for graph in loader:
        graph = graph.cuda()
        optimizer.zero_grad()
        predicts, reward, loss = model(graph, agent)
        # loss = F.nll_loss(predicts, graph.y.view(-1))
        loss.backward()
        # correct += predicts.max(1)[1].eq(graph.y.view(-1)).sum().item()
        correct_vector = predicts.max(1)[1].eq(graph.y.view(-1))
        # agent.fed_reward(correct_vector)
        correct += correct_vector.sum().item()
        total_loss += loss.item() * num_graphs(graph)
        optimizer.step()
    train_acc = correct / len(loader.dataset)
    train_loss = total_loss / len(loader.dataset)
    return train_acc, train_loss

def eval(loader, model, agent):
    correct, total_loss = 0, 0
    agent._eval()
    model.eval()
    with torch.no_grad():
        for graph in loader:
            graph = graph.cuda()
            predicts, reward, loss = model(graph, agent)
            # loss = F.nll_loss(predicts, graph.y.view(-1))
            correct += predicts.max(1)[1].eq(graph.y.view(-1)).sum().item()
            total_loss += loss.item() * num_graphs(graph)
        eval_acc = correct / len(loader.dataset)
        eval_loss = total_loss / len(loader.dataset)
    return eval_acc, eval_loss

def test(loader, model, agent):
    correct, total_loss = 0, 0
    agent._eval()
    model.eval()
    with torch.no_grad():
        for graph in loader:
            graph = graph.cuda()
            predicts, reward, loss = model(graph, agent)
            # loss = F.nll_loss(predicts, graph.y.view(-1))
            correct += predicts.max(1)[1].eq(graph.y.view(-1)).sum().item()
            total_loss += loss.item() * num_graphs(graph)
        test_acc = correct / len(loader.dataset)
        test_loss = total_loss / len(loader.dataset)
    return test_acc, test_loss

def num_graphs(data):
    if hasattr(data, 'num_graphs'):
        return data.num_graphs
    else:
        return data.x.size(0)
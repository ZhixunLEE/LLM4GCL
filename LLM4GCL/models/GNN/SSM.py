import copy
import torch
import random
import torch.nn as nn

from LLM4GCL.models import BareGNN

from torch_geometric.utils import k_hop_subgraph
from torch_geometric.utils import degree
from torch.utils.data import Subset, DataLoader

class random_subgraph_sampler(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, graph, center_node_budget, nei_budget, ids_per_cls):
        center_nodes_selected = self.node_sampler(ids_per_cls, center_node_budget)
        all_nodes_selected = self.nei_sampler(center_nodes_selected, graph, nei_budget)
        return center_nodes_selected, all_nodes_selected

    def node_sampler(self, ids_per_cls_train, budget, max_ratio_per_cls=1.0):
        store_ids = []
        for i, ids in enumerate(ids_per_cls_train):
            budget_ = min(budget, int(max_ratio_per_cls * len(ids))) if isinstance(budget, int) else int(
                budget * len(ids))
            store_ids.extend(random.sample(ids, budget_))
        return store_ids

    def nei_sampler(self, center_nodes_selected, graph, nei_budget):
        nodes_selected_current_hop = copy.deepcopy(center_nodes_selected)
        retained_nodes = copy.deepcopy(center_nodes_selected)
        for b in nei_budget:
            edge_index = graph.edge_index
            src, dst = edge_index
            neighbors = dst[torch.isin(src, torch.tensor(nodes_selected_current_hop))].tolist()
            nodes_selected_current_hop = random.choices(neighbors, k=b)
            retained_nodes.extend(nodes_selected_current_hop)
        return list(set(retained_nodes))
    

class degree_based_sampler(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, graph, center_node_budget, nei_budget, ids_per_cls):
        center_nodes_selected = self.node_sampler(ids_per_cls, center_node_budget)
        all_nodes_selected = self.nei_sampler(center_nodes_selected, graph, nei_budget)
        return center_nodes_selected, all_nodes_selected

    def node_sampler(self, ids_per_cls_train, budget, max_ratio_per_cls=1.0):
        store_ids = []
        for i, ids in enumerate(ids_per_cls_train):
            budget_ = min(budget, int(max_ratio_per_cls * len(ids))) if isinstance(budget, int) else int(
                budget * len(ids))
            store_ids.extend(random.sample(ids, budget_))
        return store_ids

    def nei_sampler(self, center_nodes_selected, graph, nei_budget):
        probs = degree(graph.edge_index[1], num_nodes=graph.num_nodes).float()
        nodes_selected_current_hop = copy.deepcopy(center_nodes_selected)
        retained_nodes = copy.deepcopy(center_nodes_selected)
        for b in nei_budget:
            if b == 0:
                continue
            edge_index = graph.edge_index
            src, dst = edge_index
            neighbors = dst[torch.isin(src, torch.tensor(nodes_selected_current_hop))].tolist()
            neighbors = list(set(neighbors) - set(retained_nodes))
            if len(neighbors) == 0:
                continue
            prob = probs[neighbors]
            sampled_neibs_ = torch.multinomial(prob, min(b, len(neighbors)), replacement=False).tolist()
            sampled_neibs = torch.tensor(neighbors)[sampled_neibs_]
            retained_nodes.extend(sampled_neibs.tolist())
        return list(set(retained_nodes))


samplers = {'uniform': random_subgraph_sampler(), 'degree':degree_based_sampler()}
class SSM(BareGNN):

    def __init__(self, task_loader, result_logger, config, checkpoint_path, dataset, model_name, seed, device):
        super(SSM, self).__init__(task_loader, result_logger, config, checkpoint_path, dataset, model_name, seed, device)

        self.sampler = samplers[config['ssm_sampler']]
        self.c_node_budget = config['ssm_c_node_budget']
        self.nei_budget = config['ssm_nei_budget']
        self.current_task = -1
        self.buffer_c_node = []
        self.buffer_all_nodes = []

    
    def train(self, curr_session, curr_epoch, model, text_dataset, train_loader, optimizer, class_num, config, device):
        id_by_class, train_idx = self.task_loader.id_by_class, self.task_loader.train_idx_per_task[curr_session]
        ids_per_cls = []
        for item in id_by_class.keys():
            ids = list(set(id_by_class[item]).intersection(set(train_idx)))
            if len(ids) == 0:
                continue
            ids_per_cls.append(ids)

        self.model.train()
        self.model.zero_grad()

        # size_g_train = labels.shape[0]
        data = text_dataset.data
        if curr_session != self.current_task:
            # store data from the current task
            self.current_task = curr_session
            c_nodes_sampled, nbs_sampled= self.sampler(data, self.c_node_budget, self.nei_budget, ids_per_cls)
            
            self.buffer_c_node.extend(c_nodes_sampled)
            self.buffer_all_nodes.append(nbs_sampled)

        if curr_session > 0:
            buffer_edge_index = copy.deepcopy(data.edge_index)
            src, dst = buffer_edge_index
            buffer_mask = torch.isin(dst, torch.tensor(self.buffer_c_node)) & torch.isin(src, torch.tensor(self.buffer_all_nodes))
            buffer_edge_index = buffer_edge_index[:, buffer_mask]
            _, train_edge_index, _, _ = k_hop_subgraph(train_idx, config['layer_num'], data.edge_index, relabel_nodes=False)
            edge_index = torch.cat([buffer_edge_index, train_edge_index], dim=1)
            edge_index = torch.unique(edge_index, dim=1)

            train_idx = train_idx + self.buffer_c_node
            train_dataset = Subset(text_dataset, train_idx)
            train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True)


        all_loss, train_num = 0., 0
        for _, batch in enumerate(train_loader):
            if batch['node_id'].size(0) < 2:
                break

            optimizer.zero_grad()
            subset, edge_index, mapping, _ = k_hop_subgraph(batch['node_id'], config['layer_num'], data.edge_index, relabel_nodes=True)
            logits = model(data.x[subset].to(device), edge_index.to(device))[mapping]

            logits = logits[:, :class_num]
            labels = batch['labels'].to(device)
            n_per_cls = [(labels == j).sum() for j in range(self.num_class)]
            loss_w = [1. / max(i, 1) for i in n_per_cls]
            loss_w = torch.tensor(loss_w[:class_num]).to(self.device)

            loss = self.loss_func(logits, labels, loss_w)
            loss.backward()
            optimizer.step()

            all_loss += loss * batch['node_id'].size(0)
            train_num += batch['node_id'].size(0)

        return all_loss / train_num
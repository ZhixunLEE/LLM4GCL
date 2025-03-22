import copy
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from LLM4GCL.models import BareGNN
from LLM4GCL.backbones import GCNNet, GATNet, SAGENet, SGCNet, SGConv
from LLM4GCL.utils import _save_checkpoint, _reload_best_model

from tqdm import tqdm
from torch import Tensor
from torch.nn import ModuleList
from torch_geometric.data import Data
from torch_geometric.nn.inits import glorot
from torch_geometric.utils import k_hop_subgraph, add_self_loops, degree


def add_edges(graph):
    num_nodes = graph.x.shape[0]
    graph = copy.deepcopy(graph)
    edge_index = graph.edge_index

    deg = degree(edge_index[0], num_nodes, dtype=torch.long)

    isolated_nodes = torch.where(deg == 1)[0].cpu().numpy()
    connected_nodes = torch.where(deg != 1)[0].cpu().numpy()

    if len(isolated_nodes) == 0 or len(connected_nodes) == 0:
        return edge_index

    random_nodes = np.random.choice(connected_nodes, size=len(isolated_nodes), replace=True)

    srcs = np.concatenate([isolated_nodes, random_nodes])
    dsts = np.concatenate([random_nodes, isolated_nodes])

    new_edges = torch.tensor([srcs, dsts], dtype=torch.long, device=edge_index.device)
    edge_index = torch.cat([edge_index, new_edges], dim=1)
    graph.edge_index = edge_index

    return graph



def drop_feature(x, drop_prob):
    drop_mask = torch.empty((x.size(1), ), dtype=torch.float32, device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0
    return x


def mask_edge(edge_index, drop_prob):
    edge_index = copy.deepcopy(edge_index)
    
    num_edges = edge_index.size(1)
    num_delete = int(drop_prob * num_edges)

    if num_delete > 0:
        delete_indices = np.random.choice(num_edges, num_delete, replace=False)
        delete_indices = torch.from_numpy(delete_indices).to(edge_index.device)

        src, dst = edge_index
        not_self_loop = src[delete_indices] != dst[delete_indices]
        delete_indices = delete_indices[not_self_loop]

        keep_mask = torch.ones(num_edges, dtype=torch.bool, device=edge_index.device)
        keep_mask[delete_indices] = False
        edge_index = edge_index[:, keep_mask]

    return edge_index


class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret


class ModelGrace(nn.Module):
    def __init__(self, model, num_hidden, num_proj_hidden, tau=0.5):
        super(ModelGrace, self).__init__()
        self.model = model
        self.tau = tau
        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)

    def forward(self, x, edge_index):
        output = self.model(x, edge_index)
        Z = F.elu(self.fc1(output))
        Z = self.fc2(Z)
        return Z

    def sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1, z2):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))
        return -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def batched_semi_loss(self, z1, z2, batch_size):
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []
        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]
            losses.append(-torch.log(between_sim[:, i * batch_size:(i + 1) * batch_size].diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))
            torch.cuda.empty_cache()
        return torch.cat(losses)

    def loss(self, h1, h2, batch_size):
        if batch_size is None:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)
        ret = (l1 + l2) * 0.5
        ret = ret.mean()
        return ret


def traingrace(modelgrace, graph, batch_size, drop_edge_prob, drop_feature_prob, epochs, lr, device):
    modelgrace.train()
    optimizer = torch.optim.Adam(modelgrace.parameters(), lr=lr, weight_decay=1e-5)

    progress_bar = tqdm(range(epochs))
    progress_bar.set_description(f'GRACE Pre-train')

    for _ in range(epochs):
        optimizer.zero_grad()
        graph_aug = mask_edge(graph.edge_index, drop_edge_prob)
        features_aug = drop_feature(graph.x, drop_feature_prob)
        Z1 = modelgrace(graph.x.to(device), graph.edge_index.to(device))
        Z2 = modelgrace(features_aug.to(device), graph_aug.to(device))
        loss = modelgrace.loss(Z1, Z2, batch_size=batch_size)
        loss.backward()
        optimizer.step()

        progress_bar.set_postfix({
            'Loss': f"{loss.item():.4f}",
        })

        progress_bar.update(1)
    progress_bar.close()


class SimplePrompt(nn.Module):
    def __init__(self, in_channels: int):
        super(SimplePrompt, self).__init__()
        self.global_emb = nn.Parameter(torch.Tensor(1, in_channels))
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.global_emb)

    def add(self, x: Tensor):
        return x + self.global_emb
    

class GPFplusAtt(nn.Module):
    def __init__(self, in_channels: int, p_num: int):
        super(GPFplusAtt, self).__init__()
        self.p_list = nn.Parameter(torch.Tensor(p_num, in_channels))
        self.a = nn.Linear(in_channels, p_num)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.p_list)
        self.a.reset_parameters()

    def add(self, x: Tensor):
        score = self.a(x)
        weight = F.softmax(score, dim=1)
        p = weight.mm(self.p_list)
        return x + p


class TPP(BareGNN):

    def __init__(self, task_loader, result_logger, config, checkpoint_path, dataset, model_name, seed, device):
        super(TPP, self).__init__(task_loader, result_logger, config, checkpoint_path, dataset, model_name, seed, device)

        self.grace_config = {
            'batch_size': config['TPP']['grace']['batch_size'],
            'drop_edge': config['TPP']['grace']['pe'],
            'drop_feature': config['TPP']['grace']['pf'],
            'epochs': config['TPP']['grace']['epochs'],
            'lr': float(config['TPP']['grace']['lr']),
        }

        num_promt = int(config['TPP']['prompts'])

        if num_promt < 2:
            prompt = SimplePrompt(self.feat_dim).to(device)
        else:
            prompt = GPFplusAtt(self.feat_dim, num_promt).to(device)

        class GNNModel(nn.Module):

            def __init__(self, gnn_type, input_dim, hidden_dim, output_dim, layer_num, dropout, num_heads, aggr, device):
                super(GNNModel, self).__init__()
                if gnn_type == 'GCN':
                    self.gnn = GCNNet(input_dim, hidden_dim, hidden_dim, layer_num, dropout).to(device)
                elif gnn_type == 'GAT':
                    self.gnn = GATNet(input_dim, hidden_dim, hidden_dim, layer_num, dropout, num_heads).to(device)
                elif gnn_type == 'SAGE':
                    self.gnn = SAGENet(input_dim, hidden_dim, hidden_dim, layer_num, dropout, aggr).to(device)
                elif gnn_type == 'SGC':
                    self.gnn = SGCNet(input_dim, hidden_dim, hidden_dim, layer_num, dropout).to(device)

            def forward(self, x, edge_index):
                x = self.gnn(x, edge_index)

                return x
            
        self.model = GNNModel(self.gnn_type, self.input_dim, self.hidden_dim, self.output_dim, self.layer_num, self.dropout, self.num_heads, self.aggr, self.device)
        cls_head = LogReg(self.hidden_dim, self.task_loader.session_size).cuda()
        self.classifiers = ModuleList([copy.deepcopy(cls_head) for _ in range(self.session_num)])
        self.prompts = ModuleList([copy.deepcopy(prompt) for _ in range(self.session_num - 1)])


    def pretrain(self, g):
        num_hidden = self.hidden_dim
        num_proj_hidden = 2 * num_hidden
        gracemodel = ModelGrace(self.model, num_hidden, num_proj_hidden, tau=0.5).cuda()

        traingrace(gracemodel, g, self.grace_config['batch_size'], self.grace_config['drop_edge'], self.grace_config['drop_feature'], self.grace_config['epochs'], self.grace_config['lr'], self.device)


    def getprototype(self, graph, train_ids, k=3):
        graph = add_edges(graph)
        neighbor_agg = SGConv(k=k).to(self.device)
        features = neighbor_agg(graph.x.to(self.device), graph.edge_index.to(self.device))
        degs = degree(graph.edge_index[0], graph.x.shape[0], dtype=torch.float).clamp(min=1)
        norm = torch.pow(degs, -0.5)
        norm = norm.to(features.device).unsqueeze(1)
        features = features * norm
        prototype = torch.mean(features[train_ids], dim=0)
        return prototype
    

    def gettaskid(self, curr_session, prototypes, graph, test_ids, k=3):
        if curr_session == 0:
            return 0
        graph = add_edges(graph)
        neighbor_agg = SGConv(k=k).to(self.device)
        features = neighbor_agg(graph.x.to(self.device), graph.edge_index.to(self.device))
        degs = degree(graph.edge_index[0], graph.x.shape[0], dtype=torch.float).clamp(min=1)
        norm = torch.pow(degs, -0.5)
        norm = norm.to(features.device).unsqueeze(1)
        features = features * norm

        testprototypes = torch.mean(features[test_ids], dim=0)

        testprototypes = testprototypes.cpu()

        dist = torch.norm(prototypes[0 : curr_session + 1] - testprototypes, dim=1)
        _, taskid = torch.min(dist, dim=0)
        return taskid.numpy()        
    

    def train(self, curr_session, curr_epoch, model, text_dataset, train_loader, optimizer, class_offset, config, device):
        model.eval()
        data = text_dataset.data
        cls_head = self.classifiers[curr_session]
        cls_head.train()

        if curr_session > 0:
            prompt = self.prompts[curr_session - 1]
            prompt.train()

        all_loss, train_num = 0., 0
        for _, batch in enumerate(train_loader):
            if batch['node_id'].size(0) < 2:
                break
            cls_head.zero_grad()
            if curr_session > 0:
                prompt.zero_grad()
            optimizer.zero_grad()
            
            subset, edge_index, mapping, _ = k_hop_subgraph(batch['node_id'], config['layer_num'], data.edge_index, relabel_nodes=True)
            if curr_session > 0:
                feats = prompt.add(data.x[subset].to(device))
            else:
                feats = data.x[subset]
            embeds = model(feats.to(device), edge_index.to(device))[mapping]

            logits = cls_head(embeds)
            labels = batch['labels'].to(device) - class_offset

            loss = self.loss_func(logits, labels)
            loss.backward()
            optimizer.step()
            all_loss += loss * batch['node_id'].size(0)
            train_num += batch['node_id'].size(0)

        return all_loss / train_num
    

    @torch.no_grad()
    def valid(self, curr_session, model, text_dataset, valid_loader, class_offset, config, device):
        return self.evaluate(curr_session, model, text_dataset, valid_loader, class_offset, config, device)

    @torch.no_grad()
    def evaluate(self, curr_session, model, text_dataset, test_loader, class_offset, config, device):
        model.eval()
        data = text_dataset.data
        cls_head = self.classifiers[curr_session]
        cls_head.eval()

        if curr_session > 0:
            prompt = self.prompts[curr_session - 1]
            prompt.eval()

        logits_list, preds_list, labels_list = [], [], []
        for _, batch in enumerate(test_loader):
            if batch['node_id'].size(0) < 2:
                break

            subset, edge_index, mapping, _ = k_hop_subgraph(batch['node_id'], config['layer_num'], data.edge_index, relabel_nodes=True)
            if curr_session > 0:
                feats = prompt.add(data.x[subset].to(device))
            else:
                feats = data.x[subset]
            embeds = model(feats.to(device), edge_index.to(device))[mapping]

            logits = cls_head(embeds)
            preds = torch.argmax(logits, dim=1)
            labels = batch['labels'] - class_offset

            logits_list.extend(logits)
            preds_list.extend(preds)
            labels_list.extend(labels)

        logits = torch.stack(logits_list, dim=0)
        preds = torch.stack(preds_list, dim=0)
        labels = torch.stack(labels_list, dim=0)

        acc, f1 = self.get_metric(logits, preds, labels)

        return acc, f1


    def fit(self, iter):
        prototypes = torch.zeros(self.session_num, self.feat_dim)

        optimizers = []
        for curr_session in range(self.session_num):
            model_param_group = []
            if curr_session == 0:
                model_param_group.append({"params":self.classifiers[curr_session].parameters()})
            else:
                model_param_group.append({"params": self.prompts[curr_session - 1].parameters()})
                model_param_group.append({"params": self.classifiers[curr_session].parameters()})
            optimizers.append(torch.optim.Adam(model_param_group, lr=float(self.config['lr']), weight_decay=float(self.config['weight_decay'])))

        for curr_session in range(self.session_num):
            session_size = self.task_loader.session_size
            class_num, text_dataset_iso, _, train_loader, valid_loader, test_loader_isolate, _ = self.task_loader.get_task(curr_session)
            class_offset = class_num - session_size
            optimizer = optimizers[curr_session]

            train_idx = self.task_loader.train_idx_per_task[curr_session]
            subset, edge_index, mapping, _ = k_hop_subgraph(train_idx, self.config['layer_num'], text_dataset_iso.data.edge_index, relabel_nodes=True)
            subgraph = Data(x=text_dataset_iso.data.x[subset], y=text_dataset_iso.data.y[subset], edge_index=edge_index)

            if curr_session == 0:
                self.pretrain(subgraph)

            progress_bar = tqdm(range(self.config['epochs']))
            progress_bar.set_description(f'Training | Iter {iter}')

            tolerate, best_acc_valid = 0, 0.
            for epoch in range(self.config['epochs']):
                loss = self.train(curr_session, epoch, self.model, text_dataset_iso, train_loader, optimizer, class_offset, self.config, self.device)
                progress_bar.write("Session: {} | Epoch: {} | Loss: {:.4f}".format(curr_session, epoch, loss))

                if epoch > 0 and epoch % self.config['valid_epoch'] == 0:
                    acc_valid, f1_valid = self.valid(curr_session, self.model, text_dataset_iso, valid_loader, class_offset, self.config, self.device)
                    progress_bar.write("Session: {} | Epoch: {} | Acc Val: {:.4f} | F1 Val: {:.4f} | Tolerate: {}".format(curr_session, epoch, acc_valid, f1_valid, tolerate))
                    if acc_valid > best_acc_valid:
                        tolerate = 0
                        best_acc_valid = acc_valid
                    else:
                        tolerate += 1
                        if tolerate > self.config['patience']: 
                            break

                progress_bar.set_postfix({
                    'Loss': f"{loss:.4f}",
                    'Best Valid ACC': f"{best_acc_valid:.4f}",
                    'Tolerate': tolerate
                })

                progress_bar.update(1)
            progress_bar.close()

            prototypes[curr_session] = self.getprototype(subgraph, mapping)

            test_idx = self.task_loader.test_idx_per_task_isolate[curr_session]
            subset, edge_index, mapping, _ = k_hop_subgraph(test_idx, self.config['layer_num'], text_dataset_iso.data.edge_index, relabel_nodes=True)
            subgraph = Data(x=text_dataset_iso.data.x[subset], y=text_dataset_iso.data.y[subset], edge_index=edge_index)

            taskid = self.gettaskid(curr_session, prototypes, subgraph, mapping)
            curr_acc_test_isolate, curr_f1_test_isolate = self.evaluate(taskid, self.model, text_dataset_iso, test_loader_isolate, class_offset, self.config, self.device)
            
            acc_list = []
            for s in range(curr_session):
                class_num, text_dataset_iso, _, _, _, test_loader_isolate, _ = self.task_loader.get_task(s)
                class_offset = class_num - session_size
                test_idx = self.task_loader.test_idx_per_task_isolate[s]
                subset, edge_index, mapping, _ = k_hop_subgraph(test_idx, self.config['layer_num'], text_dataset_iso.data.edge_index, relabel_nodes=True)
                subgraph = Data(x=text_dataset_iso.data.x[subset], y=text_dataset_iso.data.y[subset], edge_index=edge_index)

                taskid = self.gettaskid(s, prototypes, subgraph, mapping)

                prev_acc_test_isolate, prev_f1_test_isolate = self.evaluate(taskid, self.model, text_dataset_iso, test_loader_isolate, class_offset, self.config, self.device)
                acc_list.append(prev_acc_test_isolate)
            acc_list.append(curr_acc_test_isolate)

            print("Session: {} | Iso. Acc Test: {:.4f} | Iso. F1 Test: {:.4f}".format(curr_session, curr_acc_test_isolate, curr_f1_test_isolate))

            self.result_logger.add_new_results(acc_list, 0.0)

        return self.result_logger
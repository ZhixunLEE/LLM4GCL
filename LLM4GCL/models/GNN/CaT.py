import copy
import torch
import random
import torch.nn as nn
import torch.nn.functional as F

from LLM4GCL.models import BareGNN
from LLM4GCL.common.utils import _save_checkpoint, _reload_best_model

from tqdm import tqdm
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, add_self_loops, degree


def get_graph_class_ratio(data, train_idx, class_id):
    labels = data.y[train_idx]
    return (labels == class_id).sum().item() / len(labels)


class Linear(GCNConv):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels)
    
    def forward(self, x):
        return self.lin(x)


class Encoder(torch.nn.Module):
    def __init__(self, nin, nhid, nout, nlayers, activation=True):
        super().__init__()
        self.layers = torch.nn.ModuleList([])
        self.activation = activation

        if nlayers == 1:
            self.layers.append(Linear(nin, nout))
        else:
            self.layers.append(Linear(nin, nhid))
            for _ in range(nlayers - 2):
                self.layers.append(Linear(nhid, nhid))
            self.layers.append(Linear(nhid, nout))

    def initialize(self):
        for layer in self.layers:
            layer.reset_parameters()

    def encode_without_e(self, x):
        self.eval()
        for layer in self.layers[:-1]:
            x = layer(x)
            if self.activation:
                x = F.relu(x)
        return x

    def encode(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        x = self.layers[0].propagate(edge_index, x=x, edge_weight=norm)

        self.eval()
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            if self.activation:
                x = F.relu(x)

        return x


class CaT(BareGNN):

    def __init__(self, task_loader, result_logger, config, checkpoint_path, dataset, model_name, seed, device):
        super(CaT, self).__init__(task_loader, result_logger, config, checkpoint_path, dataset, model_name, seed, device)

        self.feat_init = config["cat_feat_init"]
        self.memory_bank = []
        self.budgets = self.assign_buget_per_cls(int(config["cat_budget"] * self.task_loader.data.x.shape[0]))
        self.feat_lr = float(config["cat_feat_lr"])
        self.hidden_dim = config['hidden_dim']
        self.emb_dim = config["cat_emb_dim"]
        self.layer_num = config['layer_num']
        self.n_encoders = config["cat_n_encoders"]


    def assign_buget_per_cls(self, budget):
        budgets = []
        for curr_session in range(self.session_num):
            budgets_at_task = []
            train_idx = self.task_loader.train_idx_per_task[curr_session]
            
            session_classes = torch.unique(self.task_loader.data.y[train_idx])

            for cls in session_classes:
                class_ratio = get_graph_class_ratio(self.task_loader.data, train_idx, cls)
                replay_cls_size = int(budget * class_ratio)
                if replay_cls_size == 0:
                    budgets_at_task.append(1)
                else:
                    budgets_at_task.append(replay_cls_size)
            gap = budget - sum(budgets_at_task)

            for i in range(gap):
                budgets_at_task[i % len(session_classes)] += 1
            budgets.append(budgets_at_task)

        return budgets
    

    def initialize_feature(self, curr_session, session_classes, budgets, feat_cond, method="randomChoice"):
        id_by_class, train_idx = self.task_loader.id_by_class, self.task_loader.train_idx_per_task[curr_session]
        ids_per_cls = [list(set(id_by_class[item]).intersection(set(train_idx))) for item in id_by_class.keys()]

        if method == "randomNoise":
            torch.nn.init.xavier_uniform_(feat_cond)

        elif method == "randomChoice":
            sampled_ids = []
            for i, cls in enumerate(session_classes):
                ids_at_cls = ids_per_cls[cls]
                sampled_ids += random.choices(ids_at_cls, k=budgets[i])

            sampled_feat = self.task_loader.data.x[sampled_ids]
            feat_cond.data.copy_(sampled_feat)

        return feat_cond
    

    def condense(self, curr_session, session_classes, train_loader, data, feat_cond, labels_cond, budgets):
        num_nodes = sum(budgets)
        self_loop_indices = torch.arange(num_nodes)
        edge_index_cond = torch.stack([self_loop_indices, self_loop_indices], dim=0)
        opt_feat = torch.optim.Adam([feat_cond], lr=self.feat_lr)

        progress_bar = tqdm(range(self.n_encoders))
        progress_bar.set_description(f'Generate Condense Graph | Session {curr_session}')

        encoder = Encoder(self.feat_dim, self.hidden_dim, self.emb_dim, self.layer_num).to(self.device)
        for _ in range(self.n_encoders):
            emb_real_list = []
            label_list = []
            encoder.initialize()
            with torch.no_grad():
                for _, batch in enumerate(train_loader):
                    if batch['node_id'].size(0) < 2:
                        break
                    
                    subset, edge_index, mapping, _ = k_hop_subgraph(batch['node_id'], self.config['layer_num'], data.edge_index, relabel_nodes=True)
                    emb_real = encoder.encode(data.x[subset].to(self.device), edge_index.to(self.device))[mapping]

                    emb_real = F.normalize(emb_real)
                    emb_real_list.extend(emb_real)
                    label_list.extend(batch['labels'])
                    
            emb_real = torch.stack(emb_real_list, dim=0)
            labels = torch.stack(label_list, dim=0)

            emb_cond = encoder.encode_without_e(feat_cond.to(self.device))
            emb_cond = F.normalize(emb_cond)

            loss = 0.
            for i, cls in enumerate(session_classes):
                real_emb_at_class = emb_real[labels == cls]
                cond_emb_at_class = emb_cond[labels_cond == cls]
                
                dist = torch.mean(real_emb_at_class, 0) - torch.mean(cond_emb_at_class, 0)
                loss += torch.sum(dist ** 2)
        
            opt_feat.zero_grad()
            loss.backward()
            opt_feat.step()

            progress_bar.set_postfix({
                'Loss': f"{loss.item():.4f}",
            })

            progress_bar.update(1)
        progress_bar.close()

        replayed_graph = Data(x=feat_cond.detach().cpu(), y=labels_cond, edge_index=edge_index_cond)

        return replayed_graph
    

    def memorize(self, curr_session, train_loader, data):
        labels_cond = []
        budgets = self.budgets[curr_session]
        train_idx = self.task_loader.train_idx_per_task[curr_session]    
        session_classes = torch.unique(self.task_loader.data.y[train_idx])

        for i, cls in enumerate(session_classes):
            labels_cond += [cls] * budgets[i]
        labels_cond = torch.tensor(labels_cond)

        feat_cond = torch.nn.Parameter(torch.FloatTensor(sum(budgets), self.feat_dim))
        feat_cond = self.initialize_feature(curr_session, session_classes, budgets, feat_cond, self.feat_init)
        replayed_graph = self.condense(curr_session, session_classes, train_loader, data, feat_cond, labels_cond, budgets)

        return replayed_graph
    

    def train(self, curr_session, curr_epoch, model, memory_data_list, optimizer, class_num, config, device):
        model.train()
        all_loss = 0.
        for memory_data in memory_data_list:
            optimizer.zero_grad()
            logits = model(memory_data.x.to(device), memory_data.edge_index.to(device))

            logits = logits[:, :class_num]
            labels = memory_data.y.to(device)
            n_per_cls = [(labels == j).sum() for j in range(self.num_class)]
            loss_w = [1. / max(i, 1) for i in n_per_cls]
            loss_w = torch.tensor(loss_w[:class_num]).to(self.device)

            loss = self.loss_func(logits, labels, loss_w)
            loss.backward()
            optimizer.step()
            all_loss += loss

        return all_loss / len(memory_data_list)


    def fit(self, iter):
        optimizer = self.get_optimizer(self.model)

        for curr_session in range(self.session_num):
            _, _, text_dataset_joint, train_loader, _, _, _ = self.task_loader.get_task(curr_session)
            replayed_graph = self.memorize(curr_session, train_loader, text_dataset_joint.data)
            self.memory_bank.append(replayed_graph)


        for curr_session in range(self.session_num):
            memory_data = self.memory_bank[:curr_session + 1]
            if curr_session != 0:
                _reload_best_model(self.model, self.checkpoint_path, self.dataset, self.model_name, self.seed)

            class_num, text_dataset_iso, text_dataset_joint, _, valid_loader, test_loader_isolate, test_loader_joint = self.task_loader.get_task(curr_session)

            progress_bar = tqdm(range(self.config['epochs']))
            progress_bar.set_description(f'Training | Iter {iter} | Session {curr_session}')

            tolerate, best_acc_valid = 0, 0.
            for epoch in range(self.config['epochs']):
                loss = self.train(curr_session, epoch, self.model, memory_data, optimizer, class_num, self.config, self.device)
                progress_bar.write("Session: {} | Epoch: {} | Loss: {:.4f}".format(curr_session, epoch, loss))

                if epoch > 0 and epoch % self.config['valid_epoch'] == 0:
                    acc_valid, f1_valid = self.valid(self.model, text_dataset_joint, valid_loader, class_num, self.config, self.device)
                    progress_bar.write("Session: {} | Epoch: {} | Acc Val: {:.4f} | F1 Val: {:.4f} | Tolerate: {}".format(curr_session, epoch, acc_valid, f1_valid, tolerate))
                    if acc_valid > best_acc_valid:
                        tolerate = 0
                        best_acc_valid = acc_valid
                        _save_checkpoint(self.model, optimizer, epoch, self.checkpoint_path, self.dataset, self.model_name, self.seed)
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

            _reload_best_model(self.model, self.checkpoint_path, self.dataset, self.model_name, self.seed)
            curr_acc_test_isolate, curr_f1_test_isolate = self.evaluate(self.model, text_dataset_iso, test_loader_isolate, class_num, self.config, self.device)
            curr_acc_test_joint, curr_f1_test_joint = self.evaluate(self.model, text_dataset_joint, test_loader_joint, class_num, self.config, self.device)

            acc_list = []
            for s in range(curr_session):
                _, text_dataset_iso, _, _, _, test_loader_isolate, _ = self.task_loader.get_task(s)
                prev_acc_test_isolate, prev_f1_test_isolate = self.evaluate(self.model, text_dataset_iso, test_loader_isolate, class_num, self.config, self.device)
                acc_list.append(prev_acc_test_isolate)
            acc_list.append(curr_acc_test_isolate)

            print("Session: {} | Iso. Acc Test: {:.4f} | Iso. F1 Test: {:.4f}".format(curr_session, curr_acc_test_isolate, curr_f1_test_isolate))
            print("Session: {} | Jot. Acc Test: {:.4f} | Jot. F1 Test: {:.4f}".format(curr_session, curr_acc_test_joint, curr_f1_test_joint))

            self.result_logger.add_new_results(acc_list, curr_acc_test_joint)

        return self.result_logger
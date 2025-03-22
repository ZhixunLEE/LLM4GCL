import torch
import torch.nn as nn
import torch.nn.functional as F

from LLM4GCL.backbones import GCNNet, GATNet, SAGENet, SGCNet
from LLM4GCL.models import BaseModel
from LLM4GCL.utils import _save_checkpoint, _reload_best_model

from tqdm import tqdm
from torch_geometric.utils import k_hop_subgraph

class cosine(BaseModel):

    def __init__(self, task_loader, result_logger, config, checkpoint_path, dataset, model_name, seed, device):
        super(cosine, self).__init__(task_loader, result_logger, config, checkpoint_path, dataset, model_name, seed, device)
        self.gnn_type = config['gnn']
        self.input_dim = self.feat_dim
        self.hidden_dim = config['hidden_dim']
        self.output_dim = self.num_class
        self.layer_num = config['layer_num']
        self.dropout = config['dropout']
        self.num_heads = config['num_heads']
        self.aggr = config['aggr']
        self.T = config['cosine']['T']
        self.sample_num = config['cosine']['sample_num']

        class GNNModel(nn.Module):

            def __init__(self, gnn_type, input_dim, hidden_dim, output_dim, layer_num, dropout, num_heads, aggr, T, device):
                super(GNNModel, self).__init__()
                self.T = T
                if gnn_type == 'GCN':
                    self.gnn = GCNNet(input_dim, hidden_dim, hidden_dim, layer_num, dropout).to(device)
                elif gnn_type == 'GAT':
                    self.gnn = GATNet(input_dim, hidden_dim, hidden_dim, layer_num, dropout, num_heads).to(device)
                elif gnn_type == 'SAGE':
                    self.gnn = SAGENet(input_dim, hidden_dim, hidden_dim, layer_num, dropout, aggr).to(device)
                elif gnn_type == 'SGC':
                    self.gnn = SGCNet(input_dim, hidden_dim, hidden_dim, layer_num, dropout).to(device)

                self.fc = nn.Linear(hidden_dim, output_dim).to(device)

            def forward(self, x, edge_index):
                x = self.gnn(x, edge_index)
                logits = self.fc(x)

                return logits
            
            def cosine_forward(self, x, edge_index):
                x = self.gnn(x, edge_index)
                x = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1))
                x = self.T * x

                return x

        self.model = GNNModel(self.gnn_type, self.input_dim, self.hidden_dim, self.output_dim, self.layer_num, self.dropout, self.num_heads, self.aggr, self.T, self.device)

    def update_proto(self, model, data, train_loader, task_classes, config, device):
        self.model.eval()
        embeds_list, labels_list = [], []
        for _, batch in enumerate(train_loader):
            if batch['node_id'].size(0) < 2:
                break
            subset, edge_index, mapping, _ = k_hop_subgraph(batch['node_id'], config['layer_num'], data.edge_index, relabel_nodes=True)
            embeds = model.gnn(data.x[subset].to(device), edge_index.to(device))[mapping]
            labels = batch['labels'].to(device)
            embeds_list.extend(embeds)
            labels_list.extend(labels)
        
        embeds = torch.stack(embeds_list, dim=0)
        labels = torch.stack(labels_list, dim=0)

        for class_index in task_classes:
            data_index = (labels == class_index).nonzero().squeeze(-1)
            class_embed = embeds[data_index]
            proto = class_embed.mean(0)
            self.model.fc.weight.data[class_index] = proto

    def train(self, curr_session, curr_epoch, model, text_dataset, train_loader, optimizer, class_num, config, device):
        model.train()
        data = text_dataset.data
        all_loss, train_num = 0., 0
        for _, batch in enumerate(train_loader):
            if batch['node_id'].size(0) < 2:
                break
            optimizer.zero_grad()
            
            subset, edge_index, mapping, _ = k_hop_subgraph(batch['node_id'], config['layer_num'], data.edge_index, relabel_nodes=True)
            logits = model(data.x[subset].to(device), edge_index.to(device))[mapping]

            logits = logits[:, :class_num]
            labels = batch['labels'].to(device)

            loss = self.loss_func(logits, labels)
            loss.backward()
            optimizer.step()
            all_loss += loss * batch['node_id'].size(0)
            train_num += batch['node_id'].size(0)

        return all_loss / train_num

    @torch.no_grad()
    def valid(self, model, text_dataset, valid_loader, class_num, config, device):
        data = text_dataset.data
        logits_list, preds_list, labels_list = [], [], []
        for _, batch in enumerate(valid_loader):
            if batch['node_id'].size(0) < 2:
                break

            subset, edge_index, mapping, _ = k_hop_subgraph(batch['node_id'], config['layer_num'], data.edge_index, relabel_nodes=True)
            logits = model(data.x[subset].to(device), edge_index.to(device))[mapping]

            logits = logits[:, :class_num]
            preds = torch.argmax(logits, dim=1)
            labels = batch['labels']

            logits_list.extend(logits)
            preds_list.extend(preds)
            labels_list.extend(labels)

        logits = torch.stack(logits_list, dim=0)
        preds = torch.stack(preds_list, dim=0)
        labels = torch.stack(labels_list, dim=0)

        acc, f1 = self.get_metric(logits, preds, labels)

        return acc, f1

    @torch.no_grad()
    def evaluate(self, model, text_dataset, test_loader, class_num, config, device):
        model.eval()
        data = text_dataset.data
        logits_list, preds_list, labels_list = [], [], []
        for _, batch in enumerate(test_loader):
            if batch['node_id'].size(0) < 2:
                break

            subset, edge_index, mapping, _ = k_hop_subgraph(batch['node_id'], config['layer_num'], data.edge_index, relabel_nodes=True)
            logits = model.cosine_forward(data.x[subset].to(device), edge_index.to(device))[mapping]

            logits = logits[:, :class_num]
            preds = torch.argmax(logits, dim=1)
            labels = batch['labels']

            logits_list.extend(logits)
            preds_list.extend(preds)
            labels_list.extend(labels)

        logits = torch.stack(logits_list, dim=0)
        preds = torch.stack(preds_list, dim=0)
        labels = torch.stack(labels_list, dim=0)

        acc, f1 = self.get_metric(logits, preds, labels)

        return acc, f1

    def fit(self, iter):
        optimizer = self.get_optimizer(self.model)

        class_num, text_dataset_iso, text_dataset_joint, train_loader, valid_loader, _, _ = self.task_loader.get_task(0)

        progress_bar = tqdm(range(self.config['epochs']))
        progress_bar.set_description(f'Training | Iter {iter}')

        tolerate, best_acc_valid = 0, 0.
        for epoch in range(self.config['epochs']):
            loss = self.train(0, epoch, self.model, text_dataset_iso, train_loader, optimizer, class_num, self.config, self.device)
            progress_bar.write("Session: {} | Epoch: {} | Loss: {:.4f}".format(0, epoch, loss))

            if epoch > 0 and epoch % self.config['valid_epoch'] == 0:
                acc_valid, f1_valid = self.valid(self.model, text_dataset_joint, valid_loader, class_num, self.config, self.device)
                progress_bar.write("Session: {} | Epoch: {} | Acc Val: {:.4f} | F1 Val: {:.4f} | Tolerate: {}".format(0, epoch, acc_valid, f1_valid, tolerate))
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
        class_offset = 0
        for curr_session in range(self.session_num):
            class_num, text_dataset_iso, text_dataset_joint, train_loader, _, test_loader_isolate, test_loader_joint = self.task_loader.get_task(curr_session, subset=self.sample_num)
            task_classes = [i for i in range(class_offset, class_num)]

            if curr_session != 0:
                self.update_proto(self.model, text_dataset_iso.data, train_loader, task_classes, self.config, self.device)
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
            class_offset = class_num

        return self.result_logger
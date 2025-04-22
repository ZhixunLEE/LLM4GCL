import torch
import torch.nn as nn
import torch.nn.functional as F

from LLM4GCL.backbones import GCNNet, GATNet, SAGENet, SGCNet
from LLM4GCL.models import BaseModel

from torch_geometric.utils import k_hop_subgraph

class BareGNN(BaseModel):

    def __init__(self, task_loader, result_logger, config, checkpoint_path, dataset, model_name, local_ce, seed, device):
        super(BareGNN, self).__init__(task_loader, result_logger, config, checkpoint_path, dataset, model_name, local_ce, seed, device)
        self.gnn_type = config['gnn']
        self.input_dim = self.feat_dim
        self.hidden_dim = config['hidden_dim']
        self.output_dim = self.num_class
        self.layer_num = config['layer_num']
        self.dropout = config['dropout']
        self.num_heads = config['num_heads']
        self.aggr = config['aggr']

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

                self.fc = nn.Sequential(
                    # nn.Linear(hidden_dim, hidden_dim),
                    # nn.ReLU(),
                    # nn.Dropout(p=0.1), 
                    nn.Linear(hidden_dim, output_dim),
                ).to(device)

            def forward(self, x, edge_index):
                x = self.gnn(x, edge_index)
                logits = self.fc(x)

                return logits
            
        self.model = GNNModel(self.gnn_type, self.input_dim, self.hidden_dim, self.output_dim, self.layer_num, self.dropout, self.num_heads, self.aggr, self.device)

    def train(self, curr_session, curr_epoch, model, text_dataset, train_loader, optimizer, class_src, class_dst, config, device):
        model.train()
        data = text_dataset.data
        all_loss, train_num = 0., 0
        for _, batch in enumerate(train_loader):
            if batch['node_id'].size(0) < 2:
                break
            optimizer.zero_grad()
            
            subset, edge_index, mapping, _ = k_hop_subgraph(batch['node_id'], config['layer_num'], data.edge_index, relabel_nodes=True)
            logits = model(data.x[subset].to(device), edge_index.to(device))[mapping]

            if self.local_ce:
                logits = logits[:, class_src : class_dst]
                labels = batch['labels'].to(device) - class_src
            else:
                logits = logits[:, : class_dst]
                labels = batch['labels'].to(device)

            loss = self.loss_func(logits, labels)
            loss.backward()
            optimizer.step()
            all_loss += loss * batch['node_id'].size(0)
            train_num += batch['node_id'].size(0)

        return all_loss / train_num

    @torch.no_grad()
    def valid(self, model, text_dataset, valid_loader, class_src, class_dst, config, device):
        model.eval()
        data = text_dataset.data
        logits_list, preds_list, labels_list = [], [], []
        for _, batch in enumerate(valid_loader):
            if batch['node_id'].size(0) < 2:
                break

            subset, edge_index, mapping, _ = k_hop_subgraph(batch['node_id'], config['layer_num'], data.edge_index, relabel_nodes=True)
            logits = model(data.x[subset].to(device), edge_index.to(device))[mapping]

            if self.local_ce:
                logits = logits[:, class_src : class_dst]
                labels = batch['labels'] - class_src
            else:
                logits = logits[:, : class_dst]
                labels = batch['labels']
            preds = torch.argmax(logits, dim=1)

            logits_list.extend(logits)
            preds_list.extend(preds)
            labels_list.extend(labels)

        logits = torch.stack(logits_list, dim=0)
        preds = torch.stack(preds_list, dim=0)
        labels = torch.stack(labels_list, dim=0)

        acc, f1 = self.get_metric(logits, preds, labels)

        return acc, f1

    @torch.no_grad()
    def evaluate(self, model, text_dataset, test_loader, class_dst, config, device):
        model.eval()
        data = text_dataset.data
        logits_list, preds_list, labels_list = [], [], []
        for _, batch in enumerate(test_loader):
            if batch['node_id'].size(0) < 2:
                break

            subset, edge_index, mapping, _ = k_hop_subgraph(batch['node_id'], config['layer_num'], data.edge_index, relabel_nodes=True)
            logits = model(data.x[subset].to(device), edge_index.to(device))[mapping]

            logits = logits[:, : class_dst]
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
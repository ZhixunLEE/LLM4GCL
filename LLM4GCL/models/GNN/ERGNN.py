import random
import torch
import torch.nn as nn

from LLM4GCL.models import BareGNN
from LLM4GCL.backbones import GCNNet, GATNet, SAGENet, SGCNet
from LLM4GCL.common.utils import _save_checkpoint, _reload_best_model

from tqdm import tqdm
from torch_geometric.utils import k_hop_subgraph


class CM_sampler(nn.Module):
    # sampler for ERGNN CM and CM*
    def __init__(self, plus):
        super().__init__()
        self.plus = plus

    def forward(self, ids_per_cls_train, budget, feats, reps, d, using_half=True):
        if self.plus:
            return self.sampling(ids_per_cls_train, budget, reps, d, using_half=using_half)
        else:
            return self.sampling(ids_per_cls_train, budget, feats, d, using_half=using_half)

    def sampling(self, ids_per_cls_train, budget, vecs, d, using_half=True):
        budget_dist_compute = 1000
        if using_half:
            vecs = vecs.half()

        ids_selected = []
        for i, ids in enumerate(ids_per_cls_train):
            other_cls_ids = list(range(len(ids_per_cls_train)))
            other_cls_ids.pop(i)
            ids_selected0 = ids_per_cls_train[i] if len(ids_per_cls_train[i]) < budget_dist_compute else random.choices(ids_per_cls_train[i], k=budget_dist_compute)

            dist = []
            vecs_0 = vecs[ids_selected0]
            for j in other_cls_ids:
                chosen_ids = random.choices(ids_per_cls_train[j], k=min(budget_dist_compute, len(ids_per_cls_train[j])))
                vecs_1 = vecs[chosen_ids]
                if len(chosen_ids) < 26 or len(ids_selected0) < 26:
                    # torch.cdist throws error for tensor smaller than 26
                    dist.append(torch.cdist(vecs_0.float(), vecs_1.float()).half())
                else:
                    dist.append(torch.cdist(vecs_0, vecs_1))

            #dist = [torch.cdist(vecs[ids_selected0], vecs[random.choices(ids_per_cls_train[j], k=min(budget_dist_compute,len(ids_per_cls_train[j])))]) for j in other_cls_ids]
            dist_ = torch.cat(dist,dim=-1) # include distance to all the other classes
            n_selected = (dist_<d).sum(dim=-1)
            rank = n_selected.sort()[1].tolist()
            current_ids_selected = rank[:budget]
            ids_selected.extend([ids_per_cls_train[i][j] for j in current_ids_selected])
        return ids_selected


samplers = {'CM': CM_sampler(plus=False), 'CM_plus':CM_sampler(plus=True)}
class ERGNN(BareGNN):

    def __init__(self, task_loader, result_logger, config, checkpoint_path, dataset, model_name, seed, device):
        super(ERGNN, self).__init__(task_loader, result_logger, config, checkpoint_path, dataset, model_name, seed, device)

        self.sampler = samplers[config['ergnn_sampler']]
        self.buffer_node_ids = []
        self.budget = int(config['ergnn_budget'])
        self.d_CM = float(config['ergnn_d']) # d for CM sampler of ERGNN

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

                return logits, x
            
        self.model = GNNModel(self.gnn_type, self.input_dim, self.hidden_dim, self.output_dim, self.layer_num, self.dropout, self.num_heads, self.aggr, self.device)


    def update_memory(self, curr_session, model, text_dataset, train_loader, config, device):
        # sample and store ids from current task
        id_by_class, train_idx = self.task_loader.id_by_class, self.task_loader.train_idx_per_task[curr_session]
        ids_per_cls = [list(set(id_by_class[item]).intersection(set(train_idx))) for item in id_by_class.keys()]

        data = text_dataset.data
        node_idx_list, feats_list, embeds_list = [], [], []
        for _, batch in enumerate(train_loader):
            if batch['node_id'].size(0) < 2:
                break

            subset, edge_index, mapping, _ = k_hop_subgraph(batch['node_id'], config['layer_num'], data.edge_index, relabel_nodes=True)
            _, embs = model(data.x[subset].to(device), edge_index.to(device))
            node_idx_list.extend(batch['node_id'].tolist())
            feats_list.extend(data.x[subset][mapping])
            embeds_list.extend(embs[mapping])

        feats = torch.stack(feats_list, dim=0)
        embeds = torch.stack(embeds_list, dim=0)

        node_idx_dict = {node_idx: i for i, node_idx in enumerate(node_idx_list)}
        ids_per_cls_new = []
        for sublist in ids_per_cls:
            if len(sublist) == 0:
                continue
            ids_curr_cls_new = []
            for item in sublist:
                ids_curr_cls_new.append(node_idx_dict[item])
            ids_per_cls_new.append(ids_curr_cls_new)

        sampled_ids = self.sampler(ids_per_cls_new, self.budget, feats, embeds, self.d_CM, using_half=False)
        sampled_ids_new = [node_idx_list[idx] for idx in sampled_ids]
        self.buffer_node_ids.extend(sampled_ids_new)
        subset, edge_index, mapping, _ = k_hop_subgraph(self.buffer_node_ids, config['layer_num'], data.edge_index, relabel_nodes=True)
        self.aux_feats, self.aux_edge_index, self.aux_labels, self.aux_mapping = data.x[subset].to(device), edge_index.to(device), data.y[subset][mapping].to(device), mapping.to(device)


    def train(self, curr_session, curr_epoch, model, text_dataset, train_loader, optimizer, class_num, config, device):
        self.model.train()
        data = text_dataset.data
        all_loss, train_num = 0., 0
        for _, batch in enumerate(train_loader):
            if batch['node_id'].size(0) < 2:
                break
            
            node_num_batch = batch['node_id'].size(0)
            buffer_size = len(self.buffer_node_ids)
            beta = buffer_size / (buffer_size + node_num_batch)

            optimizer.zero_grad()
            subset, edge_index, mapping, _ = k_hop_subgraph(batch['node_id'], config['layer_num'], data.edge_index, relabel_nodes=True)
            logits, _ = model(data.x[subset].to(device), edge_index.to(device))

            logits = logits[mapping][:, :class_num]
            labels = batch['labels'].to(device)
            n_per_cls = [(labels == j).sum() for j in range(self.num_class)]
            loss_w = [1. / max(i, 1) for i in n_per_cls]
            loss_w = torch.tensor(loss_w[:class_num]).to(self.device)

            loss = self.loss_func(logits, labels, loss_w)

            if curr_session != 0:
                aux_output, _ = self.model(self.aux_feats, self.aux_edge_index)
                aux_logits = aux_output[self.aux_mapping][:, :class_num]
                
                aux_n_per_cls = [(self.aux_labels == j).sum() for j in range(self.num_class)]
                aux_loss_w = [1. / max(i, 1) for i in aux_n_per_cls]
                aux_loss_w = torch.tensor(aux_loss_w[:class_num]).to(self.device)

                loss_aux = self.loss_func(aux_logits, self.aux_labels, aux_loss_w)
                loss = beta * loss + (1 - beta) * loss_aux

            loss.backward()
            optimizer.step()
            all_loss += loss * batch['node_id'].size(0)
            train_num += batch['node_id'].size(0)

        return all_loss / train_num
    
    @torch.no_grad()
    def valid(self, model, text_dataset, valid_loader, class_num, config, device):
        return self.evaluate(model, text_dataset, valid_loader, class_num, config, device)

    @torch.no_grad()
    def evaluate(self, model, text_dataset, test_loader, class_num, config, device):
        model.eval()
        data = text_dataset.data
        logits_list, preds_list, labels_list = [], [], []
        for _, batch in enumerate(test_loader):
            if batch['node_id'].size(0) < 2:
                break

            subset, edge_index, mapping, _ = k_hop_subgraph(batch['node_id'], config['layer_num'], data.edge_index, relabel_nodes=True)
            logits, _ = model(data.x[subset].to(device), edge_index.to(device))

            logits = logits[mapping][:, :class_num]
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

        for curr_session in range(self.session_num):
            if curr_session != 0:
                _reload_best_model(self.model, self.checkpoint_path, self.dataset, self.model_name, self.seed)

            class_num, text_dataset_iso, text_dataset_joint, train_loader, valid_loader, test_loader_isolate, test_loader_joint = self.task_loader.get_task(curr_session)

            progress_bar = tqdm(range(self.config['epochs']))
            progress_bar.set_description(f'Training | Iter {iter} | Session {curr_session}')

            tolerate, best_acc_valid = 0, 0.
            for epoch in range(self.config['epochs']):
                loss = self.train(curr_session, epoch, self.model, text_dataset_iso, train_loader, optimizer, class_num, self.config, self.device)
                progress_bar.write("Session: {} | Epoch: {} | Loss: {:.4f}".format(curr_session, epoch, loss))

                if epoch > 0 and epoch % self.config['valid_epoch'] == 0:
                    acc_valid, f1_valid = self.valid(self.model, text_dataset_iso, valid_loader, class_num, self.config, self.device)
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
            self.update_memory(curr_session, self.model, text_dataset_joint, train_loader, self.config, self.device)
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

import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from LLM4GCL.backbones import SGCNet
from LLM4GCL.models import BareGNN
from LLM4GCL.common.utils import _save_checkpoint, _reload_best_model

from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.loader import NeighborLoader


def match_loss(gw_syn, gw_real, dis_metric, device):
    dis = torch.tensor(0.0).to(device)
    if dis_metric == 'ours':
        for ig in range(len(gw_real)):
            gwr = gw_real[ig]
            gws = gw_syn[ig]
            dis += distance_wb(gwr, gws)
    elif dis_metric == 'mse':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = torch.sum((gw_syn_vec - gw_real_vec)**2)
    elif dis_metric == 'cos':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = 1 - torch.sum(gw_real_vec * gw_syn_vec, dim=-1) / (torch.norm(gw_real_vec, dim=-1) * torch.norm(gw_syn_vec, dim=-1) + 0.000001)
    else:
        exit('DC error: unknown distance function')

    return dis


def distance_wb(gwr, gws):
    shape = gwr.shape

    if len(gwr.shape) == 2:
        gwr = gwr.T
        gws = gws.T
    if len(shape) == 4: # conv, out*in*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2] * shape[3])
        gws = gws.reshape(shape[0], shape[1] * shape[2] * shape[3])
    elif len(shape) == 3:  # layernorm, C*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2])
        gws = gws.reshape(shape[0], shape[1] * shape[2])
    elif len(shape) == 2: # linear, out*in
        tmp = 'do nothing'
    elif len(shape) == 1: # batchnorm/instancenorm, C; groupnorm x, bias
        gwr = gwr.reshape(1, shape[0])
        gws = gws.reshape(1, shape[0])
        return 0
    dis_weight = torch.sum(1 - torch.sum(gwr * gws, dim=-1) / (torch.norm(gwr, dim=-1) * torch.norm(gws, dim=-1) + 0.000001))
    dis = dis_weight

    return dis


class GCondenser():

    def __init__(self, curr_session, class_num, ids_per_cls_train, input_dim, hidden_dim, output_dim, num_layers, dropout, graph, config, device):
        self.budget = config['delome_budget']
        self.epochs = config['delome_epochs']
        self.batch_size = config['delome_batch_size']
        self.feat_lr = float(config['delome_feat_lr'])
        self.layer_size = config['delome_layer_size']

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.curr_session = curr_session
        self.class_num = class_num
        self.ids_per_cls_train = ids_per_cls_train
        self.graph = graph
        labels = self.graph.y[self.graph.mapping]
        self.class_ids = labels.unique().tolist()

        self.syn_node_num = 0
        for i in range(len(self.class_ids)):
            self.syn_node_num = self.syn_node_num + min(self.budget, len(self.ids_per_cls_train[i]))

        self.syn_feats = nn.Parameter(torch.FloatTensor(self.syn_node_num, self.graph.x.shape[1]).to(device))
        self.syn_labels = torch.LongTensor(self.generate_labels_syn()).to(device)
        self.optimizer_feat = torch.optim.Adam([self.syn_feats], lr=self.feat_lr)

        self.device = device
        self.reset_parameters()


    def reset_parameters(self):
        self.syn_feats.data.copy_(torch.randn(self.syn_feats.size()))
    

    def generate_labels_syn(self):
        num_class_dict = {}
        syn_labels = []
        self.syn_class_indices = {}
        for i, cls in enumerate(self.class_ids):
            num_class_dict[cls] = min(self.budget, len(self.ids_per_cls_train[i]))
            self.syn_class_indices[cls] = [len(syn_labels), len(syn_labels) + num_class_dict[cls]]
            syn_labels += [cls] * num_class_dict[cls]
        self.num_class_dict = num_class_dict

        return syn_labels
    

    def get_condense_graph(self, ):
        cond_feat, cond_label, cond_edge_index = self.train()
        graph = Data(x=cond_feat, y=cond_label, edge_index=cond_edge_index)

        return graph


    def get_Igraph_feat(self, features):
        idx_selected = []
        for i, cls in enumerate(self.class_ids):
            tmp = random.sample(self.ids_per_cls_train[i], self.num_class_dict[cls])
            tmp = list(tmp)
            idx_selected = idx_selected + tmp
        idx_selected = np.array(idx_selected).reshape(-1)
        features = features[idx_selected]

        return features
    

    def train(self, ):

        edge_index = torch.tensor([[i for i in range(self.syn_node_num)], [i for i in range(self.syn_node_num)]], dtype=torch.long)
        syn_class_indices = self.syn_class_indices
        feat_sub = self.get_Igraph_feat(self.graph.x)
        self.syn_feats.data.copy_(feat_sub)
        Igraph = Data(edge_index=edge_index).to(self.device)

        progress_bar = tqdm(range(self.epochs + 1))
        progress_bar.set_description(f'Condensing Graph | Session {self.curr_session}')

        for epoch in range(self.epochs + 1):
            all_loss, sample_num = 0., 0
            model = SGCNet(self.input_dim, self.hidden_dim, self.output_dim, self.num_layers, self.dropout).to(self.device)
            model.reset_params()
            model_parameters = list(model.parameters())
            model.train()

            for i, cls in enumerate(self.class_ids):
                if cls not in self.num_class_dict:
                    continue
                dataloader = NeighborLoader(self.graph, num_neighbors=self.layer_size, input_nodes=self.ids_per_cls_train[i], batch_size=self.batch_size, shuffle=True)

                for samples in dataloader:
                    model.zero_grad()
                    output = model(samples.x.to(self.device), samples.edge_index.to(self.device))[samples.input_id]
                    loss_real = F.nll_loss(output[:, : self.class_num], samples.y[samples.input_id].to(self.device))

                    gw_real = torch.autograd.grad(loss_real, model_parameters)
                    gw_real = list((_.detach().clone() for _ in gw_real))
                    
                    ind = syn_class_indices[cls]
                    syn_output = model(self.syn_feats.to(self.device), Igraph.edge_index.to(self.device))

                    loss_syn = F.nll_loss(syn_output[ind[0] : ind[1], : self.class_num], self.syn_labels[ind[0]: ind[1]].to(self.device))
                    gw_syn = torch.autograd.grad(loss_syn, model_parameters, create_graph=True)

                    coeff = self.num_class_dict[cls] / max(self.num_class_dict.values())
                    loss = coeff * match_loss(gw_syn, gw_real, dis_metric='mse', device=self.device)

                    self.optimizer_feat.zero_grad()
                    loss.backward()
                    self.optimizer_feat.step()

                    all_loss += loss * len(samples.input_id)
                    sample_num += len(samples.input_id)

            if epoch == self.epochs:
                out_syn_feat, out_syn_labels = self.syn_feats.detach(), self.syn_labels
                self_loop_indices = torch.arange(out_syn_feat.shape[0])
                out_syn_edge_index = torch.stack([self_loop_indices, self_loop_indices], dim=0)

            progress_bar.set_postfix({
                'Loss': f"{all_loss / sample_num:.4f}",
            })

            progress_bar.update(1)
        progress_bar.close()

        return out_syn_feat, out_syn_labels, out_syn_edge_index


class DeLoMe(BareGNN):

    def __init__(self, task_loader, result_logger, config, checkpoint_path, dataset, model_name, seed, device):
        super(DeLoMe, self).__init__(task_loader, result_logger, config, checkpoint_path, dataset, model_name, seed, device)
        
        self.current_task = -1
        self.cond_num = {}
        self.budget = int(config['delome_budget'])
        self.tro = config['delome_tro']
        self.aux_g = []
        self.adjustments = 0
        self.aux_loss_w_ = []

    def train(self, curr_session, curr_epoch, model, text_dataset, train_loader, optimizer, class_num, config, device):
        data = text_dataset.data
        if curr_session != self.current_task:
            self.current_task = curr_session

            id_by_class, train_idx = self.task_loader.id_by_class, self.task_loader.train_idx_per_task[curr_session]
            ids_per_cls_train = [list(set(id_by_class[item]).intersection(set(train_idx))) for item in id_by_class.keys()]
            ids_per_cls_train = [item for item in ids_per_cls_train if len(item) > 0]

            label_frep = np.array(list(self.cond_num.values()))
            label_current = np.array([len(id) for id in ids_per_cls_train])
            label_frep_array = np.concatenate((label_frep, label_current), axis=0)
            label_frep_array = label_frep_array / label_frep_array.sum()
            adjustments = np.log(label_frep_array ** self.tro + 1e-12)
            adjustments = torch.from_numpy(adjustments)
            self.adjustments = adjustments.to(self.device)

            subset, edge_index, mapping, _ = k_hop_subgraph(train_idx, self.config['layer_num'], data.edge_index, relabel_nodes=True)
            subgraph = Data(x=data.x[subset], y=data.y[subset], edge_index=edge_index, mapping=mapping)

            ids_per_cls_train_subset = [
                [i for i, idx in enumerate(subset) if idx in ids_train]
                for ids_train in ids_per_cls_train
            ]

            gcond = GCondenser(curr_session, class_num, ids_per_cls_train_subset, self.feat_dim, self.hidden_dim, self.hidden_dim, self.layer_num, self.dropout, subgraph, config, device)
            cond_g = gcond.get_condense_graph()
            self.aux_g.append(cond_g.to(self.device))

            labels_condg = cond_g.y
            for j in range(self.output_dim):
                if (labels_condg == j).sum() != 0:
                    self.cond_num[j] = (labels_condg == j).sum().cpu()

            model.train()

        all_loss, train_num = 0., 0
        for _, batch in enumerate(train_loader):
            if batch['node_id'].size(0) < 2:
                break
            optimizer.zero_grad()
            
            subset, edge_index, mapping, _ = k_hop_subgraph(batch['node_id'], config['layer_num'], data.edge_index, relabel_nodes=True)
            logits = model(data.x[subset].to(device), edge_index.to(device))[mapping]

            logits = logits[:, :class_num] + self.adjustments[ :class_num]
            labels = batch['labels'].to(device)
            n_per_cls = [(labels == j).sum() for j in range(self.num_class)]
            loss_w = [1. / max(i, 1) for i in n_per_cls]
            loss_w = torch.tensor(loss_w[:class_num]).double().to(self.device)

            loss = self.loss_func(logits, labels, loss_w)

            if curr_session != 0:
                for old_session in range(curr_session):
                    aux_g = self.aux_g[old_session]
                    aux_logits = self.model(aux_g.x.to(self.device), aux_g.edge_index.to(self.device))
                    aux_logits = aux_logits[:, :class_num] + self.adjustments[ :class_num]
                    aux_labels = aux_g.y.to(device)

                    aux_n_per_cls = [(aux_labels == j).sum() for j in range(self.num_class)]
                    aux_loss_w = [1. / max(i, 1) for i in aux_n_per_cls]
                    aux_loss_w = torch.tensor(aux_loss_w[:class_num]).double().to(self.device)

                    loss_aux = self.loss_func(aux_logits, aux_labels, aux_loss_w)
                    loss = loss + loss_aux

            loss.backward()
            optimizer.step()
            all_loss += loss * batch['node_id'].size(0)
            train_num += batch['node_id'].size(0)

        return all_loss / train_num

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
                loss = self.train(curr_session, epoch, self.model, text_dataset_joint, train_loader, optimizer, class_num, self.config, self.device)
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


                    
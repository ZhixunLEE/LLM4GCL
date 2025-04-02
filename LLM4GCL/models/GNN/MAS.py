import torch
import torch.nn as nn
import torch.nn.functional as F

from LLM4GCL.models import BareGNN
from LLM4GCL.utils import _save_checkpoint, _reload_best_model

from tqdm import tqdm
from torch_geometric.utils import k_hop_subgraph

class MAS(BareGNN):

    def __init__(self, task_loader, result_logger, config, checkpoint_path, dataset, model_name, seed, device):
        super(MAS, self).__init__(task_loader, result_logger, config, checkpoint_path, dataset, model_name, seed, device)
        
        self.reg = config['mas_strength']
        self.current_task = 0
        self.fisher = []
        self.optpar = []
        self.n_seen_examples = 0
        self.mem_mask = None


    def new_weight(self, curr_session, curr_epoch, model, text_dataset, train_loader, optimizer, class_num, config, device):
        self.optpar = []
        new_fisher = []
        param_grad_epoch = []
        n_new_examples = 0

        model.train()
        data = text_dataset.data
        for _, batch in enumerate(train_loader):
            if batch['node_id'].size(0) < 2:
                break
            param_grad_batch = []
            n_new_examples += batch['node_id'].size(0)
            optimizer.zero_grad()
            subset, edge_index, mapping, _ = k_hop_subgraph(batch['node_id'], config['layer_num'], data.edge_index, relabel_nodes=True)
            logits = model(data.x[subset].to(device), edge_index.to(device))[mapping]

            logits = logits[:, :class_num]
            logits.pow_(2)
            loss = logits.mean()

            loss.backward()
            optimizer.step()

            for p in self.model.parameters():
                param_grad = p.grad.data.clone().pow(2)
                param_grad_batch.append(param_grad)
            param_grad_epoch.append(param_grad_batch)

        for i,p in enumerate(self.model.parameters()):
            param_grad_ = []
            for param_grad_batch_ in param_grad_epoch:
                param_grad_.append(param_grad_batch_[i])
            new_fisher.append(sum(param_grad_) / len(param_grad_))
            pd = p.data.clone()
            self.optpar.append(pd)

        if len(self.fisher) != 0:
            for i, f in enumerate(new_fisher):
                self.fisher[i] = (self.fisher[i] * self.n_seen_examples + new_fisher[i]* n_new_examples) / (self.n_seen_examples + n_new_examples)
            self.n_seen_examples += n_new_examples
        else:
            for i, f in enumerate(new_fisher):
                self.fisher.append(new_fisher[i])
            self.n_seen_examples = n_new_examples

        self.current_task += 1


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
            n_per_cls = [(labels == j).sum() for j in range(self.num_class)]
            loss_w = [1. / max(i, 1) for i in n_per_cls]
            loss_w = torch.tensor(loss_w[:class_num]).to(self.device)

            loss = self.loss_func(logits, labels, loss_w)

            if self.current_task > 0:
                for i, p in enumerate(self.model.parameters()):
                    l = self.reg * self.fisher[i]
                    l = l * (p - self.optpar[i]).pow(2)
                    loss += l.sum()

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

            self.new_weight(curr_session, epoch, self.model, text_dataset_iso, train_loader, optimizer, class_num, self.config, self.device)
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

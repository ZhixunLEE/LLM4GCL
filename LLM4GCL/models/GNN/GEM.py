import torch
import torch.nn as nn
import torch.nn.functional as F

from LLM4GCL.models import BareGNN
from LLM4GCL.utils import _save_checkpoint, _reload_best_model

from tqdm import tqdm
from torch_geometric.utils import k_hop_subgraph




import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import quadprog


def store_grad(params, grads, grad_dims, task_id):
    # store the gradients
    grads[:, task_id].fill_(0.0)
    cnt = 0
    for param in params():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            grads[beg: en, task_id].copy_(param.grad.data.view(-1))
        cnt += 1


def overwrite_grad(params, new_grad, grad_dims):
    cnt = 0
    for param in params():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = new_grad[beg: en].contiguous().view(
                param.grad.data.size())
            param.grad.data.copy_(this_grad)
        cnt += 1


def project2cone2(gradient, memories, margin=0.5, eps=1e-3):
    memories_np = memories.detach().cpu().t().double().numpy()
    gradient_np = gradient.detach().cpu().contiguous().view(-1).double().numpy()
    t = memories_np.shape[0]
    P = np.dot(memories_np, memories_np.transpose())
    P = 0.5 * (P + P.transpose()) + np.eye(t) * eps
    q = np.dot(memories_np, gradient_np) * -1
    G = np.eye(t)
    h = np.zeros(t) + margin
    v = quadprog.solve_qp(P, q, G, h)[0]
    x = np.dot(v, memories_np) + gradient_np
    gradient.copy_(torch.Tensor(x).view(-1, 1))


class GEM(BareGNN):

    def __init__(self, task_loader, result_logger, config, checkpoint_path, dataset, model_name, seed, device):
        super(GEM, self).__init__(task_loader, result_logger, config, checkpoint_path, dataset, model_name, seed, device)

        self.margin = config['GEM']['strength']
        self.n_mem = int(config['GEM']['n_mem'])

        self.memory_data = []

        self.grad_dims = []
        for param in self.model.parameters():
            self.grad_dims.append(param.data.numel())
        self.grads = torch.Tensor(sum(self.grad_dims), self.session_num).cuda()

        self.observed_tasks = []
        self.current_task = -1
        self.mem_cnt = 0

    
    def train(self, curr_session, curr_epoch, model, text_dataset, train_loader, optimizer, class_num, config, device):
        model.train()

        # create data loaders for previous tasks
        self.old_dataloaders = {}
        for old_task_idx in self.observed_tasks[:-1]:
            self.old_dataloaders[old_task_idx] = self.task_loader.get_task(old_task_idx, subset=self.n_mem)

        # compute gradient on previous tasks
        for old_task_idx in self.observed_tasks[:-1]:
            old_task_loss = 0
            
            (old_class_num, old_text_dataset, old_train_loader, _, _, _) = self.old_dataloaders[old_task_idx]
            old_data = old_text_dataset.data

            for _, batch in enumerate(old_train_loader):
                if batch['node_id'].size(0) < 2:
                    break
                self.model.zero_grad()

                subset, edge_index, mapping, _ = k_hop_subgraph(batch['node_id'], config['layer_num'], old_data.edge_index, relabel_nodes=True)
                logits = model(old_data.x[subset].to(device), edge_index.to(device))[mapping]

                logits = logits[:, :old_class_num]
                labels = batch['labels'].to(device)

                loss = self.loss_func(logits, labels)

                old_task_loss = old_task_loss + loss
            old_task_loss.backward()
            store_grad(self.model.parameters, self.grads, self.grad_dims, old_task_idx)

        no_solution_ind = 0
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

            all_loss += loss * batch['node_id'].size(0)
            train_num += batch['node_id'].size(0)

            # check if gradient violates constraints
            if len(self.observed_tasks) > 1:
                # copy gradient
                store_grad(self.model.parameters, self.grads, self.grad_dims, self.current_task)
                indx = torch.cuda.LongTensor(self.observed_tasks[:-1])
                dotp = torch.mm(self.grads[:, self.current_task].unsqueeze(0), self.grads.index_select(1, indx))
                if (dotp < 0).sum() != 0:
                    try:
                        project2cone2(self.grads[:, self.current_task].unsqueeze(1), self.grads.index_select(1, indx), self.margin)
                        # copy gradients back
                        overwrite_grad(self.model.parameters, self.grads[:, self.current_task], self.grad_dims)
                    except:
                        if no_solution_ind == 0:
                            print('no solution situation observed')
                        no_solution_ind = 1

            optimizer.step()

        return all_loss / train_num

    def fit(self, iter):
        optimizer = self.get_optimizer(self.model)

        for curr_session in range(self.session_num):
            self.observed_tasks.append(curr_session)
            self.current_task = curr_session

            if curr_session != 0:
                _reload_best_model(self.model, self.checkpoint_path, self.dataset, self.model_name, self.seed)

            class_num, text_dataset_iso, text_dataset_joint, train_loader, valid_loader, test_loader_isolate, test_loader_joint = self.task_loader.get_task(curr_session)

            progress_bar = tqdm(range(self.config['epochs']))
            progress_bar.set_description(f'Training | Iter {iter}')

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
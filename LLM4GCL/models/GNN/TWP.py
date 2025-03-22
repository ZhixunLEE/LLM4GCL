import torch

from LLM4GCL.models import BareGNN
from LLM4GCL.utils import _save_checkpoint, _reload_best_model

from tqdm import tqdm
from torch_geometric.utils import k_hop_subgraph


class TWP(BareGNN):

    def __init__(self, task_loader, result_logger, config, checkpoint_path, dataset, model_name, seed, device):
        super(TWP, self).__init__(task_loader, result_logger, config, checkpoint_path, dataset, model_name, seed, device)
        
        self.current_task = 0
        self.fisher_loss = {}
        self.fisher_att = {}
        self.optpar = {}
        self.mem_mask = None

        self.lambda_l = config['lambda_l']
        self.lambda_t = config['lambda_t']
        self.beta = config['beta']


    def new_weight(self, curr_epoch, model, text_dataset, train_loader, optimizer, class_num, config, device):
        model.train()
        data = text_dataset.data
        self.model.zero_grad()
        self.fisher_loss[self.current_task] = []
        self.fisher_att[self.current_task] = []
        self.optpar[self.current_task] = []

        pgss_floss,pgss_fatt = [], []
        for _, batch in enumerate(train_loader):
            if batch['node_id'].size(0) < 2:
                break
            pgs_floss,pgs_fatt = [], []
            optimizer.zero_grad()
            subset, edge_index, mapping, _ = k_hop_subgraph(batch['node_id'], config['layer_num'], data.edge_index, relabel_nodes=True)
            logits = model(data.x[subset].to(device), edge_index.to(device))[mapping]

            logits = logits[:, :class_num]
            labels = batch['labels'].to(device)

            loss = self.loss_func(logits, labels)
            loss.backward(retain_graph=True)

            for p in self.model.parameters():
                pg = p.grad.data.clone().pow(2)
                pgs_floss.append(pg)
            pgss_floss.append(pgs_floss)

            for p in self.model.parameters():
                pg = p.grad.data.clone().pow(2)
                pgs_fatt.append(pg)
            pgss_fatt.append(pgs_floss)

            for i,p in enumerate(self.model.parameters()):
                pg_floss_,pgs_fatt_ = [],[]
                for pgs_ in pgss_floss:
                    pg_floss_.append(pgs_[i])
                for pgs_ in pgss_fatt:
                    pgs_fatt_.append(pgs_[i])
                pd = p.data.clone()
                self.optpar[self.current_task].append(pd)
                self.fisher_loss[self.current_task].append(sum(pg_floss_)/len(pg_floss_))
                self.fisher_att[self.current_task].append(sum(pgs_fatt_)/len(pgs_fatt_))

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

            loss = self.loss_func(logits, labels)

            grad_norm = 0
            for p in self.model.parameters():
                param_grad = p.grad.data.clone()
                grad_norm += torch.norm(param_grad, p=1)

            for s in range(self.current_task):
                for i, p in enumerate(self.model.parameters()):
                    l = self.lambda_l * self.fisher_loss[s][i] + self.lambda_t * self.fisher_att[s][i]
                    l = l * (p - self.optpar[s][i]).pow(2)
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

            self.new_weight(epoch, self.model, text_dataset_iso, train_loader, optimizer, class_num, self.config, self.device)
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

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import accuracy_score, f1_score
from LLM4GCL.utils import _save_checkpoint, _reload_best_model

class BaseModel(nn.Module):

    def __init__(self, task_loader, result_logger, config, checkpoint_path, dataset, model_name, seed, device):
        super(BaseModel, self).__init__()
        self.task_loader = task_loader
        self.result_logger = result_logger
        self.session_num = task_loader.task_num
        self.feat_dim = task_loader.data.x.shape[1]
        self.num_class = task_loader.data.y.max().item() + 1
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.dataset = dataset
        self.model_name = model_name
        self.seed = seed
        self.device = device
        self.model = None

    def get_optimizer(self, model):
        optimizer = optim.Adam(model.parameters(), lr=float(self.config['lr']), weight_decay=float(self.config['weight_decay']))
        return optimizer

    def train(self, model, text_dataset, train_loader, optimizer, class_num, config, device):
        raise "The train method is not declared !"
    
    @torch.no_grad()
    def valid(self, model, text_dataset, valid_loader, class_num, config, device):
        raise "The valid method is not declared !"
    
    @torch.no_grad()
    def evaluate(self, model, text_dataset, test_loader, class_num, config, device):
        raise "The evaluate method is not declared !"
    
    def fit(self, ):
        optimizer = self.get_optimizer(self.model)

        for curr_session in range(self.session_num):
            if curr_session != 0:
                _reload_best_model(self.model, self.checkpoint_path, self.dataset, self.model_name, self.seed)

            class_num, text_dataset, train_loader, valid_loader, test_loader_isolate, test_loader_joint = self.task_loader.get_task(curr_session)

            tolerate, best_acc_valid = 0, 0.
            for epoch in range(self.config['epochs']):
                loss = self.train(self.model, text_dataset, train_loader, optimizer, class_num, self.config, self.device)
                print("Session: {} | Epoch: {} | Loss: {:.4f}".format(curr_session, epoch, loss.item()))

                if epoch > 0 and epoch % 10 == 0:
                    acc_valid, f1_valid = self.valid(self.model, text_dataset, valid_loader, class_num, self.config, self.device)
                    print("Session: {} | Epoch: {} | Acc Val: {:.4f} | F1 Val: {:.4f} | Tolerate: {}".format(curr_session, epoch, acc_valid, f1_valid, tolerate))
                    if acc_valid > best_acc_valid:
                        tolerate = 0
                        best_acc_valid = acc_valid
                        _save_checkpoint(self.model, optimizer, epoch, self.checkpoint_path, self.dataset, self.model_name, self.seed)
                    else:
                        tolerate += 1
                        if tolerate > self.config['patience']: 
                            break

            _reload_best_model(self.model, self.checkpoint_path, self.dataset, self.model_name, self.seed)
            curr_acc_test_isolate, curr_f1_test_isolate = self.evaluate(self.model, text_dataset, test_loader_isolate, class_num, self.config, self.device)
            curr_acc_test_joint, curr_f1_test_joint = self.evaluate(self.model, text_dataset, test_loader_joint, class_num, self.config, self.device)

            acc_list = []
            for s in range(curr_session):
                _, text_dataset, _, _, test_loader_isolate, _ = self.task_loader.get_task(s)
                prev_acc_test_isolate, prev_f1_test_isolate = self.evaluate(self.model, text_dataset, test_loader_isolate, class_num, self.config, self.device)
                acc_list.append(prev_acc_test_isolate)
            acc_list.append(curr_acc_test_isolate)

            print("Session: {} | Iso. Acc Test: {:.4f} | Iso. F1 Test: {:.4f}".format(curr_session, curr_acc_test_isolate, curr_f1_test_isolate))
            print("Session: {} | Jot. Acc Test: {:.4f} | Jot. F1 Test: {:.4f}".format(curr_session, curr_acc_test_joint, curr_f1_test_joint))

            self.result_logger.add_new_results(acc_list, curr_acc_test_joint)

        return self.result_logger


    def get_metric(self, logits, preds, labels):
        logits = logits.detach().cpu().numpy()
        preds = preds.cpu().numpy()
        labels = labels.cpu().numpy()
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='macro', zero_division=0)

        return acc, f1


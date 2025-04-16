import torch
import torch.nn as nn
import torch.nn.functional as F

from LLM4GCL.models import BaseModel
from LLM4GCL.backbones import RoBERTaNet, LLaMANet
from LLM4GCL.common.utils import adjust_learning_rate, _save_checkpoint, _reload_best_model

from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class SimpleCIL(BaseModel):

    def __init__(self, task_loader, result_logger, config, checkpoint_path, dataset, model_name, model_path, seed, device):
        super(SimpleCIL, self).__init__(task_loader, result_logger, config, checkpoint_path, dataset, model_name, seed, device)
        self.lm_type = config['lm']
        if self.lm_type == 'RoBERTa':
            self.hidden_dim = 1024
        elif self.lm_type == 'LLaMA':
            self.hidden_dim = 4096
        self.output_dim = self.num_class
        self.lr = float(config['lr'])
        self.weight_decay = float(config['weight_decay'])
        self.lora_config = config['LoRA']
        self.dropout = config['dropout']
        self.att_dropout = config['att_dropout']
        self.max_length = config['max_length']
        self.model_path = model_path

        self.T = config['simplecil_T']
        self.sample_num = config['simplecil_sample_num']

        class LMModel(nn.Module):

            def __init__(self, lm_type, max_length, model_path, hidden_dim, output_dim, lora_config, dropout, att_dropout, T, device):
                super(LMModel, self).__init__()
                self.device = device
                self.max_length = max_length
                self.T = T
                self.lm_type = lm_type
                
                if lm_type == 'RoBERTa':
                    self.lm = RoBERTaNet(output_dim, model_path, lora_config, dropout, att_dropout).to(device)
                elif lm_type == 'LLaMA':
                    self.lm = LLaMANet(model_path, lora_config, dropout, att_dropout).to(device)

                self.fc = nn.Linear(hidden_dim, output_dim).to(device)

            def forward(self, samples):
                tokens = self.lm.tokenizer(samples['raw_text'], padding=True, truncation=True, max_length=self.max_length, return_tensors='pt')
                tokens['input_ids'] = tokens['input_ids'].to(self.device)
                tokens['attention_mask'] = tokens['attention_mask'].to(self.device)
                outputs = self.lm(tokens['input_ids'], tokens['attention_mask'])

                if self.lm_type in ['RoBERTa']:
                    hidden_embs = outputs.hidden_states[-1][:, 0, :]
                elif self.lm_type in ['LLaMA']:
                    hidden_embs = mean_pooling(outputs.hidden_states[-1], tokens['attention_mask'])

                logits = self.fc(hidden_embs)

                return logits, hidden_embs

            def cosine_forward(self, samples):
                logits, hidden_embs = self.forward(samples)
                x = hidden_embs
                x = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1))
                logits = self.T * x

                return logits
            
        self.model = LMModel(self.lm_type, self.max_length, self.model_path, self.hidden_dim, self.output_dim, self.lora_config, self.dropout, self.att_dropout, self.T, self.device)

    @torch.no_grad()
    def update_proto(self, model, data, train_loader, task_classes, config, device):
        self.model.eval()
        embeds_list, labels_list = [], []
        for _, batch in enumerate(train_loader):
            if batch['node_id'].size(0) < 2:
                break
            _, hidden_embeds = model(batch)
            labels = batch['labels'].to(device)
            embeds_list.extend(hidden_embeds)
            labels_list.extend(labels)
        
        embeds = torch.stack(embeds_list, dim=0)
        labels = torch.stack(labels_list, dim=0)

        for class_index in task_classes:
            data_index = (labels == class_index).nonzero().squeeze(-1)
            class_embed = embeds[data_index]
            proto = class_embed.mean(0)
            self.model.fc.weight.data[class_index] = proto
    
    def get_optimizer(self, model):
        params = [p for _, p in model.named_parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            [{'params': params, 'lr': self.lr, 'weight_decay': self.weight_decay},],
            betas=(0.9, 0.95)
        )

        return optimizer

    def train(self, curr_session, curr_epoch, model, text_dataset, train_loader, optimizer, class_num, config, device):
        model.train()
        accum_loss = 0.
        all_loss, train_num = 0., 0
        for step, batch in enumerate(train_loader):
            if batch['node_id'].size(0) < 2:
                break
            optimizer.zero_grad()
            logits, _ = model(batch)
            labels = batch['labels'].to(self.device)
            n_per_cls = [(labels == j).sum() for j in range(self.num_class)]
            loss_w = [1. / max(i, 1) for i in n_per_cls]
            loss_w = torch.tensor(loss_w[:class_num]).to(self.device)

            loss = self.loss_func(logits[:, : class_num], labels, loss_w)
            loss.backward()

            clip_grad_norm_(optimizer.param_groups[0]['params'], 0.1)
            if (step + 1) % config['grad_steps'] == 0:
                adjust_learning_rate(optimizer.param_groups[0], step / len(train_loader) + curr_epoch, config)
            optimizer.step()
            accum_loss = accum_loss + loss.item()

            if (step + 1) % config['grad_steps'] == 0:
                all_loss += accum_loss * batch['node_id'].size(0)
                train_num += batch['node_id'].size(0)
                lr = optimizer.param_groups[0]["lr"]
                accum_loss = 0.

        return all_loss / train_num

    @torch.no_grad()
    def valid(self, model, text_dataset, valid_loader, class_num, config, device):
        model.eval()
        logits_list, preds_list, labels_list = [], [], []
        for _, batch in enumerate(valid_loader):
            if batch['node_id'].size(0) < 2:
                break
            logits, _ = model(batch)
            logits = logits[:, : class_num]
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
        logits_list, preds_list, labels_list = [], [], []
        for _, batch in enumerate(test_loader):
            if batch['node_id'].size(0) < 2:
                break
            logits = model.cosine_forward(batch)
            logits = logits[:, : class_num]
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
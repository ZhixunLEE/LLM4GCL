import torch
import torch.nn as nn

from LLM4GCL.models import BaseModel
from LLM4GCL.backbones import RoBERTaNet
from LLM4GCL.utils import adjust_learning_rate

from torch.nn.utils import clip_grad_norm_


class BareLM(BaseModel):

    def __init__(self, task_loader, result_logger, config, checkpoint_path, dataset, model_name, model_path, seed, device):
        super(BareLM, self).__init__(task_loader, result_logger, config, checkpoint_path, dataset, model_name, seed, device)
        self.lm_type = config['lm']
        if self.lm_type == 'RoBERTa':
            self.hidden_dim = 1024
        self.output_dim = self.num_class
        self.lr = float(config['lr'])
        self.weight_decay = float(config['weight_decay'])
        self.lora_config = config['LoRA']
        self.dropout = config['dropout']
        self.att_dropout = config['att_dropout']
        self.max_length = config['max_length']
        self.model_path = model_path

        class LMModel(nn.Module):

            def __init__(self, lm_type, max_length, model_path, hidden_dim, output_dim, lora_config, dropout, att_dropout, device):
                super(LMModel, self).__init__()
                self.device = device
                self.max_length = max_length
                
                if lm_type == 'RoBERTa':
                    self.lm = RoBERTaNet(output_dim, model_path, lora_config, dropout, att_dropout).to(device)

                self.fc = nn.Sequential(
                    # nn.Linear(hidden_dim, hidden_dim),
                    # nn.ReLU(),
                    # nn.Dropout(p=0.1), 
                    nn.Linear(hidden_dim, output_dim),
                ).to(device)

            def forward(self, samples):
                tokens = self.lm.tokenizer(samples['raw_text'], padding=True, truncation=True, max_length=self.max_length, return_tensors='pt')
                tokens['input_ids'] = tokens['input_ids'].to(self.device)
                tokens['attention_mask'] = tokens['attention_mask'].to(self.device)
                _, hidden_states = self.lm(tokens['input_ids'], tokens['attention_mask'])
                logits = self.fc(hidden_states[-1][:, 0, :])
                # logits, _ = self.lm(tokens['input_ids'], tokens['attention_mask'])

                return logits
            
        self.model = LMModel(self.lm_type, self.max_length, self.model_path, self.hidden_dim, self.output_dim, self.lora_config, self.dropout, self.att_dropout, self.device)

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
            logits = model(batch)
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
        return self.evaluate(model, text_dataset, valid_loader, class_num, config, device)

    @torch.no_grad()
    def evaluate(self, model, text_dataset, test_loader, class_num, config, device):
        model.eval()
        logits_list, preds_list, labels_list = [], [], []
        for _, batch in enumerate(test_loader):
            if batch['node_id'].size(0) < 2:
                break
            logits = model(batch)
            logits = torch.softmax(logits[:, : class_num], dim=1)
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
    
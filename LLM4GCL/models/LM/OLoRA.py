import re
import torch
import torch.nn as nn
import torch.nn.functional as F

from LLM4GCL.models import BareLM
from LLM4GCL.backbones import RoBERTaNet, LLaMANet
from LLM4GCL.utils import adjust_learning_rate, _save_checkpoint, _reload_best_model

from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
from peft import LoraConfig, get_peft_model


class OLoRA(BareLM):

    def __init__(self, task_loader, result_logger, config, checkpoint_path, dataset, model_name, model_path, seed, device):
        super(OLoRA, self).__init__(task_loader, result_logger, config, checkpoint_path, dataset, model_name, model_path, seed, device)

        if self.lm_type == 'RoBERTa':
            self.layer_num = 24
        self.lora_config = config['LoRA']

        class LMModel(nn.Module):

            def __init__(self, lm_type, max_length, model_path, hidden_dim, output_dim, lora_config, dropout, att_dropout, device):
                super(LMModel, self).__init__()
                self.device = device
                self.max_length = max_length
                
                if lm_type == 'RoBERTa':
                    self.lm = RoBERTaNet(output_dim, model_path, lora_config, dropout, att_dropout).to(device)
                elif lm_type == 'LLaMA':
                    self.lm = LLaMANet(max_length, output_dim, model_path, lora_config, dropout, att_dropout).to(device)

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
                hidden_embs = self.lm(tokens['input_ids'], tokens['attention_mask'])
                logits = self.fc(hidden_embs)

                return logits
            
        self.model = LMModel(self.lm_type, self.max_length, self.model_path, self.hidden_dim, self.output_dim, self.lora_config, self.dropout, self.att_dropout, self.device)

        self.ortho_lambda = config['olora_ortho_lambda']
        self.l2_lambda = config['olora_l2_lambda']

        self.session_cnt = 0
        self.lora_A_Q_dict = {str(i): [] for i in range(self.layer_num)}
        self.lora_A_V_dict = {str(i): [] for i in range(self.layer_num)}
    
    def new_incremental_task(self, ):
        self.session_cnt += 1
        for name, param in self.model.lm.model.named_parameters():
            if "lora_A" in name:
                layer_id =  re.search(r'layer\.(\d+)', name).group(1)
                if "query" in name:
                    self.lora_A_Q_dict[layer_id].append((name, param))
                elif "value" in name:
                    self.lora_A_V_dict[layer_id].append((name, param))
        self.model.lm.model = self.model.lm.model.merge_and_unload()

        new_config = LoraConfig(
            r=self.lora_config['lora_r'],
            lora_alpha=self.lora_config['lora_alpha'],
            target_modules=self.lora_config['target_modules'],
            lora_dropout=self.lora_config['lora_dropout'],
            task_type="SEQ_CLS",
        )
        self.model.lm.model = get_peft_model(self.model.lm.model, new_config)
        optimizer = self.get_optimizer(self.model)

        return optimizer

    def get_ortho_loss(self, ):
        
        ortho_loss = 0.
        if self.session_cnt > 0:
            for curr_name, curr_param in self.model.lm.model.named_parameters():
                if "lora_A" in curr_name:
                    layer_id =  re.search(r'layer\.(\d+)', curr_name).group(1)
                    if "query" in curr_name:
                        for name, param in self.lora_A_Q_dict[layer_id]:
                            ortho_loss += torch.abs(torch.mm(param, curr_param.T)).sum() # [r * dim] * [dim * r]
                    elif "value" in curr_name:
                        for name, param in self.lora_A_V_dict[layer_id]:
                            ortho_loss += torch.abs(torch.mm(param, curr_param.T)).sum() # [r * dim] * [dim * r]

        # l2-normalization for new lora_A/B
        l2_loss = 0.
        for name, param in self.model.lm.model.named_parameters():
            if "lora" in name and "LoRA_new" + str(self.session_cnt) in name:
                l2_loss += torch.norm(param, p=2)

        return ortho_loss, l2_loss

    def loss_func(self, logits, labels, loss_weight=None):
        loss = F.cross_entropy(logits, labels, weight=loss_weight)
        ortho_loss, l2_loss = self.get_ortho_loss()
        loss = loss + ortho_loss * self.ortho_lambda + l2_loss * self.l2_lambda

        return loss

    def fit(self, iter):

        for curr_session in range(self.session_num):
            if curr_session != 0:
                _reload_best_model(self.model, self.checkpoint_path, self.dataset, self.model_name, self.seed)
                optimizer = self.new_incremental_task()
            else:
                optimizer = self.get_optimizer(self.model)

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
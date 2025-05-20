import torch
import torch.nn as nn
import torch.nn.functional as F

from LLM4GCL.models import BaseModel
from LLM4GCL.backbones import LLaMANet
from LLM4GCL.common.utils import adjust_learning_rate, _save_checkpoint, _reload_best_model
from LLM4GCL.common.prompts import get_genreal_prompts

from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_

IGNORE_INDEX = -100

class LLaMA(BaseModel):

    def __init__(self, task_loader, result_logger, config, checkpoint_path, dataset, model_name, model_path, local_ce, seed, device):
        super(LLaMA, self).__init__(task_loader, result_logger, config, checkpoint_path, dataset, model_name, local_ce, seed, device)
        self.lm_type = config['lm']
        self.hidden_dim = 4096
        self.lr = float(config['lr'])
        self.weight_decay = float(config['weight_decay'])
        self.lora_config = config['LoRA']
        self.dropout = config['dropout']
        self.att_dropout = config['att_dropout']
        self.max_length = config['max_length']
        self.model_path = model_path

        class LMModel(nn.Module):

            def __init__(self, lm_type, max_length, model_path, lora_config, dropout, att_dropout, device):
                super(LMModel, self).__init__()
                self.device = device
                self.max_length = max_length
                self.lm_type = lm_type
                
                if lm_type == 'LLaMA':
                    self.lm = LLaMANet(model_path, lora_config, dropout, att_dropout).to(device)

            def forward(self, samples, prompts):
                tokenizer = self.lm.tokenizer
                embedding = self.lm.embeddings

                pad_embeds = embedding(torch.tensor(tokenizer.pad_token_id)).unsqueeze(0)
                bos_id = tokenizer.bos_token_id
                eos_id = tokenizer.eos_token_id

                prompt_token = tokenizer(prompts, add_special_tokens=False).to(self.device)
                input_token = tokenizer(samples['raw_text'], add_special_tokens=False).to(self.device)
                label_token = tokenizer(samples['label_text'], add_special_tokens=False).to(self.device)

                batch_inputs_embeds = []
                batch_attention_mask = []
                batch_label_input_ids = []
                for i in range(len(samples['node_id'])):
                    label_input_ids = label_token.input_ids[i] + [eos_id]
                    max_text_len = self.max_length - len(prompt_token.input_ids + label_input_ids) - 1
                    input_ids = [bos_id] + input_token.input_ids[i][:max_text_len] + prompt_token.input_ids + label_input_ids
                    inputs_embeds = embedding(torch.tensor(input_ids).to(self.device))
                    label_input_ids = [IGNORE_INDEX] * (inputs_embeds.shape[0] - len(label_input_ids)) + label_input_ids

                    batch_inputs_embeds.append(inputs_embeds)
                    batch_attention_mask.append([1] * inputs_embeds.shape[0])
                    batch_label_input_ids.append(label_input_ids)

                max_length = max([x.shape[0] for x in batch_inputs_embeds])
                for i in range(len(samples['node_id'])):
                    pad_length = max_length - batch_inputs_embeds[i].shape[0]
                    batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
                    batch_attention_mask[i] = [0] * pad_length + batch_attention_mask[i]
                    batch_label_input_ids[i] = [IGNORE_INDEX] * pad_length + batch_label_input_ids[i]

                inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.device)
                attention_mask = torch.tensor(batch_attention_mask).to(self.device)
                label_input_ids = torch.tensor(batch_label_input_ids).to(self.device)

                outputs = self.lm(inputs_embeds, attention_mask, labels=label_input_ids)

                return outputs

            def generate(self, samples, prompts):
                tokenizer = self.lm.tokenizer
                embedding = self.lm.embeddings

                pad_embeds = embedding(torch.tensor(tokenizer.pad_token_id)).unsqueeze(0)
                bos_id = tokenizer.bos_token_id

                prompt_token = tokenizer(prompts, add_special_tokens=False).to(self.device)
                input_token = tokenizer(samples['raw_text'], add_special_tokens=False).to(self.device)

                batch_inputs_embeds = []
                batch_attention_mask = []

                for i in range(len(samples['node_id'])):
                    max_text_len = self.max_length - len(prompt_token.input_ids) - 1
                    input_ids = [bos_id] + input_token.input_ids[i][:max_text_len] + prompt_token.input_ids
                    inputs_embeds = embedding(torch.tensor(input_ids).to(self.device))

                    batch_inputs_embeds.append(inputs_embeds)
                    batch_attention_mask.append([1] * inputs_embeds.shape[0])

                max_length = max([x.shape[0] for x in batch_inputs_embeds])
                for i in range(len(samples['node_id'])):
                    pad_length = max_length - batch_inputs_embeds[i].shape[0]
                    batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
                    batch_attention_mask[i] = [0] * pad_length + batch_attention_mask[i]

                inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.device)
                attention_mask = torch.tensor(batch_attention_mask).to(self.device)

                outputs = self.lm.generate(inputs_embeds, attention_mask)

                return outputs
                
        self.model = LMModel(self.lm_type, self.max_length, self.model_path, self.lora_config, self.dropout, self.att_dropout, self.device)

    
    def get_optimizer(self, model):
        params = [p for _, p in model.named_parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            [{'params': params, 'lr': self.lr, 'weight_decay': self.weight_decay},],
            betas=(0.9, 0.95)
        )

        return optimizer

    def train(self, curr_session, curr_epoch, model, text_dataset, train_loader, optimizer, class_num, config, device, prompts):
        model.train()
        all_loss, train_num = 0., 0
        for step, batch in enumerate(train_loader):
            if batch['node_id'].size(0) < 2:
                break
            optimizer.zero_grad()

            outputs = model(batch, prompts)
            loss = outputs.loss

            loss.backward()
            all_loss += loss * batch['node_id'].size(0)
            train_num += batch['node_id'].size(0)

            clip_grad_norm_(optimizer.param_groups[0]['params'], 0.1)
            if (step + 1) % config['grad_steps'] == 0:
                adjust_learning_rate(optimizer.param_groups[0], step / len(train_loader) + curr_epoch, config)
            optimizer.step()

        print(train_num)
        return all_loss / train_num

    @torch.no_grad()
    def valid(self, model, text_dataset, valid_loader, class_num, config, device, prompts):
        return self.evaluate(model, text_dataset, valid_loader, class_num, config, device, prompts)

    @torch.no_grad()
    def evaluate(self, model, text_dataset, test_loader, class_num, config, device, prompts):
        model.eval()
        logits_list, preds_list, labels_list = [], [], []
        for _, batch in enumerate(test_loader):
            if batch['node_id'].size(0) < 2:
                break

            preds = model.generate(batch, prompts)
            labels = batch['label_text']
            preds_list.extend(preds)
            labels_list.extend(labels)

        labels = labels_list
        preds = preds_list

        acc, f1 = self.get_metric(None, preds, labels)

        return acc, f1
    
    
    def fit(self, iter):
        optimizer = self.get_optimizer(self.model)
        label_text_list = self.task_loader.text_dataset.label_texts
        for curr_session in range(self.session_num):
            if curr_session != 0:
                _reload_best_model(self.model, self.checkpoint_path, self.dataset, self.model_name, self.seed)

            _, class_dst, text_dataset_iso, text_dataset_joint, train_loader, valid_loader, test_loader_isolate, test_loader_joint = self.task_loader.get_task(curr_session)

            label_text = label_text_list[ :class_dst]
            prompts = get_genreal_prompts(self.dataset, label_text)

            progress_bar = tqdm(range(self.config['epochs']))
            progress_bar.set_description(f'Training | Iter {iter} | Session {curr_session}')

            tolerate, best_acc_valid = 0, 0.
            for epoch in range(self.config['epochs']):
                loss = self.train(curr_session, epoch, self.model, text_dataset_iso, train_loader, optimizer, class_dst, self.config, self.device, prompts)
                progress_bar.write("Session: {} | Epoch: {} | Loss: {:.4f}".format(curr_session, epoch, loss))

                if epoch > 0 and epoch % self.config['valid_epoch'] == 0:
                    acc_valid, f1_valid = self.valid(self.model, text_dataset_iso, valid_loader, class_dst, self.config, self.device, prompts)
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
            curr_acc_test_isolate, curr_f1_test_isolate = self.evaluate(self.model, text_dataset_iso, test_loader_isolate, class_dst, self.config, self.device, prompts)
            curr_acc_test_joint, curr_f1_test_joint = self.evaluate(self.model, text_dataset_joint, test_loader_joint, class_dst, self.config, self.device, prompts)

            acc_list = []
            for s in range(curr_session):
                _, _, text_dataset_iso, _, _, _, test_loader_isolate, _ = self.task_loader.get_task(s)
                prev_acc_test_isolate, prev_f1_test_isolate = self.evaluate(self.model, text_dataset_iso, test_loader_isolate, class_dst, self.config, self.device, prompts)
                acc_list.append(prev_acc_test_isolate)
            acc_list.append(curr_acc_test_isolate)

            print("Session: {} | Iso. Acc Test: {:.4f} | Iso. F1 Test: {:.4f}".format(curr_session, curr_acc_test_isolate, curr_f1_test_isolate))
            print("Session: {} | Jot. Acc Test: {:.4f} | Jot. F1 Test: {:.4f}".format(curr_session, curr_acc_test_joint, curr_f1_test_joint))

            self.result_logger.add_new_results(acc_list, curr_acc_test_joint)
        
        torch.cuda.empty_cache()
        return self.result_logger
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from LLM4GCL.models import BaseModel
from LLM4GCL.backbones import LLaMANet
from LLM4GCL.common.utils import adjust_learning_rate, _save_checkpoint, _reload_best_model
from LLM4GCL.common.prompts import get_instruction_prompts

from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_

IGNORE_INDEX = -100

class InstructLM(BaseModel):

    def __init__(self, task_loader, result_logger, config, checkpoint_path, dataset, model_name, model_path, local_ce, seed, device):
        super(InstructLM, self).__init__(task_loader, result_logger, config, checkpoint_path, dataset, model_name, local_ce, seed, device)

        self.lm_type = config['lm']
        if self.lm_type == 'LLaMA':
            self.hidden_dim = 4096
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
                
                if lm_type == 'LLaMA':
                    self.lm = LLaMANet(model_path, lora_config, dropout, att_dropout).to(device)

            def forward(self, instructions):
                tokenizer = self.lm.tokenizer
                embedding = self.lm.embeddings
                pad_embeds = embedding(torch.tensor(tokenizer.pad_token_id)).unsqueeze(0)
                bos_id = tokenizer.bos_token_id
                eos_id = tokenizer.eos_token_id
                
                batch_inputs_embeds = []
                batch_attention_mask = []
                batch_label_input_ids = []
                
                for item in instructions:
                    context, question, answer = '[INST]' + item["Context"], item["Question"] + '[/INST]', item["Answer"]

                    context_tokens = tokenizer(context, add_special_tokens=False).to(self.device)
                    question_tokens = tokenizer(question, add_special_tokens=False).to(self.device)
                    answer_tokens = tokenizer(answer, add_special_tokens=False).to(self.device)

                    # <s> [INST] Context + Question [/INST] Label </s>
                    max_text_len = self.max_length - len(question_tokens.input_ids + answer_tokens.input_ids + [eos_id]) - 1
                    input_ids = [bos_id] + context_tokens.input_ids[:max_text_len] + question_tokens.input_ids + answer_tokens.input_ids + [eos_id]
                    inputs_embeds = embedding(torch.tensor(input_ids).to(self.device))

                    labels = [IGNORE_INDEX] * (len([bos_id] + context_tokens.input_ids[:max_text_len] + question_tokens.input_ids)) + answer_tokens.input_ids + [eos_id]
                    
                    batch_inputs_embeds.append(inputs_embeds)
                    batch_attention_mask.append([1] * len(input_ids))
                    batch_label_input_ids.append(labels)

                max_length = max([x.shape[0] for x in batch_inputs_embeds])
                for i in range(len(batch_inputs_embeds)):
                    pad_length = max_length - batch_inputs_embeds[i].shape[0]
                    batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
                    batch_attention_mask[i] = [0] * pad_length + batch_attention_mask[i]
                    batch_label_input_ids[i] = [IGNORE_INDEX] * pad_length + batch_label_input_ids[i]

                inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.device)
                attention_mask = torch.tensor(batch_attention_mask).to(self.device)
                labels = torch.tensor(batch_label_input_ids).to(self.device)
                
                outputs = self.lm(inputs_embeds, attention_mask, labels=labels)

                return outputs

            def generate(self, instructions):
                tokenizer = self.lm.tokenizer
                embedding = self.lm.embeddings
                bos_id = tokenizer.bos_token_id
                
                batch_inputs_embeds = []
                batch_attention_mask = []
                
                for item in instructions:
                    context, question = '[INST]' + item["Context"], item["Question"] + '[/INST]'
                    context_tokens = tokenizer(context, add_special_tokens=False).to(self.device)
                    question_tokens = tokenizer(question, add_special_tokens=False).to(self.device)

                    # <s> [INST] Context + Question [/INST]
                    max_text_len = self.max_length - len(question_tokens.input_ids) - 1
                    input_ids = [bos_id] + context_tokens.input_ids[:max_text_len] + question_tokens.input_ids
                    inputs_embeds = embedding(torch.tensor(input_ids).to(self.device))
                    
                    batch_inputs_embeds.append(inputs_embeds)
                    batch_attention_mask.append([1] * len(input_ids))
                
                max_length = max([x.shape[0] for x in batch_inputs_embeds])
                pad_embeds = embedding(torch.tensor(tokenizer.pad_token_id)).unsqueeze(0)
                for i in range(len(batch_inputs_embeds)):
                    pad_length = max_length - batch_inputs_embeds[i].shape[0]
                    batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
                    batch_attention_mask[i] = [0] * pad_length + batch_attention_mask[i]
                
                inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.device)
                attention_mask = torch.tensor(batch_attention_mask).to(self.device)
                
                outputs = self.lm.generate(inputs_embeds, attention_mask)
                
                return outputs

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
        all_loss, train_num = 0., 0
        for step, batch in enumerate(train_loader):
            if batch['node_id'].size(0) < 2:
                break
            optimizer.zero_grad()

            instructions = get_instruction_prompts(batch['node_id'], text_dataset.data, text_dataset.data.raw_texts, None, class_num, self.dataset, mode='ego')
            outputs = model(instructions)
            loss = outputs.loss
            loss.backward()
            all_loss += loss * batch['node_id'].size(0)
            train_num += batch['node_id'].size(0)

            clip_grad_norm_(optimizer.param_groups[0]['params'], 0.1)
            if (step + 1) % config['grad_steps'] == 0:
                adjust_learning_rate(optimizer.param_groups[0], step / len(train_loader) + curr_epoch, config)
            optimizer.step()

        return all_loss / train_num

    @torch.no_grad()
    def valid(self, model, text_dataset, valid_loader, class_num, config, device):
        return self.evaluate(model, text_dataset, valid_loader, class_num, config, device)

    @torch.no_grad()
    def evaluate(self, model, text_dataset, test_loader, class_num, config, device):
        model.eval()
        preds_list, labels_list = [], []
        for _, batch in enumerate(test_loader):
            if batch['node_id'].size(0) < 2:
                break
            instructions = get_instruction_prompts(batch['node_id'], text_dataset.data, text_dataset.data.raw_texts, None, class_num, self.dataset, mode='ego')
            preds = model.generate(instructions)
            labels = batch['label_text']

            preds_list.extend(preds)
            labels_list.extend(labels)

        preds = preds_list
        labels = labels_list

        acc, f1 = self.get_metric(None, preds, labels)

        return acc, f1

    
    def fit(self, iter):
        optimizer = self.get_optimizer(self.model)
        for curr_session in range(self.session_num):
            _, class_dst, text_dataset_iso, text_dataset_joint, train_loader, valid_loader, test_loader_isolate, test_loader_joint = self.task_loader.get_task(curr_session)

            if curr_session == 0:
                progress_bar = tqdm(range(self.config['epochs']))
                progress_bar.set_description(f'Training | Iter {iter} | Session {curr_session}')

                tolerate, best_acc_valid = 0, 0.
                for epoch in range(self.config['epochs']):
                    loss = self.train(curr_session, epoch, self.model, text_dataset_iso, train_loader, optimizer, class_dst, self.config, self.device)
                    progress_bar.write("Session: {} | Epoch: {} | Loss: {:.4f}".format(curr_session, epoch, loss))

                    if epoch > 0 and epoch % self.config['valid_epoch'] == 0:
                        acc_valid, f1_valid = self.valid(self.model, text_dataset_iso, valid_loader, class_dst, self.config, self.device)
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

            curr_acc_test_isolate, curr_f1_test_isolate = self.evaluate(self.model, text_dataset_iso, test_loader_isolate, class_dst, self.config, self.device)
            curr_acc_test_joint, curr_f1_test_joint = self.evaluate(self.model, text_dataset_joint, test_loader_joint, class_dst, self.config, self.device)

            acc_list = []
            for s in range(curr_session):
                _, _, text_dataset_iso, _, _, _, test_loader_isolate, _ = self.task_loader.get_task(s)
                prev_acc_test_isolate, prev_f1_test_isolate = self.evaluate(self.model, text_dataset_iso, test_loader_isolate, class_dst, self.config, self.device)
                acc_list.append(prev_acc_test_isolate)
            acc_list.append(curr_acc_test_isolate)

            print("Session: {} | Iso. Acc Test: {:.4f} | Iso. F1 Test: {:.4f}".format(curr_session, curr_acc_test_isolate, curr_f1_test_isolate))
            print("Session: {} | Jot. Acc Test: {:.4f} | Jot. F1 Test: {:.4f}".format(curr_session, curr_acc_test_joint, curr_f1_test_joint))

            self.result_logger.add_new_results(acc_list, curr_acc_test_joint)

        return self.result_logger
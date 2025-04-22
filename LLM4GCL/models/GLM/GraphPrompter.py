import torch
import torch.nn as nn

from LLM4GCL.models import BaseModel
from LLM4GCL.backbones import GCNNet, GATNet, SAGENet, SGCNet, RoBERTaNet, LLaMANet
from LLM4GCL.common.utils import adjust_learning_rate, _save_checkpoint, _reload_best_model
from LLM4GCL.common.prompts import get_genreal_prompts, get_label_text

from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
from torch_geometric.utils import k_hop_subgraph

IGNORE_INDEX = -100

def encode_graphs(feats, edge_index, mapping, gnn_encoder, projector, device):
    embeds = gnn_encoder(feats.to(device), edge_index.to(device))
    inputs_embeds = projector(embeds[mapping])

    return inputs_embeds

class GraphPrompterComponent(nn.Module):

    def __init__(self, feat_dim, num_class, config, model_path, device):
        super(GraphPrompterComponent, self).__init__()
        # LM Params
        self.lm_type = config['lm']
        if self.lm_type == 'RoBERTa':
            self.lm_hidden_dim = 1024
        elif self.lm_type == 'LLaMA':
            self.lm_hidden_dim = 4096
        self.output_dim = num_class
        self.lora_config = config['LoRA']
        self.lm_dropout = config['dropout']
        self.att_dropout = config['att_dropout']
        self.max_length = config['max_length']
        self.model_path = model_path
        self.device = device

        if self.lm_type in ['RoBERTa']:
            class LMModel(nn.Module):

                def __init__(self, lm_type, max_length, model_path, hidden_dim, output_dim, lora_config, dropout, att_dropout, device):
                    super(LMModel, self).__init__()
                    self.device = device
                    self.max_length = max_length
                    self.hidden_dim = hidden_dim
                    self.output_dim = output_dim
                    
                    if lm_type == 'RoBERTa':
                        self.lm = RoBERTaNet(output_dim, model_path, lora_config, dropout, att_dropout).to(device)

                    self.fc = nn.Sequential(
                        # nn.Linear(hidden_dim, hidden_dim),
                        # nn.ReLU(),
                        # nn.Dropout(p=0.1), 
                        nn.Linear(self.hidden_dim, self.output_dim),
                    ).to(self.device)

                def forward(self, samples, graph_embeds, prompts=None):
                    tokenizer = self.lm.tokenizer
                    embedding = self.lm.embeddings
                    batch_inputs_embeds = []
                    batch_attention_mask = []
                    batch_size = len(samples['node_id'])
                    for i in range(batch_size):
                        inputs = tokenizer(samples['raw_text'], padding=True, truncation=True, max_length=self.max_length, return_tensors='pt')
                        inputs_embeds = embedding(inputs['input_ids'][i][:self.max_length - 1].to(self.device))
                        inputs_embeds = torch.cat([inputs_embeds[:1], graph_embeds[i].unsqueeze(0), inputs_embeds[1:]], dim=0)

                        batch_inputs_embeds.append(inputs_embeds)
                        batch_attention_mask.append(torch.cat([torch.tensor([1]).to(self.device), inputs['attention_mask'][i][:self.max_length - 1].to(self.device)]))

                    inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.device)
                    attention_mask = torch.stack(batch_attention_mask, dim=0).to(self.device)

                    outputs = self.lm(inputs_embeds, attention_mask)
                    logits = self.fc(outputs.hidden_states[-1][:, 0, :])

                    return logits
        elif self.lm_type in ['LLaMA']:
            class LMModel(nn.Module):

                def __init__(self, lm_type, max_length, model_path, hidden_dim, output_dim, lora_config, dropout, att_dropout, device):
                    super(LMModel, self).__init__()
                    self.device = device
                    self.max_length = max_length
                    
                    if lm_type == 'LLaMA':
                        self.lm = LLaMANet(model_path, lora_config, dropout, att_dropout).to(device)

                def forward(self, samples, graph_embeds, prompts):
                    batch_size = len(samples['node_id'])        
                    tokenizer = self.lm.tokenizer
                    embedding = self.lm.embeddings

                    pad_embeds = embedding(torch.tensor(tokenizer.pad_token_id)).unsqueeze(0)
                    bos_embeds = embedding(torch.tensor(tokenizer.bos_token_id)).unsqueeze(0)
                    eos_id = tokenizer.eos_token_id

                    prompt_token = tokenizer(prompts, add_special_tokens=False).to(self.device)
                    input_token = tokenizer(samples['raw_text'], add_special_tokens=False).to(self.device)
                    label_token = tokenizer(samples['label_text'], add_special_tokens=False).to(self.device)

                    batch_inputs_embeds = []
                    batch_attention_mask = []
                    batch_label_input_ids = []
                    for i in range(batch_size):
                        label_input_ids = label_token.input_ids[i] + [eos_id]
                        max_text_len = self.max_length - len(prompt_token.input_ids + label_input_ids) - 2
                        input_ids = input_token.input_ids[i][:max_text_len] + prompt_token.input_ids + label_input_ids
                        inputs_embeds = embedding(torch.tensor(input_ids).to(self.device))
                        inputs_embeds = torch.cat([bos_embeds, graph_embeds[i].unsqueeze(0), inputs_embeds], dim=0)
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

                    outputs = self.lm(inputs_embeds.half(), attention_mask.half(), labels=label_input_ids)

                    return outputs
                
                def generate(self, samples, graph_embeds, prompts):
                    batch_size = len(samples['node_id'])
                    tokenizer = self.lm.tokenizer
                    embedding = self.lm.embeddings

                    pad_embeds = embedding(torch.tensor(tokenizer.pad_token_id)).unsqueeze(0)
                    bos_embeds = embedding(torch.tensor(tokenizer.bos_token_id)).unsqueeze(0)

                    prompt_token = tokenizer(prompts, add_special_tokens=False).to(self.device)
                    input_token = tokenizer(samples['raw_text'], add_special_tokens=False).to(self.device)

                    batch_inputs_embeds = []
                    batch_attention_mask = []

                    for i in range(batch_size):
                        max_text_len = self.max_length - len(prompt_token.input_ids) - 2
                        input_ids = input_token.input_ids[i][:max_text_len] + prompt_token.input_ids
                        inputs_embeds = embedding(torch.tensor(input_ids).to(self.device))
                        inputs_embeds = torch.cat([bos_embeds, graph_embeds[i].unsqueeze(0), inputs_embeds], dim=0)

                        batch_inputs_embeds.append(inputs_embeds)
                        batch_attention_mask.append([1] * inputs_embeds.shape[0])

                    max_length = max([x.shape[0] for x in batch_inputs_embeds])
                    for i in range(len(samples['node_id'])):
                        pad_length = max_length - batch_inputs_embeds[i].shape[0]
                        batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
                        batch_attention_mask[i] = [0] * pad_length + batch_attention_mask[i]

                    inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.device)
                    attention_mask = torch.tensor(batch_attention_mask).to(self.device)

                    outputs = self.lm.generate(inputs_embeds.half(), attention_mask.half())

                    return outputs
            
        self.lm_model = LMModel(self.lm_type, self.max_length, self.model_path, self.lm_hidden_dim, self.output_dim, self.lora_config, self.lm_dropout, self.att_dropout, self.device)
        self.tokenizer = self.lm_model.lm.tokenizer
        self.embeddings = self.lm_model.lm.embeddings

        # GNN Params
        self.gnn_type = config['gp_gnn']
        self.gnn_input_dim = feat_dim
        self.gnn_hidden_dim = config['gp_hidden_dim']
        self.gnn_output_dim = config['gp_output_dim']
        self.gnn_layer_num = config['gp_layer_num']
        self.gnn_dropout = config['gp_dropout']
        self.gnn_num_heads = config['gp_num_heads']
        self.gnn_aggr = config['gp_aggr']

        class GNNModel(nn.Module):

            def __init__(self, gnn_type, input_dim, hidden_dim, output_dim, layer_num, dropout, num_heads, aggr, device):
                super(GNNModel, self).__init__()
                if gnn_type == 'GCN':
                    self.gnn = GCNNet(input_dim, hidden_dim, output_dim, layer_num, dropout).to(device)
                elif gnn_type == 'GAT':
                    self.gnn = GATNet(input_dim, hidden_dim, output_dim, layer_num, dropout, num_heads).to(device)
                elif gnn_type == 'SAGE':
                    self.gnn = SAGENet(input_dim, hidden_dim, output_dim, layer_num, dropout, aggr).to(device)
                elif gnn_type == 'SGC':
                    self.gnn = SGCNet(input_dim, hidden_dim, output_dim, layer_num, dropout).to(device)

            def forward(self, x, edge_index):
                embeds = self.gnn(x, edge_index)
                return embeds

        self.gnn_encoder = GNNModel(self.gnn_type, self.gnn_input_dim, self.gnn_hidden_dim, self.gnn_output_dim, self.gnn_layer_num, self.gnn_dropout, self.gnn_num_heads, self.gnn_aggr, self.device)
        
        # Proj Params
        self.proj_hidden_dim = config['gp_proj_hidden_dim']
        self.projector = nn.Sequential(
            nn.Linear(self.gnn_output_dim, self.proj_hidden_dim),
            nn.Sigmoid(),
            nn.Linear(self.proj_hidden_dim, self.lm_hidden_dim),
        ).to(self.device)

    def forward(self, samples, data, prompts=None):

        subset, edge_index, mapping, _ = k_hop_subgraph(samples['node_id'], self.gnn_layer_num, data.edge_index, relabel_nodes=True)
        graph_embeds = encode_graphs(data.x[subset], edge_index, mapping, self.gnn_encoder, self.projector, self.device)
        outputs = self.lm_model(samples, graph_embeds, prompts)

        return outputs
    
    def generate(self, samples, data, prompts=None):

        subset, edge_index, mapping, _ = k_hop_subgraph(samples['node_id'], self.gnn_layer_num, data.edge_index, relabel_nodes=True)
        graph_embeds = encode_graphs(data.x[subset], edge_index, mapping, self.gnn_encoder, self.projector, self.device)
        outputs = self.lm_model.generate(samples, graph_embeds, prompts)

        return outputs


class GraphPrompter(BaseModel):
        
    def __init__(self, task_loader, result_logger, config, checkpoint_path, dataset, model_name, model_path, local_ce, seed, device):
        super(GraphPrompter, self).__init__(task_loader, result_logger, config, checkpoint_path, dataset, model_name, local_ce, seed, device)

        self.lm_type = config['lm']
        self.lr = float(config['lr'])
        self.weight_decay = float(config['weight_decay'])
        self.model = GraphPrompterComponent(self.feat_dim, self.num_class, config, model_path, device)

    def get_optimizer(self, model):
        params = [p for _, p in model.named_parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            [{'params': params, 'lr': self.lr, 'weight_decay': self.weight_decay},],
            betas=(0.9, 0.95)
        )

        return optimizer

    def train(self, curr_session, curr_epoch, model, text_dataset, train_loader, optimizer, class_src, class_dst, config, device, prompts=None):
        model.train()
        all_loss, train_num = 0., 0
        for step, batch in enumerate(train_loader):
            if batch['node_id'].size(0) < 2:
                break
            optimizer.zero_grad()

            if self.lm_type in ['RoBERTa']:
                logits = model(batch, text_dataset.data)
                if self.local_ce:
                    logits = logits[:, class_src : class_dst]
                    labels = batch['labels'].to(device) - class_src
                else:
                    logits = logits[:, : class_dst]
                    labels = batch['labels'].to(device)
                loss = self.loss_func(logits, labels)
            elif self.lm_type in ['LLaMA']:
                outputs = model(batch, text_dataset.data, prompts)
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
    def valid(self, model, text_dataset, valid_loader, class_src, class_dst, config, device, prompts=None):
        model.eval()
        logits_list, preds_list, labels_list = [], [], []
        for _, batch in enumerate(valid_loader):
            if batch['node_id'].size(0) < 2:
                break

            if self.lm_type in ['RoBERTa']:
                logits = model(batch, text_dataset.data)
                if self.local_ce:
                    logits = logits[:, class_src : class_dst]
                    labels = batch['labels'].to(device) - class_src
                else:
                    logits = logits[:, : class_dst]
                    labels = batch['labels'].to(device)
                logits = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                labels = batch['labels']
                logits_list.extend(logits)
                
            elif self.lm_type in ['LLaMA']:
                preds = model.generate(batch, text_dataset.data, prompts)
                labels = batch['label_text']

            preds_list.extend(preds)
            labels_list.extend(labels)

        if self.lm_type in ['RoBERTa']:
            logits = torch.stack(logits_list, dim=0)
            labels = torch.stack(labels_list, dim=0)
            preds = torch.stack(preds_list, dim=0)
        elif self.lm_type in ['LLaMA']:
            logits = None
            labels = labels_list
            preds = preds_list

        acc, f1 = self.get_metric(logits, preds, labels)

        return acc, f1

    @torch.no_grad()
    def evaluate(self, model, text_dataset, test_loader, class_dst, config, device, prompts=None):
        model.eval()
        logits_list, preds_list, labels_list = [], [], []
        for _, batch in enumerate(test_loader):
            if batch['node_id'].size(0) < 2:
                break

            if self.lm_type in ['RoBERTa']:
                logits = model(batch, text_dataset.data)
                logits = torch.softmax(logits[:, : class_dst], dim=1)
                preds = torch.argmax(logits, dim=1)
                labels = batch['labels']
                logits_list.extend(logits)
                
            elif self.lm_type in ['LLaMA']:
                preds = model.generate(batch, text_dataset.data, prompts)
                labels = batch['label_text']

            preds_list.extend(preds)
            labels_list.extend(labels)

        if self.lm_type in ['RoBERTa']:
            logits = torch.stack(logits_list, dim=0)
            labels = torch.stack(labels_list, dim=0)
            preds = torch.stack(preds_list, dim=0)
        elif self.lm_type in ['LLaMA']:
            logits = None
            labels = labels_list
            preds = preds_list

        acc, f1 = self.get_metric(logits, preds, labels)

        return acc, f1

    def fit(self, iter):
        optimizer = self.get_optimizer(self.model)

        if self.lm_type in ['LLaMA']:
            label_text_list = get_label_text(self.dataset)

        for curr_session in range(self.session_num):
            if curr_session != 0:
                _reload_best_model(self.model, self.checkpoint_path, self.dataset, self.model_name, self.seed)

            class_src, class_dst, text_dataset_iso, text_dataset_joint, train_loader, valid_loader, test_loader_isolate, test_loader_joint = self.task_loader.get_task(curr_session)
            
            prompts = None
            if self.lm_type in ['LLaMA']:
                label_text = label_text_list[ :class_dst]
                prompts = get_genreal_prompts(self.dataset, label_text)

            progress_bar = tqdm(range(self.config['epochs']))
            progress_bar.set_description(f'Training | Iter {iter} | Session {curr_session}')

            tolerate, best_acc_valid = 0, 0.
            for epoch in range(self.config['epochs']):
                loss = self.train(curr_session, epoch, self.model, text_dataset_iso, train_loader, optimizer, class_src, class_dst, self.config, self.device, prompts)
                progress_bar.write("Session: {} | Epoch: {} | Loss: {:.4f}".format(curr_session, epoch, loss))

                if epoch > 0 and epoch % self.config['valid_epoch'] == 0:
                    acc_valid, f1_valid = self.valid(self.model, text_dataset_iso, valid_loader, class_src, class_dst, self.config, self.device, prompts)
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

        return self.result_logger
import copy
import torch
import random
import contextlib
import numpy as np
import scipy.sparse as sp
import torch.nn as nn
import torch.nn.functional as F

from LLM4GCL.models import BaseModel
from LLM4GCL.backbones import LLaMANet
from LLM4GCL.common.utils import adjust_learning_rate, _save_checkpoint, _reload_best_model, normalize_adj_matrix
from LLM4GCL.common.prompts import get_LLaGA_prompts

from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset

DEFAULT_GRAPH_PAD_ID = -500 
BOS = '<s>[INST]'
EOS_USER = '[/INST]'
EOS = '</s>'
IGNORE_INDEX = -100 

def build_hopfield_emb(graph, n_layers):
    x, edge_index = graph.x, graph.edge_index
    num_nodes = x.shape[0]
    norm_adj = normalize_adj_matrix(edge_index, num_nodes, edge_index.device)
    
    all_embeds = [x]
    for _ in range(n_layers):
        x = torch.sparse.mm(norm_adj, x)
        all_embeds.append(x)

    return all_embeds


def build_laplacian_emb(k_hop=2, sample_size=10):
    n = int(((sample_size ** (k_hop+1)) - 1) / (sample_size - 1))
    edge_row, edge_col = [], []
    last_hop_start = last_hop_end = 0 
    
    for i in range(k_hop):
        edge_row.extend([x for x in range(last_hop_start, last_hop_end + 1) for _ in range(sample_size)])
        edge_col.extend(list(range(last_hop_start * sample_size + 1, last_hop_end * sample_size + sample_size +1)))
        last_hop_start = last_hop_start * sample_size + 1 
        last_hop_end = last_hop_end * sample_size + sample_size
    
    edge_row = np.array(edge_row)
    edge_col = np.array(edge_col)
    A = sp.coo_matrix((np.array([1]*len(edge_row)),(edge_col, edge_row)), shape=(n,n))
    L = sp.eye(n) - A

    _, EigVec = np.linalg.eig(L.toarray())

    PE = torch.FloatTensor(EigVec)

    return PE


class LLaGADataset(Dataset):
    def __init__(self, graph, node_index, label_text, dataset, neighbor_template, k_hop, sample_size, repeats=1):
        super(LLaGADataset, self).__init__() 
        self.graph = graph
        self.node_index = node_index
        self.label_text = label_text
        self.dataset = dataset
        self.neighbor_template = neighbor_template
        self.k_hop = k_hop
        self.sample_size = sample_size
        self.repeats = repeats 
        
        self.data_list = self._format_data()
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        return self.data_list[index]
       
    def _format_data(self, ):
        if self.neighbor_template == "ND":
            edge_list = self._prepare_edge_list(self.graph.edge_index.to("cpu"), self.graph.x.shape[0])

        SYSTEM_PROMPT, description = get_LLaGA_prompts(self.dataset, self.label_text)
        
        available_data_list = []
        for node in self.node_index:
            sample = {
                "id": node, 
                "query": f"{SYSTEM_PROMPT}\n{description}",
                "origin_txt": self.graph.raw_texts[node],
                "label": self.label_text[self.graph.y[node].item()], 
            }
            if self.neighbor_template == "ND":
                neigh_seq = self._get_node_neighbor_detail(edge_list, node, k_hop=self.k_hop, sample_size=self.sample_size) 
                sample["graph"] = neigh_seq
            
            available_data_list.extend([sample] * self.repeats)
        
        return available_data_list

    def _prepare_edge_list(self, edge_index, num_nodes):
        row, col = edge_index
        edge_list = [[] for _ in range(num_nodes)] 
        
        row, col = row.numpy(), col.numpy()
        for i in range(row.shape[0]):
            edge_list[row[i]].append(int(col[i]))
        return edge_list 


    def _get_node_neighbor_detail(self, edge_list, node_idx, k_hop, sample_size, avoid_idx=None):
        assert k_hop > 0 and sample_size > 0 
        neighbors = [[node_idx]]
        for t in range(k_hop):
            last_hop = neighbors[-1]
            current_hop = [] 
            for i in last_hop:
                if i == DEFAULT_GRAPH_PAD_ID: 
                    current_hop.extend([DEFAULT_GRAPH_PAD_ID] * sample_size)
                    continue 
                
                node_neighbor = copy.copy(edge_list[i])
                if t == 0 and avoid_idx is not None and avoid_idx in node_neighbor:
                    node_neighbor.remove(avoid_idx)
                if len(node_neighbor) > sample_size:
                    sampled_neighbor = random.sample(node_neighbor, sample_size)
                else:
                    sampled_neighbor = node_neighbor + [DEFAULT_GRAPH_PAD_ID] * (sample_size - len(node_neighbor))
                
                current_hop.extend(sampled_neighbor)
            neighbors.append(current_hop)
            
        node_sequence = [n for hop in neighbors for n in hop]

        return node_sequence


class LLaGAModel(torch.nn.Module):
    def __init__(self, graph_input_dim, config, model_path, device):
        super(LLaGAModel, self).__init__()

        self.lm_type = config['lm']
        self.hidden_dim = 4096
        self.lr = float(config['lr'])
        self.weight_decay = float(config['weight_decay'])
        self.lora_config = config['LoRA']
        self.dropout = config['dropout']
        self.att_dropout = config['att_dropout']
        self.max_length = config['max_length']
        self.model_path = model_path
        self.device = device

        self.llm_freeze = config['llm_freeze']
        self.neighbor_template = config['neighbor_template']
        self.neighbor_desc_mean_pooling = config['nd_mean']
        self.k_hop = config['k_hop']
        self.sample_size = config['sample_size']
        self.hop_field = config['hop_field']
        self.proj_layer = config['proj_layer']
        self.proj_hidden_dim = self.hidden_dim
        self.proj_output_dim = self.hidden_dim

        if self.llm_freeze:
            self.lora_config['use_lora'] = False
        self.lm = LLaMANet(model_path, self.lora_config, self.dropout, self.att_dropout).to(self.device)
        # Freeze LLM
        if self.llm_freeze:
            for _, param in self.lm.named_parameters(): 
                param.requires_grad = False 
        
        # Linear Projector
        proj_input_dim = graph_input_dim + int(((self.sample_size ** (self.k_hop + 1)) - 1) / (self.sample_size - 1)) if self.neighbor_template == "ND" else graph_input_dim
        proj_hidden_dim = self.proj_hidden_dim
        proj_output_dim = self.proj_output_dim
        proj_layers = []
        assert self.proj_layer >= 2, "Layers in Linear Projection should be greater than 2!"

        for i in range(self.proj_layer):
            if i == 0:
                proj_layers.append(nn.Linear(proj_input_dim, proj_hidden_dim))
                proj_layers.append(nn.LeakyReLU())
            elif i != self.proj_layer - 1:
                proj_layers.append(nn.Linear(proj_hidden_dim, proj_hidden_dim))
                proj_layers.append(nn.LeakyReLU())
            else:
                proj_layers.append(nn.Linear(proj_hidden_dim, proj_output_dim))
        self.graph_proj = nn.Sequential(*proj_layers).to(self.device)
    
    def maybe_autocast(self, dtype=torch.bfloat16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()
    
    def encode_subgraph_hopfield(self, node_index, graph_embed):
        node_index = node_index
        neighbor_embs = []
        for cur_layer_emb in graph_embed: 
            neighbor_embs.append(cur_layer_emb[node_index, :])
        graph_emb = torch.stack(neighbor_embs, dim=0).to(self.device)
        graph_features = self.graph_proj(graph_emb) 
        # print(graph_emb.shape, graph_features.shape)
        return graph_features
    
    def encode_subgraph_neighbordesc(self, node_indexes, graph_embed, position_embed):
        graph_embed, position_embed = graph_embed.to(self.device), position_embed.to(self.device)
        mask = node_indexes != DEFAULT_GRAPH_PAD_ID 
        masked_graph_emb = graph_embed[node_indexes[mask]]
        
        s, d = node_indexes.shape[0], masked_graph_emb.shape[1]
        embed = torch.zeros((s, d)).to(self.device)
        embed[mask] = masked_graph_emb 
        graph_embed = torch.cat([embed, position_embed], dim=1).to(self.device)
        
        graph_features = self.graph_proj(graph_embed)
        graph_features[node_indexes == DEFAULT_GRAPH_PAD_ID] = 0. 
        
        if self.neighbor_desc_mean_pooling:
            graph_features = torch.mean(graph_features, dim=0, keepdim=True)
            
        return graph_features 
    
    def forward(self, samples, graph_embed, position_embed):
        tokenizer = self.lm.tokenizer
        word_embedding = self.lm.embeddings
        id_list = [item["id"] for item in samples]
        query_list = [item["query"] for item in samples]
        label_list = [item["label"] for item in samples]

        queries = tokenizer(query_list, add_special_tokens=False) # input query
        labels = tokenizer(label_list, add_special_tokens=False) # output ground-truth label
        
        eos_tokens = tokenizer(EOS, add_special_tokens=False)
        eos_user_tokens = tokenizer(EOS_USER, add_special_tokens=False)
        bos_embeds = word_embedding(tokenizer(BOS, add_special_tokens=False, return_tensors='pt').input_ids[0].to(self.device))
        pad_embeds = word_embedding(torch.tensor(tokenizer.pad_token_id).to(self.device)).unsqueeze(0)
        
        batch_size = len(id_list)
        
        batch_inputs_embeds = []
        batch_attention_mask = []
        batch_label_input_ids = []
        
        if self.neighbor_template == "ND":
            batch_graph_indexes = torch.tensor([item["graph"] for item in samples]).to(self.device)
        else:
            batch_graph_indexes = id_list
        
        for i in range(batch_size):
            label_input_ids = labels.input_ids[i] + eos_tokens.input_ids
            input_ids = queries.input_ids[i][:self.max_length] + eos_user_tokens.input_ids + label_input_ids
            input_embeds = word_embedding(torch.tensor(input_ids).to(self.device))

            if self.neighbor_template == "ND":
                graph_embeds = self.encode_subgraph_neighbordesc(batch_graph_indexes[i], graph_embed, position_embed)
            else:
                graph_embeds = self.encode_subgraph_hopfield(batch_graph_indexes[i], graph_embed)
            
            input_embeds = torch.cat([bos_embeds, graph_embeds, input_embeds], dim=0)
            
            batch_inputs_embeds.append(input_embeds)
            batch_attention_mask.append([1] * input_embeds.shape[0])
            label_input_ids = [IGNORE_INDEX] * (input_embeds.shape[0] - len(label_input_ids)) + label_input_ids
            batch_label_input_ids.append(label_input_ids)
        
        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(batch_size):
            pad_length = max_length - batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0] * pad_length + batch_attention_mask[i]
            
            batch_label_input_ids[i] = [IGNORE_INDEX] * pad_length + batch_label_input_ids[i]
        
        input_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.device)
        label_input_ids = torch.tensor(batch_label_input_ids).to(self.device)
        
        with self.maybe_autocast():
            outputs = self.lm(
                input_embeds, 
                attention_mask=attention_mask, 
                labels=label_input_ids
            )
        return outputs
    
    def generate(self, samples, graph_embed, position_embed):
        tokenizer = self.lm.tokenizer
        word_embedding = self.lm.embeddings
        id_list = [item["id"] for item in samples]
        query_list = [item["query"] for item in samples]

        queris = tokenizer(query_list, add_special_tokens=False)
        
        eos_user_tokens = tokenizer(EOS_USER, add_special_tokens=False)
        bos_embeds = word_embedding(tokenizer(BOS, add_special_tokens=False, return_tensors='pt').input_ids[0].to(self.device))
        pad_embeds = word_embedding(torch.tensor(tokenizer.pad_token_id).to(self.device)).unsqueeze(0)
        
        batch_size = len(id_list)
        
        batch_inputs_embeds = []
        batch_attention_mask = []
        
        if self.neighbor_template == "ND":
            batch_graph_indexes = torch.tensor([item["graph"] for item in samples]).to(self.device)
        else:
            batch_graph_indexes = id_list
        
        for i in range(batch_size):
            input_ids = queris.input_ids[i][:self.max_length] + eos_user_tokens.input_ids
            input_embeds = word_embedding(torch.tensor(input_ids).to(self.device))
            
            if self.neighbor_template == "ND":
                graph_embeds = self.encode_subgraph_neighbordesc(batch_graph_indexes[i], graph_embed, position_embed)
            else:
                graph_embeds = self.encode_subgraph_hopfield(batch_graph_indexes[i], graph_embed)
            input_embeds = torch.cat([bos_embeds, graph_embeds, input_embeds], dim=0)
            
            batch_inputs_embeds.append(input_embeds)
            batch_attention_mask.append([1] * input_embeds.shape[0])
        
        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(batch_size):
            pad_length = max_length - batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0] * pad_length + batch_attention_mask[i]
            
        input_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.device)
        
        with self.maybe_autocast():
            outputs = self.lm.generate(
                input_embeds,
                attention_mask=attention_mask,
            )
        
        return outputs


class LLaGA(BaseModel):
        
    def __init__(self, task_loader, result_logger, config, checkpoint_path, dataset, model_name, model_path, local_ce, seed, device):
        super(LLaGA, self).__init__(task_loader, result_logger, config, checkpoint_path, dataset, model_name, local_ce, seed, device)

        self.lm_type = config['lm']
        self.lr = float(config['lr'])
        self.weight_decay = float(config['weight_decay'])

        self.neighbor_template = config['neighbor_template']
        self.neighbor_desc_mean_pooling = config['nd_mean']
        self.k_hop = config['k_hop']
        self.sample_size = config['sample_size']
        self.hop_field = config['hop_field']

        self.model = LLaGAModel(self.feat_dim, config, model_path, device)

    def get_optimizer(self, model):
        params = [p for _, p in model.named_parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            [{'params': params, 'lr': self.lr, 'weight_decay': self.weight_decay},],
            betas=(0.9, 0.95)
        )

        return optimizer

    def train(self, curr_session, curr_epoch, model, text_dataset, train_loader, optimizer, class_src, class_dst, config, device, label_text, graph_embed, position_embed):
        model.train()
        all_loss, train_num = 0., 0
        for step, batch in enumerate(train_loader):
            if batch['node_id'].size(0) < 2:
                break
            optimizer.zero_grad()
            samples = LLaGADataset(text_dataset.data, batch['node_id'].tolist(), label_text, self.dataset, self.neighbor_template, self.k_hop, self.sample_size)
            outputs = model(samples, graph_embed, position_embed)
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
    def valid(self, model, text_dataset, valid_loader, class_src, class_dst, config, device, label_text, graph_embed, position_embed):
        model.eval()
        logits_list, preds_list, labels_list = [], [], []
        for _, batch in enumerate(valid_loader):
            if batch['node_id'].size(0) < 2:
                break

            samples = LLaGADataset(text_dataset.data, batch['node_id'].tolist(), label_text, self.dataset, self.neighbor_template, self.k_hop, self.sample_size)
            preds = model.generate(samples, graph_embed, position_embed)
            final_preds = [pred[:pred.index("</s>")] if "</s>" in pred else pred for pred in preds]
            labels = batch['label_text']

            preds_list.extend(final_preds)
            labels_list.extend(labels)

        labels = labels_list
        preds = preds_list

        acc, f1 = self.get_metric(None, preds, labels)

        return acc, f1

    @torch.no_grad()
    def evaluate(self, model, text_dataset, test_loader, class_dst, config, device, label_text, graph_embed, position_embed):
        model.eval()
        logits_list, preds_list, labels_list = [], [], []
        for _, batch in enumerate(test_loader):
            if batch['node_id'].size(0) < 2:
                break

            samples = LLaGADataset(text_dataset.data, batch['node_id'].tolist(), label_text, self.dataset, self.neighbor_template, self.k_hop, self.sample_size)
            preds = model.generate(samples, graph_embed, position_embed)
            final_preds = [pred[:pred.index("</s>")] if "</s>" in pred else pred for pred in preds]
            labels = batch['label_text']

            preds_list.extend(final_preds)
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

            class_src, class_dst, text_dataset_iso, text_dataset_joint, train_loader, valid_loader, test_loader_isolate, test_loader_joint = self.task_loader.get_task(curr_session)
            label_text = label_text_list[ : class_dst]
            graph_embedding_iso = text_dataset_iso.data.x if self.neighbor_template == "ND" else build_hopfield_emb(text_dataset_iso.data, self.hop_field)
            graph_embedding_joint = text_dataset_joint.data.x if self.neighbor_template == "ND" else build_hopfield_emb(text_dataset_joint.data, self.hop_field)
            position_embed = build_laplacian_emb(self.k_hop, self.sample_size) if self.neighbor_template == "ND" else None

            progress_bar = tqdm(range(self.config['epochs']))
            progress_bar.set_description(f'Training | Iter {iter} | Session {curr_session}')

            tolerate, best_acc_valid = 0, 0.
            for epoch in range(self.config['epochs']):
                loss = self.train(curr_session, epoch, self.model, text_dataset_iso, train_loader, optimizer, class_src, class_dst, self.config, self.device, label_text, graph_embedding_iso, position_embed)
                progress_bar.write("Session: {} | Epoch: {} | Loss: {:.4f}".format(curr_session, epoch, loss))

                if epoch > 0 and epoch % self.config['valid_epoch'] == 0:
                    acc_valid, f1_valid = self.valid(self.model, text_dataset_iso, valid_loader, class_src, class_dst, self.config, self.device, label_text, graph_embedding_iso, position_embed)
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
            curr_acc_test_isolate, curr_f1_test_isolate = self.evaluate(self.model, text_dataset_iso, test_loader_isolate, class_dst, self.config, self.device, label_text, graph_embedding_iso, position_embed)
            curr_acc_test_joint, curr_f1_test_joint = self.evaluate(self.model, text_dataset_joint, test_loader_joint, class_dst, self.config, self.device, label_text, graph_embedding_joint, position_embed)

            acc_list = []
            for s in range(curr_session):
                _, _, text_dataset_iso, _, _, _, test_loader_isolate, _ = self.task_loader.get_task(s)
                graph_embedding_iso = text_dataset_iso.data.x if self.neighbor_template == "ND" else build_hopfield_emb(text_dataset_iso.data, self.hop_field)
                prev_acc_test_isolate, prev_f1_test_isolate = self.evaluate(self.model, text_dataset_iso, test_loader_isolate, class_dst, self.config, self.device, label_text, graph_embedding_iso, position_embed)
                acc_list.append(prev_acc_test_isolate)
            acc_list.append(curr_acc_test_isolate)

            print("Session: {} | Iso. Acc Test: {:.4f} | Iso. F1 Test: {:.4f}".format(curr_session, curr_acc_test_isolate, curr_f1_test_isolate))
            print("Session: {} | Jot. Acc Test: {:.4f} | Jot. F1 Test: {:.4f}".format(curr_session, curr_acc_test_joint, curr_f1_test_joint))

            self.result_logger.add_new_results(acc_list, curr_acc_test_joint)

        return self.result_logger
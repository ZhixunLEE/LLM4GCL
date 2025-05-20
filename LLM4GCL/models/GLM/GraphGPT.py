import os 
import sys 
import csv
import time
import json 
import torch
import random
import contextlib
import numpy as np
import torch.nn as nn

from LLM4GCL.models import BaseModel
from LLM4GCL.backbones import LLaMANet
from LLM4GCL.common.utils import adjust_learning_rate, _save_checkpoint, _reload_best_model
from LLM4GCL.common.prompts import MATCHING_TEMPLATES, GraphGPT_DESC

from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader
from torch_geometric.utils import k_hop_subgraph

IGNORE_INDEX = -100
BOS = '<s>[INST]'
EOS_USER = '[/INST]'
EOS = '</s>'
IGNORE_INDEX = -100 
DEFAULT_G_START_TOKEN = "<g_start>"
DEFAULT_G_END_TOKEN = "<g_end>"

def fetch_title(txt, max_length=512):
    title = None
    if ":" in txt:
        title= txt.split(":")[0]
    title= txt.split(".")[0]
    
    return title[:max_length]


def save_pretrain_checkpoint(model, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    state_dict = {
        k: v for k, v in model.state_dict().items()
        if v.requires_grad
    }
    save_obj = {
        "model": state_dict,
    }
    torch.save(save_obj, file_path)

def reload_pretrain_checkpoint(model, file_path):
    checkpoint = torch.load(file_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=False)


# Example Data: https://huggingface.co/datasets/Jiabin99/graph-matching
class GraphMatchingDataset(Dataset):
    def __init__(self, graph_data, node_idx, k_hop=1, num_sampled_neighbors=8, graph_type="academic_network", sample_times=1):
        self.graph_data = graph_data 
        self.node_idx = node_idx
        self.k_hop = k_hop
        self.num_sampled_neighbors = num_sampled_neighbors
        self.sample_times = sample_times
        self.query_template = MATCHING_TEMPLATES[graph_type]
        self.graph_type = graph_type
        
        self.all_data = self._prepare_matching_data()
    
    def __len__(self):
        return len(self.all_data)
    
    def __getitem__(self, idx):
        return self.all_data[idx]
        
    def _prepare_matching_data(self):
        data_samples = []
        for node in self.node_idx:
            neighbors, _, _, _ = k_hop_subgraph(node, num_hops=self.k_hop, edge_index=self.graph_data.edge_index)
            neighbors = neighbors.numpy()
            if len(neighbors.tolist()) == 0:
                continue
            
            for _ in range(self.sample_times):
                subset = np.random.choice(neighbors, size=self.num_sampled_neighbors).tolist()
                subset = list(set(subset))
                if node not in subset:
                    subset = [node] + subset[:-1] 
                else:
                    target_idx = subset.index(node)
                    subset[target_idx] = subset[0]
                    subset[0] = node
                
                if len(subset) < self.num_sampled_neighbors:
                    pad_length = self.num_sampled_neighbors - len(subset)
                    subset = subset + [node] * pad_length
                    
                assert subset[0] == node 
                
                texts = []
                for token_id in range(len(subset)): 
                    raw_text = fetch_title(self.graph_data.raw_texts[subset[token_id]])
                    texts.append([token_id, raw_text]) # origin_id in graph-tokens, corresponding text
                
                # Re-order the texts 
                random.shuffle(texts)
                tokenid2text_mapping = {pairs[0]+1: pairs[1] for text_id, pairs in enumerate(texts)}
                query_graph_texts = ". ".join([f"{text_id+1}. {pairs[1]}" for text_id, pairs in enumerate(texts)])
                
                if self.graph_type == "academic_network":
                    cur_query = self.query_template.replace("{{paper_titles}}", query_graph_texts)
                    cur_response = ". ".join([f"Graph token {k} corresponds to paper {tokenid2text_mapping[k]}" for k in sorted(tokenid2text_mapping.keys()) ])
                    cur_response = "Based on the given graph tokens and the list of paper titles, we obtain the matching of graph tokens and papers as follows: " + cur_response
                elif self.graph_type == "social_network":
                    cur_query = self.query_template.replace("{{user_profiles}}", query_graph_texts)
                    cur_response = ". ".join([f"Graph token {k} corresponds to user {tokenid2text_mapping[k]}" for k in sorted(tokenid2text_mapping.keys()) ])
                    cur_response = "Based on the given graph tokens and the descriptions of users, we obtain the matching of graph tokens and users as follows: " + cur_response
                elif self.graph_type == "ecommerce_network":
                    cur_query = self.query_template.replace("{{item_comments}}", query_graph_texts)
                    cur_response = ". ".join([f"Graph token {k} corresponds to item {tokenid2text_mapping[k]}" for k in sorted(tokenid2text_mapping.keys())])
                    cur_response = "Based on the given graph tokens and the comments of items, we obtain the matching of graph tokens and items as follows: " + cur_response

                sample = {
                    "id": node,
                    "nodes": torch.LongTensor(subset),
                    "query": cur_query,
                    "label": cur_response
                }
                data_samples.append(sample)

        return data_samples


# Example Data: https://huggingface.co/datasets/Jiabin99/Arxiv-PubMed-mix-NC-LP
class GraphInstructionTuningDataset(Dataset):
    def __init__(self, graph_data, label_texts, node_idx, k_hop=1, maximum_neighbors=4, dataset_name="cora"):
        self.graph_data = graph_data
        self.num_nodes = graph_data.x.shape[0]
        self.k_hop = k_hop 
        self.maximum_neighbors = maximum_neighbors
        self.label_names = label_texts

        self.node_idx = node_idx
        
        label_names = ", ".join(label_texts)
        self.query_prompt = GraphGPT_DESC[dataset_name].replace("{{label_names}}", label_names)
        self.data_list = self.format_data()
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        return self.data_list[index]
    
    def format_data(self):
                
        available_data_list = []
        for cur_node in self.node_idx:
            neighbors, _, _, _ = k_hop_subgraph(cur_node, num_hops=self.k_hop, edge_index=self.graph_data.edge_index)
            neighbors = neighbors.numpy().tolist()
        
            if len(neighbors) > self.maximum_neighbors:
                neighbors = np.random.choice(np.array(neighbors), size=self.maximum_neighbors).tolist()
                neighbors = [cur_node] + neighbors
            else: 
                pad_length = self.maximum_neighbors - len(neighbors) 
                neighbors = [cur_node] + neighbors + [cur_node] * pad_length
            
            assert cur_node == neighbors[0]
        
            cur_query = self.query_prompt.replace("{{raw_text}}", self.graph_data.raw_texts[cur_node])
            cur_response = self.label_names[self.graph_data.y[cur_node].item()]
        
            available_data_list.append({
                "id": cur_node, 
                "nodes": torch.LongTensor(neighbors),
                "query": cur_query,
                "label": cur_response
            })  

        return available_data_list   


class GraphGPTModel(torch.nn.Module):
    def __init__(self, config, llm_path, output_dim, graph_embedding, device, stage="matching"):
        super().__init__()
        
        self.stage = stage
        if stage == "matching":
            self.max_txt_len = config['s1_max_txt_length'] 
            self.max_new_tokens = config['s1_max_ans_length']  
        else:
            self.max_txt_len = config['s2_max_txt_length'] 
            self.max_new_tokens = config['s2_max_ans_length'] 

        self.llm = config['llm']
        self.lora_config = config['LoRA']
        self.dropout = config['dropout']
        self.att_dropout = config['att_dropout']
        self.hidden_dim = 4096
        self.device = device

        if self.llm == 'LLaMA':
            self.model= LLaMANet(llm_path, self.lora_config, self.dropout, self.att_dropout).to(self.device)
        self.graph_embedding = graph_embedding.to(self.device)

        input_dim = self.graph_embedding.shape[1]
        self.graph_projector = nn.Linear(input_dim, self.hidden_dim).to(self.device)
        
    
    def maybe_autocast(self, dtype=torch.bfloat16):
        enable_autocast = self.device != torch.device("cpu")
        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    def encode_graphs(self, node_index):
        node_embeds = self.graph_embedding[node_index]
        return self.graph_projector(node_embeds)
    
    def forward(self, samples):
        queries = self.model.tokenizer(samples["query"], add_special_tokens=False)
        labels = self.model.tokenizer(samples["label"], add_special_tokens=False)
        
        eos_tokens = self.model.tokenizer(EOS, add_special_tokens=False)
        eos_user_tokens = self.model.tokenizer(EOS_USER, add_special_tokens=False)
        gstart_embeds = self.model.embeddings(self.model.tokenizer(DEFAULT_G_START_TOKEN, add_special_tokens=False, return_tensors='pt').input_ids[0].to(self.device))
        gend_embeds = self.model.embeddings(self.model.tokenizer(DEFAULT_G_END_TOKEN, add_special_tokens=False, return_tensors='pt').input_ids[0].to(self.device))
        bos_embeds = self.model.embeddings(self.model.tokenizer(BOS, add_special_tokens=False, return_tensors='pt').input_ids[0].to(self.device))
        pad_embeds = self.model.embeddings(torch.tensor(self.model.tokenizer.pad_token_id).to(self.device)).unsqueeze(0)
        
        batch_size = len(samples["id"])
        
        batch_input_embeds = []
        batch_attention_mask = []
        batch_label_input_ids = []
        
        batch_node_indexes  = samples["nodes"].to(self.device)
        
        for i in range(batch_size):
            label_input_ids = labels.input_ids[i][:self.max_new_tokens] + eos_tokens.input_ids 
            
            input_ids = queries.input_ids[i][:self.max_txt_len] + eos_user_tokens.input_ids + label_input_ids
            input_embeds = self.model.embeddings(torch.tensor(input_ids).to(self.device))
            
            node_embeds = self.encode_graphs(batch_node_indexes[i, :])
            graph_embeds = torch.cat([gstart_embeds, node_embeds, gend_embeds], dim=0)
            input_embeds = torch.cat([bos_embeds, graph_embeds, input_embeds], dim=0)
            
            batch_input_embeds.append(input_embeds)
            batch_attention_mask.append([1] * input_embeds.shape[0])
            label_input_ids = [IGNORE_INDEX] * (input_embeds.shape[0] - len(label_input_ids)) + label_input_ids
            batch_label_input_ids.append(label_input_ids)
        
        max_length = max([x.shape[0] for x in batch_input_embeds]) 
        for i in range(batch_size):
            pad_length = max_length - batch_input_embeds[i].shape[0]
            batch_input_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_input_embeds[i]])
            batch_attention_mask[i] = [0] * pad_length + batch_attention_mask[i]
            
            batch_label_input_ids[i] = [IGNORE_INDEX] * pad_length + batch_label_input_ids[i]
            
        input_embeds = torch.stack(batch_input_embeds, dim=0).to(self.device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.device)
        label_input_ids = torch.tensor(batch_label_input_ids).to(self.device)
        
        with self.maybe_autocast():
            outputs = self.model(
                input_embeds, 
                attention_mask=attention_mask,
                labels=label_input_ids
            )
            
        return outputs.loss
    
    def inference(self, samples):
        queries = self.model.tokenizer(samples["query"])
        
        eos_user_tokens = self.model.tokenizer(EOS_USER, add_special_tokens=False)
        gstart_embeds = self.model.embeddings(self.model.tokenizer(DEFAULT_G_START_TOKEN, add_special_tokens=False, return_tensors='pt').input_ids[0].to(self.device))
        gend_embeds = self.model.embeddings(self.model.tokenizer(DEFAULT_G_END_TOKEN, add_special_tokens=False, return_tensors='pt').input_ids[0].to(self.device))
        bos_embeds = self.model.embeddings(self.model.tokenizer(BOS, add_special_tokens=False, return_tensors='pt').input_ids[0].to(self.device))
        pad_embeds = self.model.embeddings(torch.tensor(self.model.tokenizer.pad_token_id).to(self.device)).unsqueeze(0)
        
        batch_size = len(samples["id"])
        
        batch_input_embeds = []
        batch_attention_mask = []

        batch_node_indexes  = samples["nodes"].to(self.device)
        
        for i in range(batch_size):
            input_ids = queries.input_ids[i][:self.max_txt_len] + eos_user_tokens.input_ids 
            input_embeds = self.model.embeddings(torch.tensor(input_ids).to(self.device))
            
            node_embeds = self.encode_graphs(batch_node_indexes[i, :])
            graph_embeds = torch.cat([gstart_embeds, node_embeds, gend_embeds], dim=0)
            input_embeds = torch.cat([bos_embeds, graph_embeds, input_embeds], dim=0)
            
            batch_input_embeds.append(input_embeds)
            batch_attention_mask.append([1] * input_embeds.shape[0])
        
        max_length = max([x.shape[0] for x in batch_input_embeds])
        
        for i in range(batch_size):
            pad_length = max_length - batch_input_embeds[i].shape[0]
            batch_input_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_input_embeds[i]], dim=0)
            batch_attention_mask[i] = [0] * pad_length + batch_attention_mask[i]
        
        input_embeds = torch.stack(batch_input_embeds, dim=0).to(self.device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.device)
        
        with self.maybe_autocast():
            preds = self.model.generate(
                input_embeds, 
                attention_mask=attention_mask,
            )
        
        return samples["id"], preds 


class GraphGPT(BaseModel):
        
    def __init__(self, task_loader, result_logger, config, checkpoint_path, dataset, model_name, model_path, local_ce, seed, device):
        super(GraphGPT, self).__init__(task_loader, result_logger, config, checkpoint_path, dataset, model_name, local_ce, seed, device)

        self.llm = config['llm']
        self.output_dir = config['output_dir']
        self.wd = float(config['wd'])
        config['lr'] = float(config['s2_lr'])
        config['epochs'] = config['s2_epoch']

        self.do_stage1 = config['do_stage1']
        self.s1_epoch = config['s1_epoch']
        self.s1_batch_size = config['s1_batch_size']
        self.s1_lr = float(config['s1_lr'])
        self.s1_k_hop = config['s1_k_hop']
        self.s1_num_neighbors = config['s1_num_neighbors']

        self.do_stage2 = config['do_stage2']
        self.s2_epoch = config['s2_epoch']
        self.s2_batch_size = config['s2_batch_size']
        self.s2_lr = float(config['s2_lr'])
        self.s2_num_neighbors = config['s2_num_neighbors']
        self.s2_patience = config['s2_patience']

        self.model = GraphGPTModel(config, model_path, self.num_class, task_loader.data.x, self.device, stage="matching")


    def train(self, curr_session, curr_epoch, model, text_dataset, train_loader, optimizer, class_src, class_dst, config, device):
        model.train()
        all_loss, train_num = 0., 0
        for step, batch in enumerate(train_loader):
            if batch['id'].size(0) < 2:
                break
            optimizer.zero_grad()

            loss = model(batch)
            loss.backward()
            all_loss += loss * batch['id'].size(0)
            train_num += batch['id'].size(0)

            clip_grad_norm_(optimizer.param_groups[0]['params'], 0.1)
            if (step + 1) % config['grad_steps'] == 0:
                adjust_learning_rate(optimizer.param_groups[0], step / len(train_loader) + curr_epoch, config)
            optimizer.step()

        return all_loss / train_num

    @torch.no_grad()
    def valid(self, model, text_dataset, valid_loader, class_src, class_dst, config, device):
        return self.evaluate(model, text_dataset, valid_loader, class_dst, config, device)

    @torch.no_grad()
    def evaluate(self, model, text_dataset, test_loader, class_dst, config, device):
        model.eval()
        preds_list, labels_list = [], []
        for _, batch in enumerate(test_loader):
            if batch['id'].size(0) < 2:
                break

            _, preds = model.inference(batch)
            final_preds = [pred[:pred.index("</s>")] if "</s>" in pred else pred for pred in preds]
            labels = batch['label']

            preds_list.extend(final_preds)
            labels_list.extend(labels)

        logits = None
        labels = labels_list
        preds = preds_list

        acc, f1 = self.get_metric(logits, preds, labels)

        return acc, f1

    def fit(self, iter):
        output_dir = f"{self.output_dir}/{self.dataset}/{self.llm}_stage1_best.pth"
        if self.do_stage1:
            print("Preparing Stage 1 [Graph Matching] ...")
            if os.path.exists(output_dir):
                reload_pretrain_checkpoint(self.model, output_dir)
                self.s1_epoch = 0
                self.s1_lr = self.s2_lr

            params = [p for _, p in self.model.named_parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW([{'params': params, 'lr': self.s1_lr, 'weight_decay': self.wd}])

            if self.s1_epoch > 0:
                graph_type = {
                    "cora": "academic_network", 
                    "citeseer": "academic_network", 
                    "wikics": "academic_network",
                    "photo": "ecommerce_network",
                    "products": "ecommerce_network", 
                    "arxiv_23": "academic_network", 
                    "arxiv": "academic_network", 
                    
                }[self.dataset]
                pretrain_node_idx = self.task_loader.train_idx_per_task[0] + self.task_loader.valid_idx_per_task[0] + self.task_loader.test_idx_per_task_isolate[0]
                dataset = GraphMatchingDataset(graph_data=self.task_loader.dataset_per_task_isolate[0].data, node_idx=pretrain_node_idx, k_hop=self.s1_k_hop, num_sampled_neighbors=self.s1_num_neighbors, graph_type=graph_type)
                train_loader = DataLoader(dataset, batch_size=self.s1_batch_size, drop_last=True, shuffle=True)
            
                num_training_steps = self.s1_epoch * len(train_loader)
                progress_bar = tqdm(range(num_training_steps), desc='Pre-training Through Graph Matching')
            
                for epoch in range(self.s1_epoch):
                    self.model.train()
                
                    epoch_loss, accum_loss = 0.0, 0.0
                
                    for step, batch in enumerate(train_loader):
                        optimizer.zero_grad() 
                    
                        loss = self.model(batch)
                        loss.backward()
                    
                        clip_grad_norm_(optimizer.param_groups[0]['params'], 0.1)
                        optimizer.step() 
                        epoch_loss, accum_loss = epoch_loss + loss.item(), accum_loss + loss.item()
                    
                        progress_bar.update(1)
                
                    print(f"Stage 1 Training | Epoch: {epoch + 1} | Train Loss: {epoch_loss / len(train_loader):.4f}")
                    save_pretrain_checkpoint(self.model, output_dir)
            
                torch.cuda.empty_cache()
                torch.cuda.reset_max_memory_allocated()

        label_text_list = self.task_loader.text_dataset.label_texts
        for curr_session in range(self.session_num):
            if curr_session != 0:
                _reload_best_model(self.model, self.checkpoint_path, self.dataset, self.model_name, self.seed)
            else:
                if self.do_stage1 == False:
                    params = [p for _, p in self.model.named_parameters() if p.requires_grad]
                    optimizer = torch.optim.AdamW([{'params': params, 'lr': self.s2_lr, 'weight_decay': self.wd}])
                else:
                    reload_pretrain_checkpoint(self.model, output_dir)

            class_src, class_dst, text_dataset_iso, text_dataset_joint, _, _, _, _ = self.task_loader.get_task(curr_session)
            train_idx, valid_idx, test_idx_iso, test_idx_joint = self.task_loader.train_idx_per_task[curr_session], self.task_loader.valid_idx_per_task[curr_session], self.task_loader.test_idx_per_task_isolate[curr_session], self.task_loader.test_idx_per_task_joint[curr_session]

            label_text = label_text_list[ :class_dst]
            train_dataset = GraphInstructionTuningDataset(text_dataset_iso.data, label_text, train_idx, maximum_neighbors=self.s2_num_neighbors, dataset_name=self.dataset)
            val_dataset = GraphInstructionTuningDataset(text_dataset_iso.data, label_text, valid_idx, maximum_neighbors=self.s2_num_neighbors, dataset_name=self.dataset)
            test_dataset_iso = GraphInstructionTuningDataset(text_dataset_iso.data, label_text, test_idx_iso, maximum_neighbors=self.s2_num_neighbors, dataset_name=self.dataset)
            test_dataset_joint = GraphInstructionTuningDataset(text_dataset_joint.data, label_text, test_idx_joint, maximum_neighbors=self.s2_num_neighbors, dataset_name=self.dataset)
            
            train_loader = DataLoader(train_dataset, batch_size=self.s2_batch_size, drop_last=False, pin_memory=True, shuffle=True)
            valid_loader = DataLoader(val_dataset, batch_size=self.s2_batch_size, drop_last=False, pin_memory=True, shuffle=False)
            test_loader_iso = DataLoader(test_dataset_iso, batch_size=self.s2_batch_size, drop_last=False, pin_memory=True, shuffle=False)
            test_loader_joint = DataLoader(test_dataset_joint, batch_size=self.s2_batch_size, drop_last=False, pin_memory=True, shuffle=False)

            progress_bar = tqdm(range(self.config['s2_epoch']))
            progress_bar.set_description(f'Training | Iter {iter} | Session {curr_session}')

            tolerate, best_acc_valid = 0, 0
            for epoch in range(self.config['s2_epoch']):
                loss = self.train(curr_session, epoch, self.model, text_dataset_iso, train_loader, optimizer, class_src, class_dst, self.config, self.device)
                progress_bar.write("Session: {} | Epoch: {} | Loss: {:.4f}".format(curr_session, epoch, loss))

                if epoch > 0 and epoch % self.config['valid_epoch'] == 0:
                    acc_valid, f1_valid = self.valid(self.model, text_dataset_iso, valid_loader, class_src, class_dst, self.config, self.device)
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
            curr_acc_test_isolate, curr_f1_test_isolate = self.evaluate(self.model, text_dataset_iso, test_loader_iso, class_dst, self.config, self.device)
            curr_acc_test_joint, curr_f1_test_joint = self.evaluate(self.model, text_dataset_joint, test_loader_joint, class_dst, self.config, self.device)

            acc_list = []
            for s in range(curr_session):
                _, _, pre_text_dataset_iso, _, _, _, _, _ = self.task_loader.get_task(s)
                pre_test_idx_iso = self.task_loader.test_idx_per_task_isolate[s]
                pre_test_dataset_iso = GraphInstructionTuningDataset(pre_text_dataset_iso.data, label_text, pre_test_idx_iso, maximum_neighbors=self.s2_num_neighbors, dataset_name=self.dataset)
                pre_test_loader_iso = DataLoader(pre_test_dataset_iso, batch_size=self.s2_batch_size, drop_last=False, pin_memory=True, shuffle=False)
                prev_acc_test_isolate, prev_f1_test_isolate = self.evaluate(self.model, pre_text_dataset_iso, pre_test_loader_iso, class_dst, self.config, self.device)
                acc_list.append(prev_acc_test_isolate)
            acc_list.append(curr_acc_test_isolate)

            print("Session: {} | Iso. Acc Test: {:.4f} | Iso. F1 Test: {:.4f}".format(curr_session, curr_acc_test_isolate, curr_f1_test_isolate))
            print("Session: {} | Jot. Acc Test: {:.4f} | Jot. F1 Test: {:.4f}".format(curr_session, curr_acc_test_joint, curr_f1_test_joint))

            self.result_logger.add_new_results(acc_list, curr_acc_test_joint)

            torch.cuda.empty_cache()
            
        return self.result_logger

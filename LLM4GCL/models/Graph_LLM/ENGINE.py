import os
import math
import torch
import torch.nn as nn

from LLM4GCL.models import BaseModel
from LLM4GCL.backbones import GCNNet, GATNet, SAGENet, SGCNet
from LLM4GCL.utils import _save_checkpoint, _reload_best_model

from tqdm import tqdm
from torch_geometric.utils import k_hop_subgraph
from transformers.models.auto import AutoModel, AutoTokenizer

def generate_hidden_embeds(dataset, model_name, model_path, cache_path, text, batch_size, max_length, device):
    assert model_name in ['RoBERTa']

    if model_name == 'RoBERTa':
        tokenizer = AutoTokenizer.from_pretrained("roberta-large")
        model = AutoModel.from_pretrained("roberta-large", output_hidden_states=True, return_dict=True, cache_dir=model_path).cuda()
        hidden_layers = 24
    else:
        raise ValueError(f'Unsupported model {model_name}!')
    
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    batch_size = batch_size
    model.eval()
    layers = [[] for i in range(hidden_layers + 1)]
    for i in tqdm(range(math.ceil(len(text) / batch_size)), desc='Generating embeds'):
        if (i + 1) * batch_size <= len(text):
            txt = text[(i) * batch_size: (i + 1) * batch_size]
        else:
            txt = text[(i) * batch_size:]
        
        model_input = tokenizer(txt, truncation=True, padding=True, return_tensors="pt", max_length=max_length).to(device)
        with torch.no_grad():
            out = model(**model_input)
        batch_size = model_input['input_ids'].shape[0]
        hidden_states = out['hidden_states']
        
        for i, layer_hid in enumerate(hidden_states):
            layer_hid = layer_hid.cpu()
            layer_node_hid = mean_pooling(layer_hid, model_input['attention_mask'].cpu())
            layers[i].append(layer_node_hid.cpu())
            
    layers_hid = [torch.cat(xs).float() for xs in layers]

    if not os.path.exists(cache_path):
        os.makedirs(cache_path)
    torch.save(layers_hid, f=os.path.join(cache_path, 'layer_attr_' + dataset + '.pt'))


def get_hidden_embeds(dataset, cache_path, layer_select, generate_func=None, func_params=None):

    def try_load():
        if dataset in ['products', 'photo', 'arxiv']:
            xs = []
            for i in [0, 5, 10, 15, 20, 24]:
                file_path = os.path.join(cache_path, f"{i}_layer_attr_{dataset}.pt")
                if not os.path.exists(file_path):
                    return None
                xs.append(torch.load(file_path))
            return xs
        else:
            file_path = os.path.join(cache_path, f"layer_attr_{dataset}.pt")
            if not os.path.exists(file_path):
                return None
            xs = torch.load(file_path)
            return [xs[i] for i in layer_select]
    
    result = try_load()
    if result is not None:
        return result

    if generate_func is not None:
        print(f"Embedding files not found, generating...")
        generate_func(**func_params)

        result = try_load()
        if result is not None:
            return result

    raise FileNotFoundError(
        f"Failed to load embeddings for dataset {dataset} from {cache_path}\n"
        f"Expected files: {[f'{i}_layer_attr_{dataset}.pt' for i in [0,5,10,15,20,24]] if dataset in ['products','photo','arxiv'] else f'layer_attr_{dataset}.pt'}"
    )


def custom_global_mean_pool(last, edge_index, mapping):
    neighbors = []
    for node in mapping:
        src_neighbors = edge_index[1][edge_index[0] == node]
        dst_neighbors = edge_index[0][edge_index[1] == node]
        neighbors.append(torch.cat([src_neighbors, dst_neighbors], dim=0))

    pooled_features = []
    for node, neigh in zip(mapping, neighbors):
        node_features = last[node]
        neighbor_features = last[neigh]
        if neighbor_features.shape[0] == 0:
            combined_features = node_features
        else:
            combined_features = torch.cat([node_features.unsqueeze(0), neighbor_features], dim=0)
            combined_features = combined_features.mean(dim=0)
        pooled_features.append(combined_features)

    return torch.stack(pooled_features)


class ENGINEComponent(nn.Module):

    def __init__(self, lm_hidden_dim, num_class, config, device):
        super(ENGINEComponent, self).__init__()

        self.num_class = num_class
        self.device = device
        self.layer_select = config['engine_layer_select']
        self.T = config['engine_T']
        self.r = config['engine_r']

        self.k = int(lm_hidden_dim / self.r)
        self.proj_input_dim = lm_hidden_dim
        self.proj_hidden_dim = self.k
        self.proj_output_dim = self.k

        self.gnn_type = config['gnn']
        self.gnn_input_dim = self.k
        self.gnn_hidden_dim = config['hidden_dim']
        self.gnn_output_dim = self.k
        self.gnn_layer_num = config['layer_num']
        self.gnn_dropout = config['dropout']
        self.gnn_num_heads = config['num_heads']
        self.gnn_aggr = config['aggr']

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

        self.encoder_list = nn.ModuleList([
            GNNModel(gnn_type=self.gnn_type, 
                     input_dim=self.gnn_input_dim, 
                     hidden_dim=self.gnn_hidden_dim, 
                     output_dim=self.gnn_output_dim, 
                     layer_num=self.gnn_layer_num, 
                     dropout=self.gnn_dropout, 
                     num_heads=self.gnn_num_heads, 
                     aggr=self.gnn_aggr, 
                     device=self.device)
            for _ in self.layer_select
        ])
        self.proj_list = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.proj_input_dim, self.proj_hidden_dim),
                nn.LayerNorm(self.proj_hidden_dim),
                nn.ReLU(),
                nn.Linear(self.proj_hidden_dim, self.proj_output_dim)
            ) for _ in self.layer_select
        ]).to(self.device)
        self.alpha_list = nn.ParameterList([
            nn.Parameter(torch.tensor(0.0)) for _ in self.layer_select
        ]).to(self.device)

        self.classifier = nn.Linear(self.gnn_output_dim * 2, self.num_class).to(self.device)

    def forward(self, hidden_embeds, batch, data):
        
        for i, (embeds, encoder, projector, alpha) in enumerate(zip(hidden_embeds, self.encoder_list, self.proj_list, self.alpha_list)):
            subset, edge_index, mapping, _ = k_hop_subgraph(batch['node_id'], self.gnn_layer_num, data.edge_index, relabel_nodes=True)
  
            if i == 0:
                out = encoder(projector((embeds[subset]).to(self.device)), edge_index.to(self.device))
            else:
                a = torch.nn.functional.sigmoid(alpha / self.T)
                x = projector((embeds[subset]).to(self.device)) * a + last * (1 - a)
                out = encoder(x, edge_index.to(self.device))
            last = out

        out = torch.cat([last[mapping], custom_global_mean_pool(last, edge_index, mapping)], dim=1)
        logits = self.classifier(out)

        return logits

        
class ENGINE(BaseModel):
        
    def __init__(self, task_loader, result_logger, config, checkpoint_path, dataset, model_name, model_path, seed, device):
        super(ENGINE, self).__init__(task_loader, result_logger, config, checkpoint_path, dataset, model_name, seed, device)

        self.lm_type = config['lm']
        if self.lm_type == 'RoBERTa':
            self.lm_hidden_dim = 1024
        self.lr = float(config['lr'])
        self.weight_decay = float(config['weight_decay'])
        self.model = ENGINEComponent(self.lm_hidden_dim, self.num_class, config, device)
        self.model_path = model_path
        self.batch_size = config['batch_size']
        self.max_length = config['max_length']

        self.cache_path = config['cache']
        self.cache_batch_size = config['cache_batch_size']
        self.layer_select = config['engine_layer_select']

    def get_optimizer(self, model):
        optimizer = torch.optim.Adam([
            {'params': model.encoder_list.parameters()},
            {'params': model.proj_list.parameters()},
            {'params': model.alpha_list.parameters()},
            {'params': model.classifier.parameters()}
        ], lr=self.lr, weight_decay=self.weight_decay)

        return optimizer

    def train(self, curr_session, curr_epoch, model, text_dataset, train_loader, hidden_embeds, optimizer, class_num, config, device):
        all_loss, train_num = 0., 0
        for step, batch in enumerate(train_loader):
            if batch['node_id'].size(0) < 2:
                break
            optimizer.zero_grad()
            logits = model(hidden_embeds, batch, text_dataset.data)
            labels = batch['labels'].to(self.device)

            n_per_cls = [(labels == j).sum() for j in range(self.num_class)]
            loss_w = [1. / max(i, 1) for i in n_per_cls]
            loss_w = torch.tensor(loss_w[:class_num]).to(self.device)

            loss = self.loss_func(logits[:, : class_num], labels, loss_w)
            loss.backward()
            optimizer.step()
            all_loss += loss * batch['node_id'].size(0)
            train_num += batch['node_id'].size(0)

        return all_loss / train_num

    @torch.no_grad()
    def valid(self, model, text_dataset, valid_loader, hidden_embeds, class_num, config, device):
        return self.evaluate(model, text_dataset, valid_loader, hidden_embeds, class_num, config, device)

    @torch.no_grad()
    def evaluate(self, model, text_dataset, test_loader, hidden_embeds, class_num, config, device):
        model.eval()
        logits_list, preds_list, labels_list = [], [], []
        for _, batch in enumerate(test_loader):
            if batch['node_id'].size(0) < 2:
                break
            logits = model(hidden_embeds, batch, text_dataset.data)
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
        func_dict = {
            'dataset': self.dataset, 
            'model_name': self.lm_type, 
            'model_path': self.model_path, 
            'cache_path': self.cache_path, 
            'text': self.task_loader.text_dataset.raw_texts, 
            'batch_size': self.cache_batch_size, 
            'max_length': self.max_length, 
            'device': self.device
        }
        hidden_embeds = get_hidden_embeds(self.dataset, self.cache_path, self.layer_select, generate_hidden_embeds, func_dict)

        for curr_session in range(self.session_num):
            if curr_session != 0:
                _reload_best_model(self.model, self.checkpoint_path, self.dataset, self.model_name, self.seed)

            class_num, text_dataset_iso, text_dataset_joint, train_loader, valid_loader, test_loader_isolate, test_loader_joint = self.task_loader.get_task(curr_session)

            progress_bar = tqdm(range(self.config['epochs']))
            progress_bar.set_description(f'Training | Iter {iter} | Session {curr_session}')

            tolerate, best_acc_valid = 0, 0.
            for epoch in range(self.config['epochs']):
                loss = self.train(curr_session, epoch, self.model, text_dataset_iso, train_loader, hidden_embeds, optimizer, class_num, self.config, self.device)
                progress_bar.write("Session: {} | Epoch: {} | Loss: {:.4f}".format(curr_session, epoch, loss))

                if epoch > 0 and epoch % self.config['valid_epoch'] == 0:
                    acc_valid, f1_valid = self.valid(self.model, text_dataset_iso, valid_loader, hidden_embeds, class_num, self.config, self.device)
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
            curr_acc_test_isolate, curr_f1_test_isolate = self.evaluate(self.model, text_dataset_iso, test_loader_isolate, hidden_embeds, class_num, self.config, self.device)
            curr_acc_test_joint, curr_f1_test_joint = self.evaluate(self.model, text_dataset_joint, test_loader_joint, hidden_embeds, class_num, self.config, self.device)

            acc_list = []
            for s in range(curr_session):
                _, text_dataset_iso, _, _, _, test_loader_isolate, _ = self.task_loader.get_task(s)
                prev_acc_test_isolate, prev_f1_test_isolate = self.evaluate(self.model, text_dataset_iso, test_loader_isolate, hidden_embeds, class_num, self.config, self.device)
                acc_list.append(prev_acc_test_isolate)
            acc_list.append(curr_acc_test_isolate)

            print("Session: {} | Iso. Acc Test: {:.4f} | Iso. F1 Test: {:.4f}".format(curr_session, curr_acc_test_isolate, curr_f1_test_isolate))
            print("Session: {} | Jot. Acc Test: {:.4f} | Jot. F1 Test: {:.4f}".format(curr_session, curr_acc_test_joint, curr_f1_test_joint))

            self.result_logger.add_new_results(acc_list, curr_acc_test_joint)

        return self.result_logger
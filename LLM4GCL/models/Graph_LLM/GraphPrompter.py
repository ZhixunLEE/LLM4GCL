import torch
import torch.nn as nn

from LLM4GCL.models import BaseModel
from LLM4GCL.backbones import GCNNet, GATNet, SAGENet, SGCNet, RoBERTaNet
from LLM4GCL.utils import adjust_learning_rate

from torch.nn.utils import clip_grad_norm_
from torch_geometric.utils import k_hop_subgraph


class GraphPrompterComponent(nn.Module):

    def __init__(self, feat_dim, num_class, config, model_path, device):
        super(GraphPrompterComponent, self).__init__()
        # LM Params
        self.lm_type = config['lm']
        if self.lm_type == 'RoBERTa':
            self.lm_hidden_dim = 1024
        self.output_dim = num_class
        self.lora_config = config['LoRA']
        self.lm_dropout = config['dropout']
        self.att_dropout = config['att_dropout']
        self.max_length = config['max_length']
        self.model_path = model_path
        self.device = device

        class LMModel(nn.Module):

            def __init__(self, lm_type, max_length, model_path, hidden_dim, output_dim, lora_config, dropout, att_dropout, device):
                super(LMModel, self).__init__()
                self.device = device
                self.max_length = max_length
                
                if lm_type == 'RoBERTa':
                    self.lm = RoBERTaNet(output_dim, model_path, lora_config, dropout, att_dropout).to(device)

            def forward(self, inputs_embeds, attention_mask):
                _, hidden_embeds = self.lm(inputs_embeds, attention_mask)

                return hidden_embeds[-1][:, 0, :]
            
        self.lm_model = LMModel(self.lm_type, self.max_length, self.model_path, self.lm_hidden_dim, self.output_dim, self.lora_config, self.lm_dropout, self.att_dropout, self.device)
        self.tokenizer = self.lm_model.lm.tokenizer
        self.embeddings = self.lm_model.lm.embeddings

        # GNN Params
        self.gnn_type = config['GNN']['gnn']
        self.gnn_input_dim = feat_dim
        self.gnn_hidden_dim = config['GNN']['hidden_dim']
        self.gnn_output_dim = config['GNN']['output_dim']
        self.gnn_layer_num = config['GNN']['layer_num']
        self.gnn_dropout = config['GNN']['dropout']
        self.gnn_num_heads = config['GNN']['num_heads']
        self.gnn_aggr = config['GNN']['aggr']

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
        self.proj_hidden_dim = config['Projector']['hidden_dim']
        self.projector = nn.Sequential(
            nn.Linear(self.gnn_output_dim, self.proj_hidden_dim),
            nn.Sigmoid(),
            nn.Linear(self.proj_hidden_dim, self.lm_hidden_dim),
        ).to(self.device)

        self.fc = nn.Sequential(
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
            # nn.Dropout(p=0.1), 
            nn.Linear(self.lm_hidden_dim, self.output_dim),
        ).to(self.device)

    def forward(self, samples, data):

        def encode_graphs(feats, edge_index, mapping):
            embeds = self.gnn_encoder(feats.to(self.device), edge_index.to(self.device))
            inputs_embeds = self.projector(embeds[mapping])

            return inputs_embeds

        subset, edge_index, mapping, _ = k_hop_subgraph(samples['node_id'], self.gnn_layer_num, data.edge_index, relabel_nodes=True)
        graph_embeds = encode_graphs(data.x[subset], edge_index, mapping)

        batch_size = len(samples['node_id'])
        batch_inputs_embeds = []
        batch_attention_mask = []
        if self.lm_type == 'RoBERTa':
            for i in range(batch_size):
                inputs = self.tokenizer(samples['raw_text'], padding=True, truncation=True, max_length=self.max_length, return_tensors='pt')
                inputs_embeds = self.embeddings(inputs['input_ids'][i][:self.max_length - 1].to(self.device))
                inputs_embeds = torch.cat([inputs_embeds[:1], graph_embeds[i].unsqueeze(0), inputs_embeds[1:]], dim=0)

                batch_inputs_embeds.append(inputs_embeds)
                batch_attention_mask.append(torch.cat([torch.tensor([1]).to(self.device), inputs['attention_mask'][i][:self.max_length - 1].to(self.device)]))

        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.device)
        attention_mask = torch.stack(batch_attention_mask, dim=0).to(self.device)

        outputs = self.lm_model(inputs_embeds, attention_mask)
        logits = self.fc(outputs)

        return logits
    

class GraphPrompter(BaseModel):
        
    def __init__(self, task_loader, result_logger, config, checkpoint_path, dataset, model_name, model_path, seed, device):
        super(GraphPrompter, self).__init__(task_loader, result_logger, config, checkpoint_path, dataset, model_name, seed, device)

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


    def train(self, curr_session, curr_epoch, model, text_dataset, train_loader, optimizer, class_num, config, device):
        model.train()
        accum_loss = 0.
        all_loss, train_num = 0., 0
        for step, batch in enumerate(train_loader):
            if batch['node_id'].size(0) < 2:
                break
            optimizer.zero_grad()
            logits = model(batch, text_dataset.data)
            loss = self.loss_func(logits[:, : class_num], batch['labels'].cuda())
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
            logits = model(batch, text_dataset.data)
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
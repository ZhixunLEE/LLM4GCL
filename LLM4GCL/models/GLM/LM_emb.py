import os
import math
import torch
import torch.nn as nn

from LLM4GCL.models import BareGNN
from LLM4GCL.backbones import GCNNet, GATNet, SAGENet, SGCNet
from LLM4GCL.utils import _save_checkpoint, _reload_best_model

from tqdm import tqdm
from transformers.models.auto import AutoModel, AutoTokenizer

@torch.no_grad()
def generate_hidden_embeds(dataset, model_name, model_path, cache_path, text, batch_size, max_length, device):
    assert model_name in ['RoBERTa', 'LLaMA']

    if model_name == 'RoBERTa':
        tokenizer = AutoTokenizer.from_pretrained("roberta-large")
        model = AutoModel.from_pretrained("roberta-large", output_hidden_states=True, return_dict=True, cache_dir=model_path).to(device)
    elif model_name == 'LLaMA':
        model_name_str = 'Llama-3.1-8B'
        model_path = os.path.join(model_path, 'models--' + model_name_str.lower())
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path, output_hidden_states=True, return_dict=True, load_in_4bit=True).to(device)
        tokenizer.pad_token = tokenizer.eos_token
    else:
        raise ValueError(f'Unsupported model {model_name}!')

    batch_size = batch_size
    model.eval()
    hidden_embdes = []
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
        if model_name == 'RoBERTa':
            hidden_states = hidden_states[-1][:, 0, :].detach().cpu()
        elif model_name == 'LLaMA':
            hidden_states = hidden_states[-1][:, -1, :].float().detach().cpu()
        hidden_embdes.extend(hidden_states)

    hidden_embdes = torch.stack(hidden_embdes, dim=0)

    if not os.path.exists(cache_path):
        os.makedirs(cache_path)
    torch.save(hidden_embdes, f=os.path.join(cache_path, model_name + '_embeds_' + dataset + '.pt'))


def get_hidden_embeds(model_name, dataset, cache_path, generate_func=None, func_params=None):

    def try_load():
        file_path = os.path.join(cache_path, f"{model_name}_embeds_{dataset}.pt")
        if not os.path.exists(file_path):
            return None
        hidden_embeds = torch.load(file_path)
        return hidden_embeds
    
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
        f"Expected files: {f'{model_name}_embeds_{dataset}.pt'}"
    )


class LM_emb(BareGNN):

    def __init__(self, task_loader, result_logger, config, checkpoint_path, dataset, model_name, model_path, seed, device):
        super(LM_emb, self).__init__(task_loader, result_logger, config, checkpoint_path, dataset, model_name, seed, device)

        self.lm_type = config['lm']
        if self.lm_type == 'RoBERTa':
            self.lm_hidden_dim = 1024
        elif self.lm_type == 'LLaMA':
            self.lm_hidden_dim = 4096
        self.lm_max_length = config['max_length']
        self.lm_model_path = model_path
        self.cache_path = config['cache']
        self.cache_batch_size = config['cache_batch_size']
        self.input_dim = self.lm_hidden_dim

        self.lr = float(config['lr'])
        self.weight_decay = float(config['weight_decay'])
        self.batch_size = config['batch_size']

        class GNNModel(nn.Module):

            def __init__(self, gnn_type, input_dim, hidden_dim, output_dim, layer_num, dropout, num_heads, aggr, device):
                super(GNNModel, self).__init__()
                if gnn_type == 'GCN':
                    self.gnn = GCNNet(input_dim, hidden_dim, hidden_dim, layer_num, dropout).to(device)
                elif gnn_type == 'GAT':
                    self.gnn = GATNet(input_dim, hidden_dim, hidden_dim, layer_num, dropout, num_heads).to(device)
                elif gnn_type == 'SAGE':
                    self.gnn = SAGENet(input_dim, hidden_dim, hidden_dim, layer_num, dropout, aggr).to(device)
                elif gnn_type == 'SGC':
                    self.gnn = SGCNet(input_dim, hidden_dim, hidden_dim, layer_num, dropout).to(device)

                self.fc = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(p=0.1), 
                    nn.Linear(hidden_dim, output_dim),
                ).to(device)

            def forward(self, x, edge_index):
                x = self.gnn(x, edge_index)
                logits = self.fc(x)

                return logits
            
        self.model = GNNModel(self.gnn_type, self.input_dim, self.hidden_dim, self.output_dim, self.layer_num, self.dropout, self.num_heads, self.aggr, self.device)


    def fit(self, iter):
        optimizer = self.get_optimizer(self.model)
        func_dict = {
            'dataset': self.dataset, 
            'model_name': self.lm_type, 
            'model_path': self.lm_model_path, 
            'cache_path': self.cache_path, 
            'text': self.task_loader.text_dataset.raw_texts, 
            'batch_size': self.cache_batch_size, 
            'max_length': self.lm_max_length, 
            'device': self.device
        }
        hidden_embeds = get_hidden_embeds(self.lm_type, self.dataset, self.cache_path, generate_hidden_embeds, func_dict)

        for curr_session in range(self.session_num):
            if curr_session != 0:
                _reload_best_model(self.model, self.checkpoint_path, self.dataset, self.model_name, self.seed)

            class_num, text_dataset_iso, text_dataset_joint, train_loader, valid_loader, test_loader_isolate, test_loader_joint = self.task_loader.get_task(curr_session)
            text_dataset_iso.data.x = hidden_embeds
            text_dataset_joint.data.x = hidden_embeds

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
                text_dataset_iso.data.x = hidden_embeds
                prev_acc_test_isolate, prev_f1_test_isolate = self.evaluate(self.model, text_dataset_iso, test_loader_isolate, class_num, self.config, self.device)
                acc_list.append(prev_acc_test_isolate)
            acc_list.append(curr_acc_test_isolate)

            print("Session: {} | Iso. Acc Test: {:.4f} | Iso. F1 Test: {:.4f}".format(curr_session, curr_acc_test_isolate, curr_f1_test_isolate))
            print("Session: {} | Jot. Acc Test: {:.4f} | Jot. F1 Test: {:.4f}".format(curr_session, curr_acc_test_joint, curr_f1_test_joint))

            self.result_logger.add_new_results(acc_list, curr_acc_test_joint)

        return self.result_logger
 
import os
import io
import math
import yaml
import torch
import random
import itertools
import numpy as np

from ruamel.yaml import YAML
from typing import Dict, Any
from datetime import datetime
from torch_geometric import seed_everything as pyg_seed 


def load_config(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        params = yaml.safe_load(file)

    return params


def merge_params(default: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:

    return {**default, **override}


def update_config(config_path, dataset, best_params, metrics=None):
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.indent(mapping=2, sequence=4, offset=2)

    with open(config_path, 'r', encoding='utf-8') as f:
        original_lines = f.readlines()
        f.seek(0)
        config = yaml.load(f) or {}

    best_key = f'best_{dataset}'
    config[best_key] = {
        **best_params,
        '_meta': {
            'updated_at': datetime.now().isoformat(),
            'metrics': metrics or {}
        }
    }

    output = io.StringIO()
    yaml.dump(config, output)
    new_content = output.getvalue()

    header = []
    for line in original_lines:
        if line.strip().startswith('#') or not line.strip():
            header.append(line)
        else:
            break

    with open(config_path, 'w', encoding='utf-8') as f:
        f.writelines(header)
        f.write(new_content)


def seed_everything(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    pyg_seed(seed)


def _save_checkpoint(model, optimizer, cur_epoch, checkpoint_path, dataset, model_name, seed):
    os.makedirs(checkpoint_path, exist_ok=True)

    save_dir = os.path.join(checkpoint_path, model_name)
    os.makedirs(save_dir, exist_ok=True)
    save_to = os.path.join(save_dir, f"{dataset}_seed{seed}_best.pth")

    state_dict = {
        k: v for k, v in model.state_dict().items()
        if v.requires_grad
    }

    save_obj = {
        "model": state_dict,
        # "optimizer": optimizer.state_dict(),
        # "epoch": cur_epoch,
    }

    try:
        torch.save(save_obj, save_to)
        print(f"Saving checkpoint at epoch {cur_epoch} to {save_to}.")
    except Exception as e:
        print(f"Failed to save checkpoint: {e}")


def _reload_best_model(model, checkpoint_path, dataset, model_name, seed):
    path = f'{model_name}/{dataset}_seed{seed}_best.pth'
    checkpoint_path = os.path.join(checkpoint_path, path)

    print("Loading checkpoint from {}.".format(checkpoint_path))

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=False)

    return model


def adjust_learning_rate(param_group, epoch, config):
    if epoch < config['warmup_epochs']:
        lr = float(config['lr']) * epoch / config['warmup_epochs']
    else:
        lr = float(config['min_lr']) + (float(config['lr']) - float(config['min_lr'])) * 0.5 * (1.0 + math.cos(math.pi * (epoch - config['warmup_epochs']) / (config['epochs'] - config['warmup_epochs'])))
    param_group["lr"] = lr
    return lr


def select_hyperparameters(search_space, search_type, num_samples):
    all_combinations = list(itertools.product(*search_space.values()))
    
    search_results = []

    if search_type == 'grid':
        for combination in all_combinations:
            selected_params = {key: value for key, value in zip(search_space.keys(), combination)}
            search_results.append(selected_params)
    
    elif search_type == 'random':
        selected_combinations = random.sample(all_combinations, num_samples)
        for conbination in selected_combinations:
            selected_params = {key: value for key, value in zip(search_space.keys(), conbination)}
            search_results.append(selected_params)
    
    else:
        raise ValueError("Unsupported search type.")
    
    return search_results
    

def update_args_with_params(args, params):
    for key, value in params.items():
        setattr(args, key, value)
    return args
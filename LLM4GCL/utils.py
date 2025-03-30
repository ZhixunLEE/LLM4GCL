import os
import math
import yaml
import torch
import random
import numpy as np
import itertools
from torch_geometric import seed_everything as pyg_seed 

def load_config(dataset, model, file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        params = yaml.safe_load(file)

    if dataset not in params.keys() or dataset == None:
        params = params['default']
    else:
        params = params[dataset]

    print(f"--------------------------------------------")
    print(f"Load configs of {model}!")
    print(f"Params:")
    for item in params.keys():
        print(f"{item}: {params[item]}")
    print(f"--------------------------------------------")

    return params


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
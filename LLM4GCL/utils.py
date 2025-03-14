import os
import yaml
import torch
import random
import numpy as np
from torch_geometric import seed_everything as pyg_seed 

def load_config(dataset, model, file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        params = yaml.safe_load(file)

    if dataset not in params.keys():
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

    param_grad_dic = {
        k: v.requires_grad for (k, v) in model.named_parameters()
    }
    state_dict = model.state_dict()

    for k in list(state_dict.keys()):
        if k in param_grad_dic.keys() and not param_grad_dic[k]:
            # delete parameters that do not require gradient
            del state_dict[k]

    save_obj = {
        "model": state_dict,
        # "optimizer": optimizer.state_dict(),
        # "epoch": cur_epoch,
    }

    path = f'{dataset}_{model_name}_seed{seed}'
    save_to = os.path.join(checkpoint_path, path+"_checkpoint_best.pth")

    print("Saving checkpoint at epoch {} to {}.".format(cur_epoch, save_to))
    torch.save(save_obj, save_to)


def _reload_best_model(model, checkpoint_path, dataset, model_name, seed):
    path = f'{dataset}_{model_name}_seed{seed}_checkpoint_best.pth'
    checkpoint_path = os.path.join(checkpoint_path, path)

    print("Loading checkpoint from {}.".format(checkpoint_path))

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=False)

    return model
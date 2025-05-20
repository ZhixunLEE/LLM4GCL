import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import yaml
import argparse

from LLM4GCL.experiment import Experiment
from LLM4GCL.common.utils import load_config, merge_params, update_config, select_hyperparameters


model_dict = {
    'GNN': ['BareGNN', 'EWC', 'LwF', 'cosine', 'TEEN', 'TPP'],
    'LM': ['RoBERTa', 'BERT', 'LLaMA', 'SimpleCIL'], 
    'GLM': ['LM_emb', 'GraphPrompter', 'ENGINE', 'LLaGA', 'GraphGPT', 'SimGCL'], 
}

exp_settings = {
    'NCIL': {
        'ways': {'cora': 2, 'citeseer': 2, 'wikics': 3, 'photo': 3, 'products': 4, 'arxiv_23': 4, 'arxiv': 4},
        'sessions': {'cora': 3, 'citeseer': 3, 'wikics': 3, 'photo': 4, 'products': 8, 'arxiv_23': 9, 'arxiv': 10},
        'train_shots': {'cora': 100, 'citeseer': 100, 'wikics': 200, 'photo': 400, 'products': 400, 'arxiv_23': 400, 'arxiv': 800},
        'valid_shots': {'cora': 50, 'citeseer': 50, 'wikics': 50, 'photo': 50, 'products': 50, 'arxiv_23': 50, 'arxiv': 50},
        'test_shots': {'cora': 100, 'citeseer': 100, 'wikics': 200, 'photo': 400, 'products': 400, 'arxiv_23': 400, 'arxiv': 800},
    },
    'FSNCIL': {
        'base_session': {'cora': 3, 'citeseer': 2, 'wikics': 4, 'photo': 4, 'products': 11, 'arxiv_23': 13, 'arxiv': 12},
        'novel_session': {'cora': 4, 'citeseer': 4, 'wikics': 6, 'photo': 8, 'products': 20, 'arxiv_23': 24, 'arxiv': 28},
        'ways': {'cora': 2, 'citeseer': 2, 'wikics': 3, 'photo': 4, 'products': 4, 'arxiv_23': 4, 'arxiv': 4},
        'sessions': {'cora': 3, 'citeseer': 3, 'wikics': 3, 'photo': 3, 'products': 6, 'arxiv_23': 7, 'arxiv': 8},
        'base_train_shots': {'cora': 100, 'citeseer': 100, 'wikics': 200, 'photo': 400, 'products': 400, 'arxiv_23': 400, 'arxiv': 800},
        'train_shots': {'cora': 5, 'citeseer': 5, 'wikics': 5, 'photo': 5, 'products': 5, 'arxiv_23': 5, 'arxiv': 5},
        'valid_shots': {'cora': 50, 'citeseer': 50, 'wikics': 50, 'photo': 50, 'products': 50, 'arxiv_23': 50, 'arxiv': 50},
        'test_shots': {'cora': 100, 'citeseer': 100, 'wikics': 200, 'photo': 400, 'products': 400, 'arxiv_23': 400, 'arxiv': 800},
    }
}


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Dataset
    parser.add_argument('--dataset', 
                        type=str, 
                        default='cora', 
                        choices=['cora', 'citeseer', 'wikics', 'photo', 'products', 'arxiv_23', 'arxiv'], 
                        help='the name of TAG dataset')
    parser.add_argument('--data_path', type=str, default='/YOUR_PATH/data/', help='the path of TAG dataset')

    # Model
    parser.add_argument('--model_type', 
                        type=str, 
                        default='LM', 
                        choices=['GNN', 'LM', 'GLM'], 
                        help='Specify the type of model to use. '
                            ' "GNN": Use only Graph Neural Network (GNN) for training and inference. '
                            ' "LM": Use only Language Model (LM) for training and inference. '
                            ' "GLM": Combine Graph Neural Network or Graph and Language Model (LM) into a unified model.')
    parser.add_argument('--model', type=str, default='GPT', help='the name of model, must match with the model_type')
    parser.add_argument('--model_path', type=str, default='/YOUR_PATH/model/', help='the path to load pre-trained models')
    parser.add_argument('--ckpt_path', type=str, default='/YOUR_PATH/ckpt/', help='the path to store best model weights')

    # Settings
    parser.add_argument('--cl_type', type=str, default='class', choices=['class'], help='The type of CL. E.g., class is for class incremental learning')
    parser.add_argument('--task_type', type=str, default='NCIL', choices=['NCIL', 'FSNCIL'], help='The type of continual tasks.')

    # Training
    parser.add_argument('--local_ce', default=False, action='store_true')
    parser.add_argument('--ntrail', type=int, default=1, help='repetition count of experiments')
    parser.add_argument('--gpu_num', type=int, default=0, help='the selected GPU number')

    # Tuning
    parser.add_argument('--hyperparam_search', default=False, action='store_true')
    parser.add_argument('--search_type', type=str, default='grid', choices=['grid', 'random'])
    parser.add_argument('--num_samples', type=int, default=10)

    args = parser.parse_args()

    assert args.model in model_dict[args.model_type], f"Model type '{args.model_type}' does not support model '{args.model}'."
    # assert args.split_ratio[0] + args.split_ratio[1] + args.split_ratio[2] <= 1, f"The sum of split ratio is larger than 1."
    args.config_path = './configs/{}/{}.yaml'.format(args.model_type, args.model)
    config = load_config(args.config_path)

    if args.task_type == 'FSNCIL':
        args.base_session = exp_settings[args.task_type]['base_session'][args.dataset]
        args.novel_session = exp_settings[args.task_type]['novel_session'][args.dataset]
        args.base_train_shots = exp_settings[args.task_type]['base_train_shots'][args.dataset]
    elif args.task_type == 'NCIL':
        args.base_session = None
        args.novel_session = None
        args.base_train_shots = None
    args.ways = exp_settings[args.task_type]['ways'][args.dataset]
    args.sessions = exp_settings[args.task_type]['sessions'][args.dataset]
    args.train_shots = exp_settings[args.task_type]['train_shots'][args.dataset]
    args.valid_shots = exp_settings[args.task_type]['valid_shots'][args.dataset]
    args.test_shots = exp_settings[args.task_type]['test_shots'][args.dataset]

    if args.hyperparam_search:

        if 'default' not in config or 'search_space' not in config:
            raise ValueError("Hyperparameter search requires the default and search_space sections in the config file")

        default_params = config['default']
        search_space = config['search_space']
        selected_params = select_hyperparameters(search_space, args.search_type, args.num_samples)
        
        best_performance = -float('inf')
        best_params = None
        best_metric = None
        
        for params in selected_params:

            current_params = merge_params(default_params, params)
            exp = Experiment(args, current_params)
            avg_acc_iso_mean, avg_fgt_iso_mean, avg_acc_jot_mean, last_acc_jot_mean = exp.run()

            if avg_acc_iso_mean + avg_fgt_iso_mean + avg_acc_jot_mean + last_acc_jot_mean > best_performance:
                best_performance = avg_acc_iso_mean + avg_fgt_iso_mean + avg_acc_jot_mean + last_acc_jot_mean
                best_params = params
                best_metric = {
                    'Iso. Avg ACC': "{:.4f}".format(avg_acc_iso_mean),
                    'Iso. Avg FGT': "{:.4f}".format(avg_fgt_iso_mean),
                    'Jot. Avg ACC': "{:.4f}".format(avg_acc_jot_mean),
                    'Jot. Last ACC': "{:.4f}".format(last_acc_jot_mean)
                }
        
        if best_params is not None and best_metric is not None:
            update_config(args.config_path, args.dataset, best_params, best_metric)
                
    else:
        if 'default' not in config:
            raise ValueError("The config file must contain the default section")

        final_params = config['default'].copy()
        if 'best_' + args.dataset in config and config['best_' + args.dataset]:
            final_params = merge_params(final_params, config['best_' + args.dataset])

        exp = Experiment(args, final_params)
        exp.run()

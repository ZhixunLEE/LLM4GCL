import yaml
import argparse

from LLM4GCL.experiment import Experiment
from LLM4GCL.utils import load_config, merge_params, update_config, select_hyperparameters


model_dict = {
    'GNN': ['BareGNN', 'JointGNN', 'EWC', 'MAS', 'GEM', 'LwF', 'cosine', 'ERGNN', 'SSM', 'CaT', 'DeLoMe', 'TPP'],
    'LM': ['BareLM', 'SimpleCIL', 'OLoRA'], 
    'Graph_LM': ['LM_emb', 'GraphPrompter', 'ENGINE'], 
}

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Dataset
    parser.add_argument('--dataset', 
                        type=str, 
                        default='cora', 
                        choices=['cora', 'citeseer', 'wikics', 'photo', 'children', 'products', 'arxiv_23', 'arxiv'], 
                        help='the name of TAG dataset')
    parser.add_argument('--data_path', type=str, default='/root/autodl-tmp/data/', help='the path of TAG dataset')

    # Model
    parser.add_argument('--model_type', 
                        type=str, 
                        default='Graph_LM', 
                        choices=['GNN', 'LM', 'Graph_LM'], 
                        help='Specify the type of model to use. '
                            ' "GNN": Use only Graph Neural Network (GNN) for training and inference. '
                            ' "LM": Use only Language Model (LM) for training and inference. '
                            ' "Graph_LM": Combine Graph Neural Network or Graph and Language Model (LM) into a unified model.')
    parser.add_argument('--model', type=str, default='ENGINE', help='the name of model, must match with the model_type')
    parser.add_argument('--model_path', type=str, default='/root/autodl-tmp/model/', help='the path to load pre-trained models')
    parser.add_argument('--checkpoint_path', type=str, default='/root/autodl-tmp/checkpoint/', help='the path to store best model weights')

    # Settings
    parser.add_argument('--cl_type', type=str, default='class', choices=['class'], help='The type of CL. E.g., class is for class incremental learning')
    parser.add_argument('--task_type', type=str, default='normal', choices=['normal'], help='The type of continual tasks.')
    parser.add_argument('--session_size', type=int, default=2, help='The number of classes in each CL session')
    parser.add_argument('--split_ratio', type=list, default=[0.6, 0.2, 0.2], help='The ratio to split data into train/valid/test datasets.')

    # Training
    parser.add_argument('--ntrail', type=int, default=1, help='repetition count of experiments')
    parser.add_argument('--gpu_num', type=int, default=0, help='the selected GPU number')

    # Tuning
    parser.add_argument('--hyperparam_search', default=False, action='store_true')
    parser.add_argument('--search_type', type=str, default='grid', choices=['grid', 'random'])
    parser.add_argument('--num_samples', type=int, default=10)

    args = parser.parse_args()

    assert args.model in model_dict[args.model_type], f"Model type '{args.model_type}' does not support model '{args.model}'."
    assert args.split_ratio[0] + args.split_ratio[1] + args.split_ratio[2] <= 1, f"The sum of split ratio is larger than 1."
    args.config_path = './configs/{}/{}.yaml'.format(args.model_type, args.model)
    config = load_config(args.config_path)

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
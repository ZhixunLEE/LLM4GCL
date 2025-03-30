import argparse

from LLM4GCL.experiment import Experiment

model_dict = {
    'GNN': ['BareGNN', 'JointGNN', 'EWC', 'MAS', 'GEM', 'LwF', 'cosine', 'ERGNN', 'SSM', 'CaT', 'DeLoMe', 'TPP'],
    'LM': ['BareLM'], 
    'LM_emb': [],
    'Graph_LM': [], 
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
                        default='GNN', 
                        choices=['GNN', 'LM', 'LM_emb', 'Graph_LM'], 
                        help='Specify the type of model to use. '
                            ' "GNN": Use only Graph Neural Network (GNN) for training and inference. '
                            ' "LM": Use only Language Model (LM) for training and inference. '
                            ' "LM_emb": Use embeddings from a pre-trained Language Model (LM) to train the GNN. '
                            ' "Graph_LM": Combine Graph Neural Network or Graph and Language Model (LM) into a unified model.')
    parser.add_argument('--model', type=str, default='BareGNN', help='the name of model, must match with the model_type')
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
    parser.add_argument('--hyperparameter_search', action='store_true')
    parser.add_argument('--search_type', type=str, default='grid', choices=['grid', 'random'])
    parser.add_argument('--num_samples', type=int, default=10)

    args = parser.parse_args()


    assert args.model in model_dict[args.model_type], f"Model type '{args.model_type}' does not support model '{args.model}'."
    assert args.split_ratio[0] + args.split_ratio[1] + args.split_ratio[2] <= 1, f"The sum of split ratio is larger than 1."

    args.config_path = './LLM4GCL/configs/{}/{}.yaml'.format(args.model_type, args.model) # You should treat LLM4GCL as a library
    args.search_space_path = './search_space/{}/{}.yaml'.format(args.model_type, args.model)

    exp = Experiment(args)
    exp.run()

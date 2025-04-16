import os
import json
import torch
import numpy as np

from textwrap import shorten

def get_label_text(dataset):
    label_text_list = None
    with open('LLM4GCL/common/label_text.json', 'r', encoding='utf-8') as f:
        label_text_list = json.load(f)[dataset]

    return label_text_list


# Genreal Prompts for Inference
def get_genreal_prompts(dataset, label_text):
    prompts_dict = {
        'cora': f"Question: Which of the following sub-categories of AI does this paper belong to? Here are the {len(label_text)} categories: {', '.join([label.lower() for label in label_text])}. Reply only one category that you think this paper might belong to. Only reply the category phrase without any other explanation words.\n\nAnswer: ",
        'citeseer': f"Question: Which of the following theme does this paper belong to? Here are the {len(label_text)} categories: {', '.join([label.lower() for label in label_text])}. Reply only one category that you think this paper might belong to. Only reply the category full name I give you without any other words.\n\nAnswer: ",
        'wikics': f"Question: Which of the following branch of Computer science does this Wikipedia-based dataset belong to? Here are the {len(label_text)} categories: {', '.join([label.lower() for label in label_text])}. Reply only one category that you think this paper might belong to. Only reply the category full name without any other words.\n\nAnswer: ",
        'photo': f"Question: Which of the following categories does this photo item belong to? Here are the {len(label_text)} categories: {', '.join([label.lower() for label in label_text])}. Reply only one category that you think this item might belong to. Only reply the category name I give of the category without any other words.\n\nAnswer: ",
        'products': f"Question: Which of the following categories of Amazon products does this item belong to? Here are the {len(label_text)} categories: {', '.join([label.lower() for label in label_text])}. Reply only one category that you think this item might belong to. Only reply the category name without any other explanation words.\n\nAnswer: ",
        'arxiv_23': f"Question: Which of the following arXiv CS sub-categories does this dataset belong to? Here are the {len(label_text)} categories: {', '.join([label.lower() for label in label_text])}. Reply only one category that you think this paper might belong to. Only reply the category name I given without any other words, please don't use your own words.\n\nAnswer:",
        'arxiv': f"Question: Which of the following arXiv CS sub-categories does this dataset belong to? Here are the {len(label_text)} categories: {', '.join([label.lower() for label in label_text])}. Reply only one category that you think this paper might belong to. Only reply the category phrase without any other explanation words.\n\nAnswer: "
    }

    return prompts_dict[dataset]


# Genreal Prompts for Instruction LM
def get_instruction_prompts(node_index, graph_data, text, label_index, class_num, dataset, hop=[20, 20], mode="ego", include_label=False, max_node_text_len=256):

    def get_subgraph(node_idx, edge_index, hop):
        current_nodes = torch.tensor([node_idx])
        all_hop_neighbors = []

        for _ in range(len(hop)):
            mask = torch.isin(edge_index[0], current_nodes) | torch.isin(edge_index[1], current_nodes)
            new_nodes = torch.unique(torch.cat((edge_index[0][mask], edge_index[1][mask])))
            diff_nodes_set = set(new_nodes.numpy()) - set(current_nodes.numpy())
            diff_nodes = torch.tensor(list(diff_nodes_set))  
            all_hop_neighbors.append(diff_nodes.tolist())
            current_nodes = torch.unique(torch.cat((current_nodes, new_nodes)))

        return all_hop_neighbors
    
    def get_structure_prompts(node_index, label_idx, label_text, graph_data, text, all_hop_neighbors, hop, include_label, dataset, mode, max_node_text_len=256):

        prompt_str = ""
        if dataset == 'products':
            node_id_token = f'Product id: '
        elif dataset == 'photo':
            node_id_token = f'Photo id: '
        elif dataset == 'wikics':
            node_id_token = f'Page id: '
        else:
            node_id_token = f'Paper id: '

        for h in range(0, len(hop)):
            neighbors_at_hop = all_hop_neighbors[h]
            neighbors_at_hop = np.unique(np.array(neighbors_at_hop))
            np.random.shuffle(neighbors_at_hop)

            if h == 0:
                neighbors_at_hop = neighbors_at_hop[:hop[0]]
            else:
                neighbors_at_hop = neighbors_at_hop[:hop[1]]

            if len(neighbors_at_hop) > 0:
                if dataset == 'product':
                    prompt_str += f"\nKnown neighbor products at hop {h + 1} (partial, may be incomplete):\n"
                elif dataset == 'photo':
                    prompt_str += f"\nKnown neighbor photos at hop {h + 1} (partial, may be incomplete):\n"
                elif dataset == 'wikics':
                    prompt_str += f"\nKnown neighbor pages at hop {h + 1} (partial, may be incomplete):\n"
                else:
                    prompt_str += f"\nKnown neighbor papers at hop {h + 1} (partial, may be incomplete):\n"

                for neighbor_idx in neighbors_at_hop:
                    neighbor_text = shorten(text[neighbor_idx], width=max_node_text_len, placeholder="...")
                    if mode != 'pure':
                        prompt_str += f"\n{node_id_token}{neighbor_idx}\nText: {neighbor_text}\n"
                    else:
                        prompt_str += f"\n{node_id_token}{neighbor_idx}"

                    if include_label and neighbor_idx in label_idx: # Notice label leakage !!!
                        label = label_text[graph_data.y[neighbor_idx].item()]
                        prompt_str += f"Label: {label}\n"

        return prompt_str

    label_text_list = get_label_text(dataset)
    label_text = label_text_list[ :class_num]

    prefix_prompts_dict = {
        'neighbors': f'You are a good graph reasoner. Given a graph description from {dataset} dataset, understand the structure and answer the question.\n',
        'ego': f'You are a good graph reasoner. Given target node information from {dataset} dataset, answer the question.\n',
        'pure': f'You are a good graph reasoner. Given a graph description from {dataset} dataset, understand the structure and answer the question.\n',
    }

    question_prompts_dict = {
        'cora': f"Please predict which of the following sub-categories of AI does this paper belong to. Choose from the following {len(label_text)} categories: {', '.join([label.lower() for label in label_text])}",
        'citeseer': f"Please predict which of the following theme does this paper belong to. Choose from the following {len(label_text)} categories: {', '.join([label.lower() for label in label_text])}",
        'wikics': f"Please predict which branch of Computer Science this Wikipedia-based dataset belongs to. Choose from the following {len(label_text)} categories: {', '.join([label.lower() for label in label_text])}",
        'photo': f"Please predict which of the following categories does this photo item belong to. Choose from the following {len(label_text)} categories: {', '.join([label.lower() for label in label_text])}",
        'products': f"Please predict which of the following categories does this target item from Amazon belong to. Choose from the following {len(label_text)} categories: {', '.join([label.lower() for label in label_text])}",
        'arxiv_23': f"Please predict the most appropriate original arxiv identifier for the paper. Choose from the following {len(label_text)} categories: {', '.join([label.lower() for label in label_text])}.",
        'arxiv': f"Please predict the most appropriate original arxiv identifier for the paper. Choose from the following {len(label_text)} categories: {', '.join([label.lower() for label in label_text])}.",
    }


    instruction_list = []
    for id in node_index:

        if dataset == 'products':
            node_id = f'Product id: {id}\n'
        elif dataset == 'photo':
            node_id = f'Photo id: {id}\n'
        elif dataset == 'wikics':
            node_id = f'Page id: {id}\n'
        else:
            node_id = f'Paper id: {id}\n'

        prefix_prompt = prefix_prompts_dict[mode]
        context = prefix_prompt + '\n## Target node:\n' + node_id
        question = question_prompts_dict[dataset] + '\nDo not provide your reasoning.\n Answer:\n\n'
        answer = label_text[graph_data.y[id].item()]

        if mode == 'neighbors':
            raw_text = shorten(text[id], width=max_node_text_len, placeholder="...")
            context = f"{context}Text: {raw_text}\n"

            all_hop_neighbors = get_subgraph(id, graph_data.edge_index, hop)
            prompt_str = get_structure_prompts(id, label_index, label_text, graph_data, text, all_hop_neighbors, hop, include_label, dataset, mode, max_node_text_len)
            context += prompt_str

        elif mode == 'ego':
            raw_text = shorten(text[id], width=max_node_text_len, placeholder="...")
            context = f"{context}Text: {raw_text}\n"

        elif mode == 'pure':
            all_hop_neighbors = get_subgraph(id, graph_data.edge_index, hop)
            prompt_str = get_structure_prompts(id, label_index, label_text, text, graph_data, all_hop_neighbors, hop, include_label, dataset, mode, max_node_text_len)
            context += prompt_str

        else:
            raise ValueError('Invalid mode!')

        instruction_list.append({
            "Context": context,
            "Question": question,
            "Answer": answer
        })

    return instruction_list

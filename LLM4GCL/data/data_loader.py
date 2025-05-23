import copy
import json
import heapq
import torch

from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops
from torch.utils.data import Dataset
from huggingface_hub import hf_hub_download


class TextDataset(Dataset):
    def __init__(self, dataset, data_path):
        self.dataset = dataset
        self.data_path = data_path

        self.data, self.id_by_class = self._load_data()
        self.raw_texts = self.data.raw_texts
        self.label_texts = self.data.label_texts

    def __getitem__(self, idx):
        item = {}
        item['node_id'] = idx
        item["labels"] = self.data.y[idx].to(torch.long)
        item["raw_text"] = self.raw_texts[idx]
        item["label_text"] = self.label_texts[item["labels"]]

        return item

    def __len__(self):
        return len(self.raw_texts)
    
    def _get_label_text(self, dataset):
        label_text_list = None
        with open('LLM4GCL/common/label_text.json', 'r', encoding='utf-8') as f:
            label_text_list = json.load(f)[dataset]

        return label_text_list
    
    def _load_data(self, ):
        path = self.data_path + self.dataset + ".pt"
        hf_hub_download(
            repo_id="YYYumo/LLM4GCL",
            filename=self.dataset + ".pt",
            repo_type="dataset",
            local_dir=self.data_path
        )
        data = torch.load(path)

        if self.dataset == 'products' or self.dataset == 'arxiv_23':
            if self.dataset == 'products':
                empty_label = [29, 33]
                delete_label = [22, 26, 27, 30, 34, 35, 38, 39, 40, 41, 43]
            elif self.dataset == 'arxiv_23':
                empty_label = [0, 19]
                delete_label = [12]

            mask = ~torch.isin(data.y, torch.tensor(delete_label))
            to_remove_idx = (~mask).nonzero(as_tuple=True)[0]
            remaining_idx = torch.arange(data.x.size(0))[~torch.isin(torch.arange(data.x.size(0)), to_remove_idx)]

            edge_mask = ~torch.isin(data.edge_index[0], to_remove_idx) & ~torch.isin(data.edge_index[1], to_remove_idx)
            data.edge_index = data.edge_index[:, edge_mask]

            node_map = {}
            edge_index = [[], []]
            for ori_idx, curr_idx in zip(remaining_idx.tolist(), [i for i in range(len(data.x[mask]))]):
                node_map[ori_idx] = curr_idx

            for i in range(data.edge_index.size(1)):
                if data.edge_index[0][i].item() in node_map.keys():
                    edge_index[0].append(node_map[data.edge_index[0][i].item()])

            for i in range(data.edge_index.size(1)):
                if data.edge_index[1][i].item() in node_map.keys():
                    edge_index[1].append(node_map[data.edge_index[1][i].item()])

            data.edge_index = torch.stack((torch.tensor(edge_index[0]), torch.tensor(edge_index[1])), dim=0)
            
            data.x = data.x[mask]
            data.y = data.y[mask]
            data.raw_texts = [data.raw_texts[i] for i in range(len(data.raw_texts)) if mask[i]]
            data.num_nodes = data.num_nodes - 1

            labels = data.y

            delete_label.extend(empty_label)
            delete_label.sort()
            labels = [label for label in data.y if label not in delete_label]
            labels = [label - sum(label > x for x in delete_label) for label in labels]
            labels = torch.tensor(labels)
            data.y = labels

        edge_index, _ = add_self_loops(data.edge_index)
        
        new_data = Data(
            x=data.x,
            edge_index=edge_index,
            y=data.y,
            raw_texts=data.raw_texts,
        )
        data = new_data

        labels = data.y
        class_list = labels.unique().numpy()
        id_by_class = {i: [] for i in class_list}
        for id, cla in enumerate(labels):
            id_by_class[cla.item()].append(id)

        num_nodes = [len(v) for _, v in id_by_class.items()]
        sorted_class_idx = heapq.nlargest(labels.max().item() + 1, enumerate(num_nodes), key=lambda x: x[1])

        # Re-order labels
        label_texts = self._get_label_text(self.dataset)
        sorted_label_texts = copy.deepcopy(label_texts)
        for i, (id, _) in enumerate(sorted_class_idx):
            class_idx = id_by_class[id]
            labels[class_idx] = i
            sorted_label_texts[i] = label_texts[id]
        data.label_texts = sorted_label_texts

        class_list = labels.unique().numpy()
        id_by_class = {i: [] for i in class_list}
        for id, cla in enumerate(labels):
            id_by_class[cla.item()].append(id)

        print(f"--------------------------------------------")
        print(f"Load dataset {self.dataset}!")
        print(f"Node num: {data.x.shape[0]}")
        print(f"Edge num: {data.edge_index.shape[1]}")
        print(f"Class num: {data.y.max().item() + 1}")
        print(f"Feature dim: {data.x.shape[1]}")
        for cls in id_by_class.keys():
            print(f"Class {cls}: {len(id_by_class[cls])} samples")
        print(f"--------------------------------------------")

        return data, id_by_class
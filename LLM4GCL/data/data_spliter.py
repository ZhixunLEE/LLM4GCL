import copy
import torch
import random
import numpy as np

from torch.utils.data import Subset, DataLoader
from collections import defaultdict

class TaskLoader():
        
    def __init__(self, batch_size, text_dataset, cl_type, task_type, base_session, novel_session, ways, sessions, base_train_shots, train_shots, valid_shots, test_shots):
        self.batch_size = batch_size
        self.text_dataset = text_dataset
        self.data = text_dataset.data
        self.id_by_class = text_dataset.id_by_class
        self.cl_type = cl_type
        self.task_type = task_type
        self.label_num = self.data.y.max().item() + 1

        self.base_session = base_session
        self.novel_session = novel_session
        self.ways = ways
        self.sessions = sessions
        self.base_train_shots = base_train_shots
        self.train_shots = train_shots
        self.valid_shots = valid_shots
        self.test_shots = test_shots

        # Task Split
        node_idx_per_class, train_idx_per_task, valid_idx_per_task, test_idx_per_task_isolate, test_idx_per_task_joint, dataset_per_task_isolate, dataset_per_task_joint = self._split_data()
        self.node_idx_per_class = node_idx_per_class
        self.train_idx_per_task = train_idx_per_task
        self.valid_idx_per_task = valid_idx_per_task
        self.test_idx_per_task_isolate = test_idx_per_task_isolate
        self.test_idx_per_task_joint = test_idx_per_task_joint
        self.dataset_per_task_isolate = dataset_per_task_isolate
        self.dataset_per_task_joint = dataset_per_task_joint

    def get_joint_task(self, ):
        all_train_idx, all_valid_idx, all_test_idx = [], [], []
        for task_id in range(self.sessions):
            all_train_idx.extend(self.train_idx_per_task[task_id])
            all_valid_idx.extend(self.valid_idx_per_task[task_id])
            all_test_idx.extend(self.test_idx_per_task_isolate[task_id])

        text_dataset = self.dataset_per_task_joint[self.sessions - 1]

        train_dataset = Subset(text_dataset, all_train_idx)
        val_dataset = Subset(text_dataset, all_valid_idx)
        test_dataset = Subset(text_dataset, all_test_idx)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        valid_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        class_num = self.data.y[all_train_idx].max().item() + 1

        return class_num, text_dataset, train_loader, valid_loader, test_loader

    def get_task(self, task_id, subset = -1):
        if task_id >= self.sessions:
            raise f"Task id {task_id} is larger than total number of tasks {self.sessions} !"
        
        train_idx = self.train_idx_per_task[task_id]
        valid_idx = self.valid_idx_per_task[task_id]
        test_idx_isolate = self.test_idx_per_task_isolate[task_id]
        test_idx_joint = self.test_idx_per_task_joint[task_id]
        
        if subset != -1:
            train_idx = self._stratified_sample(train_idx, self.data.y[train_idx], subset)
            valid_idx = self._stratified_sample(valid_idx, self.data.y[valid_idx], subset)

        text_dataset_isolate = self.dataset_per_task_isolate[task_id]
        text_dataset_joint = self.dataset_per_task_joint[task_id]

        train_dataset = Subset(text_dataset_isolate, train_idx)
        val_dataset = Subset(text_dataset_isolate, valid_idx)
        test_dataset_isolate = Subset(text_dataset_isolate, test_idx_isolate)
        test_dataset_joint = Subset(text_dataset_joint, test_idx_joint)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        valid_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader_isolate = DataLoader(test_dataset_isolate, batch_size=self.batch_size, shuffle=False)
        test_loader_joint = DataLoader(test_dataset_joint, batch_size=self.batch_size, shuffle=False)

        if self.task_type == 'FSNCIL':
            if task_id == 0: # Base Session
                class_src, class_dst = 0, self.base_session
            else:
                class_src, class_dst = self.base_session + (task_id - 1) * self.ways, min(self.base_session + task_id * self.ways, self.label_num)
        elif self.task_type == 'NCIL':
            class_src, class_dst = task_id * self.ways, min((task_id + 1) * self.ways, self.label_num)

        return class_src, class_dst, text_dataset_isolate, text_dataset_joint, train_loader, valid_loader, test_loader_isolate, test_loader_joint

    def _split_data(self, ):
        node_idx_per_class = []
        train_idx_per_task = []
        valid_idx_per_task = []
        test_idx_per_task_isolate = []
        test_idx_per_task_joint = []
        dataset_per_task_isolate = []
        dataset_per_task_joint = []

        all_class = self.data.y.unique(sorted=True).tolist()

        for i in range(self.sessions):
            node_idx_curr_class = []
            train_idx_curr_task = []
            valid_idx_curr_task = []
            test_idx_curr_task = []

            if self.task_type == 'FSNCIL':
                if i == 0: # Base Session
                    curr_task_class_idx = all_class[ : self.base_session]
                else:
                    curr_task_class_idx = all_class[self.base_session + (i - 1) * self.ways : min(self.base_session + i * self.ways, self.label_num)]
            elif self.task_type == 'NCIL':
                curr_task_class_idx = all_class[i * self.ways : min((i + 1) * self.ways, self.label_num)]

            for cla in curr_task_class_idx:
                node_idx = self.id_by_class[cla]
                node_idx_curr_class.extend(node_idx)
                node_num = len(node_idx)

                if self.task_type == 'FSNCIL':
                    if i == 0: # Base Session
                        train_shots, valid_shots, test_shots = self.base_train_shots, self.valid_shots, self.test_shots
                    else:
                        train_shots, valid_shots, test_shots = self.train_shots, self.valid_shots, self.test_shots
                elif self.task_type == 'NCIL':
                    train_shots, valid_shots, test_shots = self.train_shots, self.valid_shots, self.test_shots

                if node_num < (train_shots + valid_shots + test_shots):
                    train_num, valid_num, test_num = int(node_num * 0.5), int(node_num * 0.1), int(node_num * 0.4)
                    if (train_num + valid_num + test_num) > node_num:
                        train_num -= train_num + valid_num + test_num - node_num
                else:
                    train_num, valid_num, test_num = train_shots, valid_shots, test_shots
                random.shuffle(node_idx)

                train_idx_curr_task.extend(node_idx[: train_num])
                valid_idx_curr_task.extend(node_idx[train_num : train_num + valid_num])
                test_idx_curr_task.extend(node_idx[train_num + valid_num: train_num + valid_num + test_num])
       
            node_idx_per_class.append(node_idx_curr_class)
            train_idx_per_task.append(train_idx_curr_task)
            valid_idx_per_task.append(valid_idx_curr_task)
            test_idx_per_task_isolate.append(test_idx_curr_task)

            if i != 0:
                test_idx_per_task_joint.append(test_idx_per_task_joint[-1] + test_idx_curr_task)
            else:
                test_idx_per_task_joint.append(test_idx_curr_task)

            if self.task_type == 'FSNCIL':
                if i == 0: # Base Session
                    curr_dataset = self._adjust_graph(all_class[ : self.base_session])
                    prev_dataset = self._adjust_graph(all_class[ : self.base_session])
                else:
                    curr_dataset = self._adjust_graph(all_class[self.base_session + (i - 1) * self.ways : min(self.base_session + i * self.ways, self.label_num)])
                    prev_dataset = self._adjust_graph(all_class[: min(self.base_session + i * self.ways, self.label_num)])
            elif self.task_type == 'NCIL':
                curr_dataset = self._adjust_graph(all_class[i * self.ways : min((i + 1) * self.ways, self.label_num)])
                prev_dataset = self._adjust_graph(all_class[: min((i + 1) * self.ways, self.label_num)])

            dataset_per_task_isolate.append(curr_dataset)
            dataset_per_task_joint.append(prev_dataset)

        return node_idx_per_class, train_idx_per_task, valid_idx_per_task, test_idx_per_task_isolate, test_idx_per_task_joint, dataset_per_task_isolate, dataset_per_task_joint


    def _adjust_graph(self, class_id):
        node_mask = torch.zeros(len(self.data.y), dtype=torch.bool)
        observe_idx = []
        for cls in class_id:
            observe_idx.extend(self.id_by_class[cls])

        node_mask[observe_idx] = True
        edge_index = self.data.edge_index

        mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
        new_edge_index = edge_index[:, mask]

        text_dataset = copy.deepcopy(self.text_dataset)
        text_dataset.data.edge_index = new_edge_index

        return text_dataset

    def _stratified_sample(self, indices, labels, n_samples):
        label_to_indices = defaultdict(list)
        for idx, label in zip(indices, labels):
            label_to_indices[label.item()].append(idx)

        unique_labels = list(label_to_indices.keys())
        label_counts = [len(label_to_indices[l]) for l in unique_labels]
        proportions = np.array(label_counts) / sum(label_counts)
        samples_per_label = (proportions * n_samples).astype(int)

        samples_per_label = np.maximum(samples_per_label, 1)
        total = sum(samples_per_label)

        while total > n_samples:
            max_idx = np.argmax(samples_per_label)
            if samples_per_label[max_idx] > 1:
                samples_per_label[max_idx] -= 1
                total -= 1

        sampled_indices = []
        for i, label in enumerate(unique_labels):
            sample_size = samples_per_label[i]
            sampled = random.sample(label_to_indices[label], min(sample_size, len(label_to_indices[label])))
            sampled_indices.extend(sampled)

        return sampled_indices

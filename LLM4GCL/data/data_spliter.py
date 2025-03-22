import copy
import torch
import random
from torch.utils.data import Subset, DataLoader


class TaskLoader():
        
    def __init__(self, batch_size, text_dataset, cl_type, task_type, session_size, split_ratio):
        self.batch_size = batch_size
        self.text_dataset = text_dataset
        self.data = text_dataset.data
        self.id_by_class = text_dataset.id_by_class
        self.cl_type = cl_type
        self.task_type = task_type
        self.session_size = session_size
        self.train_ratio = split_ratio[0]
        self.valid_ratio = split_ratio[1]
        self.test_ratio = split_ratio[2]

        if self.cl_type == 'class':
            if self.task_type == 'normal':
                self.task_num = self.data.y.max().item() // self.session_size

        # Task Split
        train_idx_per_task, valid_idx_per_task, test_idx_per_task_isolate, test_idx_per_task_joint, dataset_per_task_isolate, dataset_per_task_joint = self._split_data()
        self.train_idx_per_task = train_idx_per_task
        self.valid_idx_per_task = valid_idx_per_task
        self.test_idx_per_task_isolate = test_idx_per_task_isolate
        self.test_idx_per_task_joint = test_idx_per_task_joint
        self.dataset_per_task_isolate = dataset_per_task_isolate
        self.dataset_per_task_joint = dataset_per_task_joint

    def get_joint_task(self, ):
        all_train_idx, all_valid_idx, all_test_idx = [], [], []
        for task_id in range(self.task_num):
            all_train_idx.extend(self.train_idx_per_task[task_id])
            all_valid_idx.extend(self.valid_idx_per_task[task_id])
            all_test_idx.extend(self.test_idx_per_task_isolate[task_id])

        text_dataset = self.dataset_per_task_joint[self.task_num - 1]

        train_dataset = Subset(text_dataset, all_train_idx)
        val_dataset = Subset(text_dataset, all_valid_idx)
        test_dataset = Subset(text_dataset, all_test_idx)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        valid_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        class_num = self.data.y[all_train_idx].max().item() + 1

        return class_num, text_dataset, train_loader, valid_loader, test_loader

    def get_task(self, task_id, subset = -1):
        if task_id >= self.task_num:
            raise f"Task id {task_id} is larger than total number of tasks {self.task_num} !"
        
        train_idx = self.train_idx_per_task[task_id]
        valid_idx = self.valid_idx_per_task[task_id]
        test_idx_isolate = self.test_idx_per_task_isolate[task_id]
        test_idx_joint = self.test_idx_per_task_joint[task_id]
        
        if subset != -1:
            train_idx = random.sample(train_idx, min(len(train_idx), subset))
            valid_idx = random.sample(valid_idx, min(len(valid_idx), subset))

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

        class_num = self.data.y[train_idx].max().item() + 1

        return class_num, text_dataset_isolate, text_dataset_joint, train_loader, valid_loader, test_loader_isolate, test_loader_joint

    def _split_data(self, ):
        if self.cl_type == 'class':
            if self.task_type == 'normal':
                spliter = self._normal_cls_cl_spliter
    
        return spliter()

    def _normal_cls_cl_spliter(self, ):
        train_idx_per_task = []
        valid_idx_per_task = []
        test_idx_per_task_isolate = []
        test_idx_per_task_joint = []
        dataset_per_task_isolate = []
        dataset_per_task_joint = []

        all_class = self.data.y.unique(sorted=True).tolist()

        for i in range(self.task_num):
            curr_task_class_idx = all_class[i * self.session_size : (i + 1) * self.session_size]
            train_idx_curr_task = []
            valid_idx_curr_task = []
            test_idx_curr_task = []

            for cla in curr_task_class_idx:
                node_idx = self.id_by_class[cla]
                node_num = len(node_idx)
                train_num, valid_num, test_num = int(node_num * self.train_ratio), int(node_num * self.valid_ratio), int(node_num * self.test_ratio)
                random.shuffle(node_idx)

                train_idx_curr_task.extend(node_idx[: train_num])
                valid_idx_curr_task.extend(node_idx[train_num : train_num + valid_num])
                test_idx_curr_task.extend(node_idx[train_num + valid_num: train_num + valid_num + test_num])
       
            train_idx_per_task.append(train_idx_curr_task)
            valid_idx_per_task.append(valid_idx_curr_task)
            test_idx_per_task_isolate.append(test_idx_curr_task)

            if i != 0:
                test_idx_per_task_joint.append(test_idx_per_task_joint[-1] + test_idx_curr_task)
            else:
                test_idx_per_task_joint.append(test_idx_curr_task)

            curr_dataset = self._adjust_graph(all_class[i * self.session_size : (i + 1) * self.session_size])
            dataset_per_task_isolate.append(curr_dataset)
            prev_dataset = self._adjust_graph(all_class[: (i + 1) * self.session_size])
            dataset_per_task_joint.append(prev_dataset)

        return train_idx_per_task, valid_idx_per_task, test_idx_per_task_isolate, test_idx_per_task_joint, dataset_per_task_isolate, dataset_per_task_joint

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

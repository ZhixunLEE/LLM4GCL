import re
import os
import json
import torch
import torch.nn as nn

from LLM4GCL.models import BaseModel
from LLM4GCL.common.prompts import get_genreal_prompts

from openai import OpenAI
from tqdm import tqdm
from textwrap import shorten


def safe_extract_brackets(text_list):
    processed = []
    for text in text_list:
        text = str(text) if text is not None else ""
        match = re.search(r'\((.*?)\)', text)
        processed.append(match.group(1) if match else text)
    return processed


class GPT(BaseModel):

    def __init__(self, task_loader, result_logger, config, checkpoint_path, dataset, model_name, model_path, local_ce, seed, device):
        super(GPT, self).__init__(task_loader, result_logger, config, checkpoint_path, dataset, model_name, local_ce, seed, device)
        self.lm_type = config['lm']
        self.max_length = config['max_length']
        self.result_path = os.path.join(config['result_path'], self.dataset + '_results.json')

        class GPTModel(nn.Module):

            def __init__(self, lm_type, max_length):
                super(GPTModel, self).__init__()
                self.max_length = max_length
                self.lm_type = lm_type

                self.client = OpenAI(
                    api_key="sk-OIfFSkBR56F9lCjS69PWl3ynX9OmbvaEib4d9gDwBANjzeUr",
                    base_url="https://35.aigcbest.top/v1"
                )

            def forward(self, samples, prompts):
                preds, output_data  = [], []
                for i in range(len(samples['node_id'])):
                    text = shorten(samples['raw_text'][i], width=self.max_length, placeholder="...")
                    message = f"{text}\n{prompts}"
                    response = self.client.chat.completions.create(
                        model=self.lm_type,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": message}
                        ],
                    )
                    prediction = response.choices[0].message.content
                    preds.append(prediction)

                    true_label_text = samples['label_text'][i].strip().lower()
                    entry = {
                        "node_id": samples['node_id'][i].item(),
                        "message": message,
                        "prediction": preds[i],
                        "true_label": re.search(r'\((.*?)\)', true_label_text).group(1) if re.search(r'\((.*?)\)', true_label_text) else true_label_text
                    }
                    output_data.append(entry)

                return preds, output_data
                
        self.model = GPTModel(self.lm_type, self.max_length)


    @torch.no_grad()
    def evaluate(self, curr_session, model, text_dataset, test_loader, class_dst, config, device, prompts):
        preds_list, labels_list, output_data_list = [], [], []
        progress_bar = tqdm(range(len(test_loader)))
        progress_bar.set_description(f'Evaluating | Session {curr_session}')
        
        for _, batch in enumerate(test_loader):
            preds, output_data = model(batch, prompts)
            labels = batch['label_text']

            print(preds)
            preds = safe_extract_brackets(preds)
            labels = safe_extract_brackets(labels)

            preds_list.extend(preds)
            labels_list.extend(labels)
            output_data_list.extend(output_data)
            progress_bar.update(1)
        progress_bar.close()

        labels = labels_list
        preds = preds_list

        acc, f1 = self.get_metric(None, preds, labels)

        return acc, f1, preds, labels, output_data_list
    

    def fit(self, iter):
        label_text_list = self.task_loader.text_dataset.label_texts
        json_results = []
        for curr_session in range(self.session_num):
            all_output_data, all_preds, all_labels = [], [], []
            _, class_dst, text_dataset_iso, _, _, _, test_loader_isolate, _ = self.task_loader.get_task(curr_session)

            label_text = label_text_list[ :class_dst]
            label_text = safe_extract_brackets(label_text)
            prompts = get_genreal_prompts(self.dataset, label_text)

            curr_acc_test_isolate, curr_f1_test_isolate, curr_preds_isolate, curr_labels_isolate, curr_output_data_isolate = self.evaluate(curr_session, self.model, text_dataset_iso, test_loader_isolate, class_dst, self.config, self.device, prompts)
            all_preds.extend(curr_preds_isolate)
            all_labels.extend(curr_labels_isolate)
            all_output_data.extend(curr_output_data_isolate)

            acc_list = []
            for s in range(curr_session):
                _, _, text_dataset_iso, _, _, _, test_loader_isolate, _ = self.task_loader.get_task(s)
                prev_acc_test_isolate, prev_f1_test_isolate, prev_preds_isolate, prev_labels_isolate, prev_output_data_isolate = self.evaluate(curr_session, self.model, text_dataset_iso, test_loader_isolate, class_dst, self.config, self.device, prompts)
                all_preds.extend(prev_preds_isolate)
                all_labels.extend(prev_labels_isolate)
                all_output_data.extend(prev_output_data_isolate)
                acc_list.append(prev_acc_test_isolate)
            acc_list.append(curr_acc_test_isolate)

            curr_acc_test_joint, curr_f1_test_joint = self.get_metric(None, all_preds, all_labels)

            print("Session: {} | Iso. Acc Test: {:.4f} | Iso. F1 Test: {:.4f}".format(curr_session, curr_acc_test_isolate, curr_f1_test_isolate))
            print("Session: {} | Jot. Acc Test: {:.4f} | Jot. F1 Test: {:.4f}".format(curr_session, curr_acc_test_joint, curr_f1_test_joint))

            self.result_logger.add_new_results(acc_list, curr_acc_test_joint)
            json_results.append({'session': curr_session, 'results': all_output_data})

        os.makedirs(os.path.dirname(self.result_path), exist_ok=True)
        with open(self.result_path, "w", encoding="utf-8") as f:
            json.dump(json_results, f, ensure_ascii=False, indent=4)

        return self.result_logger
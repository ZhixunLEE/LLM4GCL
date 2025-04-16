import torch
import torch.nn as nn

from peft import (
    LoraConfig,
    get_peft_model,
)
from transformers.models.auto import AutoModel, AutoTokenizer, AutoModelForSequenceClassification


class RoBERTaNet(torch.nn.Module):

    def __init__(self, num_classes, model_path, lora_config, dropout, att_dropout):
        super(RoBERTaNet, self).__init__()
        self.model_name = 'roberta-large'
        self.num_classes = num_classes
        self.model_path = model_path
        self.lora_config = lora_config
        self.dropout = dropout
        self.att_dropout = att_dropout

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)   
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name, cache_dir=self.model_path, num_labels=num_classes)
        model.config.dropout = self.dropout
        model.config.attention_dropout = self.att_dropout
        model.config.output_hidden_states = True
        self.embeddings = model.get_input_embeddings()

        if lora_config['use_lora']:
            self.target_modules = ["query", "value"]
            self.lora_r = self.lora_config['lora_r']
            self.lora_alpha = self.lora_config['lora_alpha']
            self.lora_dropout = self.lora_config['lora_dropout']
            config = LoraConfig(
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                target_modules=self.target_modules,
                lora_dropout=self.lora_dropout,
                task_type="SEQ_CLS",
            )
            self.model = get_peft_model(model, config)
        else:
            self.model = model

    def forward(self, input, attention_mask):
        if input.dtype in (torch.int32, torch.int64) and input.dim() == 2:  # shape: [batch_size, seq_len]
            kwargs = {"input_ids": input}
        else:
            kwargs = {"inputs_embeds": input}
        
        outputs = self.model(
            **kwargs,
            attention_mask=attention_mask,
            return_dict=True,
        )
        
        return outputs


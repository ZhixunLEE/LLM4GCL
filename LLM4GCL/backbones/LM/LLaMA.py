import os
import torch
import torch.nn as nn

from peft import LoraConfig, get_peft_model

from transformers import AutoTokenizer, AutoModelForCausalLM

class LLaMANet(torch.nn.Module):

    def __init__(self, max_length, num_classes, model_path, lora_config, dropout, att_dropout):
        super(LLaMANet, self).__init__()
        self.model_name = 'Llama-3.1-8B'
        self.num_classes = num_classes
        self.model_path = os.path.join(model_path, 'models--' + self.model_name.lower())
        self.lora_config = lora_config
        self.dropout = dropout
        self.att_dropout = att_dropout
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)   
        model = AutoModelForCausalLM.from_pretrained(
            # self.model_name, 
            self.model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            load_in_4bit=True
        )

        model.config.dropout = self.dropout
        model.config.attention_dropout = self.att_dropout
        model.config.output_hidden_states = True
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.embeddings = model.get_input_embeddings()

        if lora_config['use_lora']:
            self.target_modules = ["q_proj", "v_proj"]
            self.lora_r = self.lora_config['lora_r']
            self.lora_alpha = self.lora_config['lora_alpha']
            self.lora_dropout = self.lora_config['lora_dropout']
            config = LoraConfig(
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                target_modules=self.target_modules,
                lora_dropout=self.lora_dropout,
                task_type="CAUSAL_LM",
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

        last_hidden_state = outputs.hidden_states[-1].float()  # shape: [batch_size, seq_len, hidden_size]

        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

import os
import torch
import torch.nn as nn

from peft import LoraConfig, get_peft_model

from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM

class LLaMANet(torch.nn.Module):

    def __init__(self, model_path, lora_config, dropout, att_dropout):
        super(LLaMANet, self).__init__()

        access_token = ''

        # self.model_name = 'meta-llama/Llama-3.1-8B'
        self.model_name = 'meta-llama/Llama-3.2-3B'
        # self.model_name = 'meta-llama/Llama-3.2-1B'
        self.model_path = model_path
        self.lora_config = lora_config
        self.dropout = dropout
        self.att_dropout = att_dropout
        self.max_ans_length = 32
        if self.model_name == 'meta-llama/Llama-3.1-8B':
            self.hidden_dim = 4096
        elif self.model_name == 'meta-llama/Llama-3.2-1B':
            self.hidden_dim = 2048
        elif self.model_name == 'meta-llama/Llama-3.2-3B':
            self.hidden_dim = 3072

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=access_token)   
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            cache_dir=self.model_path,
            device_map="auto",
            quantization_config=quant_config,
            token=access_token
        )

        model.config.dropout = self.dropout
        model.config.attention_dropout = self.att_dropout
        model.config.output_hidden_states = True
        
        self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = 'left'
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


    def forward(self, input, attention_mask, labels=None):
        if input.dtype in (torch.int32, torch.int64) and input.dim() == 2:  # shape: [batch_size, seq_len]
            kwargs = {"input_ids": input}
        else:
            kwargs = {"inputs_embeds": input}

        outputs = self.model(
            **kwargs,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )

        return outputs

    def generate(self, input, attention_mask):
        if input.dtype in (torch.int32, torch.int64) and input.dim() == 2:  # shape: [batch_size, seq_len]
            kwargs = {"input_ids": input}
        else:
            kwargs = {"inputs_embeds": input}
        
        outputs = self.model.generate(
            **kwargs,
            attention_mask=attention_mask,
            max_new_tokens=self.max_ans_length,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        decode_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return decode_outputs

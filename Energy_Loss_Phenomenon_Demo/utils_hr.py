import os
import sys, json, gzip
import random
import time
import jsonlines
import torch
# from torch.utils.tensorboard import SummaryWriter 
import logging 
import time
from transformers import (
    AutoTokenizer,
    default_data_collator,
    AutoModelForCausalLM
)
import argparse
import numpy as np
import os 
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import set_seed
# pip install pydantic==1.10.13
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaSdpaAttention, LlamaMLP, LlamaAttention
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer, MistralMLP

def read_json(file):
    with open(file, 'r', encoding='utf-8') as f:
        return json.load(f)
        
def set_random_seed(seed):
    if seed is not None:
        set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def stream_jsonl(filename: str):
    """
    Parses each jsonl line and yields it as a dictionary
    """
    if filename.endswith(".gz"):
        with open(filename, "rb") as gzfp:
            with gzip.open(gzfp, 'rt') as fp:
                for line in fp:
                    if any(not x.isspace() for x in line):
                        yield json.loads(line)
    else:
        with open(filename, "r") as fp:
            for line in fp:
                if any(not x.isspace() for x in line):
                    yield json.loads(line)

def read_jsonl(file):
    datalist=[]
    with open(file, "r+", encoding="utf-8") as f:
        for item in jsonlines.Reader(f):
            datalist.append(item)
    return datalist

def energy_loss(input_data, output_data):
# energy_loss = ||X||_1 - ||Y||_1
    if len(input_data.shape) == 3:
        assert input_data.shape[0] == 1, "the first dimension for input must be 1"
        input_data = input_data.squeeze(0)

    if len(output_data.shape) == 3:
        assert output_data.shape[0] == 1, "the first dimension for output must be 1"
        output_data = output_data.squeeze(0)

    output_data = output_data.to(torch.float32)
    input_data = input_data.to(torch.float32)

    # input_energy = (torch.norm(input_data, p=2, dim=-1) **2) / torch.tensor(input_data.size(1)).float()
    # output_energy = (torch.norm(output_data, p=2, dim=-1)**2) / torch.tensor(output_data.size(1)).float()
    input_energy = input_data.abs().mean(dim=-1)
    output_energy = output_data.abs().mean(dim=-1)
    loss = input_energy - output_energy
    return loss


def energy_loss_new(input_data, output_data):
# energy_loss = ||X - Y||_1
    if len(input_data.shape) == 3:
        assert input_data.shape[0] == 1, "the first dimension for input must be 1"
        input_data = input_data.squeeze(0)

    if len(output_data.shape) == 3:
        assert output_data.shape[0] == 1, "the first dimension for output must be 1"
        output_data = output_data.squeeze(0)

    output_data = output_data.to(torch.float32)
    input_data = input_data.to(torch.float32)

    loss = (input_data - output_data).abs().mean(dim=-1)
    return loss




def register_mlp_energyloss_hooks_last_layer(model, tokenizer, sample, use_new_energyloss=False, k='31'):
    energyloss_dict = {}
    energy_fn = energy_loss_new if use_new_energyloss else energy_loss
    def hook_fn(module, input, output, layer_name): #(torch.Size([1, 1169, 4096]), ) ---> torch.Size([1, 1169, 4096])
        assert input[0].shape[0:2] == output.shape[0:2], "Big Error"
        energyloss_dict[layer_name] = energy_fn(input[0], output)
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and 'lm_head' not in name and f'layers.{k}.mlp.down_proj' in name:
            handles.append(module.register_forward_hook(lambda self, i, o, name=name: hook_fn(self, i, o, name)))
    _ = model(**{k: v.to(model.device) for k, v in tokenizer(sample, return_tensors='pt').items()})
    for handle in handles:
        handle.remove() 
    return energyloss_dict

def register_mlp_energyloss_hooks_last_layer_deepseek(model, tokenizer, sample, use_new_energyloss=False):
    energyloss_dict = {}
    energy_fn = energy_loss_new if use_new_energyloss else energy_loss
    def hook_fn(module, input, output, layer_name): #(torch.Size([1, 1169, 4096]), ) ---> torch.Size([1, 1169, 4096])
        assert input[0].shape[0:2] == output.shape[0:2], "Big Error"
        energyloss_dict[layer_name] = energy_fn(input[0], output)
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and 'lm_head' not in name and '29.mlp.down_proj' in name:
            handles.append(module.register_forward_hook(lambda self, i, o, name=name: hook_fn(self, i, o, name)))
    _ = model(**{k: v.to(model.device) for k, v in tokenizer(sample, return_tensors='pt').items()})
    for handle in handles:
        handle.remove() 
    return energyloss_dict
    

def register_mlp_energyloss_hooks_last_gate_layer_deepseek(model, tokenizer, sample, use_new_energyloss=False):
    energyloss_dict = {}
    from torch import nn
    act_silu = nn.SiLU()
    energy_fn = energy_loss_new if use_new_energyloss else energy_loss
    def hook_fn(module, input, output, layer_name): #(torch.Size([1, 1169, 4096]), ) ---> torch.Size([1, 1169, 4096])
        assert input[0].shape[0:2] == output.shape[0:2], "Big Error"
        energyloss_dict[layer_name] = energy_fn(input[0], output)
        # energyloss_dict[layer_name] = energy_fn(input[0], act_silu(output))
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and 'lm_head' not in name and '29.mlp.gate_proj' in name:
            handles.append(module.register_forward_hook(lambda self, i, o, name=name: hook_fn(self, i, o, name)))
    _ = model(**{k: v.to(model.device) for k, v in tokenizer(sample, return_tensors='pt').items()})
    for handle in handles:
        handle.remove() 
    return energyloss_dict

def register_mlp_energyloss_hooks_last_gate_layer(model, tokenizer, sample, use_new_energyloss=False, k='31'):
    energyloss_dict = {}
    from torch import nn
    act_silu = nn.SiLU()
    energy_fn = energy_loss_new if use_new_energyloss else energy_loss
    def hook_fn(module, input, output, layer_name): #(torch.Size([1, 1169, 4096]), ) ---> torch.Size([1, 1169, 4096])
        assert input[0].shape[0:2] == output.shape[0:2], "Big Error"
        energyloss_dict[layer_name] = energy_fn(input[0], output)
        # energyloss_dict[layer_name] = energy_fn(input[0], act_silu(output))
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and 'lm_head' not in name and f'layers.{k}.mlp.gate_proj' in name:
            handles.append(module.register_forward_hook(lambda self, i, o, name=name: hook_fn(self, i, o, name)))
    _ = model(**{k: v.to(model.device) for k, v in tokenizer(sample, return_tensors='pt').items()})
    for handle in handles:
        handle.remove() 
    return energyloss_dict

def register_mlp_energyloss_hooks_last_up_layer(model, tokenizer, sample, use_new_energyloss=False, k='31'):
    energyloss_dict = {}
    from torch import nn
    act_silu = nn.SiLU()
    energy_fn = energy_loss_new if use_new_energyloss else energy_loss
    def hook_fn(module, input, output, layer_name): #(torch.Size([1, 1169, 4096]), ) ---> torch.Size([1, 1169, 4096])
        assert input[0].shape[0:2] == output.shape[0:2], "Big Error"
        energyloss_dict[layer_name] = energy_fn(input[0], output)
    handles = []
    linear_dict = {name: module for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and 'lm_head' not in name and f'layers.{k}.mlp.up_proj' in name:
            handles.append(module.register_forward_hook(lambda self, i, o, name=name: hook_fn(self, i, o, name)))
    _ = model(**{k: v.to(model.device) for k, v in tokenizer(sample, return_tensors='pt').items()})
    for handle in handles:
        handle.remove() 
    return energyloss_dict

def register_mlp_energyloss_hooks_last_up_layer_deepseek(model, tokenizer, sample, use_new_energyloss=False):
    energyloss_dict = {}
    from torch import nn
    act_silu = nn.SiLU()
    energy_fn = energy_loss_new if use_new_energyloss else energy_loss
    def hook_fn(module, input, output, layer_name): #(torch.Size([1, 1169, 4096]), ) ---> torch.Size([1, 1169, 4096])
        assert input[0].shape[0:2] == output.shape[0:2], "Big Error"
        energyloss_dict[layer_name] = energy_fn(input[0], output)
    handles = []
    linear_dict = {name: module for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and 'lm_head' not in name and '29.mlp.up_proj' in name:
            handles.append(module.register_forward_hook(lambda self, i, o, name=name: hook_fn(self, i, o, name)))
    _ = model(**{k: v.to(model.device) for k, v in tokenizer(sample, return_tensors='pt').items()})
    for handle in handles:
        handle.remove() 
    return energyloss_dict




def create_llama_model(model_name_or_path, device):
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16).to(device)
    use_fast_tokenizer = "LlamaForCausalLM" not in model.config.architectures
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=use_fast_tokenizer, padding_side="left", legacy=False)
    tokenizer.pad_token = tokenizer.unk_token
    return model, tokenizer



@torch.no_grad()
def evaluation_energyloss_last_layer(model, tokenizer, batch, tau = 0.01, response_tag = 'actor_latest_response', num = 40, tag = None, use_new_energyloss=False, k='31'):
    model.eval()
    sample = batch['input'] + ' ' + batch[response_tag]
    start_idx = len(tokenizer(batch['input'] + ' ')['input_ids']) - 1
    final_idx = len(tokenizer(sample)['input_ids'])-2

    if tag == "mlp_energyloss":
        activation_dict = register_mlp_energyloss_hooks_last_layer(model, tokenizer, sample, use_new_energyloss=use_new_energyloss, k=k)
    elif tag == "gate_mlp_energyloss":
        activation_dict = register_mlp_energyloss_hooks_last_gate_layer(model, tokenizer, sample, use_new_energyloss=use_new_energyloss, k=k)
    elif tag == "up_mlp_energyloss":
        activation_dict = register_mlp_energyloss_hooks_last_up_layer(model, tokenizer, sample, use_new_energyloss=use_new_energyloss, k=k)
    else:
        exit()
    num = min(num, final_idx - start_idx)
    
    activation_dict = {k:v[start_idx:start_idx + num + 1] for k,v in activation_dict.items()}
    assert len(list(activation_dict.keys())) == 1, "the number of considered later must be 1"
    return list(activation_dict.values())[0]

@torch.no_grad()
def evaluation_energyloss_last_layer_deepseek(model, tokenizer, batch, tau = 0.01, response_tag = 'actor_latest_response', num = 40, tag = None, use_new_energyloss=False):
    model.eval()
    sample = batch['input'] + ' ' + batch[response_tag]
    start_idx = len(tokenizer(batch['input'] + ' ')['input_ids']) - 1
    final_idx = len(tokenizer(sample)['input_ids'])-2

    if tag == "mlp_energyloss":
        activation_dict = register_mlp_energyloss_hooks_last_layer_deepseek(model, tokenizer, sample, use_new_energyloss=use_new_energyloss)
    elif tag == "gate_mlp_energyloss":
        activation_dict = register_mlp_energyloss_hooks_last_gate_layer_deepseek(model, tokenizer, sample, use_new_energyloss=use_new_energyloss)
    elif tag == "up_mlp_energyloss":
        activation_dict = register_mlp_energyloss_hooks_last_up_layer_deepseek(model, tokenizer, sample, use_new_energyloss=use_new_energyloss)
    else:
        exit()
    num = min(num, final_idx - start_idx)
    
    activation_dict = {k:v[start_idx:start_idx + num + 1] for k,v in activation_dict.items()}
    assert len(list(activation_dict.keys())) == 1, "the number of considered later must be 1"
    return list(activation_dict.values())[0]


@torch.no_grad()
def evaluation_energyloss_last_layer_mistral(model, tokenizer, batch, tau = 0.01, response_tag = 'actor_latest_response', num = 40, tag = None, use_new_energyloss=False, k='31'):
    model.eval()
    sample = batch['input'] + ' ' + batch[response_tag]
    start_idx = len(tokenizer(batch['input'] + ' ')['input_ids']) - 1
    final_idx = len(tokenizer(sample)['input_ids'])-2

    if tag == "mlp_energyloss":
        activation_dict = register_mlp_energyloss_hooks_last_layer(model, tokenizer, sample, use_new_energyloss=use_new_energyloss, k=k)
    elif tag == "gate_mlp_energyloss":
        activation_dict = register_mlp_energyloss_hooks_last_gate_layer(model, tokenizer, sample, use_new_energyloss=use_new_energyloss, k=k)
    elif tag == "up_mlp_energyloss":
        activation_dict = register_mlp_energyloss_hooks_last_up_layer(model, tokenizer, sample, use_new_energyloss=use_new_energyloss, k=k)

    else:
        exit()
    num = min(num, final_idx - start_idx)
    
    activation_dict = {k:v[start_idx:start_idx + num + 1] for k,v in activation_dict.items()}
    assert len(list(activation_dict.keys())) == 1, "the number of considered later must be 1"
    return list(activation_dict.values())[0]
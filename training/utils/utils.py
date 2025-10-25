# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import os
import torch
import random
import numpy as np
from transformers import set_seed, AutoTokenizer
import deepspeed
from deepspeed.accelerator import get_accelerator
import torch.nn as nn
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

def print_rank_0(msg, rank=None):
    if rank is not None and rank <= 0:
        print(msg)
    # elif is_rank_0():
    #     print(msg)


def is_rank_0():
    """Check whether it is rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            return True
        else:
            return False
    else:
        return True


def to_device(batch, device):
    output = {}
    for k, v in batch.items():
        try:
            output[k] = v.to(device)
        except:
            output[k] = v
    return output


class MovingAverage:

    def __init__(self):
        self.count = 0
        self.total = 0
        self.mean = 0

    def update(self, num):
        self.total += num
        self.count += 1
        self.mean = self.total / self.count
        return self.mean


def get_tokenizer(model_name_or_path, fast_tokenizer=True, use_unk=False):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=fast_tokenizer, trust_remote_code=True)
    if tokenizer.pad_token is None :
        tokenizer.pad_token = tokenizer.eos_token
    if use_unk:
        tokenizer.pad_token = tokenizer.unk_token
    if 'gptneox' in tokenizer.__class__.__name__.lower():
        tokenizer.pad_token_id = 1
    if 'llama3' in model_name_or_path.lower():
        tokenizer.pad_token = "<|eot_id|>"
    if 'qwen2.5' in model_name_or_path.lower():
        tokenizer.pad_token = '<|endoftext|>' # 151643
        tokenizer.eos_token = '<|im_end|>' # 151645
        tokenizer.bos_token = '<|im_start|>' # 151644
    if 'qwen2' in tokenizer.__class__.__name__.lower():
        tokenizer.bos_token = "<|im_start|>"
        tokenizer.pad_token = "<|endoftext|>"
    print(f'the pad token of current tokenizer is {tokenizer.pad_token}')
    # make sure tokenizer is right pad in our logic
    tokenizer.padding_side = 'right'
    return tokenizer

def get_llama3_roleplay_tokenizer(model_name_or_path, fast_tokenizer=True, use_unk=False):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=fast_tokenizer, trust_remote_code=True)
    tokenizer.pad_token = "<|reserved_special_token_0|>"
    print(f'the pad token of current tokenizer is {tokenizer.pad_token}')
    # make sure tokenizer is right pad in our logic
    tokenizer.padding_side = 'right'
    return tokenizer


def load_hf_tokenizer(model_name_or_path, fast_tokenizer=True, use_unk=False):
    tokenizer = get_tokenizer(model_name_or_path, fast_tokenizer=fast_tokenizer, use_unk=use_unk)
    return tokenizer

def set_random_seed(seed):
    if seed is not None:
        set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        get_accelerator().manual_seed_all(seed)


def get_all_reduce_mean(tensor):
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    tensor = tensor / torch.distributed.get_world_size()
    return tensor


# This function is a modified version of code available in the from_pretrained API of HuggingFace Transformers
# The code is copied and modified from: https://github.com/huggingface/transformers/blob/5ee9693a1c77c617ebc43ef20194b6d3b674318e/src/transformers/modeling_utils.py#L498
# This function helps load a HF format checkpoint into a DeepSpeed wrapped model that has been sharded using ZeRO Stage 3
def load_state_dict_into_model(model_to_load: nn.Module,
                               state_dict=None,
                               start_prefix="",
                               zero_stage=0):
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    error_msgs = []

    # PyTorch's `_load_from_state_dict` does not copy parameters in a module's descendants
    # so we need to apply the function recursively.
    def load(module: nn.Module, state_dict, prefix=""):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        args = (state_dict, prefix, local_metadata, True, [], [], error_msgs)
        # Parameters of module and children will start with prefix. We can exit early if there are none in this
        # state_dict
        if len([key for key in state_dict if key.startswith(prefix)]) > 0:
            if zero_stage == 3:
                # In sharded models, each shard has only part of the full state_dict, so only gather
                # parameters that are in the current state_dict.
                named_parameters = dict(module.named_parameters(prefix=prefix[:-1], recurse=False))
                params_to_gather = [
                    named_parameters[k] for k in state_dict.keys()
                    if k in named_parameters
                ]
                if len(params_to_gather) > 0:
                    # because zero3 puts placeholders in model params, this context
                    # manager gathers (unpartitions) the params of the current layer, then loads from
                    # the state dict and then re-partitions them again
                    with deepspeed.zero.GatheredParameters(params_to_gather,
                                                           modifier_rank=0):
                        if torch.distributed.get_rank() == 0:
                            module._load_from_state_dict(*args)
            else:
                module._load_from_state_dict(*args)

        for name, child in module._modules.items():
            if child is not None:
                load(child, state_dict, prefix + name + ".")

    load(model_to_load, state_dict, prefix=start_prefix)
    # Delete `state_dict` so it could be collected by GC earlier. Note that `state_dict` is a copy of the argument, so
    # it's safe to delete it.
    del state_dict
    return error_msgs

def get_optimizer_grouped_parameters_specialweight(
    model,
    weight_decay,
    actor_learning_rate,
    lora_lr=5e-4,
    no_decay_name_list=["bias", "LayerNorm.weight"],
    lora_name_list=["lora_right_weight", "lora_left_weight"],
    special_layer_weight=[],
    special_lr_multiplier=1.0
):
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if (not any(nd in n for nd in no_decay_name_list)
                    and p.requires_grad and not any(nd in n
                                                    for nd in lora_name_list)
                    and not any(nd in n for nd in special_layer_weight))
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (not any(nd in n for nd in no_decay_name_list)
                    and not any(nd in n for nd in special_layer_weight)
                    and p.requires_grad and any(nd in n
                                                for nd in lora_name_list))
            ],
            "weight_decay": weight_decay,
            "lr": lora_lr
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (any(nd in n
                        for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (any(nd in n
                        for nd in special_layer_weight) and p.requires_grad)
            ],
            "weight_decay": weight_decay,
            "lr": special_lr_multiplier * actor_learning_rate,
        },
    ]

    non_empty_groups = []
    for group in optimizer_grouped_parameters:
        if group["params"]:
            non_empty_groups.append(group)
    return non_empty_groups


def get_optimizer_grouped_parameters(
    model,
    weight_decay,
    lora_lr=5e-4,
    no_decay_name_list=["bias", "LayerNorm.weight"],
    lora_name_list=["lora_right_weight", "lora_left_weight"],
):
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if (not any(nd in n for nd in no_decay_name_list)
                    and p.requires_grad and not any(nd in n
                                                    for nd in lora_name_list))
            ],
            "weight_decay":
            weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (not any(nd in n for nd in no_decay_name_list)
                    and p.requires_grad and any(nd in n
                                                for nd in lora_name_list))
            ],
            "weight_decay":
            weight_decay,
            "lr":
            lora_lr
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (any(nd in n
                        for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay":
            0.0,
        },
    ]

    non_empty_groups = []
    for group in optimizer_grouped_parameters:
        if group["params"]:
            non_empty_groups.append(group)
    return non_empty_groups


def moving_average(model, model_ema, beta=0.992, device=None, zero_stage=0):
    zero_stage_3 = (zero_stage == 3)
    with torch.no_grad():
        for param, param_ema in zip(model.parameters(),
                                    model_ema.parameters()):
            # TODO: use prefiltering for efficiency
            params_to_fetch = _z3_params_to_fetch([param, param_ema
                                                   ]) if zero_stage_3 else []
            should_gather_param = len(params_to_fetch) > 0
            with deepspeed.zero.GatheredParameters(
                    params_to_fetch, enabled=should_gather_param):
                data = param.data
                if device is not None:
                    data = data.to(device)
                param_ema.data.copy_(torch.lerp(data, param_ema.data, beta))

def _z3_params_to_fetch(param_list):
    return [
        p for p in param_list
        if hasattr(p, 'ds_id') and p.ds_status == ZeroParamStatus.NOT_AVAILABLE
    ]

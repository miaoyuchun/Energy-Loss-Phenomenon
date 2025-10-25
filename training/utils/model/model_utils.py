# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import os
import io
import torch
import time
import deepspeed
import json
from transformers import AutoConfig, AutoModel
from transformers.integrations.deepspeed import HfDeepSpeedConfig
from transformers.modeling_utils import get_parameter_dtype, shard_checkpoint
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from petrel_client.client import Client
try:
    from .reward_model import RewardModel, RewardLlama
    from utils.model.reward_mistral import RewardMistral
    from utils.model.reward_deepseek import RewardDeepseek
except:
    print("can not import reward model class")
from ..utils import load_state_dict_into_model
from ..utils import print_rank_0

def configure_dropout(model_config, dropout):
    if dropout is not None:
        for key in ['attention_dropout']:
            if hasattr(model_config, key):
                print(f"Setting model_config.{key} to {dropout}")
                setattr(model_config, key, dropout)


def create_hf_model(model_class,
                    model_name_or_path,
                    dtype,
                    ds_config=None,
                    rlhf_training=False,
                    dropout=None,
                    lastlayer_dropout=0):
    model_config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    configure_dropout(model_config, dropout)
    # Note: dschf is defined in function scope to avoid global effects
    # https://huggingface.co/docs/transformers/main_classes/deepspeed#nontrainer-deepspeed-integration
    if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
        dschf = HfDeepSpeedConfig(ds_config)
    else:
        dschf = None
    if lastlayer_dropout == 0.0:
        model = model_class.from_pretrained(
            model_name_or_path, 
            trust_remote_code=True,
            config=model_config, 
            use_flash_attention_2 = model_config.model_type =='llama',
            torch_dtype=dtype,
            use_safetensors = False if 'qwen' not in model_name_or_path.lower() and 'llama3' not in model_name_or_path.lower()  and 'mistral' not in model_name_or_path.lower() and 'phi' not in model_name_or_path.lower() and 'gemma' not in model_name_or_path.lower() else True
        )
    else:
        exit()
    return model

def create_llama_critic_model(
        args,
        model_name_or_path,
        tokenizer,
        ds_config,
        dtype,
        num_padding_at_beginning=0,
        dropout=None,
        zero_stage=0,
        rlhf_training=False):
    model_config = AutoConfig.from_pretrained(model_name_or_path)
    if model_config.model_type == 'llama' and 'llama' in model_name_or_path.lower():
        model_class = RewardLlama
    elif model_config.model_type == 'llama' and 'deepseek' in model_name_or_path.lower():
        model_class = RewardDeepseek
    elif model_config.model_type == 'mistral':
        print("mistral is utilized")
        model_class =RewardMistral
    else:
        print("Erroe!!!! reward model type only support llama, mistral, and deepseek !!!!")
        exit()

    configure_dropout(model_config, dropout)
    print(f"loading from {model_name_or_path}")

    args.use_sigmoid = False
    args.use_simcse = False
    args.use_variational_inference = False
    args.use_encoder_sigmoid = False
    args.use_decoder_sigmoid = False
    args.no_sample = False
    args.fast_version = False
    args.introduce_supervision = False
    args.internal_entropy = False
    args.complex_fcn_decode = False
    args.beta = -1
    args.gamma = -1
    args.lamda = -1
    args.omega = -1
    
    if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
        dschf = HfDeepSpeedConfig(ds_config)
    else:
        dschf = None
    critic_model = model_class.from_pretrained(
        model_name_or_path, 
        config=model_config,
        torch_dtype=dtype,
        tokenizer=tokenizer,
        pad_id=tokenizer.pad_token_id, 
        num_padding_at_beginning=num_padding_at_beginning,
        use_sigmoid=args.use_sigmoid,
        only_reward_final_token=args.only_reward_final_token,
        use_flash_attention_2=model_config.model_type =='llama',
        use_variational_inference=args.use_variational_inference,
        use_encoder_sigmoid=args.use_encoder_sigmoid,
        use_decoder_sigmoid=args.use_decoder_sigmoid,
        use_simcse=args.use_simcse,
        beta=args.beta,
        gamma=args.gamma,
        lamda=args.lamda,
        omega=args.omega,
        fast_version=args.fast_version,
        no_sample = args.no_sample,
        complex_fcn_decode = args.complex_fcn_decode,
        introduce_supervision = args.introduce_supervision,
        dim_factor = args.dim_factor,
        factor_num = args.factor_num,
        factor_width = args.factor_width,
        internal_entropy = args.internal_entropy
    )


    return critic_model
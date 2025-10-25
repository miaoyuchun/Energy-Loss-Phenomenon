# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import os
import time
import torch
import deepspeed
from deepspeed.ops.adam import FusedAdam
from deepspeed.ops.adam import DeepSpeedCPUAdam
from transformers import AutoModelForCausalLM, get_scheduler, GPTNeoXForCausalLM, LlamaForCausalLM
from utils.ds_utils import get_train_ds_config, get_eval_ds_config
from utils.module.lora import (
    convert_linear_layer_to_lora,
    only_optimize_lora_parameters, 
    make_model_gradient_checkpointing_compatible
)
from utils.model.model_utils import create_hf_model, create_llama_critic_model
from utils.utils import get_optimizer_grouped_parameters, get_optimizer_grouped_parameters_specialweight, get_all_reduce_mean


"""
TODOs:
  * support HF models for critic (for debugging), must be a previously saved ckpt from step-2
  * determine ds_config/zero_stage based on model size, gpu style, world size, etc
    - get model size by creating simple meta model
    - 1.3b: zero-2 for actor/ref models, zero-0 for others
    - 13b+: zero-3 for all models
"""


def log_init(model_name, stime=None):
    if torch.distributed.get_rank() == 0:
        tag = "start" if stime is None else "end"
        suffix = "ing" if stime is None else "ed"
        duration = ""
        if stime is not None:
            duration = "(duration: {:.2f}s)".format(time.time() - stime)
        msg = f"[{tag}] Initializ{suffix} {model_name} Model [{tag}] {duration}"
        stars = (90 - len(msg)) // 2
        extra_star = "*" if (90 - len(msg)) % 2 == 1 else ""
        print("*" * stars + msg + "*" * stars + extra_star)
        return time.time()

class DeepSpeedRLHFEngine():
    def __init__(self, actor_model_name_or_path, critic_model_name_or_path,
                 tokenizer, args, num_total_iters, gold_reward_tokenizer=None):
        self.args = args
        self.num_total_iters = num_total_iters
        self.tokenizer = tokenizer
        self.gold_reward_tokenizer = gold_reward_tokenizer
        self.actor = self._init_actor(actor_model_name_or_path=actor_model_name_or_path)
        self.ref = self._init_ref(actor_model_name_or_path=actor_model_name_or_path)
        # self.ref = self._init_ref(actor_model_name_or_path=actor_model_name_or_path)
        self.actor_ema = None
        if self.args.enable_ema:
            self.actor_ema = self._init_ema(actor_model_name_or_path=actor_model_name_or_path)
        self.critic = self._init_critic(critic_model_name_or_path=actor_model_name_or_path)
        # self.critic = self._init_critic(critic_model_name_or_path=critic_model_name_or_path)
        self.reward = self._init_reward(reward_model_name_or_path=critic_model_name_or_path)
        # self.critic, self.reward, self.ref = None, None, None

        if self.args.critic_gradient_checkpointing:
            self.critic.gradient_checkpointing_enable()

    def _init_actor(self, actor_model_name_or_path):
        stime = log_init("Actor")

        # DS Config
        ds_config = get_train_ds_config(
            offload=self.args.offload,
            dtype=self.args.dtype,
            stage=self.args.actor_zero_stage,
            enable_hybrid_engine=self.args.enable_hybrid_engine,
            inference_tp_size=self.args.inference_tp_size,
            release_inference_cache=self.args.release_inference_cache,
            pin_parameters=(not self.args.unpin_actor_parameters),
            tp_gather_partition_size=self.args.tp_gather_partition_size,
            enable_tensorboard=self.args.enable_tensorboard,
            enable_mixed_precision_lora=self.args.enable_mixed_precision_lora,
            tb_path=self.args.tensorboard_path,
            tb_name="step3_actor")
        ds_config[
            'train_micro_batch_size_per_gpu'] = self.args.per_device_train_batch_size
        ds_config[
            'train_batch_size'] = self.args.per_device_train_batch_size * torch.distributed.get_world_size(
            ) * self.args.gradient_accumulation_steps_actor

        # Model
        model_class=AutoModelForCausalLM

        actor_model = create_hf_model(
            model_class,
            actor_model_name_or_path,
            torch.float16 if self.args.dtype == "fp16" else torch.bfloat16,
            ds_config=ds_config,
            dropout=self.args.actor_dropout,
            lastlayer_dropout=self.args.lastlayer_dropout
            )

        # LoRA
        if self.args.actor_lora_dim > 0:
            actor_model = convert_linear_layer_to_lora(
                actor_model, self.args.actor_lora_module_name,
                self.args.actor_lora_dim)
            if self.args.only_optimize_lora:
                actor_model = only_optimize_lora_parameters(actor_model)
                actor_model = make_model_gradient_checkpointing_compatible(
                    actor_model)
                
        if  self.args.special_lr_multiplier == 0:
            for n, p in actor_model.named_parameters():
                if n in self.args.special_layer_weight:
                    p.requires_grad = False
                    print(f"{n} is set to no grad !!!")

        # Optimizer
        AdamOptimizer = DeepSpeedCPUAdam if self.args.offload else FusedAdam

        if self.args.special_lr_multiplier > 0 and self.args.special_lr_multiplier < 1:
            optim_params = get_optimizer_grouped_parameters_specialweight(
                actor_model, self.args.actor_weight_decay, self.args.actor_learning_rate, self.args.actor_lora_learning_rate,
                special_layer_weight=self.args.special_layer_weight, special_lr_multiplier=self.args.special_lr_multiplier)
        else:
            optim_params = get_optimizer_grouped_parameters(
                actor_model, self.args.actor_weight_decay,
                self.args.actor_lora_learning_rate)
            
        optim = AdamOptimizer(optim_params,
                            lr=self.args.actor_learning_rate,
                            betas=(0.9, 0.95))



        # LR Scheduler
        lr_scheduler = get_scheduler(
            name=self.args.lr_scheduler_type,
            optimizer=optim,
            num_warmup_steps=self.args.num_warmup_steps,
            num_training_steps=self.num_total_iters,
        )
        self.actor_engine, *_ = deepspeed.initialize(model=actor_model,
                                                optimizer=optim,
                                                lr_scheduler=lr_scheduler,
                                                config=ds_config,
                                                dist_init_required=True)

        log_init("Actor", stime=stime)
        return self.actor_engine

    def _init_ref(self, actor_model_name_or_path):
        stime = log_init("Ref")
        # DS Config
        zero_stage = self.args.actor_zero_stage
        if zero_stage != 3:
            zero_stage = 3
        ds_config = get_eval_ds_config(self.args.offload_reference_model, self.args.dtype, zero_stage)
        ds_config[
            'train_micro_batch_size_per_gpu'] = self.args.per_device_train_batch_size
        ds_config[
            'train_batch_size'] = self.args.per_device_train_batch_size * torch.distributed.get_world_size(
            ) * self.args.gradient_accumulation_steps_actor

        model_class=AutoModelForCausalLM

        ref_model = create_hf_model(model_class,
                                    actor_model_name_or_path, 
                                    torch.float16 if self.args.dtype == "fp16" else torch.bfloat16,
                                    ds_config)

        ref_engine, *_ = deepspeed.initialize(model=ref_model,
                                              config=ds_config,
                                              dist_init_required=True)

        ref_engine.eval()
        log_init("Ref", stime=stime)     
        return ref_engine

    def _init_ema(self, actor_model_name_or_path):
        stime = log_init("EMA")
        # DS Config
        zero_stage = self.args.actor_zero_stage
        if zero_stage != 3:
            # If actor is ZeRO-3 then we use it for everything, otherwise assume we have enough memory
            zero_stage = 0
        ds_config = get_eval_ds_config(self.args.offload_reference_model,
                                       self.args.dtype, zero_stage)
        ds_config[
            'train_micro_batch_size_per_gpu'] = self.args.per_device_train_batch_size
        #TODO(jeff): we should probably set grad accumlation steps here as well for clarity
        ds_config[
            'train_batch_size'] = self.args.per_device_train_batch_size * torch.distributed.get_world_size(
            ) * self.args.gradient_accumulation_steps_actor

        actor_model_ema = create_hf_model(AutoModelForCausalLM,
                                          actor_model_name_or_path,
                                          torch.float16 if self.args.dtype == "fp16" else torch.bfloat16,
                                          ds_config)
        if self.args.actor_lora_dim > 0:
            actor_model_ema = convert_linear_layer_to_lora(
                actor_model_ema, self.args.actor_lora_module_name,
                self.args.actor_lora_dim)

        ema_engine, *_ = deepspeed.initialize(model=actor_model_ema,
                                              config=ds_config,
                                              dist_init_required=True)

        log_init("EMA", stime=stime)
        return ema_engine

    def _init_critic(self, critic_model_name_or_path):
        stime = log_init("Critic")
        ds_config = get_train_ds_config(
            offload=False,
            dtype='bf16',
            stage=self.args.critic_zero_stage,
            enable_tensorboard=self.args.enable_tensorboard,
            tb_path=self.args.tensorboard_path,
            tb_name="step3_critic")
        ds_config[
            'train_micro_batch_size_per_gpu'] = self.args.per_device_train_batch_size
        #TODO(jeff): we should probably set grad accumlation steps here as well for clarity
        ds_config[
            'train_batch_size'] = self.args.per_device_train_batch_size * torch.distributed.get_world_size(
            ) * self.args.gradient_accumulation_steps
        
        # Model
        critic_model = create_llama_critic_model(
            self.args,
            critic_model_name_or_path,
            self.tokenizer,
            ds_config,
            torch.bfloat16,
            rlhf_training=False,
            dropout=self.args.critic_dropout,
            zero_stage=self.args.critic_zero_stage
        )


        # Optimizer
        AdamOptimizer = DeepSpeedCPUAdam if self.args.offload else FusedAdam
        optim_params = get_optimizer_grouped_parameters(
            critic_model, self.args.critic_weight_decay, self.args.critic_lora_learning_rate)
        optim = AdamOptimizer(optim_params,
                              lr=self.args.critic_learning_rate,
                              betas=(0.9, 0.95))

        # LR Scheduler
        lr_scheduler = get_scheduler(
            name=self.args.lr_scheduler_type,
            optimizer=optim,
            num_warmup_steps=self.args.num_warmup_steps,
            num_training_steps=self.num_total_iters,
        )

        # DeepSpeed Engine
        self.critic_engine, *_ = deepspeed.initialize(model=critic_model,
                                                 optimizer=optim,
                                                 lr_scheduler=lr_scheduler,
                                                 config=ds_config,
                                                 dist_init_required=True)

        log_init("Critic", stime=stime)
        return self.critic_engine

    def _init_reward(self, reward_model_name_or_path):
        stime = log_init("Reward")
        # DS Config
        zero_stage = self.args.critic_zero_stage
        if zero_stage != 3:
            # If critic is ZeRO-3 then we use it for everything, otherwise assume we have enough memory
            zero_stage = 3

        ds_eval_config = get_eval_ds_config(offload=False, dtype='bf16', stage=zero_stage)

        # We need to set train batch size and micro batch size here to pass the sanity check of DeepSpeed engine.
        ds_eval_config[
            'train_micro_batch_size_per_gpu'] = self.args.per_device_train_batch_size
        ds_eval_config[
            'train_batch_size'] = self.args.per_device_train_batch_size * torch.distributed.get_world_size(
            ) * self.args.gradient_accumulation_steps

        # Model
        args = self.args

        reward_model = create_llama_critic_model(
            args,
            reward_model_name_or_path,
            self.tokenizer,
            ds_eval_config,
            torch.bfloat16,
            rlhf_training=False,
            dropout=self.args.critic_dropout,
            zero_stage=self.args.critic_zero_stage
        )
        reward_engine, *_ = deepspeed.initialize(model=reward_model,
                                                 config=ds_eval_config,
                                                 dist_init_required=True)
        reward_engine.eval()
        log_init("Reward", stime=stime)
        return reward_engine
import os, glob
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import random
import time
import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter 
import logging 
import time
from transformers import (
    AutoTokenizer,
    default_data_collator,
)
from transformers.models.llama.modeling_llama import LlamaMLP
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXMLP
import deepspeed
import numpy as np
from ppo_trainer import DeepSpeedPPOTrainer, DeepSpeedPPOTrainerUnsupervised
from rlhf_engine import DeepSpeedRLHFEngine
from deepspeed.accelerator import get_accelerator
from utils.data.data_utils import create_prompt_dataset, MiniDataset, DataCollatorRLHF, get_unsupervised_data, DataCollatorSFT
from utils.model.model_utils import save_hf_format, save_zero_three_model
from utils.utils import (
    print_rank_0, 
    to_device, 
    set_random_seed, 
    get_all_reduce_mean, 
    moving_average, 
    load_hf_tokenizer
)
from utils.utils_hr import register_mlpblock_energyloss_hooks_last_layer, register_mlpblock_energyloss_hooks_last_layer_misture
from utils.module.lora import convert_lora_to_linear_layer
from utils.perf import print_throughput_step3
from utils.parse import parse_args


def create_datasets(args, tokenizer, train_phase=3):
    unsupervised_training_enabled = args.unsupervised_dataset_name and args.unsupervised_dataset_config_name
    prompt_train_dataset, _ = create_prompt_dataset(args, train_phase, tokenizer, args.end_of_conversation_token)

    if unsupervised_training_enabled:
        unsupervised_train_dataset = get_unsupervised_data(args, tokenizer)
    else:
        unsupervised_train_dataset = None

    # DataLoaders creation:
    if train_phase==3:
        data_collator = DataCollatorRLHF(
            pad_token_id=tokenizer.pad_token_id,
            inference_tp_size=args.inference_tp_size
        )
    elif train_phase==1:
        data_collator = DataCollatorSFT(
        pad_token_id=tokenizer.pad_token_id
    )
    if args.local_rank == -1:
        prompt_train_sampler = RandomSampler(prompt_train_dataset)
        if unsupervised_training_enabled:
            unsupervised_train_sampler = RandomSampler(
                unsupervised_train_dataset)
    else:
        prompt_train_sampler = DistributedSampler(prompt_train_dataset, shuffle=True)
        if unsupervised_training_enabled:
            unsupervised_train_sampler = DistributedSampler(
                unsupervised_train_dataset)
    prompt_train_dataloader = DataLoader(
        prompt_train_dataset,
        collate_fn=data_collator,
        sampler=prompt_train_sampler,
        drop_last=args.drop_last,
        batch_size=args.per_device_generation_batch_size)
    if unsupervised_training_enabled:
        unsupervised_train_dataloader = DataLoader(
            unsupervised_train_dataset,
            collate_fn=default_data_collator,
            sampler=unsupervised_train_sampler,
            drop_last=args.drop_last,
            batch_size=args.per_device_generation_batch_size)
    else:
        unsupervised_train_dataloader = [None] * len(
            prompt_train_dataloader)  # basically a dummy dataloader

    num_update_steps_per_epoch = min(len(prompt_train_dataloader), len(unsupervised_train_dataloader)) * \
        (args.per_device_generation_batch_size / args.per_device_train_batch_size) * \
        args.ppo_epochs / args.gradient_accumulation_steps
    num_total_iters = int(args.num_train_epochs * num_update_steps_per_epoch)

    return prompt_train_dataloader, unsupervised_train_dataloader, num_total_iters


def main():
    
    args = parse_args()
    args.local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)

    args.use_external_eval = False
    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.distributed.init_process_group(backend='nccl')
    args.global_rank = torch.distributed.get_rank()
    print("current process is {} of global rank {}".format(args.local_rank, args.global_rank))
    if args.enable_tensorboard and args.global_rank == 0:
        print(
            f"Tensorboard logs going to: {args.tensorboard_path}/step3_tensorboard_logs"
        )
        writer = SummaryWriter(f"{args.tensorboard_path}/step3_tensorboard_logs")
        
    # Validate settings
    if args.inference_tp_size > 1:
        assert (
            args.actor_zero_stage == 3
        ), "Zero stage 3 must be used to do Tensor sharding in the hybrid engine"

    if args.actor_zero_stage == 2 and args.critic_zero_stage == 2 and args.enable_hybrid_engine and args.offload and args.actor_lora_dim == 0:
        raise ValueError(
            "The combination of [actor_zero_stage==2, critic_zero_stage==2, enable_hybrid_engine=True, offload=True, lora=False] is currently unsupported due to training instability!"
        )
    
    if 'vicuna' in args.template_path:
        args.template_start_end = ['USER:','ASSISTANT:']
    elif 'alpaca' in args.template_path:
        args.template_start_end = ['Human:\n','Assistant:\n']
    print_rank_0("template_start_end and template_path is set as {} and {}".format(args.template_start_end, os.path.basename(args.template_path)), args.global_rank)
    
    # get unspuervised training dataset
    unsupervised_training_enabled = args.unsupervised_dataset_name and args.unsupervised_dataset_config_name
    if unsupervised_training_enabled:
        # if we enable unsupervised training, we need to double the batch size for actor model
        args.gradient_accumulation_steps_actor = args.gradient_accumulation_steps * 2
    else:
        args.gradient_accumulation_steps_actor = args.gradient_accumulation_steps
    # If passed along, set the training seed now.
    set_random_seed(args.seed)
    torch.distributed.barrier()

    # load_hf_tokenizer will get the correct tokenizer and set padding tokens based on the model family
    tokenizer = load_hf_tokenizer(args.actor_model_name_or_path, fast_tokenizer=True, use_unk=args.use_unk)
    if args.global_rank == 0:
        print("==========================")
        print("=> Step 3: init tokenizer: {}".format(tokenizer))
        print("=> Step 3: init tokenizer.pad_token_id: {}".format(tokenizer.pad_token_id))
        print("=> Step 3: init tokenizer.eos_token_id: {}".format(tokenizer.eos_token_id))
        print("=> Step 3: len(tokenizer): {}".format(len(tokenizer)))
        print("==========================\n\n")

    print(f"Begining to process training dataset with recipe path {args.recipe} !!!")    
    prompt_train_dataloader, unsupervised_train_dataloader, num_total_iters = create_datasets(
        args=args, tokenizer=tokenizer, train_phase=3)
        
    # RLHF engine is responsible for creating models, loading checkpoints, ds-initialize models/optims/lr-schedulers
    rlhf_engine = DeepSpeedRLHFEngine(
        actor_model_name_or_path=args.actor_model_name_or_path,
        critic_model_name_or_path=args.critic_model_name_or_path,
        tokenizer=tokenizer,
        num_total_iters=num_total_iters,
        args=args)

    # Mixed Precision ZeRO++
    if args.enable_mixed_precision_lora:
        assert args.actor_lora_dim > 0, "Mixed Precision LoRA requires LoRA to be enabled"
        assert args.actor_zero_stage == 3, "Mixed Precision LoRA requires Zero stage 3"
        rlhf_engine.actor.optimizer.quantize_nontrainable_params()
        print_rank_0("Mixed Precision ZeRO++ enabled")
        
    if args.actor_gradient_checkpointing:
        rlhf_engine.actor.gradient_checkpointing_enable()

    ppo_trainer = DeepSpeedPPOTrainerUnsupervised if unsupervised_training_enabled else DeepSpeedPPOTrainer
    trainer = ppo_trainer(rlhf_engine, args)

    # first number is how many experience-batch to generate, second number is the training batch size, which is the micro-batch size used
    exp_mini_dataset = MiniDataset(args.generation_batches,
                                   args.per_device_train_batch_size)
    unsup_mini_dataset = MiniDataset(args.generation_batches,
                                     args.per_device_train_batch_size)

    # Train!
    print_rank_0("***** Running training *****", args.global_rank)

    non_overflow_step_count = 0
    current_step = 0
    # Initial global step for checkpint saving.
    checkpoints_ratio = len(prompt_train_dataloader)
    ##########for reproduce hacking ###############
    args.num_train_epochs = 1
    if args.checkpoints_save_strategy == "num":
        checkpoints_ratio = args.num_train_epochs * len(prompt_train_dataloader) // args.checkpoints_save_num

    for epoch in range(args.num_train_epochs):
        print_rank_0(
            f"Beginning of Epoch {epoch+1}/{args.num_train_epochs}, Total Generation Batches {min(len(prompt_train_dataloader), len(unsupervised_train_dataloader))}",
            args.global_rank)
            
        for step, (batch_prompt, batch_unsupervised) in enumerate(
                zip(prompt_train_dataloader, unsupervised_train_dataloader)):
            # print('debug info: the shape of current promt is {}'.format(batch_prompt['prompt'].shape))
            if batch_prompt['prompt'].shape[0] == 1:
                print("Attention!!! the shape of current promt is 1")
                continue
            # print("+++++++++the type of promt is {}".format(batch_prompt['prompt'].dtype))
            current_step += 1
            batch_prompt = to_device(batch_prompt, device)
            # try:
            out = trainer.generate_experience(batch_prompt['prompt'].long(),
                                            batch_prompt['prompt_att_mask'].long(),
                                            current_step)

            del out['gold_score']
            print(f"generate_experience is finished at {current_step} at Line 188")
            # except Exception as e:
            #     print('debug info: Error occurs {}'.format(e))
            #     print('debug info: the shape of current promt is {}'.format(batch_prompt['prompt'].shape))
            #     continue
            training_start = time.time()
            if batch_unsupervised is not None:
                batch_unsupervised = to_device(batch_unsupervised, device)
                unsup_dataset = unsup_mini_dataset.add(batch_unsupervised)
            else:
                unsup_dataset = unsup_mini_dataset.add([[None] * args.per_device_generation_batch_size])
            try:
                exp_dataset = exp_mini_dataset.add(out)
            except Exception as e:
                print("debug info: Error {}".format_map(e))
                print('debug info: the shape of current promt is {}'.format(batch_prompt['prompt'].shape))
                continue

            if exp_dataset is not None:
                inner_iter = 0
                actor_loss_sum, critic_loss_sum, unsup_loss_sum, average_energy_loss, average_dst_loss, average_dst_actor, average_dst_ref= torch.tensor(0.).cuda(), torch.tensor(0.).cuda(), torch.tensor(0.).cuda(), torch.tensor(0.).cuda(), torch.tensor(0.).cuda(), torch.tensor(0.).cuda(), torch.tensor(0.).cuda()
                average_reward = 0
                average_length = 0
                random.shuffle(exp_dataset)
                random.shuffle(unsup_dataset)
                if args.actor_gradient_checkpointing:
                    rlhf_engine.actor.gradient_checkpointing_enable()
                # print(f"Now go to Line 205 with {current_step}")
                # for ppo_ep in range(args.ppo_epochs):
                # Modified by baorong in 2023/12/07 to support buffer experience training

                actor_averages_energy_list, ref_averages_energy_list = [], []
                for (exp_data, unsup_data) in zip(exp_dataset, unsup_dataset):
                    actor_loss, critic_loss = trainer.train_rlhf(exp_data)
                    actor_loss_sum += actor_loss.item()
                    critic_loss_sum += critic_loss.item()
                    average_reward += exp_data["ori_rewards"].mean()
                    average_length += exp_data["ans_length"].mean()
                    average_energy_loss += exp_data["energy_loss"].mean()
                    average_dst_loss += exp_data["dst_loss"].mean()
                    average_dst_actor += exp_data["dst_actor"].mean()
                    average_dst_ref += exp_data["dst_ref"].mean()
                    actor_averages_energy_list.append(exp_data['actor_averages_energy'].mean(dim=0))
                    ref_averages_energy_list.append(exp_data['ref_averages_energy'].mean(dim=0))
                    if unsupervised_training_enabled:
                        unsup_loss = trainer.train_unsupervised(unsup_data, args.unsup_coef)
                        unsup_loss_sum += unsup_loss.item()
                    inner_iter += 1
                    
                    if args.enable_ema:
                        moving_average(rlhf_engine.actor,
                                        rlhf_engine.actor_ema,
                                        zero_stage=args.actor_zero_stage)
                    random.shuffle(exp_dataset)
                    random.shuffle(unsup_dataset)
                    if inner_iter >= args.ppo_epochs:
                        break

                actor_averages_energy = torch.sum(torch.stack(actor_averages_energy_list), dim=0)
                ref_averages_energy = torch.sum(torch.stack(ref_averages_energy_list), dim=0)

                print_rank_0(
                    f'Epoch: {epoch} | Step: {current_step} | Actor Loss: {actor_loss_sum/inner_iter} | Critic Loss: {critic_loss_sum/inner_iter}', args.global_rank)
                average_reward = get_all_reduce_mean(average_reward).item()
                average_length = get_all_reduce_mean(average_length).item()
                average_energy_loss = get_all_reduce_mean(average_energy_loss).item()
                average_dst_loss = get_all_reduce_mean(average_dst_loss).item()
                average_dst_actor = get_all_reduce_mean(average_dst_actor).item()
                average_dst_ref = get_all_reduce_mean(average_dst_ref).item()
                try:
                    # if not calculate actor_averages_energy, the following commands would report error!!!
                    actor_averages_energy = get_all_reduce_mean(actor_averages_energy).cpu().numpy()
                    ref_averages_energy = get_all_reduce_mean(ref_averages_energy).cpu().numpy()
                    print("the shape of actor_averages_energy is {}".format(actor_averages_energy.shape))
                except:
                    actor_averages_energy = -100
                    ref_averages_energy = -100



                print_rank_0(
                    f"Average reward score: {average_reward/inner_iter}, average answer length {average_length/inner_iter}, average energy loss {average_energy_loss/inner_iter}, diff reward score: {(average_reward/inner_iter) - args.energy_loss_co * (average_energy_loss/inner_iter)}, average dst loss {average_dst_loss/inner_iter}",
                    args.global_rank)

                print_rank_0(
                    "-------------------------------------------------------------------------------------",
                    args.global_rank)
                # NOTE: tensorboard logging of step-wise average statistics
                actor_loss_avg = get_all_reduce_mean(actor_loss_sum).item()
                critic_loss_avg = get_all_reduce_mean(critic_loss_sum).item()
                if unsupervised_training_enabled:
                    unsup_loss_avg = get_all_reduce_mean(unsup_loss_sum).item()

                if args.enable_tensorboard and torch.distributed.get_rank() == 0:
                    print("=> Writing step-wise tensorboard information...")
                    writer.add_scalar('train-by-step/reward', average_reward / inner_iter, global_step=current_step)
                    writer.add_scalar('train-by-step/ans_length', average_length / inner_iter, global_step=current_step)
                    writer.add_scalar('train-by-step/energy_loss', average_energy_loss / inner_iter, global_step=current_step)
                    writer.add_scalar('train-by-step/dst_loss', average_dst_loss / inner_iter, global_step=current_step)
                    writer.add_scalar("train-by-step/actor_lr", rlhf_engine.actor_engine.optimizer.get_lr(), global_step=current_step)
                    writer.add_scalar("train-by-step/critic_lr", rlhf_engine.critic_engine.optimizer.get_lr(), global_step=current_step)
                    writer.add_scalar("train-by-step/avg_actor_loss", actor_loss_avg/inner_iter, global_step=current_step)
                    writer.add_scalar("train-by-step/avg_critic_loss", critic_loss_avg/inner_iter, global_step=current_step)
                    if unsupervised_training_enabled:
                        writer.add_scalar("train-by-step/unsuper_loss", unsup_loss_avg/inner_iter, global_step=current_step)
                    abs_mean_kl, meal_kl = trainer.get_kl()
                    writer.add_scalar("train-by-step/kl_divergence", meal_kl, global_step=current_step)
                    print("=> Finished")
                    
                    writer.flush()
            
            if args.actor_gradient_checkpointing:
                rlhf_engine.actor.gradient_checkpointing_disable()
            
            
            if current_step % checkpoints_ratio == 0:
                ... # save model
              

if __name__ == "__main__":
    main()

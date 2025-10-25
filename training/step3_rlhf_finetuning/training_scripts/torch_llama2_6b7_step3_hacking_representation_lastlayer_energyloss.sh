#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
NNODES=$1
GPUS=$2
Actor_Model=$3
Critic_Model=$4
Actor_Lr=$5
Critic_Lr=$6
BS=$7
KLCTL=$8
OUTPUT=$9
PREFIX=${10}
DIM_FACTOR=${11}
FACTOR_NUM=${12}
FACTOR_WIDTH=${13}
recipe_path=${14}
MASTER_PORT=${15}
Gold_Model=${16}
lr_scheduler_type=${17}
epoch=${18}
energy_loss_ta=${19}
energyloss_co=${20}
energy_loss_clip_ratio=${21}

recipename=$(basename "$recipe_path")
recipe="${recipename%.json}"


if [ "$NNODES" == "1" ]; then
    MASTER_ADDR=localhost
    RANK=0
fi

if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output_step3_llama
fi


if [ "$PREFIX" == "" ]; then
    PREFIX=llama2_7b_rlhf
fi
mkdir -p $OUTPUT/$PREFIX

if [ "$MASTER_PORT" == "" ]; then
    MASTER_PORT=6669
fi

if [ "$lr_scheduler_type" == "" ]; then
    lr_scheduler_type=cosine
fi

if [ "$DIM_FACTOR" == "" ]; then
    DIM_FACTOR=-1
fi

if [ "$FACTOR_NUM" == "" ]; then
    FACTOR_NUM=-1
fi

if [ "$NNODES" == "2" ]; then
    Actor_Lr=$(awk "BEGIN {print $Actor_Lr * sqrt(2)}")
    Critic_Lr=$(awk "BEGIN {print $Critic_Lr * sqrt(2)}")
fi

if [ "$NNODES" == "4" ]; then
    Actor_Lr=$(awk "BEGIN {print $Actor_Lr * 2}")
    Critic_Lr=$(awk "BEGIN {print $Critic_Lr * 2}")
fi

if [ "$epoch" == "" ]; then
    epoch=1
fi

echo "After Adjusting with "${NNODES}" NNODES, Actor_Lr is "$Actor_Lr" and Critic_Lr is "${Critic_Lr}
torchrun --nproc_per_node  $GPUS --nnodes $NNODES --master_addr $MASTER_ADDR --master_port $MASTER_PORT --node_rank $RANK ./training/step3_rlhf_finetuning/llama_step3_hacking_representation.py \
   --data_path ours/sample_alignment_stage2 \
   --data_split 0,0,10 \
   --prompt_input \
   --recipe $recipe_path \
   --data_output_path ./cache/llama2_only_${recipe}_vicuna \
   --end_of_conversation_token "</s>" \
   --actor_model_name_or_path ${Actor_Model} \
   --critic_model_name_or_path ${Critic_Model} \
   --template_path ./training/step3_rlhf_finetuning/templates/vicuna.json \
   --use_unk \
   --num_padding_at_beginning 0 \
   --only_reward_final_token \
   --per_device_generation_batch_size $BS \
   --per_device_train_batch_size $BS \
   --generation_batches 1 \
   --ppo_epochs 1 \
   --max_seq_len 1024 \
   --max_answer_seq_len 512 \
   --actor_learning_rate ${Actor_Lr} \
   --critic_learning_rate ${Critic_Lr} \
   --kl_ctl_weight $KLCTL \
   --actor_weight_decay 0.1 \
   --critic_weight_decay 0.1 \
   --num_train_epochs ${epoch} \
   --lr_scheduler_type  $lr_scheduler_type\
   --gradient_accumulation_steps 1 \
   --actor_gradient_checkpointing \
   --critic_gradient_checkpointing \
   --offload_reference_model \
   --actor_dropout 0.0 \
   --num_warmup_steps 10 \
   --deepspeed \
   --no_save_afs \
   --seed 42 \
   --dtype bf16 \
   --output_dir $OUTPUT/$PREFIX \
   --tensorboard_path $OUTPUT/$PREFIX \
   --checkpoints_save_strategy num \
   --checkpoints_save_num 5 \
   --enable_tensorboard \
   --release_inference_cache \
   --actor_zero_stage 2 \
   --critic_zero_stage 3 \
   --dim_factor $DIM_FACTOR \
   --factor_num $FACTOR_NUM \
   --factor_width $FACTOR_WIDTH \
   --energy_loss_ta ${energy_loss_ta} \
   --energy_loss_co ${energyloss_co} \
   --energy_loss_clip_ratio ${energy_loss_clip_ratio} 2>&1 | tee -a $OUTPUT/$PREFIX/$HOSTNAME-training.log

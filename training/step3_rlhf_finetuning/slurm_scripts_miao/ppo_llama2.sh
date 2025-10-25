export TOKENIZERS_PARALLELISM=true
export OMP_NUM_THREADS=8
export MAX_JOBS=32
export NCCL_NET_PLUGIN=none
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_ENABLE=1
export NCCL_P2P_ENABLE=1

WORKSPACE=/path/to/Energy_Loss_Phenomenon
cd $WORKSPACE

actor_model=/path/to/SFT/model
critic_model=/path/to/RM/model
gold_model=None

recipe_path=./training/step3_rlhf_finetuning/recipes/recipe_version1.json
recipename=$(basename "$recipe_path")
recipe="${recipename%.json}"
actor_lr=5e-7
critic_lr=1e-6
bs=8
klctl=0
dim_factor=-1
factor_num=-1
factor_width=-1
MASTER_PORT=6768
scheduler=cosine
epoch=5
node=1
output_dir="./output/real_vicuna_"${recipe}"_hacking_representation"
mkdir $output_dir
prefix_task_name="llama2_7b_klctl"$klctl"_actor_sharegpt2e-5_"$(basename $actor_model)"_lr"$actor_lr"_criticreal_hhtradeoffv1_lr"$critic_lr"_bs"$bs"_"$recipe"_"${scheduler}"_epoch"${epoch}"_node"${node}"_max_length5"

echo $prefix_task_name
echo actor_model is from ${actor_model}
echo critic model is from ${critic_model}
echo prefix is ${prefix_task_name}
bash training_scripts/vicuna/torch_llama2_6b7_step3_hacking_representations.sh ${node} 8 ${actor_model} ${critic_model} $actor_lr $critic_lr $bs $klctl ${output_dir} ${prefix_task_name} ${dim_factor} ${factor_num} ${factor_width} ${recipe_path} ${MASTER_PORT} ${gold_model} ${scheduler} ${epoch} ${length_penalty_N}
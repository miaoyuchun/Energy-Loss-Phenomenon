import argparse
import os
import deepspeed
from transformers import SchedulerType

def parse_args(extra_args_provider=None, ignore_unknown_args=False):
    """Parse all arguments."""
    parser = argparse.ArgumentParser(description='Deepspeed-Chat Arguments', allow_abbrev=False)
    # Standard arguments.
    parser = _add_initialization_args(parser)
    parser = _add_reward_args(parser)
    parser = _add_regularization_args(parser)
    parser = _add_training_args(parser)
    parser = _add_learning_rate_args(parser)
    parser = _add_checkpointing_args(parser)
    parser = _add_distributed_args(parser)
    parser = _add_validation_args(parser)
    parser = _add_data_args(parser)
    parser = _add_logging_args(parser)
    parser = _add_zero_args(parser)
    parser = _add_activation_checkpoint_args(parser)
    # Add custom arguments by baorong in 2023-09-14
    parser = _add_supervise_finetune_args(parser)
    parser = _add_rlhf_args(parser)
    parser = _add_ib_args(parser)
    parser = _add_dpo_args(parser)
    parser = _add_reset_args(parser)
    

    # Custom arguments.
    if extra_args_provider is not None:
        parser = extra_args_provider(parser)

    parser = deepspeed.add_config_arguments(parser)

    # Parse.
    if ignore_unknown_args:
        args, _ = parser.parse_known_args()
    else:
        args = parser.parse_args()

    # Args from environment
    args.rank = int(os.getenv('RANK', '0'))
    args.world_size = int(os.getenv("WORLD_SIZE", '1'))
    args.gold_reward_model = eval(args.gold_reward_model)
    args.reset = eval(args.reset)
    if args.eval_recipe != None and args.eval_data_path != None and args.eval_data_split != None:
        args.use_external_eval = True
    else:
        args.use_external_eval = False
    print("use_external_eval is {}".format(args.use_external_eval))
    if args.special_layer_weight == None:
        args.special_layer_weight = []
        args.special_lr_multiplier = 1
    else:
        exit()

    if args.energy_loss_ta == 'v1':
        print(f"energy_loss v1: last mlp block energyloss is calculated with coefficient {args.energy_loss_co} and energy_loss_clip_ratio {args.energy_loss_clip_ratio}!!!")
    elif args.energy_loss_ta == 'v2':
        print(f"energy_loss v2: last three mlp layers energyloss is calculated with coefficient {args.energy_loss_co} and energy_loss_clip_ratio {args.energy_loss_clip_ratio}!!!")
    return args

def _add_ib_args(parser):
    group = parser.add_argument_group(title='informarion bottleneck')
    group.add_argument('--use_unk',  action='store_true', help='if use unk as pad token')
    group.add_argument('--use_variational_inference',  action='store_true', help='if use the variation inference')
    group.add_argument('--use_encoder_sigmoid',  action='store_true', help='if use the sigmoid at encoder')
    group.add_argument('--use_simcse',  action='store_true', help='if use the use_simcse')
    group.add_argument('--use_decoder_sigmoid',  action='store_true', help='if use the sigmoid at decoder')
    group.add_argument('--beta', type=float, default=0.5, help='coefficient of the KL regularization')
    group.add_argument('--gamma', type=float, default=0.1, help='coefficient of the conditional loss')
    group.add_argument('--lamda', type=float, default=0.1, help='coefficient of the internal entropy')
    group.add_argument('--omega', type=float, default=0.1, help='coefficient of the simcse')
    group.add_argument('--fast_version', action='store_true', help='whether use fast_version')
    group.add_argument('--no_sample', action='store_true', help='whether use fast_version')
    group.add_argument('--complex_fcn_decode', action='store_true', help='whether use complex_fcn_decode')
    group.add_argument('--introduce_supervision', action='store_true', help='whether use introduce_supervision')
    group.add_argument('--internal_entropy', action='store_true', help='whether use internal_entropy')
    group.add_argument('--factor_num', type=int, default=-1, help='the number of factors')
    group.add_argument('--dim_factor', type=int, default=-1, help='dimensions per factor')
    group.add_argument('--factor_width', type=int, default=-1, help='how many dimensions conresponde to a factor')
    group.add_argument('--drop_last', action='store_true', help='whether drop last in the dataloader')
    group.add_argument('--reverse', action='store_true', help='whether reverse in the dataloader')
    group.add_argument('--sampler_seed', type=int, default=100, help='random seed for PrompDataset')
    group.add_argument('--lastlayer_dropout', type=float, default=0.0 , help='dropout parameter for the last layer')
    group.add_argument('--special_lr_multiplier', type=float, default=1.0 , help='lr for some special layer')
    group.add_argument('--special_layer_weight', type=str, default=None , help='v1: only model.layers.31.mlp.down_proj.weight')
    group.add_argument('--energy_loss_ta', type=str, default=None , help='v1: mlpblock v2 mlplayers')
    group.add_argument('--energy_loss_co', type=float, default=0, help='coefficient of the enerfy loss')
    group.add_argument('--energy_loss_clip_ratio', type=float, default=0.0, help='ratio of max energy loss compared to reward value')
    group.add_argument('--rmenaemble_uwo', type=float, default=0.0, help='ratio of uncertainty in uwo')
    group.add_argument('--length_penalty_N', type=int, default=0, help='parameter N for length penalty')
    group.add_argument('--dst_penalty_co', type=float, default=0, help='coefficient of the dst penalty')
    group.add_argument('--use_dst_penalty',action='store_true', help='whether use the dst penalty')

    return parser


def _add_dpo_args(parser):
    group = parser.add_argument_group(title='reward mdoel')
    group.add_argument('--beta_dpo', type=float, default=0.1, help='dpo parameter')
    group.add_argument('--ftx_gamma', type=float, default=0.0, help='ftx_gamma parameter')
    group.add_argument('--reference_zero_stage', type=int, default=3, help='the zero stage of reference model')
    group.add_argument('--loss_type', type=str, default="sigmoid", help='loss_type of dpo loss function')
    return parser


def _add_reset_args(parser):
    group = parser.add_argument_group(title='reset')
    group.add_argument('--reset_step', type=int, default=100, help='step to reset')
    group.add_argument('--reset_type', type=str, default='reset_all_attention_5', help='reset_type')
    group.add_argument('--reset_module', type=str, default='actor', help='reset_type')
    group.add_argument('--reset', type=str, default='False', help='whether reset')
    return parser


def _add_reward_args(parser):
    group = parser.add_argument_group(title='reward mdoel')
    group.add_argument('--num_padding_at_beginning', type=int, default=0,
                    help='number of pad tokens at begining')
    group.add_argument('--aoss_dir', type=str, default=None,
                    help='number of pad tokens at begining')
    return parser

def _add_initialization_args(parser):
    group = parser.add_argument_group(title='initialization')

    group.add_argument('--seed', type=int, default=42,
                       help='Random seed used for python, numpy, '
                       'pytorch, and cuda.')
    group.add_argument('--data-parallel-random-init', action='store_true',
                       help='Enable random initialization of params '
                       'across data parallel ranks')
    group.add_argument('--init-method-std', type=float, default=0.02,
                       help='Standard deviation of the zero mean normal '
                       'distribution used for weight initialization.')
    group.add_argument('--init-method-xavier-uniform', action='store_true',
                       help='Enable Xavier uniform parameter initialization')

    return parser

def _add_logging_args(parser):
    group = parser.add_argument_group(title='logging')
    group.add_argument('--print_loss', action='store_true',
                       help='Prints loss at each step.')
    group.add_argument('--enable_tensorboard', action='store_true',
                        help='Enable tensorboard logging')
    group.add_argument('--tensorboard_path', type=str,
                        default="step1_tensorboard")
    parser.add_argument('--print_answers', action='store_true',
                        help='Print prompt and answers during training')
    return parser


def _add_regularization_args(parser):
    group = parser.add_argument_group(title='regularization')

    group.add_argument('--attention-dropout', type=float, default=0.1,
                       help='Post attention dropout probability.')
    group.add_argument('--hidden-dropout', type=float, default=0.1,
                       help='Dropout probability for hidden state transformer.')
    group.add_argument('--weight-decay', type=float, default=0.01,
                       help='Weight decay coefficient for L2 regularization.')
    group.add_argument('--start-weight-decay', type=float,
                       help='Initial weight decay coefficient for L2 regularization.')
    group.add_argument('--end-weight-decay', type=float,
                       help='End of run weight decay coefficient for L2 regularization.')
    group.add_argument('--weight-decay-incr-style', type=str, default='constant',
                       choices=['constant', 'linear', 'cosine'],
                       help='Weight decay increment function.')
    group.add_argument('--clip-grad', type=float, default=1.0,
                       help='Gradient clipping based on global L2 norm.')
    group.add_argument('--adam-beta1', type=float, default=0.9,
                       help='First coefficient for computing running averages '
                       'of gradient and its square')
    group.add_argument('--adam-beta2', type=float, default=0.999,
                       help='Second coefficient for computing running averages '
                       'of gradient and its square')
    group.add_argument('--adam-eps', type=float, default=1e-08,
                       help='Term added to the denominator to improve'
                       'numerical stability')
    group.add_argument('--sgd-momentum', type=float, default=0.9,
                       help='Momentum factor for sgd')

    return parser


def _add_training_args(parser):
    group = parser.add_argument_group(title='training')
    group.add_argument("--model_name_or_path", type=str,
                       help="Path to pretrained model or model identifier from huggingface.co/models.")
    group.add_argument("--per_device_train_batch_size", type=int, default=16,
                       help="Batch size (per device) for the training dataloader.")
    group.add_argument("--per_device_eval_batch_size", type=int, default=16, 
                       help="Batch size (per device) for the evaluation dataloader.")
    group.add_argument("--max_seq_len", type=int, default=512,
                       help="The maximum sequence length.")
    group.add_argument("--weight_decay", type=float,default=0.,
                        help="Weight decay to use.")
    group.add_argument("--num_train_epochs", type=int, default=1,
                        help="Total number of training epochs to perform.")
    group.add_argument("--gradient_accumulation_steps", type=int,default=1,
                       help="Number of updates steps to accumulate before performing a backward/update pass.")
    group.add_argument("--lr_scheduler_type", type=SchedulerType, default="cosine",
                       help="The scheduler type to use.",
                       choices=[
                            "linear", "cosine", "cosine_with_restarts", "polynomial",
                            "constant", "constant_with_warmup"
                        ])
    group.add_argument('--use_flash_attn', action='store_true',
                       help='use FlashAttention implementation of attention. '
                       'https://arxiv.org/abs/2205.14135')
    group.add_argument("--num_warmup_steps", type=int, default=0,
                       help="Number of steps for the warmup in the lr scheduler.")
    group.add_argument('--gradient_checkpointing', action='store_true',
                        help='Enable HF gradient checkpointing for model.')
    group.add_argument("--dropout", type=float, default=None,
                       help="If dropout configured, use it. "
                       "Otherwise, keep the default dropout configuration of the model.")
    group.add_argument("--lora_dim", type=int, default=0,
                       help="If > 0, use LoRA for efficient training.")
    group.add_argument("--lora_module_name", type=str, default="decoder.layers.",
                        help="The scope of LoRA.")
    group.add_argument('--only_optimize_lora', action='store_true',
                        help='Only optimize the LoRA parameters.')
    group.add_argument("--lora_learning_rate", type=float, default=5e-4,
                       help="Initial LoRA learning rate (after the potential warmup period) to use.")
    group.add_argument('--compute_fp32_loss', action='store_true',
                       help='Relevant for low precision dtypes (fp16, bf16, etc.). '
                       'If specified, loss is calculated in fp32.')
    return parser


def _add_learning_rate_args(parser):
    group = parser.add_argument_group(title='learning rate')
    group.add_argument("--learning_rate", type=float, default=1e-3,
                       help="Initial learning rate (after the potential warmup period) to use.")
    group.add_argument('--lr-decay-style', type=str, default='linear',
                       choices=['constant', 'linear', 'cosine', 'inverse-square-root'],
                       help='Learning rate decay function.')
    group.add_argument('--lr-decay-iters', type=int, default=None,
                       help='number of iterations to decay learning rate over,'
                       ' If None defaults to `--train-iters`')
    group.add_argument('--lr-decay-samples', type=int, default=None,
                       help='number of samples to decay learning rate over,'
                       ' If None defaults to `--train-samples`')
    group.add_argument('--lr-decay-tokens', type=int, default=None,
                       help='number of tokens to decay learning rate over,'
                       ' If not None will override iter/sample-based decay')
    group.add_argument('--lr-warmup-fraction', type=float, default=None,
                       help='fraction of lr-warmup-(iters/samples) to use '
                       'for warmup (as a float)')
    group.add_argument('--lr-warmup-iters', type=int, default=0,
                       help='number of iterations to linearly warmup '
                       'learning rate over.')
    group.add_argument('--lr-warmup-samples', type=int, default=0,
                       help='number of samples to linearly warmup '
                       'learning rate over.')
    group.add_argument('--lr-warmup-tokens', type=int, default=None,
                       help='number of tokens to linearly warmup '
                       'learning rate over.')
    group.add_argument('--warmup', type=int, default=None,
                       help='Old lr warmup argument, do not use. Use one of the'
                       '--lr-warmup-* arguments above')
    group.add_argument('--min-lr', type=float, default=0.0,
                       help='Minumum value for learning rate. The scheduler'
                       'clip values below this threshold.')
    group.add_argument('--override-opt_param-scheduler', action='store_true',
                       help='Reset the values of the scheduler (learning rate,'
                       'warmup iterations, minimum learning rate, maximum '
                       'number of iterations, and decay style from input '
                       'arguments and ignore values from checkpoints. Note'
                       'that all the above values will be reset.')
    group.add_argument('--use-checkpoint-opt_param-scheduler', action='store_true',
                       help='Use checkpoint to set the values of the scheduler '
                       '(learning rate, warmup iterations, minimum learning '
                       'rate, maximum number of iterations, and decay style '
                       'from checkpoint and ignore input arguments.')
    return parser


def _add_checkpointing_args(parser):
    group = parser.add_argument_group(title='checkpointing')
    group.add_argument('--oss_bucket_prefix', type=str, default=None,
                       help='oss prefix to save checkpoints to.')
    group.add_argument('--oss_output_dir', type=str, default=None,
                       help='Directory containing a model checkpoint.')
    group.add_argument('--checkpoints_save_strategy', type=str, default='epoch',
                       help='oss prefix to save checkpoints to.')
    group.add_argument('--output_dir', type=str, default=None,
                       help='Output directory to save checkpoints to.')
    group.add_argument('--save_interval', type=int, default=None,
                       help='Number of iterations between checkpoint saves.')
    group.add_argument('--checkpoints_save_num', type=int, default=3,
                       help='Number of iterations between checkpoint saves.')
    group.add_argument('--no_save_afs', action='store_true', default=None,
                       help='Do not save current optimizer.')
    group.add_argument('--no-save-rng', action='store_true', default=None,
                       help='Do not save current rng state.')
    return parser


def _add_rlhf_args(parser):
    group = parser.add_argument_group(title='(Step 3) RLHF training arguments')
    group.add_argument('--use_sigmoid', action='store_true', 
                       help='Do not save current rng state.')
    group.add_argument('--use_reward_norm', action='store_true', default=None,
                       help='Do not save current rng state.')
    group.add_argument('--use_reward_scaling', action='store_true', default=None,
                       help='Do not save current rng state.')
    group.add_argument('--only_reward_final_token', action='store_true',
                       help='Do not save current rng state.')
    group.add_argument('--only_update_vhead', action='store_true', default=None,
                       help='Do not save current rng state.')
    group.add_argument("--unsupervised_dataset_name", type=str, default=None,
                       help="The name of the dataset to use (via the datasets library).")
    group.add_argument("--unsupervised_dataset_config_name", type=str, default=None,
                       help="The configuration name of the dataset to use (via the datasets library).")
    group.add_argument("--unsup_coef", type=float, default=27.8,
                        help='''gamma in Equation 2 from InstructGPT paper''')
    ## Make parameters for VLLM fast inference
    group.add_argument("--actor_model_name_or_path", type=str,
                       help="Path to pretrained model or model identifier from huggingface.co/models.")
    ## Used for ensemble
    group.add_argument("--critic_model_name_or_path", type=str,
                       help="Path to pretrained model or model identifier from huggingface.co/models.")
    group.add_argument("--critic_model_name_or_path1", type=str, default=None,
                       help="Path to pretrained model or model identifier from huggingface.co/models.")
    group.add_argument("--critic_model_name_or_path2", type=str, default=None,
                       help="Path to pretrained model or model identifier from huggingface.co/models.")
    group.add_argument("--critic_model_name_or_path3", type=str, default=None,
                       help="Path to pretrained model or model identifier from huggingface.co/models.")
    ## 
    group.add_argument("--generation_batches", type=int, default=1,
                        help="Generate x batches to go to training mode.")
    group.add_argument("--per_device_generation_batch_size", type=int, default=16,
                       help="Batch size (per device) for the training dataloader and generation purpose.")
    group.add_argument("--ppo_epochs", type=int, default=1,
                       help="For generated data, how many ppo training epochs to run.")
    group.add_argument("--max_prompt_seq_len", type=int, default=2048,
                       help="The maximum sequence length.")
    group.add_argument("--max_answer_seq_len", type=int, default=256,
                        help="The maximum sequence length.")
    group.add_argument("--top_p", type=float, default=0.9, 
                       help="Float that controls the cumulative probability of the top tokens")
    group.add_argument("--top_k", type=int, default=5, 
                       help="Integer that controls the number of top tokens to consider. Set to -1 to consider all tokens.")
    group.add_argument("--temperature", type=float, default=1.0, 
                       help="Float that controls the randomness of the sampling. Lower values make the model more deterministic, while higher values make the model more random. Zero means greedy sampling.")
    group.add_argument("--frequency_penalty", type=float,default=0.0,
                       help="Float that penalizes new tokens based on their frequency in the generated text so far. Values > 0 encourage the model to use new tokens, while values < 0 encourage the model to repeat tokens.")
    group.add_argument("--repetition_penalty", type=float,default=1.0,
                       help="Float that penalizes new tokens based on whether they appear in the prompt and the generated text so far. Values > 1 encourage the model to use new tokens, while values < 1 encourage the model to repeat tokens.")
    group.add_argument("--length_penalty", type=float,default=1.0,
                       help="Float that penalizes sequences based on their length. Used in beam search.")
    group.add_argument("--gpu_memory_utilization", type=float,default=0.5,
                       help="Make gpu save memory for vllm, which differs from model size!")
    group.add_argument("--actor_learning_rate", type=float,default=9.65e-6,
                       help="Initial learning rate (after the potential warmup period) to use.")
    group.add_argument("--critic_learning_rate", type=float, default=5e-6,
                       help="Initial learning rate (after the potential warmup period) to use.")
    group.add_argument("--actor_weight_decay", type=float, default=0.,
                        help="Weight decay to use.")
    group.add_argument("--critic_weight_decay", type=float, default=0.,
                        help="Weight decay to use.")
    # DeepSpeed
    group.add_argument("--enable_hybrid_engine", action='store_true',
                       help="Enable hybrid engine for actor model to optimize both \
                        inference and training through DeepSpeed.")
    group.add_argument("--unpin_actor_parameters", action='store_true',
                       help="Unpin actor's parameters during generation. \
                        This makes generation slower but requires less memory.")
    group.add_argument("--release_inference_cache",action='store_true',
                       help="Release the memory cache used for inference. \
                        This makes generation preparation slower but might increase e2e throughput by using larger batch size.")
    group.add_argument("--inference_tp_size", type=int, default=1,
                       help="Tensor-parallelism degree used for the inference-optimization. \
                        Please note hybrid-engine need to be enabled when using this feature.")
    group.add_argument("--tp_gather_partition_size", type=int,default=1,
                       help="Granularity to bring in layers for TP sharding inside the hybrid engine. \
                        Please note hybrid-engine and tp_inference_size > 1 need to be true when using this feature.")
    group.add_argument('--offload_reference_model', action='store_true',
                       help='Enable ZeRO Offload techniques for reference model')
    group.add_argument('--actor_zero_stage', type=int, default=0,
                       help='ZeRO optimization stage for Actor model (and clones).')
    group.add_argument('--critic_zero_stage', type=int, default=0,
                       help='ZeRO optimization stage for Critic model (and reward).')
    group.add_argument('--actor_gradient_checkpointing', action='store_true',
                       help='Enable HF gradient checkpointing for Actor model.')
    group.add_argument('--critic_gradient_checkpointing', action='store_true',
                       help='Enable HF gradient checkpointing for Critic model.')
    group.add_argument("--actor_dropout", type=float, default=None,
                       help="If actor dropout configured, use it. "
                       "Otherwise, keep the default dropout configuration of the actor model.")
    group.add_argument("--critic_dropout", type=float, default=None,
                       help="If critic dropout configured, use it. "
                       "Otherwise, keep the default dropout configuration of the critic model.")
    ## LoRA for efficient training setting
    group.add_argument("--actor_lora_dim", type=int, default=0,
                        help="If > 0, use LoRA for efficient training.")
    group.add_argument("--actor_lora_module_name", type=str, default="decoder.layers.",
                        help="The scope of LoRA.")
    group.add_argument("--critic_lora_dim", type=int, default=0,
                        help="If > 0, use LoRA for efficient training.")
    group.add_argument("--critic_lora_module_name", type=str, default="decoder.layers.",
                        help="The scope of LoRA.")
    group.add_argument("--actor_lora_learning_rate", type=float, default=5e-4,
                        help="Initial actor LoRA learning rate (after the potential warmup period) to use.")
    group.add_argument("--critic_lora_learning_rate", type=float, default=5e-4,
                        help="Initial critic LoRA learning rate (after the potential warmup period) to use.")
    ## Make EMA as an optional feature
    group.add_argument('--enable_ema', action='store_true',
                        help='Enable EMA checkpoint for the model.')
    ## Mixed Precision ZeRO++
    group.add_argument('--enable_mixed_precision_lora', action='store_true',
                        help='Enable Mixed Precision ZeRO++ for training and generation.')
    ## Actor/critic model overflow alignment
    group.add_argument('--align_overflow', action='store_true',
                        help='Align loss scale overflow between actor and critic')
    ## Testing
    group.add_argument('--enable_test_mode', action='store_true',
                        help='Enable a testing mode that terminates training based on args.test_stop_step')
    group.add_argument("--test_stop_step", type=int, default=0,
                        help="Training non-overflow step at which to terminate training during testing.")
    group.add_argument("--kl_ctl_weight", 
                        type=float, 
                        default=0.02,
                        help="the weight of kl_ctl used in ppo_trainer.py")
    group.add_argument("--gold_reward_model", type=str, default='False', help="if the gold_reward_model is utilized !")
    group.add_argument("--gold_reward_name_or_path", type=str, default='', help="the path of gold model path")
    group.add_argument('--gold_reward_zero_stage', type=int, default=0, help='ZeRO optimization stage for Actor model (and clones).')
    
    return parser


def _add_distributed_args(parser):
    group = parser.add_argument_group(title='distributed')

    group.add_argument('--tensor-model-parallel-size', type=int, default=1,
                       help='Degree of tensor model parallelism.')
    group.add_argument('--enable-expert-tensor-parallelism', action='store_true',
                        default=False,
                        help="use tensor parallelism for expert layers in MoE")
    group.add_argument('--pipeline-model-parallel-size', type=int, default=1,
                       help='Degree of pipeline model parallelism.')
    group.add_argument('--pipeline-model-parallel-split-rank',
                       type=int, default=None,
                       help='Rank where encoder and decoder should be split.')
    group.add_argument('--moe-expert-parallel-size', type=int, default=1,
                       help='Degree of the MoE expert parallelism.')
    group.add_argument('--num-layers-per-virtual-pipeline-stage', type=int, default=None,
                       help='Number of layers per virtual pipeline stage')
    group.add_argument('--overlap-p2p-communication',
                       action='store_true',
                       help='overlap pipeline parallel communication with forward and backward chunks',
                       dest='overlap_p2p_comm')
    group.add_argument('--distributed-backend', default='nccl',
                       choices=['nccl', 'gloo', 'ccl'],
                       help='Which backend to use for distributed training.')
    group.add_argument('--distributed-timeout-minutes', type=int, default=10,
                       help='Timeout minutes for torch.distributed.')
    group.add_argument('--DDP-impl', default='local',
                       choices=['local', 'torch', 'FSDP'],
                       help='which DistributedDataParallel implementation '
                       'to use.')
    group.add_argument('--no-contiguous-buffers-in-local-ddp',
                       action='store_false', help='If set, dont use '
                       'contiguous buffer in local DDP.',
                       dest='use_contiguous_buffers_in_local_ddp')
    group.add_argument('--no-scatter-gather-tensors-in-pipeline', action='store_false',
                       help='Use scatter/gather to optimize communication of tensors in pipeline',
                       dest='scatter_gather_tensors_in_pipeline')
    group.add_argument('--use-ring-exchange-p2p', action='store_true',
                       default=False, help='If set, use custom-built ring exchange '
                       'for p2p communications. Note that this option will require '
                       'a custom built image that support ring-exchange p2p.')
    group.add_argument('--local_rank', type=int, default=-1,
                       help='local rank passed from distributed launcher.')
    group.add_argument('--global_rank', type=int, default=-1,
                       help='global rank passed from distributed launcher.')
    group.add_argument('--lazy-mpu-init', type=bool, required=False,
                       help='If set to True, initialize_megatron() '
                       'skips DDP initialization and returns function to '
                       'complete it instead.Also turns on '
                       '--use-cpu-initialization flag. This is for '
                       'external DDP manager.' )
    group.add_argument('--use-cpu-initialization', action='store_true',
                       default=None, help='If set, affine parallel weights '
                       'initialization uses CPU' )
    group.add_argument('--empty-unused-memory-level', default=0, type=int,
                       choices=[0, 1, 2],
                       help='Call torch.cuda.empty_cache() each iteration '
                       '(training and eval), to reduce fragmentation.'
                       '0=off, 1=moderate, 2=aggressive.')
    group.add_argument('--standalone-embedding-stage', action='store_true',
                       default=False, help='If set, *input* embedding layer '
                       'is placed on its own pipeline stage, without any '
                       'transformer layers. (For T5, this flag currently only '
                       'affects the encoder embedding.)')
    group.add_argument('--use-distributed-optimizer', action='store_true',
                       help='Use distributed optimizer.')

    return parser


def _add_validation_args(parser):
    group = parser.add_argument_group(title='validation')
    group.add_argument('--eval_iters', type=int, default=5,
                       help='Number of iterations to run for evaluation'
                       'validation/test for.')
    group.add_argument('--eval-interval', type=int, default=1000,
                       help='Interval between running evaluation on '
                       'validation set.')
    group.add_argument('--skip_eval', action='store_true',
                       default=False, help='If set, bypass the training loop, '
                       'optionally do evaluation for validation/test, and exit.')

    return parser


def _add_data_args(parser):
    group = parser.add_argument_group(title='data and dataloader')
    # Below are adding by baorong in 2023/08/29 to support training one epoch.
    
    group.add_argument('--data_home', type=str, default=None,
                       help='Data Home for all datasets root path.')
    # End of adding by baorong in 2023/08/29 to support training one epoch.
    group.add_argument('--aml-data-download-path', type=str, default=None,
                       help='Path to mounted input dataset')
    group.add_argument('--data-path', nargs='*', default=None,
                       help='Path to the training dataset. Accepted format:'
                       '1) a single data path, 2) multiple datasets in the'
                       'form: dataset1-weight dataset1-path dataset2-weight '
                       'dataset2-path ... It is used with --split when a '
                       'single dataset used for all three: train, valid '
                       'and test. It is exclusive to the other '
                       '--*-data-path args')
    group.add_argument('--split', type=str, default='969, 30, 1',
                       help='Comma-separated list of proportions for training,'
                       ' validation, and test split. For example the split '
                       '`90,5,5` will use 90%% of data for training, 5%% for '
                       'validation and 5%% for test.')
    group.add_argument('--train-data-path', nargs='*', default=None,
                       help='Path to the training dataset. Accepted format:'
                       '1) a single data path, 2) multiple datasets in the'
                       'form: dataset1-weight dataset1-path dataset2-weight '
                       'dataset2-path ...')
    group.add_argument('--valid-data-path', nargs='*', default=None,
                       help='Path to the validation dataset. Accepted format:'
                       '1) a single data path, 2) multiple datasets in the'
                       'form: dataset1-weight dataset1-path dataset2-weight '
                       'dataset2-path ...')
    group.add_argument('--test-data-path', nargs='*', default=None,
                       help='Path to the test dataset. Accepted format:'
                       '1) a single data path, 2) multiple datasets in the'
                       'form: dataset1-weight dataset1-path dataset2-weight '
                       'dataset2-path ...')
    group.add_argument('--data-cache-path', default=None,
                       help='Path to a directory to hold cached index files.')

    group.add_argument('--vocab-size', type=int, default=None,
                       help='Size of vocab before EOD or padding.')
    group.add_argument('--vocab-file', type=str, default=None,
                       help='Path to the vocab file.')
    group.add_argument('--merge-file', type=str, default=None,
                       help='Path to the BPE merge file.')
    group.add_argument('--vocab-extra-ids', type=int, default=0,
                       help='Number of additional vocabulary tokens. '
                            'They are used for span masking in the T5 model')
    group.add_argument('--max_reward_samples', type=int, default=8,
                       help='Maximum sequence length to process.')
    group.add_argument('--encoder-seq-length', type=int, default=None,
                       help='Maximum encoder sequence length to process.'
                       'This should be exclusive of --seq-length')
    group.add_argument('--decoder-seq-length', type=int, default=None,
                       help="Maximum decoder sequence length to process.")
    group.add_argument('--retriever-seq-length', type=int, default=256,
                       help='Maximum sequence length for the biencoder model '
                       'for retriever')
    group.add_argument('--sample-rate', type=float, default=1.0,
                       help='sample rate for training data. Supposed to be 0 '
                            ' < sample_rate < 1')
    group.add_argument('--mask-prob', type=float, default=0.15,
                       help='Probability of replacing a token with mask.')
    group.add_argument('--short-seq-prob', type=float, default=0.1,
                       help='Probability of producing a short sequence.')
    group.add_argument('--mmap-warmup', action='store_true',
                       help='Warm up mmap files.')
    group.add_argument('--tokenizer-type', type=str,
                       default=None,
                       choices=['BertWordPieceLowerCase',
                                'BertWordPieceCase',
                                'GPT2BPETokenizer',
                                'SentencePieceTokenizer',
                                'GPTSentencePieceTokenizer',
                                'NullTokenizer'],
                       help='What type of tokenizer to use.')
    group.add_argument('--tokenizer-model', type=str, default=None,
                       help='Sentencepiece tokenizer model.')
    group.add_argument('--data-impl', type=str, default='infer',
                       choices=['mmap', 'infer'],
                       help='Implementation of indexed datasets.')
    group.add_argument('--reset-position-ids', action='store_true',
                       help='Reset posistion ids after end-of-document token.')
    group.add_argument('--reset-attention-mask', action='store_true',
                       help='Reset self attention maske after '
                       'end-of-document token.')
    group.add_argument('--eod-mask-loss', action='store_true',
                       help='Mask loss for the end of document tokens.')
    group.add_argument('--train-data-exact-num-epochs', type=int, default=None,
                       help='When building the train dataset, force it to be '
                       'an exact number of epochs of the raw data')
    group.add_argument('--return-data-index', action='store_true',
                       help='Return the index of data sample.')
    group.add_argument('--data-efficiency-curriculum-learning', action='store_true',
                       help='Use DeepSpeed data efficiency library curriculum learning feature.')
    group.add_argument('--train-idx-path', type=str, default=None,
                       help='Force to use certain index file.')
    group.add_argument('--train-desc-path', type=str, default=None,
                       help='Force to use certain index file.')
    group.add_argument('--train-doc-idx-path', type=str, default=None,
                       help='Force to use certain index file.')
    group.add_argument('--train-sample-idx-path', type=str, default=None,
                       help='Force to use certain index file.')
    group.add_argument('--train-shuffle-idx-path', type=str, default=None,
                       help='Force to use certain index file.')
    return parser

def _add_zero_args(parser):
    """Text generate arguments."""

    group = parser.add_argument_group('ZeRO configurations', 'configurations')
    group.add_argument("--zero_stage", type=int, default=1)
    group.add_argument('--offload',action='store_true',
                        help='Enable ZeRO Offload techniques.')
    group.add_argument('--dtype', type=str, default='fp16', choices=['fp16', 'bf16'],
                        help='Training data type')
    group.add_argument('--zero-reduce-scatter', action='store_true',
                       help='Use reduce scatter if specified')
    group.add_argument('--zero-contigious-gradients', action='store_true',
                       help='Use contigious memory optimizaiton if specified')
    group.add_argument("--zero-reduce-bucket-size", type=int, default=0.0)
    group.add_argument("--zero-allgather-bucket-size", type=int, default=0.0)
    group.add_argument('--remote-device', type=str, default='none', choices=['none', 'cpu', 'nvme'],
                      help='Remote device for ZeRO-3 initialized parameters.')
    group.add_argument('--use-pin-memory', action='store_true',
                     help='Use pinned CPU memory for ZeRO-3 initialized model parameters.')
    return parser

def _add_activation_checkpoint_args(parser):
    group = parser.add_argument_group('Activation Checkpointing',
                                      'Checkpointing Configurations')
    group.add_argument('--deepspeed-activation-checkpointing', action='store_true',
                       help='uses activation checkpointing from deepspeed')
    group.add_argument('--partition-activations', action='store_true',
                       help='partition Activations across GPUs before checkpointing.')
    group.add_argument('--contigious-checkpointing', action='store_true',
                       help='Contigious memory checkpointing for activatoins.')
    group.add_argument('--checkpoint-in-cpu', action='store_true',
                       help='Move the activation checkpoints to CPU.')
    group.add_argument('--synchronize-each-layer', action='store_true',
                       help='does a synchronize at the beginning and end of each checkpointed layer.')
    group.add_argument('--profile-backward', action='store_true',
                       help='Enables backward pass profiling for checkpointed layers.')
    return parser

def _add_supervise_finetune_args(parser):
    group = parser.add_argument_group('Supervised Finetuning',
                                    'SFT Training Configurations')
    group.add_argument('--data_path', nargs='*', default=['ours/sample_instruct_stage1'],
                        help='Path to the training dataset. Accepted format:'
                        '1) a single data path, 2) multiple datasets in the'
                        'form: dataset1-path dataset2-path ...')
    group.add_argument('--data_split', type=str, default='10,0,0',
                        help='Comma-separated list of proportions for training'
                        'phase 1, 2, and 3 data. For example the split `2,4,4`'
                        'will use 60% of data for phase 1, 20% for phase 2'
                        'and 20% for phase 3.')
    group.add_argument('--end_of_conversation_token', type=str, default="</s>",
                        help='End of conversation words with one more tokens.')
    group.add_argument('--sft_only_data_path', nargs='*',default=[],
                        help='Path to the dataset for only using in SFT phase.')
    group.add_argument('--prefix_instruction', action='store_true',
                       help='Whether to use prefix instruction.')
    group.add_argument('--adding_demonstrations', action='store_true',
                       help='Whether to use prefix instruction.')
    group.add_argument('--prompt_input', action='store_true',
                       help='Whether to use prefix instruction.')
    group.add_argument("--recipe", type=str, default="",
                       help="the dataest reicpe")
    group.add_argument("--eval_recipe", type=str, default=None,
                       help="the dataest reicpe")
    group.add_argument("--eval_data_split", type=str, default=None,
                       help="the dataest split")
    group.add_argument("--eval_data_path", nargs='*', default=None)
    group.add_argument("--threshhold", type=float, default=0.1,
                       help="threshhold for dormant neuron")
    group.add_argument("--cache_dir", type=str, default="/mnt/data/temp/cache/",
                       help="the path of cache")
    group.add_argument("--template_path", type=str, default=None,
                       help="the path of template")
    group.add_argument('--data_output_path', type=str, default='/mnt/data/temp/cache',
                       help='Where to store the data-related files such as shuffle index. '
                       'This needs to be on a local storage of a node (not on a shared storage)')
    group.add_argument( "--tokenizer_path", type=str,
                       help="Path to pretrained model or model identifier from huggingface.co/models.")
    group.add_argument("--num_workers", type=int, default=64,
                       help="Number of workers for processing the data.")
    group.add_argument("--train_phase", type=int, default=1,
                        help="the training phase")
    group.add_argument("--sft_train_epochs", type=int, default=2,
                        help="SFT data training epochs")
    group.add_argument("--lse_square_scale", type=float, default=0.0,
                       help="Auxiliary loss scale for Z-loss adopted from Palm")
    group.add_argument('--inplace_backward', action='store_true',
                       help='Loss backward in inplace.')
    group.add_argument('--pairwise_allresponse', action='store_true',
                       help='Loss backward in inplace.')
    group.add_argument('--template_start_end', nargs='*', default=['Human:\n','Assistant:\n'],
                        help='Template start and end words for all sthree steps,'
                        'Very important for correctly add final token when input text'
                        'is reaching the max_seq_len')
    group.add_argument('--concat_prompt', action='store_true',
                        help='whether concate prompt to max length in the stage 1')
    return parser

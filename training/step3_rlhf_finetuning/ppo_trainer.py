# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import os
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import time
import deepspeed
from typing import Tuple
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from deepspeed.accelerator import get_accelerator
from utils.utils import print_rank_0, to_device
from transformers.models.llama.modeling_llama import LlamaMLP
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXMLP
from utils.utils_hr import register_mlpblock_energyloss_hooks_last_layer, register_mlp_energyloss_hooks_last_layer_deepseek, register_mlp_energyloss_hooks_last_layer_mistral, compute_dst, register_mlp_energyloss_hooks_last_layer_icml_rebuttal
import concurrent.futures
import requests

def remove_leading_zeros(lst, remove_id):
    while lst and lst[0] == remove_id:
        lst.pop(0)
    return lst

def gen(prompt_list):
    url = "http://10.119.29.22:6007/v1/completions"
    headers = {
        'User-Agent': "Apifox/1.0.0",
        'Content-Type': "application/json",
        'Authorization': "Bearer 7f6cd4662cffb99bd51b61ed75b5e138"
    }
    raws = {
        "prompt": prompt_list,
        "model": "InitiAI-v1.4-music",
        "max_tokens": 512
        }

    response = requests.post(
        url=url,
        headers=headers,
        json=raws
    )
    res = response.json()['choices']
    assert len(res) == 1, "Error"
    response = res[0]['text']
    return response

def infre_generate(tokenizer, context):
    print(context.shape)
    prompts_str = [tokenizer.decode(context[i].tolist()) for i in range(context.shape[0])]
    all_prompt = [[remove_leading_zeros(context[i].tolist(), tokenizer.pad_token_id)] for i in range(context.shape[0])]

    def process_group(index, line_list):
        res = gen(prompt_list=line_list[index])
        return [{"id": index, "response": res}]

    output_list = []
    line_list = all_prompt
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(line_list)) as executor:
        futures = []
        for i in range(0, len(line_list)):  # Adjust group size to your needs
            futures.append(executor.submit(process_group, i, line_list))

        for future in concurrent.futures.as_completed(futures):
            output_list.extend(future.result())

    output_list.sort(key=lambda x: x['id'])
    response_str = [o['response'] for o in output_list]
    final_results = pad_sequence(
        [torch.tensor(tokenizer.encode(p + o + tokenizer.eos_token)[1:]) for p, o in zip(prompts_str, response_str)], 
        padding_value=tokenizer.pad_token_id, 
        batch_first=True
    )
    print(final_results.shape)
    return final_results.cuda()

def print_all_ranks(tag, value, rank):
    world_size = torch.distributed.get_world_size()
    all_tensor = torch.zeros(world_size, dtype=torch.float32).to(
        get_accelerator().current_device_name())
    all_tensor[rank] = value
    torch.distributed.all_reduce(all_tensor, op=torch.distributed.ReduceOp.SUM)
    print_rank_0(f'{tag} {all_tensor}', rank)


def get_model_norm(model):
    with torch.no_grad():
        total = 0.0
        for param in model.parameters():
            should_gather = hasattr(
                param,
                'ds_id') and param.ds_status == ZeroParamStatus.NOT_AVAILABLE
            with deepspeed.zero.GatheredParameters(param,
                                                   enabled=should_gather):
                total += float(param.float().norm())

    return total


def gather_log_probs(logits, labels):
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)

# NOTE: Used in step 3 for reward normalization and scaling
class RunningMoments:
    def __init__(self):
        """
        Calculates the running mean and standard deviation of a data stream. Modified version of
        https://github.com/DLR-RM/stable-baselines3/blob/a6f5049a99a4c21a6f0bcce458ca3306cef310e0/stable_baselines3/common/running_mean_std.py
        """
        self.mean = 0
        self.std = 1
        self.var = 1
        self.count = 1e-24

    @torch.no_grad()
    def update(self, xs: torch.Tensor) -> Tuple[float, float]:
        """
        Updates running moments from batch's moments computed across ranks
        """
        xs_count = xs.numel()
        xs_var, xs_mean = torch.var_mean(xs, unbiased=False)
        xs_mean, xs_var = xs_mean.float(), xs_var.float()

        delta = xs_mean - self.mean
        tot_count = self.count + xs_count

        new_sum = xs_var * xs_count
        # correct old_sum deviation accounting for the new mean
        old_sum = self.var * self.count + delta**2 * self.count * xs_count / tot_count
        tot_sum = old_sum + new_sum

        self.mean += delta * xs_count / tot_count
        self.var = tot_sum / tot_count
        self.std = (self.var * tot_count / (tot_count - 1)).float().sqrt()
        self.count = tot_count

        return xs_mean.item(), (xs_var * xs_count / (xs_count - 1)).float().sqrt().item()
    
class DeepSpeedPPOTrainer():

    def __init__(self, rlhf_engine, args):
        self.rlhf_engine = rlhf_engine
        self.actor_model = self.rlhf_engine.actor
        self.critic_model = self.rlhf_engine.critic
        self.ref_model = self.rlhf_engine.ref
        self.reward_model = self.rlhf_engine.reward
        self.tokenizer = self.rlhf_engine.tokenizer
        self.args = args
        self.max_answer_seq_len = args.max_answer_seq_len
        self.z3_enabled = args.actor_zero_stage == 3
        self.compute_fp32_loss = self.args.compute_fp32_loss

        # Those value can be changed
        self.kl_ctl = args.kl_ctl_weight
        self.clip_reward_value = 5
        self.cliprange = 0.2
        self.cliprange_value = 0.2
        self.gamma = 1.0
        self.lam = 0.95
        # NOTE: mean_kl is used for logging
        self.mean_kl = 0.
        self.generate_time = 0.0
        self.running_reward_stats = RunningMoments()
        self.running_energyloss_stats = RunningMoments()
    
    def _generate_sequence(self, prompts, mask, step):

        if self.actor_model.module.config.model_type == "llama" or self.args.temperature == 0:
            # kwargs = dict(do_sample=True)
            kwargs = dict(do_sample=False)
        else:
            kwargs = dict()
        
        with torch.no_grad():
            seq = self.actor_model.module.generate(
                prompts,
                attention_mask=mask,
                max_new_tokens=self.max_answer_seq_len,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                synced_gpus=self.z3_enabled,
                temperature=self.args.temperature,
                repetition_penalty=self.args.repetition_penalty,
                top_p=self.args.top_p,
                top_k=self.args.top_k,
                use_cache=True,
                **kwargs)
        # Filter out seq with no answers (or very short). This happens when users directly use the pre-training ckpt without supervised finetuning
        # NOTE: this will causes each GPU has different number of examples
        batch_size = seq.shape[0]
        # print("an example of seq at Line 135 is {}".format(seq[0]))
        prompt_length = prompts.shape[1]
        self.prompt_length = prompt_length
        first_occurrence_mask = (seq[:,prompt_length:] == self.tokenizer.eos_token_id).float().cumsum(1).cumsum(1)
        seq[:,prompt_length:] = seq[:,prompt_length:].masked_fill(first_occurrence_mask > 1, self.tokenizer.pad_token_id)
        ans = seq[:, prompt_length:]
        ans = seq
        valid_ans_len = (ans != self.tokenizer.pad_token_id).sum(dim=-1)

        out_seq = []
        for i in range(batch_size):
            if valid_ans_len[i] <= 1:  # if the answer is shorter than 1 token, drop it
                continue
            else:
                out_seq.append(seq[i:i+1])
        out_seq = torch.cat(out_seq, dim=0)  # concate output in the batch dim
        return out_seq
    
    @torch.inference_mode()
    def _get_ref_energy(self, prompts, mask, target=None, cal_length=1024):
        # This has been added due to a probability/nan error that happens after
        # meta-llama/Llama-2-7b-hf enabled do_sample:
        # https://huggingface.co/meta-llama/Llama-2-7b-hf/commit/6fdf2e60f86ff2481f2241aaee459f85b5b0bbb9
        # self.tokenizer.decode(output[0,:], skip_special_tokens=True)
        if target == 'v1':
            fn = register_mlpblock_energyloss_hooks_last_layer
        elif target == 'v2':
            if 'llama' in self.args.actor_model_name_or_path:
                ############### icml original version #################
                # fn = register_mlp_energyloss_hooks_last_layer
                ############### icml rebuttal version #################
                fn = register_mlp_energyloss_hooks_last_layer_icml_rebuttal
            elif 'deepseek' in self.args.actor_model_name_or_path:
                fn = register_mlp_energyloss_hooks_last_layer_deepseek
            elif 'mistral' in self.args.actor_model_name_or_path:
                fn = register_mlp_energyloss_hooks_last_layer_mistral

        output = infre_generate(self.tokenizer, prompts)
        start_idx = prompts.shape[1] - 1
        final_idx = output.shape[1] - 2
        attention_mask = (output != self.tokenizer.pad_token_id).long()
        activation_dict = fn(self.ref_model, output, attention_mask)
        activation_dict = {k:v[:, start_idx:final_idx] for k,v in activation_dict.items()} # remove prompts
        activation_list = list(activation_dict.values())
        mask = ((output != self.tokenizer.pad_token_id) & (output != self.tokenizer.eos_token_id)).long()[:,start_idx:final_idx]
        if mask.shape[1] > cal_length + 2:
            mask[:,cal_length:] = 0
        # if mask.shape[1] > cal_length + 2:
        #     mask[:,:-cal_length] = 0
        averages_energy = [torch.sum(activation*mask, dim=1) / torch.clamp(torch.sum(mask, dim=1), min=1) for activation in activation_list]
        rank = torch.distributed.get_rank()
        print_rank_0(self.tokenizer.decode(output[0,:], skip_special_tokens=False), rank)
        return averages_energy

    @torch.inference_mode()
    def _get_actor_energy_no_generation(self, prompts, output, target=None, cal_length=1024, ref_list=None):
        if target == 'v1':
            fn = register_mlpblock_energyloss_hooks_last_layer
        elif target == 'v2':
            if 'llama' in self.args.actor_model_name_or_path:
                ############### icml original version #################
                # fn = register_mlp_energyloss_hooks_last_layer
                ############### icml rebuttal version #################
                fn = register_mlp_energyloss_hooks_last_layer_icml_rebuttal
            elif 'deepseek' in self.args.actor_model_name_or_path:
                fn = register_mlp_energyloss_hooks_last_layer_deepseek
            elif 'mistral' in self.args.actor_model_name_or_path:
                fn = register_mlp_energyloss_hooks_last_layer_mistral
        start_idx = prompts.shape[1] - 1
        final_idx = output.shape[1] - 2
        attention_mask = (output != self.tokenizer.pad_token_id).long()
        activation_dict = fn(self.actor_model, output, attention_mask)
        activation_dict = {k:v[:, start_idx:final_idx] for k,v in activation_dict.items()} # remove prompts
        activation_list = list(activation_dict.values())
        print("begin to calculate cal_length abs_differences")
        mask = ((output != self.tokenizer.pad_token_id) & (output != self.tokenizer.eos_token_id)).long()[:,start_idx:final_idx]
        if mask.shape[1] > cal_length + 2:
            mask[:,cal_length:] = 0
        # if mask.shape[1] > cal_length + 2:
        #     mask[:,:-cal_length] = 0
        averages_energy = [torch.sum(activation*mask, dim=1) / torch.clamp(torch.sum(mask, dim=1), min=1) for activation in activation_list]

        return averages_energy


        
    def generate_experience(self, prompts, mask, step):
        self.eval()
        generate_start = time.time()
        seq = self._generate_sequence(prompts, mask, step)
        generate_end = time.time()
        ans_length = seq[:,prompts.shape[-1]:].not_equal(self.tokenizer.pad_token_id).sum(dim=1).float()
        attention_mask = seq.not_equal(self.tokenizer.pad_token_id)
        
        if self.tokenizer.pad_token == self.tokenizer.eos_token:
            print("correcting attention mask")
            bs, seq_len = attention_mask.shape
            for i in range(bs):
                last_one_pos = (attention_mask[i] == 1).nonzero(as_tuple=True)[0][-1].item()
                if last_one_pos < seq_len -1:
                    first_zero_after_last_one = last_one_pos + 1
                    attention_mask[i, first_zero_after_last_one] = 1
                    ans_length[i] = ans_length[i] + 1
                
                
        with torch.no_grad():
            output = self.actor_model(seq, attention_mask=attention_mask.long())
            output_ref = self.ref_model(seq, attention_mask=attention_mask.long())
            reward_score = self.reward_model.forward_value(seq, attention_mask,
                prompt_length=prompts.shape[1])['chosen_end_scores'].detach()
            
            # Used for logging
            ori_rewards = reward_score.clone().detach() #torch.Size([2])
            values = self.critic_model.forward_value(
                seq, attention_mask, return_value_only=True).detach()[:, :-1]
        logits = output.logits
        logits_ref = output_ref.logits
        if self.compute_fp32_loss:
            logits = logits.to(torch.float)
            logits_ref = logits_ref.to(torch.float)
        self.generate_time = generate_end - generate_start


        # Using zhengrui code to normalize/scaling the reward
        self.running_reward_stats.update(reward_score)
        if self.args.use_reward_norm:
            reward_score = (reward_score - self.running_reward_stats.mean) / self.running_reward_stats.std
        elif self.args.use_reward_scaling:
            reward_score /= self.running_reward_stats.std
        else:
            pass

        if self.args.length_penalty_N != 0:
            start_idx = prompts.shape[1] - 1
            final_idx = seq.shape[1] - 2
            mask_ = ((seq != self.tokenizer.pad_token_id) & (seq != self.tokenizer.eos_token_id)).long()[:,start_idx:final_idx]
            response_length = torch.clamp(torch.sum(mask_, dim=1), min=1)
            print("activating length penalty with response_length {}".format(response_length))
            reward_score = reward_score + (1 - response_length/self.args.length_penalty_N) * self.running_reward_stats.std
        
        if self.args.use_dst_penalty:
            dst_actor = compute_dst(self.actor_model.module, prompts, seq, self.tokenizer)
            seq_ref = infre_generate(self.tokenizer, prompts)
            dst_ref = compute_dst(self.ref_model.module, prompts, seq_ref, self.tokenizer)
            print("activating dst penalty with dst_ref {} and dst_actor {}".format(dst_ref, dst_actor))
            dst_loss = torch.abs(dst_actor - dst_ref)
            reward_score = reward_score - self.args.dst_penalty_co * dst_loss

        # use energyloss to panel the reward
        if self.args.energy_loss_ta != None:
            ref_averages_energy = self._get_ref_energy(prompts, mask, self.args.energy_loss_ta, cal_length=1024)
            actor_averages_energy = self._get_actor_energy_no_generation(prompts, seq, self.args.energy_loss_ta, cal_length=1024)

            energy_loss = torch.cat([torch.abs(t1 - t2).unsqueeze(0) for t1, t2 in zip(actor_averages_energy, ref_averages_energy)]).sum(dim=0)

            actor_averages_energy_save = torch.cat([t1.unsqueeze(0) for t1 in actor_averages_energy]).T # (layer_num, bs).T
            ref_averages_energy_save = torch.cat([t1.unsqueeze(0) for t1 in ref_averages_energy]).T # (layer_num, bs).T
            print("the shape of actor_averages_energy_save is {}".format(actor_averages_energy_save.shape))

            assert reward_score.shape == energy_loss.shape, "Big Error!!!"

            min_reward_score = torch.min(reward_score - self.args.energy_loss_co * energy_loss)
            max_reward_ratio = torch.max((self.args.energy_loss_co * energy_loss) / reward_score)
            if self.args.energy_loss_clip_ratio != -100.0:
                print("cliping energy loss")
                reward_score = torch.max(reward_score - self.args.energy_loss_co * energy_loss, self.args.energy_loss_clip_ratio * reward_score)
            else:
                reward_score = reward_score - self.args.energy_loss_co * energy_loss

        gold_score = self.get_glod_reward(seq, parallel=True) if self.args.gold_reward_model else [-1]*len(ans_length)
        energy_loss = torch.ones_like(reward_score) * -100 if self.args.energy_loss_ta == None else energy_loss
        actor_averages_energy_save = torch.ones_like(reward_score) * -100 if self.args.energy_loss_ta == None else actor_averages_energy_save
        ref_averages_energy_save = torch.ones_like(reward_score) * -100 if self.args.energy_loss_ta == None else ref_averages_energy_save
        dst_actor = torch.ones_like(reward_score) * -100 if not self.args.use_dst_penalty else dst_actor
        dst_ref = torch.ones_like(reward_score) * -100 if not self.args.use_dst_penalty else dst_ref
        dst_loss = torch.ones_like(reward_score) * -100 if not self.args.use_dst_penalty else dst_loss


        return {
            'prompts': prompts,
            'logprobs': gather_log_probs(logits[:, :-1, :], seq[:, 1:]),
            'ref_logprobs': gather_log_probs(logits_ref[:, :-1, :], seq[:,1:]),
            'value': values,
            'rewards': reward_score,
            'input_ids': seq,
            "attention_mask": attention_mask,
            "ori_rewards": ori_rewards,
            "ans_length": ans_length,
            "gold_score": gold_score,
            "generation_mask": mask,
            "energy_loss": energy_loss.detach(),
            "actor_averages_energy": actor_averages_energy_save.detach(),
            "ref_averages_energy": ref_averages_energy_save.detach(),
            "dst_loss": dst_loss.detach(),
            "dst_actor": dst_actor.detach(),
            "dst_ref": dst_ref.detach()
        }

    def compute_rewards(self, prompts, log_probs, ref_log_probs, reward_score, action_mask):
        kl_divergence_estimate = -self.kl_ctl * (log_probs - ref_log_probs)
        rewards = kl_divergence_estimate
        
        
        # Sen: mean_kl is used for logging
        # self.mean_kl = kl_divergence_estimate.mean().float()
        self.mean_kl = (ref_log_probs - log_probs).mean().float()
        self.abs_mean_kl = (ref_log_probs - log_probs).abs().mean().float()

        
        start = prompts.shape[1] - 1
        ends = start + action_mask[:, start:].sum(1) + 1
        reward_clip = torch.clamp(reward_score, -self.clip_reward_value, self.clip_reward_value)
        batch_size = log_probs.shape[0]
        for j in range(batch_size):
            rewards[j, start:ends[j]][-1] += reward_clip[j]
        return rewards

    def train_rlhf(self, inputs):
        # train the rlhf mode here
        ### process the old outputs
        prompts = inputs['prompts']
        log_probs = inputs['logprobs']
        ref_log_probs = inputs['ref_logprobs']
        reward_score = inputs['rewards']
        values = inputs['value']
        attention_mask = inputs['attention_mask']
        seq = inputs['input_ids']

        start = prompts.size()[-1] - 1
        action_mask = attention_mask[:, 1:]
        old_values = values
        with torch.no_grad():
            old_rewards = self.compute_rewards(prompts, log_probs,
                                               ref_log_probs, reward_score,
                                               action_mask)
            ends = start + action_mask[:, start:].sum(1) + 1
            for i in range(old_rewards.shape[0]):
                old_rewards[i, ends[i]:] = 0
                old_values[i, ends[i]:] = 0
            advantages, returns = self.get_advantages_and_returns(old_values, old_rewards, start)
            # zheng rui function 7-8
        ### process the new outputs
        self.train()
        try:
            self.actor_model.optimizer.zero_grad()
            self.critic_model.optimizer.zero_grad()
        except:
            print("fail to zreo grad")
        batch = {'input_ids': seq, "attention_mask": attention_mask}
        actor_prob = self.actor_model(**batch, use_cache=False).logits
        actor_log_prob = gather_log_probs(actor_prob[:, :-1, :], seq[:, 1:])
        actor_loss = self.actor_loss_fn(actor_log_prob[:, start:],
                                        log_probs[:, start:], advantages,
                                        action_mask[:, start:])
        
        self.actor_model.backward(actor_loss)

        if not self.args.align_overflow:
            self.actor_model.step()

        value = self.critic_model.forward_value(**batch,
                                                return_value_only=True,
                                                use_cache=False)[:, :-1]
        critic_loss = self.critic_loss_fn(value[:, start:], old_values[:,
                                                                       start:],
                                          returns, action_mask[:, start:])
        self.critic_model.backward(critic_loss)

        if self.args.align_overflow:
            
            actor_overflow = self.actor_model.optimizer.check_overflow()
            critic_overflow = self.critic_model.optimizer.check_overflow()

            rank = torch.distributed.get_rank()
            if actor_overflow:
                self.actor_model.optimizer.skip_step = True
                print_rank_0(
                    "OVERFLOW: actor overflow, skipping both actor and critic steps",
                    rank)
            elif critic_overflow:
                self.critic_model.optimizer.skip_step = True
                print_rank_0(
                    "OVERFLOW: critic overflow, skipping both actor and critic steps",
                    rank)
            else:
                pass
            self.actor_model.step()
        self.critic_model.step()

        return actor_loss, critic_loss

    def get_overflow(self):
        # Overflow is not expected when using bf16
        # Therefore, DeepSpeed's BF16_Optimizer does not maintain an overflow indication
        if self.args.dtype == "bf16":
            return False, False

        actor_overflow = self.actor_model.optimizer.overflow
        critic_overflow = self.critic_model.optimizer.overflow

        return actor_overflow, critic_overflow

    def actor_loss_fn(self, logprobs, old_logprobs, advantages, mask):
        ## policy gradient loss
        log_ratio = (logprobs - old_logprobs) * mask
        ratio = torch.exp(log_ratio)
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - self.cliprange,
                                             1.0 + self.cliprange)
        pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2) * mask) / mask.sum()
        return pg_loss

    def critic_loss_fn(self, values, old_values, returns, mask):
        ## value loss
        values_clipped = torch.clamp(
            values,
            old_values - self.cliprange_value,
            old_values + self.cliprange_value,
        )
        if self.compute_fp32_loss:
            values = values.float()
            values_clipped = values_clipped.float()
        vf_loss1 = (values - returns)**2
        vf_loss2 = (values_clipped - returns)**2
        vf_loss = 0.5 * torch.sum(
            torch.max(vf_loss1, vf_loss2) * mask) / mask.sum()
        return vf_loss

    def get_advantages_and_returns(self, values, rewards, start):
        # Adopted from https://github.com/CarperAI/trlx/blob/main/trlx/models/modeling_ppo.py#L134
        lastgaelam = 0
        advantages_reversed = []
        length = rewards.size()[-1]
        for t in reversed(range(start, length)):
            nextvalues = values[:, t + 1] if t < length - 1 else 0.0
            delta = rewards[:, t] + self.gamma * nextvalues - values[:, t]
            lastgaelam = delta + self.gamma * self.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values[:, start:]
        return advantages.detach(), returns

    def _validate_training_mode(self):
        assert self.actor_model.module.training
        assert self.critic_model.module.training

    def _validate_evaluation_mode(self):
        assert not self.actor_model.module.training
        assert not self.critic_model.module.training
        assert not self.ref_model.module.training
        assert not self.reward_model.module.training

    def train(self):
        self.actor_model.train()
        self.critic_model.train()

    def eval(self):
        self.actor_model.eval()
        self.critic_model.eval()
        self.reward_model.eval()
        self.ref_model.eval()

    def dump_model_norms(self, tag):
        actor_model_norm = get_model_norm(self.actor_model)
        ref_model_norm = get_model_norm(self.ref_model)
        critic_model_norm = get_model_norm(self.critic_model)
        reward_model_norm = get_model_norm(self.reward_model)
        print_all_ranks(f'{tag} global_actor_model_norm', actor_model_norm,
                        self.args.local_rank)
        print_all_ranks(f'{tag} global_ref_model_norm', ref_model_norm,
                        self.args.local_rank)
        print_all_ranks(f'{tag} global_critic_model_norm', critic_model_norm,
                        self.args.local_rank)
        print_all_ranks(f'{tag} global_reward_model_norm', reward_model_norm,
                        self.args.local_rank)
    
    def get_kl(self):
        return self.abs_mean_kl, self.mean_kl
    
    def get_glod_reward(self, seq, parallel):
        with torch.no_grad():
            device = torch.device("cuda", self.args.local_rank)
            seq_decode = self.tokenizer.batch_decode(seq, skip_special_tokens=True)
            seq_gold_encode = [to_device(self.rlhf_engine.gold_reward_tokenizer(item + self.rlhf_engine.gold_reward_tokenizer.eos_token, return_tensors='pt'), device) for item in seq_decode]
            if not parallel:
                gold_score_list = []
                for sample in seq_gold_encode:
                    gold_score = self.rlhf_engine.gold_reward.forward_value(**sample, print_example=False)['chosen_end_scores'].detach().squeeze().item()
                    gold_score_list.append(gold_score)
                return torch.tensor(gold_score_list).to(device)
            else:
                input_ids = [x['input_ids'].squeeze().tolist() for x in seq_gold_encode]
                attention_mask = [x['attention_mask'].squeeze().tolist() for x in seq_gold_encode]
                input_ids = pad_sequence([torch.tensor(x, dtype=torch.long).to(device) for x in input_ids], padding_value=self.rlhf_engine.gold_reward_tokenizer.pad_token_id, batch_first=True)
                attention_mask = pad_sequence([torch.tensor(x, dtype=torch.long).to(device) for x in attention_mask], padding_value=self.rlhf_engine.gold_reward_tokenizer.pad_token_id, batch_first=True)
                assert attention_mask.size(0) == input_ids.size(0) and attention_mask.size(1) == input_ids.size(1)
                gold_score = self.rlhf_engine.gold_reward.forward_value(input_ids, attention_mask, print_example=False)['chosen_end_scores'].detach().squeeze()
                return gold_score
    
class DeepSpeedPPOTrainerUnsupervised(DeepSpeedPPOTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_unsupervised(self, inputs, unsup_coef):
        # Train the unsupervised model here
        self._validate_training_mode()

        outputs = self.actor_model(**inputs, use_cache=False)
        loss = outputs.loss
        self.actor_model.backward(unsup_coef * loss)
        self.actor_model.step()

        return loss
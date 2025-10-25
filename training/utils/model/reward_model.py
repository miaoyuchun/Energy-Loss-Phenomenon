# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import torch
from torch import nn
from transformers import LlamaPreTrainedModel, LlamaModel, LlamaConfig
from torch.autograd import Variable
## Note that the following code is modified from
## https://github.com/CarperAI/trlx/blob/main/examples/summarize_rlhf/reward_model/reward_model.py
from utils.utils import print_rank_0
import torch
import torch.nn.functional as F

def calculate_entropy_torch(vec):
    p = F.softmax(vec, dim=-1)
    p_log_p = (-1) * p * torch.log(p+1e-10)
    return p_log_p.sum()  # 计算熵

def entropy_unnormalized(vec):# torch.Size([2*bs, self.factor_num, self.dim_factor])
    bs = vec.shape[0]//2
    # 使用softmax函数将vec转换为概率分布
    p = F.softmax(vec, dim=-1) # torch.Size([2*bs, self.factor_num, self.dim_factor])
    p_log_p = (-1) * p * torch.log(p+1e-10)
    p_log_p[p == 0] = 0.0  # torch.Size([2*bs, self.factor_num, self.dim_factor])
    inner_entropy = p_log_p.sum(-1)[:bs,:].mean() + p_log_p.sum(-1)[bs:,:].mean()
    return inner_entropy

# 创建一个没有归一化的向量，包含正数和负数

def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5*(1 + logvar - mu.pow(2) - logvar.exp()) # torch.Size([64, 32])
    total_kld = klds.sum(1).mean(0, True) # torch.Size([1])
    dimension_wise_kld = klds.mean(0) # torch.Size([32])
    mean_kld = klds.mean(1).mean(0, True) # torch.Size([1])
    
    return total_kld, dimension_wise_kld, mean_kld
    
class RewardModel(nn.Module):
    def __init__(self,
                 base_model,
                 tokenizer,
                 num_padding_at_beginning=0,
                 compute_fp32_loss=False):
        super().__init__()
        self.config = base_model.config
        self.num_padding_at_beginning = num_padding_at_beginning
        if hasattr(self.config, "word_embed_proj_dim"):
            # `OPT` models use word_embed_proj_dim as final output
            # https://github.com/huggingface/transformers/blob/main/src/transformers/models/opt/modeling_opt.py#L497
            self.v_head = nn.Linear(self.config.word_embed_proj_dim, 1, bias=False)
        else:
            # `gpt-neo(x)` models use `hidden_size` attribute names instead of `n_embd``
            self.config.n_embd = self.config.hidden_size if hasattr(
                self.config, "hidden_size") else self.config.n_embd
            self.v_head = nn.Linear(self.config.n_embd, 1, bias=False)
        self.model = base_model
        self.PAD_ID = tokenizer.pad_token_id
        self.compute_fp32_loss = compute_fp32_loss

    def gradient_checkpointing_enable(self):
        self.model.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.model.gradient_checkpointing_disable()

    def forward(self,
                input_ids=None,
                past_key_values=None,
                attention_mask=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                use_cache=False,
                category=None):
        loss = None

        if self.config.model_type == "llama":
            kwargs = dict()
        else:
            kwargs = dict(head_mask=head_mask)

        transformer_outputs = self.model(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs)

        hidden_states = transformer_outputs[0]
        rewards = self.v_head(hidden_states).squeeze(-1)
        chosen_mean_scores = []
        rejected_mean_scores = []

        # Split the inputs and rewards into two parts, chosen and rejected
        assert len(input_ids.shape) == 2
        bs = input_ids.shape[0] // 2
        seq_len = input_ids.shape[1]

        chosen_ids = input_ids[:bs]  # bs x seq x 1
        rejected_ids = input_ids[bs:]
        chosen_rewards = rewards[:bs]
        rejected_rewards = rewards[bs:]

        # Compute pairwise loss. Only backprop on the different tokens before padding
        loss = 0.
        for i in range(bs):
            chosen_id = chosen_ids[i]
            rejected_id = rejected_ids[i]
            chosen_reward = chosen_rewards[i]
            rejected_reward = rejected_rewards[i]

            c_inds = (chosen_id == self.PAD_ID).nonzero()
            c_ind = c_inds[self.num_padding_at_beginning].item() if len(
                c_inds
            ) > self.num_padding_at_beginning else seq_len  # OPT model pads the first token, so we need to use the second padding token as the end of the sequence
            check_divergence = (chosen_id != rejected_id).nonzero()

            if len(check_divergence) == 0:
                end_ind = rejected_reward.size(-1)
                divergence_ind = end_ind - 1
                r_ind = c_ind
            else:
                # Check if there is any padding otherwise take length of sequence
                r_inds = (rejected_id == self.PAD_ID).nonzero()
                r_ind = r_inds[self.num_padding_at_beginning].item(
                ) if len(r_inds) > self.num_padding_at_beginning else seq_len
                end_ind = max(c_ind, r_ind)
                divergence_ind = check_divergence[0]
            assert divergence_ind > 0
            c_truncated_reward = chosen_reward[divergence_ind:end_ind]
            r_truncated_reward = rejected_reward[divergence_ind:end_ind]
            chosen_mean_scores.append(
                chosen_reward[c_ind - 1])  #use the end score for reference
            rejected_mean_scores.append(rejected_reward[r_ind - 1])

            if self.compute_fp32_loss:
                c_truncated_reward = c_truncated_reward.float()
                r_truncated_reward = r_truncated_reward.float()
            loss += -torch.nn.functional.logsigmoid(c_truncated_reward -
                                                    r_truncated_reward).mean()

        loss = loss / bs
        chosen_mean_scores = torch.stack(chosen_mean_scores)
        rejected_mean_scores = torch.stack(rejected_mean_scores)
        return {
            "loss": loss,
            "chosen_mean_scores": chosen_mean_scores,
            "rejected_mean_scores": rejected_mean_scores,
        }

    def forward_value(self,
                      input_ids=None,
                      attention_mask=None,
                      past_key_values=None,
                      position_ids=None,
                      head_mask=None,
                      inputs_embeds=None,
                      return_value_only=False,
                      prompt_length=0,
                      use_cache=False):

        if self.config.model_type == "llama":
            kwargs = dict()
        else:
            kwargs = dict(head_mask=head_mask)

        transformer_outputs = self.model(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs)
        hidden_states = transformer_outputs[0]
        values = self.v_head(hidden_states).squeeze(-1)
        if return_value_only:
            return values
        else:
            # [0 0 0 0 prompt, answer, 0 0 0 0 ] for step 3, we have padding at the beginning
            # [prompt, answer, 0, 0, 0, 0] this is normal
            # assert prompt_length > 1, "prompt_length must be greater than 1 to help select the end score"
            bs = values.size(0)
            seq_len = input_ids.shape[1]
            chosen_end_scores = [
            ]  # we use this name for consistency with the original forward function
            for i in range(bs):
                input_id = input_ids[i]
                value = values[i]

                c_inds = (input_id[prompt_length:] == self.PAD_ID).nonzero()
                # here we only use the answer part of the sequence so we do not need to care about the padding at the beginning
                c_ind = c_inds[0].item() + prompt_length if len(
                    c_inds) > 0 else seq_len
                chosen_end_scores.append(value[c_ind - 1])
            return {
                "values": values,
                "chosen_end_scores": torch.stack(chosen_end_scores),
            }

class RewardLlama(LlamaPreTrainedModel):
    _tied_weights_keys = ["v_head.weight"]
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.config = config
         # NOTE: tokenizer is used for internal outputs in step 2
        self.tokenizer = kwargs.pop("tokenizer", None)
        self.use_sigmoid = kwargs.pop("use_sigmoid", False)
        self.use_encoder_sigmoid = kwargs.pop("use_encoder_sigmoid", False)
        self.use_decoder_sigmoid = kwargs.pop("use_decoder_sigmoid", False)
        self.only_reward_final_token = kwargs.pop("only_reward_final_token", False)
        self.use_variational_inference = kwargs.pop("use_variational_inference", False)
        self.use_simcse = kwargs.pop("use_simcse", False)
        self.num_padding_at_beginning = kwargs.pop("num_padding_at_beginning", 0)
        self.factor_num = kwargs.pop("factor_num", 3) # the number of factors
        self.dim_factor = kwargs.pop("dim_factor", 1) # the dimension per factors
        self.factor_width =  kwargs.pop("factor_width", 1) # the dimension occupied by each factors
        self.latent_dim =  self.factor_width * self.factor_num * self.dim_factor # the total dimention at the latent space
        self.weight_auxiliary=kwargs.pop("weight_auxiliary", 0.5)
        self.beta = kwargs.pop("beta", 0.5)
        self.gamma = kwargs.pop("gamma", 1.0)
        self.lamda = kwargs.pop("lamda", 0.1)
        self.omega = kwargs.pop("omega", 0.0)
        self.PAD_ID = kwargs.pop("pad_id", None)
        self.no_sample = kwargs.pop("no_sample", False)
        self.complex_fcn_decode = kwargs.pop("complex_decode", False)
        self.introduce_supervision = kwargs.pop("introduce_supervision", True)
        self.internal_entropy = kwargs.pop("internal_entropy", False)


        assert self.PAD_ID is not None, "pad_id must be given for RewardLlama"
        
        hidden_size = self.config.hidden_size if hasattr(self.config, "hidden_size") else self.config.n_embd
        hidden_size = hidden_size if hasattr(self.config, "word_embed_proj_dim") else hidden_size
        if self.use_variational_inference:
            if self.use_encoder_sigmoid:
                self.encode_head = nn.Sequential(nn.Linear(hidden_size, self.latent_dim*2, bias=False), nn.Sigmoid())
            else:
                self.encode_head = nn.Linear(hidden_size, self.latent_dim*2, bias=False)
            if self.use_decoder_sigmoid:
                self.decode_head = nn.Sequential(nn.Linear(self.latent_dim, 1, bias=False), nn.Sigmoid())
            else:
                if self.complex_fcn_decode:
                    self.decode_head = nn.Sequential(nn.Linear(self.latent_dim, self.latent_dim*8, bias=True), nn.Sigmoid(), nn.Linear(self.latent_dim*8, self.latent_dim*16, bias=True), nn.Sigmoid(), nn.Linear(self.latent_dim*16, self.latent_dim*32, bias=True), nn.Sigmoid(), nn.Linear(self.latent_dim*32, self.latent_dim*16, bias=True), nn.Sigmoid(), nn.Linear(self.latent_dim*16, self.latent_dim*8, bias=True), nn.Sigmoid(), nn.Linear(self.latent_dim*8, 1, bias=True))
                else:
                    self.decode_head = nn.Linear(self.latent_dim, 1, bias=False)
        if self.use_sigmoid:
            self.v_head = nn.Sequential(nn.Linear(hidden_size, 1, bias=False), nn.Sigmoid())
        else:
            self.v_head = nn.Linear(hidden_size, 1, bias=False)
        self.model = LlamaModel(config)
        self.post_init()
    
    def gradient_checkpointing_enable(self):
        self.model.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.model.gradient_checkpointing_disable()

    def reward(self,
                input_ids=None, # torch.Size([4, 418])
                past_key_values=None,
                attention_mask=None, # torch.Size([4, 418])
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                use_cache=False,
                category=None, 
                 **kwargs):
        with torch.no_grad():
            transformer_outputs = self.model(
                input_ids, # torch.Size([batch_Size, max_length])
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache) 
            hidden_states = transformer_outputs[0] # torch.Size([batchsize, max_length, 4096])
            if self.use_variational_inference:
                encode_result = self.encode_head(hidden_states) # torch.Size([4, 418, self.latent_dim*2])
                mu = encode_result[:, :, :self.latent_dim] # torch.Size([4, 418, self.latent_dim])
                logvar = encode_result[:, :, self.latent_dim:] # torch.Size([4, 418, self.latent_dim])
                std = logvar.div(2).exp() # torch.Size([4, 418, self.latent_dim])
                eps = Variable(std.data.new(std.size()).normal_()) # torch.Size([4, 418, self.latent_dim])
                decode_input = mu  #+ std*eps if not self.no_sample else  mu # torch.Size([4, 418, self.factor_num])
                batch_sequence_rewards = self.decode_head(decode_input).squeeze(-1) # torch.Size([4, 418])
                mu = mu.view(input_ids.shape[0], input_ids.shape[1],  self.factor_num, self.factor_width, self.dim_factor) # torch.Size([4, 418, self.factor_num, self.factor_width, self.dim_factor])
                std = std.view(input_ids.shape[0], input_ids.shape[1],  self.factor_num, self.factor_width, self.dim_factor) # torch.Size([4, 418, self.factor_num, self.factor_width, self.dim_factor])、
                if self.factor_width == 1 and self.dim_factor == 1:
                    print("Before the self.factor_num, self.factor_width, self.dim_factor is {}, {}, {}".format(mu.shape[2], mu.shape[3], mu.shape[4]))
                    mu = mu.permute(0, 1, 4, 3, 2)
                    std =  std.permute(0, 1, 4, 3, 2)
                    print("Now the self.factor_num, self.factor_width, self.dim_factor is {}, {}, {}".format(mu.shape[2], mu.shape[3], mu.shape[4]))
                index = attention_mask.sum(dim=1) - 1
                print("score is obtained at token id {}".format(input_ids[torch.arange(input_ids.shape[0]), index]))
                save_mu = [mu[i,index[i],:,:,:].tolist() for i in range(mu.shape[0])] # 
                save_std = [std[i,index[i],:,:,:].tolist() for i in range(std.shape[0])] 
                rewards = batch_sequence_rewards.gather(1, index.unsqueeze(-1)).squeeze(1).tolist()
                assert len(rewards) == len(save_mu) == len(save_std)
            else:
                batch_sequence_rewards = self.v_head(hidden_states).squeeze(-1)
                index = attention_mask.sum(dim=1) - 1
                rewards  = batch_sequence_rewards.gather(1, index.unsqueeze(-1)).squeeze(1).tolist()
                # mu =  torch.stack([batch_sequence_rewards] * self.latent_dim, dim=2).view(input_ids.shape[0], input_ids.shape[1], self.factor_num, self.factor_width, self.dim_factor)
                # std = torch.ones_like(mu)
                hidden_states=transformer_outputs[0]
                save_mu = [hidden_states[i,index[i],:].unsqueeze(0).unsqueeze(0).tolist() for i in range(hidden_states.shape[0])]
                save_std = [hidden_states[i,index[i],:].unsqueeze(0).unsqueeze(0).tolist() for i in range(hidden_states.shape[0])]
                # save_mu = [None] * len(rewards)
                # save_std = [None] * len(rewards)
            # index = attention_mask.sum(dim=1) - 1
            # print("score is obtained at token id {}".format(input_ids[torch.arange(input_ids.shape[0]), index]))
            # rewards = batch_sequence_rewards.gather(1, index.unsqueeze(-1)).squeeze(1).tolist()

            # save_mu = [mu[i,index[i],:,:,:].squeeze().tolist() for i in range(mu.shape[0])] # 
            # save_std = [std[i,index[i],:,:,:].squeeze().tolist() for i in range(std.shape[0])] 
            # print("length of save_mu is {}".format(len(save_mu)))
            # print("length of save_mu[0] is {}".format(len(save_mu[0])))
            # print("length of save_mu[0][0] is {}".format(len(save_mu[0][0])))
            
            # batchsize = batch_sequence_rewards.shape[0]
            # end_token_rewards = []
            # batch_input_ids = input_ids
            # for idx in range(batchsize):
            #     sequence_reward = batch_sequence_rewards[idx]
            #     input_ids = batch_input_ids[idx]
            #     seq_len = input_ids.shape[0]
            #     pad_indx = (input_ids == self.tokenizer.pad_token_id).nonzero()
            #     pad_start_indx = pad_indx[self.num_padding_at_beginning] \
            #         if len(pad_indx) > self.num_padding_at_beginning else seq_len
                    
            #     end_token_reward = sequence_reward[pad_start_indx - 1]
            #     end_token_rewards.append(end_token_reward.item())
            # rewards = end_token_rewards
        return rewards, save_mu, save_std



    def forward(self,
                input_ids=None, # torch.Size([4, 418])
                past_key_values=None,
                attention_mask=None, # torch.Size([4, 418])
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                use_cache=False,
                category=None,
                labels=None): # ['helpful', 'harmless', 'helpful', 'harmless']
        loss = None

        if self.config.model_type == "llama":
            kwargs = dict()
        else:
            kwargs = dict(head_mask=head_mask)

        transformer_outputs = self.model(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs) 
        
        if self.use_variational_inference:
            hidden_states = transformer_outputs[0] # torch.Size([4, 418, 4096])
            encode_result = self.encode_head(hidden_states) # torch.Size([4, 418, self.latent_dim*2])
            mu = encode_result[:, :, :self.latent_dim] # torch.Size([4, 418, self.latent_dim])
            logvar = encode_result[:, :, self.latent_dim:] # torch.Size([4, 418, self.latent_dim])
            std = logvar.div(2).exp() # torch.Size([4, 418, self.latent_dim])
            eps = Variable(std.data.new(std.size()).normal_()) # torch.Size([4, 418, self.latent_dim])
            decode_input = mu if self.no_sample else mu + std * eps # torch.Size([4, 418, self.latent_dim])
            rewards = self.decode_head(decode_input).squeeze(-1) # torch.Size([4, 418]) [2*bs, seq]
            
        else:
            hidden_states = transformer_outputs[0] # torch.Size([4, 418, 4096])
            rewards = self.v_head(hidden_states).squeeze(-1)

        # From this line the code is different between official deepspeed code!
        assert len(input_ids.shape) == 2
        bs = input_ids.shape[0] // 2
        seq_len = input_ids.shape[1]
        chosen_ids = input_ids[:bs]  # bs x seq x 1
        rejected_ids = input_ids[bs:] # torch.Size([bs, seq])
        chosen_mask = attention_mask[:bs] # torch.Size([bs, seq])
        rejected_mask = attention_mask[bs:] # torch.Size([bs, seq])
        # Select the index of the first non-eos token index as the final token always br eos_token
        chosen_index = chosen_mask.sum(dim=1) - 1 #torch.Size([bs])
        print("chosen score is obtained at token id {}".format(chosen_ids[torch.arange(chosen_ids.shape[0]), chosen_index]))
        c_truncated_reward= rewards[:bs].gather(1, chosen_index.unsqueeze(-1)).squeeze(1) # torch.Size([bs])
        rejected_index = rejected_mask.sum(dim=1) - 1 #torch.Size([bs])
        print("reject score is obtained at token id {}".format(rejected_ids[torch.arange(rejected_ids.shape[0]), rejected_index]))
        r_truncated_reward = rewards[bs:].gather(1, rejected_index.unsqueeze(-1)).squeeze(1) # torch.Size([bs])
        loss = -torch.log(torch.sigmoid(c_truncated_reward - r_truncated_reward)).mean()
        
        if self.use_variational_inference:
            # compute KL Loss between differnt group
            all_index = torch.cat((chosen_index,rejected_index), dim=0) # torch.Size([2*bs])
            mu_loss = mu[torch.arange(input_ids.shape[0]), all_index, :] # torch.Size([2*bs, self.latent_dim])
            logvar_loss = logvar[torch.arange(input_ids.shape[0]), all_index, :]  # torch.Size([2*bs, self.latent_dim])
            factor_mu = mu_loss.view(input_ids.shape[0], self.factor_width * self.factor_num, self.dim_factor).mean(dim=-1) # torch.Size([2*bs, self.factor_width * self.factor_num])
            factor_logvar = logvar_loss.view(input_ids.shape[0], self.factor_num * self.factor_width, self.dim_factor).mean(dim=-1) # torch.Size([2*bs, self.factor_width * self.factor_num])
            total_kld, dim_wise_kld, mean_kld = kl_divergence(factor_mu, factor_logvar)
            loss_kl = self.beta * total_kld

            # compute internal entropy of every factor group
            if self.internal_entropy:
                mu_group = mu_loss.view(input_ids.shape[0], self.factor_width * self.factor_num, self.dim_factor) # torch.Size([2*bs, self.factor_width * self.factor_num, self.dim_factor])
                mu_internal_entropy = entropy_unnormalized(mu_group)
                var_group = logvar_loss.view(input_ids.shape[0], self.factor_width * self.factor_num, self.dim_factor) # torch.Size([2*bs, self.factor_width * self.factor_num, self.dim_factor])
                var_internal_entropy = entropy_unnormalized(var_group)
                loss_entropy = self.lamda * (mu_internal_entropy + var_internal_entropy)
            else:
                loss_entropy = 0

 
            if self.introduce_supervision:
                latent_state = decode_input[torch.arange(input_ids.shape[0]), all_index, :] # torch.Size([2*bs, self.latent_dim])
                latent_factor = latent_state.view(input_ids.shape[0], self.factor_num, self.factor_width * self.dim_factor).mean(dim=-1) # torch.Size([2*bs, self.factor_num])
                # compute conditional prior loss
                helpful_index = [index for index, element in enumerate(category) if element == 'helpful'] #[0,2]
                harmless_index = [index for index, element in enumerate(category) if element == 'harmless'] #[1,3]
                helpful_factor, helpful_num= latent_factor[helpful_index, 0], int(len(helpful_index)/2) # torch.Size([2]) 默认第一个factor是helpful
                harmless_factor, harmless_num = latent_factor[harmless_index, 1], int(len(harmless_index)/2) # torch.Size([2]) 默认第二个维度是harmless
                loss_conditional_helpful = -torch.log(torch.sigmoid(helpful_factor[:helpful_num] - helpful_factor[helpful_num:])).mean() if helpful_num != 0 else 0
                loss_conditional_harmless = -torch.log(torch.sigmoid(harmless_factor[:harmless_num] - harmless_factor[harmless_num:])).mean() if harmless_num != 0 else 0
                loss_conditional = self.gamma * (loss_conditional_helpful + loss_conditional_harmless)
            else:
                loss_conditional = 0

        else:
            loss_kl, loss_conditional, loss_entropy = 0, 0, 0

        if self.use_simcse:
            index = attention_mask.sum(dim=1) - 1
            last_tokens = torch.stack([hidden_states[i,index[i],:].squeeze() for i in range(hidden_states.shape[0])], dim=0)
            emb1 = last_tokens[:bs]
            emb2 = last_tokens[bs:]
            sim_matrix = F.cosine_similarity(emb1.unsqueeze(1), emb2.unsqueeze(0), dim=-1)
            sim_matrix = sim_matrix / self.opt.simcse_temperature
            labels = torch.arange(bs).long().to(last_tokens.device)
            loss_simcse = self.omega * F.cross_entropy(sim_matrix, labels)
        else:
            loss_simcse = 0

            

        # if self.use_variational_inference:
        #     # print("loss:{}-loss_kl:{}-loss_entropy:{}-loss-conditional:{}".format(loss.item(), loss_kl.item() if self.use_variational_inference else loss_kl, loss_entropy.item() if self.introduce_supervision else loss_entropy, loss_conditional.item() if self.introduce_supervision else loss_conditional))
        print("loss:{}-loss_kl:{}-loss_entropy:{}-loss-conditional:{}-loss-simcse:{}".format(loss.item(), loss_kl.item() if self.use_variational_inference else loss_kl, loss_entropy.item() if self.internal_entropy else loss_entropy, loss_conditional if self.introduce_supervision else loss_conditional, loss_simcse.item() if self.use_simcse else loss_simcse))
    
    
        return {
            "loss": loss + loss_kl + loss_conditional + loss_entropy + loss_simcse,
            "chosen_mean_scores": c_truncated_reward,
            "rejected_mean_scores": r_truncated_reward,
        }

    def forward_value(self,
                      input_ids=None,
                      attention_mask=None,
                      past_key_values=None,
                      position_ids=None,
                      head_mask=None,
                      inputs_embeds=None,
                      return_value_only=False,
                      prompt_length=0,
                      use_cache=False,
                      local_rank=-1,
                      print_example=True):

        if self.config.model_type == "llama":
            kwargs = dict()
        else:
            kwargs = dict(head_mask=head_mask)

        transformer_outputs = self.model(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs)
        hidden_states = transformer_outputs[0]
        if self.use_variational_inference:
            encode_result = self.encode_head(hidden_states) # torch.Size([4, 418, self.latent_dim*2])
            mu = encode_result[:, :, :self.latent_dim] # torch.Size([4, 418, self.latent_dim])
            decode_input = mu  #+ std*eps if not self.no_sample else  mu # torch.Size([4, 418, self.factor_num])
            rewards = self.decode_head(decode_input).squeeze(-1) # torch.Size([4, 418])
        else:
            rewards = self.v_head(hidden_states).squeeze(-1)

        if return_value_only:
            return rewards
        else:
            # [0 0 0 0 prompt, answer, 0 0 0 0 ] for step 3, we have padding at the beginning
            # [prompt, answer, 0, 0, 0, 0] this is normal
            # From this line the code is different between official deepspeed code!
            # assert prompt_length > 1, "prompt_length must be greater than 1 to help select the end score"
            seq_len = input_ids.shape[1]
            eos_pos = torch.argmax(torch.eq(input_ids[:,prompt_length:], self.PAD_ID).int(), dim=1)
            eos_pos += prompt_length # 第一个eos对应的索引
            eos_pos = eos_pos.where(eos_pos != prompt_length, torch.tensor(seq_len, device=eos_pos.device)) # eos_pos 张量中等于 prompt_length 的元素替换为 seq_len
            # print("score is obtianed at token id {}".format(input_ids[torch.arange(input_ids.shape[0]), eos_pos-1]))
            # eos_pos is the final first eos token index in the input_ids
            if self.only_reward_final_token:
                eos_pos = eos_pos - 1
                chosen_rewards = rewards.gather(1, eos_pos.unsqueeze(-1)).squeeze(1)
            else:
                divergence_mask = torch.arange(seq_len).expand_as(input_ids).to(input_ids.device)
                divergence_mask = torch.le(divergence_mask, eos_pos.unsqueeze(1))
                divergence_mask[:,:prompt_length] = 0
                divergence_mask = divergence_mask.to(rewards.dtype)
                chosen_rewards = torch.mul(rewards, divergence_mask).sum(dim=1)
                chosen_rewards = torch.divide(chosen_rewards, divergence_mask.sum(dim=1))
                
            if print_example:
                print_rank_0("++++++++++++++++>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<++++++++++++++++", local_rank)
                print_rank_0("score is obtianed at token id {}".format(input_ids[torch.arange(input_ids.shape[0]), eos_pos]), local_rank)
                print_rank_0("-----------------------", local_rank)
                print_rank_0("an example of input_id: {}".format(self.tokenizer.decode(input_ids[0])), local_rank)
                print_rank_0("-----------------------", local_rank)
                print_rank_0("an example of input_id[prompt_length:c_ind]: {}".format(self.tokenizer.decode(input_ids[0, prompt_length:eos_pos[0]])), local_rank)
                print_rank_0("++++++++++++++++>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<++++++++++++++++", local_rank)
            else:
                print_rank_0("score is obtianed at token id {}".format(input_ids[torch.arange(input_ids.shape[0]), eos_pos]), local_rank)
                print_rank_0("-----------------------", local_rank)
                print_rank_0("an example of input_id: {}".format(self.tokenizer.decode(input_ids[0])), local_rank)

            return {
                "values": rewards,
                "chosen_end_scores": chosen_rewards,
            }

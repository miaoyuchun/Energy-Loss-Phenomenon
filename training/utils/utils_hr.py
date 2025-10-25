from transformers.models.llama.modeling_llama import LlamaMLP
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXMLP
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer, MistralMLP
import torch
from torch.nn import CrossEntropyLoss
def energy_loss(input_data, output_data):

    # if len(input_data.shape) == 3:
    #     assert input_data.shape[0] == 1, "the first dimension for input must be 1"
    #     input_data = input_data.squeeze(0)

    # if len(output_data.shape) == 3:
    #     assert output_data.shape[0] == 1, "the first dimension for output must be 1"
    #     output_data = output_data.squeeze(0)

    output_data = output_data.to(torch.float32)
    input_data = input_data.to(torch.float32)

    # input_energy = (torch.norm(input_data, p=2, dim=-1) **2) / torch.tensor(input_data.size(1)).float()
    # output_energy = (torch.norm(output_data, p=2, dim=-1)**2) / torch.tensor(output_data.size(1)).float()
    input_energy = input_data.abs().mean(dim=-1)
    output_energy = output_data.abs().mean(dim=-1)
    loss = input_energy - output_energy
    return loss

def energy_loss_icml_rebuttal(input_data, output_data):

    # if len(input_data.shape) == 3:
    #     assert input_data.shape[0] == 1, "the first dimension for input must be 1"
    #     input_data = input_data.squeeze(0)

    # if len(output_data.shape) == 3:
    #     assert output_data.shape[0] == 1, "the first dimension for output must be 1"
    #     output_data = output_data.squeeze(0)

    output_data = output_data.to(torch.float32)
    input_data = input_data.to(torch.float32)

    # input_energy = (torch.norm(input_data, p=2, dim=-1) **2) / torch.tensor(input_data.size(1)).float()
    # output_energy = (torch.norm(output_data, p=2, dim=-1)**2) / torch.tensor(output_data.size(1)).float()
    input_energy = input_data.abs().mean(dim=-1)
    output_energy = output_data.abs().mean(dim=-1)
    loss = output_energy
    return loss

def register_mlpblock_energyloss_hooks_last_layer(model, output, attention_mask):
    energyloss_dict = {}
    target = LlamaMLP
    def hook_fn(module, input, output, layer_name): #(torch.Size([1, 1169, 4096]), ) ---> torch.Size([1, 1169, 4096])
        assert input[0].shape[0:2] == output.shape[0:2], "Big Error"
        energyloss_dict[layer_name] = energy_loss(input[0], output)
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, target) and '31' in name:
            handles.append(module.register_forward_hook(lambda self, i, o, name=name: hook_fn(self, i, o, name)))
    _ = model(output, attention_mask, use_cache=False)
    for handle in handles:
        handle.remove() 
    return energyloss_dict


def register_mlpblock_energyloss_hooks_last_layer_misture(model, output, attention_mask):
    energyloss_dict = {}
    target = MistralMLP
    def hook_fn(module, input, output, layer_name): #(torch.Size([1, 1169, 4096]), ) ---> torch.Size([1, 1169, 4096])
        assert input[0].shape[0:2] == output.shape[0:2], "Big Error"
        energyloss_dict[layer_name] = energy_loss(input[0], output)
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, target) and '31' in name:
            handles.append(module.register_forward_hook(lambda self, i, o, name=name: hook_fn(self, i, o, name)))
    _ = model(output, attention_mask, use_cache=False)
    for handle in handles:
        handle.remove() 
    return energyloss_dict


def register_mlp_energyloss_hooks_last_layer(model, output, attention_mask):
    energyloss_dict = {}
    def hook_fn(module, input, output, layer_name): #(torch.Size([1, 1169, 4096]), ) ---> torch.Size([1, 1169, 4096])
        assert input[0].shape[0:2] == output.shape[0:2], "Big Error"
        energyloss_dict[layer_name] = energy_loss(input[0], output)
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and 'lm_head' not in name and '31.mlp' in name:
            handles.append(module.register_forward_hook(lambda self, i, o, name=name: hook_fn(self, i, o, name)))
    _ = model(output, attention_mask, use_cache=False)
    for handle in handles:
        handle.remove() 
    return energyloss_dict

def register_mlp_energyloss_hooks_last_layer_icml_rebuttal(model, output, attention_mask):
    energyloss_dict = {}
    def hook_fn(module, input, output, layer_name): #(torch.Size([1, 1169, 4096]), ) ---> torch.Size([1, 1169, 4096])
        assert input[0].shape[0:2] == output.shape[0:2], "Big Error"
        energyloss_dict[layer_name] = energy_loss_icml_rebuttal(input[0], output)
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and 'lm_head' not in name and '31.mlp' in name:
            handles.append(module.register_forward_hook(lambda self, i, o, name=name: hook_fn(self, i, o, name)))
    _ = model(output, attention_mask, use_cache=False)
    for handle in handles:
        handle.remove() 
    return energyloss_dict

def register_mlp_energyloss_hooks_last_layer_deepseek(model, output, attention_mask):
    energyloss_dict = {}
    def hook_fn(module, input, output, layer_name): #(torch.Size([1, 1169, 4096]), ) ---> torch.Size([1, 1169, 4096])
        assert input[0].shape[0:2] == output.shape[0:2], "Big Error"
        energyloss_dict[layer_name] = energy_loss(input[0], output)
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and 'lm_head' not in name and '29.mlp' in name:
            handles.append(module.register_forward_hook(lambda self, i, o, name=name: hook_fn(self, i, o, name)))
    _ = model(output, attention_mask, use_cache=False)
    for handle in handles:
        handle.remove() 
    return energyloss_dict



def register_mlp_energyloss_hooks_last_layer_mistral(model, output, attention_mask):
    energyloss_dict = {}
    def hook_fn(module, input, output, layer_name): #(torch.Size([1, 1169, 4096]), ) ---> torch.Size([1, 1169, 4096])
        assert input[0].shape[0:2] == output.shape[0:2], "Big Error"
        energyloss_dict[layer_name] = energy_loss(input[0], output)
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and 'lm_head' not in name and '31.mlp' in name:
            handles.append(module.register_forward_hook(lambda self, i, o, name=name: hook_fn(self, i, o, name)))
    _ = model(output, attention_mask, use_cache=False)
    for handle in handles:
        handle.remove() 
    return energyloss_dict


def register_mlp_energyloss_hooks_last_layer_mistral_v6(model, output, attention_mask):
    energyloss_dict = {}
    def hook_fn(module, input, output, layer_name): #(torch.Size([1, 1169, 4096]), ) ---> torch.Size([1, 1169, 4096])
        assert input[0].shape[0:2] == output.shape[0:2], "Big Error"
        energyloss_dict[layer_name] = energy_loss(input[0], output)
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and 'lm_head' not in name and ('31.mlp' in name or '30.mlp' in name or '29.mlp' in name or '28.mlp' in name):
            handles.append(module.register_forward_hook(lambda self, i, o, name=name: hook_fn(self, i, o, name)))
    _ = model(output, attention_mask, use_cache=False)
    for handle in handles:
        handle.remove() 
    return energyloss_dict

def register_mlp_energyloss_hooks_last_layer_mistral_v7(model, output, attention_mask):
    energyloss_dict = {}
    def hook_fn(module, input, output, layer_name): #(torch.Size([1, 1169, 4096]), ) ---> torch.Size([1, 1169, 4096])
        assert input[0].shape[0:2] == output.shape[0:2], "Big Error"
        energyloss_dict[layer_name] = energy_loss(input[0], output)
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and 'lm_head' not in name and ('31.mlp' in name or '30.mlp' in name or '29.mlp' in name or '28.mlp' in name or '27.mlp' in name or '26.mlp' in name):
            handles.append(module.register_forward_hook(lambda self, i, o, name=name: hook_fn(self, i, o, name)))
    _ = model(output, attention_mask, use_cache=False)
    for handle in handles:
        handle.remove() 
    return energyloss_dict

def register_mlp_energyloss_hooks_last_layer_mistral_v8(model, output, attention_mask):
    energyloss_dict = {}
    def hook_fn(module, input, output, layer_name): #(torch.Size([1, 1169, 4096]), ) ---> torch.Size([1, 1169, 4096])
        assert input[0].shape[0:2] == output.shape[0:2], "Big Error"
        energyloss_dict[layer_name] = energy_loss(input[0], output)
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and 'lm_head' not in name and ('31.mlp' in name or '30.mlp' in name):
            handles.append(module.register_forward_hook(lambda self, i, o, name=name: hook_fn(self, i, o, name)))
    _ = model(output, attention_mask, use_cache=False)
    for handle in handles:
        handle.remove() 
    return energyloss_dict


def register_mlp_energyloss_hooks_last_layer_mistral_v9(model, output, attention_mask):
    energyloss_dict = {}
    def hook_fn(module, input, output, layer_name): #(torch.Size([1, 1169, 4096]), ) ---> torch.Size([1, 1169, 4096])
        assert input[0].shape[0:2] == output.shape[0:2], "Big Error"
        energyloss_dict[layer_name] = energy_loss(input[0], output)
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and 'lm_head' not in name and ('29.mlp' in name or '28.mlp' in name):
            handles.append(module.register_forward_hook(lambda self, i, o, name=name: hook_fn(self, i, o, name)))
    _ = model(output, attention_mask, use_cache=False)
    for handle in handles:
        handle.remove() 
    return energyloss_dict


def register_mlp_energyloss_hooks_last_gate_up_layer_mistral(model, output, attention_mask):
    energyloss_dict = {}
    def hook_fn(module, input, output, layer_name): #(torch.Size([1, 1169, 4096]), ) ---> torch.Size([1, 1169, 4096])
        assert input[0].shape[0:2] == output.shape[0:2], "Big Error"
        energyloss_dict[layer_name] = energy_loss(input[0], output)
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and 'lm_head' not in name:
            if '31.mlp.gate_proj' in name or '31.mlp.up_proj' in name:
                handles.append(module.register_forward_hook(lambda self, i, o, name=name: hook_fn(self, i, o, name)))
    _ = model(output, attention_mask, use_cache=False)
    for handle in handles:
        handle.remove() 
    return energyloss_dict

def register_mlp_energyloss_hooks_last_gate_up_layer_mistral_v5(model, output, attention_mask):
    energyloss_dict = {}
    from torch import nn
    act_silu = nn.SiLU()
    def hook_fn(module, input, output, layer_name): #(torch.Size([1, 1169, 4096]), ) ---> torch.Size([1, 1169, 4096])
        assert input[0].shape[0:2] == output.shape[0:2], "Big Error"
        energyloss_dict[layer_name] = {'input': input[0], 'output': output}
        
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and ('31.mlp.up_proj' in name or '31.mlp.gate_proj' in name):
            handles.append(module.register_forward_hook(lambda self, i, o, name=name: hook_fn(self, i, o, name)))
    _ = model(output, attention_mask, use_cache=False)
    for handle in handles:
        handle.remove()
    assert len(energyloss_dict.keys()) == 2 and torch.equal(energyloss_dict['model.layers.31.mlp.up_proj']['input'], energyloss_dict['model.layers.31.mlp.gate_proj']['input']), "Big Error!"
    result = energy_loss(energyloss_dict['model.layers.31.mlp.up_proj']['input'], act_silu(energyloss_dict['model.layers.31.mlp.gate_proj']['output']) * energyloss_dict['model.layers.31.mlp.up_proj']['output'])
    return {"model.layers.31.mlp.up_gate_proj": result}


def register_last_mlp_energyloss_hooks_last_layer(model, output, attention_mask):
    energyloss_dict = {}
    def hook_fn(module, input, output, layer_name): #(torch.Size([1, 1169, 4096]), ) ---> torch.Size([1, 1169, 4096])
        assert input[0].shape[0:2] == output.shape[0:2], "Big Error"
        energyloss_dict[layer_name] = energy_loss(input[0], output)
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and 'lm_head' not in name and '31.mlp.down_proj' in name:
            handles.append(module.register_forward_hook(lambda self, i, o, name=name: hook_fn(self, i, o, name)))
    _ = model(output, attention_mask, use_cache=False)
    for handle in handles:
        handle.remove() 
    return energyloss_dict


# def register_mlpblock_energyloss_hooks_last_layer(model, output, attention_mask):
#     energyloss_dict = {}
#     target = GPTNeoXMLP
#     def hook_fn(module, input, output, layer_name): #(torch.Size([1, 1169, 4096]), ) ---> torch.Size([1, 1169, 4096])
#         assert input[0].shape[0:2] == output.shape[0:2], "Big Error"
#         energyloss_dict[layer_name] = energy_loss(input[0], output)
#     handles = []
#     for name, module in model.named_modules():
#         if isinstance(module, target) and '23' in name:
#             handles.append(module.register_forward_hook(lambda self, i, o, name=name: hook_fn(self, i, o, name)))
#     _ = model(output, attention_mask)
#     for handle in handles:
#         handle.remove() 
#     return energyloss_dict

# def register_mlp_energyloss_hooks_last_layer(model, output, attention_mask):
#     energyloss_dict = {}
#     def hook_fn(module, input, output, layer_name): #(torch.Size([1, 1169, 4096]), ) ---> torch.Size([1, 1169, 4096])
#         assert input[0].shape[0:2] == output.shape[0:2], "Big Error"
#         energyloss_dict[layer_name] = energy_loss(input[0], output)
#     handles = []
#     for name, module in model.named_modules():
#         if isinstance(module, torch.nn.Linear) and 'lm_head' not in name and '23' in name:
#             handles.append(module.register_forward_hook(lambda self, i, o, name=name: hook_fn(self, i, o, name)))
#     _ = model(output, attention_mask)
#     for handle in handles:
#         handle.remove() 
#     return energyloss_dict


IGNORE_INDEX = -100
def compute_ppl(model, logits, labels, nums):
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    loss_fct = CrossEntropyLoss(reduction='none')
    shift_logits = shift_logits.view(-1, model.config.vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)
    loss = loss.view(labels.size(0), -1)  # reshape loss back to sequence length

    batch_ppl = []
    for i, num in enumerate(nums):
        avg_loss = loss[i, -num:].mean()
        batch_ppl.append(torch.exp(avg_loss).float().cpu().item())
    return batch_ppl


@torch.no_grad()
def compute_single_ppl(model, data_list,batch_size=1):
    single_ppl = [float('inf') for _ in range(len(data_list))]
    with torch.no_grad():
        model.eval()
        for i in range(0, len(data_list), batch_size):
            batch = data_list[i:i+batch_size]
            nums = [len(b) - 1 for b in batch]
            inputs = torch.tensor(batch).cuda()
            labels = inputs.clone()
            logits = model(input_ids=inputs)[0]
            batch_ppl = compute_ppl(model, logits, labels, nums)
            single_ppl[i:i+batch_size] = batch_ppl
    return single_ppl

@torch.no_grad()
def compute_pair_ppl(model, data_pairs_list, batch_size=1):
    pair_ppl = [float('inf') for _ in range(len(data_pairs_list))]
    with torch.no_grad():
        model.eval()
        for batch_start in range(0, len(data_pairs_list), batch_size):
            batch_pairs = data_pairs_list[batch_start:batch_start+batch_size]
            nums = [len(response) - 1 for instruction, response in batch_pairs]
            inputs = [instruction + response for instruction, response in batch_pairs]
            inputs = torch.tensor(inputs).cuda()
            labels = torch.tensor([[IGNORE_INDEX] * len(instruction) + response for instruction, response in batch_pairs]).cuda()
            logits = model(input_ids=inputs)[0]
            batch_ppl = compute_ppl(model, logits, labels, nums)
            pair_ppl[batch_start:batch_start+batch_size] = batch_ppl
    return pair_ppl



def trim_special_tokens(input_ids, tokenizer):
    """
    去除列表两端的 pad 和 eos token。

    参数：
    - input_ids (list): 输入的token ID列表
    - pad_token_id (int): pad token的ID，默认是0
    - eos_token_id (int): eos token的ID，默认是1

    返回：
    - list: 去除pad和eos token后的有效token列表
    """
    start = 0
    end = len(input_ids)

    # 找到从前往后第一个非pad、非eos的位置
    while start < end and (input_ids[start] == tokenizer.pad_token_id or input_ids[start] == tokenizer.eos_token_id):
        start += 1

    # 找到从后往前第一个非pad、非eos的位置
    while end > start and (input_ids[end - 1] == tokenizer.pad_token_id or input_ids[end - 1] == tokenizer.eos_token_id):
        end -= 1

    # 返回去除特殊token后的有效部分
    return input_ids[start:end]



@torch.no_grad()
def compute_dst(model, prompts, seq, tokenizer):
    model.eval()
    start_idx = prompts.shape[1] - 1
    final_idx = seq.shape[1] - 2
    response_list = seq[:,start_idx:final_idx].tolist()
    instruction_list = prompts.tolist()
    clean_response_list = [trim_special_tokens(res, tokenizer) for res in response_list]
    clean_instruction_list = [trim_special_tokens(ins, tokenizer) for ins in instruction_list]
    all_data_list = [(a, b) for a, b in zip(clean_instruction_list, clean_response_list)]

    single_ppl = compute_single_ppl(model, clean_response_list)
    # print(single_ppl)
    pair_ppl = compute_pair_ppl(model, all_data_list)
    # print(pair_ppl)
    assert len(single_ppl) == len(pair_ppl), "Error"
    dst = []
    for i in range(len(single_ppl)):
        dst.append((single_ppl[i] - pair_ppl[i])/single_ppl[i])

    return torch.tensor(dst).cuda()
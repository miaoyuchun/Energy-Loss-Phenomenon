# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Part of the code was adopted from https://github.com/microsoft/Megatron-DeepSpeed/blob/main/megatron/data/dataset_utils.py
"""
import torch
import os
import re
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset, load_from_disk
from itertools import chain
from . import new_datasets
import json
import hashlib
import datasets
import math
import random
IGNORE_INDEX=-100
from typing import Any, Dict, List, Sequence, Tuple

def get_raw_dataset(
    dataset_name, 
    output_path, 
    seed, 
    local_rank, 
    recipe:str="", 
    prompt_input=False, 
    cache_dir="/mnt/data/instruction_data/cache/", 
    adding_demonstrations=True,
    template_path=None,
    end_of_conversation="</s>"
    ):
    if dataset_name == "ours/sample_dialog_sft":
        assert recipe != "", "recipe can't be empty string"
        return new_datasets.Sample_Dialog_SFT_Dataset(
            recipe, output_path, seed, local_rank, prompt_input, cache_dir, 
            adding_demonstrations, template_path, end_of_conversation)
    #adding our prepared datasets for stage 2&3
    elif dataset_name == "ours/sample_alignment_stage2":
        assert recipe != "", "recipe can't be empty string"
        return new_datasets.Sampled_Alignment_Stage2_Dataset(
            recipe, output_path, seed, local_rank, prompt_input, cache_dir, template_path, end_of_conversation)
    elif dataset_name == "ours/sample_alignment_stage4_no_template":
        assert recipe != "", "recipe can't be empty string"
        return new_datasets.Sampled_Alignment_Stage4_Dataset_no_Template(
            recipe, output_path, seed, local_rank, prompt_input, cache_dir, template_path, end_of_conversation)
    #adding our prepared instruction-tuning datasets
    else:
        raise RuntimeError(
            f"We do not have configs for dataset {dataset_name}, but you can add it by yourself in raw_datasets.py."
        )

class PromptDataset(Dataset):

    def __init__(self, prompt_dataset, chosen_dataset, reject_dataset,
                 pad_token_id, train_phase, args=None) -> None:
        super().__init__()
        self.prompt_dataset = prompt_dataset
        self.chosen_dataset = chosen_dataset
        self.reject_dataset = reject_dataset
        self.pad_token_id = pad_token_id
        self.train_phase = train_phase
        self.args = args
        if self.args!=None and self.args.reverse and self.train_phase == 3:
            print(f"reverse with seed {self.args.sampler_seed}!!!")
            global_random = random.Random(self.args.sampler_seed)
            self.numbers = list(range(0, len(self.prompt_dataset)))
            global_random.shuffle(self.numbers)
        else:
            self.numbers = None

    def __len__(self):
        if self.train_phase == 3:
            length = len(self.prompt_dataset)
        else:
            length = len(self.chosen_dataset)
        return length

    def __getitem__(self, idx):
        if self.train_phase == 1:
            input_ids = torch.tensor(self.chosen_dataset[idx]["input_ids"], dtype=torch.long)
            attention_mask = torch.tensor(self.chosen_dataset[idx]["attention_mask"], dtype=torch.long)
            labels = input_ids.clone() if "labels" not in self.chosen_dataset[idx] \
                else torch.tensor(self.chosen_dataset[idx]["labels"], dtype=torch.long)
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }
        elif self.train_phase == 2 or self.train_phase == 4:
            data  = self.chosen_dataset[idx]
            return data
        elif self.train_phase == 3:
            if self.numbers != None:
                # idx = len(self.prompt_dataset) - idx -1
                idx = self.numbers[idx]
            if 'dataset' in self.prompt_dataset[idx]:
                print(self.prompt_dataset[idx]['dataset'])
            return torch.tensor(self.prompt_dataset[idx]["input_ids"]), \
                torch.tensor(self.prompt_dataset[idx]["attention_mask"])


def create_dataset(args, dataset_name, train_phase, tokenizer, fname, end_of_conversation_token):
    datasets.disable_caching()
    raw_dataset = get_raw_dataset(
        dataset_name, 
        args.data_output_path, 
        args.seed, 
        args.local_rank, 
        args.recipe, 
        args.prompt_input, 
        args.cache_dir,
        args.adding_demonstrations,
        args.template_path,
        end_of_conversation_token
    )
    train_dataset = raw_dataset.get_train_data() 
    eval_dataset = raw_dataset.get_eval_data()

    def process_data(tmp_data):
        if train_phase == 1:
            if "prefix_instruction" in args and args.prefix_instruction:
                sentence, prefix_sentence = raw_dataset.get_prompt_and_chosen_and_prefix(tmp_data)
            else:
                sentence = raw_dataset.get_prompt_and_chosen(tmp_data)
            sentence  = sentence.strip()#.replace(end_of_conversation_token, ' '+end_of_conversation_token).strip()
            assert len(sentence) > 0, "the chosen sentence can't be empty"
            
            add_bos = False
            if tokenizer(sentence)['input_ids'][0] != tokenizer.bos_token_id:
                sentence = tokenizer.bos_token + sentence
                add_bos = True # bos have been added to the setence
            chosen_token = tokenizer(sentence,return_tensors="pt")
            chosen_token = {key: value.squeeze() for key, value in chosen_token.items()}
            target = chosen_token["input_ids"].clone()
            turns = sentence.split(args.end_of_conversation_token)
            if turns[0] == '':
                assert add_bos, "begin with end of conversation!!!"
                # Now bos = end of conversation
                del(turns[0])
                turns[0] = args.end_of_conversation_token + turns[0]
            cur_len = 1
            target[:cur_len] = IGNORE_INDEX
            for i, turn in enumerate(turns):
                if i == 0:
                    turn_len = len(tokenizer(turn).input_ids) # considering the end of conversation token
                else:
                    turn_len = len(tokenizer(turn).input_ids) if not add_bos else len(tokenizer(turn).input_ids) + 1 # considering the end of conversation token

                if turn == "" or cur_len + turn_len > args.max_seq_len:
                    break
                parts = turn.split(args.template_start_end[1])
                if len(parts) != 2:
                    print(tmp_data['index'])
                    assert len(parts) == 2, "Big error at line 200!"
                parts[0] += args.template_start_end[1]
                if i == 0:
                    instruction_len = len(tokenizer(parts[0]).input_ids) - 2 # one for bos, the other one for endding kongge that will combied with the next token in targets
                else:
                    instruction_len = len(tokenizer(parts[0]).input_ids) - 2 if not add_bos else len(tokenizer(parts[0]).input_ids) - 1
                assert tokenizer.decode(target[cur_len : cur_len + instruction_len]).strip().endswith(args.template_start_end[1].strip()), "Errors in IGNORE_TOKEN_ID"
                if i != 0:
                    assert tokenizer.decode(target[cur_len : cur_len + instruction_len]).strip().startswith(args.template_start_end[0].strip()), "Errors in IGNORE_TOKEN_ID"
                target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
                cur_len += turn_len
            target[cur_len:] = IGNORE_INDEX
            chosen_token["labels"] = target
            chosen_token = {key: value[:cur_len].tolist() for key, value in chosen_token.items()}

            return chosen_token

        elif train_phase == 3:
            prompt = raw_dataset.get_prompt(tmp_data).rstrip()
            if prompt is not None:
                if not hasattr(tokenizer, 'add_bos_token'):
                    prompt = tokenizer.bos_token + prompt
                prompt_token = tokenizer(prompt, max_length=args.max_seq_len, truncation=True, return_tensors="pt")
                # for key_word in ["input_ids", "attention_mask"]:
                y = prompt_token["input_ids"].squeeze(0)
                c = prompt_token['attention_mask'].squeeze(0)
                if y.size(0) >= args.max_seq_len:
                    input_text = tokenizer.decode(y, skip_special_tokens=True)
                    start = args.template_start_end[0].strip()
                    end = args.template_start_end[1].strip()
                    start_matches = re.findall(start, input_text)
                    end_matches = re.findall(end, input_text)
                    if len(start_matches) > len(end_matches):
                        end_tokens = tokenizer(end, return_tensors="pt")
                        y = torch.cat((y, end_tokens['input_ids'].squeeze(0)), dim=0)
                        c = torch.ones_like(y, dtype=prompt_token['attention_mask'].dtype)
                        # prompt_token['attention_mask'] = attention_mask
                        reformat_input = tokenizer.decode(y, skip_special_tokens=True)
                        print(f"exceed max seq len\n{input_text} with reformat\n{reformat_input}\n")
                prompt_token["input_ids"] = y.flip(0)
                prompt_token["attention_mask"] = c.flip(0)
                if 'dataset' in tmp_data:
                    prompt_token['dataset'] = tmp_data['dataset']
                return prompt_token
            else:
                print(f"Big error!!!! prompt is None!")
                exit(0)

    def process_all_reponses_and_ranks(tmp_data):
        # tmp_data = {key: values[0] for key, values in tmp_data.items()}
        # Rewrite by baorong in 2023-10-30 to sample the rank response data
        responses_and_ranks = raw_dataset.get_prompt_and_all_response_and_ranks(tmp_data)
        if responses_and_ranks is not None:
            # Do not all end of conversation token because we already adding them in get_prompt_and_all_response_and_ranks
            if not hasattr(tokenizer, 'add_bos_token'):
                inputs = [tokenizer.bos_token + input_  for input_ in list(list(zip(*responses_and_ranks))[0])]
            else:
                inputs = [input_  for input_ in list(list(zip(*responses_and_ranks))[0])]
            ranks = list(list(zip(*responses_and_ranks))[1])
            ranks = [float(ranks[idx]) for idx in range(len(ranks))]
            responses_tokens = tokenizer(inputs)
            # print(f"length of inputs is {len(inputs)} and close_power_2 is {close_power_2} and ranks {ranks}")
            all_tuples = [(x, y) for x in range(0,len(ranks)) for y in range(x+1,len(ranks))]
            filter_tuples = [data for data in all_tuples if ranks[data[0]] < ranks[data[1]]]
            print("{}-{}".format(ranks, len(filter_tuples)))
            close_power_2 = 2 ** math.floor(math.log(len(filter_tuples), 2))
            chosen_tuples = random.sample(filter_tuples, min(close_power_2, args.max_reward_samples))
            result = {
                "chose_ids":[],
                "reject_ids":[],
                "chose_mask":[],
                "reject_mask":[],
                "category":tmp_data['category']
            }
            for x, y in chosen_tuples:
                assert ranks[x] < ranks[y], "the rank of x must be smaller than y"
                result["chose_ids"].append(responses_tokens["input_ids"][x][-args.max_seq_len:])
                result["chose_mask"].append(responses_tokens["attention_mask"][x][-args.max_seq_len:])
                result["reject_ids"].append(responses_tokens["input_ids"][y][-args.max_seq_len:])
                result["reject_mask"].append(responses_tokens["attention_mask"][y][-args.max_seq_len:])
            # result = [{key: value[i] for key, value in result.items()} for i in range(len(chosen_tuples))]
            return result
        
    def process_all_reponses_and_ranks_and_labels(tmp_data):
        # tmp_data = {key: values[0] for key, values in tmp_data.items()}
        # Rewrite by baorong in 2023-10-30 to sample the rank response data
        prompt, responses_and_ranks = raw_dataset.get_prompt_and_all_response_and_ranks_and_labels(tmp_data)
        if responses_and_ranks is not None:
            # Do not all end of conversation token because we already adding them in get_prompt_and_all_response_and_ranks
            if hasattr(tokenizer, 'add_bos_token'):
                tokenizer.add_bos_token = False
            if not prompt.startswith(tokenizer.bos_token):
                prompt = tokenizer.bos_token + prompt

            responses = list(list(zip(*responses_and_ranks))[0])
            ranks = list(list(zip(*responses_and_ranks))[1])
            assert ranks == [0, 1], "big errors!!!"
            prompt_ids= tokenizer(prompt)
            chosen_ids = tokenizer(responses[0])
            rejected_ids = tokenizer(responses[1])
            results = {
                "prompt_ids": prompt_ids["input_ids"],
                "chosen_ids": chosen_ids["input_ids"],
                "rejected_ids": rejected_ids["input_ids"],
            }
            return results
            
    def concat_prompt_and_chosen_stage1(examples):
        text_column_name = ['input_ids', 'attention_mask', 'labels']
        from itertools import zip_longest
        zipped_list = zip_longest(*[examples[key] for key in text_column_name], fillvalue="")
        data = {"input_ids": [], "attention_mask": [], "labels": []}
        input_ids, attention_mask, labels = [], [], []
        for idsattnlabels in zipped_list:
            if len(input_ids) + len(idsattnlabels[0]) > args.max_seq_len:
                data["input_ids"].append(input_ids)
                data["attention_mask"].append(attention_mask)
                data["labels"].append(labels)
                input_ids, attention_mask, labels = idsattnlabels[0], idsattnlabels[1], idsattnlabels[2]
            else:
                input_ids += idsattnlabels[0]
                attention_mask += idsattnlabels[1]
                labels += idsattnlabels[2]
        if len(input_ids) > 0:
            data["input_ids"].append(input_ids)
            data["attention_mask"].append(attention_mask)
            data["labels"].append(labels)
        return data

    def get_dataset(current_dataset, split):
        prompt_dataset = None
        chosen_dataset = None
        reject_dataset = None
        if train_phase==1:
            skip_columns = ["input_ids", "labels", "attention_mask"]
            original = len(current_dataset)
            current_dataset = current_dataset.filter(lambda x: args.end_of_conversation_token not in ''.join([item['instruction'] + item['output'] for item in x['conversations']]))
            print("debug info: {} samples are filtered".format(len(current_dataset) - original))
            mid_dataset=current_dataset.map(
                process_data,
                num_proc=args.num_workers,
                remove_columns=[name for name in current_dataset.column_names if name not in skip_columns],
                desc="Tokenizing and reformatting instruction data"
            )
            if args.concat_prompt:
                chosen_dataset = mid_dataset.map(
                    concat_prompt_and_chosen_stage1,
                    num_proc=args.num_workers,
                    batched=True,
                    remove_columns=mid_dataset.column_names,
                    desc="Tokenizing and reformatting instruction data to max_seq_len"
                )
            else:
                chosen_dataset = mid_dataset

        elif train_phase==2:
            print(f"Debug info we are now enter line 339 for process_all_reponses_and_ranks")
            skip_columns = ["category", "chose_ids", "chose_mask", "reject_ids", "reject_mask", "ranks"]
            chosen_dataset = current_dataset.map(
                process_all_reponses_and_ranks,
                batched=False,
                num_proc=args.num_workers,
                remove_columns=[name for name in current_dataset.column_names if name not in skip_columns],
                desc="Tokenizing and reformatting alignment data"
            )
            # print("the original length of dataset is {}".format(len(current_dataset)))
            # chosen_dataset = current_dataset.map(
            #     process_all_reponses_and_ranks,
            #     batched=True,
            #     num_proc=args.num_workers,
            #     remove_columns=[name for name in current_dataset.column_names if name not in skip_columns],
            #     batch_size=1,
            #     desc="Tokenizing and reformatting alignment data"
            # )
            # print("the after-process length of dataset is {}".format(len(chosen_dataset)))

        elif train_phase==4:
            skip_columns = ["prompt_ids","chosen_ids","rejected_ids"]
            chosen_dataset = current_dataset.map(
                process_all_reponses_and_ranks_and_labels,
                batched=False,
                num_proc=args.num_workers,
                remove_columns=[name for name in current_dataset.column_names if name not in skip_columns],
                desc="Tokenizing and reformatting alignment data"
            )        
            original = len(chosen_dataset)
            chosen_dataset = chosen_dataset.filter(lambda x: max(len(x['prompt_ids'])+len(x['chosen_ids']), len(x['prompt_ids'])+len(x['rejected_ids'])) < args.max_seq_len)
            print("debug info: {} samples are filtered due to length".format(len(chosen_dataset) - original))
        elif train_phase ==3:
            skip_columns = ["input_ids", "attention_mask", "dataset"]
            prompt_dataset=current_dataset.map(
                process_data,
                num_proc=args.num_workers,
                remove_columns=[name for name in current_dataset.column_names if name not in skip_columns],
                desc="Tokenizing and reformatting stage 3 data"
            )

        if prompt_dataset is not None:
            prompt_dataset.save_to_disk("{}/prompt_dataset.{}".format(fname, split))
        if chosen_dataset is not None:
            chosen_dataset.save_to_disk("{}/chosen_dataset.{}".format(fname, split))
        if reject_dataset is not None:
            reject_dataset.save_to_disk("{}/reject_dataset.{}".format(fname, split))

        return PromptDataset(prompt_dataset=prompt_dataset, 
                             chosen_dataset=chosen_dataset, 
                             reject_dataset=reject_dataset,
                             pad_token_id=tokenizer.pad_token_id, 
                             train_phase=train_phase) 
        
    train_dataset=get_dataset(train_dataset, "train")
    eval_dataset=get_dataset(eval_dataset, "eval")

    return train_dataset, eval_dataset

def create_prompt_dataset(args, train_phase, tokenizer, end_of_conversation_token):
    """
    Creates the prompt dataset
    """
    datasets.disable_caching()
    os.makedirs(args.data_output_path, exist_ok=True)
    fname = '_'.join(args.data_path)
    if "recipe" not in args or args.recipe == "":
        if "prefix_instruction" in args and args.prefix_instruction:
            fname = f"{fname}_split{args.data_split}_phase{train_phase}_seed{args.seed}_seqlen{args.max_seq_len}_prefix"
        else:
            fname = f"{fname}_split{args.data_split}_phase{train_phase}_seed{args.seed}_seqlen{args.max_seq_len}"
    else:
        recipe = json.load(open(args.recipe,"r"))
        recipe_str = json.dumps(recipe) + str(tokenizer) + str(end_of_conversation_token)
        hash_value = hashlib.md5(recipe_str.encode("utf-8")).hexdigest()
        # NOTE: from step 2
        if "prefix_instruction" in args and args.prefix_instruction:
            fname = fname = f"{fname}_split{args.data_split}_phase{train_phase}_seed{args.seed}_seqlen{args.max_seq_len}_prefix_{hash_value}_useunk_{args.use_unk}"
        else:
            fname = f"{fname}_split{args.data_split}_phase{train_phase}_seed{args.seed}_seqlen{args.max_seq_len}_ignore_prefix_{hash_value}_useunk_{args.use_unk}"
        if args.adding_demonstrations:
            fname=fname+"_with_demonstrations"

    fname = '_'.join(fname.split('/'))
    fname = f"{args.data_output_path}/{fname}"
    
    cache_found = None
    if train_phase==1 or train_phase==2 or train_phase==4:
        train_fname = f"{fname}/chosen_dataset.train"
        eval_fname = f"{fname}/chosen_dataset.eval"
        cache_found = os.path.exists(train_fname) and os.path.exists(eval_fname)

    elif train_phase==3:
        train_fname = f"{fname}/prompt_dataset.train"
        eval_fname = f"{fname}/prompt_dataset.eval"
        cache_found = os.path.exists(train_fname) and os.path.exists(eval_fname)
    
    print(f'debug info: file_fname is {fname} with cache_found {cache_found}')
    cache_success = 1
    # create_dataset(args, args.data_path[0], train_phase, tokenizer, fname, end_of_conversation_token)
    if torch.distributed.get_rank() == 0 and not cache_found:
        assert len(args.data_path) == 1, "only support one dataset for now"
        train_dataset, eval_dataset = \
            create_dataset(args, args.data_path[0], train_phase, tokenizer, fname, end_of_conversation_token)
    
        # try:
        #     train_dataset, eval_dataset = \
        #         create_dataset(args, args.data_path[0], train_phase, tokenizer, fname, end_of_conversation_token)
        # except Exception as e:
        #     print(f"create_dataset failed with exception {e}")
        #     cache_success = 0
    
    counts = torch.LongTensor([cache_success]).to('cuda')
    torch.distributed.all_reduce(counts)
    if counts[0].item() != torch.distributed.get_world_size():
        print("Data index creation unsuccessful, exiting.")
        exit()

    train_dataset, eval_dataset=None, None
    prompt_dataset_train, chosen_dataset_train, reject_dataset_train = None, None, None
    prompt_dataset_eval, chosen_dataset_eval, reject_dataset_eval = None, None, None

    if train_phase==1 or train_phase==2 or train_phase==4:
        chosen_dataset_train=load_from_disk(train_fname)
        chosen_dataset_eval=load_from_disk(eval_fname)
    elif train_phase==3:
        prompt_dataset_train=load_from_disk(train_fname)
        prompt_dataset_eval=load_from_disk(eval_fname)

    train_dataset=PromptDataset(prompt_dataset_train, chosen_dataset_train, reject_dataset_train, tokenizer.bos_token_id, train_phase, args)
    eval_dataset=PromptDataset(prompt_dataset_eval, chosen_dataset_eval, reject_dataset_eval, tokenizer.bos_token_id, train_phase, args)
    
    return train_dataset, eval_dataset

class DataCollatorSFT:
    
    def __init__(self, attention_pad=0, pad_token_id=0, ingor_index = -100):
        self.attention_pad = attention_pad
        self.pad_token_id = pad_token_id
        self.ignore_index = ingor_index
        
    def __call__(self, data):
        input_ids = pad_sequence(
            [f['input_ids'] for f in data], 
            padding_value=self.pad_token_id, 
            batch_first=True
        )
        attention_mask = pad_sequence(
            [f["attention_mask"] for f in data],
            padding_value=self.attention_pad,
            batch_first=True
        )
        labels = pad_sequence(
            [f["labels"] for f in data],
            padding_value=self.ignore_index,
            batch_first=True
        )
        return {"input_ids": input_ids, 
                "attention_mask": attention_mask, 
                "labels": labels
                }

class DataCollatorReward:
    def __init__(self, tokenizer, pad_token_id=0, ignore_index=-100):
        self.tokenizer = tokenizer
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index

    def __call__(self, samples):
        # print(f"debug info: samples length {len(samples)} is {samples}")
        # if len(samples[0]["chose_ids"]) > 1:
        #     print("length of x[chose_ids] is {}".format(len(samples[0]["chose_ids"])))
        chose_ids = [x["chose_ids"][0] for x in samples]
        reject_ids = [x["reject_ids"][0] for x in samples]
        category_list = [x['category'] for x in samples] * 2
        input_ids = chose_ids + reject_ids
        input_ids = pad_sequence(
            [torch.tensor(x, dtype=torch.long) for x in input_ids],
            padding_value=self.pad_token_id,
            batch_first=True
        )
        chose_mask = [x["chose_mask"][0] for x in samples]
        reject_mask = [x["reject_mask"][0] for x in samples]
        attention_mask = chose_mask + reject_mask
        attention_mask = pad_sequence(
            [torch.tensor(x, dtype=torch.long) for x in attention_mask],
            padding_value=0,
            batch_first=True
        )
        assert attention_mask.size(1) == input_ids.size(1)
        assert attention_mask.size(0) == input_ids.size(0) == len(category_list)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "category": category_list
            }

class DataCollatorRLHF:
    # This RLHF data collator is totally different from deepspeed team original code!
    def __init__(self, pad_token_id=0, inference_tp_size=1):
        self.pad_token_id = pad_token_id
        self.inference_tp_size = inference_tp_size

    def __call__(self, data):
        batch = {}
        prompt = pad_sequence([f[0] for f in data],
                              padding_value=self.pad_token_id,
                              batch_first=True)
        prompt_mask = pad_sequence([f[1] for f in data],
                                   padding_value=0,
                                   batch_first=True)
        batch["prompt"] = prompt.flip(1)
        batch["prompt_att_mask"] = prompt_mask.flip(1)
        return batch

class DataCollatorDPO:
    def __init__(self, tokenizer, pad_token_id=0, ignore_index=-100):
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index
        self.return_tensors = 'pt'

    def _pad_labels(self, batch: torch.Tensor, positions: List[Tuple[int, int]]) -> torch.Tensor:
        padded_labels = []
        for feature, (prompt_len, answer_len) in zip(batch, positions):
            # if self.tokenizer.padding_side == "left":
            #     start, end = feature.size(0) - answer_len, feature.size(0)
            # else:
            #     start, end = prompt_len, prompt_len + answer_len
            start, end = prompt_len, prompt_len + answer_len
            padded_tensor = self.ignore_index * torch.ones_like(feature)
            padded_tensor[start:end] = feature[start:end]
            padded_labels.append(padded_tensor)
        return torch.stack(padded_labels, dim=0).contiguous()  # in contiguous memory
    
    def __call__(self, features):
        r"""
        Pads batched data to the longest sequence in the batch.

        We generate 2 * n examples where the first n examples represent chosen examples and
        the last n examples represent rejected examples.
        """
        concatenated_features = []
        label_positions = []
        for key in ("chosen_ids", "rejected_ids"):
            for feature in features:
                prompt_len, answer_len = len(feature["prompt_ids"]), len(feature[key])
                concatenated_features.append(
                    {
                        "input_ids": feature["prompt_ids"] + feature[key],
                        "attention_mask": [1] * (prompt_len + answer_len),
                    }
                )
                label_positions.append((prompt_len, answer_len))

        batch = self.tokenizer.pad(
            concatenated_features,
            return_tensors=self.return_tensors,
        )
        batch["labels"] = self._pad_labels(batch["input_ids"], label_positions)
        return batch


def get_unsupervised_data(args, tokenizer):
    unsupervised_raw_datasets = load_dataset(
        args.unsupervised_dataset_name, args.unsupervised_dataset_config_name)
    column_names = unsupervised_raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    tokenized_datasets = unsupervised_raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=args.num_workers,
        remove_columns=column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )

    block_size = args.max_prompt_seq_len + args.max_answer_seq_len

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {
            k: list(chain(*examples[k]))
            for k in examples.keys()
        }
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k:
            [t[i:i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=args.num_workers,
        load_from_cache_file=True,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    train_dataset = lm_datasets["train"]

    return train_dataset


class MiniDataset:

    def __init__(self, max_size, small_batch_size):
        self.dataset = []
        self.max_size = max_size
        self.small_batch_size = small_batch_size

    def seperate(self):
        small_dataset = []
        for large_batch in self.dataset:
            if type(large_batch) == list or type(large_batch) == tuple:
                large_size = len(large_batch[0])
            elif type(large_batch) == dict:
                large_size = len(large_batch[list(large_batch.keys())[0]])
            else:
                large_size = len(large_batch)
            for i in range(0, large_size, self.small_batch_size):
                if type(large_batch) == list or type(large_batch) == tuple:
                    small_dataset.append(
                        [x[i:i + self.small_batch_size] for x in large_batch])
                elif type(large_batch) == dict:
                    small_dataset.append({
                        k: v[i:i + self.small_batch_size]
                        for k, v in large_batch.items()
                    })
                else:
                    small_dataset.append(large_batch[i:i +
                                                     self.small_batch_size])
        self.free()

        return small_dataset

    def add(self, data):
        if len(self.dataset) < self.max_size:
            self.dataset.append(data)
            if len(self.dataset) == self.max_size:
                return self.seperate()
            else:
                return None
        else:
            raise ValueError(
                "The dataset is full but we did not stop it. There is a bug in the code."
            )

    def free(self):
        self.dataset = []


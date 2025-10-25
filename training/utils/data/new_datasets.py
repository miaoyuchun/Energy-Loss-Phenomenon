# coding=utf-8
# Copyright (c) 2023, Initi AI.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from datasets import load_dataset, concatenate_datasets, load_from_disk, Dataset
from torch.utils.data import Subset
import re
import random
import json
import warnings
import hashlib
import os
import copy
import pprint

PROMPT_DICT = {
    "prompt_input_chosen": "Human:\n{instruction} {input}\nAssistant:\n{output}",
    "prompt_no_input_chosen": "Human:\n{instruction}\nAssistant:\n{output}",
    "prompt_input_instruct": "Human:\n{instruction} {input}\nAssistant:\n",
    "prompt_no_input_instruct": "Human:\n{instruction}\nAssistant:\n"
}
PROMPT_DICT_FEW_SHOT = {
    "prompt_input_chosen": "Human:\n{instruction} {demonstrations} {input}\nAssistant:\n{output}",
    "prompt_no_input_chosen": "Human:\n{instruction} {demonstrations}\nAssistant:\n{output}",
    "prompt_input_instruct": "Human:\n{instruction} {demonstrations} {input}\nAssistant:\n",
    "prompt_no_input_instruct": "Human:\n{instruction} {demonstrations}\nAssistant:\n"
}

DEFAULT_SYS_PROMPT="""<<SYS>>You are a helpful, respectful and honest assistant. Always answer as helpfully as possible. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don’t know the answer to a question, please don’t share false information.\n<</SYS>>\n"""
# The template prompt dataset class that all new dataset porting needs to
# follow in order to have a unified API and unified data format.


class PromptRawDataset(object):

    def __init__(self, output_path, seed, local_rank):
        self.output_path = output_path
        self.seed = seed
        self.local_rank = local_rank

    def get_train_data(self):
        return

    def get_eval_data(self):
        return

    def get_prompt(self, sample):
        return

    # The chosen response should be in the format of: " " + actual_response_sentence
    def get_chosen(self, sample):
        return

    # The rejected response should be in the format of: " " + actual_response_sentence
    # If the dataset does not have rejected response, return None
    def get_rejected(self, sample):
        return

    def get_prompt_and_chosen(self, sample):
        return

    def get_prompt_and_chosen_and_prefix(self, sample):
        return

    def get_prompt_and_rejected(self, sample):
        return

class RewriteFlanv2RawDataset_Stage3(PromptRawDataset):
    """
    Flanv2 dataset before and after rewriting (2023-08-08)
    """
    def __init__(self,
                 output_path=None,  
                 seed=None, 
                 local_rank=None, 
                 prompt_input=False, 
                 adding_demonstrations=True,
                 template_path=None, 
                 rewrite_type="new",
                 flan_task=None,
                 flan_source=None):
        super().__init__(output_path, seed, local_rank)
        self.seed = 6666 if seed is None else seed
        if rewrite_type is None or rewrite_type not in ["origin", "new"]:
            raise ValueError("--rewrite_type is not correct")
        self.rewrite_type = rewrite_type
        self.dataset_name = "flan_rewrite_dataset"
        self.dataset_name_clean = "flan_rewrite_dataset"
        self.path = "/mnt/data/alignment_data/use_final_2048/flanv2_new"
        self.flan_task = flan_task
        self.flan_source = flan_source
        self.data_path=["new_flan_prompt.jsonl"]
        
        print_dp = ", ".join(self.data_path)
        print(f"********\n => data path: {print_dp}")
        
        self.raw_datasets = self.read_raw_data()
        self.split_dataset(eval_ratio=0.02)
        
        self.prompt_input = prompt_input
        self.adding_demonstrations = adding_demonstrations
        
        print("******\n =>: template_path: {}".format(template_path))

        if template_path is None:
            self.prompt_input_chosen, self.prompt_no_input_chosen = PROMPT_DICT[
                "prompt_input_chosen"], PROMPT_DICT["prompt_no_input_chosen"]
            self.prompt_input_instruct, self.prompt_no_input_instruct = PROMPT_DICT[
                "prompt_input_instruct"], PROMPT_DICT["prompt_no_input_instruct"]

            self.prompt_input_chosen_few_shot, self.prompt_no_input_chosen_few_shot = PROMPT_DICT_FEW_SHOT[
                "prompt_input_chosen"], PROMPT_DICT_FEW_SHOT["prompt_no_input_chosen"]
            self.prompt_input_instruct_few_shot, self.prompt_no_input_instruct_few_shot = PROMPT_DICT_FEW_SHOT[
                "prompt_input_instruct"], PROMPT_DICT_FEW_SHOT["prompt_no_input_instruct"] 
        else:
            with open(template_path, "r") as r:
                template_data=json.load(r) 

            self.prompt_input_chosen, self.prompt_no_input_chosen = template_data[0][
                "prompt_input_chosen"], template_data[0]["prompt_no_input_chosen"]
            self.prompt_input_instruct, self.prompt_no_input_instruct = template_data[0][
                "prompt_input_instruct"], template_data[0]["prompt_no_input_instruct"]

            self.prompt_input_chosen_few_shot, self.prompt_no_input_chosen_few_shot = template_data[1][
                "prompt_input_chosen"], template_data[1] ["prompt_no_input_chosen"]
            self.prompt_input_instruct_few_shot, self.prompt_no_input_instruct_few_shot = template_data[1][
                "prompt_input_instruct"], template_data[1]["prompt_no_input_instruct"] 
            

    def read_raw_data(self):
        split_datasets=[]
        for data_path in self.data_path:
            if data_path.split(".")[-1] == "json" or data_path.split(".")[-1] == "jsonl":
                data_path=os.path.join(self.path, data_path)
                sub_dataset = load_dataset("json",data_files=data_path)["train"]
                
                # Select specific flan_task/source if given
                if self.flan_task is not None:
                    assert self.flan_source is None 
                    task_name = " ".join(self.flan_task.split("_"))
                    sub_dataset = sub_dataset.filter(lambda x: x["task_name"] == task_name)
                    
                if self.flan_source is not None:
                    assert self.flan_task is None
                    sub_dataset = sub_dataset.filter(lambda x: x["source"] == self.flan_source)
                
                shuf_dataset = sub_dataset.shuffle(seed=self.seed)
                split_datasets.append(shuf_dataset)

        all_datasets = concatenate_datasets(split_datasets)
        return all_datasets

    def split_dataset(self, eval_ratio=0.05):
        self.raw_datasets = self.raw_datasets.train_test_split(
            test_size=eval_ratio, seed=self.seed)

    def get_data(self):
        return self.raw_datasets

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        out_sample = {
            "instruction": sample[self.rewrite_type],
            "input": "",
        }
        if not self.prompt_input:
            instruction_sentence = out_sample['instruction'] + "\n" + out_sample['input'] + "\n"
        else:
            instruction_sentence = self.prompt_input_instruct.format_map(out_sample) if out_sample.get(
                "input", "") != "" else self.prompt_no_input_instruct.format_map(out_sample)

        return instruction_sentence

class Sample_Dialog_SFT_Dataset(PromptRawDataset):
    """
        ### 2023-11-14 written by baorong
        Langauge: English & Chinese
        Source Datasets: ShareGPT & zero-shot (alpaca, gpt4all, camal-ai)
    """
    def __init__(self,
                 recipe_config=None,
                 output_path=None,
                 seed=None,
                 local_rank=None,
                 prompt_input=False,
                 cache_dir=None,
                 adding_demonstrations=True,
                 template_path=None,
                 end_of_conversation="</s>"):
        super().__init__(output_path, seed, local_rank)
        self.re_digits = re.compile(r'(\d+)')
        if not self.seed:
            self.seed = 6666
        self.dataset_name = "multi_turns_dataset"
        self.dataset_name_clean = "multi_turns_dataset"
        self.cache_dir = cache_dir
        self.prompt_input = prompt_input
        self.end_of_conversation = end_of_conversation
        self.adding_demonstrations = adding_demonstrations
        self.template_path = template_path

        with open(recipe_config, "r") as f:
            self.recipe = json.load(f)
        try:
            if self.recipe["root_dir"] != "":
                self.path = self.recipe["root_dir"]
            if self.recipe["cache_dir"] != "":
                self.cache_dir = self.recipe["cache_dir"]
            if self.recipe["output_shffle"] != "":
                self.output_shuffle = eval(self.recipe["output_shffle"])
            else:
                self.output_shuffle = False
        except:
            print("please add root_dir&cache_dir key in your recipe!")
            exit(0)

        print("current cache_dir is :{}".format(self.cache_dir))
        if template_path is None:
            self.prompt_input_chosen, self.prompt_no_input_chosen = PROMPT_DICT[
                "prompt_input_chosen"], PROMPT_DICT["prompt_no_input_chosen"]
            self.prompt_input_instruct, self.prompt_no_input_instruct = PROMPT_DICT[
                "prompt_input_instruct"], PROMPT_DICT["prompt_no_input_instruct"]
            self.prompt_input_chosen_few_shot, self.prompt_no_input_chosen_few_shot = PROMPT_DICT_FEW_SHOT[
                "prompt_input_chosen"], PROMPT_DICT_FEW_SHOT["prompt_no_input_chosen"]
            self.prompt_input_instruct_few_shot, self.prompt_no_input_instruct_few_shot = PROMPT_DICT_FEW_SHOT[
                "prompt_input_instruct"], PROMPT_DICT_FEW_SHOT["prompt_no_input_instruct"] 
        else:
            template_data=json.load(open(template_path, "r"))
            self.prompt_input_chosen, self.prompt_no_input_chosen = template_data[0][
                "prompt_input_chosen"], template_data[0]["prompt_no_input_chosen"]
            self.prompt_input_instruct, self.prompt_no_input_instruct = template_data[0][
                "prompt_input_instruct"], template_data[0]["prompt_no_input_instruct"]
            self.prompt_input_chosen_few_shot, self.prompt_no_input_chosen_few_shot = template_data[1][
                "prompt_input_chosen"], template_data[1]["prompt_no_input_chosen"]
            self.prompt_input_instruct_few_shot, self.prompt_no_input_instruct_few_shot = template_data[1][
                "prompt_input_instruct"], template_data[1]["prompt_no_input_instruct"] 

        self.raw_datasets = self.read_raw_data(recipe=self.recipe)
        print("self.raw_datasets:", self.raw_datasets)
        self.split_dataset(eval_ratio=0.01)
    
    def read_raw_data(self, recipe: dict):
        recipe_str = json.dumps(recipe)
        hash_value = hashlib.md5(recipe_str.encode("utf-8")).hexdigest()
        save_dir = os.path.join(self.cache_dir, hash_value)
        if os.path.exists(save_dir):
            print(f"find cached dataset in {save_dir}")
            return load_from_disk(save_dir)
        
        # record the actual data num in the recipe
        num_recipe = copy.copy(recipe)
        type_datasets = []
        # iterate data type
        for type_data in recipe.keys():
            # add some exception keys used to save meta info！！！
            if type_data in ["root_dir", "cache_dir", "output_shffle","default_max_num"]:
                continue
            type_ratio = recipe[type_data]["ratio"]
            type_shuffle=eval(recipe[type_data]["type_shuffle"])    
            assert type_ratio > 0.0, "type dataset ratio must greater than 0"
            sub_recipe = recipe[type_data]["datasets"]
            sub_datasets = []
            for sub_dataset_name, sub_ratio in sub_recipe.items():
                print(f"processing {type_data}/{sub_dataset_name}")
                # construct a sub dataset
                sub_dataset_path = os.path.join(self.path, type_data, sub_dataset_name)
                if sub_dataset_path.split(".")[-1] == "json" or sub_dataset_path.split(".")[-1] == "jsonl":
                    # load every jsonl file and add to the dataset
                    sub_dataset = load_dataset("json", data_files=sub_dataset_path,)["train"]
                else:
                    warnings.warn(f"load sub dataset fail: {sub_dataset_path}")

                assert sub_ratio > 0.0, "sub dataset ratio must greater than 0"
                downsample_sub_dataset = sub_dataset
                if sub_ratio < 1.0:
                    downsample_sub_dataset = sub_dataset.train_test_split(train_size=sub_ratio, seed=self.seed)["train"]
                    sub_datasets.append(downsample_sub_dataset)
                else:
                    sub_ratio = int(sub_ratio)
                    for _ in range(sub_ratio):
                        sub_datasets.append(sub_dataset)

                print(f"processing {type_data}/{sub_dataset_name} finished!")
                # record the actual used num
                num_recipe[type_data]["datasets"][sub_dataset_name] = len(downsample_sub_dataset) * sub_ratio

            type_dataset = concatenate_datasets(sub_datasets)
            if type_shuffle:
                type_dataset = type_dataset.shuffle(seed=self.seed)
            downsample_type_dataset = type_dataset
            if type_ratio < 1.0:
                downsample_type_dataset = type_dataset.train_test_split(train_size=type_ratio, seed=self.seed)["train"]
                type_datasets.append(downsample_type_dataset)
            else:
                type_ratio = int(type_ratio)
                for _ in range(type_ratio):
                    type_datasets.append(downsample_type_dataset)
            num_recipe[type_data]["ratio"] = len(downsample_type_dataset) * type_ratio

        all_datasets = concatenate_datasets(type_datasets)
        def add_id(example, index):
            example['id'] = index
            return example
        all_datasets = all_datasets.map(add_id, with_indices=True)

        if self.output_shuffle:
            all_datasets = all_datasets.shuffle(seed=self.seed)

        self.create_dir(save_dir)
        all_datasets.save_to_disk(save_dir)
        print(f"final num_recipe:{num_recipe}")
        return all_datasets

    def create_dir(self, path):
        """create a dir and its parent dir recursively"""
        dir_name = os.path.dirname(path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            
    def split_dataset(self, eval_ratio=0.05):
        self.raw_datasets = self.raw_datasets.train_test_split(
            test_size=eval_ratio, seed=self.seed)

    def get_data(self):
        return self.raw_datasets

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]
    
    def get_template_output(self, sample):
        if not self.prompt_input:
            if sample.get("demonstrations", "") != "" and self.adding_demonstrations:
                prefix_sentence = sample['instruction'] + sample['demonstrations'] + "\n"+ sample['input'] + "\n"
            else:
                prefix_sentence = sample['instruction'] + "\n" + sample['input'] + "\n"
        else:
            if sample.get("demonstrations", "") != "" and self.adding_demonstrations:
                prefix_sentence = self.prompt_input_instruct_few_shot.format_map(sample) if sample.get(
                    "input", "") != "" else self.prompt_no_input_instruct_few_shot.format_map(sample)
            else:
                prefix_sentence = self.prompt_input_instruct.format_map(sample) if sample.get(
                    "input", "") != "" else self.prompt_no_input_instruct.format_map(sample)
        return prefix_sentence
    
    def get_sample_prefix(self, sample):
        sample["sys_prompt"] = '' if sample["sys_prompt"] is None else sample["sys_prompt"]
        if sample["conversations"] is not None:
            conversations = sample["conversations"]
            rounds = random.randint(0, len(conversations)-1)
            prefix_sentence = ""
            for idx, conversation in enumerate(conversations):
                if idx == 0:
                    conversation["sys_prompt"] = sample["sys_prompt"]
                else:
                    conversation["sys_prompt"] = ""
                conversation['conversations'] = None
                if idx >= rounds:
                    prefix_sentence = self.get_template_output(conversation)
                else:
                    prefix_sentence += self.get_prompt_and_chosen(conversation)
                if idx >= rounds:
                    break
        else:
            prefix_sentence = self.get_template_output(sample)
        return prefix_sentence

    def get_prompt_and_chosen(self, sample):
        if 'sys_prompt' not in sample.keys() or sample["sys_prompt"] is None:
            sample["sys_prompt"] = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
        if sample["conversations"] is not None:
            conversations = sample["conversations"]
            chosen_sentence = ""
            for idx, conversation in enumerate(conversations):
                if idx == 0:
                    conversation["sys_prompt"] = sample["sys_prompt"]
                else:
                    conversation["sys_prompt"] = ""
                conversation['conversations'] = None
                chosen_sentence += self.get_prompt_and_chosen(conversation)
        else:
            if not self.prompt_input:
                if sample.get("demonstrations", "") != "" and self.adding_demonstrations:
                    chosen_sentence = sample['instruction'] + "\n" + sample['demonstrations'] + "\n"+ \
                    sample['input'] + "\n" + sample['output'] + self.end_of_conversation
                else:
                    chosen_sentence = sample['instruction'] + "\n" + \
                        sample['input'] + "\n" + sample['output'] + self.end_of_conversation                        
            else:
                if sample.get("demonstrations", "") != "" and self.adding_demonstrations:
                    chosen_sentence = self.prompt_input_chosen_few_shot.format_map(sample) if sample.get(
                        "input", "") != "" else self.prompt_no_input_chosen_few_shot.format_map(sample)
                    chosen_sentence += self.end_of_conversation
                else:
                    chosen_sentence = self.prompt_input_chosen.format_map(sample) if sample.get(
                        "input", "") != "" else self.prompt_no_input_chosen.format_map(sample)
                    chosen_sentence += self.end_of_conversation
        return chosen_sentence

    def get_prompt_and_chosen_and_prefix(self, sample):
        chosen_sentence = self.get_prompt_and_chosen(sample)
        instruction_sentence = self.get_sample_prefix(sample)
        return chosen_sentence, instruction_sentence


class Sampled_Alignment_Stage2_Dataset(PromptRawDataset):
    """
        ### 2023-11-08 rewrite by baorong
        Langauge: English
        Source Datasets: full-hh-rlhf, HC3_en, mkqa, oasst1, piqa, stackexchange, 
        rlhf-reward-datasets, SHP, SHP_valtest, summarize_from_feeedback,
        synthetic-instruct-gptj-pairwise, webgpt
        This func supports a Alignment dataset recipe, which are organized as dialog format.
    """

    def __init__(self, 
                 recipe_config="alignment_data_final1024_config.json",
                 output_path=None, 
                 seed=None, 
                 local_rank=None,
                 prompt_input=False,
                 cache_dir="/mnt/data/instruction_data/cache/",
                 template_path=None,
                 end_of_conversation="</s>"):
        super().__init__(output_path, seed, local_rank)
        self.seed = 6666 if self.seed is None else self.seed
        self.dataset_name = "sample_alignment_dataset"
        self.dataset_name_clean = "sample_alignment_dataset"
        self.cache_dir = cache_dir
        self.need_scores = False
        self.prompt_input = prompt_input
        self.end_of_conversation = end_of_conversation
        self.recipe_config=recipe_config
        
        # the path should be a dir
        with open(recipe_config, "r") as f:
            self.recipe = json.load(f)
        try:
            if self.recipe["root_dir"] != "":
                self.path = self.recipe["root_dir"]
            if self.recipe["cache_dir"] != "":
                self.cache_dir = self.recipe["cache_dir"]
            if self.recipe["output_shffle"] != "":
                self.output_shuffle = eval(self.recipe["output_shffle"])
            else:
                self.output_shuffle = False
        except:
            print("please add root_dir&cache_dir key in your recipe!")
            exit(0)

        self.raw_datasets = self.read_raw_data(recipe=self.recipe)
        print("self.raw_datasets:", self.raw_datasets)
        self.split_dataset(eval_ratio=0.05)        

        self.prompt_no_input_chosen = None
        self.prompt_no_input_instruct = None
        self.template_path = template_path
        
        if template_path is None:
            self.prompt_input_chosen, self.prompt_no_input_chosen = PROMPT_DICT[
                "prompt_input_chosen"], PROMPT_DICT["prompt_no_input_chosen"]
            self.prompt_input_instruct, self.prompt_no_input_instruct = PROMPT_DICT[
                "prompt_input_instruct"], PROMPT_DICT["prompt_no_input_instruct"]
            self.prompt_input_chosen_few_shot, self.prompt_no_input_chosen_few_shot = PROMPT_DICT_FEW_SHOT[
                "prompt_input_chosen"], PROMPT_DICT_FEW_SHOT["prompt_no_input_chosen"]
            self.prompt_input_instruct_few_shot, self.prompt_no_input_instruct_few_shot = PROMPT_DICT_FEW_SHOT[
                "prompt_input_instruct"], PROMPT_DICT_FEW_SHOT["prompt_no_input_instruct"] 
        else:
            template_data=json.load(open(template_path, "r"))
            self.prompt_input_chosen, self.prompt_no_input_chosen = template_data[0][
                "prompt_input_chosen"], template_data[0]["prompt_no_input_chosen"]
            self.prompt_input_instruct, self.prompt_no_input_instruct = template_data[0][
                "prompt_input_instruct"], template_data[0]["prompt_no_input_instruct"]
            self.prompt_input_chosen_few_shot, self.prompt_no_input_chosen_few_shot = template_data[1][
                "prompt_input_chosen"], template_data[1]["prompt_no_input_chosen"]
            self.prompt_input_instruct_few_shot, self.prompt_no_input_instruct_few_shot = template_data[1][
                "prompt_input_instruct"], template_data[1]["prompt_no_input_instruct"]
            
    def read_raw_data(self, recipe: dict):
        recipe_str = json.dumps(recipe)
        hash_value = hashlib.md5(recipe_str.encode("utf-8")).hexdigest()
        save_dir = os.path.join(self.cache_dir, hash_value)
        if os.path.exists(save_dir):
            print(f"find cached dataset in {save_dir}")
            return load_from_disk(save_dir)
        
        # record the actual data num in the recipe
        num_recipe = copy.copy(recipe)
        type_datasets = []
        # iterate data type
        for type_data in recipe.keys():
            # add some exception keys used to save meta info！！！
            if type_data in ["root_dir", "cache_dir", "output_shffle","default_max_num", "need_scores"]:
                continue
            type_ratio = recipe[type_data]["ratio"]
            type_shuffle=eval(recipe[type_data]["type_shuffle"])    
            assert type_ratio > 0.0, "type dataset ratio must greater than 0"
            sub_recipe = recipe[type_data]["datasets"]
            sub_datasets = []
            for sub_dataset_name, sub_ratio in sub_recipe.items():
                print(f"processing {type_data}/{sub_dataset_name}")
                # construct a sub dataset
                sub_dataset_path = os.path.join(self.path, type_data, sub_dataset_name)
                if sub_dataset_path.split(".")[-1] == "json" or sub_dataset_path.split(".")[-1] == "jsonl":
                    # load every jsonl file and add to the dataset
                    sub_dataset = load_dataset("json", data_files=sub_dataset_path)["train"]
                    sub_dataset_len = len(sub_dataset)
                    if type(sub_ratio) == dict:
                        file_num=sub_ratio["max_num"]
                        sub_ratio = sub_ratio["ratio"]
                        file_ratio=file_num / sub_dataset_len
                        sub_ratio = min(sub_ratio,file_ratio)
                    downsample_sub_dataset = sub_dataset
                    if sub_ratio < 1.0:
                        downsample_sub_dataset = sub_dataset.train_test_split(train_size=sub_ratio, seed=self.seed)["train"]
                    # downsample_sub_dataset = downsample_sub_dataset.remove_columns(["scores",'score_range'])
                    sub_datasets.append(downsample_sub_dataset)
                else:
                    warnings.warn(f"load sub dataset fail: {sub_dataset_path}")
    
                print(f"processing {type_data}/{sub_dataset_name} finished!")
                # record the actual used num
                num_recipe[type_data]["datasets"][sub_dataset_name] = len(downsample_sub_dataset)
                
            type_dataset = concatenate_datasets(sub_datasets)
            type_len=len(type_dataset)
            if "max_num" in recipe[type_data].keys():
                max_num = recipe[type_data]["max_num"]
                num_ratio=max_num / type_len
                type_ratio = min(num_ratio,type_ratio)
            
            if type_shuffle:
                type_dataset = type_dataset.shuffle(seed=self.seed)
            downsample_type_dataset = type_dataset
            if type_ratio < 1.0:
                downsample_type_dataset = type_dataset.train_test_split(train_size=type_ratio, seed=self.seed)["train"]
                
            def add_category(example):
                if 'helpful' in type_data:
                    category = 'helpful'
                elif 'harmless' in type_data:
                    category = 'harmless'
                else:
                    category = 'Unknown'
                example['category'] = category
                # print(category)
                return example
            downsample_type_dataset = downsample_type_dataset.map(add_category, with_indices=False, num_proc=64)
            type_datasets.append(downsample_type_dataset)
            num_recipe[type_data]["ratio"] = len(downsample_type_dataset)
        
        print(f"final num_recipe:{num_recipe}")
        all_datasets = concatenate_datasets(type_datasets)

        # 添加自动递增的 ID 字段
        def add_length(example,index):
            if 'ranked_responses' in example:
                example['length'] = len(example['ranked_responses'])
            else:
                example['length'] = 0
            return example
        all_datasets = all_datasets.map(add_length, with_indices=True, num_proc=64)
        self.create_dir(save_dir)
        if 'flan' not in self.recipe_config and 'ranks' in all_datasets.column_names:
            all_datasets = all_datasets.filter(lambda x: max(x['ranks']) > min(x['ranks']))
        all_datasets.save_to_disk(save_dir)
        return all_datasets

    def create_dir(self, path):
        """create a dir and its parent dir recursively"""
        dir_name = os.path.dirname(path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

    def split_dataset(self, eval_ratio=0.05):
        self.raw_datasets = self.raw_datasets.train_test_split(
            test_size=eval_ratio, seed=self.seed)

    def get_data(self):
        return self.raw_datasets

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]
    
    def get_prompt_and_chosen(self, sample):
        if 'sys_prompt' not in sample.keys() or sample["sys_prompt"] is None:
            # when self.end_of_conversation == "<|endoftext|>", it means we are training pythia model instead of vicuna model
            if 'qwen' in self.template_path:
                sample["sys_prompt"] = '<|im_start|>system\nYou are a helpful assistant<|im_end|>'
            elif 'vicuna' not in self.template_path or self.end_of_conversation == "<|endoftext|>": 
                sample["sys_prompt"] = ''
            else:
                sample["sys_prompt"] = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
        chosen_sentence = []
        # recursively call this function to get the chosen sentence, if length of chosen sentence is 1, then return the sentence as a string!
        if sample["conversations"] is not None:
            conversations = sample["conversations"]
            for idx, conversation in enumerate(conversations):
                if idx == 0:
                    conversation["sys_prompt"] = sample["sys_prompt"]
                else:
                    conversation["sys_prompt"] = ""
                conversation['conversations'] = None
                chosen_sentence.append(self.get_prompt_and_chosen(conversation))
        else:
            if not self.prompt_input:
                if sample.get("input", "") != "":
                    chosen_sentence = sample['instruction'] + "\n" + sample['input'] + "\n"+ sample['output']
                else:
                    chosen_sentence = sample['instruction'] + "\n" + sample['output']
            else:
                if sample.get("input", "") != "":
                    chosen_sentence = self.prompt_input_chosen.format_map(sample)
                else:
                    chosen_sentence = self.prompt_no_input_chosen.format_map(sample)
        return chosen_sentence
    
    def get_prompt_and_all_response_and_ranks(self, sample):
        chosen_sentence = self.get_prompt_and_chosen(sample)
        concat_sentence = f"{self.end_of_conversation}".join(chosen_sentence)
        responses_and_scores = []
        for rank_response, rank_score in zip(sample['ranked_responses'], sample['ranks']):
            responses_and_scores.append((concat_sentence + rank_response + self.end_of_conversation, rank_score))
        return responses_and_scores
    
    def get_prompt(self, sample):
        chosen_sentence = self.get_prompt_and_chosen(sample)
        concat_sentence = f"{self.end_of_conversation}".join(chosen_sentence)
        return concat_sentence
    
    def get_prompt_and_all_response_and_ranks_and_labels(self, sample):
        chosen_sentence = self.get_prompt_and_chosen(sample)
        prompt = f"{self.end_of_conversation}".join(chosen_sentence)
        assert sample['ranks'] == [0, 1], "big error!!!"
        responses_and_ranks = []
        for rank_response, rank_score in zip(sample['ranked_responses'], sample['ranks']):
            responses_and_ranks.append((rank_response + self.end_of_conversation, rank_score))
        return prompt, responses_and_ranks
    




class Sampled_Alignment_Stage4_Dataset_no_Template(PromptRawDataset):
    """
        ### 2023-11-08 rewrite by baorong
        Langauge: English
        Source Datasets: full-hh-rlhf, HC3_en, mkqa, oasst1, piqa, stackexchange, 
        rlhf-reward-datasets, SHP, SHP_valtest, summarize_from_feeedback,
        synthetic-instruct-gptj-pairwise, webgpt
        This func supports a Alignment dataset recipe, which are organized as dialog format.
    """

    def __init__(self, 
                 recipe_config="alignment_data_final1024_config.json",
                 output_path=None, 
                 seed=None, 
                 local_rank=None,
                 prompt_input=False,
                 cache_dir="/mnt/data/instruction_data/cache/",
                 template_path=None,
                 end_of_conversation="</s>"):
        super().__init__(output_path, seed, local_rank)
        self.seed = 6666 if self.seed is None else self.seed
        self.dataset_name = "sample_alignment_dataset"
        self.dataset_name_clean = "sample_alignment_dataset"
        self.cache_dir = cache_dir
        self.need_scores = False
        self.prompt_input = prompt_input
        self.end_of_conversation = end_of_conversation
        self.recipe_config=recipe_config
        
        # the path should be a dir
        with open(recipe_config, "r") as f:
            self.recipe = json.load(f)
        try:
            if self.recipe["root_dir"] != "":
                self.path = self.recipe["root_dir"]
            if self.recipe["cache_dir"] != "":
                self.cache_dir = self.recipe["cache_dir"]
            if self.recipe["output_shffle"] != "":
                self.output_shuffle = eval(self.recipe["output_shffle"])
            else:
                self.output_shuffle = False
        except:
            print("please add root_dir&cache_dir key in your recipe!")
            exit(0)

        self.raw_datasets = self.read_raw_data(recipe=self.recipe)
        print("self.raw_datasets:", self.raw_datasets)
        self.split_dataset(eval_ratio=0.05)

        self.prompt_no_input_chosen = None
        self.prompt_no_input_instruct = None
        # self.template_path = template_path
        
        # if template_path is None:
        #     self.prompt_input_chosen, self.prompt_no_input_chosen = PROMPT_DICT[
        #         "prompt_input_chosen"], PROMPT_DICT["prompt_no_input_chosen"]
        #     self.prompt_input_instruct, self.prompt_no_input_instruct = PROMPT_DICT[
        #         "prompt_input_instruct"], PROMPT_DICT["prompt_no_input_instruct"]
        #     self.prompt_input_chosen_few_shot, self.prompt_no_input_chosen_few_shot = PROMPT_DICT_FEW_SHOT[
        #         "prompt_input_chosen"], PROMPT_DICT_FEW_SHOT["prompt_no_input_chosen"]
        #     self.prompt_input_instruct_few_shot, self.prompt_no_input_instruct_few_shot = PROMPT_DICT_FEW_SHOT[
        #         "prompt_input_instruct"], PROMPT_DICT_FEW_SHOT["prompt_no_input_instruct"] 
        # else:
        #     template_data=json.load(open(template_path, "r"))
        #     self.prompt_input_chosen, self.prompt_no_input_chosen = template_data[0][
        #         "prompt_input_chosen"], template_data[0]["prompt_no_input_chosen"]
        #     self.prompt_input_instruct, self.prompt_no_input_instruct = template_data[0][
        #         "prompt_input_instruct"], template_data[0]["prompt_no_input_instruct"]
        #     self.prompt_input_chosen_few_shot, self.prompt_no_input_chosen_few_shot = template_data[1][
        #         "prompt_input_chosen"], template_data[1]["prompt_no_input_chosen"]
        #     self.prompt_input_instruct_few_shot, self.prompt_no_input_instruct_few_shot = template_data[1][
        #         "prompt_input_instruct"], template_data[1]["prompt_no_input_instruct"]
            
    def read_raw_data(self, recipe: dict):
        recipe_str = json.dumps(recipe)
        hash_value = hashlib.md5(recipe_str.encode("utf-8")).hexdigest()
        save_dir = os.path.join(self.cache_dir, hash_value)
        if os.path.exists(save_dir):
            print(f"find cached dataset in {save_dir}")
            return load_from_disk(save_dir)
        
        # record the actual data num in the recipe
        num_recipe = copy.copy(recipe)
        type_datasets = []
        # iterate data type
        for type_data in recipe.keys():
            # add some exception keys used to save meta info！！！
            if type_data in ["root_dir", "cache_dir", "output_shffle","default_max_num", "need_scores"]:
                continue
            type_ratio = recipe[type_data]["ratio"]
            type_shuffle=eval(recipe[type_data]["type_shuffle"])    
            assert type_ratio > 0.0, "type dataset ratio must greater than 0"
            sub_recipe = recipe[type_data]["datasets"]
            sub_datasets = []
            for sub_dataset_name, sub_ratio in sub_recipe.items():
                print(f"processing {type_data}/{sub_dataset_name}")
                # construct a sub dataset
                sub_dataset_path = os.path.join(self.path, type_data, sub_dataset_name)
                if sub_dataset_path.split(".")[-1] == "json" or sub_dataset_path.split(".")[-1] == "jsonl":
                    # load every jsonl file and add to the dataset
                    sub_dataset = load_dataset("json", data_files=sub_dataset_path)["train"]
                    sub_dataset_len = len(sub_dataset)
                    if type(sub_ratio) == dict:
                        file_num=sub_ratio["max_num"]
                        sub_ratio = sub_ratio["ratio"]
                        file_ratio=file_num / sub_dataset_len
                        sub_ratio = min(sub_ratio,file_ratio)
                    downsample_sub_dataset = sub_dataset
                    if sub_ratio < 1.0:
                        downsample_sub_dataset = sub_dataset.train_test_split(train_size=sub_ratio, seed=self.seed)["train"]
                    # downsample_sub_dataset = downsample_sub_dataset.remove_columns(["scores",'score_range'])
                    sub_datasets.append(downsample_sub_dataset)
                else:
                    warnings.warn(f"load sub dataset fail: {sub_dataset_path}")
    
                print(f"processing {type_data}/{sub_dataset_name} finished!")
                # record the actual used num
                num_recipe[type_data]["datasets"][sub_dataset_name] = len(downsample_sub_dataset)
                
            type_dataset = concatenate_datasets(sub_datasets)
            type_len=len(type_dataset)
            if "max_num" in recipe[type_data].keys():
                max_num = recipe[type_data]["max_num"]
                num_ratio=max_num / type_len
                type_ratio = min(num_ratio,type_ratio)
            
            if type_shuffle:
                type_dataset = type_dataset.shuffle(seed=self.seed)
            downsample_type_dataset = type_dataset
            if type_ratio < 1.0:
                downsample_type_dataset = type_dataset.train_test_split(train_size=type_ratio, seed=self.seed)["train"]
                
            def add_category(example):
                if 'helpful' in type_data:
                    category = 'helpful'
                elif 'harmless' in type_data:
                    category = 'harmless'
                else:
                    category = 'Unknown'
                example['category'] = category
                # print(category)
                return example
            downsample_type_dataset = downsample_type_dataset.map(add_category, with_indices=False, num_proc=64)
            type_datasets.append(downsample_type_dataset)
            num_recipe[type_data]["ratio"] = len(downsample_type_dataset)
        
        print(f"final num_recipe:{num_recipe}")
        all_datasets = concatenate_datasets(type_datasets)

        # 添加自动递增的 ID 字段
        def add_length(example,index):
            if 'ranked_responses' in example:
                example['length'] = len(example['ranked_responses'])
            else:
                example['length'] = 0
            return example
        all_datasets = all_datasets.map(add_length, with_indices=True, num_proc=64)
        self.create_dir(save_dir)
        if 'flan' not in self.recipe_config and 'ranks' in all_datasets.column_names:
            all_datasets = all_datasets.filter(lambda x: max(x['ranks']) > min(x['ranks']))
        all_datasets.save_to_disk(save_dir)
        return all_datasets

    def create_dir(self, path):
        """create a dir and its parent dir recursively"""
        dir_name = os.path.dirname(path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

    def split_dataset(self, eval_ratio=0.05):
        self.raw_datasets = self.raw_datasets.train_test_split(
            test_size=eval_ratio, seed=self.seed)

    def get_data(self):
        return self.raw_datasets

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]
    
    def get_prompt_and_chosen(self, sample):
        raise
        if 'sys_prompt' not in sample.keys() or sample["sys_prompt"] is None:
            # when self.end_of_conversation == "<|endoftext|>", it means we are training pythia model instead of vicuna model
            if 'qwen' in self.template_path:
                sample["sys_prompt"] = '<|im_start|>system\nYou are a helpful assistant<|im_end|>'
            elif 'vicuna' not in self.template_path or self.end_of_conversation == "<|endoftext|>": 
                sample["sys_prompt"] = ''
            else:
                sample["sys_prompt"] = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
        chosen_sentence = []
        # recursively call this function to get the chosen sentence, if length of chosen sentence is 1, then return the sentence as a string!
        if sample["conversations"] is not None:
            conversations = sample["conversations"]
            for idx, conversation in enumerate(conversations):
                if idx == 0:
                    conversation["sys_prompt"] = sample["sys_prompt"]
                else:
                    conversation["sys_prompt"] = ""
                conversation['conversations'] = None
                chosen_sentence.append(self.get_prompt_and_chosen(conversation))
        else:
            if not self.prompt_input:
                if sample.get("input", "") != "":
                    chosen_sentence = sample['instruction'] + "\n" + sample['input'] + "\n"+ sample['output']
                else:
                    chosen_sentence = sample['instruction'] + "\n" + sample['output']
            else:
                if sample.get("input", "") != "":
                    chosen_sentence = self.prompt_input_chosen.format_map(sample)
                else:
                    chosen_sentence = self.prompt_no_input_chosen.format_map(sample)
        return chosen_sentence
    
    def get_prompt_and_all_response_and_ranks(self, sample):
        raise
        chosen_sentence = self.get_prompt_and_chosen(sample)
        concat_sentence = f"{self.end_of_conversation}".join(chosen_sentence)
        responses_and_scores = []
        for rank_response, rank_score in zip(sample['ranked_responses'], sample['ranks']):
            responses_and_scores.append((concat_sentence + rank_response + self.end_of_conversation, rank_score))
        return responses_and_scores
    
    def get_prompt(self, sample):
        raise
        chosen_sentence = self.get_prompt_and_chosen(sample)
        concat_sentence = f"{self.end_of_conversation}".join(chosen_sentence)
        return concat_sentence
    
    def get_prompt_and_all_response_and_ranks_and_labels(self, sample):
        # chosen_sentence = self.get_prompt_and_chosen(sample)
        # prompt = f"{self.end_of_conversation}".join(chosen_sentence)
        prompt = sample['instruction']
        assert sample['ranks'] == [0, 1], "big error!!!"
        responses_and_ranks = []
        for rank_response, rank_score in zip(sample['ranked_responses'], sample['ranks']):
            responses_and_ranks.append((rank_response + self.end_of_conversation, rank_score))
        return prompt, responses_and_ranks
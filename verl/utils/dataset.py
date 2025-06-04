# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import math
import os
from collections import defaultdict
from io import BytesIO
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import datasets
from datasets import load_dataset, concatenate_datasets
from PIL import Image
from PIL.Image import Image as ImageObject
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..models.transformers.qwen2_vl import get_rope_index
from . import torch_functional as VF


def collate_fn(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    for feature in features:
        for key, value in feature.items():
            if isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)

    for key, value in tensors.items():
        tensors[key] = torch.stack(value, dim=0)

    for key, value in non_tensors.items():
        non_tensors[key] = np.array(value, dtype=object)

    return {**tensors, **non_tensors}


class ImageProcessMixin:
    max_pixels: int
    min_pixels: int

    def process_image(self, image: Union[Dict[str, Any], ImageObject]) -> ImageObject:
        if isinstance(image, dict):
            image = Image.open(BytesIO(image["bytes"]))
        elif isinstance(image, bytes):
            image = Image.open(BytesIO(image))

        if (image.width * image.height) > self.max_pixels:
            resize_factor = math.sqrt(self.max_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))

        if (image.width * image.height) < self.min_pixels:
            resize_factor = math.sqrt(self.min_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))

        if image.mode != "RGB":
            image = image.convert("RGB")

        return image


class RLHFDataset(Dataset, ImageProcessMixin):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_path: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        prompt_key: str = "prompt",
        answer_key: str = "answer",
        image_key: str = "images",
        max_prompt_length: int = 1024,
        truncation: str = "error",
        format_prompt: str = None,
        max_pixels: int = None,
        min_pixels: int = None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.image_key = image_key
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.format_prompt = format_prompt
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        
        print(f"Loading dataset from {data_path}")
        
        if isinstance(data_path, list):
            # 新版：处理data_path为list的情况
            self.data_files = data_path
            dataframes = []
            for parquet_file in self.data_files:
                dataframe = load_dataset("parquet", data_files=parquet_file)["train"]
                dataframes.append(dataframe)
            self.dataset = concatenate_datasets(dataframes)
            print(f'dataset len: {len(self.dataset)}')

            # # 临时处理no thinking的情况
            # self.data_files = data_path
            # dataframes = []
            # for parquet_file in self.data_files:
            #     dataframe = load_dataset("parquet", data_files=parquet_file)["train"]
            #     dataframes.append(dataframe)
            # self.dataset = concatenate_datasets(dataframes)
            # print(f'dataset len: {len(self.dataset)}')
            
            # # 替换instruction_following为no thinking版本
            # def replace_instruction(example):
            #     if "problem" in example:
            #         # 直接以"You FIRST think about"作为分界点
            #         instruction_index = example["problem"].find("You FIRST think about")
                    
            #         if instruction_index != -1:
            #             # 保留分界点之前的内容
            #             question_part = example["problem"][:instruction_index].rstrip()
            #             # 添加新指令
            #             new_instruction = "Directly output the answer in the format <point>[[x1, y1], [x2, y2], ...]</point>."
            #             # 重建problem
            #             example["problem"] = question_part + "\n" + new_instruction
            #         else:
            #             # 如果没找到旧指令，直接在末尾添加新指令
            #             new_instruction = "Directly output the answer in the format <point>[[x1, y1], [x2, y2], ...]</point>."
            #             example["problem"] = example["problem"].rstrip() + "\n" + new_instruction
                        
            #     return example
                
            # self.dataset = self.dataset.map(replace_instruction, desc="替换指令为no thinking版本")

            # # # 临时修改3d dataset
            # self.data_files = data_path
            # dataframes = []
            # for parquet_file in self.data_files:
            #     dataframe = load_dataset("parquet", data_files=parquet_file)["train"]
            #     dataframes.append(dataframe)
            # self.dataset = concatenate_datasets(dataframes)
            # print(f'dataset len: {len(self.dataset)}')
            
            # # 替换instruction_following为3d dataset版本
            # def replace_instruction(example):
            #     if "problem" in example:
            #         # 计算<image>标签的数量
            #         image_count = example["problem"].count("<image>")
            #         print(f"一共有{image_count}个<image>标签")
                    
            #         # 在第一个<image>前添加RGB Image:
            #         if "<image>" in example["problem"]:
            #             first_image_index = example["problem"].find("<image>")
            #             example["problem"] = example["problem"][:first_image_index] + "RGB Image: " + example["problem"][first_image_index:]
                    
            #         # 如果有第二个<image>，在前面添加Depth Image:
            #         if image_count >= 2:
            #             # 找到第一个<image>后的位置
            #             first_image_pos = example["problem"].find("<image>")
            #             # 从第一个<image>后开始查找第二个<image>
            #             second_image_index = example["problem"].find("<image>", first_image_pos + 7)  # 7是<image>的长度
            #             if second_image_index != -1:
            #                 example["problem"] = example["problem"][:second_image_index] + "Depth Image: " + example["problem"][second_image_index:]
                
            #     return example
                
            # self.dataset = self.dataset.map(replace_instruction, desc="替换指令为3d dataset版本")

            # filter out too long prompts
            self.filter_overlong_prompts = True  # todo: 需要根据实际情况设置
            if self.filter_overlong_prompts:
                tokenizer = self.tokenizer
                prompt_key = self.prompt_key
                self.dataset = self.dataset.filter(
                    lambda doc: len(tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True)
                                ) <= self.max_prompt_length,
                    num_proc=8,
                    desc=f"Filtering prompts longer than {self.max_prompt_length} tokens")

                print(f'filter dataset len: {len(self.dataset)}')
        else:
            # 原版：处理data_path为str的情况
            if "@" in data_path:
                data_path, data_split = data_path.split("@")
            else:
                data_split = "train"
                
            if os.path.isdir(data_path):
                self.dataset = load_dataset("parquet", data_dir=data_path, split="train")
            elif os.path.isfile(data_path):
                self.dataset = load_dataset("parquet", data_files=data_path, split="train")
            else:  # remote dataset
                self.dataset = load_dataset(data_path, split=data_split)
            print(f'dataset len: {len(self.dataset)}')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        row_dict: dict = self.dataset[index]
        prompt_str: str = row_dict[self.prompt_key]
        if self.format_prompt:
            prompt_str = prompt_str + " " + self.format_prompt.strip()

        if self.image_key in row_dict:
            # https://huggingface.co/docs/transformers/en/tasks/image_text_to_text
            content_list = []
            for i, content in enumerate(prompt_str.split("<image>")):
                if i != 0:
                    content_list.append({"type": "image"})

                if content:
                    content_list.append({"type": "text", "text": content})

            messages = [{"role": "user", "content": content_list}]
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            images = [self.process_image(image) for image in row_dict.pop(self.image_key)]
            model_inputs = self.processor(images, [prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            row_dict["multi_modal_data"] = {"image": images}
            row_dict["multi_modal_inputs"] = dict(model_inputs)

            # qwen2vl mrope
            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=model_inputs["image_grid_thw"],
                attention_mask=attention_mask,
            )  # (3, seq_length)
        else:
            messages = [{"role": "user", "content": prompt_str}]
            prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.tokenizer([prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)  # (seq_length,)

        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        row_dict["input_ids"] = input_ids
        row_dict["attention_mask"] = attention_mask
        row_dict["position_ids"] = position_ids
        row_dict["raw_prompt_ids"] = self.tokenizer.encode(prompt, add_special_tokens=False)
        row_dict["ground_truth"] = row_dict.pop(self.answer_key)
        row_dict["data_type"] = row_dict.pop("data_type")
        return row_dict

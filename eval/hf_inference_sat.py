# python3 -m accelerate.commands.launch --num_processes=4 --main_process_port=54321 eval/hf_inference_embspatial.py

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoTokenizer
import torch
from datasets import load_dataset, load_from_disk
import pandas as pd
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.utils.dataclasses import DistributedType
import json
import os
from tqdm import tqdm
import time
import logging
import random
import textwrap

def process_vision_info(messages, return_video_kwargs=False):
    image_inputs = []
    for msg in messages:
        for content in msg["content"] if isinstance(msg["content"], list) else [msg["content"]]:
            if isinstance(content, dict) and content.get("type") == "image":
                image_inputs.append(content["image"])

    video_inputs = None
    video_kwargs = {}

    if return_video_kwargs:
        return image_inputs, video_inputs, video_kwargs
    else:
        return image_inputs, video_inputs

def process_sample(sample, model, processor, device, reasoning_model, instruct_following, gen_args):
    question = sample['question'].strip()
    answer_choices = sample['answers']
    random.shuffle(answer_choices)
    answer = ", ".join(answer_choices[:-1]) + " or " + answer_choices[-1]
    prompt = f"{question} Choose between the following options: {answer}"
    problem = f"Answer in natural language. {prompt}."
    problem = problem + '\n' + instruct_following
    problem = textwrap.dedent(problem).strip()
    correct_answer = sample['correct_answer']
    correct_answer = str(correct_answer)

    question = problem
    answer = correct_answer

    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    } for image in sample["image_bytes"]
                ] + [{"type": "text", "text": question}],
            },
        ]
        
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _ = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(device)
        
        with torch.no_grad():
            generated_ids = model.generate(**inputs, **gen_args)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        predicted = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        original_predicted = predicted
        
        if reasoning_model:
            import re
            pattern = r"<answer>(.*?)<\/answer>"
            match = re.search(pattern, predicted)
            if match:
                predicted = match.group(1).strip()
            else:
                logger.info(f"oracle prediction: {predicted}")
                predicted = ""
        
        logger.info("------------------------------------------------------")
        logger.info(f"question: {question}")
        logger.info(f"original_predicted: {original_predicted}")
        logger.info(f"predicted: {predicted}")
        logger.info(f"ground_truth: {answer}")
        logger.info("------------------------------------------------------")

        return {
            'question': question,
            'predicted': predicted,
            'ground_truth': answer,
            'original_predicted': original_predicted,
            'accuracy': answer in predicted,
        }
    
    except Exception as e:
        logger.error(f"Error processing sample: {e}")
        return None

def main(task_name, model_name, model_path, reasoning_model, max_pixels, min_pixels, gen_args, instruct_following, huggingface_dataset_name, split):
    accelerator = Accelerator()
    device = accelerator.device
    logger.info(f"device: {device}")
    set_seed(42)

    if accelerator.num_processes > 1:
        local_device = torch.device(f"cuda:{accelerator.local_process_index}")
        device_map = f"cuda:{accelerator.local_process_index}"
    else:
        local_device = device
        device_map = "auto"
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        attn_implementation="flash_attention_2",
    )
    
    processor = AutoProcessor.from_pretrained(model_path, max_pixels=max_pixels, min_pixels=min_pixels)
    
    start_time = time.time()
    if accelerator.num_processes > 1:
        if accelerator.distributed_type == DistributedType.FSDP:
            model = accelerator.prepare(model)
        else:
            model = accelerator.prepare_model(model, evaluation_mode=True)
        
        if accelerator.is_main_process:
            logger.info(f"Using {accelerator.num_processes} devices for data parallel processing")
    
    with accelerator.main_process_first():
        dataset = load_dataset(huggingface_dataset_name)
        test_data = dataset[split]
    
    process_idx = accelerator.process_index
    num_processes = accelerator.num_processes
    
    samples_per_process = len(test_data) // num_processes
    start_idx = process_idx * samples_per_process
    end_idx = start_idx + samples_per_process if process_idx < num_processes - 1 else len(test_data)
    
    local_results = []
    
    if accelerator.is_main_process:
        iterator = tqdm(range(start_idx, end_idx), desc=f"Process {process_idx}")
    else:
        iterator = range(start_idx, end_idx)
    
    for idx in iterator:
        result = process_sample(test_data[idx], model, processor, local_device, reasoning_model, instruct_following, gen_args)
        if result is not None:
            local_results.append(result)
    
    all_results = accelerator.gather_for_metrics(local_results)
    
    if accelerator.is_main_process:
        if isinstance(all_results[0], list):
            final_results = [item for sublist in all_results for item in sublist]
        else:
            final_results = all_results
        
        total_samples = len(final_results)
        correct_predictions = sum(1 for result in final_results if result.get("accuracy") == True)
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0
        
        os.makedirs('logs/results', exist_ok=True)
        result_file_name = f'logs/results/{task_name}_{model_name}_{reasoning_model}_{split}.json'
        with open(result_file_name, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, ensure_ascii=False, indent=4)

        end_time = time.time()
        total_time = end_time - start_time

        logger.info("------------------------------------------------------")
        logger.info(f"Model Name: {model_name}")
        logger.info("------------------------------------------------------")
        logger.info(f"Finished processing. Total samples: {total_samples}")
        logger.info(f"Number of correct predictions: {correct_predictions}")
        logger.info(f"Average accuracy: {accuracy:.4f}")
        logger.info(f"Total time taken: {total_time:.2f} seconds")
        logger.info("------------------------------------------------------")

if __name__ == "__main__":
    task_name = "SAT-Real"
    huggingface_dataset_name = "array/SAT"
    split = "test"

    model_name = "Embodied-R1-3B"
    model_path = "/mnt/path/iffyuan/EasyR1/workdir/embodiedr1_qwen2_5_vl_3b_version_reward_v5_data_exp_date_0428_stage_1_qa/global_step_394/actor/huggingface"
    reasoning_model = True

    max_pixels = 1605632
    min_pixels = 256 * 28 * 28
    gen_args = {
        "temperature": 0,
        "top_p": 1,
        "max_new_tokens": 2048,
        "repetition_penalty": 1.05,
        "do_sample": False,
    }

    if reasoning_model:
        instruct_following = (
            r'You FIRST think about the reasoning process as an internal monologue and then provide the final answer. '
            r'The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags. '
            r'The overall format being: '
            r'<think> reasoning process here </think><answer>your answer here</answer>'
        )
    else:
        instruct_following = "Directly output the answer."
    current_time = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs('logs', exist_ok=True)
    log_file_name = f"logs/inference_{task_name}_{model_name}_{current_time}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - Process %(process)d - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file_name)
        ]
    )
    logger = logging.getLogger(f"{task_name}_{model_name}")
    main(task_name, model_name, model_path, reasoning_model, max_pixels, min_pixels, gen_args, instruct_following, huggingface_dataset_name, split)
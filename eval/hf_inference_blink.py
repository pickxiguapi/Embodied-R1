# python3 -m accelerate.commands.launch --num_processes=8 --main_process_port=54321 eval/hf_inference_blink.py

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoTokenizer
import torch
from datasets import load_dataset
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.utils.dataclasses import DistributedType
import json
import os
from tqdm import tqdm
import time
import logging
import textwrap
from datasets import concatenate_datasets
import re


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

    prompt = sample['prompt'].strip()
    problem = prompt + '\n' + instruct_following
    problem = textwrap.dedent(problem).strip()
    answer = str(sample['answer'])
    question = problem
    sub_task = sample['sub_task']
    idx = sample['idx']

    text_option = sample["choices"]
    mapping_dict = {
        "(A)": 0,
        "(B)": 1,
        "(C)": 2,
        "(D)": 3,
        "(E)": 4,
        "(F)": 5,
        "(G)": 6,
        "(H)": 7,
        "(I)": 8,
        "(J)": 9,
    }
    text_answer = text_option[mapping_dict[answer]]

    try:
        image_contents = []
        for i in range(1, 5):
            image_key = f"image_{i}"
            if image_key in sample and sample[image_key]:
                image_contents.append({"type": "image", "image": sample[image_key]})

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": image_contents + [{"type": "text", "text": question}],
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
            pattern = r"<answer>(.*?)<\/answer>"
            match = re.search(pattern, predicted)
            if match:
                predicted = match.group(1).strip()
            else:
                predicted = ""
        
        logger.info("------------------------------------------------------")
        logger.info(f"question: {question}")
        logger.info(f"original_predicted: {original_predicted}")
        logger.info(f"predicted: {predicted}")
        logger.info(f"ground_truth: {answer}")
        logger.info(f"text_answer: {text_answer}")
        logger.info(f"accuracy: {answer[1] in predicted or text_answer in predicted}")
        logger.info("------------------------------------------------------")

        return {
            'question': question,
            'predicted': predicted,
            'ground_truth': answer,
            'original_predicted': original_predicted,
            # todo: (A) and A are the same.
            'accuracy': answer[1] in predicted or text_answer in predicted,  # todo: some models may not have the instruction following capability.
            'sub_task': sub_task,
            'idx': idx,
        }
    
    except Exception as e:
        logger.error(f"Error processing sample: {e}")
        return None

def main(task_name, model_name, model_path, reasoning_model, max_pixels, min_pixels, gen_args, instruct_following, huggingface_dataset_name, split, subsets):
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
        attn_implementation=None,
    )
    
    processor = AutoProcessor.from_pretrained(model_path, max_pixels=max_pixels, min_pixels=min_pixels)
    
    start_time = time.time()
    if accelerator.num_processes > 1:
        if accelerator.distributed_type == DistributedType.FSDP:
            model = accelerator.prepare(model)
        else:
            model = accelerator.prepare_model(model, evaluation_mode=True)
    
    with accelerator.main_process_first():
        dataset = load_dataset(huggingface_dataset_name, subsets[0], split=split)
        for subset in subsets[1:]:
            dataset = concatenate_datasets([dataset, load_dataset(huggingface_dataset_name, subset, split=split)])
        test_data = dataset

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
        result_file_name = f'logs/results/{task_name}_{model_name}_{reasoning_model}.json'
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

        sub_task_results = {}
        for result in final_results:
            sub_task = result['sub_task']
            if sub_task not in sub_task_results:
                sub_task_results[sub_task] = {
                    "total": 0,
                    "correct": 0
                }
            sub_task_results[sub_task]["total"] += 1
            if result.get("accuracy") == True:
                sub_task_results[sub_task]["correct"] += 1

        logger.info("------------------------------------------------------")
        logger.info("Accuracy by Sub-task:")
        for sub_task, counts in sub_task_results.items():
            sub_task_accuracy = counts["correct"] / counts["total"] if counts["total"] > 0 else 0
            logger.info(f"Sub-task: {sub_task}, Accuracy: {sub_task_accuracy:.4f} ({counts['correct']} / {counts['total']})")
        logger.info("------------------------------------------------------")

if __name__ == "__main__":
    task_name = "BLINK"
    huggingface_dataset_name = "BLINK-Benchmark/BLINK"
    split = "val"
    subset = ["Multi-view_Reasoning", "Object_Localization", "Relative_Depth", "Spatial_Relation"]

    model_name = "Embodied-R1-3B"
    model_path = "IffYuan/Embodied-R1-3B-v1"
    reasoning_model = True

    max_pixels = 2097152
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
            r'<think> reasoning process here </think><answer>answer example:(A)</answer>'
        )
    else:
        instruct_following = "Directly output the option."
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
    main(task_name, model_name, model_path, reasoning_model, max_pixels, min_pixels, gen_args, instruct_following, huggingface_dataset_name, split, subset)
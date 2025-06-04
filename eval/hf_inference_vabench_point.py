from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import torch
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
import pandas as pd 
import PIL
from PIL import Image, ImageDraw
from io import BytesIO
import re
import numpy as np

def process_vision_info(messages):
    image_inputs = []
    for msg in messages:
        for content in msg["content"] if isinstance(msg["content"], list) else [msg["content"]]:
            if isinstance(content, dict) and content.get("type") == "image":
                image_inputs.append(content["image"])
    return image_inputs, None

def parse_points_from_output(output):
    points = []
    
    pattern = r"<point>\[(.*?)\]</point>"
    match = re.search(pattern, output)
    if match:
        try:
            points_str = match.group(1)
            if points_str.strip():
                points_str = points_str.replace("'", "\"")
                points_str = f"[{points_str}]"
                try:
                    points = json.loads(points_str)
                except json.JSONDecodeError:
                    coord_pattern = r"\[(\d+\.?\d*),\s*(\d+\.?\d*)\]"
                    coords = re.findall(coord_pattern, points_str)
                    points = [[float(x), float(y)] for x, y in coords]
        except Exception as e:
            logger.error(f"Error parsing points: {e}")
    return points

def check_points_in_bbox(points, bbox):
    if not points:
        return 0, 0
    
    points_in_bbox = 0
    for point in points:
        if len(point) == 2:
            x, y = point
            if bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]:
                points_in_bbox += 1
    
    return points_in_bbox, len(points)

def visualize_bbox_and_points(image, bbox, points, save_path):
    draw = ImageDraw.Draw(image)
    
    # 绘制预测的点
    for point in points:
        if len(point) == 2:
            x, y = point
            in_bbox = bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]
            
            outer_radius = 7
            draw.ellipse([(x-outer_radius, y-outer_radius), 
                          (x+outer_radius, y+outer_radius)], fill="white")
            
            inner_radius = 5
            inner_color = "crimson" if in_bbox else "lightblue"  
            draw.ellipse([(x-inner_radius, y-inner_radius), 
                          (x+inner_radius, y+inner_radius)], fill=inner_color)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    image.save(save_path)
    return save_path

def process_sample(sample, model, processor, device, reasoning_model, instruct_following, gen_args, vis_dir, disable_visualization=True):
    question = sample['problem'].strip()
    question = question + '\n' + instruct_following
    question = textwrap.dedent(question).strip()
    
    answer = sample['answer']
    answer = answer.replace("<type>fsd_free_point</type>", "")
    json_answer = json.loads(answer)
    bbox_answer = json_answer['bbox']
    # point_answer = json_answer['free_points']

    try:
        image_bytes = BytesIO(sample["images"][0])
        image = PIL.Image.open(image_bytes).convert("RGB")
        doc_id = sample['id']
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": question}
                ],
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
                logger.info(f"oracle prediction: {predicted}")
                predicted = ""
        
        points = parse_points_from_output(predicted)
        
        points_in_bbox, total_points = check_points_in_bbox(points, bbox_answer)
        
        accuracy_score = points_in_bbox / total_points if total_points > 0 else 0
        
        vis_path = None
        if not disable_visualization and total_points > 0 and random.random() < 1:
            vis_path = os.path.join(vis_dir, f"sample_{doc_id}_{accuracy_score:.2f}.jpg")
            visualize_bbox_and_points(image, bbox_answer, points, vis_path)
        
        logger.info("------------------------------------------------------")
        logger.info(f"ID: {doc_id}")
        logger.info(f"question: {question}")
        logger.info(f"predicted: {original_predicted}")
        logger.info(f"parsed points: {points}")
        logger.info(f"points in bbox: {points_in_bbox}/{total_points}")
        logger.info(f"accuracy score: {accuracy_score:.4f}")
        if vis_path:
            logger.info(f"visualization saved to: {vis_path}")
        logger.info("------------------------------------------------------")

        return {
            'question_id': doc_id,
            'question': question,
            'predicted': predicted,
            'original_predicted': original_predicted,
            'points': points,
            'points_in_bbox': points_in_bbox,
            'total_points': total_points,
            'accuracy_score': accuracy_score,
        }
    
    except Exception as e:
        logger.error(f"Error processing sample: {e}")
        return None

def main(task_name, model_name, model_path, reasoning_model, max_pixels, min_pixels, gen_args, instruct_following, dataset_path, split, vis_dir, disable_visualization=True, use_flash_attention=False):
    accelerator = Accelerator()
    device = accelerator.device
    logger.info(f"device: {device}")
    set_seed(42)

    if accelerator.num_processes > 1:
        local_device = torch.device(f"cuda:{accelerator.local_process_index}")
        device_map = f"cuda:{accelerator.local_process_index}"
        logger.info(f"Process {accelerator.process_index} using GPU: {accelerator.local_process_index}")
    else:
        local_device = device
        device_map = "auto"
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        attn_implementation="flash_attention_2" if use_flash_attention else None,
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
        logger.info(f"Loading dataset: {dataset_path}")
        dataset = pd.read_parquet(dataset_path)
        test_data = dataset.to_dict(orient='records')
        logger.info(f"Test data samples: {len(test_data)}")
    
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
        result = process_sample(
            test_data[idx], model, processor, local_device, 
            reasoning_model, instruct_following, gen_args, vis_dir, disable_visualization
        )
        if result is not None:
            local_results.append(result)
    
    all_results = accelerator.gather_for_metrics(local_results)
    
    if accelerator.is_main_process:
        if isinstance(all_results[0], list):
            final_results = [item for sublist in all_results for item in sublist]
        else:
            final_results = all_results
        
        total_points = sum(result.get('total_points', 0) for result in final_results)
        total_points_in_bbox = sum(result.get('points_in_bbox', 0) for result in final_results)
        overall_points_accuracy = total_points_in_bbox / total_points if total_points > 0 else 0
        
        os.makedirs('logs/results', exist_ok=True)
        result_file_name = f'logs/results/{task_name}_{model_name}_{reasoning_model}_{split}.json'
        with open(result_file_name, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, ensure_ascii=False, indent=4)

        end_time = time.time()
        total_time = end_time - start_time

        logger.info("------------------------------------------------------")
        logger.info(f"Model Name: {model_name}")
        logger.info("------------------------------------------------------")
        logger.info(f"Finished processing. Total samples: {len(final_results)}")
        logger.info(f"Total points: {total_points}")
        logger.info(f"Points in bbox: {total_points_in_bbox}")
        logger.info(f"Overall points accuracy: {overall_points_accuracy:.4f}")
        logger.info(f"Total time taken: {total_time:.2f} seconds")
        logger.info("------------------------------------------------------")

if __name__ == "__main__":
    task_name = "VABench_Point"
    split = "test"
    disable_visualization = False
    use_flash_attention = False

    model_name = "Embodied-R1-3B"
    model_path = "/mnt/kaiwu-group-x4/iffyuan/EasyR1/workdir/stage_2_embodiedr1_qwen2_5_vl_3b_version_reward_v7_date_0515/global_step_2064/actor/huggingface"
    dataset_path = "/mnt/kaiwu-group-x4/iffyuan/all-seeing/all-seeing-v2/process_rl_data/FSD_points_rft_fsd_free_point_train_32790_test_300_0425/test.parquet"
    reasoning_model = True
    
    vis_dir = f"logs/visualizations/{task_name}_{model_name}"
    os.makedirs(vis_dir, exist_ok=True)

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
            r'The answer consists only of several coordinate points, with the overall format being: '
            r'<think> reasoning process here </think><answer><point>[[x1, y1], [x2, y2], ...]</point></answer>'
        )
    else:
        instruct_following = "Use 2D points to mark the region mentioned in the task with format <point>[[x1, y1], [x2, y2], ...]</point>."
    
    current_time = time.strftime("%Y%m%d_%H%M%S")
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
    
    main(task_name, model_name, model_path, reasoning_model, max_pixels, min_pixels, gen_args, instruct_following, dataset_path, split, vis_dir, disable_visualization, use_flash_attention)
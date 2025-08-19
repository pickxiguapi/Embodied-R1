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
from scipy.interpolate import interp1d

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

def interpolate_trajectory(trajectory, new_length):
    if len(trajectory) <= 1 or new_length <= 1:
        return trajectory
    
    old_indices = np.arange(len(trajectory))
    new_indices = np.linspace(0, len(trajectory) - 1, new_length)
    
    x_coords = [p[0] for p in trajectory]
    y_coords = [p[1] for p in trajectory]
    
    x_interpolator = interp1d(old_indices, x_coords, kind='linear')
    y_interpolator = interp1d(old_indices, y_coords, kind='linear')
    
    new_x_coords = x_interpolator(new_indices)
    new_y_coords = y_interpolator(new_indices)
    
    return [[x, y] for x, y in zip(new_x_coords, new_y_coords)]

def calculate_rmse_mae(pred_trajectory, ans_trajectory):
    if len(pred_trajectory) != len(ans_trajectory):
        logger.warning(f"Trajectory length mismatch: pred={len(pred_trajectory)}, ans={len(ans_trajectory)}")
        return None, None
    
    squared_diffs = []
    abs_diffs = []
    
    for pred_point, ans_point in zip(pred_trajectory, ans_trajectory):
        dx = pred_point[0] - ans_point[0]
        dy = pred_point[1] - ans_point[1]
        
        squared_diff = dx**2 + dy**2
        squared_diffs.append(squared_diff)
        
        abs_diff = (abs(dx) + abs(dy)) / 2
        abs_diffs.append(abs_diff)
    
    rmse = np.sqrt(np.mean(squared_diffs))
    mae = np.mean(abs_diffs)
    
    return rmse, mae

def interpolate_color(start_color, end_color, ratio):
    r = int(start_color[0] * (1 - ratio) + end_color[0] * ratio)
    g = int(start_color[1] * (1 - ratio) + end_color[1] * ratio)
    b = int(start_color[2] * (1 - ratio) + end_color[2] * ratio)
    return (r, g, b)

def draw_diamond(draw, center_x, center_y, size, fill_color, outline_color=(255, 255, 255)):
    points = [
        (center_x, center_y - size),
        (center_x + size, center_y),
        (center_x, center_y + size),
        (center_x - size, center_y)
    ]
    draw.polygon(points, fill=fill_color, outline=outline_color, width=2)

def draw_triangle(draw, center_x, center_y, size, fill_color, outline_color=(255, 255, 255)):
    points = [
        (center_x, center_y - size),
        (center_x - size, center_y + size),
        (center_x + size, center_y + size)
    ]
    draw.polygon(points, fill=fill_color, outline=outline_color, width=2)

def visualize_trajectory_and_points(image, pred_trajectory, ans_trajectory, save_path):
    draw = ImageDraw.Draw(image)
    
    # 注释掉绘制ans_trajectory（绿色线）的代码
    # if len(ans_trajectory) > 1:
    #     for i in range(len(ans_trajectory) - 1):
    #         start_point = tuple(ans_trajectory[i])
    #         end_point = tuple(ans_trajectory[i + 1])
    #         draw.line([start_point, end_point], fill=(0, 255, 0), width=2)
    #     
    #     start_x, start_y = ans_trajectory[0]
    #     draw.ellipse([(start_x-6, start_y-6), (start_x+6, start_y+6)], fill=(0, 255, 0), outline=(255, 255, 255), width=1)
    #     
    #     end_x, end_y = ans_trajectory[-1]
    #     draw.ellipse([(end_x-6, end_y-6), (end_x+6, end_y+6)], fill=(0, 255, 0), outline=(255, 255, 255), width=1)
    
    if len(pred_trajectory) > 1:
        start_color = (255, 0, 0)
        end_color = (0, 0, 255)
        
        for i in range(len(pred_trajectory) - 1):
            ratio = i / (len(pred_trajectory) - 1)
            line_color = interpolate_color(start_color, end_color, ratio)
            
            start_point = tuple(pred_trajectory[i])
            end_point = tuple(pred_trajectory[i + 1])
            draw.line([start_point, end_point], fill=line_color, width=3)
        
        start_x, start_y = pred_trajectory[0]
        draw_diamond(draw, int(start_x), int(start_y), 8, (255, 0, 0))
        
        end_x, end_y = pred_trajectory[-1]
        draw_triangle(draw, int(end_x), int(end_y), 8, (0, 0, 255))
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    image.save(save_path)
    return save_path

def process_sample(sample, model, processor, device, reasoning_model, instruct_following, gen_args, vis_dir, disable_visualization=True):
    question = sample['problem'].strip()
    question = question + '\n' + instruct_following
    question = textwrap.dedent(question).strip()
    
    answer = sample['answer']
    answer = answer.replace("<type>fsd_visual_trace</type>", "")
    json_answer = json.loads(answer)
    ans_traj = json_answer["trajectory"]

    try:
        image_bytes = BytesIO(sample["images"][0])
        image = PIL.Image.open(image_bytes).convert("RGB")
        width, height = image.size
        doc_id = sample.get('id', str(hash(question))[:8])
        
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
        
        pred_traj = parse_points_from_output(predicted)
        
        max_side = max(height, width)
        
        ans_traj_scaled = []
        for point in ans_traj:
            x, y = point
            x = (x / width) * (1000 - 1)
            y = (y / height) * (1000 - 1)
            ans_traj_scaled.append([x, y])
        ans_traj = ans_traj_scaled
        
        pred_traj_scaled = []
        for point in pred_traj:
            x, y = point
            x = (x / width) * (1000 - 1)
            y = (y / height) * (1000 - 1)
            pred_traj_scaled.append([x, y])
        pred_traj = pred_traj_scaled
        
        if len(pred_traj) > 1 and len(ans_traj) > 1:
            new_length = max(len(pred_traj), len(ans_traj))
            pred_traj_interp = interpolate_trajectory(pred_traj, new_length)
            ans_traj_interp = interpolate_trajectory(ans_traj, new_length)
            
            rmse, mae = calculate_rmse_mae(pred_traj_interp, ans_traj_interp)
        else:
            rmse, mae = None, None
            pred_traj_interp = pred_traj
            ans_traj_interp = ans_traj
            
        pred_traj_visual = []
        for point in pred_traj:
            x, y = point
            x = (x / (1000 - 1)) * width
            y = (y / (1000 - 1)) * height
            pred_traj_visual.append([x, y])
            
        ans_traj_visual = []
        for point in ans_traj:
            x, y = point
            x = (x / (1000 - 1)) * width
            y = (y / (1000 - 1)) * height
            ans_traj_visual.append([x, y])
        
        vis_path = None
        if not disable_visualization and len(pred_traj) > 0 and random.random() < 1:
            vis_path = os.path.join(vis_dir, f"sample_{doc_id}_{rmse:.2f}_{mae:.2f}.jpg")
            visualize_trajectory_and_points(image, pred_traj_visual, ans_traj_visual, vis_path)
        
        logger.info("------------------------------------------------------")
        logger.info(f"id: {doc_id}")
        logger.info(f"question: {question}")
        logger.info(f"predicted: {original_predicted}")
        logger.info(f"parsed trajectory points: {len(pred_traj)}")
        logger.info(f"ground truth trajectory points: {len(ans_traj)}")
        if rmse is not None and mae is not None:
            logger.info(f"RMSE: {rmse:.4f}")
            logger.info(f"MAE: {mae:.4f}")
        if vis_path:
            logger.info(f"visualization saved to: {vis_path}")
        logger.info("------------------------------------------------------")

        return {
            'question_id': doc_id,
            'question': question,
            'predicted': predicted,
            'original_predicted': original_predicted,
            'pred_traj': pred_traj,
            'ans_traj': ans_traj,
            'rmse': rmse,
            'mae': mae,
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
        
        rmses = []
        maes = []
        valid_samples = 0
        
        for result in final_results:
            if 'rmse' in result and 'mae' in result and result['rmse'] is not None and result['mae'] is not None:
                rmses.append(result['rmse'])
                maes.append(result['mae'])
                valid_samples += 1
                
        avg_rmse = np.mean(rmses) if rmses else None
        avg_mae = np.mean(maes) if maes else None
        
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
        logger.info(f"Valid samples: {valid_samples}")
        if avg_rmse is not None and avg_mae is not None:
            logger.info(f"Average RMSE: {avg_rmse:.4f}")
            logger.info(f"Average MAE: {avg_mae:.4f}")
        logger.info(f"Total time taken: {total_time:.2f} seconds")
        logger.info("------------------------------------------------------")

if __name__ == "__main__":
    task_name = "VABench_VisualTrace"
    split = "test"
    disable_visualization = False
    use_flash_attention = False

    model_name = "Embodied-R1-3B"
    model_path = "/mnt/path/iffyuan/EasyR1/workdir/stage_2_embodiedr1_qwen2_5_vl_3b_version_reward_v7_date_0515/global_step_2064/actor/huggingface"
    dataset_path = "/mnt/path/iffyuan/all-seeing/all-seeing-v2/process_rl_data/FSD_visual_trace_rft_fsd_visual_trace_train_32790_test_300_0514/test.parquet"
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
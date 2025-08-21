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
import re
import textwrap
from PIL import Image
from collections import defaultdict
import numpy as np

def pixel_to_world(x, y, dep, img_width=2160, img_height=1440, camera_view_matrix_inv=np.array([[0., 1., 0., 0.],[0.9028605, 0., -0.42993355, 0.],[-0.42993355, 0., -0.9028605, 0.],[1., 0., 1.2, 1.]]), camera_proj_matrix=np.array([[1.7320507, 0., 0., 0.],[0., 2.5980759, 0., 0.],[0., 0., 0., -1.],[0., 0., 0.05, 0.]])):
    vinv = torch.tensor(camera_view_matrix_inv).float()
    proj = torch.tensor(camera_proj_matrix).float()
    
    fu = 2 / proj[0, 0]
    fv = 2 / proj[1, 1]
    
    centerU = img_width / 2
    centerV = img_height / 2
    
    x_tensor = torch.tensor(x).float()
    y_tensor = torch.tensor(y).float()
    dep_tensor = torch.tensor(dep).float()
    
    X_cam = (x_tensor - centerU) / img_width * dep_tensor * fu
    Y_cam = (y_tensor - centerV) / img_height * dep_tensor * fv
    Z_cam = dep_tensor
    
    point_cam = torch.tensor([X_cam, Y_cam, Z_cam, 1.0])
    point_world = point_cam @ vinv
    
    return point_world[:3].cpu().numpy().tolist()

def evaluate_posi(tar_pos, mode, sel_pos=None, sel_pos_1=None, sel_pos_2=None, sel_pos_all=None):
    succ = 0
    if mode in ["left", "right", "front", "back", "behind", "top"]:
        if mode == "left":
            succ += sel_pos[1] > tar_pos[1]
        elif mode == "right":
            succ += sel_pos[1] < tar_pos[1]
        elif mode == "front":
            succ += sel_pos[0] < tar_pos[0]
        elif mode == "back" or mode == "behind":
            succ += sel_pos[0] > tar_pos[0]
        elif mode == "top":
            succ += sel_pos[2] <= tar_pos[2]
    elif mode == "between":
        max_sel_pos_x = np.max([sel_pos_1[0], sel_pos_2[0]])
        max_sel_pos_y = np.max([sel_pos_1[1], sel_pos_2[1]])
        min_sel_pos_x = np.min([sel_pos_1[0], sel_pos_2[0]])
        min_sel_pos_y = np.min([sel_pos_1[1], sel_pos_2[1]])
        succ += (min_sel_pos_x < tar_pos[0] < max_sel_pos_x) or (min_sel_pos_y < tar_pos[0] < max_sel_pos_y)
    elif mode == "center":
        max_sel_pos_x = np.max(sel_pos_all, axis=0)[0]
        min_sel_pos_x = np.min(sel_pos_all, axis=0)[0]
        max_sel_pos_y = np.max(sel_pos_all, axis=0)[1]
        min_sel_pos_y = np.min(sel_pos_all, axis=0)[1]
        succ += (min_sel_pos_x < tar_pos[0] < max_sel_pos_x) and (min_sel_pos_y < tar_pos[1] < max_sel_pos_y)
    return succ

def calculate_xy_depth_reward(object:list,direction:str,predict_answer:list)->float:
    real_pos = predict_answer[0]
    if direction in ["left", "right", "front", "behind", "top"]:
            success = evaluate_posi(real_pos, direction, object[0])
    elif direction == "between":
            sel_pos_1 = object[0]
            sel_pos_2 = object[1]
            success = evaluate_posi(real_pos, direction, sel_pos_1=sel_pos_1, sel_pos_2=sel_pos_2)
    elif direction == "center":
            success = evaluate_posi(real_pos, direction, sel_pos_all=object)
    else:
        return 0.0

    return success

def open6dor_compute_score(predict_str: str, gt_data, logger) -> float:
    object = gt_data.get("object", [])
    direction = gt_data.get("direction", [])
    
    points = parse_points_from_output(predict_str)
    predict_depth = [1 for _ in range(len(points))]

    predict_answer = []
    for i in range(len(points)):
        predict_answer.append(pixel_to_world(points[i][0], points[i][1], predict_depth[i]))

    point_distance_reward = 0
    for i in range(len(predict_answer)):
        point_distance_reward += calculate_xy_depth_reward(object, direction, [predict_answer[i]])
    if point_distance_reward > 0:
        point_distance_reward = 1
    logger.info(f"point_distance_reward: {point_distance_reward}")
    return point_distance_reward

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

def process_sample(sample, model, processor, device, reasoning_model, instruct_following, gen_args):
    image_path = sample["images"][0].replace("\\", "/")
    base_path = "/mnt/path/iffyuan/all-seeing/all-seeing-v2/process_rl_data"
    abs_image_path = os.path.join(base_path, image_path)

    with open(abs_image_path, "rb") as image_file:
        image = Image.open(image_file).convert("RGB")

    task_instruction = sample["position_instruction"]
    question = f"You are currently a robot performing robotic manipulation tasks. The task instruction is: {task_instruction}. " \
                "Use 2D points to mark the target location where the object you need to manipulate in the task should ultimately be moved."
    question = question + '\n' + instruct_following
    question = textwrap.dedent(question).strip()
    
    answer = sample["answer"]
    position_tag = sample.get("position_tag", "unknown")
    doc_id = sample.get("id", "unknown")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {"type": "text", "text": question},
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
    score = open6dor_compute_score(predicted, answer, logger)

    logger.info("------------------------------------------------------")
    logger.info(f"ID: {doc_id}")
    logger.info(f"question: {question}")
    logger.info(f"predicted: {original_predicted}")
    logger.info(f"parsed points: {points}")
    logger.info(f"ground_truth: {answer}")
    logger.info(f"score: {score}")
    logger.info("------------------------------------------------------")

    return {
        'question_id': doc_id,
        'question': question,
        'predicted': predicted,
        'original_predicted': original_predicted,
        'points': points,
        'ground_truth': answer,
        'accuracy_score': float(score) if isinstance(score, (np.number, np.ndarray)) else score,
        'position_tag': position_tag
    }

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def main(task_name, model_name, model_path, reasoning_model, max_pixels, min_pixels, gen_args, instruct_following, dataset_path, use_flash_attention=False):
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
        with open(dataset_path, 'r') as f:
            test_data = json.load(f)
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
            reasoning_model, instruct_following, gen_args
        )
        if result is not None:
            local_results.append(result)
    
    all_results = accelerator.gather_for_metrics(local_results)
    
    if accelerator.is_main_process:
        if isinstance(all_results[0], list):
            final_results = [item for sublist in all_results for item in sublist]
        else:
            final_results = all_results
        
        tag_results = defaultdict(list)
        for result in final_results:
            for key, value in result.items():
                if isinstance(value, (np.number, np.ndarray)):
                    result[key] = float(value) if isinstance(value, (np.floating, np.float32, np.float64)) else int(value)
            
            tag = result.get('position_tag', 'unknown')
            tag_results[tag].append(result)
        
        total_samples = len(final_results)
        correct_predictions = sum(1 for result in final_results if result.get("accuracy_score") > 0)
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0
        
        tag_accuracies = {}
        for tag, results in tag_results.items():
            tag_total = len(results)
            tag_correct = sum(1 for result in results if result.get("accuracy_score") > 0)
            tag_accuracies[tag] = {
                'total': tag_total,
                'correct': tag_correct,
                'accuracy': tag_correct / tag_total if tag_total > 0 else 0
            }
        
        os.makedirs('logs/results', exist_ok=True)
        result_file_name = f'logs/results/{task_name}_{model_name}_{reasoning_model}.json'
        with open(result_file_name, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, ensure_ascii=False, indent=4, cls=NumpyEncoder)
            
        tag_result_file_name = f'logs/results/{task_name}_{model_name}_{reasoning_model}_tag_accuracies.json'
        with open(tag_result_file_name, 'w', encoding='utf-8') as f:
            json.dump(tag_accuracies, f, ensure_ascii=False, indent=4)

        end_time = time.time()
        total_time = end_time - start_time

        logger.info("------------------------------------------------------")
        logger.info(f"Model Name: {model_name}")
        logger.info("------------------------------------------------------")
        logger.info(f"Finished processing. Total samples: {total_samples}")
        logger.info(f"Number of correct predictions: {correct_predictions}")
        logger.info(f"Overall accuracy: {accuracy:.4f}")
        logger.info("------------------------------------------------------")
        logger.info("Accuracy by position_tag:")
        for tag, tag_acc in tag_accuracies.items():
            logger.info(f"  {tag}: {tag_acc['accuracy']:.4f} ({tag_acc['correct']}/{tag_acc['total']})")
        logger.info("------------------------------------------------------")
        logger.info(f"Total time taken: {total_time:.2f} seconds")
        logger.info("------------------------------------------------------")

if __name__ == "__main__":
    task_name = "Open6dor-Custom"
    model_name = "Embodied-R1-3B-2D"
    model_path = "IffYuan/Embodied-R1-3B-v1"
    dataset_path = "3d_dataset.json"
    reasoning_model = True
    use_flash_attention = False

    max_pixels = 3110400
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
    
    main(task_name, model_name, model_path, reasoning_model, max_pixels, min_pixels, gen_args, instruct_following, dataset_path, use_flash_attention)
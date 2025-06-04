from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoTokenizer
import torch
import torch.distributed as dist
from datasets import load_dataset
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
from PIL import Image, ImageDraw
import numpy as np
import io

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

def parse_points_from_output(output):
    import re
    points = []
    
    # <point>[[x1,y1], [x2,y2], ...]</point>
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


def check_points_in_mask(points, mask_image):
    if not points or mask_image is None:
        return 0, 0

    if mask_image.mode != 'L':
        logger.warning(f"Mask image mode is {mask_image.mode}, converting to 'L'.")
        mask_image = mask_image.convert('L')

    points_in_mask_foreground = 0
    width, height = mask_image.size

    for point in points:
        if len(point) == 2:
            x, y = point
            ix, iy = int(round(x)), int(round(y))

            if 0 <= ix < width and 0 <= iy < height:
                pixel_value = mask_image.getpixel((ix, iy))
                if pixel_value > 0:
                    points_in_mask_foreground += 1
    
    return points_in_mask_foreground, len(points)

def visualize_mask_and_points(image_pil, mask_pil, points, save_path):
    draw_image = image_pil.convert("RGBA")
    draw = ImageDraw.Draw(draw_image)
    
    if mask_pil:
        if mask_pil.mode != 'L':
            mask_pil = mask_pil.convert('L')
        
        mask_overlay = Image.new('RGBA', draw_image.size, (0,0,0,0))
        mask_draw = ImageDraw.Draw(mask_overlay)
        mask_color_rgb = (255, 0, 0)
        mask_alpha = 100

        for x in range(mask_pil.width):
            for y in range(mask_pil.height):
                if mask_pil.getpixel((x,y)) > 0:
                    mask_draw.point((x,y), fill=mask_color_rgb + (mask_alpha,))
        
        draw_image = Image.alpha_composite(draw_image, mask_overlay)
        draw = ImageDraw.Draw(draw_image)

    for point in points:
        if len(point) == 2:
            x, y = point
            ix, iy = int(round(x)), int(round(y))
            
            in_mask_foreground = False
            if mask_pil and 0 <= ix < mask_pil.width and 0 <= iy < mask_pil.height:
                if mask_pil.getpixel((ix, iy)) > 0:
                    in_mask_foreground = True
            
            outer_radius = 7
            draw.ellipse([(x-outer_radius, y-outer_radius), (x+outer_radius, y+outer_radius)], fill="white")
            
            inner_radius = 5
            inner_color = "lightgreen" if in_mask_foreground else "lightblue"
            draw.ellipse([(x-inner_radius, y-inner_radius), (x+inner_radius, y+inner_radius)], fill=inner_color)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    draw_image.convert("RGB").save(save_path)
    return save_path

def process_sample(sample, model, processor, device, reasoning_model, instruct_following, gen_args, vis_dir, disable_visualization=True):
    instruction = sample['problem'].strip()
    question = f"Provide one or more points coordinate of objects region" \
    f" this sentence describes: {instruction}. The results are presented in a format" \
    f" <point>[[x1,y1], [x2,y2], ...]</point>."

    if instruct_following:
        question = question + '\n' + instruct_following
    question = textwrap.dedent(question).strip()

    pil_image = sample.get('image')
    mask_image = sample.get('mask')
    doc_id = sample.get('question_id')
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": pil_image,
                }] + [{"type": "text", "text": question}],
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
        match = re.search(pattern, predicted, re.DOTALL)
        if match:
            predicted = match.group(1).strip()
        else:
            logger.info(f"oracle prediction: {predicted}")
            predicted = ""
    
    points = parse_points_from_output(predicted)
    
    points_in_mask, total_points = check_points_in_mask(points, mask_image)
    accuracy_score = points_in_mask / total_points if total_points > 0 else 0
    
    vis_path = None
    if not disable_visualization and total_points > 0 and random.random() < 0.1:
        vis_path = os.path.join(vis_dir, f"sample_{doc_id}_{accuracy_score:.2f}.jpg")
        visualize_mask_and_points(pil_image, mask_image, points, vis_path)
    
    logger.info("------------------------------------------------------")
    logger.info(f"Sample ID: {doc_id}")
    logger.info(f"question: {question}")
    logger.info(f"predicted: {original_predicted}")
    logger.info(f"parsed points: {points}")
    logger.info(f"points in mask: {points_in_mask}/{total_points}")
    logger.info(f"accuracy score: {accuracy_score:.4f}")
    if vis_path:
        logger.info(f"visualization saved to: {vis_path}")
    logger.info("------------------------------------------------------")

    return {
        'question_id': doc_id,
        'problem': instruction,
        'question': question,
        'original_predicted': original_predicted,
        'points': points,
        'points_in_mask': points_in_mask,
        'total_points': total_points,
        'accuracy_score': accuracy_score,
    }

def main(task_name, model_name, model_path, reasoning_model, max_pixels, min_pixels, gen_args, instruct_following, huggingface_dataset_name, split, vis_dir, disable_visualization=True, use_flash_attention=False):
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
    
    with accelerator.main_process_first():
        logger.info(f"Loading dataset from Hugging Face: {huggingface_dataset_name}")
        try:
            test_data = load_dataset(huggingface_dataset_name, split=split if split else "train")
            logger.info(f"Dataset loaded. Number of samples: {len(test_data)}")
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            return

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
        try:
            result = process_sample(
                test_data[idx], model, processor, local_device,
                reasoning_model, instruct_following, 
                gen_args, vis_dir, disable_visualization
            )
            if result is not None:
                local_results.append(result)
        except Exception as e:
            logger.error(f"Error processing sample at index {idx}: {e}", exc_info=True)

    all_results = accelerator.gather_for_metrics(local_results)
    
    if accelerator.is_main_process:
        if isinstance(all_results[0], list):
            final_results = [item for sublist in all_results for item in sublist]
        else:
            final_results = all_results
        
        total_samples_processed = len(final_results)
        if total_samples_processed > 0:
            avg_accuracy_score = sum(result.get('accuracy_score', 0) for result in final_results) / total_samples_processed
            total_points_predicted_all = sum(result.get('total_points', 0) for result in final_results)
            total_points_in_mask_all = sum(result.get('points_in_mask', 0) for result in final_results)
            overall_points_accuracy = total_points_in_mask_all / total_points_predicted_all if total_points_predicted_all > 0 else 0
        else:
            avg_accuracy_score = 0
            overall_points_accuracy = 0
            total_points_predicted_all = 0
            total_points_in_mask_all = 0
            
        os.makedirs('logs/results', exist_ok=True)
        result_file_name = f'logs/results/{task_name}_{model_name}_{reasoning_model}_{split}.json'
        with open(result_file_name, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, ensure_ascii=False, indent=4)
        
        end_time = time.time()
        total_time = end_time - start_time

        logger.info("------------------------------------------------------")
        logger.info(f"Model Name: {model_name}")
        logger.info("------------------------------------------------------")
        logger.info(f"Finished processing. Total samples: {total_samples_processed}")
        logger.info(f"Average accuracy score: {avg_accuracy_score:.4f}")
        logger.info(f"Total points predicted: {total_points_predicted_all}")
        logger.info(f"Points in mask: {total_points_in_mask_all}")
        logger.info(f"Overall points accuracy: {overall_points_accuracy:.4f}")
        logger.info(f"Total time taken: {total_time:.2f} seconds")
        logger.info("------------------------------------------------------")

if __name__ == "__main__":
    task_name = "Part-Affordance-2K"
    huggingface_dataset_name = "quark404/rgb_part_affordance" 
    split = "train" 
    disable_visualization = False
    use_flash_attention = False

    model_name = "Embodied-R1-3B"
    #model_path = "/mnt/kaiwu-group-x4/iffyuan/EasyR1/workdir/stage_2_embodiedr1_qwen2_5_vl_3b_version_reward_v7_date_0515/global_step_2064/actor/huggingface"
    model_path = "/mnt/kaiwu-group-x4/iffyuan/EasyR1/workdir/stage_2_point_rec_embodiedr1_qwen2_5_vl_3b_version_reward_v6_date_0503/global_step_784/actor/huggingface"
    reasoning_model = True
    
    vis_dir = f"logs/visualizations/{task_name}_{model_name}"
    os.makedirs(vis_dir, exist_ok=True)

    max_pixels = 3200000 
    min_pixels = 256 * 28 * 28 
    gen_args = {
        "temperature": 0.0,
        "top_p": None,
        "max_new_tokens": 512,
        "repetition_penalty": 1.0,
        "do_sample": False,
    }

    instruct_following = (
        r'You FIRST think about the reasoning process as an internal monologue and then provide the final answer. '
        r'The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags. '
        r'The answer consists only of several coordinate points, with the overall format being: '
        r'<think> reasoning process here </think><answer><point>[[x1, y1], [x2, y2], ...]</point></answer>'
    )

    current_time = time.strftime("%Y%m%d_%H%M%S")
    log_file_name = f"logs/inference_{task_name}_{model_name}_{current_time}.log"
    os.makedirs("logs", exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - Process %(process)d - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file_name)
        ]
    )
    logger = logging.getLogger(f"{task_name}_{model_name}")
    
    main(task_name, model_name, model_path, reasoning_model, max_pixels, min_pixels, gen_args, instruct_following, huggingface_dataset_name, split, vis_dir, disable_visualization, use_flash_attention) 
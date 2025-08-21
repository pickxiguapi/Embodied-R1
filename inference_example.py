import os
import re
import torch
import json
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

def process_vision_info(messages):
    image_inputs = []
    for msg in messages:
        for content in msg["content"] if isinstance(msg["content"], list) else [msg["content"]]:
            if isinstance(content, dict) and content.get("type") == "image":
                image_inputs.append(content["image"])
    return image_inputs, None

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

DEFAULT_CKPT_PATH = 'IffYuan/Embodied-R1-3B-v1'

EXAMPLE = [
    {
        "image": "exsample_data/put the red block on top of the yellow block.png",
        "text": "put the red block on top of the yellow block",
        "mode": "VTG"
    },
    {
        "image": "exsample_data/put pepper in pan.png",
        "text": "put pepper in pan",
        "mode": "RRG"
    },
    {
        "image":"exsample_data/roborefit_18992.png",
        "text":"bring me the camel model",
        "mode":"REG"
    },
    {
        "image":"exsample_data/handal_090002.png",
        "text":"loosening stuck bolts",
        "mode":"OFG"
    },
]

CONF_MODE = {
    "REG": {
        "template": (
            "Provide one or more points coordinate of objects region {instruction}. "
            "The results are presented in a format <point>[[x1,y1], [x2,y2], ...]</point>. "
            "You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
            "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags. "
            "The answer consists only of several coordinate points, with the overall format being: "
            "<think> reasoning process here </think><answer><point>[[x1, y1], [x2, y2], ...]</point></answer>"
        ),
        "description": "Referring Expression Grounding - Locating the coordinates of specified object regions within an image."
    },
    "OFG": {
        "template": (
            "Please provide the 2D points coordinate of the region this sentence describes: {instruction}. "
            "The results are presented in a format <point>[[x1,y1], [x2,y2], ...]</point>. "
            "You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
            "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags. "
            "The answer consists only of several coordinate points, with the overall format being: "
            "<think> reasoning process here </think><answer><point>[[x1, y1], [x2, y2], ...]</point></answer>"
        ),
        "description": "Object Affordance Grounding - Locating the 2D coordinates of specified object regions based on descriptions."
    },
    "RRG": {
        "template": (
            "You are currently a robot performing robotic manipulation tasks. The task instruction is: {instruction}. "
            "Use 2D points to mark the target location where the object you need to manipulate in the task should ultimately be moved. "
            "You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
            "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags. "
            "The answer consists only of several coordinate points, with the overall format being: "
            "<think> reasoning process here </think><answer><point>[[x1, y1], [x2, y2], ...]</point></answer>"
        ),
        "description": "Region Referring Grounding - Locating specific spatial target locations for robotic manipulation tasks."
    },
    "VTG": {
        "template": (
            "You are currently a robot performing robotic manipulation tasks. The task instruction is: {instruction}. "
            "Use 2D points to mark the manipulated object-centric waypoints to guide the robot to successfully complete the task. "
            "You must provide the points in the order of the trajectory, and the number of points must be 8. "
            "You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
            "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags. "
            "The answer consists only of several coordinate points, with the overall format being: "
            "<think> reasoning process here </think><answer><point>[[x1, y1], [x2, y2], ..., [x8, y8]]</point></answer>."
        ),
        "description": "Visual Trace Generation - Generating waypoints centered on manipulated objects to guide robots to complete tasks."
    }
}

def _visualize_coordinates_on_image(image, coordinates_str, mode="REG"):
    """Draw coordinates on image for visualization"""
    if not image or not coordinates_str:
        return image
    
    try:
        import ast
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import numpy as np
        from PIL import Image
        import io
        from matplotlib.collections import LineCollection
        from scipy.interpolate import interp1d
        
        # Parse coordinates
        coords = ast.literal_eval(coordinates_str)
        if not isinstance(coords, list):
            return image
        
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image.copy()
        
        # Calculate dynamic point size based on image resolution
        img_height, img_width = img_array.shape[:2]
        # Base size calculation: use the smaller dimension to avoid overly large points
        base_size = min(img_width, img_height)
        # Scale factor: larger images get bigger points, with reasonable min/max bounds
        if base_size <= 500:
            point_radius = max(3, int(base_size * 0.008))  # Small images: 3-4 pixels
        elif base_size <= 1000:
            point_radius = max(4, int(base_size * 0.010))  # Medium images: 4-10 pixels  
        else:
            point_radius = max(12, int(base_size * 0.018))  # Large images: 12+ pixels (unchanged)
        
        # Different size boost for different image sizes
        if base_size <= 500:
            point_radius = min(point_radius, 18)  # Small images: no extra boost, max 18px
        elif base_size <= 1000:
            point_radius = min(point_radius + 1, 20)  # Medium images: +1px, max 20px
        else:
            point_radius = min(point_radius + 2, 25)  # Large images: +2px, max 25px (unchanged)
        
        line_width = max(2, point_radius // 3)  # Line width proportional to point size
        
        print(f"Image size: {img_width}x{img_height}, Point radius: {point_radius}, Line width: {line_width}")
        
        # Create matplotlib figure
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.imshow(img_array)
        
        if mode == "VTG" and len(coords) >= 2:
            # VTG mode: Draw interpolated curve with gradient colors
            points = np.array(coords)
            x_coords, y_coords = points[:, 0], points[:, 1]
            
            # Create interpolated curve using scipy
            if len(coords) >= 3:
                # Use cubic interpolation for smooth curve
                t = np.linspace(0, 1, len(coords))
                t_new = np.linspace(0, 1, 100)  # 100 points for smooth curve
                
                fx = interp1d(t, x_coords, kind='cubic')
                fy = interp1d(t, y_coords, kind='cubic')
                
                x_new = fx(t_new)
                y_new = fy(t_new)
            else:
                # Linear interpolation for 2 points
                x_new = np.linspace(x_coords[0], x_coords[-1], 100)
                y_new = np.linspace(y_coords[0], y_coords[-1], 100)
            
            # Create gradient line segments
            points_curve = np.array([x_new, y_new]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points_curve[:-1], points_curve[1:]], axis=1)
            
            # Draw trajectory line with purple/magenta color similar to the reference image
            for i in range(len(segments)):
                # Create gradient from red to purple to blue
                ratio = i / max(1, len(segments) - 1)
                if ratio <= 0.5:
                    # Red to purple
                    r = 1.0 - ratio
                    g = 0.0
                    b = ratio * 2
                else:
                    # Purple to blue
                    r = 1.0 - ratio
                    g = 0.0 
                    b = 1.0
                
                color = (r, g, b, 0.8)
                line = plt.Line2D([segments[i][0][0], segments[i][1][0]], 
                                [segments[i][0][1], segments[i][1][1]], 
                                color=color, linewidth=line_width, alpha=0.8)
                ax.add_line(line)
            
            # Mark end point with square shape (size based on point_radius)
            end_x, end_y = coords[-1]
            square_size = point_radius * 2
            square = patches.Rectangle((end_x-point_radius, end_y-point_radius), square_size, square_size, 
                                     facecolor='blue', edgecolor='white', linewidth=line_width)
            ax.add_patch(square)
            
            # Mark all waypoints (including start point) with circles (no labels)
            for i, point in enumerate(coords):
                x, y = int(point[0]), int(point[1])
                # Skip the end point since it's already marked as a square
                if i < len(coords) - 1:
                    circle = patches.Circle((x, y), radius=point_radius, facecolor='purple', edgecolor='white', linewidth=line_width, alpha=0.7)
                    ax.add_patch(circle)
        
        else:
            # Other modes: Draw points with light blue inner circle and white border (no labels)
            for i, point in enumerate(coords):
                if len(point) == 2:
                    x, y = int(point[0]), int(point[1])
                    # Draw a circle at the coordinate - light blue inner, white outer (size based on image resolution)
                    circle = patches.Circle((x, y), radius=point_radius, facecolor='lightblue', edgecolor='white', linewidth=line_width)
                    ax.add_patch(circle)
        
        ax.axis('off')  # Hide axis
        plt.tight_layout()
        
        # Convert matplotlib figure to PIL Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=150)
        buf.seek(0)
        result_image = Image.open(buf)
        plt.close(fig)  # Close figure to free memory
        
        return result_image
            
    except Exception as e:
        print(f"Error visualizing coordinates: {e}")
        return image

def _load_model_processor(checkpoint_path=DEFAULT_CKPT_PATH, cpu_only=False, flash_attn2=False):
    if cpu_only:
        device_map = 'cpu'
    else:
        device_map = 'auto'

    if flash_attn2:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            checkpoint_path,
            torch_dtype='auto',
            attn_implementation=None,
            device_map=device_map
        )
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            checkpoint_path, 
            device_map=device_map
        )

    processor = AutoProcessor.from_pretrained(checkpoint_path)
    return model, processor

def _extract_model_output_parts(text):
    think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    think_content = think_match.group(1).strip() if think_match else ""
    
    answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    answer_content = answer_match.group(1).strip() if answer_match else ""
    
    point_match = re.search(r'<point>(.*?)</point>', answer_content, re.DOTALL)
    coordinates = point_match.group(1).strip() if point_match else ""
    
    return think_content, answer_content, coordinates

def _transform_messages(original_messages):
    transformed_messages = []
    for message in original_messages:
        new_content = []
        for item in message['content']:
            if 'image' in item:
                new_item = {'type': 'image', 'image': item['image']}
            elif 'text' in item:
                new_item = {'type': 'text', 'text': item['text']}
            elif 'video' in item:
                new_item = {'type': 'video', 'video': item['video']}
            else:
                continue
            new_content.append(new_item)

        new_message = {'role': message['role'], 'content': new_content}
        transformed_messages.append(new_message)

    return transformed_messages

def generate_response(model, processor, messages):
    messages = _transform_messages(messages)
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors='pt')
    inputs = inputs.to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=4096,
            temperature=0,
            top_p=1,
            repetition_penalty=1.05,
            do_sample=False
        )
    
    generated_text = processor.batch_decode(
        generated_ids[:, inputs['input_ids'].shape[1]:], 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )[0]
    
    return generated_text

def process_single_example(model, processor, example, base_image_dir="", output_dir="output"):

    image_path = os.path.join(base_image_dir, example["image"])
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Fail to load image {image_path}: {e}")
        return None
    

    mode = example["mode"]
    text = example["text"]
    

    if mode in CONF_MODE:
        template = CONF_MODE[mode]["template"]
        formatted_text = template.format(instruction=text)
    else:
        formatted_text = text
    

    messages = [{
        'role': 'user',
        'content': [
            {'type': 'image', 'image': image},
            {'type': 'text', 'text': formatted_text}
        ]
    }]
    
    print(f"Processing example: {example['image']}")
    print(f"Mode: {mode}")
    print(f"Instruction: {text}")
    

    try:
        response = generate_response(model, processor, messages)
        print(f"Model response: {response}")
        

        think_content, answer_content, coordinates = _extract_model_output_parts(response)
        
        visual_image = None
        if coordinates:
            visual_image = _visualize_coordinates_on_image(image, coordinates, mode)
            
  
            os.makedirs(output_dir, exist_ok=True)
            image_name = os.path.splitext(os.path.basename(example["image"]))[0]
            visual_image_path = os.path.join(output_dir, f"{image_name}_visualized.png")
            visual_image.save(visual_image_path)
            print(f"Visualization image saved: {visual_image_path}")
        

        return {
            "image": example["image"],
            "mode": mode,
            "instruction": text,
            "think_content": think_content,
            "answer_content": answer_content,
            "coordinates": coordinates,
            "full_response": response,
            "visual_image_path": visual_image_path if coordinates else None
        }
    except Exception as e:
        print(f"Error processing example: {e}")
        return None

def main():
    checkpoint_path = DEFAULT_CKPT_PATH
    cpu_only = False
    flash_attn2 = False
    base_image_dir = ""  
    output_dir = "output_results" 
    
    os.makedirs(output_dir, exist_ok=True)
    

    print("Loading model and processor...")
    model, processor = _load_model_processor(checkpoint_path, cpu_only, flash_attn2)
    print("Model loaded successfully")
    
    results = []
    for i, example in enumerate(EXAMPLE):
        print(f"\nProcessing example {i+1}/{len(EXAMPLE)}")
        result = process_single_example(model, processor, example, base_image_dir, output_dir)
        if result:
            results.append(result)
    

    output_file = os.path.join(output_dir, "inference_results.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nProcessing completed! Results saved to: {output_file}")
    

if __name__ == '__main__':
    main()

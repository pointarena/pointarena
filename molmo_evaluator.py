import os
import json
import argparse
import random
import csv
import re  # Add explicit import for re
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional, Union
import numpy as np
from PIL import Image, ImageDraw
from dotenv import load_dotenv
import io

# Import the same model interfaces and helpers as the main app
# from openai import OpenAI
# import google.generativeai as genai
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoProcessor, 
    AutoTokenizer, 
    GenerationConfig,
    # Qwen2_5_VLForConditionalGeneration, 
    AutoModelForVision2Seq,
    # LlavaOnevisionForConditionalGeneration
)
import base64
# import anthropic

# Load environment variables
load_dotenv()

# Configure API keys and clients
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
# xai_client = OpenAI(
#     api_key=os.getenv("XAI_API_KEY"),
#     base_url="https://api.x.ai/v1",
# )
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Constants
IMAGES_DIR = Path("images")
MASKS_DIR = Path("masks")
POINT_ON_MASK_DIR = Path("point_on_mask")  # New directory for visualization images

# Create the point_on_mask directory if it doesn't exist
POINT_ON_MASK_DIR.mkdir(exist_ok=True, parents=True)

# Load the image_filename to points mapping from CSV file
IMAGE_POINTS_MAP = {}
try:
    with open('pixmo_metadata.csv', 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['points'] and row['points'] != '[]':
                IMAGE_POINTS_MAP[row['image_filename']] = json.loads(row['points'])
    print(f"Loaded points data for {len(IMAGE_POINTS_MAP)} images from pixmo_metadata.csv")
except Exception as e:
    print(f"Error loading pixmo_metadata.csv: {e}")
    IMAGE_POINTS_MAP = {}

# Available models
OPENAI_MODELS = ["gpt-4o", "gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1-nano"]
GEMINI_MODELS = ["gemini-2.5-flash-preview-04-17", "gemini-2.5-pro-preview-05-06","gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-1.5-flash", "gemini-1.5-flash-8b", "gemini-1.5-pro"]
MOLMO_MODELS = ["Molmo-7B-D-0924", "Molmo-7B-O-0924", "Molmo-72B-0924"]
QWEN_MODELS = ["Qwen2.5-VL-7B-Instruct"]
LLAVA_MODELS = ["llava-onevision-qwen2-7b-ov-hf"]
CLAUDE_MODELS = ["claude-3-7-sonnet-20250219"]
GROK_MODELS = ["grok-2-vision-latest"]

# Use local models
USE_LOCAL_MODELS = True
if USE_LOCAL_MODELS:
    SAVED_MODELS_DIR = Path(os.getenv("SAVED_MODELS_DIR", "models"))
    SAVED_MODELS_DIR.mkdir(exist_ok=True, parents=True)
else:
    SAVED_MODELS_DIR = None

# Initialize Molmo model and processor (lazy loading)
molmo_model = None
molmo_processor = None

# Initialize Qwen model and processor (lazy loading)
qwen_model = None
qwen_processor = None

# Initialize LLaVA model and processor (lazy loading)
llava_model = None
llava_processor = None

# Add a utility function to print complete prompts near the beginning of the file, after imports
def print_complete_prompt(system_content, user_content, model_name, image_path):
    """Print the complete prompt including system content and user content."""
    print("\n" + "="*80)
    print(f"COMPLETE PROMPT FOR {model_name} ON {image_path}:")
    print("-"*80)
    if system_content:
        print(f"SYSTEM CONTENT:\n{system_content}")
        print("-"*80)
    print(f"USER CONTENT:\n{user_content}")
    print("="*80 + "\n")

def initialize_molmo(model_name="allenai/Molmo-7B-D-0924"):
    """Initialize Molmo model and processor if not already initialized."""
    global molmo_model, molmo_processor
    
    if molmo_model is None or molmo_processor is None:
        # Get model short name
        model_short_name = model_name.split('/')[-1]
        
        if USE_LOCAL_MODELS:
            # Use local model
            local_model_dir = SAVED_MODELS_DIR / model_short_name
            
            if not local_model_dir.exists():
                raise ValueError(f"Model directory does not exist: {local_model_dir}. Please ensure the model has been downloaded to this directory.")
            
            print(f"Loading Molmo model from local directory: {local_model_dir}")
            
            # Load from local directory
            molmo_processor = AutoProcessor.from_pretrained(
                local_model_dir,
                trust_remote_code=True,
                torch_dtype='auto',
                device_map='auto'
            )
            
            molmo_model = AutoModelForCausalLM.from_pretrained(
                local_model_dir,
                trust_remote_code=True,
                torch_dtype='auto',
                device_map='auto'
            )
        else:
            # Use remote model
            print(f"Loading Molmo model from Hugging Face: {model_name}")
            
            # Load processor from remote
            molmo_processor = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype='auto',
                device_map='auto'
            )
            
            # Load model from remote
            molmo_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype='auto',
                device_map='auto'
            )
        
    return molmo_model, molmo_processor


def get_original_points_info(image_path, category):
    """
    Get information about original points for steerable images.
    
    Args:
        image_path (str): Path to the image file
        category (str): Image category
        
    Returns:
        str: Information string about original points or empty string if not applicable
    """
    # Initialize with empty string
    original_points_info = ""
    
    # Only process for steerable category
    if category != "steerable":
        return original_points_info
    
    # Get the filename from the path
    image_filename = os.path.basename(image_path)
    
    # Check if we have original points data for this image
    if image_filename not in IMAGE_POINTS_MAP:
        return original_points_info
    
    # Get image dimensions
    img = Image.open(image_path)
    img_width, img_height = img.size
    
    # Get original points and convert to pixel coordinates
    original_points = IMAGE_POINTS_MAP[image_filename]
    original_points_in_pixels = []
    
    for point in original_points:
        # Convert percentage to pixel coordinates
        pixel_x = point["x"] * img_width / 100
        pixel_y = point["y"] * img_height / 100
        original_points_in_pixels.append(f"[{pixel_x:.1f}, {pixel_y:.1f}]")
    
    # Create information string if we have points
    if original_points_in_pixels:
        original_points_str = ", ".join(original_points_in_pixels)
        original_points_info = f"\nThe image contains an existing original point at pixel coordinates: {original_points_str}.\nThe query refers to this existing point."
    
    return original_points_info



def extract_points(text, image_w, image_h):
    """Extract points from text using multiple regex patterns.
    
    Extracts normalized coordinates (0-100 range) and converts them to pixel coordinates.
    Handles multiple formats like Click(x,y), (x,y), x="x" y="y", and p=xxx,yyy.
    
    Args:
        text: Text containing coordinate information
        image_w: Image width in pixels
        image_h: Image height in pixels
        
    Returns:
        List of points as numpy arrays in pixel coordinates
    """
    all_points = []
    for match in re.finditer(r"Click\(([0-9]+\.[0-9]), ?([0-9]+\.[0-9])\)", text):
        try:
            point = [float(match.group(i)) for i in range(1, 3)]
        except ValueError:
            pass
        else:
            point = np.array(point)
            if np.max(point) > 100:
                # Treat as an invalid output
                continue
            point /= 100.0
            point = point * np.array([image_w, image_h])
            all_points.append(point)

    for match in re.finditer(r"\(([0-9]+\.[0-9]),? ?([0-9]+\.[0-9])\)", text):
        try:
            point = [float(match.group(i)) for i in range(1, 3)]
        except ValueError:
            pass
        else:
            point = np.array(point)
            if np.max(point) > 100:
                # Treat as an invalid output
                continue
            point /= 100.0
            point = point * np.array([image_w, image_h])
            all_points.append(point)
    for match in re.finditer(r'x\d*="\s*([0-9]+(?:\.[0-9]+)?)"\s+y\d*="\s*([0-9]+(?:\.[0-9]+)?)"', text):
        try:
            point = [float(match.group(i)) for i in range(1, 3)]
        except ValueError:
            pass
        else:
            point = np.array(point)
            if np.max(point) > 100:
                # Treat as an invalid output
                continue
            point /= 100.0
            point = point * np.array([image_w, image_h])
            all_points.append(point)
    for match in re.finditer(r'(?:\d+|p)\s*=\s*([0-9]{3})\s*,\s*([0-9]{3})', text):
        try:
            point = [int(match.group(i)) / 10.0 for i in range(1, 3)]
        except ValueError:
            pass
        else:
            point = np.array(point)
            if np.max(point) > 100:
                # Treat as an invalid output
                continue
            point /= 100.0
            point = point * np.array([image_w, image_h])
            all_points.append(point)
    return all_points

def call_molmo(image_path, object_name, model_name="allenai/Molmo-7B-D-0924", category=None):
    """Call Molmo model to get points for the specified object."""
    try:
        # Initialize model and processor if not already done
        model, processor = initialize_molmo(model_name)
        
        # Load the image
        image = Image.open(image_path)
        img_width, img_height = image.size
        
        # Get information about original points for steerable images
        original_points_info = get_original_points_info(image_path, category)
        
    
        # Prepare the prompt based on category
        if category == "counting":
            prompt = f"""
            {object_name}
            {original_points_info}
            """
        else:
            prompt = f"""
            {object_name}
            {original_points_info}
            """
       
        
        # Process the image and text
        inputs = processor.process(
            images=[image],
            text=prompt
        )
        
        # Move inputs to the correct device and make a batch of size 1
        inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}
        
        # Generate output with torch.autocast for better performance
        with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", enabled=True, dtype=torch.bfloat16):
            output = model.generate_from_batch(
                inputs,
                GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
                tokenizer=processor.tokenizer
            )
        
        # Only get generated tokens; decode them to text
        generated_tokens = output[0, inputs['input_ids'].size(1):]
        content = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # First try to extract points using our enhanced extraction patterns
        extracted_points = extract_points(content, img_width, img_height)
        
        if extracted_points:
            # Convert to the standard format expected by the app
            points = [{"point": [float(p[0]), float(p[1])]} for p in extracted_points]
            print(f"Extracted {len(points)} points using extract_points function")
            
            # If not counting category and more than one point was returned, limit to first point
            if category != "counting" and len(points) > 1:
                points = [points[0]]
            
            return points
        
        # If no points found with enhanced extraction, try the original methods
        # Extract JSON from the response
        json_start = content.find('[')
        json_end = content.rfind(']') + 1
        if json_start != -1 and json_end != -1:
            json_str = content[json_start:json_end]
            
            # Try to extract coordinates using regex
            coords = re.findall(r'\[(\d+\.?\d*),\s*(\d+\.?\d*)\]', json_str)
            if coords:
                # Convert to standard [x, y] pixel coords format
                points = [{"point": [float(x), float(y)]} for x, y in coords]
                
                # If not counting category and more than one point was returned, limit to first point
                if category != "counting" and len(points) > 1:
                    points = [points[0]]
                
                return points
            
            # If regex fails, try to parse as JSON
            try:
                # Try to fix common JSON format errors
                raw_points = json.loads(json_str)
                
                # Handle different possible formats
                points = []
                if isinstance(raw_points, list):
                    for item in raw_points:
                        if isinstance(item, list) and len(item) == 2:
                            # Direct [x, y] format
                            x, y = item
                            points.append({"point": [float(x), float(y)]})
                        elif isinstance(item, dict) and "point" in item:
                            # {"point": [x, y]} format
                            if isinstance(item["point"], list) and len(item["point"]) == 2:
                                x, y = item["point"]
                                points.append({"point": [float(x), float(y)]})
                
                if points:
                    # If not counting category and more than one point was returned, limit to first point
                    if category != "counting" and len(points) > 1:
                        points = [points[0]]
                    return points
                
                # Fallback: attempt to extract just the coordinates
                numbers = re.findall(r'\d+\.?\d*', json_str)
                if len(numbers) >= 2:
                    # Try to pair them up as x,y coordinates
                    points = []
                    for i in range(0, len(numbers)-1, 2):
                        # Use direct x, y coordinate
                        points.append({"point": [float(numbers[i]), float(numbers[i+1])]})
                    
                    # If not counting category and more than one point was returned, limit to first point
                    if category != "counting" and len(points) > 1:
                        points = [points[0]]
                    
                    print(f"Extracted {len(points)} points using number extraction")
                    return points
                
                return []
            except Exception as e:
                print(f"Error parsing coordinates from {model_name} on {image_path}: {e}")
                return []
        else:
            print(f"Unable to extract coordinates from {model_name} on {image_path}")
            return []
    except Exception as e:
        print(f"Error calling {model_name} on {image_path}: {e}")
        return []


def load_mask(mask_path):
    """Load a binary mask from a PNG file."""
    try:
        # Load the mask image
        mask_img = Image.open(mask_path)
        
        # Convert to numpy array (values will be 0 for black and 255 for white)
        mask_array = np.array(mask_img)
        
        # Normalize to binary (True/False) mask
        # For grayscale, consider any value > 127 as True
        if len(mask_array.shape) == 2:
            binary_mask = mask_array > 127
        # For RGB, consider any channel > 127 as True (if any channel is white)
        elif len(mask_array.shape) == 3:
            binary_mask = np.any(mask_array > 127, axis=2)
        else:
            raise ValueError(f"Unexpected mask shape: {mask_array.shape}")
        
        return binary_mask
    except Exception as e:
        print(f"Error loading mask {mask_path}: {e}")
        return None

def is_point_in_mask(point, mask, img_width, img_height):
    """Check if a point is inside the mask."""
    if mask is None or point is None:
        print(f"[DEBUG MASK] Invalid mask or point: mask={mask is not None}, point={point}")
        return False
    
    # Unpack point (x, y format in pixel coordinates)
    x, y = point["point"]
    print(f"[DEBUG MASK] Checking point x={x}, y={y} (pixel coordinates)")
    
    # Convert to integers for indexing
    pixel_x = int(x)
    pixel_y = int(y)
    print(f"[DEBUG MASK] Pixel coordinates: x={pixel_x}, y={pixel_y}, image size: {img_width}x{img_height}")
    
    # Ensure coordinates are within image bounds
    if pixel_y < 0 or pixel_y >= img_height or pixel_x < 0 or pixel_x >= img_width:
        print(f"[DEBUG MASK] Point outside image bounds: x={pixel_x}, y={pixel_y}")
        return False
    
    # Check if point falls within the mask
    is_in_mask = mask[pixel_y, pixel_x]
    print(f"[DEBUG MASK] Point in mask: {is_in_mask}")
    return is_in_mask

def visualize_points_on_mask(image_path, mask, points, output_path, img_width, img_height):
    """Create a visualization of points overlaid on the mask and save it."""
    try:
        print(f"\n[DEBUG VISUALIZATION] Creating visualization for {output_path}")
        print(f"[DEBUG VISUALIZATION] Image dimensions: {img_width}x{img_height}")
        print(f"[DEBUG VISUALIZATION] Points to visualize: {points}")
        
        # Create a visualization of the mask (white foreground, black background)
        mask_vis = np.zeros((img_height, img_width, 3), dtype=np.uint8)
        mask_vis[mask] = 255  # White mask
        
        # Convert to PIL image
        mask_image = Image.fromarray(mask_vis, mode="RGB")
        
        # Draw points on the mask image
        draw = ImageDraw.Draw(mask_image)
        for point in points:
            # Unpack point (x, y format in pixel coordinates)
            x, y = point["point"]
            print(f"[DEBUG VISUALIZATION] Processing point: x={x}, y={y} (pixel coordinates)")
            
            # Convert to integers for drawing
            pixel_x = int(x)
            pixel_y = int(y)
            print(f"[DEBUG VISUALIZATION] Drawing at pixel coordinates: x={pixel_x}, y={pixel_y}")
            
            # Draw a cross at the point location (red for better visibility on white mask)
            point_size = max(5, min(img_width, img_height) // 100)  # Adaptive point size
            print(f"[DEBUG VISUALIZATION] Drawing point with size {point_size} at ({pixel_x}, {pixel_y})")
            draw.line((pixel_x - point_size, pixel_y, pixel_x + point_size, pixel_y), fill=(255, 0, 0), width=3)
            draw.line((pixel_x, pixel_y - point_size, pixel_x, pixel_y + point_size), fill=(255, 0, 0), width=3)
            
            # Add a circle around the point
            draw.ellipse((pixel_x - point_size, pixel_y - point_size, 
                         pixel_x + point_size, pixel_y + point_size), 
                         outline=(255, 0, 0), width=2)
        
        # Save the image
        mask_image.save(output_path)
        print(f"[DEBUG VISUALIZATION] Visualization saved to {output_path}")
        return True
    except Exception as e:
        print(f"[DEBUG VISUALIZATION] Error creating visualization: {e}")
        print(f"Error creating visualization: {e}")
        return False

def evaluate_model(model_name, model_type, progress_callback=None, resume=True):
    """Evaluate model performance on the dataset."""
    # Load data.json file
    try:
        with open("data.json", "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading data.json: {e}")
        if progress_callback:
            progress_callback(f"Error loading data.json: {e}")
        return
    
    # Select the appropriate model call function based on model type
    if model_type.lower() == "openai":
        model_func = call_openai
    elif model_type.lower() == "gemini":
        model_func = call_gemini
    elif model_type.lower() == "molmo":
        # For Molmo models, we need to add the complete path prefix if not already present
        if not model_name.startswith("allenai/"):
            model_name = f"allenai/{model_name}"
        model_func = call_molmo
    elif model_type.lower() == "qwen":
        # For Qwen models, we need to add the complete path prefix if not already present
        if not model_name.startswith("Qwen/"):
            model_name = f"Qwen/{model_name}"
        model_func = call_qwen
    elif model_type.lower() == "llava":
        # For LLaVA models, we need to add the complete path prefix if not already present
        if not model_name.startswith("llava-hf/"):
            model_name = f"llava-hf/{model_name}"
        model_func = call_llava
    elif model_type.lower() == "claude":
        model_func = call_claude
    elif model_type.lower() == "grok":
        model_func = call_grok
    else:
        print(f"Unknown model type: {model_type}")
        if progress_callback:
            progress_callback(f"Unknown model type: {model_type}")
        return
    
    # Define results file name
    results_file = f"results_{model_type}_{model_name.replace('/', '_')}_simple_prompt.json"
    if not os.path.exists("static_results"):
        os.makedirs("static_results")
    results_file = f"static_results/{results_file}"
    
    # Initialize or load existing results
    if resume and os.path.exists(results_file):
        try:
            with open(results_file, "r") as f:
                results = json.load(f)
            print(f"Resuming from existing results file with {results['success']} successes and {results['failure']} failures")
            if progress_callback:
                progress_callback(f"Resuming from existing results file with {results['success']} successes and {results['failure']} failures")
                
            # Get the list of already processed images
            processed_images = set(detail["image"] for detail in results["details"])
        except Exception as e:
            print(f"Error loading existing results file: {e}")
            if progress_callback:
                progress_callback(f"Error loading existing results file: {e}")
            results = {
                "total": 0,
                "success": 0,
                "failure": 0,
                "details": []
            }
            processed_images = set()
    else:
        results = {
            "total": 0,
            "success": 0,
            "failure": 0,
            "details": []
        }
        processed_images = set()
    
    # Process each image in the dataset
    item_count = 0
    for i, item in enumerate(data):
        # Skip items without mask_filename
        if "mask_filename" not in item:
            continue
        
        # Get image filename
        image_filename = item["image_filename"]
        
        # Skip already processed images if resuming
        if image_filename in processed_images:
            print(f"Skipping already processed image: {image_filename}")
            if progress_callback:
                progress_callback(f"Skipping already processed image: {image_filename}")
            continue
        
        results["total"] += 1
        item_count += 1
        
        # Update progress
        if progress_callback:
            progress_callback(f"Processing image {i+1}/{len(data)}: {image_filename}")
        
        # Get category from the data item
        category = item.get("category", "")
        
        # Find the image using both filename and category
        image_path = None
        if category:
            # Direct lookup using category and filename
            category_dir = IMAGES_DIR / category
            potential_path = category_dir / image_filename
            if category_dir.is_dir() and potential_path.exists():
                image_path = str(potential_path)
        
      
        
        if image_path is None:
            print(f"Image not found: {image_filename} in category: {category}")
            if progress_callback:
                progress_callback(f"Image not found: {image_filename} in category: {category}")
            results["failure"] += 1
            results["details"].append({
                "image": image_filename,
                "success": False,
                "reason": f"Image not found in category: {category}"
            })
            continue
        
        # Get mask path
        mask_filename = item["mask_filename"]
        mask_path = MASKS_DIR / mask_filename
        
        if not mask_path.exists():
            print(f"Mask not found: {mask_filename}")
            if progress_callback:
                progress_callback(f"Mask not found: {mask_filename}")
            results["failure"] += 1
            results["details"].append({
                "image": image_filename,
                "success": False,
                "reason": "Mask not found"
            })
            continue
        
        # Get query and category
        query = item["user_input"]
        expected_count = item.get("count", 1)  # Default to 1 for non-counting categories
        
        # Load image dimensions
        try:
            img = Image.open(image_path)
            img_width, img_height = img.size
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            if progress_callback:
                progress_callback(f"Error loading image {image_path}: {e}")
            results["failure"] += 1
            results["details"].append({
                "image": image_filename,
                "success": False,
                "reason": f"Error loading image: {e}"
            })
            continue
        
        # Load mask
        try:
            mask = load_mask(mask_path)
            if mask is None:
                raise ValueError("Failed to load mask")
        except Exception as e:
            print(f"Error loading mask {mask_path}: {e}")
            if progress_callback:
                progress_callback(f"Error loading mask {mask_path}: {e}")
            results["failure"] += 1
            results["details"].append({
                "image": image_filename,
                "success": False,
                "reason": f"Error loading mask: {e}"
            })
            continue
        
        # Call model to get points
        try:
            print(f"Testing {model_name} on image {image_filename} with query: '{query}'")
            if progress_callback:
                progress_callback(f"Testing {model_name} on image {image_filename} with query: '{query}'")
            points = model_func(image_path, query, model_name, category)
            
            # Check if the model returned any points
            if not points:
                print(f"No points returned for {image_filename}")
                if progress_callback:
                    progress_callback(f"No points returned for {image_filename}")
                results["failure"] += 1
                results["details"].append({
                    "image": image_filename,
                    "success": False,
                    "reason": "No points returned"
                })
                continue
            
            # Generate visualization of points on mask
            vis_filename = f"{Path(image_filename).stem}_{model_type}_{model_name.split('/')[-1]}.jpg"
            vis_path = POINT_ON_MASK_DIR / vis_filename
            visualize_points_on_mask(image_path, mask, points, vis_path, img_width, img_height)
            
            # For counting category, check if the number of points matches the expected count
            if category == "counting" and len(points) != expected_count:
                print(f"Count mismatch for {image_filename}: expected {expected_count}, got {len(points)}")
                if progress_callback:
                    progress_callback(f"Count mismatch for {image_filename}: expected {expected_count}, got {len(points)}")
                results["failure"] += 1
                results["details"].append({
                    "image": image_filename,
                    "success": False,
                    "reason": f"Count mismatch: expected {expected_count}, got {len(points)}"
                })
                continue
            
            # Check if all points are within the mask
            points_in_mask = True
            for point in points:
                if not is_point_in_mask(point, mask, img_width, img_height):
                    points_in_mask = False
                    break
            
            if points_in_mask:
                print(f"Success for {image_filename}")
                if progress_callback:
                    progress_callback(f"Success for {image_filename}")
                results["success"] += 1
                results["details"].append({
                    "image": image_filename,
                    "success": True,
                    "points_count": len(points),
                    "visualization": str(vis_path)  # Add visualization path to results
                })
            else:
                print(f"Failure for {image_filename}: points outside mask")
                if progress_callback:
                    progress_callback(f"Failure for {image_filename}: points outside mask")
                results["failure"] += 1
                results["details"].append({
                    "image": image_filename,
                    "success": False,
                    "reason": "Points outside mask",
                    "visualization": str(vis_path)  # Add visualization path to results
                })
        except Exception as e:
            print(f"Error processing {image_filename} with {model_name}: {e}")
            if progress_callback:
                progress_callback(f"Error processing {image_filename} with {model_name}: {e}")
            results["failure"] += 1
            results["details"].append({
                "image": image_filename,
                "success": False,
                "reason": f"Processing error: {e}"
            })
        
        # Save intermediate results every 100 images
        if item_count % 100 == 0:
            # Calculate current success rate
            if results["total"] > 0:
                success_rate = results["success"] / results["total"] * 100
                print(f"\nIntermediate results after {item_count} processed images:")
                print(f"Total images: {results['total']}")
                print(f"Successful predictions: {results['success']}")
                print(f"Failed predictions: {results['failure']}")
                print(f"Current success rate: {success_rate:.2f}%")
                
                if progress_callback:
                    progress_callback(f"\nIntermediate results after {item_count} processed images:")
                    progress_callback(f"Total images: {results['total']}")
                    progress_callback(f"Successful predictions: {results['success']}")
                    progress_callback(f"Failed predictions: {results['failure']}")
                    progress_callback(f"Current success rate: {success_rate:.2f}%")
            
            # Save intermediate results
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Intermediate results saved to {results_file}")
            if progress_callback:
                progress_callback(f"Intermediate results saved to {results_file}")
    
    # Calculate final success rate
    if results["total"] > 0:
        success_rate = results["success"] / results["total"] * 100
        print(f"\nEvaluation results for {model_name}:")
        print(f"Total images: {results['total']}")
        print(f"Successful predictions: {results['success']}")
        print(f"Failed predictions: {results['failure']}")
        print(f"Success rate: {success_rate:.2f}%")
        print(f"Visualizations saved to {POINT_ON_MASK_DIR}/")
        
        if progress_callback:
            progress_callback(f"\nEvaluation results for {model_name}:")
            progress_callback(f"Total images: {results['total']}")
            progress_callback(f"Successful predictions: {results['success']}")
            progress_callback(f"Failed predictions: {results['failure']}")
            progress_callback(f"Success rate: {success_rate:.2f}%")
            progress_callback(f"Visualizations saved to {POINT_ON_MASK_DIR}/")
        
        # Save final results
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Final results saved to {results_file}")
        if progress_callback:
            progress_callback(f"Final results saved to {results_file}")
    else:
        print("No images were processed. Check that data.json contains valid entries and masks exist.")
        if progress_callback:
            progress_callback("No images were processed. Check that data.json contains valid entries and masks exist.")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate model performance on point prediction tasks")
    parser.add_argument("--model", required=True, help="Model name to evaluate")
    parser.add_argument("--type", required=True, choices=["openai", "gemini", "molmo", "qwen", "llava", "claude", "grok"], 
                        help="Model type (openai, gemini, molmo, qwen, llava, claude, or grok)")
    parser.add_argument("--resume", action="store_true", help="Resume from previous evaluation state if available")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Start evaluation from beginning")
    parser.set_defaults(resume=True)
    
    args = parser.parse_args()
    
    # Validate model name based on type
    valid_models = {
        "openai": OPENAI_MODELS,
        "gemini": GEMINI_MODELS,
        "molmo": MOLMO_MODELS,
        "qwen": QWEN_MODELS,
        "llava": LLAVA_MODELS,
        "claude": CLAUDE_MODELS,
        "grok": GROK_MODELS
    }
    
    if args.type in valid_models and args.model not in valid_models[args.type]:
        print(f"Warning: {args.model} is not in the list of known {args.type} models.")
        print(f"Available {args.type} models: {', '.join(valid_models[args.type])}")
        confirm = input("Do you want to continue anyway? (y/n): ")
        if confirm.lower() != "y":
            return
    
    # Evaluate the specified model
    evaluate_model(args.model, args.type, resume=args.resume)

if __name__ == "__main__":
    main() 
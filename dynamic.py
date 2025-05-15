import os
import io
import json
import random
import time
import datetime
import uuid
import re
from pathlib import Path
from io import BytesIO
from dotenv import load_dotenv
import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from openai import OpenAI
import google.generativeai as genai
import base64
import boto3
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoProcessor, 
    GenerationConfig,
    Qwen2_5_VLForConditionalGeneration,
    # LlavaOnevisionForConditionalGeneration
)
import requests
import pandas as pd

# Load environment variables
load_dotenv()

# Configure API keys and clients
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
xai_client = OpenAI(
    api_key=os.getenv("XAI_API_KEY"),
    base_url="https://api.x.ai/v1",
)

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# Cloudflare R2 configuration
r2 = boto3.client(
    's3',
    endpoint_url=os.getenv("R2_ENDPOINT_URL"),
    aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY"),
)
R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME")

# Constants
IMAGES_DIR = Path("images")
USER_PHOTOS_DIR = Path("user_photos")
DYNAMIC_RESULTS_DIR = Path("dynamic_results")
DYNAMIC_RESULTS_FILE = Path("dynamic_results.json")

# Ensure directories exist
IMAGES_DIR.mkdir(exist_ok=True)
USER_PHOTOS_DIR.mkdir(exist_ok=True)
DYNAMIC_RESULTS_DIR.mkdir(exist_ok=True)

# Use local models
USE_LOCAL_MODELS = True
# If using local models, use the specified directory; otherwise use remote models
if USE_LOCAL_MODELS:
    SAVED_MODELS_DIR = Path(os.getenv("SAVED_MODELS_DIR", "models"))
    # Ensure the model saving directory exists
    SAVED_MODELS_DIR.mkdir(exist_ok=True, parents=True)
else:
    # If not using local models, will use Hugging Face's remote models
    SAVED_MODELS_DIR = None

# Available models
OPENAI_MODELS = ["gpt-4o"]
GEMINI_MODELS = ["gemini-2.5-flash-preview-04-17"]
# Molmo models
MOLMO_MODELS = ["Molmo-7B-D-0924"]
# Qwen models
QWEN_MODELS = ["Qwen2.5-VL-7B-Instruct"]
# LLaVA models
# LLAVA_MODELS = ["llava-onevision-qwen2-7b-ov-hf"]
# Claude models
# CLAUDE_MODELS = ["claude-3-7-sonnet-20250219"]
# Grok models
GROK_MODELS = ["grok-2-vision-latest"]
# UI display names
UI_MODEL_NAMES = ["Model A", "Model B"]
POINT_COLORS = ["red", "yellow"]  # Colors for the points from different models

# Molmo API configuration
MOLMO_API_URL = os.getenv("MOLMO_API_URL", "http://10.64.77.56:8000")

# Initialize Molmo model and processor (this will be replaced with API calls)
molmo_model = None
molmo_processor = None

# Initialize Qwen model and processor (lazy loading - will be initialized when first used)
qwen_model = None
qwen_processor = None

# Initialize LLaVA model and processor (lazy loading - will be initialized when first used)
llava_model = None
llava_processor = None

def initialize_qwen(model_name="Qwen/Qwen2.5-VL-7B-Instruct"):
    """Initialize Qwen model and processor if not already initialized."""
    global qwen_model, qwen_processor
    
    if qwen_model is None or qwen_processor is None:
        # Get model short name
        model_short_name = model_name.split('/')[-1]
        
        if USE_LOCAL_MODELS:
            # Use local model
            local_model_dir = SAVED_MODELS_DIR / model_short_name
            
            if not local_model_dir.exists():
                raise ValueError(f"Model directory does not exist: {local_model_dir}. Please ensure the model has been downloaded to this directory.")
            
            print(f"Loading Qwen model from local directory: {local_model_dir}")
            
            # Load from local directory
            qwen_processor = AutoProcessor.from_pretrained(
                local_model_dir,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map='auto'
            )
            
            qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                local_model_dir,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map='auto'
            )
        else:
            # Use remote model
            print(f"Loading Qwen model from Hugging Face: {model_name}")
            
            # Load processor from remote
            qwen_processor = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map='auto',
            )
            
            # Load model from remote
            qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map='auto'
            )
        
    return qwen_model, qwen_processor


def get_random_image():
    """Get a random image from the images directory."""
    # Define the subdirectories
    subdirs = ["affordable", "counting", "spatial", "reasoning", "steerable"]
    
    # Collect images from all subdirectories
    image_files = []
    for subdir in subdirs:
        subdir_path = IMAGES_DIR / subdir
        if subdir_path.exists() and subdir_path.is_dir():
            image_files.extend(list(subdir_path.glob("*.jpg")))
            image_files.extend(list(subdir_path.glob("*.png")))
            image_files.extend(list(subdir_path.glob("*.jpeg")))
            image_files.extend(list(subdir_path.glob("*.webp")))
    
    if not image_files:
        raise ValueError("No images found in the images directory or its subdirectories.")
    
    image_path = random.choice(image_files)
    # Get category from the parent directory name
    category = image_path.parent.name
    return str(image_path), image_path.name, category

def call_openai(image_path, object_name, model_name="gpt-4o", category=None):
    """Call OpenAI model to get points for the specified object."""
    # Read the image file
    with open(image_path, "rb") as image_file:
        # Encode the image as base64
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    
    # Get image dimensions
    img = Image.open(image_path)
    img_width, img_height = img.size
    
    # Determine MIME type based on file extension
    file_extension = os.path.splitext(image_path)[1].lower()
    if file_extension == '.png':
        mime_type = "image/png"
    elif file_extension in ['.jpg', '.jpeg']:
        mime_type = "image/jpeg"
    elif file_extension == '.webp':
        mime_type = "image/webp"
    elif file_extension == '.gif':
        mime_type = "image/gif"
    else:
        # Default to jpeg for other formats
        mime_type = "image/jpeg"
    
    # Unified prompt format that works for both counting and non-counting
    prompt = f"""
    Point to {object_name}.
    The image dimensions are width={img_width}px, height={img_height}px.
    The answer should follow the json format: [{{"point": <point>}}, ...]. 
    IMPORTANT: The points MUST be in [x, y] format where x is the horizontal position (left-to-right) and y is the vertical position (top-to-bottom) in PIXEL COORDINATES (not normalized).
    Example: For a point in the center of the image, return [width/2, height/2].
    """
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that can identify objects in images and provide their coordinates."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}}
                ]}
            ],
        )
        
        content = response.choices[0].message.content
        # Extract JSON from the response
        json_start = content.find('[')
        json_end = content.rfind(']') + 1
        if json_start != -1 and json_end != -1:
            json_str = content[json_start:json_end]
            points = json.loads(json_str)
            return points
        else:
            return []
    except Exception as e:
        print(f"Error with {model_name} on {image_path}: {e}")
        return []

def call_gemini(image_path, object_name, model_name="gemini-2.0-flash", category=None):
    """Call Gemini to get points for the specified object and return unnormalized pixel coordinates."""
    try:
        model = genai.GenerativeModel(model_name)

        # Load and process image
        img = Image.open(image_path)
        img_width, img_height = img.size
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Convert to bytes
        img_byte_arr = io.BytesIO()
        ext = os.path.splitext(image_path)[1].lower()
        img_format = {
            '.png': 'PNG',
            '.jpg': 'JPEG',
            '.jpeg': 'JPEG',
            '.webp': 'WEBP',
            '.gif': 'GIF'
        }.get(ext, 'JPEG')
        img.save(img_byte_arr, format=img_format)
        image_data = img_byte_arr.getvalue()

        # Determine MIME type
        mime_type = {
            'PNG': 'image/png',
            'JPEG': 'image/jpeg',
            'WEBP': 'image/webp',
            'GIF': 'image/gif'
        }.get(img_format, 'image/jpeg')

        # Prompt
        prompt = f"""
        {object_name}
        The answer should follow the json format: [{{"point": [y, x], "label": "a"}}, ...]. The points are normalized to 0-1000 and use [y, x] format.
        """

        # Construct API call
        prompt_parts = [prompt.strip(), {"mime_type": mime_type, "data": image_data}]
        print(f"\n[INFO] Calling Gemini ({model_name}) on image {image_path}...")
        response = model.generate_content(prompt_parts)

        content = response.text if response.parts else ""
        print(f"[DEBUG] Raw Gemini Output:\n{content}")

        # Extract JSON block
        json_start = content.find("[")
        json_end = content.rfind("]") + 1
        if json_start == -1 or json_end == -1:
            raise ValueError("Failed to extract JSON array from output.")
        
        raw_points = json.loads(content[json_start:json_end])

        # Convert points from [y, x] normalized to [x, y] in pixels
        points = []
        for item in raw_points:
            if isinstance(item, dict) and "point" in item:
                y, x = item["point"]
                pixel_x = (x / 1000.0) * img_width
                pixel_y = (y / 1000.0) * img_height
                points.append({"point": [pixel_x, pixel_y]})

        # Return only the points (limit to one if not counting)
        if category != "counting" and len(points) > 1:
            points = [points[0]]

        return points

    except Exception as e:
        print(f"[ERROR] Failed to call Gemini: {e}")
        import traceback
        traceback.print_exc()
        return []
    
def call_grok(image_path, object_name, model_name="grok-2-vision-latest", category=None):
    """Call Grok to get points for the specified object."""
    try:
        # Determine MIME type based on file extension
        file_extension = os.path.splitext(image_path)[1].lower()
        if file_extension == '.png':
            mime_type = "image/png"
        elif file_extension in ['.jpg', '.jpeg']:
            mime_type = "image/jpeg"
        elif file_extension == '.webp':
            mime_type = "image/webp"
        elif file_extension == '.gif':
            mime_type = "image/gif"
        else:
            # Default to jpeg for other formats
            mime_type = "image/jpeg"
        
        # Read the image file as base64
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Get image dimensions
        img = Image.open(image_path)
        img_width, img_height = img.size
        
        # Unified prompt format that works for both counting and non-counting
        prompt = f"""
        Point to {object_name}.
        The image dimensions are width={img_width}px, height={img_height}px.
        The answer should follow the json format: [{{"point": <point>}}, ...]. 
        IMPORTANT: The points MUST be in [x, y] format where x is the horizontal position (left-to-right) and y is the vertical position (top-to-bottom) in PIXEL COORDINATES (not normalized).
        Example: For a point in the center of the image, return [width/2, height/2].
        """
        
        # Set up messages for the XAI API call
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{image_data}",
                            "detail": "high",
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            },
        ]
        
        # Call the XAI API
        response = xai_client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.01,
        )
        
        content = response.choices[0].message.content
        # Extract JSON from the response
        json_start = content.find('[')
        json_end = content.rfind(']') + 1
        if json_start != -1 and json_end != -1:
            json_str = content[json_start:json_end]
            points = json.loads(json_str)
            return points
        else:
            return []
    except Exception as e:
        print(f"Error with {model_name} on {image_path}: {e}")
        return []

def call_qwen(image_path, object_name, model_name="Qwen/Qwen2.5-VL-7B-Instruct", category=None):
    """Call Qwen model to get points for the specified object."""
    try:
        # Initialize model and processor if not already done
        model, processor = initialize_qwen(model_name)
        
        # Load the image
        image = Image.open(image_path)
        img_width, img_height = image.size
        print(f"[DEBUG] Image dimensions: {img_width}x{img_height}")
        
        # Get information about original points for steerable images
        original_points_info = get_original_points_info(image_path, category)
        
        # Define system content
        system_content = "You are a helpful assistant."
        
        # Prepare the prompt based on category
        if category == "counting":
            prompt = f"""
            {object_name}
            Output its coordinates in XML format <points x y>object</points>.
            {original_points_info}
            """
        else:
            prompt = f"""
            {object_name}
            Output its coordinates in XML format <points x y>object</points>.
            {original_points_info}
           """
        
        # Print complete prompt
        print_complete_prompt(system_content, prompt, model_name, image_path)
        
        # Qwen2.5-VL uses a specific format for multimodal inputs
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]}
        ]
        
        # Apply chat template
        text = processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Process the input
        inputs = processor(
            text=text,
            images=image,
            return_tensors="pt"
        ).to(model.device)
        
        # Generate output with torch.autocast for better performance
        with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", enabled=True, dtype=torch.bfloat16):
            output_ids = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False
            )
        
        # Decode the generated tokens
        content = processor.tokenizer.decode(output_ids[0][inputs.input_ids.size(1):], skip_special_tokens=True)
        
        print(f"\n[DEBUG] Raw Qwen output for {object_name} in {image_path}:")
        print(content)
        
        # First try to parse XML format: <points x y>object</points>
        import re
        # xml_pattern = r'<points\s+x1="(\d+\.?\d*)"\s+y1="(\d+\.?\d*)".*?>.*?</points>'
        # xml_matches = re.findall(xml_pattern, content)

        # 1) find all xN/yN pairs
        coord_pattern = re.compile(r'x(\d+)="(\d+\.?\d*)"\s+y\1="(\d+\.?\d*)"')
        # returns list of (index, x, y)
        raw = coord_pattern.findall(content)
        coords = [(float(x), float(y)) for _, x, y in raw]
        xml_matches = coords
        
        if xml_matches:
            print(f"[DEBUG] XML points format detected: {xml_matches}")
            # Convert to standard point format
            points = [{"point": [float(x), float(y)]} for x, y in xml_matches]
            print(f"[DEBUG] Extracted points from XML: {points}")
            
            # If not counting category and more than one point was returned, limit to first point
            if category != "counting" and len(points) > 1:
                print(f"[DEBUG] Multiple points detected but not counting category. Limiting to first point.")
                points = [points[0]]
            
            return points
        
        # If no XML format found, try to extract JSON as a fallback
        # Extract JSON from the response
        json_start = content.find('[')
        json_end = content.rfind(']') + 1
        if json_start != -1 and json_end != -1:
            json_str = content[json_start:json_end]
            print(f"[DEBUG] Extracted JSON string: {json_str}")
            
            # Try to extract coordinates using regex first
            import re
            
            # First try to find point_2d format which returns pixel coordinates
            pixel_coords = re.findall(r'"point_2d":\s*\[(\d+\.?\d*),\s*(\d+\.?\d*)\]', json_str)
            if pixel_coords:
                print(f"[DEBUG] Pixel coordinates extracted via 'point_2d': {pixel_coords}")
                # These are already in pixel coordinates
                points = [{"point": [float(x), float(y)]} for x, y in pixel_coords]
                print(f"[DEBUG] Extracted points: {points}")
                
                # If not counting category and more than one point was returned, limit to first point
                if category != "counting" and len(points) > 1:
                    print(f"[DEBUG] Multiple points detected but not counting category. Limiting to first point.")
                    points = [points[0]]
                
                return points
            
            # If no point_2d, try regular [x,y] format
            coords = re.findall(r'\[(\d+\.?\d*),\s*(\d+\.?\d*)\]', json_str)
            if coords:
                print(f"[DEBUG] Coordinates extracted via regex: {coords}")
                # Convert to standard pixel format
                points = [{"point": [float(x), float(y)]} for x, y in coords]
                print(f"[DEBUG] Extracted points: {points}")
                
                # If not counting category and more than one point was returned, limit to first point
                if category != "counting" and len(points) > 1:
                    print(f"[DEBUG] Multiple points detected but not counting category. Limiting to first point.")
                    points = [points[0]]
                
                return points
            
            # If regex fails, try to parse as JSON
            try:
                # Try to fix common JSON format errors
                raw_points = json.loads(json_str)
                print(f"[DEBUG] Raw points parsed from JSON: {raw_points}")
                
                # Handle different possible formats
                points = []
                if isinstance(raw_points, list):
                    for item in raw_points:
                        # Check for point_2d format (direct pixel coordinates)
                        if isinstance(item, dict) and "point_2d" in item:
                            if isinstance(item["point_2d"], list) and len(item["point_2d"]) == 2:
                                x, y = item["point_2d"]
                                points.append({"point": [float(x), float(y)]})
                        # Check for direct [x, y] format
                        elif isinstance(item, list) and len(item) == 2:
                            x, y = item
                            points.append({"point": [float(x), float(y)]})
                        # Check for {"point": [x, y]} format
                        elif isinstance(item, dict) and "point" in item:
                            if isinstance(item["point"], list) and len(item["point"]) == 2:
                                x, y = item["point"]
                                points.append({"point": [float(x), float(y)]})
                
                if points:
                    print(f"[DEBUG] Points after parsing: {points}")
                    # If not counting category and more than one point was returned, limit to first point
                    if category != "counting" and len(points) > 1:
                        print(f"[DEBUG] Multiple points detected but not counting category. Limiting to first point.")
                        points = [points[0]]
                    return points
                
                print("[DEBUG] No valid points extracted from JSON")
                
                # As a last resort, check for any pair of numbers in the content
                number_pairs = re.findall(r'(\d+\.?\d*)\s*[,\s]\s*(\d+\.?\d*)', content)
                if number_pairs:
                    print(f"[DEBUG] Found potential coordinate pairs: {number_pairs}")
                    # Use the first pair as a point
                    x, y = number_pairs[0]
                    points = [{"point": [float(x), float(y)]}]
                    return points
                
                return []
            except Exception as e:
                print(f"[DEBUG] Error parsing coordinates from JSON: {e}")
                print(f"Error parsing coordinates from {model_name} on {image_path}: {e}")
                
                # As a last resort, check for any pair of numbers in the content
                number_pairs = re.findall(r'(\d+\.?\d*)\s*[,\s]\s*(\d+\.?\d*)', content)
                if number_pairs:
                    print(f"[DEBUG] Found potential coordinate pairs: {number_pairs}")
                    # Use the first pair as a point
                    x, y = number_pairs[0]
                    points = [{"point": [float(x), float(y)]}]
                    return points
                
                return []
        else:
            # If no JSON format detected, try to find any pair of numbers as coordinates
            print(f"[DEBUG] No JSON brackets found in response. Looking for coordinate pairs.")
            number_pairs = re.findall(r'(\d+\.?\d*)\s*[,\s]\s*(\d+\.?\d*)', content)
            if number_pairs:
                print(f"[DEBUG] Found potential coordinate pairs: {number_pairs}")
                # Convert to points
                points = [{"point": [float(x), float(y)]} for x, y in number_pairs]
                
                # If not counting category and more than one point was returned, limit to first point
                if category != "counting" and len(points) > 1:
                    points = [points[0]]
                
                return points
            
            print(f"[DEBUG] Unable to extract coordinates from {model_name} on {image_path}")
            return []
    except Exception as e:
        print(f"Error calling {model_name} on {image_path}: {e}")
        print(f"Exception details: {str(e)}")
        import traceback
        traceback.print_exc()
        return []
    


def call_molmo(image_path, object_name, model_name="allenai/Molmo-7B-D-0924", category=None):
    """Call Molmo model to get points for the specified object using the API."""
    try:
        # Clean model name (remove allenai/ prefix if present)
        model_name_clean = model_name.replace("allenai/", "")
        
        # Read the image file as base64
        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # Prepare API request data
        request_data = {
            "image_base64": image_base64,
            "object_name": object_name,
            "model_name": model_name_clean,
            "category": category
        }
        
        # Call the Molmo API
        response = requests.post(
            f"{MOLMO_API_URL}/molmo/point",
            json=request_data,
            timeout=60  # 60 second timeout
        )
        
        # Check response status
        if response.status_code != 200:
            print(f"Error calling Molmo API: {response.status_code} - {response.text}")
            return []
        
        # Parse the response
        result = response.json()
        
        # Check for errors
        if result.get("error"):
            print(f"Error from Molmo API: {result['error']}")
            return []
        
        # Return the points
        return result.get("points", [])
    except Exception as e:
        print(f"Error calling Molmo API: {e}")
        return []


def draw_points_on_image(image_path, points, color="red"):
    """Draw points on the image."""
    try:
        # Convert string paths to Path objects for consistency
        if isinstance(image_path, str):
            image_path = Path(image_path)
        
        # Check if file exists
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Open the image
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)
        
        img_width, img_height = img.size
        
        for point_data in points:
            if "point" in point_data:
                # Points are in [x, y] format in pixel coordinates
                x, y = point_data["point"]
                
                # Convert to integers for drawing
                pixel_x = int(x)
                pixel_y = int(y)
                
                # Draw a circle at the point
                radius = max(8, min(img_width, img_height) // 100)  # Adaptive radius based on image size
                draw.ellipse(
                    [(pixel_x - radius, pixel_y - radius), (pixel_x + radius, pixel_y + radius)],
                    fill=color,
                    outline=color
                )
                
                # Draw a cross at the point for better visibility
                draw.line((pixel_x - radius, pixel_y, pixel_x + radius, pixel_y), fill=color, width=2)
                draw.line((pixel_x, pixel_y - radius, pixel_x, pixel_y + radius), fill=color, width=2)
        
        return img
    except Exception as e:
        import traceback
        print(f"Error in draw_points_on_image: {e}")
        traceback.print_exc()
        # Return a blank image with error message
        err_img = Image.new('RGB', (400, 400), color='white')
        ImageDraw.Draw(err_img).text((10, 10), f"Error: {e}", fill='black')
        return err_img

def save_image_to_dynamic_results(image, filename_prefix):
    """Save an image to the dynamic_results directory."""
    filename = f"{filename_prefix}_{uuid.uuid4()}.png"
    filepath = DYNAMIC_RESULTS_DIR / filename
    image.save(filepath)
    return filename

def save_result_to_json(test_data):
    """Save test data to dynamic_results.json."""
    if DYNAMIC_RESULTS_FILE.exists():
        try:
            with open(DYNAMIC_RESULTS_FILE, "r") as f:
                results = json.load(f)
        except json.JSONDecodeError:
            results = []
    else:
        results = []
    
    results.append(test_data)
    
    with open(DYNAMIC_RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved results to {DYNAMIC_RESULTS_FILE}")

def process_test(object_name, image_path, image_filename, category=None):
    """Process the test with randomly selected models and return the results."""
    try:
        print(f"\n===== PROCESS TEST =====")
        print(f"Object name: {object_name}")
        print(f"Image path: {image_path}")
        print(f"Image filename: {image_filename}")
        print(f"Category: {category}")
        
        # Verify the image path exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Try to open the image to verify it's valid
        try:
            test_image = Image.open(image_path)
            print(f"Successfully loaded image: {test_image.size} {test_image.mode}")
            test_image.close()
        except Exception as img_err:
            raise ValueError(f"Failed to load image: {img_err}")
        
        # Create a list of all available models
        all_models = []
        
        # Add all OpenAI models
        for model_name in OPENAI_MODELS:
            all_models.append((model_name, call_openai))
        
        # Add all Gemini models
        for model_name in GEMINI_MODELS:
            all_models.append((model_name, call_gemini))
        
        # Add all Claude models
        # for model_name in CLAUDE_MODELS:
        #     all_models.append((model_name, call_claude))
            
        # Add all Grok models
        for model_name in GROK_MODELS:
            all_models.append((model_name, call_grok))
        
        # Add all Molmo models
        for model_name in MOLMO_MODELS:
            # For Molmo models, we need to add the complete path prefix
            full_model_name = f"allenai/{model_name}" if not model_name.startswith("allenai/") else model_name
            all_models.append((full_model_name, call_molmo))
        
        # Add all Qwen models
        for model_name in QWEN_MODELS:
            # For Qwen models, we need to add the complete path prefix
            full_model_name = f"Qwen/{model_name}" if not model_name.startswith("Qwen/") else model_name
            all_models.append((full_model_name, call_qwen))
            
        # Add all LLaVA models
        # for model_name in LLAVA_MODELS:
        #     # For LLaVA models, we need to add the complete path prefix
        #     full_model_name = f"llava-hf/{model_name}" if not model_name.startswith("llava-hf/") else model_name
        #     all_models.append((full_model_name, call_llava))
        
        # Randomly select two models for comparison
        selected_models = random.sample(all_models, 2)
        while "grok-2-vision-latest" in selected_models and "gpt-4o" in selected_models:
            selected_models = random.sample(all_models, 2)
        
        # Assign the selected models to Model A and Model B
        model1_name, model1_func = selected_models[0]
        model2_name, model2_func = selected_models[1]
        
        print(f"Selected models: Model A = {model1_name}, Model B = {model2_name}")
        
        # Call both models with their respective functions
        try:
            print(f"Calling model1: {model1_name}")
            model1_points = model1_func(image_path, object_name, model1_name, category)
            print(f"Model1 returned {len(model1_points)} points")
        except Exception as e:
            print(f"Error calling {model1_name}: {e}")
            model1_points = []
        
        try:
            print(f"Calling model2: {model2_name}")
            model2_points = model2_func(image_path, object_name, model2_name, category)
            print(f"Model2 returned {len(model2_points)} points")
        except Exception as e:
            print(f"Error calling {model2_name}: {e}")
            model2_points = []
        
        # Draw points on images
        try:
            print(f"Drawing points for model1")
            model1_image = draw_points_on_image(image_path, model1_points, POINT_COLORS[0])
        except Exception as e:
            print(f"Error drawing points for model1: {e}")
            # Create a blank image with error text
            model1_image = Image.new('RGB', (400, 400), color='white')
            ImageDraw.Draw(model1_image).text((10, 10), f"Error: {e}", fill='black')
        
        try:
            print(f"Drawing points for model2")
            model2_image = draw_points_on_image(image_path, model2_points, POINT_COLORS[1])
        except Exception as e:
            print(f"Error drawing points for model2: {e}")
            # Create a blank image with error text
            model2_image = Image.new('RGB', (400, 400), color='white')
            ImageDraw.Draw(model2_image).text((10, 10), f"Error: {e}", fill='black')
        
        # Save images to dynamic_results directory
        model1_image_filename = save_image_to_dynamic_results(model1_image, f"model1_{model1_name.replace('/', '_')}")
        model2_image_filename = save_image_to_dynamic_results(model2_image, f"model2_{model2_name.replace('/', '_')}")
        
        # Prepare test data
        test_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "image_filename": image_filename,
            "object_name": object_name,
            "category": category,
            "model1_name": model1_name,
            "model2_name": model2_name,
            "model1_image_filename": model1_image_filename,
            "model2_image_filename": model2_image_filename,
            "model1_points": model1_points,  # Save the actual coordinate points for model1
            "model2_points": model2_points,  # Save the actual coordinate points for model2
        }
        
        return model1_image, model2_image, test_data
    except Exception as e:
        import traceback
        print(f"Error in process_test: {e}")
        traceback.print_exc()
        raise

def save_selection(test_data, selected_model_index):
    """Save the user's selection to the JSON file."""
    # Determine the winning model name
    if selected_model_index == "both_good":
        winning_model = "both_good"
    elif selected_model_index == "both_bad":
        winning_model = "both_bad"
    else:
        winning_model = test_data[f"model{selected_model_index+1}_name"]
    
    # Update test data with the winner
    test_data["winning_model"] = winning_model
    
    # Save to JSON file
    save_result_to_json(test_data)
    
    return "Your selection has been saved."

def save_user_photo(image):
    """Save a user-uploaded photo to the user_photos directory."""
    try:
        # Generate random filename
        filename = f"user_photo_{uuid.uuid4()}.jpg"
        filepath = USER_PHOTOS_DIR / filename
        
        # More detailed debug info
        print(f"Saving user photo: {type(image)}")
        print(f"User photo directory: {USER_PHOTOS_DIR.absolute()}")
        print(f"User photo filepath: {filepath.absolute()}")
        
        # Handle different types of image inputs
        if image is None:
            raise ValueError("Received None instead of an image")
        elif isinstance(image, np.ndarray):
            # Convert numpy array to PIL Image
            print(f"Converting numpy array to PIL Image, shape: {image.shape}")
            if image.ndim == 3 and image.shape[2] == 3:
                # RGB image
                pil_image = Image.fromarray(image)
            else:
                # Try to convert other array formats
                pil_image = Image.fromarray(np.asarray(image, dtype=np.uint8))
        elif isinstance(image, Image.Image):
            # Already a PIL image
            print(f"Image is already a PIL Image with size: {image.size}")
            pil_image = image
        elif isinstance(image, str) and os.path.isfile(image):
            # It's a file path, load the image
            print(f"Loading image from path: {image}")
            pil_image = Image.open(image)
        else:
            # Try to interpret the image data
            print(f"Attempting to interpret image data")
            try:
                # Try loading as file path
                if isinstance(image, str) and os.path.exists(image):
                    pil_image = Image.open(image)
                else:
                    # Last resort - try direct conversion
                    pil_image = Image.fromarray(np.array(image, dtype=np.uint8))
            except Exception as img_error:
                raise TypeError(f"Could not convert to PIL Image: {img_error}, type: {type(image)}")
        
        # Ensure it's in RGB mode for JPG output
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Save the image
        print(f"Saving PIL Image with size: {pil_image.size}")
        pil_image.save(str(filepath))
        print(f"Saved user photo to: {filepath}")
        
        # Verify file was saved
        if not filepath.exists():
            raise IOError(f"Failed to save image to {filepath}")
        
        # Add a small delay to ensure file is fully saved
        time.sleep(0.5)
        
        # Return path and filename, and "user_upload" as the category
        return str(filepath), filename, "user_upload"
    except Exception as e:
        import traceback
        print(f"Error saving user photo: {e}")
        traceback.print_exc()
        raise

def calculate_expected_score(rating_a, rating_b):
    """
    Calculate the expected score for player A in a match against player B.
    Expected score is a number between 0 and 1 (essentially a probability).
    """
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

def update_elo(rating_a, rating_b, score_a):
    """
    Update the Elo rating for player A based on their score against player B.
    Score is 1 for a win, 0 for a loss.
    """
    K_FACTOR = 32  # Standard K-factor for Elo
    expected_a = calculate_expected_score(rating_a, rating_b)
    return rating_a + K_FACTOR * (score_a - expected_a)

def generate_elo_leaderboard():
    """Generate a leaderboard of model performance based on ELO ratings."""
    INITIAL_RATING = 1000  # Initial rating for all models
    
    # Load the battle results
    if not DYNAMIC_RESULTS_FILE.exists():
        return "No results available yet.", None
    
    try:
        with open(DYNAMIC_RESULTS_FILE, 'r') as f:
            results = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return "Error loading results file.", None
    
    if not results:
        return "No results available yet.", None
    
    # Initialize ratings dictionary with default rating
    ratings = {
        "Qwen/Qwen2.5-VL-7B-Instruct": INITIAL_RATING,
        "allenai/Molmo-7B-D-0924": INITIAL_RATING,
        "gpt-4o": INITIAL_RATING,
        "grok-2-vision-latest": INITIAL_RATING,
        "gemini-2.5-flash-preview-04-17": INITIAL_RATING
    }
    
    # Track wins, losses, and ties for each model
    stats = {model: {"wins": 0, "losses": 0, "ties": 0} for model in ratings.keys()}
    
    # Process all battles
    valid_battles = 0
    for battle in results:
        model1 = battle.get("model1_name", "")
        model2 = battle.get("model2_name", "")
        winner = battle.get("winning_model", "")
        
        # Skip battles with incomplete data or where both models are good or bad
        if not model1 or not model2 or not winner or winner in ["both_good", "both_bad"]:
            continue
        
        # Skip if models aren't in our ratings dictionary
        if model1 not in ratings or model2 not in ratings:
            continue
        
        valid_battles += 1
        
        # Update stats
        if winner == model1:
            stats[model1]["wins"] += 1
            stats[model2]["losses"] += 1
            
            # Update Elo ratings
            new_rating_1 = update_elo(ratings[model1], ratings[model2], 1)
            new_rating_2 = update_elo(ratings[model2], ratings[model1], 0)
            
            ratings[model1] = new_rating_1
            ratings[model2] = new_rating_2
            
        elif winner == model2:
            stats[model2]["wins"] += 1
            stats[model1]["losses"] += 1
            
            # Update Elo ratings
            new_rating_1 = update_elo(ratings[model1], ratings[model2], 0)
            new_rating_2 = update_elo(ratings[model2], ratings[model1], 1)
            
            ratings[model1] = new_rating_1
            ratings[model2] = new_rating_2
    
    # Create a DataFrame for the leaderboard
    leaderboard = []
    for model, rating in ratings.items():
        model_stats = stats[model]
        win_rate = model_stats["wins"] / (model_stats["wins"] + model_stats["losses"]) if (model_stats["wins"] + model_stats["losses"]) > 0 else 0
        leaderboard.append({
            "Model": model,
            "Elo Rating": round(rating, 1),
            "Wins": model_stats["wins"],
            "Losses": model_stats["losses"],
            "Win Rate": f"{win_rate:.2%}"
        })
    
    leaderboard_df = pd.DataFrame(leaderboard)
    leaderboard_df = leaderboard_df.sort_values(by="Elo Rating", ascending=False)
    
    # Create a bar chart visualization
    plt.figure(figsize=(10, 6))
    models = leaderboard_df["Model"]
    model_ratings = leaderboard_df["Elo Rating"]
    
    # Create a color map
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(models)))
    
    plt.bar(models, model_ratings, color=colors)
    plt.axhline(y=INITIAL_RATING, color='r', linestyle='--', alpha=0.7, label='Initial Rating (1000)')
    
    plt.title('Model Elo Ratings Leaderboard', fontsize=16)
    plt.ylabel('Elo Rating', fontsize=14)
    plt.ylim(min(model_ratings) - 50 if len(model_ratings) > 0 else 950, max(model_ratings) + 50 if len(model_ratings) > 0 else 1050)
    
    # Rotate model names for better readability
    plt.xticks(rotation=30, ha='right', fontsize=12)
    
    # Add rating values on top of bars
    for i, v in enumerate(model_ratings):
        plt.text(i, v + 5, str(round(v, 1)), ha='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.legend()
    
    # Save to BytesIO
    img_bytes = BytesIO()
    plt.savefig(img_bytes, format='png', dpi=150, bbox_inches='tight')
    img_bytes.seek(0)
    
    # Convert BytesIO to PIL Image that Gradio can handle
    img_pil = Image.open(img_bytes)
    
    plt.close()
    
    # Format the leaderboard as markdown for display
    markdown_table = "| Rank | Model | Elo Rating | Wins | Losses | Win Rate |\n"
    markdown_table += "|------|-------|-----------|------|--------|----------|\n"
    
    for i, row in leaderboard_df.iterrows():
        markdown_table += f"| {i+1} | {row['Model']} | {row['Elo Rating']} | {row['Wins']} | {row['Losses']} | {row['Win Rate']} |\n"
    
    markdown_summary = f"### ELO Leaderboard\nBased on {valid_battles} valid battles\n\n{markdown_table}"
    
    return markdown_summary, img_pil

def get_original_points_info(image_path, category):
    """Get information about original points for steerable images."""
    # This function is a placeholder - we should implement this if needed
    return ""

def print_complete_prompt(system_content, prompt, model_name, image_path):
    """Print the complete prompt for debugging purposes."""
    print(f"\n[DEBUG] Complete prompt for {model_name} on {image_path}:")
    print(f"System: {system_content}")
    print(f"User: {prompt}")

def ui_components():
    """Create and return the Gradio UI components."""
    with gr.Blocks(title="Point Battle", theme=gr.themes.Soft()) as app:
        gr.Markdown("# Point Battle")
        gr.Markdown("### Comparing Different LLMs on Image Pointing Tasks")
        
        # State variables
        current_image_path = gr.State("")
        current_image_filename = gr.State("")
        current_image_category = gr.State("")
        current_test_data = gr.State(None)
        active_tab = gr.State(0)  # 0 for Random Image, 1 for Upload Photo
        
        with gr.Tabs() as main_tabs:
            with gr.Tab("Battle"):
                with gr.Row():
                    with gr.Column(scale=1):
                        # Image selection tab group
                        with gr.Tabs() as tabs:
                            with gr.Tab("Random Image"):
                                # Random image display
                                random_image_display = gr.Image(label="Test Image", type="filepath")
                                
                                # New random image button
                                new_image_btn = gr.Button("Get New Random Image", variant="secondary")
                            
                            with gr.Tab("Upload Photo"):
                                # Image upload component - modified to automatically use uploaded images
                                uploaded_image = gr.Image(
                                    label="Upload or Take Photo", 
                                    type="pil",  # Use "pil" type for better compatibility
                                    sources=["upload", "webcam"],
                                    image_mode="RGB",  # Force RGB mode
                                    height=400,
                                    width=400,
                                    elem_id="upload-image"
                                )
                                
                                # Status message specifically for uploads
                                upload_status = gr.Markdown("Photo will be processed automatically after upload")
                                
                                # Hide the debug textbox
                                uploaded_image_path = gr.Textbox(
                                    label="Uploaded Image Path (Debug)",
                                    value="",
                                    visible=False  # Hidden in production
                                )
                        
                        # Object input
                        with gr.Row():
                            gr.Markdown("### I want to point to")
                            object_input = gr.Textbox(placeholder="Enter object to point to", label="")
                        
                        # Submit button
                        submit_btn = gr.Button("Submit", variant="primary")
                        
                        # Status message
                        status_msg = gr.Markdown("")
                    
                    with gr.Column(scale=2):
                        with gr.Row():
                            # Model outputs
                            model1_output = gr.Image(label=UI_MODEL_NAMES[0], type="pil")
                            model2_output = gr.Image(label=UI_MODEL_NAMES[1], type="pil")
                        
                        with gr.Row():
                            # Selection buttons
                            model1_btn = gr.Button(f"Select {UI_MODEL_NAMES[0]}", interactive=False)
                            model2_btn = gr.Button(f"Select {UI_MODEL_NAMES[1]}", interactive=False)
                        
                        with gr.Row():
                            # Both Good and Both Bad buttons
                            both_good_btn = gr.Button("Both Good", interactive=False)
                            both_bad_btn = gr.Button("Both Bad", interactive=False)
                        
                        # Selection result
                        selection_result = gr.Markdown("")
            
            with gr.Tab("Leaderboard"):
                # Leaderboard display
                with gr.Row():
                    refresh_btn = gr.Button("Refresh Leaderboard", variant="primary")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        leaderboard_text = gr.Markdown("")
                    
                    with gr.Column(scale=1):
                        leaderboard_plot = gr.Image(type="pil", label="ELO Ratings Visualization")
        
        # Function to refresh leaderboard
        def refresh_leaderboard():
            text, plot = generate_elo_leaderboard()
            return text, plot
        
        # Connect refresh button
        refresh_btn.click(
            refresh_leaderboard,
            inputs=[],
            outputs=[leaderboard_text, leaderboard_plot]
        )
        
        # Load leaderboard on page load
        def load_initial_leaderboard():
            text, plot = generate_elo_leaderboard()
            return text, plot
        
        # Load a random image on page load
        def load_random_image():
            try:
                image_path, image_filename, category = get_random_image()
                # Return values for state variables and UI components
                # Set active tab to 0 (Random Image)
                return (
                    image_path, image_path, image_filename, category, 0, 
                    None, None, None, "",
                    gr.update(interactive=False), gr.update(interactive=False),
                    gr.update(interactive=False), gr.update(interactive=False)
                )
            except ValueError as e:
                return (
                    None, None, "", "", 0, 
                    None, None, None, str(e),
                    gr.update(interactive=False), gr.update(interactive=False),
                    gr.update(interactive=False), gr.update(interactive=False)
                )
        
        # Set active tab to Random Image
        def set_random_tab():
            return 0
        
        # Function to handle user photo upload - now triggered automatically when image is uploaded
        def handle_upload(image):
            print(f"\n===== UPLOAD DEBUG =====")
            print(f"Image received type: {type(image)}")
            
            if image is None:
                print("ERROR: Received None image")
                return None, None, "", "", 1, ""
            
            try:
                if isinstance(image, np.ndarray):
                    print(f"Image is numpy array with shape: {image.shape}")
                elif isinstance(image, Image.Image):
                    print(f"Image is PIL Image with size: {image.size}")
                elif isinstance(image, str):
                    print(f"Image is a file path: {image}")
                else:
                    print(f"Unknown image type: {type(image)}")
                
                print(f"Processing uploaded image: {type(image)}")
                image_path, image_filename, category = save_user_photo(image)
                print(f"Saved user photo to: {image_path}")
                print(f"Saved user photo filename: {image_filename}")
                print(f"Category: {category}")
                
                # Check if file exists before returning
                if not os.path.exists(image_path):
                    print(f"WARNING: Saved image path does not exist: {image_path}")
                
                # Return the path and set active tab to 1 (Upload Photo)
                return image_path, image_filename, category, "Photo uploaded, please enter the object to point to", 1, image_path
            except Exception as e:
                import traceback
                print(f"ERROR in handle_upload: {str(e)}")
                traceback.print_exc()
                return None, None, "", f"Error saving photo: {str(e)}", 1, ""
        
        # Process the test when submit button is clicked
        def on_submit(object_name, image_path, image_filename, image_category, tab_index):
            print(f"\n===== SUBMIT DEBUG =====")
            print(f"Object name: {object_name}")
            print(f"Image path: {image_path}")
            print(f"Image filename: {image_filename}")
            print(f"Tab index: {tab_index}")
            
            if not object_name.strip():
                print("ERROR: No object name provided")
                return None, None, None, "Please enter the object to point to", gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False)
            
            if not image_path:
                print("ERROR: No image path provided")
                return None, None, None, "Please select or upload an image first", gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False)
            
            # Verify image path exists
            image_file = Path(image_path)
            if not image_file.exists():
                error_msg = f"ERROR: Image file does not exist: {image_path}"
                print(error_msg)
                return None, None, None, f"Image file does not exist: {image_path}", gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False)
            
            print(f"Submit using image path: {image_path} (tab: {tab_index}, category: {image_category})")
            
            try:
                model1_img, model2_img, test_data = process_test(object_name, image_path, image_filename, image_category)
                print(f"Test completed successfully, model outputs ready")
                # Enable model selection buttons
                return model1_img, model2_img, test_data, "Please select which model's marking is more accurate", gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True)
            except Exception as e:
                import traceback
                print(f"ERROR in on_submit: {str(e)}")
                traceback.print_exc()
                return None, None, None, f"Error: {str(e)}", gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False)
        
        # Function to handle model selection and load next image
        def on_model_selected_then_next(test_data, model_index):
            if test_data:
                # Get the selected model name for display
                if model_index == "both_good":
                    selected_model_display = "Both models are good"
                elif model_index == "both_bad":
                    selected_model_display = "Both models are bad"
                else:
                    # Get the actual model name, not just "Model A" or "Model B"
                    selected_model_display = test_data[f"model{model_index+1}_name"]
                
                # Save the selection
                result = save_selection(test_data, model_index)
                # Include the model name in the result message
                result = f"Your selection has been saved. You selected {selected_model_display}."
                
                # Also update the leaderboard
                leaderboard_text, leaderboard_plot = generate_elo_leaderboard()
            else:
                result = "Please submit a test first"
                leaderboard_text, leaderboard_plot = None, None
            
            # Generate a new random image
            try:
                new_path, new_filename, category = get_random_image()
                return result, gr.update(interactive=False), gr.update(interactive=False), new_path, new_path, new_filename, category, 0, None, None, None, "", gr.update(interactive=False), gr.update(interactive=False), leaderboard_text, leaderboard_plot
            except ValueError as e:
                return result, gr.update(interactive=False), gr.update(interactive=False), None, None, "", "", 0, None, None, None, str(e), gr.update(interactive=False), gr.update(interactive=False), leaderboard_text, leaderboard_plot
        
        # Connect buttons to functions
        new_image_btn.click(
            load_random_image,
            outputs=[
                current_image_path, random_image_display, current_image_filename, current_image_category, active_tab, 
                model1_output, model2_output, current_test_data, status_msg,
                model1_btn, model2_btn, both_good_btn, both_bad_btn
            ]
        )
        
        # Connect the image upload component directly to the handle_upload function
        # This will automatically process the image when it's uploaded
        uploaded_image.change(
            handle_upload,
            inputs=[uploaded_image],
            outputs=[current_image_path, current_image_filename, current_image_category, upload_status, active_tab, uploaded_image_path]
        )
        
        submit_btn.click(
            on_submit,
            inputs=[object_input, current_image_path, current_image_filename, current_image_category, active_tab],
            outputs=[model1_output, model2_output, current_test_data, status_msg, model1_btn, model2_btn, both_good_btn, both_bad_btn]
        )
        
        model1_btn.click(
            on_model_selected_then_next,
            inputs=[current_test_data, gr.State(0)],  # 0 for model1
            outputs=[selection_result, model1_btn, model2_btn, current_image_path, random_image_display, current_image_filename, current_image_category, active_tab, model1_output, model2_output, current_test_data, status_msg, both_good_btn, both_bad_btn, leaderboard_text, leaderboard_plot]
        )
        
        model2_btn.click(
            on_model_selected_then_next,
            inputs=[current_test_data, gr.State(1)],  # 1 for model2
            outputs=[selection_result, model1_btn, model2_btn, current_image_path, random_image_display, current_image_filename, current_image_category, active_tab, model1_output, model2_output, current_test_data, status_msg, both_good_btn, both_bad_btn, leaderboard_text, leaderboard_plot]
        )
        
        # Add click handlers for both_good and both_bad buttons
        both_good_btn.click(
            on_model_selected_then_next,
            inputs=[current_test_data, gr.State("both_good")],
            outputs=[selection_result, model1_btn, model2_btn, current_image_path, random_image_display, current_image_filename, current_image_category, active_tab, model1_output, model2_output, current_test_data, status_msg, both_good_btn, both_bad_btn, leaderboard_text, leaderboard_plot]
        )
        
        both_bad_btn.click(
            on_model_selected_then_next,
            inputs=[current_test_data, gr.State("both_bad")],
            outputs=[selection_result, model1_btn, model2_btn, current_image_path, random_image_display, current_image_filename, current_image_category, active_tab, model1_output, model2_output, current_test_data, status_msg, both_good_btn, both_bad_btn, leaderboard_text, leaderboard_plot]
        )
        
        # Load a random image and leaderboard on page load
        app.load(
            lambda: (*load_random_image(), *load_initial_leaderboard()),
            outputs=[
                current_image_path, random_image_display, current_image_filename, current_image_category, active_tab, 
                model1_output, model2_output, current_test_data, status_msg,
                model1_btn, model2_btn, both_good_btn, both_bad_btn,
                leaderboard_text, leaderboard_plot
            ]
        )
    
    return app

if __name__ == "__main__":
    app = ui_components()
    app.launch(share=True, server_name="0.0.0.0", server_port=7860) 
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
from openai import OpenAI
import google.generativeai as genai
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoProcessor, 
    AutoTokenizer, 
    GenerationConfig,
    Qwen2_5_VLForConditionalGeneration, 
    AutoModelForVision2Seq,
    LlavaOnevisionForConditionalGeneration
)
import base64
import anthropic

# Load environment variables
load_dotenv()

# Configure API keys and clients
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
xai_client = OpenAI(
    api_key=os.getenv("XAI_API_KEY"),
    base_url="https://api.x.ai/v1",
)
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

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
OPENAI_MODELS = ["gpt-4o", "gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4v"]
GEMINI_MODELS = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.5-flash-preview-04-17", "gemini-2.5-pro-preview-05-06","gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-1.5-flash", "gemini-1.5-flash-8b", "gemini-1.5-pro"]
MOLMO_MODELS = ["Molmo-7B-D-0924", "Molmo-7B-O-0924", "Molmo-72B-0924"]
QWEN_MODELS = ["Qwen2.5-VL-7B-Instruct", "Qwen2.5-VL-32B-Instruct", "Qwen2.5-VL-72B-Instruct"]
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
                # attn_implementation="flash_attention_2",
                device_map='auto'
            )

            print(qwen_model.hf_device_map)
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
                # attn_implementation="flash_attention_2",
                device_map='auto'
            )
        
    return qwen_model, qwen_processor

def initialize_llava(model_name="llava-hf/llava-onevision-qwen2-7b-ov-hf"):
    """Initialize LLaVA-OneVision model and processor if not already initialized."""
    global llava_model, llava_processor
    
    if llava_model is None or llava_processor is None:
        # Get model short name
        model_short_name = model_name.split('/')[-1]
        
        if USE_LOCAL_MODELS:
            # Use local model
            local_model_dir = SAVED_MODELS_DIR / model_short_name
            
            if not local_model_dir.exists():
                raise ValueError(f"Model directory does not exist: {local_model_dir}. Please ensure the model has been downloaded to this directory.")
            
            print(f"Loading LLaVA-OneVision model from local directory: {local_model_dir}")
            
            # Load the model and processor using standard approach - LLaVA-OneVision HF should work with this
            try:
                print("[DEBUG] Loading processor from local directory")
                llava_processor = AutoProcessor.from_pretrained(
                    local_model_dir,
                    trust_remote_code=True,
                    torch_dtype=torch.float16
                )
                
                print("[DEBUG] Loading model from local directory")
                # Use the specialized model class for LLaVA-OneVision HF version
                llava_model = LlavaOnevisionForConditionalGeneration.from_pretrained(
                    local_model_dir,
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    device_map='auto'
                )
                print(f"[DEBUG] Model type: {type(llava_model).__name__}")
            except Exception as e:
                print(f"[DEBUG] Error loading model: {e}")
                raise
        else:
            # Use remote model
            print(f"Loading LLaVA-OneVision model from Hugging Face: {model_name}")
            
            # Load processor and model using standard approach
            try:
                print("[DEBUG] Loading processor from Hugging Face")
                llava_processor = AutoProcessor.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.float16
                )
                
                print("[DEBUG] Loading model from Hugging Face")
                # Use the specialized model class for LLaVA-OneVision HF version
                llava_model = LlavaOnevisionForConditionalGeneration.from_pretrained(
                    model_name,
                    trust_remote_code=True, 
                    torch_dtype=torch.float16,
                    device_map='auto'
                )
                print(f"[DEBUG] Model type: {type(llava_model).__name__}")
            except Exception as e:
                print(f"[DEBUG] Error loading model: {e}")
                raise
        
    return llava_model, llava_processor

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

def call_openai(image_path, object_name, model_name="gpt-4o", category=None):
    """Call OpenAI model to get points for the specified object."""
    # Modify the filename to add "_marked" before the extension (SOM)
    base_name, file_extension = os.path.splitext(image_path)
    image_path = f"{base_name}_marked{file_extension}"

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
    
    # Get information about original points for steerable images
    original_points_info = get_original_points_info(image_path, category)
    
    # Check if category is counting - limit points accordingly
    if category == "counting":
        prompt = f"""
        This image contains numbered marks (e.g., [1], [2]) overlaid on various objects or regions. 
        Using these visual prompts in numbered mark to help reason step by step for the given tasks:{object_name}.
        The image dimensions are width={img_width}px, height={img_height}px.{original_points_info}
        The answer should follow the json format: [{{"point": <point>}}, ...]. 
        IMPORTANT: The points MUST be in [x, y] format where x is the horizontal position (left-to-right) and y is the vertical position (top-to-bottom) in PIXEL COORDINATES (not normalized).
        Example: For a point in the center of the image, return [width/2, height/2].
        """
    else:
        prompt = f"""
        This image contains numbered marks (e.g., [1], [2]) overlaid on various objects or regions. 
        Using these visual prompts in numbered mark to help reason step by step for the given tasks:{object_name}.
        The image dimensions are width={img_width}px, height={img_height}px.{original_points_info}
        The answer should follow the json format: [{{"point": <point>}}]. 
        IMPORTANT: Return EXACTLY ONE POINT. The point MUST be in [x, y] format where x is the horizontal position (left-to-right) and y is the vertical position (top-to-bottom) in PIXEL COORDINATES (not normalized).
        Example: For a point in the center of the image, return [width/2, height/2].
        """
    
    # Define system content
    system_content = "You are a helpful assistant that can identify objects in images and provide their coordinates."
    
    # Print complete prompt
    print_complete_prompt(system_content, prompt, model_name, image_path)
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_content},
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
            # If not counting category and more than one point was returned, limit to first point
            if category != "counting" and len(points) > 1:
                points = [points[0]]
            return points
        else:
            return []
    except Exception as e:
        print(f"Error with {model_name} on {image_path}: {e}")
        return []

def call_gemini(image_path, object_name, model_name="gemini-2.0-flash", category=None):
    """Call Gemini to get points for the specified object."""
    try:
        # Configure the model
        model = genai.GenerativeModel(model_name)
        
        # Get image dimensions
        img = Image.open(image_path)
        img_width, img_height = img.size
        
        # Ensure image is in a supported format
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Determine MIME type based on file extension
        file_extension = os.path.splitext(image_path)[1].lower()
        if file_extension == '.png':
            mime_type = "image/png"
            img_format = 'PNG'
        elif file_extension in ['.jpg', '.jpeg']:
            mime_type = "image/jpeg"
            img_format = 'JPEG'
        elif file_extension == '.webp':
            mime_type = "image/webp"
            img_format = 'WEBP'
        elif file_extension == '.gif':
            mime_type = "image/gif"
            img_format = 'GIF'
        else:
            # Default to jpeg for other formats
            mime_type = "image/jpeg"
            img_format = 'JPEG'
        
        # Convert image to bytes
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format=img_format)
        image_data = img_byte_arr.getvalue()
        
        # Get information about original points for steerable images
        original_points_info = get_original_points_info(image_path, category)
        
    
        # NOTE: Gemini uses a different coordinate system: [y, x] format and 0-1000 normalization
        # Check if category is counting - limit points accordingly
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
        
        # Prepare the content parts in the order of text first, then image
        prompt_parts = [
            prompt,
            {
                "mime_type": mime_type,
                "data": image_data
            }
        ]
        
        print(f"\nSending prompt to Gemini ({model_name}) with image {image_path}...")
        
        # Make the API call
        response = model.generate_content(prompt_parts)
        
        # Check if response parts exist and have content
        if response.parts:
            content = response.text
        elif hasattr(response, 'prompt_feedback') and response.prompt_feedback and hasattr(response.prompt_feedback, 'block_reason'):
            raise ValueError(f"Content blocked: {getattr(response.prompt_feedback, 'block_reason_message', '') or response.prompt_feedback.block_reason}")
        else:
            raise ValueError("No text content received from Gemini, or response was empty/unexpected.")
        
        print(f"\n[DEBUG] Raw Gemini output for {object_name} in {image_path}:")
        print(content)
        
        # Extract JSON from the response
        json_start = content.find('[')
        json_end = content.rfind(']') + 1
        if json_start != -1 and json_end != -1:
            json_str = content[json_start:json_end]
            print(f"[DEBUG] Extracted JSON string: {json_str}")
            
            # Parse the JSON
            raw_points = json.loads(json_str)
            
            # Convert from Gemini's format ([y, x] in 0-1000 range) to standard format ([x, y] in pixels)
            points = []
            for item in raw_points:
                if isinstance(item, dict) and "point" in item:
                    if isinstance(item["point"], list) and len(item["point"]) == 2:
                        # Gemini format: [y, x] normalized to 0-1000
                        # We need to: 1) swap coordinates and 2) convert to pixels
                        y, x = item["point"]
                        # Convert normalized coordinates (0-1000) to pixel coordinates
                        pixel_x = (x / 1000.0) * img_width
                        pixel_y = (y / 1000.0) * img_height
                        # Add to points list in standard format
                        points.append({"point": [pixel_x, pixel_y]})
            
            print(f"[DEBUG] Converted points: {points}")
            
            # If no valid points were found or conversion failed, try regex to extract coordinates
            if not points:
                import re
                # Look for patterns like [y, x] or [number, number]
                coords = re.findall(r'\[(\d+\.?\d*),\s*(\d+\.?\d*)\]', json_str)
                if coords:
                    print(f"[DEBUG] Coordinates extracted via regex: {coords}")
                    # First coordinate is y, second is x in Gemini's format
                    for y_str, x_str in coords:
                        try:
                            y, x = float(y_str), float(x_str)
                            # Convert normalized coordinates (0-1000) to pixel coordinates
                            pixel_x = (x / 1000.0) * img_width
                            pixel_y = (y / 1000.0) * img_height
                            points.append({"point": [pixel_x, pixel_y]})
                        except ValueError:
                            continue
                    print(f"[DEBUG] Points after regex extraction: {points}")
            
            # If not counting category and more than one point was returned, limit to first point
            if category != "counting" and len(points) > 1:
                points = [points[0]]
            
            return points
        else:
            return []
    except Exception as e:
        print(f"Error with {model_name} on {image_path}: {e}")
        import traceback
        traceback.print_exc()
        return []

def call_claude(image_path, object_name, model_name="claude-3-7-sonnet-20250219", category=None):
    """Call Claude to get points for the specified object."""
    try:
        # Read the image file as base64
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
        
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
        
        # Get information about original points for steerable images
        original_points_info = get_original_points_info(image_path, category)
        
        # Define system content 
        system_content = "You are a helpful assistant that can identify objects in images and provide their coordinates."
        
        # Check if category is counting - limit points accordingly
        if category == "counting":
            prompt = f"""
            {object_name}.
            The image dimensions are width={img_width}px, height={img_height}px.{original_points_info}
            The answer should follow the json format: [{{"point": <point>}}, ...]. 
            IMPORTANT: The points MUST be in [x, y] format where x is the horizontal position (left-to-right) and y is the vertical position (top-to-bottom) in PIXEL COORDINATES (not normalized).
            Example: For a point in the center of the image, return [width/2, height/2].
            """
        else:
            prompt = f"""
            {object_name}.
            The image dimensions are width={img_width}px, height={img_height}px.{original_points_info}
            The answer should follow the json format: [{{"point": <point>}}]. 
            IMPORTANT: Return EXACTLY ONE POINT. The point MUST be in [x, y] format where x is the horizontal position (left-to-right) and y is the vertical position (top-to-bottom) in PIXEL COORDINATES (not normalized).
            Example: For a point in the center of the image, return [width/2, height/2].
            """
        
        # Print complete prompt
        print_complete_prompt(system_content, prompt, model_name, image_path)
        
        # Call the Claude API
        response = anthropic_client.messages.create(
            model=model_name,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": mime_type,
                                "data": image_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": f"{system_content}\n\n{prompt}"
                        }
                    ],
                }
            ],
        )
        
        content = response.content[0].text
        # Extract JSON from the response
        json_start = content.find('[')
        json_end = content.rfind(']') + 1
        if json_start != -1 and json_end != -1:
            json_str = content[json_start:json_end]
            points = json.loads(json_str)
            # If not counting category and more than one point was returned, limit to first point
            if category != "counting" and len(points) > 1:
                points = [points[0]]
            return points
        else:
            return []
    except Exception as e:
        print(f"Error with {model_name} on {image_path}: {e}")
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
        
        # Get information about original points for steerable images
        original_points_info = get_original_points_info(image_path, category)
        
        # Define system content
        system_content = "You are a helpful assistant that can identify objects in images and provide their coordinates."
        
        # Check if category is counting - limit points accordingly
        if category == "counting":
            prompt = f"""
            {object_name}.
            The image dimensions are width={img_width}px, height={img_height}px.{original_points_info}
            The answer should follow the json format: [{{"point": <point>}}, ...]. 
            IMPORTANT: The points MUST be in [x, y] format where x is the horizontal position (left-to-right) and y is the vertical position (top-to-bottom) in PIXEL COORDINATES (not normalized).
            Example: For a point in the center of the image, return [width/2, height/2].
            """
        else:
            prompt = f"""
            {object_name}.
            The image dimensions are width={img_width}px, height={img_height}px.{original_points_info}
            The answer should follow the json format: [{{"point": <point>}}]. 
            IMPORTANT: Return EXACTLY ONE POINT. The point MUST be in [x, y] format where x is the horizontal position (left-to-right) and y is the vertical position (top-to-bottom) in PIXEL COORDINATES (not normalized).
            Example: For a point in the center of the image, return [width/2, height/2].
            """
        
        # Print complete prompt
        print_complete_prompt(system_content, prompt, model_name, image_path)
        
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
                        "text": f"{system_content}\n\n{prompt}",
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
            # If not counting category and more than one point was returned, limit to first point
            if category != "counting" and len(points) > 1:
                points = [points[0]]
            return points
        else:
            return []
    except Exception as e:
        print(f"Error with {model_name} on {image_path}: {e}")
        return []

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
        # Print the raw response for debugging
        print(f"[DEBUG] Looking for XML patterns in content: '{content[:200]}...'")
        
        # Try several patterns to match different possible XML formats
        xml_patterns = [
            # Format: <points x1="790" y1="46" alt="...">text</points> - with double or single quotes
            r'<points\s+x1=["\'"]?(\d+\.?\d*)["\'"]?\s+y1=["\'"]?(\d+\.?\d*)["\'"]?.*?>.*?</points>',
            # Format: <points x="123" y="456">text</points> - with double or single quotes
            r'<points\s+x=["\'"]?(\d+\.?\d*)["\'"]?\s+y=["\'"]?(\d+\.?\d*)["\'"]?.*?>.*?</points>',
            # Format: <points 123 456>text</points>
            r'<points\s+(\d+\.?\d*)\s+(\d+\.?\d*)>.*?</points>'
        ]
        
        points = []
        xml_matches_found = False
        
        # Try each pattern
        for pattern in xml_patterns:
            xml_matches = re.findall(pattern, content)
            if xml_matches:
                xml_matches_found = True
                print(f"[DEBUG] XML points format detected with pattern '{pattern}': {xml_matches}")
                
                # Convert to standard point format
                for match in xml_matches:
                    points.append({"point": [float(match[0]), float(match[1])]})
        
        if xml_matches_found:
            print(f"[DEBUG] Extracted points from XML: {points}")
            
            # If not counting category and more than one point was returned, limit to first point
            if category != "counting" and len(points) > 1:
                print(f"[DEBUG] Multiple points detected but not counting category. Limiting to first point.")
                points = [points[0]]
            
            return points
        
        # If no XML matches found, try simple pattern matching for coordinate pairs
        if not xml_matches_found:
            print("[DEBUG] No XML matches found, trying to extract any coordinate pairs")
            
            # Simple number pair extraction
            number_pairs = re.findall(r'(?:x|x1)[=:" ]+(\d+\.?\d*)[ ",]*(?:y|y1)[=:" ]+(\d+\.?\d*)', content)
            if number_pairs:
                print(f"[DEBUG] Found coordinate pairs from attribute-like text: {number_pairs}")
                # Convert to points
                for x, y in number_pairs:
                    points.append({"point": [float(x), float(y)]})
                
                print(f"[DEBUG] Extracted points: {points}")
                
                # If not counting category and more than one point was returned, limit to first point
                if category != "counting" and len(points) > 1:
                    points = [points[0]]
                
                return points
        
        # If we have points from any method, return them
        if points:
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

def call_llava(image_path, object_name, model_name="llava-hf/llava-onevision-qwen2-7b-ov-hf", category=None):
    """Call LLaVA-OneVision model to get points for the specified object."""
    try:
        # Initialize model and processor if not already done
        model, processor = initialize_llava(model_name)
        
        # Load the image
        image = Image.open(image_path)
        img_width, img_height = image.size
        print(f"[DEBUG] Image dimensions: {img_width}x{img_height}")
        
        # Get information about original points for steerable images
        original_points_info = get_original_points_info(image_path, category)
        
        # Define system content
        system_content = "You are a helpful assistant that can identify objects in images and provide their coordinates."
        
        # Prepare the prompt based on category
        if category == "counting":
            prompt = f"""
            {object_name}. 
            The image dimensions are width={img_width}px, height={img_height}px.{original_points_info}
            For each point, give EXACT PIXEL COORDINATES in [x, y] format, where x is horizontal (left-to-right) and y is vertical (top-to-bottom).
            Output format should be: [x, y], [x, y], etc. for multiple points.
            ONLY return the coordinates with no additional text or explanations.
            """
        else:
            prompt = f"""
            {object_name}.
            The image dimensions are width={img_width}px, height={img_height}px.{original_points_info}
            Give EXACT PIXEL COORDINATES in [x, y] format, where x is horizontal (left-to-right) and y is vertical (top-to-bottom).
            ONLY return the coordinates with no additional text or explanations.
            """
        
        # Print complete prompt
        print_complete_prompt(system_content, prompt, model_name, image_path)
        
        # Format the prompt correctly for LLaVA-OneVision HF version
        # Use the chat template approach from the HF model card
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt}
            ]}
        ]
        
        # Use the processor's apply_chat_template method
        print("[DEBUG] Applying chat template")
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        
        # Process inputs with the processor
        print("[DEBUG] Processing inputs")
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
        
        # Generate output
        print("[DEBUG] Generating output")
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False
            )
        
        # Decode the output
        print("[DEBUG] Decoding output")
        content = processor.decode(output_ids[0][2:], skip_special_tokens=True)
        
        print(f"\n[DEBUG] Raw LLaVA output for {object_name} in {image_path}:")
        print(content)
        
        # Use a robust approach to extract coordinates from the response
        # First, try to extract using regex for [x, y] pattern
        import re
        
        # Look for coordinate pairs like [x, y]
        coord_pattern = r'\[(\d+\.?\d*),\s*(\d+\.?\d*)\]'
        coords = re.findall(coord_pattern, content)
        
        if coords:
            print(f"[DEBUG] Coordinates extracted via regex: {coords}")
            
            # Convert to standard point format
            points = [{"point": [float(x), float(y)]} for x, y in coords]
            print(f"[DEBUG] Points after extraction: {points}")
            
            # If not counting category and more than one point was returned, limit to first point
            if category != "counting" and len(points) > 1:
                print(f"[DEBUG] Multiple points detected but not counting category. Limiting to first point.")
                points = [points[0]]
                print(f"[DEBUG] Final point: {points}")
            
            return points
        else:
            # No coordinate pattern found, try other patterns
            
            # Look for numbers that might be coordinates (fallback)
            number_pairs = re.findall(r'(\d+\.?\d*)\s*,\s*(\d+\.?\d*)', content)
            if number_pairs:
                print(f"[DEBUG] Found potential coordinate pairs: {number_pairs}")
                # Convert each pair to points
                points = [{"point": [float(x), float(y)]} for x, y in number_pairs]
                
                # Limit to first point if not counting
                if category != "counting" and len(points) > 1:
                    points = [points[0]]
                
                return points
            
            # Look for individual numbers as last resort
            numbers = re.findall(r'\b(\d+\.?\d*)\b', content)
            if len(numbers) >= 2:
                print(f"[DEBUG] Found individual numbers: {numbers}")
                # Try to pair them up as x,y coordinates
                points = []
                for i in range(0, len(numbers)-1, 2):
                    x, y = float(numbers[i]), float(numbers[i+1])
                    points.append({"point": [x, y]})
                
                # Limit to first point if not counting
                if category != "counting" and len(points) > 1:
                    points = [points[0]]
                
                return points
            
            print("[DEBUG] No valid coordinates found in response")
            return []
            
    except Exception as e:
        print(f"Error calling {model_name} on {image_path}: {e}")
        print(f"Exception details: {str(e)}")
        import traceback
        traceback.print_exc()
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
        with open("where2place/point_questions.jsonl", "r") as f:
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
    results_file = f"results_{model_type}_{model_name.replace('/', '_')}_other_dataset_experiment.json"
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
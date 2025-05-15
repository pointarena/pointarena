import os
import json
import random
import time
import datetime
import uuid
import csv
import re  # Add explicit import for re
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
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
import math

# Import SAM helper
from segment_utils import SegmentAnythingHelper

# Load environment variables
load_dotenv()

# Configure API keys and clients
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Constants
IMAGES_DIR = Path("images")
# Ensure image directory exists
IMAGES_DIR.mkdir(exist_ok=True)

# Create masks directory for saving mask images
MASKS_DIR = Path("masks")
MASKS_DIR.mkdir(exist_ok=True)

# Define the 5 subfolders
IMAGE_CATEGORIES = ["affordable", "counting", "spatial", "reasoning", "steerable"]

# Grid size for image annotation (increased from 30 to 50 for smaller grid cells)
GRID_SIZE = 50

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

# Use local models
USE_LOCAL_MODELS = True
# If using local models, use the specified directory; otherwise use remote models
if USE_LOCAL_MODELS:
    SAVED_MODELS_DIR = Path(os.getenv("SAVED_MODELS_DIR"))
    # Ensure the model saving directory exists
    SAVED_MODELS_DIR.mkdir(exist_ok=True, parents=True)
else:
    # If not using local models, will use Hugging Face's remote models
    SAVED_MODELS_DIR = None

# Available models
OPENAI_MODELS = ["gpt-4o", "gpt-4o-mini",  "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano"] # o1 is not supported yet
GEMINI_MODELS = ["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-1.5-flash", "gemini-1.5-flash-8b", "gemini-1.5-pro"]
# Molmo models
MOLMO_MODELS = ["Molmo-7B-D-0924", "Molmo-7B-O-0924", "Molmo-7B-O-0924"]
# Qwen models
QWEN_MODELS = ["Qwen2.5-VL-7B-Instruct"]
# LLaVA models
LLAVA_MODELS = ["llava-onevision-qwen2-7b-ov-hf"]
# UI display names
UI_MODEL_NAMES = ["Model A", "Model B", "Model C"]
POINT_COLORS = ["red", "blue", "yellow"]  # Colors for the points from different models

# Initialize Molmo model and processor (lazy loading - will be initialized when first used)
molmo_model = None
molmo_processor = None

# Initialize Qwen model and processor (lazy loading - will be initialized when first used)
qwen_model = None
qwen_processor = None

# Initialize LLaVA model and processor (lazy loading - will be initialized when first used)
llava_model = None
llava_processor = None

# Initialize SAM helper
try:
    sam_helper = SegmentAnythingHelper()
    SAM_ENABLED = True
    print("SAM helper initialized successfully")
except Exception as e:
    print(f"Failed to initialize SAM helper: {e}")
    print("SAM functionality will be disabled")
    SAM_ENABLED = False

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
                device_map='auto',
                use_fast=True  # Add this parameter
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
                device_map='auto',
                use_fast=True  # Add this parameter
            )
            
            # Load model from remote
            molmo_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype='auto',
                device_map='auto'
            )
        
    return molmo_model, molmo_processor

def get_random_image():
    """Get a random image from one of the subfolders in the images directory and return appropriate prompt instructions."""
    # Check if all category folders exist
    category_folders = []
    for category in IMAGE_CATEGORIES:
        folder_path = IMAGES_DIR / category
        if folder_path.exists() and folder_path.is_dir():
            category_folders.append(category)
    
    if not category_folders:
        raise ValueError("No category folders found in the images directory.")
    
    # Check data.json file to see which images have already been annotated
    data_file = Path("data.json")
    already_annotated = set()
    
    if data_file.exists():
        try:
            with open(data_file, "r") as f:
                annotation_data = json.load(f)
                for item in annotation_data:
                    if "image_filename" in item:
                        already_annotated.add(item["image_filename"])
            print(f"Found {len(already_annotated)} already annotated images in data.json")
        except Exception as e:
            print(f"Error reading data.json: {e}")
            already_annotated = set()
    
    # Try to find an unannotated image
    max_attempts = 10
    for attempt in range(max_attempts):
        # Randomly select a category
        selected_category = random.choice(category_folders)
        category_path = IMAGES_DIR / selected_category
        
        # Get all image files from the selected category
        image_files = list(category_path.glob("*.jpg")) + list(category_path.glob("*.png")) + list(category_path.glob("*.jpeg")) + list(category_path.glob("*.webp")) + list(category_path.glob("*.avif"))
        if not image_files:
            continue  # Try another category if this one has no images
        
        # Filter out already annotated images
        available_images = [img for img in image_files if img.name not in already_annotated]
        
        if available_images:
            # Select a random image from available (not annotated) images
            image_path = random.choice(available_images)
            break
        elif attempt == max_attempts - 1:
            # If we've tried all categories and can't find unannotated images, use any image
            print("Warning: Could not find unannotated images after multiple attempts. Using an already annotated image.")
            image_path = random.choice(image_files)
            break
    else:
        # This will run if we exhaust all attempts without finding an image
        raise ValueError("No suitable images found after multiple attempts.")
    
    # Generate prompt instructions based on the category
    prompt_instructions = ""
    if selected_category == "affordable":
        prompt_instructions = "For this image, create a question about USING TOOLS. For example, if you see a pen, ask 'Point to the tool that people can use to write.'"
    elif selected_category == "counting":
        prompt_instructions = "For this image, create a question about COUNTING OBJECTS. For example, if you see many cars, ask 'Point to all the cars.'"
    elif selected_category == "spatial":
        prompt_instructions = "For this image, create a question about SPATIAL RELATIONSHIPS. For example, if you see two bottles, ask 'Point to the object to the right of the left bottle.'"
    elif selected_category == "reasoning":
        prompt_instructions = "For this image, create a question about REASONING. For example, if you see a car, ask 'Point to the moving direction of the car.'"
    elif selected_category == "steerable":
        prompt_instructions = "For this image, create a question RELATED TO THE BLUE POINT already marked on the image. For example, if there's a cat with a blue point on its nose, ask 'Point to the left of the existing point on the image.'"
    
    # If the image is from the steerable category, check if we have points for it in our mapping
    displayed_image = None
    if selected_category == "steerable" and image_path.name in IMAGE_POINTS_MAP:
        # Transform points from the CSV format to the format used by draw_points_on_image
        points_from_csv = []
        for point_dict in IMAGE_POINTS_MAP[image_path.name]:
            # Create a point in the format expected by draw_points_on_image
            # CSV has x,y in 0-100 range, we need to convert to actual pixel coordinates
            img = Image.open(image_path)
            img_width, img_height = img.size
            points_from_csv.append({
                "point": [float(point_dict["x"]) * img_width / 100, float(point_dict["y"]) * img_height / 100]
            })
        
        # Draw the points on the image with blue color
        if points_from_csv:
            displayed_image = draw_points_on_image(str(image_path), points_from_csv, "blue")
            print(f"Added {len(points_from_csv)} blue points to steerable image {image_path.name}")
    
    return str(image_path), image_path.name, selected_category, prompt_instructions, displayed_image

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
    elif file_extension == '.avif':
        mime_type = "image/avif"
    else:
        # Default to jpeg for other formats
        mime_type = "image/jpeg"
    
    # Check if category is counting - limit points accordingly
    if category == "counting":
        prompt = f"""
        Point to {object_name}.
        The image dimensions are width={img_width}px, height={img_height}px.
        The answer should follow the json format: [{{"point": <point>}}, ...]. 
        IMPORTANT: The points MUST be in [x, y] format where x is the horizontal position (left-to-right) and y is the vertical position (top-to-bottom) in PIXEL COORDINATES (not normalized).
        Example: For a point in the center of the image, return [width/2, height/2].
        """
    else:
        prompt = f"""
        Point to {object_name}.
        The image dimensions are width={img_width}px, height={img_height}px.
        The answer should follow the json format: [{{"point": <point>}}]. 
        IMPORTANT: Return EXACTLY ONE POINT. The point MUST be in [x, y] format where x is the horizontal position (left-to-right) and y is the vertical position (top-to-bottom) in PIXEL COORDINATES (not normalized).
        Example: For a point in the center of the image, return [width/2, height/2].
        """
    
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that can identify objects in images and provide their coordinates."},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}}
            ]}
        ],
        # max_tokens=300
    )
    
    try:
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
        print(f"Error parsing {model_name} response: {e}")
        return []

def call_gemini(image_path, object_name, model_name="gemini-2.0-flash", category=None):
    """Call Gemini to get points for the specified object."""
    model = genai.GenerativeModel(model_name)
    
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
    
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
    elif file_extension == '.avif':
        mime_type = "image/avif"
    else:
        # Default to jpeg for other formats
        mime_type = "image/jpeg"
    
    # Check if category is counting - limit points accordingly
    if category == "counting":
        prompt = f"""
        Point to {object_name}.
        The image dimensions are width={img_width}px, height={img_height}px.
        The answer should follow the json format: [{{"point": <point>}}, ...]. 
        IMPORTANT: The points MUST be in [x, y] format where x is the horizontal position (left-to-right) and y is the vertical position (top-to-bottom) in PIXEL COORDINATES (not normalized).
        Example: For a point in the center of the image, return [width/2, height/2].
        """
    else:
        prompt = f"""
        Point to {object_name}.
        The image dimensions are width={img_width}px, height={img_height}px.
        The answer should follow the json format: [{{"point": <point>}}]. 
        IMPORTANT: Return EXACTLY ONE POINT. The point MUST be in [x, y] format where x is the horizontal position (left-to-right) and y is the vertical position (top-to-bottom) in PIXEL COORDINATES (not normalized).
        Example: For a point in the center of the image, return [width/2, height/2].
        """
    
    response = model.generate_content([prompt, {"mime_type": mime_type, "data": image_data}])
    
    try:
        content = response.text
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
        print(f"Error parsing {model_name} response: {e}")
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

def call_molmo(image_path, object_name, model_name="allenai/Molmo-7B-D-0924", category=None):
    """Call Molmo model to get points for the specified object."""
    try:
        # Initialize model and processor if not already done
        model, processor = initialize_molmo(model_name)
        
        # Load the image
        image = Image.open(image_path)
        img_width, img_height = image.size
        
        # Prepare the prompt based on category
        if category == "counting":
            prompt = f"""
            Point to {object_name}.
            The image dimensions are width={img_width}px, height={img_height}px.
            The answer can use either coordinates like (45.6, 78.9) or Click(45.6, 78.9) where numbers are percentages (0-100).
            You can also use x="45.6" y="78.9" format or p=456,789 format (where numbers are divided by 10 to get percentages).
            IMPORTANT: You can return multiple points if needed.
            """
        else:
            prompt = f"""
            Point to {object_name}.
            The image dimensions are width={img_width}px, height={img_height}px.
            The answer can use either coordinates like (45.6, 78.9) or Click(45.6, 78.9) where numbers are percentages (0-100).
            You can also use x="45.6" y="78.9" format or p=456,789 format (where numbers are divided by 10 to get percentages).
            IMPORTANT: Return EXACTLY ONE POINT.
            """
        
        print(f"Processing image and prompt: {object_name}")
        
        # Process the image and text
        inputs = processor.process(
            images=[image],
            text=prompt
        )
        
        # Move inputs to the correct device and make a batch of size 1
        inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}
        
        print(f"Generating output...")
        
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
        
        print(f"Model output: {content[:100]}...")  # Only print the first 100 characters
        
        # First try to extract points using various patterns
        extracted_points = extract_points(content, img_width, img_height)
        
        if extracted_points:
            # Convert to the standard format expected by the app
            points = [{"point": [float(p[0]), float(p[1])]} for p in extracted_points]
            print(f"Extracted {len(points)} points using extract_points function")
            
            # If not counting category and more than one point was returned, limit to first point
            if category != "counting" and len(points) > 1:
                points = [points[0]]
            
            return points
        
        # If no points found with extract_points, try the original methods
        # Extract JSON from the response
        json_start = content.find('[')
        json_end = content.rfind(']') + 1
        if json_start != -1 and json_end != -1:
            json_str = content[json_start:json_end]
            
            # Try to extract coordinates using regex
            coords = re.findall(r'\[(\d+\.?\d*),\s*(\d+\.?\d*)\]', json_str)
            if coords:
                # Use x, y format directly since we're now using pixel coordinates
                points = [{"point": [float(x), float(y)]} for x, y in coords]
                print(f"Extracted {len(points)} points using original regex")
                
                # If not counting category and more than one point was returned, limit to first point
                if category != "counting" and len(points) > 1:
                    points = [points[0]]
                
                return points
            
            # If regex fails, try to parse as JSON
            try:
                # Try to fix common JSON format errors
                try:
                    raw_points = json.loads(json_str)
                    
                    # Handle different possible formats
                    points = []
                    if isinstance(raw_points, list):
                        for item in raw_points:
                            if isinstance(item, list) and len(item) == 2:
                                # Direct [x, y] format
                                x, y = item
                                # Use direct x, y coordinate
                                points.append({"point": [float(x), float(y)]})
                            elif isinstance(item, dict) and "point" in item:
                                # {"point": [x, y]} format
                                if isinstance(item["point"], list) and len(item["point"]) == 2:
                                    x, y = item["point"]
                                    # Use direct x, y coordinate
                                    points.append({"point": [float(x), float(y)]})
                    
                    if points:
                        # If not counting category and more than one point was returned, limit to first point
                        if category != "counting" and len(points) > 1:
                            points = [points[0]]
                        return points
                    
                except json.JSONDecodeError as e:
                    print(f"JSON parsing error: {e}. Attempting to fix format...")
                    
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
                print(f"Error parsing coordinates: {e}")
                return []
        else:
            print(f"No JSON found in response")
            return []
    except Exception as e:
        print(f"Error calling Molmo: {e}")
        return []

def draw_molmo_points_on_image(image_path, points, color="red"):
    """Draw points specifically for Molmo model outputs with enhanced visualization.
    
    Args:
        image_path: Path to the image file
        points: List of points in the format [{"point": [x, y]}, ...]
        color: Color to use for drawing points
        
    Returns:
        PIL Image with points drawn on it
    """
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    
    img_width, img_height = img.size
    
    for point_data in points:
        if "point" in point_data:
            # Points are in [x, y] format in pixel coordinates
            x, y = point_data["point"]
            
            # Draw a more visible marker for Molmo points
            radius = max(7, min(img_width, img_height) // 80)  # Slightly larger radius for better visibility
            
            # Draw a filled circle with outline for better visibility
            draw.ellipse(
                [(x - radius, y - radius), (x + radius, y + radius)],
                fill=color,
                outline="white"  # Add white outline for better contrast
            )
            
            # Add crosshair for precision
            line_length = radius * 1.5
            draw.line([(x - line_length, y), (x + line_length, y)], fill="white", width=2)
            draw.line([(x, y - line_length), (x, y + line_length)], fill="white", width=2)
    
    return img

def draw_points_on_image(image_path, points, color="red"):
    """Draw points on the image using pixel coordinates directly."""
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    
    img_width, img_height = img.size
    
    for point_data in points:
        if "point" in point_data:
            # Points are in [x, y] format in pixel coordinates
            x, y = point_data["point"]
            
            # Draw a circle at the point
            radius = max(5, min(img_width, img_height) // 100)  # Adaptive radius based on image size
            draw.ellipse(
                [(x - radius, y - radius), (x + radius, y + radius)],
                fill=color,
                outline=color
            )
    
    return img

def draw_grid_on_image(image, grid_size=GRID_SIZE, line_color="lightgray", line_width=1):
    """Draw a grid on the image with square cells regardless of image aspect ratio."""
    img_width, img_height = image.size
    draw = ImageDraw.Draw(image)
    
    # Calculate cell size based on the smaller dimension to ensure square cells
    cell_size = min(img_width, img_height) / grid_size
    
    # Calculate actual grid dimensions (rows and columns)
    num_cols = math.ceil(img_width / cell_size)  # Use math.ceil to ensure we cover the entire image
    num_rows = math.ceil(img_height / cell_size)  # Use math.ceil to ensure we cover the entire image
    
    # Draw horizontal lines
    for i in range(num_rows + 1):
        y = i * cell_size
        if y <= img_height:  # Only draw if still within image bounds
            draw.line([(0, y), (img_width, y)], fill=line_color, width=line_width)
    
    # Draw vertical lines
    for i in range(num_cols + 1):
        x = i * cell_size
        if x <= img_width:  # Only draw if still within image bounds
            draw.line([(x, 0), (x, img_height)], fill=line_color, width=line_width)
    
    return image

def points_to_grid_cells(points, grid_size=GRID_SIZE, image_path=None):
    """Convert pixel coordinate points to grid cell indices."""
    grid_cells = set()
    
    # If no points, return empty list
    if not points:
        return []
    
    # If no image path provided, try to extract from the first point
    if image_path is None and "image_path" in points[0]:
        image_path = points[0]["image_path"]
    
    # If still no image path, we can't calculate grid cells
    if image_path is None:
        print("Warning: No image path provided for grid cell calculation")
        return []
    
    try:
        # Open the image to get dimensions
        img = Image.open(image_path)
        img_width, img_height = img.size
        
        # Calculate cell size based on the image dimensions
        cell_size_x = img_width / grid_size
        cell_size_y = img_height / grid_size
        
        for point_data in points:
            if "point" in point_data:
                # Points are in [x, y] format in pixel coordinates
                x, y = point_data["point"]
                
                # Convert to integers for computation
                x, y = int(x), int(y)
                
                # Calculate grid cell indices
                grid_col = int(x / cell_size_x)
                grid_row = int(y / cell_size_y)
                
                # Ensure the grid coordinates are within bounds
                grid_row = max(0, min(grid_size - 1, grid_row))
                grid_col = max(0, min(grid_size - 1, grid_col))
                
                # Add the grid cell to the set
                grid_cells.add((grid_row, grid_col))
                
    except Exception as e:
        print(f"Error calculating grid cells: {e}")
    
    return list(grid_cells)

def draw_selected_grid_cells(image, grid_cells, grid_size=GRID_SIZE, fill_color="yellow", alpha=0.3):
    """Draw selected grid cells on the image with transparency, ensuring square cells."""
    img_width, img_height = image.size
    
    # Create a transparent overlay
    overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    # Calculate the size of a single square cell based on the smaller dimension
    cell_size = min(img_width, img_height) / grid_size
    
    # Calculate maximum row and column indices for this image
    num_rows = math.ceil(img_height / cell_size)
    num_cols = math.ceil(img_width / cell_size)
    
    # Convert color name to RGB tuple
    if fill_color == "yellow":
        rgb_color = (255, 255, 0)
    elif fill_color == "red":
        rgb_color = (255, 0, 0)
    elif fill_color == "green":
        rgb_color = (0, 255, 0)
    elif fill_color == "blue":
        rgb_color = (0, 0, 255)
    else:
        rgb_color = (255, 255, 0)  # Default yellow
    
    # Draw each selected grid cell
    for row, col in grid_cells:
        # Skip cells outside the image bounds
        if row >= num_rows or col >= num_cols or row < 0 or col < 0:
            continue
            
        x0 = col * cell_size
        y0 = row * cell_size
        x1 = min((col + 1) * cell_size, img_width)  # Ensure we don't exceed image boundaries
        y1 = min((row + 1) * cell_size, img_height) # Ensure we don't exceed image boundaries
        
        # Draw a semi-transparent rectangle
        rgba_color = rgb_color + (int(alpha * 255),)
        draw.rectangle([x0, y0, x1, y1], fill=rgba_color)
        
        # Draw a small marker at the center of the cell
        center_x = (x0 + x1) / 2
        center_y = (y0 + y1) / 2
        marker_size = max(2, cell_size / 10)  # Adaptive marker size
        draw.ellipse(
            [(center_x - marker_size, center_y - marker_size), 
             (center_x + marker_size, center_y + marker_size)],
            fill=rgb_color + (255,)  # Fully opaque marker
        )
    
    # Composite the overlay onto the original image
    result = Image.alpha_composite(image.convert('RGBA'), overlay)
    
    return result.convert('RGB')  # Convert back to RGB for compatibility


def save_to_json_file(annotation_data):
    """Save annotation data to local JSON file."""
    data_file = Path("data.json")
    
    # Load existing data if file exists
    if data_file.exists():
        try:
            with open(data_file, "r") as f:
                existing_data = json.load(f)
        except json.JSONDecodeError:
            # If file exists but is invalid JSON, start with empty list
            existing_data = []
    else:
        existing_data = []
    
    # Add timestamp to annotation data
    annotation_data["timestamp"] = datetime.datetime.now().isoformat()
    
    # Append new data
    existing_data.append(annotation_data)
    
    # Save back to file
    with open(data_file, "w") as f:
        json.dump(existing_data, f, indent=2)
    
    print(f"Saved annotation data to {data_file}")

def save_mask_image(mask, original_image_path):
    """
    Save a binary mask as a PNG image.
    
    Args:
        mask: Binary numpy array mask (True/False values)
        original_image_path: Path to the original image to derive the mask filename
    
    Returns:
        mask_filename: Name of the saved mask file
    """
    # Get original image filename without extension
    original_filename = Path(original_image_path).stem
    
    # Create mask filename
    mask_filename = f"{original_filename}_mask.png"
    mask_path = MASKS_DIR / mask_filename
    
    # Create pure black and white mask image using SAM helper
    mask_img = sam_helper.create_binary_mask(mask)
    
    # Save mask
    mask_img.save(mask_path)
    
    print(f"Saved mask to {mask_path}")
    return mask_filename

def process_test(object_name, image_path, image_filename, category=None):
    """Process the test with randomly selected models and return the results."""
    # Create a list of all available models
    all_models = []
    
    # # Add all OpenAI models
    # for model_name in OPENAI_MODELS:
    #     all_models.append((model_name, call_openai))
    
    # # Add all Gemini models
    # for model_name in GEMINI_MODELS:
    #     all_models.append((model_name, call_gemini))
    
    # Add all Molmo models
    for model_name in MOLMO_MODELS:
        # For Molmo models, we need to add the complete path prefix
        full_model_name = f"allenai/{model_name}" if not model_name.startswith("allenai/") else model_name
        all_models.append((full_model_name, call_molmo))
    
    # Randomly select three models for comparison (without category restriction)
    selected_models = random.sample(all_models, 3)
    
    # Assign the selected models to Model A, Model B, and Model C
    model1_name, model1_func = selected_models[0]
    model2_name, model2_func = selected_models[1]
    model3_name, model3_func = selected_models[2]
    
    print(f"Selected models: Model A = {model1_name}, Model B = {model2_name}, Model C = {model3_name}")
    
    # Call all three models with their respective functions, passing the category parameter
    model1_points = model1_func(image_path, object_name, model1_name, category)
    model2_points = model2_func(image_path, object_name, model2_name, category)
    model3_points = model3_func(image_path, object_name, model3_name, category)
    
    # Convert points to grid cells, passing the image_path
    model1_grid_cells = points_to_grid_cells(model1_points, image_path=image_path)
    model2_grid_cells = points_to_grid_cells(model2_points, image_path=image_path)
    model3_grid_cells = points_to_grid_cells(model3_points, image_path=image_path)
    
    # Draw points on images - use special drawing function for Molmo models
    # Check if model1 is a Molmo model
    if model1_name.startswith("allenai/") or model1_name in MOLMO_MODELS:
        model1_image = draw_molmo_points_on_image(image_path, model1_points, POINT_COLORS[0])
    else:
        model1_image = draw_points_on_image(image_path, model1_points, POINT_COLORS[0])
    
    # Check if model2 is a Molmo model
    if model2_name.startswith("allenai/") or model2_name in MOLMO_MODELS:
        model2_image = draw_molmo_points_on_image(image_path, model2_points, POINT_COLORS[1])
    else:
        model2_image = draw_points_on_image(image_path, model2_points, POINT_COLORS[1])
    
    # Check if model3 is a Molmo model
    if model3_name.startswith("allenai/") or model3_name in MOLMO_MODELS:
        model3_image = draw_molmo_points_on_image(image_path, model3_points, POINT_COLORS[2])
    else:
        model3_image = draw_points_on_image(image_path, model3_points, POINT_COLORS[2])
    
    # Prepare test data
    test_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "image_filename": image_filename,
        "object_name": object_name,
        "image_path": image_path,
        "category": category,
        "model1_name": model1_name,
        "model2_name": model2_name,
        "model3_name": model3_name,
        "model1_grid_cells": model1_grid_cells,
        "model2_grid_cells": model2_grid_cells,
        "model3_grid_cells": model3_grid_cells,
    }
    
    return model1_image, model2_image, model3_image, test_data


def ui_components():
    """Create and return the Gradio UI components."""
    with gr.Blocks(title="Image Annotation Tool", theme=gr.themes.Soft()) as app:
        gr.Markdown("# Image Annotation Tool")
        gr.Markdown("### Evaluate model annotations and create gold standards")
        
        # State variables
        current_image_path = gr.State("")
        current_image_filename = gr.State("")
        current_category = gr.State("")
        current_prompt_instructions = gr.State("")
        current_test_data = gr.State(None)
        accepted_models = gr.State([False, False, False])  # Track which models are accepted
        gold_standard_cells = gr.State([])  # Store the gold standard grid cells
        all_rejected = gr.State(False)  # Track if all models have been rejected
        cell_count = gr.State(0)  # Track the count of manually selected grid cells for counting category
        
        # Main container (will hold the content of both tabs)
        with gr.Group() as main_container:
            # Image display and input (common to both tabs)
            with gr.Row():
                # Left column for image and input
                with gr.Column(scale=1):
                    # Image display
                    image_display = gr.Image(label="Test Image", type="filepath")
                    
                    # Category info and prompt instructions
                    category_display = gr.Markdown("")
                    prompt_instructions_display = gr.Markdown("")
                    
                    # User input
                    user_input = gr.Textbox(placeholder="Enter what you want to annotate", label="I want to point to... (e.g. 'the center of the image')")
                    
                    # Submit button
                    submit_btn = gr.Button("Submit", variant="primary")
                    
                    # New test button
                    new_test_btn = gr.Button("New Test", variant="secondary")
                    
                    # Status message
                    status_msg = gr.Markdown("")
        
        # Create tabs container (moved to bottom)
        with gr.Tabs() as tabs:
            # Model Evaluation tab
            with gr.Tab("Model Evaluation") as model_eval_tab:
                # Model outputs in a horizontal layout
                with gr.Row():
                    # Model A
                    with gr.Column():
                        model1_output = gr.Image(label=UI_MODEL_NAMES[0], type="pil", elem_id="model1_img", show_download_button=True, height=300)
                        model1_accept_btn = gr.Button("✓ Accept", variant="secondary")
                    
                    # Model B
                    with gr.Column():
                        model2_output = gr.Image(label=UI_MODEL_NAMES[1], type="pil", elem_id="model2_img", show_download_button=True, height=300)
                        model2_accept_btn = gr.Button("✓ Accept", variant="secondary")
                    
                    # Model C
                    with gr.Column():
                        model3_output = gr.Image(label=UI_MODEL_NAMES[2], type="pil", elem_id="model3_img", show_download_button=True, height=300)
                        model3_accept_btn = gr.Button("✓ Accept", variant="secondary")
                
                # Selection result
                selection_result = gr.Markdown("")
                manual_instruction = gr.Markdown("*If 2 or more models provide satisfactory annotations, this means the question is too easy. Please load a new test. If 0 or 1 model provides a satisfactory annotation, please switch to the Manual Annotation tab to create your own annotation.*")
            
            # Manual annotation tab
            with gr.Tab("Manual Annotation", interactive=True) as manual_tab:
                with gr.Row():
                    with gr.Column():
                        # State variables
                        selected_grid_cells = gr.State([])
                        current_mask = gr.State(None)  # To store current mask from SAM
                        mask_applied = gr.State(False)  # New state to track if a mask has been applied
                        
                        # Display image with grid, using regular image component instead of selection tool
                        manual_image = gr.Image(label="Click on image area to annotate", type="pil", elem_id="manual_image", interactive=True, height=500)
                        
                        # Display currently selected grid cells
                        selected_cells_display = gr.Markdown("No areas selected")
                        
                        # Add SAM-related UI elements
                        with gr.Row(visible=SAM_ENABLED) as sam_ui_row:
                            with gr.Column(scale=2):
                                sam_mask_image = gr.Image(label="SAM Generated Mask", type="pil", show_download_button=True, height=300)
                            
                            with gr.Column(scale=1):
                                # Add Start Mask button
                                start_mask_btn = gr.Button("▶ Start Mask Generation", variant="primary")
                                
                                # Acceptance/rejection buttons for SAM
                                with gr.Row():
                                    accept_mask_btn = gr.Button("✓ Accept Mask", variant="primary")
                                    reject_mask_btn = gr.Button("✗ Reject Mask", variant="secondary")
                                
                                # Status display for SAM
                                sam_status = gr.Markdown("")
                        
                        # Common buttons
                        with gr.Row():
                            clear_selection_btn = gr.Button("Clear Selection", variant="secondary")
                            load_manual_btn = gr.Button("Load Current Image for Manual Annotation", variant="primary")
                        
                        # Finish button 
                        finish_btn = gr.Button("Finish Annotation", variant="primary")
                        
                        # Status message
                        manual_status = gr.Markdown("")
                        
                        # Add HTML component for page refresh
                        refresh_html = gr.HTML(visible=True)
        
        # Load a random image on page load and when new test button is clicked
        def load_random_image():
            try:
                image_path, image_filename, selected_category, prompt_instructions, displayed_image = get_random_image()
                # Reset accepted models state and all_rejected state
                # Reset image container IDs to remove any "model-accepted" class
                model1_img_reset = gr.update(elem_id="model1_img")
                model2_img_reset = gr.update(elem_id="model2_img")
                model3_img_reset = gr.update(elem_id="model3_img")
                
                # Format category display and prompt instructions
                category_md = f"**Category: {selected_category.capitalize()}**"
                
                # If we have a pre-annotated image (for steerable category), use it
                # Otherwise use the original image path
                image_display = displayed_image if displayed_image is not None else image_path
                
                return (
                    image_path,  # current_image_path
                    image_display,  # image_display
                    image_filename,  # current_image_filename
                    selected_category,  # current_category
                    prompt_instructions,  # current_prompt_instructions
                    category_md,  # category_display
                    prompt_instructions,  # prompt_instructions_display
                    model1_img_reset,  # model1_output
                    model2_img_reset,  # model2_output
                    model3_img_reset,  # model3_output
                    None,  # current_test_data
                    [False, False, False],  # accepted_models
                    [],  # gold_standard_cells
                    False,  # all_rejected
                    "",  # status_msg
                    gr.update(visible=True, interactive=True),  # manual_tab - restore visibility AND interactivity
                )
            except ValueError as e:
                # Reset image container IDs
                model1_img_reset = gr.update(elem_id="model1_img")
                model2_img_reset = gr.update(elem_id="model2_img")
                model3_img_reset = gr.update(elem_id="model3_img")
                
                return None, None, "", "", "", "", "", model1_img_reset, model2_img_reset, model3_img_reset, None, [False, False, False], [], False, str(e), gr.update(visible=True, interactive=True)
        
        # Process the test when submit button is clicked
        def on_submit(user_input, image_path, image_filename, category):
            if not user_input.strip():
                return None, None, None, None, [False, False, False], [], False, "Please enter annotation content", gr.update(visible=True, interactive=True)
            
            try:
                model1_img, model2_img, model3_img, test_data = process_test(user_input, image_path, image_filename, category)
                # Category is already included in test_data within process_test now
                return (
                    model1_img,  # model1_output
                    model2_img,  # model2_output
                    model3_img,  # model3_output
                    test_data,  # current_test_data
                    [False, False, False],  # accepted_models
                    [],  # gold_standard_cells
                    False,  # all_rejected
                    "Processing complete. Please evaluate each model's annotation.",  # status_msg
                    gr.update(visible=True, interactive=True),  # manual_tab
                )
            except Exception as e:
                return None, None, None, None, [False, False, False], [], False, f"Error: {str(e)}", gr.update(visible=True, interactive=True)
        
        # Handle model acceptance/rejection
        def on_model_accept(model_index, accepted_models, test_data, gold_standard_cells, cell_count):
            if test_data is None:
                return accepted_models, gold_standard_cells, False, "Please submit a test first", gr.update(visible=True, interactive=True), None, None, None, None, None, None
            
            # Update accepted models state
            new_accepted_models = accepted_models.copy()
            new_accepted_models[model_index] = True
            
            # Update gold standard cells if this is the first accepted model
            new_gold_standard_cells = gold_standard_cells
            if not any(accepted_models) and new_accepted_models[model_index]:
                # This is the first accepted model, use its grid cells as gold standard
                model_key = f"model{model_index+1}_grid_cells"
                if model_key in test_data:
                    new_gold_standard_cells = test_data[model_key]
                    
                    # If this is a counting category, update the cell count
                    if test_data.get("category", "") == "counting":
                        # Count the number of grid cells as the count
                        new_cell_count = len(new_gold_standard_cells)
                        print(f"Setting cell count to {new_cell_count} for counting category")
                    else:
                        new_cell_count = cell_count
            else:
                new_cell_count = cell_count
            
            # Prepare visual updates for all models (buttons)
            model1_btn_update = gr.update(value="✓ Accepted" if new_accepted_models[0] else "✓ Accept", 
                                          variant="success" if model_index == 0 else "secondary", 
                                          interactive=not new_accepted_models[0])
            model2_btn_update = gr.update(value="✓ Accepted" if new_accepted_models[1] else "✓ Accept", 
                                          variant="success" if model_index == 1 else "secondary", 
                                          interactive=not new_accepted_models[1])
            model3_btn_update = gr.update(value="✓ Accepted" if new_accepted_models[2] else "✓ Accept", 
                                          variant="success" if model_index == 2 else "secondary", 
                                          interactive=not new_accepted_models[2])
            
            # Prepare visual updates for the image containers
            model1_img_update = gr.update(elem_id="model1_img" + (" model-accepted" if new_accepted_models[0] else ""))
            model2_img_update = gr.update(elem_id="model2_img" + (" model-accepted" if new_accepted_models[1] else ""))
            model3_img_update = gr.update(elem_id="model3_img" + (" model-accepted" if new_accepted_models[2] else ""))
            
            # Check if we need to proceed to the next test
            if sum(new_accepted_models) == 3:
                # All models accepted, this image is too easy
                return new_accepted_models, new_gold_standard_cells, False, "All models accepted, this image is too simple. Loading new test...", gr.update(visible=True, interactive=True), model1_btn_update, model2_btn_update, model3_btn_update, model1_img_update, model2_img_update, model3_img_update, new_cell_count
            
            return new_accepted_models, new_gold_standard_cells, False, f"Model {UI_MODEL_NAMES[model_index]} accepted", gr.update(visible=True, interactive=True), model1_btn_update, model2_btn_update, model3_btn_update, model1_img_update, model2_img_update, model3_img_update, new_cell_count
        
    
        # Check if we should save the annotation
        def check_and_save_annotation(accepted_models, test_data, gold_standard_cells, cell_count=0):
            if test_data is None:
                return "Please submit a test first"
            
            # If all models are accepted, don't save (too easy)
            if all(accepted_models):
                # Trigger loading a new test
                return "All models accepted, not saving this annotation. Loading new test..."
            
            # If at least one model is accepted, save the annotation
            if any(accepted_models):
                # Get the category from test_data
                category = test_data.get("category", "")
                is_counting_category = category == "counting"
                
                # Prepare annotation data
                annotation_data = {
                    "image_filename": test_data["image_filename"],
                    "user_input": test_data["object_name"],
                    "category": category
                }
                
                # Add count field if this is a counting category
                if is_counting_category:
                    annotation_data["count"] = cell_count
                    print(f"Adding count {cell_count} to annotation data for counting category")
                else:
                    annotation_data["count"] = 1
                
                # Create a mask from the grid cells
                try:
                    # Get the image dimensions
                    image_path = test_data.get("image_path")
                    img = Image.open(image_path)
                    img_width, img_height = img.size
                    
                    # Create a mask from selected cells
                    mask = np.zeros((img_height, img_width), dtype=bool)
                    
                    # Calculate cell size for square cells
                    cell_size = min(img_width, img_height) / GRID_SIZE
                    
                    # Fill in the mask for each selected cell
                    for row, col in gold_standard_cells:
                        # Calculate cell bounds
                        x_min = int(col * cell_size)
                        y_min = int(row * cell_size)
                        x_max = min(int((col + 1) * cell_size), img_width)
                        y_max = min(int((row + 1) * cell_size), img_height)
                        
                        # Set region in mask to True
                        mask[y_min:y_max, x_min:x_max] = True
                    
                    # Save this generated mask
                    mask_filename = save_mask_image(mask, image_path)
                    annotation_data["mask_filename"] = mask_filename
                except Exception as e:
                    print(f"Error creating mask from grid cells: {e}")
                
                # Save to JSON file
                save_to_json_file(annotation_data)
                # Return message to trigger new test loading
                return "Annotation saved. Loading new test..."
            
            # If no models are accepted, prompt for manual annotation
            return "No model selected. Please either accept a model or switch to the Manual Annotation tab."
        
        # Prepare image for manual annotation
        def load_image_for_manual_annotation(image_path, test_data):
            if not image_path or image_path == "":
                return None, [], "Please load an image first", "No areas selected", None, False, 0
            
            try:
                # Load the image and draw grid
                img = Image.open(image_path)
                img_width, img_height = img.size
                
                # Calculate cell size based on the smaller dimension for square cells
                cell_size = min(img_width, img_height) / GRID_SIZE
                
                # Calculate actual grid dimensions (rows and columns) - use math.ceil to get proper boundaries
                num_cols = math.ceil(img_width / cell_size)
                num_rows = math.ceil(img_height / cell_size)
                
                # Draw grid
                img_with_grid = draw_grid_on_image(img.copy())
                
                # Check if we're dealing with a 'counting' category for display
                category = test_data.get("category", "") if test_data is not None else ""
                is_counting_category = category == "counting"
                
                # Update display text with count for counting category
                if is_counting_category:
                    cells_display = "Count: 0"
                else:
                    cells_display = "No areas selected"
                
                # Update status message to show actual grid dimensions
                status = f"Please select grid cells on the image. Using {GRID_SIZE}×{GRID_SIZE} square grid cells. Each image has between {num_rows}×{num_cols} cells depending on its aspect ratio."
                
                return img_with_grid, [], status, cells_display, None, False, 0
            except Exception as e:
                return None, [], f"Failed to load image: {str(e)}", "No areas selected", None, False, 0
        
        # Process image click event
        def on_image_click(evt: gr.SelectData, image, selected_cells, test_data, current_mask, mask_applied, cell_count):
            if test_data is None or image is None:
                return selected_cells, image, "No areas selected", None, None, mask_applied, gr.update(visible=False), "Please load an image first", cell_count
            
            try:
                # Get click coordinates
                x, y = evt.index
                print(f"Click coordinates: x={x}, y={y}")
                
                # Get image dimensions
                img_width, img_height = image.size
                print(f"Image dimensions: width={img_width}, height={img_height}")
                
                # Calculate cell size based on the smaller dimension, ensuring square cells
                cell_size = min(img_width, img_height) / GRID_SIZE
                
                # Important fix: Use precise floating point calculation for corresponding grid cells
                # Ensure coordinates start from 0, not from integer grid indices
                grid_row = int(y / cell_size)
                grid_col = int(x / cell_size)
                
                # Add debug information
                exact_row = y / cell_size
                exact_col = x / cell_size
                print(f"Exact grid position: row={exact_row}, col={exact_col}")
                print(f"Grid cell: row={grid_row}, col={grid_col}")
                
                # Calculate actual grid dimensions (rows and columns)
                num_rows = math.ceil(img_height / cell_size)
                num_cols = math.ceil(img_width / cell_size)
                
                # Ensure grid coordinates are within actual grid boundaries
                if grid_row >= num_rows or grid_col >= num_cols or grid_row < 0 or grid_col < 0:
                    print(f"Click is outside grid bounds: row={grid_row}, col={grid_col}, max_row={num_rows-1}, max_col={num_cols-1}")
                    # If click is outside valid grid area, skip processing
                    return selected_cells, image, "Click is outside valid grid area", current_mask, None, mask_applied, gr.update(visible=SAM_ENABLED and len(selected_cells) > 0), "Click is outside valid grid area", cell_count
                
                # Check if this cell is already selected
                grid_cell = (grid_row, grid_col)
                
                # Ensure selected_cells is a list
                if selected_cells is None:
                    selected_cells = []
                if not isinstance(selected_cells, list):
                    print(f"selected_cells is not a list, but {type(selected_cells)}, converting to list")
                    selected_cells = []
                
                new_selected_cells = selected_cells.copy()
                print(f"Currently selected grid cells: {new_selected_cells}")
                
                # Check if this is a "counting" category
                category = test_data.get("category", "")
                is_counting_category = category == "counting"
                
                # If we have a mask already applied, we need to update both grid cells and mask
                new_cell_count = cell_count
                image_path = test_data.get("image_path")
                
                if mask_applied and current_mask is not None:
                    # Toggle the grid cell as normal
                    if grid_cell in new_selected_cells:
                        # If already selected, deselect but DO NOT change count for counting category
                        new_selected_cells.remove(grid_cell)
                        print(f"Deselected grid cell: {grid_cell}, count remains: {new_cell_count}")
                    else:
                        # If not selected, add to selection list but DO NOT change count for counting category
                        new_selected_cells.append(grid_cell)
                        print(f"Selected grid cell: {grid_cell}, count remains: {new_cell_count}")
                    
                    # Update the mask based on the click
                    img = Image.open(image_path)
                    
                    # Calculate cell bounds
                    x_min = int(grid_col * cell_size)
                    y_min = int(grid_row * cell_size)
                    x_max = min(int((grid_col + 1) * cell_size), img_width)
                    y_max = min(int((grid_row + 1) * cell_size), img_height)
                    
                    # Convert mask to numpy if needed
                    if not isinstance(current_mask, np.ndarray):
                        print("Warning: current_mask is not a numpy array, attempting conversion")
                        try:
                            current_mask = np.array(current_mask, dtype=bool)
                        except:
                            print(f"Error converting mask: type={type(current_mask)}")
                            # Create a new empty mask if conversion fails
                            current_mask = np.zeros((img_height, img_width), dtype=bool)
                    
                    # Create a copy of the mask for modification
                    updated_mask = current_mask.copy()
                    
                    # Get current state of this region in mask
                    cell_region = updated_mask[y_min:y_max, x_min:x_max]
                    # Check if any part of the region is True
                    if np.any(cell_region):
                        # If any part is True, set the whole cell to False (toggle off)
                        updated_mask[y_min:y_max, x_min:x_max] = False
                    else:
                        # Otherwise set it to True (toggle on)
                        updated_mask[y_min:y_max, x_min:x_max] = True
                    
                    # Draw the updated mask over the grid
                    img_with_grid = draw_grid_on_image(img.copy())
                    img_with_mask = sam_helper._create_mask_overlay(np.array(img_with_grid), updated_mask)
                    
                    # Update cells display text with count for counting category
                    if is_counting_category:
                        mask_pixel_count = np.count_nonzero(updated_mask)
                        cells_display = f"Count: {new_cell_count} [FIXED]"
                    else:
                        cells_display = "Mask applied"
                    
                    # Show SAM UI
                    sam_ui_visible = gr.update(visible=SAM_ENABLED)
                    
                    return new_selected_cells, img_with_mask, cells_display, updated_mask, None, True, sam_ui_visible, "", new_cell_count
                
                # If no mask has been applied yet, just handle grid cell selection normally
                else:
                    # If it's a counting category and no mask has been applied, update the cell count
                    # If a mask has been applied, maintain the existing count to prevent changes
                    if is_counting_category:
                        if grid_cell in new_selected_cells:
                            # If already selected, deselect and decrease count
                            new_selected_cells.remove(grid_cell)
                            new_cell_count = max(0, cell_count - 1)
                            print(f"Deselected grid cell: {grid_cell}, new count: {new_cell_count}")
                        else:
                            # If not selected, add to selection list and increase count
                            new_selected_cells.append(grid_cell)
                            new_cell_count = cell_count + 1
                            print(f"Selected grid cell: {grid_cell}, new count: {new_cell_count}")
                    else:
                        # For non-counting categories, only toggle selection without affecting count
                        if grid_cell in new_selected_cells:
                            # If already selected, deselect
                            new_selected_cells.remove(grid_cell)
                            print(f"Deselected grid cell: {grid_cell}")
                        else:
                            # If not selected, add to selection list
                            new_selected_cells.append(grid_cell)
                            print(f"Selected grid cell: {grid_cell}")
                    
                    print(f"Updated selected grid cells: {new_selected_cells}")
                    
                    # Update image with grid cells
                    img = Image.open(image_path)
                    img_with_grid = draw_grid_on_image(img.copy())
                    
                    # Color selected grid cells
                    img_with_selected = draw_selected_grid_cells(img_with_grid, new_selected_cells)
                    
                    # Update cells display text based on counting category
                    if is_counting_category:
                        cells_display = f"Count: {new_cell_count}"
                    else:
                        cells_display = "Areas selected"
                    
                    # If there are points and SAM is enabled, show SAM UI row
                    sam_ui_visible = gr.update(visible=SAM_ENABLED and len(new_selected_cells) > 0)
                    
                    return new_selected_cells, img_with_selected, cells_display, current_mask, None, mask_applied, sam_ui_visible, "", new_cell_count
            
            except Exception as e:
                import traceback
                traceback_str = traceback.format_exc()
                print(f"Error processing click event: {str(e)}\n{traceback_str}")
                return selected_cells, image, f"Error processing click event: {str(e)}", current_mask, None, mask_applied, gr.update(visible=False), "", cell_count
        
        # Handle accepting SAM mask
        def on_accept_mask(current_mask, test_data, cell_count):
            if current_mask is None:
                return [], None, "No mask to accept", "No areas selected", gr.update(visible=True), True, "Please generate a mask first", cell_count
            
            try:
                # Get image dimensions
                image_path = test_data.get("image_path")
                img = Image.open(image_path)
                img_width, img_height = img.size
                
                # We're no longer converting mask to grid cells
                # Instead, we'll keep the mask as-is for saving later
                
                # For visualization, we'll still overlay it on the image with the grid
                img_with_grid = draw_grid_on_image(img.copy())
                
                # Create a visual representation of the mask
                img_with_mask = sam_helper._create_mask_overlay(np.array(img_with_grid), current_mask)
                
                # Convert mask to grid cells just for handling click events
                grid_cells = sam_helper.mask_to_grid_cells(current_mask, (img_width, img_height), GRID_SIZE)
                
                # Check if we're dealing with a 'counting' category
                category = test_data.get("category", "")
                is_counting_category = category == "counting"
                
                # For counting category, preserve the cell_count from before the mask was generated
                # Update cells display text with count for counting category
                if is_counting_category:
                    # Get the number of mask pixels (for information only)
                    mask_pixel_count = np.count_nonzero(current_mask)
                    cells_display = f"Count: {cell_count} [FIXED]"
                    user_message = "Mask accepted. You can manually adjust the mask by clicking grid cells, but the count value is now fixed. Click Finish Annotation to save it."
                else:
                    cells_display = "Mask applied and ready for saving"
                    user_message = "Mask accepted. You can manually adjust the mask by clicking grid cells, then click Finish Annotation to save it."
                
                # Keep the SAM UI row visible for potential mask adjustment
                sam_ui_visible = gr.update(visible=True)
                
                # Return the grid cells so the user can continue editing, but mask_applied is True to indicate
                # we'll use the mask when saving
                return grid_cells, img_with_mask, user_message, cells_display, sam_ui_visible, True, "Mask applied! You can adjust it by clicking on the image. Click Finish Annotation when done.", cell_count
            except Exception as e:
                import traceback
                traceback_str = traceback.format_exc()
                print(f"Error accepting mask: {str(e)}\n{traceback_str}")
                return [], None, f"Error accepting mask: {str(e)}", "No areas selected", gr.update(visible=False), False, "", cell_count
        
        # Handle rejecting SAM mask
        def on_reject_mask(selected_cells, test_data, cell_count):
            try:
                # Just go back to manual selection mode
                image_path = test_data.get("image_path")
                img = Image.open(image_path)
                img_with_grid = draw_grid_on_image(img.copy())
                
                # Draw selected grid cells
                img_with_selected = draw_selected_grid_cells(img_with_grid, selected_cells)
                
                return None, img_with_selected, "Mask rejected. Continue manual selection.", gr.update(visible=True), "Mask rejected. Continue with manual selection.", cell_count
            except Exception as e:
                return None, None, f"Error rejecting mask: {str(e)}", gr.update(visible=False), "", cell_count
        
        # Connect events
        submit_btn.click(
            on_submit,
            inputs=[user_input, current_image_path, current_image_filename, current_category],
            outputs=[model1_output, model2_output, model3_output, current_test_data, accepted_models, gold_standard_cells, all_rejected, status_msg, manual_tab]
        )
        
        new_test_btn.click(
            load_random_image,
            outputs=[current_image_path, image_display, current_image_filename, current_category, current_prompt_instructions, category_display, prompt_instructions_display, model1_output, model2_output, model3_output, current_test_data, accepted_models, gold_standard_cells, all_rejected, status_msg, manual_tab]
        )
        
        # Manual annotation tab
        load_manual_btn.click(
            load_image_for_manual_annotation,
            inputs=[current_image_path, current_test_data],
            outputs=[manual_image, selected_grid_cells, manual_status, selected_cells_display, current_mask, mask_applied, cell_count]
        )

        # Add image click event
        manual_image.select(
            on_image_click,
            inputs=[manual_image, selected_grid_cells, current_test_data, current_mask, mask_applied, cell_count],
            outputs=[selected_grid_cells, manual_image, selected_cells_display, current_mask, sam_mask_image, mask_applied, sam_ui_row, sam_status, cell_count]
        )

        # Add clear selection button event
        clear_selection_btn.click(
            clear_selection,
            inputs=[current_image_path, current_test_data, cell_count],
            outputs=[selected_grid_cells, manual_image, manual_status, selected_cells_display, current_mask, mask_applied, sam_ui_row, sam_status, cell_count]
        )

        # Add connection for the start mask button
        start_mask_btn.click(
            on_start_mask,
            inputs=[selected_grid_cells, current_test_data, cell_count],
            outputs=[current_mask, sam_mask_image, sam_status, sam_ui_row, cell_count]
        )

        # Add accept mask button event
        accept_mask_btn.click(
            on_accept_mask,
            inputs=[current_mask, current_test_data, cell_count],
            outputs=[selected_grid_cells, manual_image, sam_status, selected_cells_display, sam_ui_row, mask_applied, manual_status, cell_count]
        )

        # Add reject mask button event
        reject_mask_btn.click(
            on_reject_mask,
            inputs=[selected_grid_cells, current_test_data, cell_count],
            outputs=[current_mask, manual_image, sam_status, sam_ui_row, manual_status, cell_count]
        )

        def refresh_page():
            return """
            <script>
                window.location.reload(true);
            </script>
            """

        # Then modify the finish_btn.click chain
        finish_btn.click(
            on_manual_annotation_complete,
            inputs=[selected_grid_cells, current_test_data, cell_count, current_mask, mask_applied],
            outputs=[manual_status, tabs, manual_tab]
        ).then(
            refresh_page,
            outputs=[refresh_html]  # Use the HTML component for page refresh
        ).then(
            load_random_image,
            outputs=[current_image_path, image_display, current_image_filename, current_category, current_prompt_instructions, category_display, prompt_instructions_display, model1_output, model2_output, model3_output, current_test_data, accepted_models, gold_standard_cells, all_rejected, status_msg, manual_tab]
        )
        
        # Load a random image on page load
        app.load(
            load_random_image,
            outputs=[current_image_path, image_display, current_image_filename, current_category, current_prompt_instructions, category_display, prompt_instructions_display, model1_output, model2_output, model3_output, current_test_data, accepted_models, gold_standard_cells, all_rejected, status_msg, manual_tab]
        )
        
        # Add tab change event handler
        # This will automatically load the current image when the Manual Annotation tab is selected
        tabs.select(
            fn=load_image_for_manual_annotation,
            inputs=[current_image_path, current_test_data],
            outputs=[manual_image, selected_grid_cells, manual_status, selected_cells_display, current_mask, mask_applied, cell_count],
            trigger_mode="multiple",
        )

    return app

# Clear selection function
def clear_selection(image_path, test_data, cell_count):
    if not image_path or test_data is None:
        return [], None, "Please load an image first", "No areas selected", None, False, gr.update(visible=False), "", 0
    
    try:
        # Reload image with grid, but don't select any cells
        img = Image.open(image_path)
        img_with_grid = draw_grid_on_image(img.copy())
        
        # Show SAM UI row (but hidden) if SAM is enabled
        sam_ui_visible = gr.update(visible=False)
        
        # Reset cell count to 0
        new_cell_count = 0
        
        # Check if we're dealing with a 'counting' category for display
        category = test_data.get("category", "")
        is_counting_category = category == "counting"
        
        # Update cells display text with count for counting category
        if is_counting_category:
            cells_display = f"Count: {new_cell_count}"
        else:
            cells_display = "No areas selected"
        
        return [], img_with_grid, "All selections cleared", cells_display, None, False, sam_ui_visible, "", new_cell_count
    except Exception as e:
        return [], None, f"Failed to clear selection: {str(e)}", "No areas selected", None, False, gr.update(visible=False), "", 0

# Handle manual annotation completion
def on_manual_annotation_complete(selected_cells, test_data, cell_count, current_mask=None, mask_applied=False):
    if test_data is None:
        return "Please submit a test first", gr.update(selected="Model Evaluation"), gr.update(visible=False)
    
    try:
        # Get the image path and filename from test_data
        image_path = test_data.get("image_path")
        image_filename = test_data.get("image_filename")
        
        # Get the category from test_data
        category = test_data.get("category", "")
        is_counting_category = category == "counting"
        
        # Prepare annotation data
        annotation_data = {
            "image_filename": image_filename,
            "user_input": test_data["object_name"],
            "category": category
        }
        
        # Add count field if this is a counting category
        if is_counting_category:
            annotation_data["count"] = cell_count
            print(f"Adding count {cell_count} to annotation data for counting category")
        
        # If we have an applied mask, save it and record its filename
        if mask_applied and current_mask is not None:
            # Save the mask as PNG image
            mask_filename = save_mask_image(current_mask, image_path)
            annotation_data["mask_filename"] = mask_filename
        elif selected_cells:
            # If no mask was applied but grid cells were selected, save them
            
            # Get the image dimensions to create a mask from selected cells
            img = Image.open(image_path)
            img_width, img_height = img.size
            
            # Create a mask from selected cells
            mask = np.zeros((img_height, img_width), dtype=bool)
            
            # Calculate cell size for square cells
            cell_size = min(img_width, img_height) / GRID_SIZE
            
            # Fill in the mask for each selected cell
            for row, col in selected_cells:
                # Calculate cell bounds
                x_min = int(col * cell_size)
                y_min = int(row * cell_size)
                x_max = min(int((col + 1) * cell_size), img_width)
                y_max = min(int((row + 1) * cell_size), img_height)
                
                # Set region in mask to True
                mask[y_min:y_max, x_min:x_max] = True
            
            # Save this generated mask
            mask_filename = save_mask_image(mask, image_path)
            annotation_data["mask_filename"] = mask_filename
        else:
            # If neither mask nor cells are available, return error
            return "Please either select grid cells or apply a mask", None, None
        
        # Save to local JSON file
        save_to_json_file(annotation_data)
        
        return "Annotation saved. Loading new test...", gr.update(selected="Model Evaluation"), gr.update(visible=False)
    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
        print(f"Annotation save error: {str(e)}\n{traceback_str}")
        return f"Failed to save annotation: {str(e)}", None, None

# Add a function to handle the "Start Mask" button click
def on_start_mask(selected_cells, test_data, cell_count):
    if not selected_cells or test_data is None:
        return None, None, "Please select at least one point on the image first", gr.update(), cell_count
    
    try:
        # Load the image and set it in SAM
        image_path = test_data.get("image_path")
        img = Image.open(image_path)
        
        # Set the image in SAM
        sam_helper.set_image(image_path)
        
        # Convert grid cells to pixel coordinates
        points = sam_helper.convert_grid_cells_to_points(selected_cells, img.size, GRID_SIZE)
        
        # Generate mask from points
        mask, score, mask_image = sam_helper.predict_masks_from_points(points)
        
        print(f"Generated mask with score {score}, mask shape: {mask.shape}, {mask.dtype}")
        
        return mask, mask_image, "Mask generated. Click Accept to use this mask or Reject to continue with manual selection", gr.update(), cell_count
    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
        print(f"Error generating mask: {str(e)}\n{traceback_str}")
        return None, None, f"Error generating mask: {str(e)}", gr.update(), cell_count 
    

if __name__ == "__main__":
    # Preload models
    print("Preloading Molmo models...")
    try:
        for model_name in MOLMO_MODELS:
            full_model_name = f"allenai/{model_name}"
            print(f"Preloading model: {full_model_name}")
            initialize_molmo(full_model_name)
        print("All models preloaded successfully")
    except Exception as e:
        print(f"Error preloading models: {e}")
        import traceback
        traceback.print_exc()
        print("Continuing to start the application, will try to load models when needed")

    app = ui_components()
    app.launch(share=True, server_name="0.0.0.0", server_port=7860) 
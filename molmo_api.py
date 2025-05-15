import os
import json
import re
import base64
import numpy as np
import argparse
import signal
import sys
from pathlib import Path
from io import BytesIO
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from PIL import Image
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoProcessor, 
    GenerationConfig
)
import uvicorn

# Signal handler for clean shutdown
def signal_handler(sig, frame):
    print('Received signal to shut down, cleaning up...')
    # Clean up GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("Cleared CUDA cache")
    print("Exiting gracefully")
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Configure GPU memory growth to avoid consuming all memory
def set_gpu_memory_growth():
    if torch.cuda.is_available():
        # Use a smaller fraction of GPU memory initially
        device = torch.device("cuda:0")  # Use device 0 (which is GPU 1 due to CUDA_VISIBLE_DEVICES=1)
        torch.cuda.set_per_process_memory_fraction(0.7, device)  # Use 70% of GPU memory
        print(f"Set GPU memory fraction to 70% for device {device}")
        
        # Empty cache to ensure a clean start
        torch.cuda.empty_cache()
        print(f"Cleared CUDA cache for device {device}")

# Call the function to set memory growth
set_gpu_memory_growth()

# Load environment variables
load_dotenv()

# Constants
SAVED_MODELS_DIR = Path(os.getenv("SAVED_MODELS_DIR", "models"))
SAVED_MODELS_DIR.mkdir(exist_ok=True, parents=True)

# Molmo models
MOLMO_MODELS = ["Molmo-7B-D-0924", "Molmo-7B-O-0924", "Molmo-72B-0924"]

# Initialize FastAPI app
app = FastAPI(title="Molmo API", description="API for Molmo models")

# Initialize Molmo model and processor (lazy loading)
molmo_model = None
molmo_processor = None

class MolmoRequest(BaseModel):
    image_base64: str
    object_name: str
    model_name: str
    category: str = None

class MolmoResponse(BaseModel):
    points: list
    error: str = None

def initialize_molmo(model_name="allenai/Molmo-7B-D-0924"):
    """Initialize Molmo model and processor if not already initialized."""
    global molmo_model, molmo_processor
    
    if molmo_model is None or molmo_processor is None:
        # Get model short name
        model_short_name = model_name.split('/')[-1]
        
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
            device_map={"": 0}  # Explicitly use CUDA device 0 (which is GPU 1 due to CUDA_VISIBLE_DEVICES=1)
        )
        
        molmo_model = AutoModelForCausalLM.from_pretrained(
            local_model_dir,
            trust_remote_code=True,
            torch_dtype='auto',
            device_map={"": 0}  # Explicitly use CUDA device 0 (which is GPU 1 due to CUDA_VISIBLE_DEVICES=1)
        )
        
    return molmo_model, molmo_processor

def extract_points(text, image_w, image_h):
    """Extract points from text using multiple regex patterns."""
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
            all_points.append({"point": point.tolist()})

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
            all_points.append({"point": point.tolist()})
            
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
            all_points.append({"point": point.tolist()})
            
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
            all_points.append({"point": point.tolist()})
    return all_points

def call_molmo_internal(image, object_name, model_name="allenai/Molmo-7B-D-0924", category=None):
    """Process the Molmo model inference."""
    try:
        # Initialize model and processor if not already done
        model, processor = initialize_molmo(model_name)
        
        # Get image dimensions
        img_width, img_height = image.size
        
        # Unified prompt format that works for both counting and non-counting
        prompt = f"""
        pointing: {object_name} """
        
        # Process the image and text
        inputs = processor.process(
            images=[image],
            text=prompt
        )
        
        # Move inputs to the correct device and make a batch of size 1
        inputs = {k: v.to("cuda:0").unsqueeze(0) for k, v in inputs.items()}  # Explicitly use cuda:0
        
        # Generate output with torch.autocast for better performance
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):  # Explicitly use device_index=0
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
            print(f"Extracted {len(extracted_points)} points using extract_points function")
            return extracted_points
        
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
                    return points
                
                # Fallback: attempt to extract just the coordinates
                numbers = re.findall(r'\d+\.?\d*', json_str)
                if len(numbers) >= 2:
                    # Try to pair them up as x,y coordinates
                    points = []
                    for i in range(0, len(numbers)-1, 2):
                        # Use direct x, y coordinate
                        points.append({"point": [float(numbers[i]), float(numbers[i+1])]})
                    
                    print(f"Extracted {len(points)} points using number extraction")
                    return points
                
                return []
            except Exception as e:
                print(f"Error parsing coordinates from {model_name}: {e}")
                return []
        else:
            print(f"Unable to extract coordinates from {model_name}")
            return []
    except Exception as e:
        print(f"Error calling {model_name}: {e}")
        return []

@app.post("/molmo/point", response_model=MolmoResponse)
async def get_point(request: MolmoRequest = Body(...)):
    try:
        # Decode the base64 image
        image_data = base64.b64decode(request.image_base64)
        image = Image.open(BytesIO(image_data))
        
        # Add "allenai/" prefix if not already present
        model_name = request.model_name
        if not model_name.startswith("allenai/"):
            model_name = f"allenai/{model_name}"
            
        # Call the Molmo model
        points = call_molmo_internal(
            image=image,
            object_name=request.object_name,
            model_name=model_name,
            category=request.category
        )
        
        return MolmoResponse(points=points)
    except Exception as e:
        return MolmoResponse(points=[], error=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/models")
async def list_models():
    return {"models": MOLMO_MODELS}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Molmo API Server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to run the server on (use 0.0.0.0 for public access)")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    parser.add_argument("--public", action="store_true", help="Make the API publicly accessible (equivalent to --host=0.0.0.0)")
    
    args = parser.parse_args()
    
    # If --public flag is used, override the host
    if args.public:
        args.host = "0.0.0.0"
        
    print(f"Starting Molmo API server on {args.host}:{args.port}")
    
    # Show GPU information
    if torch.cuda.is_available():
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "Not Set")
        print(f"CUDA is available. CUDA_VISIBLE_DEVICES={cuda_visible_devices}")
        print(f"Using device: cuda:0 (which corresponds to GPU {cuda_visible_devices} due to CUDA_VISIBLE_DEVICES setting)")
        print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"Available GPU memory: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB reserved, {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB allocated")
    else:
        print("CUDA is not available. Running in CPU mode.")
    
    if args.host == "0.0.0.0":
        print("WARNING: Server is publicly accessible to all network interfaces")
        print("To access from another machine, use your server's IP address or hostname")
    
    uvicorn.run(app, host=args.host, port=args.port) 
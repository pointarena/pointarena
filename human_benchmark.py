import os
import json
import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import gradio as gr
import datetime

# Constants
IMAGES_DIR = Path("images")
MASKS_DIR = Path("masks")
HUMAN_BENCHMARK_DIR = Path("human_benchmark")
HUMAN_BENCHMARK_DIR.mkdir(exist_ok=True, parents=True)

# Result file
RESULT_FILE = "human_benchmark.json"

def load_data():
    """Load data from data.json file."""
    try:
        with open("data.json", "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading data.json: {e}")
        return []

def save_results(results):
    """Save benchmark results to human_benchmark.json file."""
    try:
        with open(RESULT_FILE, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {RESULT_FILE}")
    except Exception as e:
        print(f"Error saving results: {e}")

def draw_point(image, x, y, color="red", label=None):
    """Draw a point on the image and optionally add a label."""
    draw = ImageDraw.Draw(image)
    img_width, img_height = image.size
    
    # Draw a circle at the point
    radius = max(5, min(img_width, img_height) // 100)  # Adaptive radius based on image size
    draw.ellipse(
        [(x - radius, y - radius), (x + radius, y + radius)],
        fill=color,
        outline=color
    )
    
    # Add label if provided
    if label is not None:
        draw.text((x + radius + 2, y - radius - 2), str(label), fill=color)
    
    return image

def normalize_point(x, y, img_width, img_height):
    """Normalize point coordinates to 0-1000 range in [y, x] format."""
    norm_y = y * 1000 / img_height
    norm_x = x * 1000 / img_width
    return {"point": [norm_y, norm_x]}

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

def is_point_in_mask(y, x, mask):
    """Check if a point is inside the mask."""
    if mask is None:
        return False
    
    # Ensure coordinates are within image bounds
    if y < 0 or y >= mask.shape[0] or x < 0 or x >= mask.shape[1]:
        return False
    
    # Check if point falls within the mask
    return mask[y, x]

def human_benchmark_ui():
    """Create the Gradio UI for human benchmark."""
    # Load data from data.json
    data = load_data()
    if not data:
        raise ValueError("No data found in data.json")
    
    # Initialize results
    results = {
        "total": 0,
        "success": 0,
        "failure": 0,
        "details": []
    }
    
    # Global variables
    current_data_item = None
    current_image_path = None
    current_points = []
    data_index = 0
    
    # Find first valid image
    def find_valid_image(index):
        nonlocal current_data_item, current_image_path, current_points, data_index
        
        # Reset points
        current_points = []
        
        # Check if we've reached the end of the data
        if index >= len(data):
            return None, None, "Test completed", "Test completed", "", "", None, ""
        
        # Get the current item
        item = data[index]
        data_index = index
        current_data_item = item
        
        # Get query and category
        query = item.get("user_input", "")
        category = item.get("category", "")
        
        # Find the image path
        image_filename = item.get("image_filename", "")
        image_path = None
        
        # Look for the image in category subfolders
        for img_category in os.listdir(IMAGES_DIR):
            category_dir = IMAGES_DIR / img_category
            if not category_dir.is_dir():
                continue
                
            potential_path = category_dir / image_filename
            if potential_path.exists():
                image_path = str(potential_path)
                current_image_path = image_path
                break
        
        if image_path is None:
            print(f"Image not found: {image_filename}")
            # Skip this image and move to the next one
            return find_valid_image(index + 1)
        
        # Update progress text
        progress = f"Image {index + 1}/{len(data)}"
        stats = f"Success: {results['success']}/{results['total']} ({results['success']/results['total']*100:.2f}%)" if results['total'] > 0 else "No data yet"
        status = f"Click on the image to mark '{query}'"
        
        return image_path, query, category, status, progress, stats, 0, ""
    
    def process_click(image, evt: gr.SelectData, points_label):
        nonlocal current_points
        
        if image is None:
            return image, points_label, "Please load an image first"
        
        try:
            # Get click coordinates
            x, y = evt.index
            
            # Get image dimensions
            img_width, img_height = image.size
            
            # Create a normalized point
            norm_point = normalize_point(x, y, img_width, img_height)
            
            # Add to current points
            current_points.append(norm_point)
            
            # Create a copy of the image to draw on
            img_copy = image.copy()
            
            # Draw all points
            for i, point in enumerate(current_points):
                y, x = point["point"]
                # Convert back to pixel coordinates
                pixel_y = int(y * img_height / 1000)
                pixel_x = int(x * img_width / 1000)
                # Draw the point
                draw_point(img_copy, pixel_x, pixel_y, label=(i+1) if current_data_item.get("category") == "counting" else None)
            
            # Update points label
            new_points_label = f"Points: {len(current_points)}"
            
            return img_copy, new_points_label, f"Added point at ({x}, {y})"
            
        except Exception as e:
            print(f"Error processing click: {e}")
            return image, points_label, f"Error: {str(e)}"
    
    def undo_point(image):
        nonlocal current_points
        
        if not current_points:
            return image, "No points to undo", "No points to undo"
        
        # Remove the last point
        current_points.pop()
        
        # Redraw the image from scratch
        if image is not None and current_image_path is not None:
            # Reload the original image
            img = Image.open(current_image_path)
            
            # Draw all remaining points
            for i, point in enumerate(current_points):
                y, x = point["point"]
                # Convert back to pixel coordinates
                img_width, img_height = img.size
                pixel_y = int(y * img_height / 1000)
                pixel_x = int(x * img_width / 1000)
                # Draw the point
                draw_point(img, pixel_x, pixel_y, label=(i+1) if current_data_item.get("category") == "counting" else None)
            
            return img, f"Points: {len(current_points)}", "Last point removed"
        
        return image, f"Points: {len(current_points)}", "Last point removed"
    
    def clear_points(image):
        nonlocal current_points
        
        current_points = []
        
        # Reload the original image
        if current_image_path is not None:
            img = Image.open(current_image_path)
            return img, "Points: 0", "All points cleared"
        
        return image, "Points: 0", "All points cleared"
    
    def submit_annotation():
        nonlocal data_index, current_points, current_data_item, results
        
        if current_data_item is None:
            return None, "No data loaded", "No category", "Points: 0", "Progress", "No stats", "Please load data first"
        
        # Get item details
        category = current_data_item.get("category", "")
        query = current_data_item.get("user_input", "")
        image_filename = current_data_item.get("image_filename", "")
        mask_filename = current_data_item.get("mask_filename", "")
        expected_count = current_data_item.get("count", 1)
        
        # Validate
        is_success = True
        failure_reason = ""
        
        # For counting category, check if number of points matches expected count
        if category == "counting" and len(current_points) != expected_count:
            is_success = False
            failure_reason = f"Count mismatch: expected {expected_count}, got {len(current_points)}"
        
        # For other categories with mask, check if any point is on the mask
        elif mask_filename and len(current_points) > 0:
            # Load the mask for validation
            mask_path = MASKS_DIR / mask_filename
            if mask_path.exists():
                mask = load_mask(mask_path)
                if mask is not None:
                    # Load image to get dimensions
                    img = Image.open(current_image_path)
                    img_width, img_height = img.size
                    
                    # Check if at least one point is inside the mask
                    points_in_mask = False
                    for point in current_points:
                        y, x = point["point"]
                        # Convert to pixel coordinates
                        pixel_y = int(y * img_height / 1000)
                        pixel_x = int(x * img_width / 1000)
                        
                        if is_point_in_mask(pixel_y, pixel_x, mask):
                            points_in_mask = True
                            break
                    
                    if not points_in_mask:
                        is_success = False
                        failure_reason = "No points inside the mask"
            else:
                print(f"Warning: Mask not found: {mask_filename}")
        
        # Save image with points
        output_filename = f"{Path(image_filename).stem}_human.jpg"
        output_path = HUMAN_BENCHMARK_DIR / output_filename
        
        # Create visualization
        if current_image_path:
            img = Image.open(current_image_path)
            img_width, img_height = img.size
            
            # Draw points
            for i, point in enumerate(current_points):
                y, x = point["point"]
                # Convert to pixel coordinates
                pixel_y = int(y * img_height / 1000)
                pixel_x = int(x * img_width / 1000)
                # Draw point
                draw_point(img, pixel_x, pixel_y, label=(i+1) if category == "counting" else None)
            
            # Save image
            img.save(output_path)
        
        # Update results
        results["total"] += 1
        if is_success:
            results["success"] += 1
        else:
            results["failure"] += 1
        
        # Add detail to results
        results["details"].append({
            "image": image_filename,
            "query": query,
            "category": category,
            "points": current_points,
            "success": is_success,
            "reason": failure_reason,
            "visualization": str(output_path),
            "timestamp": datetime.datetime.now().isoformat()
        })
        
        # Save results to file
        save_results(results)
        
        # Move to the next image
        next_index = data_index + 1
        return find_valid_image(next_index)
    
    # Create the UI
    with gr.Blocks(title="Human Benchmark") as demo:
        gr.Markdown("# Human Benchmark")
        gr.Markdown("Click on the image to mark points. Submit when done to move to the next image.")
        
        with gr.Row():
            # Left column
            with gr.Column(scale=3):
                # Display the query/prompt
                query_text = gr.Textbox(label="Find this object", interactive=False)
                
                # Display the image
                image = gr.Image(type="pil", label="Click to mark points")
                
                # Display category
                category_text = gr.Textbox(label="Category", interactive=False)
                
                # Display points count
                points_text = gr.Textbox(label="Task/Points", value="Points: 0", interactive=False)
                
                # Buttons for annotation
                with gr.Row():
                    undo_btn = gr.Button("Undo Last Point")
                    clear_btn = gr.Button("Clear All Points")
                    submit_btn = gr.Button("Submit", variant="primary")
            
            # Right column
            with gr.Column(scale=1):
                progress_text = gr.Textbox(label="Progress", interactive=False)
                stats_text = gr.Textbox(label="Statistics", interactive=False)
        
        # Status message
        status_text = gr.Textbox(label="Status", interactive=False)
        
        # Set up event handlers
        
        # Handle image click
        image.select(
            process_click,
            inputs=[image, points_text],
            outputs=[image, points_text, status_text]
        )
        
        # Handle undo button
        undo_btn.click(
            undo_point,
            inputs=[image],
            outputs=[image, points_text, status_text]
        )
        
        # Handle clear button
        clear_btn.click(
            clear_points,
            inputs=[image],
            outputs=[image, points_text, status_text]
        )
        
        # Handle submit button
        submit_btn.click(
            submit_annotation,
            inputs=[],
            outputs=[image, query_text, category_text, points_text, progress_text, stats_text, status_text]
        )
        
        # Initialize
        demo.load(
            lambda: find_valid_image(0),
            inputs=[],
            outputs=[image, query_text, category_text, status_text, progress_text, stats_text, points_text, status_text]
        )
    
    return demo

def main():
    parser = argparse.ArgumentParser(description="Human benchmark for object localization")
    parser.add_argument("--share", action="store_true", help="Create a public link for the interface")
    args = parser.parse_args()
    
    print("Starting Human Benchmark UI...")
    print(f"Results will be saved to {RESULT_FILE}")
    print(f"Visualizations will be saved to {HUMAN_BENCHMARK_DIR}/")
    
    # Create the UI and launch it
    demo = human_benchmark_ui()
    demo.launch(share=args.share)

if __name__ == "__main__":
    main() 
import json
import os
import openai
import time
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

def load_data(file_path: str) -> List[Dict[str, Any]]:
    """Load the JSON data from the given file path."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def save_data(data: List[Dict[str, Any]], file_path: str) -> None:
    """Save the data back to the JSON file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def optimize_text(text: str, client) -> str:
    """Optimize the text using OpenAI API."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # Use an appropriate model
            messages=[
                {"role": "system", "content": "You are a helpful assistant that improves grammar and expression accuracy without altering the original meaning. If the text begins with a noun, prepend it with “Point to.” Return only the revised text, without any explanations or additional content."},
                {"role": "user", "content": f"Optimize this text for grammar and expression accuracy: '{text}'"}
            ]
        )
        optimized_text = response.choices[0].message.content.strip()
        # Remove quotes if present
        if optimized_text.startswith('"') and optimized_text.endswith('"'):
            optimized_text = optimized_text[1:-1]
        elif optimized_text.startswith("'") and optimized_text.endswith("'"):
            optimized_text = optimized_text[1:-1]
        return optimized_text
    except Exception as e:
        print(f"Error optimizing text: {e}")
        return text  # Return original text if optimization fails

def main():
    # Set up OpenAI API
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = input("Please enter your OpenAI API key: ")
    
    client = openai.OpenAI(api_key=api_key)
    
    # Load data
    file_path = "data.json"
    data = load_data(file_path)
    
    # Track changes
    changes_made = 0
    
    # Process each item
    for i, item in enumerate(data):
        original_text = item.get("user_input", "")
        if original_text:
            print(f"Processing item {i+1}/{len(data)}: '{original_text}'")
            
            # Optimize the text
            optimized_text = optimize_text(original_text, client)
            
            # Update if changed
            if optimized_text != original_text:
                print(f"  Original: '{original_text}'")
                print(f"  Optimized: '{optimized_text}'")
                item["user_input"] = optimized_text
                changes_made += 1
            else:
                print("  No changes needed")
            
            # Avoid rate limits
            time.sleep(0.5)
    
    # Save the updated data
    if changes_made > 0:
        save_data(data, file_path)
        print(f"\nOptimization complete. Made {changes_made} changes.")
    else:
        print("\nNo changes were necessary.")

if __name__ == "__main__":
    main() 
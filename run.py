import os
import sys
import argparse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_env_vars():
    """Check if all required environment variables are set."""
    required_vars = [
        "OPENAI_API_KEY",
        "GOOGLE_API_KEY",
        "SUPABASE_URL",
        "SUPABASE_KEY",
        "R2_ENDPOINT_URL",
        "R2_ACCESS_KEY_ID",
        "R2_SECRET_ACCESS_KEY",
        "R2_BUCKET_NAME"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("Error: Missing the following environment variables:")
        for var in missing_vars:
            print(f"  - {var}")
        print("\nPlease set these variables in the .env file or in your environment.")
        return False
    
    return True

def main():
    """Main function to run the application."""
    parser = argparse.ArgumentParser(description="Point Battle - Compare different LLMs on image pointing tasks")
    parser.add_argument("--app", choices=["main", "stats"], default="main", help="Application to run (main or stats)")
    parser.add_argument("--port", type=int, default=7860, help="Application port")
    parser.add_argument("--host", default="0.0.0.0", help="Application host")
    parser.add_argument("--share", action="store_true", help="Whether to share the application")
    parser.add_argument("--debug", action="store_true", help="Whether to enable debug mode")
    
    args = parser.parse_args()
    
    # Check environment variables
    if not check_env_vars():
        sys.exit(1)
    
    # Run the selected application
    if args.app == "main":
        from app import ui_components
        print("Starting main application...")
    else:
        from stats import ui_components
        print("Starting stats application...")
    
    app = ui_components()
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        debug=args.debug,
        allowed_paths=["*"]
    )

if __name__ == "__main__":
    main() 
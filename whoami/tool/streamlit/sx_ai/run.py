"""
Simple script to run the SX AI application.
"""

import os
import sys
import importlib

def main():
    """
    Run the SX AI application using the improved_main.py file.
    """
    # Print startup message
    print("=== Starting SX AI Chat Application ===")
    
    # Try to import the config to check if environment is set up correctly
    try:
        import config
        print(f"Server configuration:")
        print(f"  Host: {config.SERVER_CONFIG['host']}")
        print(f"  Port: {config.SERVER_CONFIG['port']}")
        print(f"  Base URL Path: {config.SERVER_CONFIG['base_url_path']}")
        print(f"API URL: {config.API_BASE_URL}")
    except ImportError as e:
        print(f"Error importing configuration: {e}")
        sys.exit(1)
    
    # Import and run the main application
    try:
        print("Starting application...")
        # Use improved_main by default
        import main
    except Exception as e:
        print(f"Error starting application: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
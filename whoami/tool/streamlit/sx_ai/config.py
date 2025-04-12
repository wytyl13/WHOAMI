"""
Configuration settings for the SX AI application.
Load from environment variables if available, otherwise use defaults.
"""

import os
from dotenv import load_dotenv

# Try to load environment variables from .env file
load_dotenv()

# API Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://1.71.15.121:8888")
DEFAULT_USER_ID = os.getenv("DEFAULT_USER_ID", "13D6F349200080712111957107")

# Server configuration
SERVER_CONFIG = {
    "port": int(os.getenv("SERVER_PORT", 3000)),
    "host": os.getenv("SERVER_HOST", "0.0.0.0"),
    "base_url_path": os.getenv("SERVER_BASE_URL_PATH", "/ai/chat_sys/chat_health_report")
}

# Application configuration
APP_TITLE = "SX AI"
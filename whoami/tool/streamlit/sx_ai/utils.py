"""
Utility functions for the SX AI application.
"""

import uuid
from typing import List, Dict, Any

def create_conversation_id() -> str:
    """
    Generate a new unique conversation ID.
    
    Returns:
        A new UUID string.
    """
    return str(uuid.uuid4())

def format_chat_title(message: str, max_length: int = 20) -> str:
    """
    Format a chat title from the first message.
    
    Args:
        message: The message text.
        max_length: Maximum length before truncation.
        
    Returns:
        Formatted title string.
    """
    if len(message) > max_length:
        return message[:max_length] + "..."
    return message

def is_valid_message(message: Dict[str, Any]) -> bool:
    """
    Check if a message object is valid.
    
    Args:
        message: The message object to validate.
        
    Returns:
        True if the message is valid, False otherwise.
    """
    return (
        isinstance(message, dict) and 
        "role" in message and 
        "content" in message and
        message["role"] in ["user", "assistant"]
    )
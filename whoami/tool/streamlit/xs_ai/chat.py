"""
Chat interface components and functionality for the SX AI application.
"""

import streamlit as st
from typing import List, Dict, Any, Optional

import api
import utils

def initialize_chat_state():
    """Initialize session state variables for chat."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

def apply_chat_styles():
    """Apply styles specific to the chat interface."""
    st.markdown("""
    <style>
        /* Chat message styling */
        .stChatMessage {
            padding: 10px 15px !important;
            border-radius: 10px !important;
            margin-bottom: 10px !important;
        }
        
        /* User message styling - Right alignment */
        [data-testid="stChatMessageContent"][data-test="chatAvatarIconUser"] > div,
        [data-testid="stChatMessageContent"] div[data-testid="chatAvatarIconUser"] + div {
            display: flex !important;
            justify-content: flex-end !important;
        }
        
        /* Force user message container to right */
        .stChatMessage[data-testid="chat-message-user"] {
            background-color: #e1f5fe !important;
            float: right !important;
            clear: both !important;
            max-width: 80% !important;
        }
        
        /* Assistant message styling - Left alignment */
        .stChatMessage[data-testid="chat-message-assistant"] {
            background-color: #f5f5f5 !important;
            float: left !important;
            clear: both !important;
            max-width: 80% !important;
        }
        
        /* Fix for chat container to handle floats */
        .chat-content-area::after {
            content: "";
            display: table;
            clear: both;
        }
        
        /* Chat input styling */
        .stChatInput {
            border-radius: 30px !important;
            padding: 10px 20px !important;
        }
        
        .stChatInput > div {
            background-color: #f5f5f5 !important;
            border-radius: 30px !important;
            border: 1px solid #e0e0e0 !important;
            box-shadow: none !important;
        }
        
        .stChatInput input {
            font-size: 16px !important;
        }
        
        /* Send button styling */
        .stChatInput button {
            border-radius: 50% !important;
            background-color: #f5f5f5 !important;
        }
    </style>
    """, unsafe_allow_html=True)

def render_chat_interface():
    """Render the chat interface."""
    # Apply chat specific styles
    apply_chat_styles()
    
    # Create a container for the scrollable chat content
    chat_content = st.container()
    
    # Display chat history in the container
    with chat_content:
        st.markdown('<div class="chat-content-area">', unsafe_allow_html=True)
        
        # Display the current conversation title
        current_title = get_current_conversation_title()
        
        # Display chat messages
        if st.session_state.messages:
            for message in st.session_state.messages:
                if utils.is_valid_message(message):
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
                else:
                    print(f"跳过格式不正确的消息: {message}")
        else:
            st.info("没有聊天记录，开始新对话吧！")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Create fixed input area at the bottom
    render_chat_input()

def render_chat_input():
    """Render the chat input area at the bottom of the screen."""
    st.markdown('<div class="chat-input-container">', unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("请输入您的问题..."):
        handle_new_message(prompt)
    
    st.markdown('</div>', unsafe_allow_html=True)

def handle_new_message(prompt: str):
    """
    Handle a new user message.
    
    Args:
        prompt: The user's message text.
    """
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Show thinking state and get AI response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("思考中...")
        
        # Get response from API
        assistant_response = api.send_chat_message(
            prompt, 
            st.session_state.user_id, 
            st.session_state.current_conversation_id
        )
        
        # Update with actual response
        message_placeholder.markdown(assistant_response)
    
    # Add assistant message to history
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
    
    # Handle new conversation case
    handle_first_message_in_new_conversation(prompt)

def handle_first_message_in_new_conversation(prompt: str):
    """
    Handle special case when this is the first message in a new conversation.
    
    Args:
        prompt: The user's message that started this conversation.
    """
    if not any(conv["id"] == st.session_state.current_conversation_id for conv in st.session_state.conversations):
        # Use the first message as the conversation title
        title = utils.format_chat_title(prompt)
        
        # Add to local conversation list
        st.session_state.conversations.append({
            "id": st.session_state.current_conversation_id,
            "title": title
        })
        
        # Refresh conversation list from server
        refresh_conversation_list()
        
        # Force re-render
        st.rerun()

def get_current_conversation_title() -> str:
    """
    Get the title of the current conversation.
    
    Returns:
        The conversation title or "New chat" if not found.
    """
    current_title = "New chat"
    for conv in st.session_state.conversations:
        if conv["id"] == st.session_state.current_conversation_id:
            current_title = conv["title"]
            break
    return current_title

def refresh_conversation_list():
    """Refresh the conversation list from the server."""
    try:
        conversations_data = api.get_conversation_history(st.session_state.user_id)
        
        # Format the conversation list
        conversations = []
        for conv in conversations_data:
            conv_id = conv.get("conversation_id", "")
            if conv_id and conv.get("messages"):
                messages = conv.get("messages", [])
                if messages:
                    first_msg = messages[0].get("content", "New chat") if isinstance(messages[0], dict) else "New chat"
                    title = utils.format_chat_title(first_msg)
                    conversations.append({
                        "id": conv_id,
                        "title": title
                    })
        
        # Update session state
        st.session_state.conversations = conversations
    except Exception as e:
        print(f"刷新历史会话出错: {str(e)}")
        import traceback
        print(traceback.format_exc())
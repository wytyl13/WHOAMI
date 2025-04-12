"""
Simplified sidebar components for the SX AI application.
"""

import streamlit as st
from typing import Optional, Callable
import time

import api
import utils
from config import DEFAULT_USER_ID


def initialize_sidebar_state():
    """Initialize session state variables for the sidebar."""
    if "conversations" not in st.session_state:
        st.session_state.conversations = []
    
    if "current_conversation_id" not in st.session_state:
        st.session_state.current_conversation_id = ""  # Empty string indicates new conversation
    
    if "user_id" not in st.session_state:
        st.session_state.user_id = DEFAULT_USER_ID


def load_initial_conversations():
    """
    Load initial conversations from the API if none are loaded yet.
    """
    if not st.session_state.conversations:
        try:
            conversations_data = api.get_conversation_history(st.session_state.user_id)
            
            conversations = []
            for conv in conversations_data:
                conv_id = conv.get("conversation_id", "")
                if conv_id and conv.get("messages"):
                    # Use the first message as the title
                    messages = conv.get("messages", [])
                    if messages:
                        first_msg = messages[0].get("content", "New chat") if isinstance(messages[0], dict) else "New chat"
                        title = utils.format_chat_title(first_msg)
                        conversations.append({
                            "id": conv_id,
                            "title": title
                        })
            
            st.session_state.conversations = conversations
        except Exception as e:
            st.error(f"Error loading conversations: {str(e)}")


def render_sidebar():
    """Simplified sidebar rendering with direct Streamlit components."""
    # Apply base styles
    st.markdown("""
    <style>
        /* Base sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #f5f5f5;
        }
        
        /* Title styling */
        .sidebar-title {
            margin-bottom: 20px;
            font-size: 24px;
            font-weight: bold;
        }
        
        /* New chat button styling */
        div[data-testid="element-container"]:has(div.stButton > button:contains("New chat")) button {
            width: 100%;
            text-align: left;
            background-color: #f0f2f6;
            border: none;
            padding: 10px;
            margin-bottom: 20px;
            color: #262730;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            font-size: 14px;
            border-radius: 4px;
        }
        
        div[data-testid="element-container"]:has(div.stButton > button:contains("New chat")) button:hover {
            background-color: #e6e9ef;
        }

        /* Put "+" symbol before New chat text */
        div[data-testid="element-container"]:has(div.stButton > button:contains("New chat")) button::before {
            content: "+ ";
            color: #f85a3e;
            font-size: 18px;
            font-weight: bold;
            margin-right: 8px;
        }
        
        /* Conversation buttons styling */
        div[data-testid="element-container"]:has(div.row-widget.stButton > button) button {
            background-color: transparent;
            border: none;
            text-align: left;
            padding: 6px 10px;
            margin: 0;
            border-radius: 4px;
            min-height: 30px;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            font-size: 13px;
            line-height: 1;
            color: #262730;
        }
        
        div[data-testid="element-container"]:has(div.row-widget.stButton > button) button:hover {
            background-color: rgba(0, 0, 0, 0.05);
        }
        
        /* Delete button styling */
        button:has(div:contains("🗑️")) {
            background-color: transparent !important;
            border: none !important;
            color: #888 !important;
            padding: 6px !important;
            min-width: auto !important;
        }
        
        button:has(div:contains("🗑️")):hover {
            color: #ff4d4d !important;
        }
        
        /* Recents heading */
        .recents-heading {
            color: #666;
            font-size: 13px;
            margin: 15px 0 8px 0;
            padding-left: 10px;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }
        
        /* Minimize vertical spacing */
        div[data-testid="stVerticalBlock"] > div {
            padding-bottom: 0 !important;
            margin-bottom: 1px !important;
        }
        
        /* Remove extra vertical space inside buttons */
        button > div {
            margin-top: 0 !important;
            margin-bottom: 0 !important;
            padding-top: 0 !important;
            padding-bottom: 0 !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        # Title
        st.markdown('<div class="sidebar-title">舜熙AI</div>', unsafe_allow_html=True)
        
        # Force initialization of conversations data
        initialize_sidebar_state()
        load_initial_conversations()
        
        # New chat button - simple direct approach
        if st.button("New chat", key="new_chat_button", use_container_width=True):
            handle_new_chat()
        
        # Conversations heading
        st.markdown('<div class="recents-heading">Recents</div>', unsafe_allow_html=True)
        
        # Reverse the conversations list to show newest first
        conversations_to_display = list(reversed(st.session_state.conversations))
        
        # For each conversation, render buttons
        for idx, conversation in enumerate(conversations_to_display):
            # Calculate the original index in the session_state.conversations list
            original_idx = len(st.session_state.conversations) - 1 - idx
            
            col1, col2 = st.columns([9, 1])
            with col1:
                if st.button(conversation["title"], key=f"conv_{conversation['id']}", use_container_width=True):
                    load_conversation(conversation["id"])
            with col2:
                if st.button("🗑️", key=f"del_{conversation['id']}"):
                    delete_conversation(original_idx, conversation["id"])


def handle_new_chat():
    """Handle the new chat button click."""
    # Save current conversation to history if it has messages
    if "messages" in st.session_state and st.session_state.messages:
        if not any(conv["id"] == st.session_state.current_conversation_id for conv in st.session_state.conversations):
            first_msg = st.session_state.messages[0]["content"] if st.session_state.messages else "New chat"
            title = utils.format_chat_title(first_msg)
            st.session_state.conversations.append({
                "id": st.session_state.current_conversation_id,
                "title": title
            })
    
    # Create new conversation
    st.session_state.current_conversation_id = utils.create_conversation_id()
    st.session_state.messages = []
    st.rerun()


def load_conversation(conversation_id: str):
    """
    Load a conversation when clicked in the sidebar.
    
    Args:
        conversation_id: ID of the conversation to load.
    """
    try:
        conversations_data = api.get_conversation_history(
            st.session_state.user_id, 
            conversation_id
        )
        
        if conversations_data and len(conversations_data) > 0:
            st.session_state.messages = conversations_data[0].get("messages", [])
            st.session_state.current_conversation_id = conversation_id
            st.rerun()
        else:
            st.error("无法加载会话数据")
    except Exception as e:
        st.error(f"加载会话出错: {str(e)}")


def delete_conversation(idx: int, conversation_id: str):
    """
    Delete a conversation.
    
    Args:
        idx: Index of the conversation in the list.
        conversation_id: ID of the conversation to delete.
    """
    if api.delete_conversation(st.session_state.user_id, conversation_id):
        # Remove from local list
        st.session_state.conversations.pop(idx)
        st.success("会话已删除!")
        
        # If the deleted conversation is the current one, clear it
        if st.session_state.current_conversation_id == conversation_id:
            st.session_state.messages = []
            st.session_state.current_conversation_id = utils.create_conversation_id()
        
        st.rerun()
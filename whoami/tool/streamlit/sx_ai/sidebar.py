"""
Sidebar components and functionality for the SX AI application.
"""

import streamlit as st
from typing import Optional, Callable

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

def render_sidebar():
    """Render the sidebar with conversation history and new chat button."""
    with st.sidebar:
        # Title and new chat button
        st.markdown("<h2 style='margin-bottom: 20px;'>SX AI</h2>", unsafe_allow_html=True)
        
        # New chat button in two columns layout
        col1, col2 = st.columns([1, 5])
        with col1:
            new_chat_clicked = st.button("➕", key="new_chat_button", use_container_width=True)
        with col2:
            st.markdown("<div style='padding-top: 8px;'>New chat</div>", unsafe_allow_html=True)
        
        if new_chat_clicked:
            handle_new_chat()
        
        # Conversations list header
        st.markdown("<h3 style='margin-top:20px; font-size:16px;'>Recents</h3>", unsafe_allow_html=True)
        
        # Display conversation list
        render_conversation_list()

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

def render_conversation_list():
    """Render the list of conversations in the sidebar."""
    for idx, conversation in enumerate(st.session_state.conversations):
        col1, col2 = st.columns([5, 1])
        with col1:
            if st.button(f"{conversation['title']}", key=f"conv_{conversation['id']}", 
                        use_container_width=True):
                load_conversation(conversation["id"])
        
        with col2:
            if st.button("🗑️", key=f"del_{conversation['id']}"):
                delete_conversation(idx, conversation["id"])

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
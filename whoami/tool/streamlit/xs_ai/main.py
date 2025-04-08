"""
Main entry point for the SX AI application.
"""

import streamlit as st

# Import modules
import config
import styles
import sidebar
import chat
import audio

def main():
    """Main application function."""
    # Set page config
    st.set_page_config(
        page_title=config.PAGE_TITLE,
        layout=config.PAGE_LAYOUT
    )
    
    # Apply styles
    styles.apply_base_styles()
    styles.apply_sidebar_styles()
    styles.apply_toggle_sidebar_script()
    styles.apply_chat_styles()
    
    # Initialize session state
    sidebar.initialize_sidebar_state()
    chat.initialize_chat_state()
    
    # Load initial data
    sidebar.load_initial_conversations()
    
    # Render sidebar
    sidebar.render_sidebar()
    
    # Render chat interface
    chat.render_chat_interface()

if __name__ == "__main__":
    main()
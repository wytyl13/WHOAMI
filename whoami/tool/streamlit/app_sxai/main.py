"""
Main entry point for the SX AI application.
"""

import streamlit as st

# Import modules
import config
import styles
import sidebar
import chat

def main():
    """Main application function."""
    # Set page config
    st.set_page_config(
        page_title=config.PAGE_TITLE,
        layout=config.PAGE_LAYOUT
    )
    
    # Apply global styles
    styles.apply_global_styles()
    styles.apply_scroll_to_bottom_script()
    
    # Initialize session state
    sidebar.initialize_sidebar_state()
    chat.initialize_chat_state()
    
    # Load initial data
    sidebar.load_initial_conversations()
    
    # Render sidebar
    sidebar.render_sidebar()
    
    # Render chat interface with voice recording
    chat.render_chat_interface()

if __name__ == "__main__":
    main()
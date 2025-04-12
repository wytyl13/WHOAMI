"""
Global layout styles for the SX AI application.
"""

import streamlit as st

def apply_global_styles():
    """Apply global layout styles to the application."""
    st.markdown("""
    <style>
        /* Global layout structure */
        body {
            overflow: hidden;
        }
        
        /* Main container structure */
        .main .block-container {
            padding-top: 1rem;
            padding-bottom: 5rem;
            max-width: 100%;
        }
        
        /* Chat content area structure */
        .chat-content-area {
            height: calc(100vh - 180px);
            overflow-y: auto;
            padding-bottom: 100px;
            margin-bottom: 20px;
        }
        
        /* Fixed bottom chat input container */
        .chat-input-container {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            padding: 15px;
            background: white;
            z-index: 99;
            box-shadow: 0 -2px 10px rgba(0,0,0,0.05);
            border-top: 1px solid #f0f0f0;
        }
        
        /* Ensure content is not hidden behind the fixed input box */
        .main-content {
            margin-bottom: 80px;
        }
        
        /* Global scrollbar styling */
        ::-webkit-scrollbar {
            width: 6px;
        }
        
        ::-webkit-scrollbar-track {
            background: #f1f1f1;
        }
        
        ::-webkit-scrollbar-thumb {
            background: #c1c1c1;
            border-radius: 3px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: #a8a8a8;
        }
        
        /* Hide default Streamlit header */
        header[data-testid="stHeader"] {
            display: none;
        }
    </style>
    """, unsafe_allow_html=True)

def apply_scroll_to_bottom_script():
    """Add JavaScript to scroll chat to bottom when loaded."""
    st.markdown("""
    <script>
        // Scroll to bottom of chat content when page loads
        document.addEventListener('DOMContentLoaded', function() {
            setTimeout(function() {
                const chatContent = document.querySelector('.chat-content-area');
                if (chatContent) {
                    chatContent.scrollTop = chatContent.scrollHeight;
                }
            }, 500);
        });
    </script>
    """, unsafe_allow_html=True)
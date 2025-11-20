"""
utils.py - Utility functions for loading external files and managing resources
"""

import streamlit as st
import os

def load_css(file_name):
    """Load external CSS file"""
    try:
        with open(file_name, 'r', encoding='utf-8') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"CSS file '{file_name}' not found. Please ensure it exists in the project directory.")
    except Exception as e:
        st.error(f"Error loading CSS file '{file_name}': {str(e)}")

def load_js(file_name):
    """Load external JavaScript file"""
    try:
        with open(file_name, 'r', encoding='utf-8') as f:
            st.markdown(f'<script>{f.read()}</script>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"JavaScript file '{file_name}' not found. Please ensure it exists in the project directory.")
    except Exception as e:
        st.error(f"Error loading JavaScript file '{file_name}': {str(e)}")

def inject_script(script_name):
    """Inject a specific script function call"""
    script_calls = {
        'mobile_detection': """
        <script>
        if (typeof initializeMobileDetection === 'function') {
            initializeMobileDetection();
        }
        </script>
        """,
        
        'auto_scroll': """
        <script>
        if (typeof autoScrollChat === 'function') {
            autoScrollChat();
        }
        </script>
        """,
        
        'textarea_resize': """
        <script>
        if (typeof initializeTextareaResize === 'function') {
            initializeTextareaResize();
        }
        </script>
        """
    }
    
    if script_name in script_calls:
        st.markdown(script_calls[script_name], unsafe_allow_html=True)
    else:
        st.error(f"Script '{script_name}' not found in available scripts.")

def validate_environment_variables(required_vars):
    """Validate that required environment variables are set"""
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    return missing_vars

def get_file_size(file_path):
    """Get file size in bytes"""
    try:
        return os.path.getsize(file_path)
    except OSError:
        return 0

def check_file_exists(file_path):
    """Check if file exists"""
    return os.path.isfile(file_path)
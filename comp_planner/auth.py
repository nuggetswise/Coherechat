"""
Authentication module for Compensation Planning Assistant.
Handles user authentication using email-based login.
"""
import streamlit as st
import hmac
import os
from datetime import datetime, timedelta

# Configuration for authentication
ALLOWED_EMAIL_DOMAINS = [
    "company.com",
    "partner.com"
]

# For more sophisticated access control, you could maintain a list of specific emails
ALLOWED_EMAILS = [
    "admin@company.com",
    "analyst@partner.com"
]

def check_password():
    """
    Returns `True` if the user had the correct email, or if they haven't tried yet.
    """
    if "authentication_status" in st.session_state:
        return st.session_state["authentication_status"]
    
    # Check if user email exists in session state
    if "user_email" not in st.session_state:
        st.session_state["user_email"] = None
    
    # Create login form
    with st.form("login_form"):
        email = st.text_input("Email")
        submitted = st.form_submit_button("Login")
        
        if submitted:
            # For a real app, verify email against database or allowlist
            # This is a simple placeholder implementation
            # In production, you'd likely use a service like Auth0, Cognito, or Firebase Auth
            
            # Example: Check if email is from an allowed domain or in allowed list
            is_valid = any(email.endswith(f"@{domain}") for domain in ALLOWED_EMAIL_DOMAINS)
            is_valid = is_valid or email in ALLOWED_EMAILS
            
            # For demo purposes, allow any email
            # Remove this line in production
            is_valid = True
            
            if is_valid:
                st.session_state["authentication_status"] = True
                st.session_state["user_email"] = email
                
                # Store Cohere API key from secrets for this session
                if hasattr(st, "secrets") and "COHERE_API_KEY" in st.secrets:
                    st.session_state["cohere_api_key"] = st.secrets["COHERE_API_KEY"]
                
                return True
            else:
                st.error("Invalid email. Please use a company email address.")
                return False
        
        return False
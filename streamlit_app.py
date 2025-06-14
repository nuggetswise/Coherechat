import streamlit as st
import os
import cohere
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Coherechat Assistant",
    page_icon="💬",
)

# Function to get API keys from environment or secrets
def get_api_keys():
    # OpenAI API Key
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key and hasattr(st, "secrets"):
        if "OPENAI_API_KEY" in st.secrets:
            openai_key = st.secrets["OPENAI_API_KEY"]
        elif "openai" in st.secrets and "OPENAI_API_KEY" in st.secrets.openai:
            openai_key = st.secrets.openai["OPENAI_API_KEY"]
    
    # Cohere API Key
    cohere_key = os.getenv("COHERE_API_KEY")
    if not cohere_key and hasattr(st, "secrets"):
        if "COHERE_API_KEY" in st.secrets:
            cohere_key = st.secrets["COHERE_API_KEY"]
        elif "cohere" in st.secrets and "COHERE_API_KEY" in st.secrets.cohere:
            cohere_key = st.secrets.cohere["COHERE_API_KEY"]
    
    return openai_key, cohere_key

# Title and introduction
st.title("💬 Coherechat Assistant")
st.caption("AI-powered assistant - OpenAI primary, Cohere fallback")

# Custom styling
st.markdown("""
<style>
div.stButton > button {
    background-color: #4CAF50;
    color: white;
    padding: 8px 20px;
    border-radius: 4px;
}
.provider-indicator {
    position: fixed;
    top: 10px;
    right: 10px;
    background: rgba(0,0,0,0.8);
    color: white;
    padding: 5px 10px;
    border-radius: 15px;
    font-size: 12px;
    z-index: 1000;
}
</style>
""", unsafe_allow_html=True)

# Initialize API keys
openai_key, cohere_key = get_api_keys()

# API key input if needed
if not openai_key and not cohere_key:
    with st.expander("⚙️ API Key Settings"):
        col1, col2 = st.columns(2)
        with col1:
            openai_key = st.text_input("OpenAI API Key (Primary):", type="password")
        with col2:
            cohere_key = st.text_input("Cohere API Key (Fallback):", type="password")
        
        if openai_key or cohere_key:
            st.success("API key(s) set for this session")

# Initialize clients
openai_client = None
cohere_client = None

if openai_key:
    openai_client = OpenAI(api_key=openai_key)
if cohere_key:
    cohere_client = cohere.Client(api_key=cohere_key)

# Function to get chat response with fallback
def get_chat_response(messages, user_message):
    # Try OpenAI first
    if openai_client:
        try:
            # Convert to OpenAI format
            openai_messages = []
            for msg in messages:
                role = "assistant" if msg["role"] == "CHATBOT" else msg["role"].lower()
                openai_messages.append({"role": role, "content": msg["message"]})
            
            stream = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=openai_messages,
                stream=True,
            )
            
            full_response = ""
            placeholder = st.empty()
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
                    placeholder.markdown(full_response + "▌")
            
            placeholder.markdown(full_response)
            return full_response, "OpenAI"
            
        except Exception as e:
            st.warning(f"OpenAI failed: {str(e)[:50]}... Trying Cohere fallback")
    
    # Fallback to Cohere
    if cohere_client:
        try:
            # Convert to Cohere format
            cohere_history = []
            for msg in messages[:-1]:  # Exclude current message
                cohere_history.append({
                    "role": msg["role"],
                    "message": msg["message"]
                })
            
            response = cohere_client.chat_stream(
                model="command-r-plus",
                message=user_message,
                chat_history=cohere_history,
                temperature=0.3,
            )
            
            full_response = ""
            placeholder = st.empty()
            
            for event in response:
                if event.event_type == "text-generation":
                    full_response += event.text
                    placeholder.markdown(full_response + "▌")
            
            placeholder.markdown(full_response)
            return full_response, "Cohere"
            
        except Exception as e:
            st.error(f"Both OpenAI and Cohere failed: {str(e)}")
            return "I'm sorry, I'm experiencing technical difficulties. Please try again.", "Error"
    
    st.error("No API keys available. Please provide at least one API key.")
    return "Please provide an API key to continue.", "Error"

# Only proceed if we have at least one API key
if openai_client or cohere_client:
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "CHATBOT", "message": "Hello! I'm your AI assistant. How can I help you today?"}
        ]
    
    # Display current provider status
    provider_status = "🟢 OpenAI Ready" if openai_client else ""
    if cohere_client:
        provider_status += (" + 🟠 Cohere Backup" if provider_status else "🟠 Cohere Only")
    
    st.markdown(f'<div class="provider-indicator">{provider_status}</div>', unsafe_allow_html=True)
    
    # Display chat messages
    for message in st.session_state.messages:
        role = "assistant" if message["role"] == "CHATBOT" else "user"
        with st.chat_message(role):
            st.markdown(message["message"])
    
    # Chat input
    if prompt := st.chat_input("What would you like to know?"):
        # Add user message
        st.session_state.messages.append({"role": "USER", "message": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response, provider = get_chat_response(st.session_state.messages, prompt)
                
                # Show which provider was used
                if provider != "Error":
                    st.caption(f"Response from: {provider}")
        
        # Add assistant response
        st.session_state.messages.append({"role": "CHATBOT", "message": response})
    
    # Clear chat button
    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button("Clear Chat"):
            st.session_state.messages = [
                {"role": "CHATBOT", "message": "Hello! I'm your AI assistant. How can I help you today?"}
            ]
            st.rerun()
else:
    st.info("Please provide at least one API key to start chatting.", icon="ℹ️")

# Footer
st.markdown("---")
st.caption("Made with Streamlit • OpenAI + Cohere")

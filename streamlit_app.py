import streamlit as st
import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Coherechat Assistant",
    page_icon="💬",
)

# Function to get API key from environment or secrets
def get_openai_api_key():
    # First try environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    
    # Then try secrets (checking both root level and openai section)
    if not api_key and hasattr(st, "secrets"):
        if "OPENAI_API_KEY" in st.secrets:
            api_key = st.secrets["OPENAI_API_KEY"]
        elif "openai" in st.secrets and "OPENAI_API_KEY" in st.secrets.openai:
            api_key = st.secrets.openai["OPENAI_API_KEY"]
    
    return api_key

# Title and introduction
st.title("💬 Coherechat Assistant")
st.caption("Your AI-powered conversational assistant")

# Custom styling for a more user-friendly interface
st.markdown("""
<style>
div.stButton > button {
    background-color: #4CAF50;
    color: white;
    padding: 8px 20px;
    border-radius: 4px;
}
.chat-message {
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    display: flex;
}
</style>
""", unsafe_allow_html=True)

# Initialize API key
api_key = get_openai_api_key()

# Only show API key input if needed
if not api_key:
    with st.expander("⚙️ API Key Settings"):
        api_key = st.text_input("Enter your OpenAI API key:", type="password")
        if api_key:
            st.success("API key set for this session")

# Only proceed if we have an API key
if api_key:
    # Initialize the OpenAI client
    client = OpenAI(api_key=api_key)
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! How can I assist you today?"}
        ]
    
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Accept user input
    if prompt := st.chat_input("What would you like to know?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            # Display a spinner while waiting for the response
            with st.spinner("Thinking..."):
                try:
                    # Create a completion stream using OpenAI API
                    stream = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": m["role"], "content": m["content"]} 
                                 for m in st.session_state.messages],
                        stream=True,
                    )
                    
                    # Process the streaming response
                    for chunk in stream:
                        if chunk.choices[0].delta.content:
                            full_response += chunk.choices[0].delta.content
                            message_placeholder.markdown(full_response + "▌")
                    
                    # Update with the complete response
                    message_placeholder.markdown(full_response)
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    full_response = "I'm sorry, I encountered an error. Please try again."
                    message_placeholder.markdown(full_response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})
    
    # Add a button to clear the chat history
    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button("Clear Chat"):
            st.session_state.messages = [
                {"role": "assistant", "content": "Hello! How can I assist you today?"}
            ]
            st.rerun()  # Updated from st.experimental_rerun()
else:
    st.info("Please provide an OpenAI API key to start chatting.", icon="ℹ️")

# Footer
st.markdown("---")
st.caption("Made with Streamlit • Powered by OpenAI")

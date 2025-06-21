#!/bin/bash
echo "🚀 Installing Compensation Planner dependencies..."

# Upgrade pip first
pip install --upgrade pip

# Install core dependencies first (without tiktoken)
echo "📦 Installing core dependencies..."
pip install streamlit==1.36.0 \
  cohere==5.15.0 \
  openai==1.38.0 \
  google-generativeai==0.3.2 \
  langchain==0.1.13 \
  langchain-core==0.1.42 \
  langsmith==0.1.21 \
  pydantic==1.10.13 \
  pandas==2.2.3 \
  numpy==1.26.3 \
  pypdf==4.3.1 \
  python-docx==1.1.2 \
  beautifulsoup4==4.12.3 \
  requests==2.32.4 \
  python-dotenv==1.0.0 \
  typing-extensions==4.9.0 \
  streamlit-option-menu==0.4.0

# Try to install tiktoken with different approaches for Python 3.13+ compatibility
echo "🔧 Attempting to install tiktoken (optional)..."
echo "   Note: tiktoken may fail on Python 3.13+ and some Linux systems"

# Method 1: Try standard installation
if pip install "tiktoken>=0.5.0"; then
    echo "✅ tiktoken installed successfully"
elif pip install --no-build-isolation tiktoken; then
    echo "✅ tiktoken installed with --no-build-isolation"
elif pip install --only-binary=all tiktoken; then
    echo "✅ tiktoken installed from binary"
else
    echo "⚠️ tiktoken installation failed - this is normal on Python 3.13+ and some Linux systems"
    echo "   The app will work perfectly without tiktoken, just with token counting warnings"
    echo "   If you really need tiktoken, try:"
    echo "   - pip install tiktoken --no-build-isolation"
    echo "   - or use Python 3.11/3.12 instead of 3.13+"
fi

echo "✅ All dependencies installed successfully!"
echo "🚀 You can now run: streamlit run pages/Compensation_Planner.py" 
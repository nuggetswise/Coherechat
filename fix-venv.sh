#!/bin/bash
echo "🧼 Resetting virtual environment..."

rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate

echo "📦 Installing pinned dependencies..."
pip install --upgrade pip

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

echo "🔧 Attempting to install tiktoken (optional)..."
echo "   Note: tiktoken may fail on Python 3.13+ and some Linux systems"

# Try tiktoken installation with fallback options
if pip install "tiktoken>=0.5.0"; then
    echo "✅ tiktoken installed successfully"
elif pip install --no-build-isolation tiktoken; then
    echo "✅ tiktoken installed with --no-build-isolation"
else
    echo "⚠️ tiktoken installation failed - app will work without it"
fi

echo "✅ Done. Activate with: source .venv/bin/activate"
# Multi-user persistent storage version with Supabase
import streamlit as st
import os
import tempfile
import cohere
from openai import OpenAI
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.tools import DuckDuckGoSearchRun
from typing import List
import json
import pandas as pd
import requests
from urllib.parse import urlparse

# OCR imports
try:
    import pytesseract
    from PIL import Image
    import pdf2image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    st.warning("OCR libraries not installed. Install with: pip install pytesseract pillow pdf2image")

# Supabase imports
try:
    from supabase import create_client, Client
    import hashlib
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    st.warning("Supabase not installed. Install with: pip install supabase")

load_dotenv()

st.set_page_config(
    page_title="Coherechat Multi-User",
    page_icon="🔮",
    layout="wide"
)

# Function to get API keys
def get_api_keys():
    openai_key = os.getenv("OPENAI_API_KEY")
    cohere_key = os.getenv("COHERE_API_KEY")
    
    if not openai_key and hasattr(st, "secrets"):
        if "OPENAI_API_KEY" in st.secrets:
            openai_key = st.secrets["OPENAI_API_KEY"]
        elif "openai" in st.secrets and "OPENAI_API_KEY" in st.secrets.openai:
            openai_key = st.secrets.openai["OPENAI_API_KEY"]
    
    if not cohere_key and hasattr(st, "secrets"):
        if "COHERE_API_KEY" in st.secrets:
            cohere_key = st.secrets["COHERE_API_KEY"]
        elif "cohere" in st.secrets and "COHERE_API_KEY" in st.secrets.cohere:
            cohere_key = st.secrets.cohere["COHERE_API_KEY"]
    
    return openai_key, cohere_key

def init_session_state():
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'local_vectorstore' not in st.session_state:
        st.session_state.local_vectorstore = None
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = []

def process_image_with_ocr(image_file):
    """Extract text from images using free Tesseract OCR."""
    if not OCR_AVAILABLE:
        st.error("OCR not available. Please install: pip install pytesseract pillow pdf2image")
        return []
    
    try:
        # Open image
        image = Image.open(image_file)
        
        # Configure Tesseract for better handwriting recognition
        custom_config = r'--oem 3 --psm 6 -l eng'
        
        # Extract text
        text = pytesseract.image_to_string(image, config=custom_config)
        
        if text.strip():
            # Create document-like structure
            from langchain.schema import Document
            doc = Document(
                page_content=text.strip(),
                metadata={
                    'source_file': image_file.name,
                    'source_type': 'image_ocr',
                    'extraction_method': 'tesseract'
                }
            )
            return [doc]
        else:
            st.warning(f"No text found in {image_file.name}")
            return []
            
    except Exception as e:
        st.error(f"OCR failed for {image_file.name}: {str(e)}")
        return []

def process_mixed_files(uploaded_files, cohere_key):
    """Process mixed file types (PDFs, images) with appropriate methods."""
    if not cohere_key:
        st.error("Cohere API key required for document processing")
        return None
    
    all_texts = []
    
    with st.spinner('Processing uploaded files with OCR support...'):
        for uploaded_file in uploaded_files:
            if uploaded_file.type == "application/pdf":
                # Process PDF
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                try:
                    loader = PyPDFLoader(tmp_path)
                    documents = loader.load()
                    
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000, 
                        chunk_overlap=200
                    )
                    texts = text_splitter.split_documents(documents)
                    
                    for text in texts:
                        text.metadata['source_file'] = uploaded_file.name
                        text.metadata['extraction_method'] = 'text'
                    
                    all_texts.extend(texts)
                    st.session_state.processed_files.append(uploaded_file.name)
                    
                finally:
                    os.unlink(tmp_path)
                
            elif uploaded_file.type in ["image/jpeg", "image/png", "image/jpg", "image/webp"]:
                # Process image with OCR
                docs = process_image_with_ocr(uploaded_file)
                all_texts.extend(docs)
                if uploaded_file.name not in st.session_state.processed_files:
                    st.session_state.processed_files.append(uploaded_file.name)
                
            else:
                st.warning(f"Unsupported file type: {uploaded_file.type}")
    
    if all_texts:
        # Create embeddings
        embeddings = CohereEmbeddings(
            model="embed-english-v3.0",
            cohere_api_key=cohere_key
        )
        
        # Create or update vector store
        if st.session_state.local_vectorstore:
            st.session_state.local_vectorstore.add_documents(all_texts)
            vectorstore = st.session_state.local_vectorstore
        else:
            vectorstore = FAISS.from_documents(all_texts, embeddings)
        
        st.success(f"✅ Processed {len(all_texts)} chunks from {len(uploaded_files)} files")
        return vectorstore
    
    return None

def generate_response(query, context_docs, context_text, openai_key, cohere_key):
    """Generate response using OpenAI (primary) or Cohere (fallback)."""
    
    # Prepare context
    if context_docs:
        context = "\n\n".join([doc.page_content for doc in context_docs])
    else:
        context = context_text
    
    if context:
        prompt = f"""Based on the following context, answer the question. If the context doesn't contain relevant information, say so.

Context:
{context}

Question: {query}

Answer:"""
    else:
        prompt = f"Please answer this question: {query}"
    
    # Try OpenAI first
    if openai_key:
        try:
            client = OpenAI(api_key=openai_key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1000
            )
            return response.choices[0].message.content, "OpenAI"
        except Exception as e:
            st.warning(f"OpenAI failed: {str(e)[:50]}... Trying Cohere")
    
    # Fallback to Cohere
    if cohere_key:
        try:
            co = cohere.Client(api_key=cohere_key)
            response = co.chat(
                model="command-r-plus",
                message=prompt,
                temperature=0.3
            )
            return response.text, "Cohere"
        except Exception as e:
            return f"Error generating response: {e}", "Error"
    
    return "No AI providers available. Please configure API keys.", "Error"

def web_search_fallback(query):
    """Fallback to web search using DuckDuckGo."""
    try:
        search = DuckDuckGoSearchRun(num_results=3)
        results = search.run(query)
        return results, ["🌐 Web Search (DuckDuckGo)"]
    except Exception as e:
        return f"Web search failed: {e}", ["❌ Search Unavailable"]

# Supabase configuration
def init_supabase():
    """Initialize Supabase client for persistent storage."""
    if not SUPABASE_AVAILABLE:
        return None
    
    supabase_url = os.getenv("SUPABASE_URL") or (
        st.secrets.get("supabase", {}).get("url") if hasattr(st, "secrets") else None
    )
    supabase_key = os.getenv("SUPABASE_ANON_KEY") or (
        st.secrets.get("supabase", {}).get("anon_key") if hasattr(st, "secrets") else None
    )
    
    if supabase_url and supabase_key:
        try:
            supabase: Client = create_client(supabase_url, supabase_key)
            # Test connection and create table if needed
            try:
                # Try to query the table
                supabase.table("documents").select("id").limit(1).execute()
            except Exception:
                # Table might not exist, provide setup instructions
                st.error("Database table not found. Please run the setup SQL in your Supabase dashboard.")
                st.code("""
-- SAFE Setup SQL for Supabase (run this in your SQL Editor):

-- Step 1: Create the table
CREATE TABLE IF NOT EXISTS documents (
  id BIGSERIAL PRIMARY KEY,
  user_id TEXT NOT NULL,
  doc_id TEXT UNIQUE NOT NULL,
  content TEXT NOT NULL,
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Step 2: Add indexes for performance
CREATE INDEX IF NOT EXISTS idx_documents_user_id ON documents(user_id);
CREATE INDEX IF NOT EXISTS idx_documents_content ON documents USING gin(to_tsvector('english', content));

-- Step 3: Enable Row Level Security (optional but recommended)
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;

-- Step 4: Create access policies (only run if you haven't created them before)
-- Check if policies exist first by going to Authentication > Policies in Supabase dashboard
-- If no policies exist, run these commands:

-- For authenticated users:
CREATE POLICY "authenticated_users_all_access" 
ON documents FOR ALL 
TO authenticated 
USING (true) 
WITH CHECK (true);

-- For anonymous users (if using anon key):
CREATE POLICY "anon_users_all_access" 
ON documents FOR ALL 
TO anon 
USING (true) 
WITH CHECK (true);

-- Step 5: Grant necessary permissions
GRANT ALL ON documents TO authenticated;
GRANT ALL ON documents TO anon;
GRANT USAGE ON SEQUENCE documents_id_seq TO authenticated;
GRANT USAGE ON SEQUENCE documents_id_seq TO anon;
                """, language="sql")
                
                st.info("""
                **Setup Instructions:**
                1. Copy the SQL above
                2. Go to your Supabase Dashboard → SQL Editor
                3. Paste and run the SQL
                4. Refresh this page
                
                **Note:** This is safe SQL that won't delete existing data.
                """)
                return None
            
            return supabase
        except Exception as e:
            st.error(f"Supabase connection failed: {e}")
            st.info("Make sure you've set SUPABASE_URL and SUPABASE_ANON_KEY in your environment or secrets")
            return None
    else:
        st.warning("Supabase credentials not found. Set SUPABASE_URL and SUPABASE_ANON_KEY to enable cloud storage.")
        return None

def get_user_id():
    """Get or create a unique user ID for session isolation."""
    if 'user_id' not in st.session_state:
        # Create a simple user identifier (in production, use proper auth)
        user_email = st.text_input("Enter your email (for data isolation):", placeholder="user@example.com")
        if user_email:
            # Hash email for privacy
            user_hash = hashlib.md5(user_email.encode()).hexdigest()[:8]
            st.session_state.user_id = user_hash
            st.session_state.user_email = user_email
            return user_hash
        else:
            st.warning("Please enter an email to access your personal data space")
            return None
    return st.session_state.user_id

def save_to_supabase(supabase, documents, user_id):
    """Save processed documents to Supabase with user isolation."""
    try:
        for i, doc in enumerate(documents):
            doc_data = {
                "user_id": user_id,
                "doc_id": f"{user_id}_{i}_{hash(doc.page_content[:100])}",
                "content": doc.page_content,
                "metadata": doc.metadata,
                "created_at": "now()"
            }
            
            # Insert document
            result = supabase.table("documents").upsert(doc_data).execute()
            
        st.success(f"✅ Saved {len(documents)} documents to cloud storage")
        return True
        
    except Exception as e:
        st.error(f"Failed to save to Supabase: {e}")
        return False

def load_from_supabase(supabase, user_id, query=None):
    """Load user's documents from Supabase."""
    try:
        # Get user's documents
        query_builder = supabase.table("documents").select("*").eq("user_id", user_id)
        
        # Add text search if query provided
        if query:
            query_builder = query_builder.text_search("content", query)
        
        result = query_builder.execute()
        
        documents = []
        for row in result.data:
            from langchain.schema import Document
            doc = Document(
                page_content=row["content"],
                metadata=row["metadata"]
            )
            documents.append(doc)
        
        return documents
        
    except Exception as e:
        st.error(f"Failed to load from Supabase: {e}")
        return []

def check_user_limits(supabase, user_id):
    """Check user's usage limits."""
    try:
        # Count user's documents
        result = supabase.table("documents").select("doc_id", count="exact").eq("user_id", user_id).execute()
        doc_count = len(result.data) if result.data else 0
        
        # Set limits (configurable)
        MAX_DOCS_PER_USER = 100
        MAX_FILE_SIZE_MB = 50
        
        if doc_count >= MAX_DOCS_PER_USER:
            st.error(f"Document limit reached ({MAX_DOCS_PER_USER}). Please delete some documents.")
            return False
        
        st.info(f"📊 Usage: {doc_count}/{MAX_DOCS_PER_USER} documents")
        return True
        
    except Exception as e:
        st.warning(f"Could not check limits: {e}")
        return True

# Main app
st.title("🔮 Coherechat - Multi-User Cloud Storage")
st.caption("Persistent storage with user isolation • Up to 10 users")

# Initialize services
init_session_state()
supabase = init_supabase()

# Get API keys
openai_key, cohere_key = get_api_keys()

# User authentication
user_id = get_user_id()
if not user_id:
    st.stop()

# API Key configuration
if not openai_key and not cohere_key:
    with st.expander("⚙️ Configure API Keys"):
        col1, col2 = st.columns(2)
        with col1:
            openai_key = st.text_input("OpenAI API Key:", type="password")
        with col2:
            cohere_key = st.text_input("Cohere API Key:", type="password")
        
        if openai_key or cohere_key:
            st.success("API key(s) configured for this session")

# Sidebar for file upload and settings
with st.sidebar:
    st.header("👤 User Info")
    st.write(f"**Email:** {st.session_state.get('user_email', 'Not set')}")
    st.write(f"**User ID:** {user_id}")
    
    # Check limits
    if supabase:
        check_user_limits(supabase, user_id)
    
    st.header("📊 Storage Options")
    
    # Storage mode selection
    storage_mode = st.radio(
        "Storage Mode:",
        ["Local Session", "Cloud Persistent"] if supabase else ["Local Session"]
    )
    
    st.header("📁 File Upload")
    
    # Check OCR availability
    if OCR_AVAILABLE:
        st.success("🔍 OCR enabled - supports images and handwritten text!")
        file_types = ["pdf", "jpg", "jpeg", "png", "webp"]
        help_text = "Upload PDFs, images, or scanned documents. OCR will extract text from images and handwritten notes."
    else:
        st.warning("📄 Text-only mode - install OCR for image support")
        file_types = ["pdf"]
        help_text = "Upload PDF files for text extraction. For image support, install: pip install pytesseract pillow pdf2image"
    
    uploaded_files = st.file_uploader(
        "Upload Files",
        type=file_types,
        accept_multiple_files=True,
        help=help_text
    )
    
    if uploaded_files and cohere_key:
        if st.button("Process Files (with OCR)", type="primary"):
            # Check limits first
            if storage_mode == "Cloud Persistent" and supabase:
                if not check_user_limits(supabase, user_id):
                    st.stop()
            
            vectorstore = process_mixed_files(uploaded_files, cohere_key)
            if vectorstore:
                st.session_state.local_vectorstore = vectorstore
                
                # Save to cloud if enabled
                if storage_mode == "Cloud Persistent" and supabase:
                    save_to_supabase(supabase, uploaded_files, user_id)
                
                st.rerun()
    
    # Show processed files
    if st.session_state.processed_files:
        st.subheader("📋 Processed Files")
        for filename in st.session_state.processed_files:
            st.write(f"✅ {filename}")
    
    # Load cloud data
    if storage_mode == "Cloud Persistent" and supabase:
        if st.button("📥 Load My Cloud Data"):
            with st.spinner("Loading your documents..."):
                cloud_docs = load_from_supabase(supabase, user_id)
                if cloud_docs and cohere_key:
                    embeddings = CohereEmbeddings(
                        model="embed-english-v3.0",
                        cohere_api_key=cohere_key
                    )
                    st.session_state.local_vectorstore = FAISS.from_documents(cloud_docs, embeddings)
                    st.success(f"✅ Loaded {len(cloud_docs)} documents from cloud")
    
    # Clear data
    if st.button("Clear All Data"):
        st.session_state.clear()
        st.rerun()

# Main chat interface
st.header("💬 Ask Questions")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if query := st.chat_input("Ask a question about your documents or anything else..."):
    # Add user message
    st.session_state.chat_history.append({"role": "user", "content": query})
    
    with st.chat_message("user"):
        st.markdown(query)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            sources = []
            response = None
            provider = None
            
            # Search documents if available
            if st.session_state.local_vectorstore and cohere_key:
                try:
                    retriever = st.session_state.local_vectorstore.as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": 5}
                    )
                    docs = retriever.get_relevant_documents(query)
                    
                    if docs:
                        # Check if the documents actually contain relevant information
                        response, provider = generate_response(query, docs, "", openai_key, cohere_key)
                        sources.append("📄 Document search")
                        
                        # Check if the AI says the context doesn't contain relevant information
                        if any(phrase in response.lower() for phrase in [
                            "context doesn't contain", 
                            "context does not contain",
                            "no relevant information",
                            "not mentioned in the context",
                            "context doesn't have",
                            "not found in the context",
                            "context doesn't provide",
                            "based on the context, i cannot"
                        ]):
                            st.info("📄 Documents searched but no relevant information found. Searching the web...")
                            # Fall back to web search
                            web_results, web_sources = web_search_fallback(query)
                            response, provider = generate_response(query, [], web_results, openai_key, cohere_key)
                            sources = web_sources  # Replace document sources with web sources
                    else:
                        # No documents found at all, try web search
                        st.info("📄 No relevant documents found. Searching the web...")
                        web_results, web_sources = web_search_fallback(query)
                        response, provider = generate_response(query, [], web_results, openai_key, cohere_key)
                        sources.extend(web_sources)
                        
                except Exception as e:
                    st.error(f"Document search failed: {e}")
                    # Fall back to web search on error
                    st.info("🌐 Falling back to web search...")
                    web_results, web_sources = web_search_fallback(query)
                    response, provider = generate_response(query, [], web_results, openai_key, cohere_key)
                    sources.extend(web_sources)
            else:
                # No documents uploaded, try web search first for better answers
                if any(word in query.lower() for word in ['news', 'current', 'today', 'latest', 'recent', 'what is', 'who is', 'when did', 'where is']):
                    st.info("🌐 No documents uploaded. Searching the web for current information...")
                    web_results, web_sources = web_search_fallback(query)
                    response, provider = generate_response(query, [], web_results, openai_key, cohere_key)
                    sources.extend(web_sources)
                else:
                    # General chat without web search
                    response, provider = generate_response(query, [], "", openai_key, cohere_key)
            
            st.markdown(response)
            st.caption(f"Response from: {provider}")
            
            if sources:
                with st.expander("Sources"):
                    for source in sources:
                        st.write(f"• {source}")
    
    # Add assistant response
    st.session_state.chat_history.append({"role": "assistant", "content": response})

# Footer
st.markdown("---")
if storage_mode == "Cloud Persistent":
    st.caption("🔮 Multi-User Cloud Storage • Data persists across sessions • User isolated")
else:
    st.caption("🔮 Local Session Storage • Data cleared when you leave")
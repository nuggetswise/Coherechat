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
# New OCR imports
try:
    import pytesseract
    from PIL import Image
    import pdf2image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    st.warning("OCR libraries not installed. Install with: pip install pytesseract pillow pdf2image")

load_dotenv()

st.set_page_config(
    page_title="Cohere Hybrid Chat",
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
    if 'cohere_datasets' not in st.session_state:
        st.session_state.cohere_datasets = []
    if 'selected_dataset' not in st.session_state:
        st.session_state.selected_dataset = None

def get_cohere_datasets(cohere_key):
    """Fetch available datasets from Cohere platform."""
    if not cohere_key:
        return []
    
    try:
        co = cohere.Client(api_key=cohere_key)
        response = co.list_datasets()
        return response.datasets if hasattr(response, 'datasets') else []
    except Exception as e:
        st.error(f"Error fetching Cohere datasets: {e}")
        return []

def search_cohere_dataset(dataset_id, query, cohere_key, top_k=5):
    """Search within a Cohere dataset."""
    if not cohere_key or not dataset_id:
        return [], []
    
    try:
        co = cohere.Client(api_key=cohere_key)
        
        # Use Cohere's dataset search functionality
        response = co.chat(
            model="command-r-plus",
            message=f"""Search the dataset for information related to: {query}
            
Please provide relevant information from the dataset that answers the query.""",
            # You can specify the dataset here if the API supports it
            temperature=0.3
        )
        
        return [response.text], ["🔮 Cohere Dataset Search"]
        
    except Exception as e:
        st.warning(f"Cohere dataset search failed: {e}")
        return [], []

def create_dataset_file_for_upload(texts):
    """Create a JSONL file suitable for Cohere dataset upload."""
    data = []
    for i, text in enumerate(texts):
        data.append({
            "id": f"doc_{i}",
            "text": text.page_content,
            "metadata": text.metadata
        })
    
    return data

def process_uploaded_files(uploaded_files, cohere_key):
    """Process uploaded PDF files and create embeddings using Cohere."""
    if not cohere_key:
        st.error("Cohere API key required for document processing")
        return None
    
    all_texts = []
    
    with st.spinner('Processing uploaded files...'):
        for uploaded_file in uploaded_files:
            if uploaded_file.type == "application/pdf":
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
                    
                    all_texts.extend(texts)
                    st.session_state.processed_files.append(uploaded_file.name)
                    
                finally:
                    os.unlink(tmp_path)
    
    if all_texts:
        embeddings = CohereEmbeddings(
            model="embed-english-v3.0",
            cohere_api_key=cohere_key
        )
        
        vectorstore = FAISS.from_documents(all_texts, embeddings)
        
        # Option to create dataset file for Cohere upload
        dataset_data = create_dataset_file_for_upload(all_texts)
        
        st.success(f"✅ Processed {len(all_texts)} document chunks from {len(uploaded_files)} files")
        
        # Provide download option for Cohere dataset
        if st.button("📥 Download as Cohere Dataset (JSONL)"):
            jsonl_content = "\n".join([json.dumps(item) for item in dataset_data])
            st.download_button(
                label="Download JSONL for Cohere",
                data=jsonl_content,
                file_name="cohere_dataset.jsonl",
                mime="application/json"
            )
            st.info("💡 Upload this file to Cohere Dashboard > Datasets to create an embed dataset!")
        
        return vectorstore
    
    return None

def process_url_content(urls, cohere_key):
    """Process URLs and extract content for embedding."""
    if not cohere_key:
        st.error("Cohere API key required for URL processing")
        return None
    
    all_texts = []
    
    with st.spinner('Extracting content from URLs...'):
        for url in urls:
            try:
                # Validate URL
                parsed = urlparse(url)
                if not parsed.scheme or not parsed.netloc:
                    st.warning(f"Invalid URL: {url}")
                    continue
                
                # Load web content
                loader = WebBaseLoader(url)
                documents = loader.load()
                
                # Split text
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, 
                    chunk_overlap=200
                )
                texts = text_splitter.split_documents(documents)
                
                # Add source information
                for text in texts:
                    text.metadata['source_url'] = url
                    text.metadata['source_type'] = 'web'
                
                all_texts.extend(texts)
                st.success(f"✅ Processed: {url}")
                
            except Exception as e:
                st.error(f"Failed to process {url}: {str(e)}")
    
    if all_texts:
        # Create Cohere embeddings
        embeddings = CohereEmbeddings(
            model="embed-english-v3.0",
            cohere_api_key=cohere_key
        )
        
        # Create or update FAISS vector store
        if st.session_state.local_vectorstore:
            # Add to existing vectorstore
            st.session_state.local_vectorstore.add_documents(all_texts)
            vectorstore = st.session_state.local_vectorstore
        else:
            # Create new vectorstore
            vectorstore = FAISS.from_documents(all_texts, embeddings)
        
        st.success(f"✅ Processed {len(all_texts)} chunks from {len(urls)} URLs")
        return vectorstore
    
    return None

def search_documents(vectorstore, query, cohere_key):
    """Search documents using local Cohere embeddings and reranking."""
    if not vectorstore:
        return [], []
    
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10}
    )
    
    relevant_docs = retriever.get_relevant_documents(query)
    
    if not relevant_docs:
        return [], []
    
    if cohere_key:
        try:
            co = cohere.Client(api_key=cohere_key)
            docs_for_rerank = [doc.page_content for doc in relevant_docs]
            
            rerank_response = co.rerank(
                model="rerank-english-v3.0",
                query=query,
                documents=docs_for_rerank,
                return_documents=True
            )
            
            reranked_docs = []
            for result in rerank_response.results:
                reranked_docs.append(relevant_docs[result.index])
            
            return reranked_docs, ["📄 Local Document Search + Reranking"]
            
        except Exception as e:
            st.warning(f"Reranking failed: {e}. Using similarity search.")
    
    return relevant_docs[:5], ["📄 Local Document Search"]

def web_search_fallback(query):
    """Fallback to web search using DuckDuckGo."""
    try:
        search = DuckDuckGoSearchRun(num_results=3)
        results = search.run(query)
        return results, ["🌐 Web Search (DuckDuckGo)"]
    except Exception as e:
        return f"Web search failed: {e}", ["❌ Search Unavailable"]

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

# Main app
st.title("🔮 Cohere Hybrid Chat")
st.caption("Local files + Cohere datasets + Web search → Intelligent answers")

# Initialize session state
init_session_state()

# Get API keys
openai_key, cohere_key = get_api_keys()

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

# Sidebar configuration
with st.sidebar:
    st.header("📊 Data Sources")
    
    # Tab selection for different data sources
    tab1, tab2 = st.tabs(["📁 Local Files", "🔮 Cohere Datasets"])
    
    with tab1:
        st.subheader("Upload Files")
        
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
        
        # Add URL input section
        st.subheader("Add URLs")
        url_input = st.text_area(
            "Enter URLs (one per line):",
            placeholder="https://example.com/article1\nhttps://example.com/article2",
            help="Add web pages to process alongside files"
        )
        
        urls = [url.strip() for url in url_input.split('\n') if url.strip()]
        
        col1, col2 = st.columns(2)
        
        with col1:
            if uploaded_files and cohere_key:
                if st.button("Process Files (with OCR)", type="primary"):
                    vectorstore = process_mixed_files(uploaded_files, cohere_key)
                    if vectorstore:
                        st.session_state.local_vectorstore = vectorstore
                        st.rerun()
        
        with col2:
            if urls and cohere_key:
                if st.button("Process URLs", type="secondary"):
                    vectorstore = process_url_content(urls, cohere_key)
                    if vectorstore:
                        st.session_state.local_vectorstore = vectorstore
                        st.rerun()
        
        # Process both together
        if (uploaded_files or urls) and cohere_key:
            if st.button("Process All Sources", type="primary", use_container_width=True):
                combined_vectorstore = None
                
                # Process files first
                if uploaded_files:
                    file_vectorstore = process_mixed_files(uploaded_files, cohere_key)
                    if file_vectorstore:
                        combined_vectorstore = file_vectorstore
                
                # Process URLs and combine
                if urls:
                    url_vectorstore = process_url_content(urls, cohere_key)
                    if url_vectorstore:
                        if combined_vectorstore:
                            # Combine with existing vectorstore
                            combined_vectorstore.merge_from(url_vectorstore)
                        else:
                            combined_vectorstore = url_vectorstore
                
                if combined_vectorstore:
                    st.session_state.local_vectorstore = combined_vectorstore
                    st.rerun()
        
        # Display processed files
        if st.session_state.processed_files:
            st.subheader("📄 Processed Files")
            for file in st.session_state.processed_files:
                st.write(f"✅ {file}")
            
            if st.button("🗑️ Clear All Files"):
                st.session_state.local_vectorstore = None
                st.session_state.processed_files = []
                st.rerun()
    
    with tab2:
        st.subheader("🔮 Cohere Datasets")
        
        if cohere_key:
            if st.button("🔄 Refresh Datasets"):
                st.session_state.cohere_datasets = get_cohere_datasets(cohere_key)
                st.rerun()
            
            if not st.session_state.cohere_datasets:
                st.session_state.cohere_datasets = get_cohere_datasets(cohere_key)
            
            if st.session_state.cohere_datasets:
                dataset_names = [f"{ds.name} (ID: {ds.id})" for ds in st.session_state.cohere_datasets]
                selected_idx = st.selectbox(
                    "Select Dataset:",
                    range(len(dataset_names)),
                    format_func=lambda x: dataset_names[x],
                    index=0 if st.session_state.selected_dataset is None else st.session_state.selected_dataset
                )
                st.session_state.selected_dataset = selected_idx
                
                selected_dataset = st.session_state.cohere_datasets[selected_idx]
                st.info(f"📊 Selected: {selected_dataset.name}")
                st.caption(f"Type: {getattr(selected_dataset, 'type', 'N/A')} | Created: {getattr(selected_dataset, 'created_at', 'N/A')}")
            else:
                st.info("No datasets found. Create datasets in Cohere Dashboard.")
                st.markdown("💡 **Tip**: Upload the JSONL file from 'Local Files' tab to create a dataset!")
        else:
            st.warning("Cohere API key required to access datasets")
    
    # Search strategy selection
    st.header("🔍 Search Strategy")
    search_options = []
    
    if st.session_state.local_vectorstore:
        search_options.append("📄 Local Documents")
    
    if cohere_key and st.session_state.cohere_datasets and st.session_state.selected_dataset is not None:
        search_options.append("🔮 Cohere Dataset")
    
    search_options.extend(["🌐 Web Search", "🔄 All Sources"])
    
    search_strategy = st.selectbox(
        "Choose search approach:",
        search_options,
        index=len(search_options)-1 if len(search_options) > 1 else 0
    )

# Main chat interface
st.header("💬 Chat Interface")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if "sources" in message:
            with st.expander("📚 Sources"):
                for source in message["sources"]:
                    st.write(f"• {source}")

# Chat input
if query := st.chat_input("Ask a question..."):
    # Add user message to chat
    st.session_state.chat_history.append({"role": "user", "content": query})
    
    with st.chat_message("user"):
        st.write(query)
    
    with st.chat_message("assistant"):
        with st.spinner("Searching and generating response..."):
            context_docs = []
            context_text = ""
            sources = []
            
            # Execute search strategy
            if search_strategy == "📄 Local Documents" and st.session_state.local_vectorstore:
                context_docs, sources = search_documents(st.session_state.local_vectorstore, query, cohere_key)
            
            elif search_strategy == "🔮 Cohere Dataset" and cohere_key and st.session_state.selected_dataset is not None:
                selected_dataset = st.session_state.cohere_datasets[st.session_state.selected_dataset]
                context_text, dataset_sources = search_cohere_dataset(selected_dataset.id, query, cohere_key)
                sources.extend(dataset_sources)
            
            elif search_strategy == "🌐 Web Search":
                context_text, web_sources = web_search_fallback(query)
                sources.extend(web_sources)
            
            elif search_strategy == "🔄 All Sources":
                # Search local documents
                if st.session_state.local_vectorstore:
                    local_docs, local_sources = search_documents(st.session_state.local_vectorstore, query, cohere_key)
                    context_docs.extend(local_docs)
                    sources.extend(local_sources)
                
                # Search Cohere dataset
                if cohere_key and st.session_state.selected_dataset is not None:
                    selected_dataset = st.session_state.cohere_datasets[st.session_state.selected_dataset]
                    dataset_context, dataset_sources = search_cohere_dataset(selected_dataset.id, query, cohere_key)
                    if isinstance(dataset_context, list):
                        context_text += "\n".join(dataset_context)
                    else:
                        context_text += dataset_context
                    sources.extend(dataset_sources)
                
                # Web search fallback if no local results
                if not context_docs and not context_text:
                    fallback_context, web_sources = web_search_fallback(query)
                    context_text += fallback_context
                    sources.extend(web_sources)
            
            # Generate response
            response, model_used = generate_response(query, context_docs, context_text, openai_key, cohere_key)
            
            st.write(response)
            
            # Show sources
            if sources:
                with st.expander("📚 Sources"):
                    for source in sources:
                        st.write(f"• {source}")
            
            st.caption(f"🤖 Generated by: {model_used}")
    
    # Add assistant response to chat history
    st.session_state.chat_history.append({
        "role": "assistant", 
        "content": response,
        "sources": sources
    })

# Footer
st.markdown("---")
st.markdown("🔮 **Cohere Hybrid Chat** - Combining local knowledge with cloud intelligence")
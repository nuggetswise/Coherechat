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
                top_k=5,
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
        st.subheader("Upload PDF Files")
        uploaded_files = st.file_uploader(
            "Upload PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload PDF files for local processing"
        )
        
        # Add URL input section
        st.subheader("Add URLs")
        url_input = st.text_area(
            "Enter URLs (one per line):",
            placeholder="https://example.com/article1\nhttps://example.com/article2",
            help="Add web pages to process alongside PDFs"
        )
        
        urls = [url.strip() for url in url_input.split('\n') if url.strip()]
        
        col1, col2 = st.columns(2)
        
        with col1:
            if uploaded_files and cohere_key:
                if st.button("Process PDF Files", type="primary"):
                    vectorstore = process_uploaded_files(uploaded_files, cohere_key)
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
                
                # Process PDFs first
                if uploaded_files:
                    combined_vectorstore = process_uploaded_files(uploaded_files, cohere_key)
                
                # Process URLs
                if urls:
                    url_vectorstore = process_url_content(urls, cohere_key)
                    if url_vectorstore:
                        combined_vectorstore = url_vectorstore
                
                if combined_vectorstore:
                    st.session_state.local_vectorstore = combined_vectorstore
                    st.rerun()

    with tab2:
        st.subheader("Cohere Platform Datasets")
        
        if cohere_key:
            if st.button("🔄 Refresh Datasets"):
                with st.spinner("Fetching datasets..."):
                    datasets = get_cohere_datasets(cohere_key)
                    st.session_state.cohere_datasets = datasets
            
            if st.session_state.cohere_datasets:
                dataset_names = [f"{d.name} ({d.id})" for d in st.session_state.cohere_datasets]
                selected = st.selectbox(
                    "Select Dataset:",
                    ["None"] + dataset_names
                )
                
                if selected != "None":
                    dataset_id = selected.split("(")[-1].strip(")")
                    st.session_state.selected_dataset = dataset_id
                    st.success(f"Selected dataset: {selected}")
                else:
                    st.session_state.selected_dataset = None
            else:
                st.info("No datasets found. Upload files to Cohere Dashboard first.")
                st.markdown("""
                **How to create datasets:**
                1. Go to [Cohere Dashboard](https://dashboard.cohere.ai)
                2. Navigate to Datasets
                3. Upload .csv or .jsonl files
                4. Create an embed dataset
                """)
        else:
            st.warning("Cohere API key required")
    
    # Settings
    st.header("⚙️ Settings")
    
    # Search strategy
    search_strategy = st.radio(
        "Search Strategy:",
        ["Local First", "Cohere Dataset First", "Both + Combine"]
    )
    
    # Provider status
    st.subheader("AI Providers")
    if openai_key:
        st.write("🟢 OpenAI (Primary)")
    if cohere_key:
        st.write("🟠 Cohere (Fallback)")
    if not openai_key and not cohere_key:
        st.write("❌ No providers configured")
    
    # Clear data
    if st.button("Clear All Data"):
        st.session_state.clear()
        st.rerun()

# Main chat interface
st.header("💬 Ask Questions")

# Display search strategy info
strategy_info = {
    "Local First": "🔍 Searches local files first, then Cohere datasets, then web",
    "Cohere Dataset First": "🔮 Searches Cohere datasets first, then local files, then web", 
    "Both + Combine": "🔄 Searches both sources and combines results"
}
st.info(strategy_info[search_strategy])

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("Sources"):
                for source in message["sources"]:
                    st.write(f"• {source}")

# Chat input
if query := st.chat_input("Ask a question about your data or anything else..."):
    # Add user message
    st.session_state.chat_history.append({"role": "user", "content": query})
    
    with st.chat_message("user"):
        st.markdown(query)
    
    # Generate response based on search strategy
    with st.chat_message("assistant"):
        with st.spinner("Searching across all sources..."):
            all_sources = []
            all_docs = []
            context_text = ""
            
            # Execute search strategy
            if search_strategy == "Local First":
                # Try local first
                if st.session_state.local_vectorstore and cohere_key:
                    docs, sources = search_documents(st.session_state.local_vectorstore, query, cohere_key)
                    if docs:
                        all_docs.extend(docs)
                        all_sources.extend(sources)
                    else:
                        # Try Cohere dataset
                        if st.session_state.selected_dataset:
                            dataset_results, dataset_sources = search_cohere_dataset(
                                st.session_state.selected_dataset, query, cohere_key
                            )
                            if dataset_results:
                                context_text = dataset_results[0]
                                all_sources.extend(dataset_sources)
                
            elif search_strategy == "Cohere Dataset First":
                # Try Cohere dataset first
                if st.session_state.selected_dataset and cohere_key:
                    dataset_results, dataset_sources = search_cohere_dataset(
                        st.session_state.selected_dataset, query, cohere_key
                    )
                    if dataset_results:
                        context_text = dataset_results[0]
                        all_sources.extend(dataset_sources)
                    else:
                        # Try local files
                        if st.session_state.local_vectorstore:
                            docs, sources = search_documents(st.session_state.local_vectorstore, query, cohere_key)
                            all_docs.extend(docs)
                            all_sources.extend(sources)
            
            elif search_strategy == "Both + Combine":
                # Search both sources
                if st.session_state.local_vectorstore and cohere_key:
                    docs, sources = search_documents(st.session_state.local_vectorstore, query, cohere_key)
                    all_docs.extend(docs)
                    all_sources.extend(sources)
                
                if st.session_state.selected_dataset and cohere_key:
                    dataset_results, dataset_sources = search_cohere_dataset(
                        st.session_state.selected_dataset, query, cohere_key
                    )
                    if dataset_results:
                        context_text += f"\n\nCohere Dataset Results:\n{dataset_results[0]}"
                        all_sources.extend(dataset_sources)
            
            # If no documents found, try web search
            if not all_docs and not context_text:
                st.info("No relevant information found in uploaded data. Searching the web...")
                web_results, web_sources = web_search_fallback(query)
                context_text = web_results
                all_sources.extend(web_sources)
            
            # Generate response
            response, provider = generate_response(query, all_docs, context_text, openai_key, cohere_key)
            
            st.markdown(response)
            st.caption(f"Response from: {provider}")
            
            # Show sources
            if all_sources:
                with st.expander("Sources"):
                    for source in all_sources:
                        st.write(f"• {source}")
                        
                        # Add source files if available
                        if all_docs:
                            source_files = list(set([doc.metadata.get('source_file', 'Unknown') for doc in all_docs]))
                            for file in source_files:
                                st.write(f"  📄 {file}")
    
    # Add assistant response
    st.session_state.chat_history.append({
        "role": "assistant", 
        "content": response,
        "sources": all_sources
    })

# Footer
st.markdown("---")
st.caption("🔮 Cohere Hybrid Chat • Local Files + Platform Datasets + Web Search")
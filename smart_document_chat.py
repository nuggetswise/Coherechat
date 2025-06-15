import streamlit as st
import os
import tempfile
import cohere
from openai import OpenAI
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain import hub
from langchain_cohere import ChatCohere
from typing import List
import numpy as np

load_dotenv()

st.set_page_config(
    page_title="Smart Document Chat",
    page_icon="📚",
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
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = []

def process_uploaded_files(uploaded_files, cohere_key):
    """Process uploaded PDF files and create embeddings using Cohere."""
    if not cohere_key:
        st.error("Cohere API key required for document processing")
        return None
    
    all_texts = []
    
    with st.spinner('Processing uploaded files...'):
        for uploaded_file in uploaded_files:
            if uploaded_file.type == "application/pdf":
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                try:
                    # Load PDF
                    loader = PyPDFLoader(tmp_path)
                    documents = loader.load()
                    
                    # Split text
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000, 
                        chunk_overlap=200
                    )
                    texts = text_splitter.split_documents(documents)
                    
                    # Add source information
                    for text in texts:
                        text.metadata['source_file'] = uploaded_file.name
                    
                    all_texts.extend(texts)
                    st.session_state.processed_files.append(uploaded_file.name)
                    
                finally:
                    # Clean up temp file
                    os.unlink(tmp_path)
    
    if all_texts:
        # Create Cohere embeddings
        embeddings = CohereEmbeddings(
            model="embed-english-v3.0",
            cohere_api_key=cohere_key
        )
        
        # Create FAISS vector store (local, no external dependencies)
        vectorstore = FAISS.from_documents(all_texts, embeddings)
        
        st.success(f"✅ Processed {len(all_texts)} document chunks from {len(uploaded_files)} files")
        return vectorstore
    
    return None

def search_documents(vectorstore, query, cohere_key):
    """Search documents using Cohere embeddings and reranking."""
    if not vectorstore:
        return [], []
    
    # Get initial candidates
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10}
    )
    
    relevant_docs = retriever.get_relevant_documents(query)
    
    if not relevant_docs:
        return [], []
    
    # Use Cohere reranking if available
    if cohere_key:
        try:
            co = cohere.Client(api_key=cohere_key)
            
            # Prepare documents for reranking
            docs_for_rerank = [doc.page_content for doc in relevant_docs]
            
            rerank_response = co.rerank(
                model="rerank-english-v3.0",
                query=query,
                documents=docs_for_rerank,
                return_documents=True
            )
            
            # Reorder documents based on reranking
            reranked_docs = []
            for result in rerank_response.results:
                reranked_docs.append(relevant_docs[result.index])
            
            return reranked_docs, ["📄 Document search with Cohere reranking"]
            
        except Exception as e:
            st.warning(f"Reranking failed: {e}. Using similarity search.")
    
    return relevant_docs[:5], ["📄 Document similarity search"]

def web_search_fallback(query):
    """Fallback to web search using DuckDuckGo."""
    try:
        search = DuckDuckGoSearchRun(num_results=3)
        results = search.run(query)
        return results, ["🌐 Web search (DuckDuckGo)"]
    except Exception as e:
        return f"Web search failed: {e}", ["❌ Search unavailable"]

def generate_response(query, context_docs, openai_key, cohere_key):
    """Generate response using OpenAI (primary) or Cohere (fallback)."""
    
    # Prepare context from documents
    context = "\n\n".join([doc.page_content for doc in context_docs])
    
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
st.title("📚 Smart Document Chat")
st.caption("Upload documents → Ask questions → Get intelligent answers with web fallback")

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

# Sidebar for file upload and settings
with st.sidebar:
    st.header("📁 Document Upload")
    
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDF files to search through"
    )
    
    if uploaded_files and cohere_key:
        if st.button("Process Documents", type="primary"):
            vectorstore = process_uploaded_files(uploaded_files, cohere_key)
            if vectorstore:
                st.session_state.vectorstore = vectorstore
                st.rerun()
    
    # Show processed files
    if st.session_state.processed_files:
        st.subheader("📋 Processed Files")
        for filename in st.session_state.processed_files:
            st.write(f"✅ {filename}")
    
    # Settings
    st.header("⚙️ Settings")
    
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

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("Sources"):
                for source in message["sources"]:
                    st.write(f"• {source}")

# Chat input
if query := st.chat_input("Ask a question about your documents or anything else..."):
    # Add user message
    st.session_state.chat_history.append({"role": "user", "content": query})
    
    with st.chat_message("user"):
        st.markdown(query)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching and thinking..."):
            sources = []
            
            # First try document search
            if st.session_state.vectorstore and cohere_key:
                docs, doc_sources = search_documents(st.session_state.vectorstore, query, cohere_key)
                sources.extend(doc_sources)
                
                if docs:
                    # Generate response from documents
                    response, provider = generate_response(query, docs, openai_key, cohere_key)
                    
                    st.markdown(response)
                    st.caption(f"Response from: {provider}")
                    
                    # Add source files
                    source_files = list(set([doc.metadata.get('source_file', 'Unknown') for doc in docs]))
                    sources.extend([f"📄 {file}" for file in source_files])
                    
                else:
                    # No relevant documents found, try web search
                    st.info("No relevant documents found. Searching the web...")
                    
                    web_results, web_sources = web_search_fallback(query)
                    sources.extend(web_sources)
                    
                    # Generate response from web results
                    response, provider = generate_response(f"{query}\n\nWeb search results:\n{web_results}", [], openai_key, cohere_key)
                    
                    st.markdown(response)
                    st.caption(f"Response from: {provider} (Web search)")
            else:
                # No documents uploaded, go straight to web search
                if not st.session_state.vectorstore:
                    st.info("No documents uploaded. Searching the web...")
                
                web_results, web_sources = web_search_fallback(query)
                sources.extend(web_sources)
                
                # Generate response
                response, provider = generate_response(f"{query}\n\nWeb search results:\n{web_results}", [], openai_key, cohere_key)
                
                st.markdown(response)
                st.caption(f"Response from: {provider}")
            
            # Show sources
            if sources:
                with st.expander("Sources"):
                    for source in sources:
                        st.write(f"• {source}")
    
    # Add assistant response
    st.session_state.chat_history.append({
        "role": "assistant", 
        "content": response,
        "sources": sources
    })

# Footer
st.markdown("---")
st.caption("Smart Document Chat • Upload → Search → Ask • Powered by OpenAI + Cohere + DuckDuckGo")
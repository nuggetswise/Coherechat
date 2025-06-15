import os
import streamlit as st
import tempfile
import json
import pandas as pd
from cohere import Client
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_cohere import ChatCohere
import time
from typing import List, Dict, Any, Optional
from supabase import create_client, Client as SupabaseClient
from datetime import datetime
import uuid

# Supabase configuration
def init_supabase() -> Optional[SupabaseClient]:
    """Initialize Supabase client"""
    try:
        supabase_url = st.secrets.get("SUPABASE_URL", "")
        supabase_key = st.secrets.get("SUPABASE_ANON_KEY", "")
        
        if not supabase_url or not supabase_key:
            st.warning("⚠️ Supabase credentials not found in secrets. Some features will be limited.")
            return None
            
        return create_client(supabase_url, supabase_key)
    except Exception as e:
        st.error(f"Failed to initialize Supabase: {e}")
        return None

def init_session_state():
    if 'cohere_api_key' not in st.session_state:
        st.session_state.cohere_api_key = ""
    if 'api_key_submitted' not in st.session_state:
        st.session_state.api_key_submitted = False
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'datasets' not in st.session_state:
        st.session_state.datasets = []
    if 'selected_dataset' not in st.session_state:
        st.session_state.selected_dataset = None
    if 'user_id' not in st.session_state:
        st.session_state.user_id = str(uuid.uuid4())  # Generate unique user ID
    if 'supabase' not in st.session_state:
        st.session_state.supabase = init_supabase()
    if 'data_source_mode' not in st.session_state:
        st.session_state.data_source_mode = "hybrid"  # hybrid, cohere_only, local_only

def save_chat_to_supabase(user_id: str, message: str, response: str, dataset_id: Optional[str] = None):
    """Save chat interaction to Supabase"""
    if not st.session_state.supabase:
        return
    
    try:
        data = {
            "user_id": user_id,
            "message": message,
            "response": response,
            "dataset_id": dataset_id,
            "timestamp": datetime.now().isoformat(),
            "session_id": st.session_state.get('session_id', str(uuid.uuid4()))
        }
        
        st.session_state.supabase.table("chat_history").insert(data).execute()
    except Exception as e:
        st.error(f"Failed to save chat history: {e}")

def load_chat_history_from_supabase(user_id: str) -> List[Dict]:
    """Load chat history from Supabase"""
    if not st.session_state.supabase:
        return []
    
    try:
        response = st.session_state.supabase.table("chat_history")\
            .select("*")\
            .eq("user_id", user_id)\
            .order("timestamp", desc=False)\
            .limit(50)\
            .execute()
        
        return response.data if response.data else []
    except Exception as e:
        st.error(f"Failed to load chat history: {e}")
        return []

def save_dataset_metadata(user_id: str, dataset_id: str, name: str, source_type: str, file_name: Optional[str] = None):
    """Save dataset metadata to Supabase"""
    if not st.session_state.supabase:
        return
    
    try:
        data = {
            "user_id": user_id,
            "dataset_id": dataset_id,
            "name": name,
            "source_type": source_type,  # 'cohere', 'local_upload'
            "file_name": file_name,
            "created_at": datetime.now().isoformat()
        }
        
        st.session_state.supabase.table("user_datasets").insert(data).execute()
    except Exception as e:
        st.error(f"Failed to save dataset metadata: {e}")

def sidebar_api_form():
    with st.sidebar:
        st.header("🔑 API Configuration")
        
        if st.session_state.api_key_submitted:
            st.success("✅ Cohere API key verified")
            if st.button("Reset Credentials"):
                st.session_state.clear()
                st.rerun()
            return True
        
        with st.form("api_credentials"):
            cohere_key = st.text_input("Cohere API Key", type="password")
            
            # Optional: Allow users to input dataset/connector IDs
            st.subheader("Optional IDs")
            existing_dataset_id = st.text_input("Existing Dataset ID (optional)", 
                                               help="If you have a specific Cohere dataset ID to use")
            existing_connector_id = st.text_input("Custom Connector ID (optional)", 
                                                 help="If you have a custom connector setup")
            
            if st.form_submit_button("Submit API Key"):
                try:
                    # Test the API key
                    client = Client(api_key=cohere_key)
                    client.models.list()  # Simple test call
                    
                    st.session_state.cohere_api_key = cohere_key
                    st.session_state.existing_dataset_id = existing_dataset_id
                    st.session_state.existing_connector_id = existing_connector_id
                    st.session_state.api_key_submitted = True
                    st.success("API key verified!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Invalid API key: {str(e)}")
        return False

def sidebar_data_source_selector():
    """Allow users to choose their data source preference"""
    with st.sidebar:
        st.divider()
        st.subheader("📊 Data Source Mode")
        
        mode = st.radio(
            "Choose your data workflow:",
            ["hybrid", "cohere_only", "local_only"],
            index=0,
            format_func=lambda x: {
                "hybrid": "🔄 Hybrid (Cohere + Local)",
                "cohere_only": "☁️ Cohere Cloud Only", 
                "local_only": "💻 Local Processing Only"
            }[x],
            help="Hybrid: Use both Cohere datasets and upload new files\nCohere Only: Use existing Cohere datasets\nLocal: Traditional local processing"
        )
        
        st.session_state.data_source_mode = mode
        
        if mode == "cohere_only":
            st.info("💡 Using only existing Cohere datasets. Upload tab will be hidden.")
        elif mode == "local_only":
            st.info("💡 Using local processing only. Cohere dataset features will be limited.")
        else:
            st.info("💡 Full hybrid mode: Upload new data OR use existing Cohere datasets.")

def extract_text_from_pdf(file) -> List[str]:
    """Extract text from PDF and split into chunks"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(file.getvalue())
            tmp_path = tmp_file.name
            
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200
        )
        texts = text_splitter.split_documents(documents)
        
        os.unlink(tmp_path)
        
        # Convert to text chunks for dataset upload
        text_chunks = [doc.page_content for doc in texts]
        return text_chunks
        
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return []

def create_dataset_file(text_chunks: List[str], filename: str) -> str:
    """Create a JSONL file for Cohere dataset upload"""
    try:
        # Create temporary JSONL file
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl')
        
        for i, text in enumerate(text_chunks):
            # Each line should be a JSON object with text field
            data = {
                "text": text,
                "id": f"{filename}_chunk_{i}",
                "source": filename
            }
            temp_file.write(json.dumps(data) + '\n')
        
        temp_file.close()
        return temp_file.name
        
    except Exception as e:
        st.error(f"Error creating dataset file: {e}")
        return None

def upload_to_cohere(file_path: str, dataset_name: str) -> str:
    """Upload dataset to Cohere and return dataset ID"""
    try:
        client = Client(api_key=st.session_state.cohere_api_key)
        
        with open(file_path, 'rb') as f:
            response = client.datasets.create(
                name=dataset_name,
                type="embed-input",  # For embedding datasets
                data=f,
                keep_original_file=True
            )
        
        # Clean up temp file
        os.unlink(file_path)
        
        return response.id
        
    except Exception as e:
        st.error(f"Error uploading to Cohere: {e}")
        return None

def create_embed_job(dataset_id: str, job_name: str) -> str:
    """Create an embedding job for the dataset"""
    try:
        client = Client(api_key=st.session_state.cohere_api_key)
        
        response = client.embed_jobs.create(
            model="embed-english-v3.0",
            dataset_id=dataset_id,
            input_type="search_document",
            name=job_name
        )
        
        return response.id
        
    except Exception as e:
        st.error(f"Error creating embed job: {e}")
        return None

def check_embed_job_status(job_id: str) -> Dict[str, Any]:
    """Check the status of an embedding job"""
    try:
        client = Client(api_key=st.session_state.cohere_api_key)
        response = client.embed_jobs.get(id=job_id)
        return {
            "status": response.status,
            "output_dataset_id": getattr(response, 'output_dataset_id', None)
        }
    except Exception as e:
        st.error(f"Error checking job status: {e}")
        return {"status": "error"}

def list_cohere_datasets() -> List[Dict[str, Any]]:
    """List all datasets from Cohere"""
    try:
        client = Client(api_key=st.session_state.cohere_api_key)
        datasets = client.datasets.list()
        return datasets.datasets if hasattr(datasets, 'datasets') else []
    except Exception as e:
        st.error(f"Error listing datasets: {e}")
        return []

def get_dataset_details(dataset_id: str) -> Optional[Dict]:
    """Get detailed information about a specific dataset including API details"""
    try:
        client = Client(api_key=st.session_state.cohere_api_key)
        dataset = client.datasets.get(dataset_id)
        
        return {
            "id": dataset.id,
            "name": dataset.name,
            "dataset_type": dataset.dataset_type,
            "validation_status": getattr(dataset, 'validation_status', 'Unknown'),
            "created_at": getattr(dataset, 'created_at', 'Unknown'),
            "download_urls": getattr(dataset, 'download_urls', []),
            "validation_error": getattr(dataset, 'validation_error', None),
            "size_bytes": getattr(dataset, 'size_bytes', 0),
            "examples_count": getattr(dataset, 'examples_count', 0)
        }
    except Exception as e:
        st.error(f"Error fetching dataset details: {e}")
        return None

def create_fine_tuning_job(dataset_id: str, model_type: str, model_name: str, base_model: str = "command") -> Optional[str]:
    """Create a fine-tuning job"""
    try:
        client = Client(api_key=st.session_state.cohere_api_key)
        
        if model_type == "chat":
            response = client.fine_tuning.create_finetuned_model(
                request={
                    "name": model_name,
                    "settings": {
                        "base_model": {
                            "base_type": base_model
                        },
                        "dataset_id": dataset_id
                    }
                }
            )
        elif model_type == "classify":
            response = client.fine_tuning.create_finetuned_model(
                request={
                    "name": model_name,
                    "settings": {
                        "base_model": {
                            "base_type": base_model
                        },
                        "dataset_id": dataset_id,
                        "task_type": "CLASSIFY"
                    }
                }
            )
        elif model_type == "rerank":
            response = client.fine_tuning.create_finetuned_model(
                request={
                    "name": model_name,
                    "settings": {
                        "base_model": {
                            "base_type": "rerank"
                        },
                        "dataset_id": dataset_id
                    }
                }
            )
        else:
            st.error("Unsupported model type")
            return None
            
        return response.finetuned_model.id if hasattr(response, 'finetuned_model') else None
        
    except Exception as e:
        st.error(f"Error creating fine-tuning job: {e}")
        return None

def list_fine_tuned_models() -> List[Any]:
    """List all fine-tuned models available to the user"""
    try:
        client = Client(api_key=st.session_state.cohere_api_key)
        models = client.fine_tuning.list_finetuned_models()
        return models.finetuned_models if hasattr(models, 'finetuned_models') else []
    except Exception as e:
        st.error(f"Error listing fine-tuned models: {e}")
        return []

def save_model_metadata(user_id: str, model_id: str, model_name: str, model_type: str, dataset_id: str, status: str = "training"):
    """Save fine-tuned model metadata to Supabase"""
    if not st.session_state.supabase:
        return
    
    try:
        data = {
            "user_id": user_id,
            "model_id": model_id,
            "model_name": model_name,
            "model_type": model_type,
            "dataset_id": dataset_id,
            "status": status,
            "created_at": datetime.now().isoformat()
        }
        
        st.session_state.supabase.table("fine_tuned_models").insert(data).execute()
    except Exception as e:
        st.error(f"Failed to save model metadata: {e}")

def query_with_custom_model(query: str, model_id: str, model_type: str) -> str:
    """Query using a fine-tuned model"""
    try:
        client = Client(api_key=st.session_state.cohere_api_key)
        
        if model_type == "chat":
            response = client.chat(
                model=model_id,
                message=query,
                temperature=0.3,
                max_tokens=800
            )
            return response.text
            
        elif model_type == "classify":
            response = client.classify(
                model=model_id,
                inputs=[query]
            )
            classifications = response.classifications[0] if response.classifications else None
            if classifications:
                return f"Classification: {classifications.prediction} (Confidence: {classifications.confidence:.2f})"
            return "Unable to classify"
            
        elif model_type == "rerank":
            # For rerank, you'd typically provide documents to rerank
            st.info("Rerank models require documents to rerank. Use the RAG functionality for better results.")
            return "Please provide documents to rerank with this model."
            
    except Exception as e:
        return f"Error querying model: {e}"

# Initialize session state
init_session_state()

# Main app layout
st.title("🌐 Cohere Hybrid RAG & Fine-Tuning Platform")
st.caption("Unified platform for RAG workflows, fine-tuning, and custom model management")

# Sidebar configuration
sidebar_data_source_selector()

# API key setup
if not sidebar_api_form():
    st.info("👈 Please enter your Cohere API key in the sidebar to continue.")
    st.stop()

# Enhanced tab configuration
if st.session_state.data_source_mode == "cohere_only":
    tabs = st.tabs(["📊 Datasets", "🎯 Fine-Tuning", "💬 Chat"])
    tab_upload, tab_datasets, tab_finetune, tab_chat = None, tabs[0], tabs[1], tabs[2]
elif st.session_state.data_source_mode == "local_only":
    tabs = st.tabs(["📤 Upload Data", "💬 Chat"])
    tab_upload, tab_datasets, tab_finetune, tab_chat = tabs[0], None, None, tabs[1]
else:  # hybrid
    tabs = st.tabs(["📤 Upload Data", "📊 Datasets", "🎯 Fine-Tuning", "💬 Chat", "📈 Analytics"])
    tab_upload, tab_datasets, tab_finetune, tab_chat, tab_analytics = tabs[0], tabs[1], tabs[2], tabs[3], tabs[4]

# Upload Data Tab (hidden in cohere_only mode)
if tab_upload:
    with tab_upload:
        st.header("Upload Documents to Cohere")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Choose a PDF file", 
                type=["pdf"],
                help="Upload a PDF document to create a dataset on Cohere's platform"
            )
            
        with col2:
            st.info("**Supported:**\n- PDF files\n- Auto-chunking\n- Cloud embedding")
        
        if uploaded_file is not None:
            st.success(f"File uploaded: {uploaded_file.name}")
            
            dataset_name = st.text_input(
                "Dataset Name", 
                value=f"dataset_{uploaded_file.name.replace('.pdf', '')}"
            )
            
            if st.button("🚀 Process and Upload to Cohere", type="primary"):
                with st.spinner("Extracting text from PDF..."):
                    text_chunks = extract_text_from_pdf(uploaded_file)
                    
                if text_chunks:
                    st.success(f"Extracted {len(text_chunks)} text chunks")
                    
                    with st.spinner("Creating dataset file..."):
                        dataset_file = create_dataset_file(text_chunks, uploaded_file.name)
                    
                    if dataset_file:
                        with st.spinner("Uploading to Cohere..."):
                            dataset_id = upload_to_cohere(dataset_file, dataset_name)
                        
                        if dataset_id:
                            st.success(f"✅ Dataset uploaded! ID: {dataset_id}")
                            
                            # Save metadata to Supabase
                            save_dataset_metadata(
                                st.session_state.user_id, 
                                dataset_id, 
                                dataset_name, 
                                "local_upload", 
                                uploaded_file.name
                            )
                            
                            # Create embedding job
                            with st.spinner("Creating embedding job..."):
                                job_name = f"embed_{dataset_name}"
                                job_id = create_embed_job(dataset_id, job_name)
                            
                            if job_id:
                                st.success(f"🔄 Embedding job started! ID: {job_id}")
                                st.info("The embedding job will process in the background. Check the 'Cohere Datasets' tab for status updates.")

# Cohere Datasets Tab (hidden in local_only mode)
if tab_datasets:
    with tab_datasets:
        st.header("📊 Dataset Management")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("🔄 Refresh Datasets"):
                st.session_state.datasets = list_cohere_datasets()
        
        with col2:
            # Show existing dataset ID if provided
            if st.session_state.get('existing_dataset_id'):
                st.info(f"📌 Predefined Dataset: {st.session_state.existing_dataset_id}")
        
        if not st.session_state.datasets:
            st.session_state.datasets = list_cohere_datasets()
        
        if st.session_state.datasets:
            st.subheader("📋 Available Datasets")
            
            # Enhanced dataset display with detailed information
            for idx, dataset in enumerate(st.session_state.datasets):
                with st.expander(f"📄 {dataset.name} ({dataset.dataset_type})", expanded=False):
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        st.write(f"**Dataset ID:** `{dataset.id}`")
                        st.write(f"**Type:** {dataset.dataset_type}")
                        st.write(f"**Status:** {getattr(dataset, 'validation_status', 'Unknown')}")
                        
                        # Copy dataset ID button
                        if st.button(f"📋 Copy ID", key=f"copy_{idx}"):
                            st.code(dataset.id)
                            st.success("Dataset ID ready to copy!")
                    
                    with col2:
                        if st.button(f"🔍 Details", key=f"details_{idx}"):
                            details = get_dataset_details(dataset.id)
                            if details:
                                st.json(details)
                    
                    with col3:
                        # Quick select for chat
                        if dataset.dataset_type == 'embed-input':
                            if st.button(f"✅ Select", key=f"select_{idx}"):
                                st.session_state.selected_dataset = dataset.id
                                st.success(f"Selected: {dataset.name}")
            
            # Dataset type filter
            st.subheader("🔍 Filter by Type")
            dataset_types = list(set([d.dataset_type for d in st.session_state.datasets]))
            selected_type = st.selectbox("Filter datasets:", ["All"] + dataset_types)
            
            if selected_type != "All":
                filtered_datasets = [d for d in st.session_state.datasets if d.dataset_type == selected_type]
                st.write(f"**{len(filtered_datasets)} datasets** of type `{selected_type}`")
        else:
            st.info("No datasets found. Upload a document to get started.")

# Fine-Tuning Tab
if tab_finetune:
    with tab_finetune:
        st.header("🎯 Fine-Tuning & Custom Models")
        
        # Model type selection
        model_tabs = st.tabs(["🔮 Chat Models", "🔍 Rerank Models", "🏷️ Classify Models", "📋 Your Models"])
        
        # Chat Model Fine-tuning
        with model_tabs[0]:
            st.subheader("💬 Create Chat Model")
            st.info("Train a conversational AI model on your specific dialogue patterns and knowledge.")
            
            # Get chat datasets
            chat_datasets = [d for d in st.session_state.get('datasets', []) if d.dataset_type == 'chat-finetune-input']
            
            if chat_datasets:
                chat_dataset_options = {f"{d.name} ({d.id})": d.id for d in chat_datasets}
                selected_chat_dataset = st.selectbox("Select Chat Dataset:", list(chat_dataset_options.keys()))
                
                if selected_chat_dataset:
                    dataset_id = chat_dataset_options[selected_chat_dataset]
                    
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        model_name = st.text_input("Model Name:", value=f"chat_model_{datetime.now().strftime('%Y%m%d')}")
                    with col2:
                        base_model = st.selectbox("Base Model:", ["command", "command-light"])
                    
                    if st.button("🚀 Start Chat Model Training", type="primary"):
                        with st.spinner("Creating fine-tuning job..."):
                            job_id = create_fine_tuning_job(dataset_id, "chat", model_name, base_model)
                        
                        if job_id:
                            st.success(f"✅ Fine-tuning job started! Model ID: {job_id}")
                            save_model_metadata(st.session_state.user_id, job_id, model_name, "chat", dataset_id)
            else:
                st.warning("No chat datasets found. Upload chat training data first.")
        
        # Rerank Model Fine-tuning
        with model_tabs[1]:
            st.subheader("🔍 Create Rerank Model")
            st.info("Improve search relevance by training on your specific query-document pairs.")
            
            rerank_datasets = [d for d in st.session_state.get('datasets', []) if d.dataset_type == 'reranker-finetune-input']
            
            if rerank_datasets:
                rerank_dataset_options = {f"{d.name} ({d.id})": d.id for d in rerank_datasets}
                selected_rerank_dataset = st.selectbox("Select Rerank Dataset:", list(rerank_dataset_options.keys()))
                
                if selected_rerank_dataset:
                    dataset_id = rerank_dataset_options[selected_rerank_dataset]
                    model_name = st.text_input("Model Name:", value=f"rerank_model_{datetime.now().strftime('%Y%m%d')}")
                    
                    if st.button("🚀 Start Rerank Model Training", type="primary"):
                        with st.spinner("Creating fine-tuning job..."):
                            job_id = create_fine_tuning_job(dataset_id, "rerank", model_name)
                        
                        if job_id:
                            st.success(f"✅ Fine-tuning job started! Model ID: {job_id}")
                            save_model_metadata(st.session_state.user_id, job_id, model_name, "rerank", dataset_id)
            else:
                st.warning("No rerank datasets found. Upload rerank training data first.")
        
        # Classify Model Fine-tuning
        with model_tabs[2]:
            st.subheader("🏷️ Create Classification Model")
            st.info("Build a text classifier for your specific categories and use cases.")
            
            classify_datasets = [d for d in st.session_state.get('datasets', []) if d.dataset_type == 'classifier-finetune-input']
            
            if classify_datasets:
                classify_dataset_options = {f"{d.name} ({d.id})": d.id for d in classify_datasets}
                selected_classify_dataset = st.selectbox("Select Classification Dataset:", list(classify_dataset_options.keys()))
                
                if selected_classify_dataset:
                    dataset_id = classify_dataset_options[selected_classify_dataset]
                    model_name = st.text_input("Model Name:", value=f"classify_model_{datetime.now().strftime('%Y%m%d')}")
                    
                    if st.button("🚀 Start Classification Model Training", type="primary"):
                        with st.spinner("Creating fine-tuning job..."):
                            job_id = create_fine_tuning_job(dataset_id, "classify", model_name)
                        
                        if job_id:
                            st.success(f"✅ Fine-tuning job started! Model ID: {job_id}")
                            save_model_metadata(st.session_state.user_id, job_id, model_name, "classify", dataset_id)
            else:
                st.warning("No classification datasets found. Upload classification training data first.")
        
        # Your Models
        with model_tabs[3]:
            st.subheader("🤖 Your Fine-Tuned Models")
            
            if st.button("🔄 Refresh Models"):
                pass  # Trigger refresh
            
            models = list_fine_tuned_models()
            
            if models:
                for model in models:
                    with st.expander(f"🤖 {model.name}", expanded=False):
                        col1, col2, col3 = st.columns([2, 1, 1])
                        
                        with col1:
                            st.write(f"**Model ID:** `{model.id}`")
                            st.write(f"**Status:** {getattr(model, 'status', 'Unknown')}")
                            st.write(f"**Created:** {getattr(model, 'created_at', 'Unknown')}")
                        
                        with col2:
                            if st.button(f"📋 Copy ID", key=f"model_copy_{model.id}"):
                                st.code(model.id)
                                st.success("Model ID ready to copy!")
                        
                        with col3:
                            if st.button(f"🧪 Test", key=f"test_{model.id}"):
                                st.session_state[f'test_model_{model.id}'] = True
                        
                        # Test interface
                        if st.session_state.get(f'test_model_{model.id}'):
                            test_query = st.text_input(f"Test query for {model.name}:", key=f"query_{model.id}")
                            if test_query and st.button(f"Run Test", key=f"run_{model.id}"):
                                result = query_with_custom_model(test_query, model.id, getattr(model, 'model_type', 'chat'))
                                st.write("**Result:**", result)
            else:
                st.info("No fine-tuned models found. Create one using the tabs above!")

# Enhanced Chat Tab
with tab_chat:
    st.header("💬 Chat with Your Data & Models")
    
    # Model selection
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        model_type = st.selectbox("Model Type:", ["Base Models", "Fine-Tuned Models"])
    
    with col2:
        if model_type == "Base Models":
            selected_model = st.selectbox("Select Model:", ["command-r-plus", "command-r", "command"])
        else:
            fine_tuned_models = list_fine_tuned_models()
            if fine_tuned_models:
                model_options = {f"{m.name} ({m.id})": m.id for m in fine_tuned_models}
                selected_model_key = st.selectbox("Select Fine-Tuned Model:", list(model_options.keys()))
                selected_model = model_options.get(selected_model_key, "command-r-plus")
            else:
                st.warning("No fine-tuned models available")
                selected_model = "command-r-plus"
    
    with col3:
        st.metric("Mode", st.session_state.data_source_mode.title())
    with col4:
        dataset_status = "Selected" if st.session_state.selected_dataset else "None"
        st.metric("Dataset", dataset_status)
    
    # Load chat history from Supabase
    if st.button("📜 Load Previous Chats"):
        saved_chats = load_chat_history_from_supabase(st.session_state.user_id)
        if saved_chats:
            st.session_state.chat_history = [
                {"role": "user", "content": chat["message"]} for chat in saved_chats[-10:]
            ] + [
                {"role": "assistant", "content": chat["response"]} for chat in saved_chats[-10:]
            ]
            st.success(f"Loaded {len(saved_chats)} previous conversations")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input with model-aware processing
    if query := st.chat_input("Ask a question..."):
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                if model_type == "Fine-Tuned Models" and fine_tuned_models:
                    # Use fine-tuned model
                    model_info = next((m for m in fine_tuned_models if m.id == selected_model), None)
                    if model_info:
                        response = query_with_custom_model(query, selected_model, getattr(model_info, 'model_type', 'chat'))
                    else:
                        response = "Model not found"
                else:
                    # Use base model with RAG
                    dataset_ids = [st.session_state.selected_dataset] if st.session_state.selected_dataset else None
                    response = query_with_hybrid_rag(query, dataset_ids, st.session_state.data_source_mode)
                
                st.markdown(response)
                
                # Save to Supabase
                save_chat_to_supabase(
                    st.session_state.user_id, 
                    query, 
                    response, 
                    st.session_state.selected_dataset
                )
        
        st.session_state.chat_history.append({"role": "assistant", "content": response})

# Analytics Tab (only in hybrid mode)
if 'tab_analytics' in locals():
    with tab_analytics:
        st.header("📈 Usage Analytics")
        
        if st.session_state.supabase:
            try:
                # Get user stats
                user_stats = st.session_state.supabase.table("chat_history")\
                    .select("*")\
                    .eq("user_id", st.session_state.user_id)\
                    .execute()
                
                if user_stats.data:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Conversations", len(user_stats.data))
                    with col2:
                        dataset_usage = len([x for x in user_stats.data if x.get('dataset_id')])
                        st.metric("Dataset Queries", dataset_usage)
                    with col3:
                        recent_chats = len([x for x in user_stats.data if 
                                          datetime.fromisoformat(x['timestamp']).date() == datetime.now().date()])
                        st.metric("Today's Chats", recent_chats)
                
                st.subheader("Recent Activity")
                if user_stats.data:
                    recent_data = user_stats.data[-10:]
                    for chat in recent_data:
                        with st.expander(f"💬 {chat['message'][:50]}..."):
                            st.write(f"**Query:** {chat['message']}")
                            st.write(f"**Response:** {chat['response'][:200]}...")
                            st.write(f"**Time:** {chat['timestamp']}")
                            
            except Exception as e:
                st.error(f"Failed to load analytics: {e}")
        else:
            st.info("📊 Analytics require Supabase connection. Configure in secrets.")

# Enhanced sidebar info
with st.sidebar:
    st.divider()
    st.subheader("🔧 Required Cohere IDs")
    st.markdown("""
    **Essential:**
    - ✅ Cohere API Key (required)
    
    **Optional:**
    - 📊 Dataset IDs (for existing datasets)
    - 🔌 Connector IDs (for custom connectors)
    - 🔄 Embed Job IDs (auto-generated)
    """)
    
    st.divider()
    st.subheader("🗄️ Supabase Features")
    supabase_status = "✅ Connected" if st.session_state.supabase else "❌ Not configured"
    st.write(f"**Status:** {supabase_status}")
    st.markdown("""
    **Features:**
    - 👤 User session management
    - 💾 Persistent chat history
    - 📊 Usage analytics
    - 🏷️ Dataset metadata
    """)
    
    st.divider()
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()
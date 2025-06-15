# Multi-user persistent storage version with Supabase
import streamlit as st
import os
import tempfile
import cohere
from openai import OpenAI
from dotenv import load_dotenv
# ...existing imports...
try:
    from supabase import create_client, Client
    import hashlib
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    st.warning("Supabase not installed. Install with: pip install supabase")

# ...existing code...

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
-- Run this SQL in your Supabase SQL Editor:

CREATE TABLE IF NOT EXISTS documents (
  id BIGSERIAL PRIMARY KEY,
  user_id TEXT NOT NULL,
  doc_id TEXT UNIQUE NOT NULL,
  content TEXT NOT NULL,
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Add indexes for performance
CREATE INDEX IF NOT EXISTS idx_documents_user_id ON documents(user_id);
CREATE INDEX IF NOT EXISTS idx_documents_content ON documents USING gin(to_tsvector('english', content));

-- Enable Row Level Security (optional - for better security)
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;

-- Simple policy that allows all operations (you can restrict this later)
CREATE POLICY IF NOT EXISTS "Enable all operations for authenticated users" 
ON documents FOR ALL 
TO authenticated 
USING (true) 
WITH CHECK (true);

-- Policy for anonymous users (if using anon key)
CREATE POLICY IF NOT EXISTS "Enable all operations for anon users" 
ON documents FOR ALL 
TO anon 
USING (true) 
WITH CHECK (true);
                """, language="sql")
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

# ...existing code...

# Main app modifications
st.title("🔮 Coherechat - Multi-User Cloud Storage")
st.caption("Persistent storage with user isolation • Up to 10 users")

# Initialize services
init_session_state()
supabase = init_supabase()

# User authentication
user_id = get_user_id()
if not user_id:
    st.stop()

# ...existing API key code...

# Sidebar modifications
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
    
    if storage_mode == "Cloud Persistent" and not supabase:
        st.error("Supabase not configured")
        storage_mode = "Local Session"
    
    st.header("📁 Data Management")
    
    # Load user's cloud data
    if storage_mode == "Cloud Persistent" and supabase:
        if st.button("📥 Load My Cloud Data"):
            with st.spinner("Loading your documents..."):
                cloud_docs = load_from_supabase(supabase, user_id)
                if cloud_docs:
                    # Create vector store from cloud documents
                    if cohere_key:
                        embeddings = CohereEmbeddings(
                            model="embed-english-v3.0",
                            cohere_api_key=cohere_key
                        )
                        st.session_state.local_vectorstore = FAISS.from_documents(cloud_docs, embeddings)
                        st.success(f"✅ Loaded {len(cloud_docs)} documents from cloud")
                    else:
                        st.error("Cohere API key required for embeddings")
                else:
                    st.info("No documents found in cloud storage")
        
        # Delete user data
        if st.button("🗑️ Delete My Cloud Data", type="secondary"):
            if st.checkbox("I understand this will delete all my data"):
                try:
                    supabase.table("documents").delete().eq("user_id", user_id).execute()
                    st.success("✅ All your data has been deleted")
                    st.session_state.clear()
                    st.rerun()
                except Exception as e:
                    st.error(f"Deletion failed: {e}")
    
    # ...existing tabs code...
    
    # Modified file processing
    with tab1:
        # ...existing file upload code...
        
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
                        # Get documents from vectorstore
                        all_docs = []
                        for file in uploaded_files:
                            docs = vectorstore.similarity_search("", k=1000, filter={"source_file": file.name})
                            all_docs.extend(docs)
                        
                        save_to_supabase(supabase, all_docs, user_id)
                    
                    st.rerun()

# ...existing chat interface code...

# Enhanced footer
st.markdown("---")
if storage_mode == "Cloud Persistent":
    st.caption("🔮 Multi-User Cloud Storage • Data persists across sessions • User isolated")
else:
    st.caption("🔮 Local Session Storage • Data cleared when you leave")
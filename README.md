# 🔮 Coherechat - Advanced AI Chat & Document Processing System

A comprehensive multi-modal AI system featuring OpenAI + Cohere integration, local & cloud document processing, web search capabilities, and intelligent RAG (Retrieval Augmented Generation) pipelines.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://mandip.streamlit.app/)

## 🚀 Features Overview

### **Multiple Chat Applications**
- **Simple Chat** (`streamlit_app.py`) - OpenAI primary with Cohere fallback
- **Smart Document Chat** (`smart_document_chat.py`) - Local RAG with web fallback
- **Hybrid Chat** (`cohere_hybrid_chat.py`) - Local + Cohere datasets + URLs
- **Cohere Playground** (`cohere_playground.py`) - Experiment with all Cohere APIs

### **Advanced Document Processing**
- **PDF Upload & Processing** - Automatic text extraction and chunking
- **URL Content Extraction** - Web scraping and content processing
- **Local Vector Storage** - FAISS-based similarity search (no cloud required)
- **Cohere Platform Integration** - Access datasets uploaded to Cohere Dashboard
- **Multi-source Search** - PDFs + URLs + Web + Platform datasets

### **Intelligent AI Features**
- **Dual AI Providers** - OpenAI (primary) + Cohere (fallback) for reliability
- **Advanced Embeddings** - Cohere embed-english-v3.0 (1024 dimensions)
- **Smart Reranking** - Cohere rerank-english-v3.0 for improved relevance
- **Fallback Chain** - Documents → Platform datasets → Web search
- **Source Attribution** - Always know where answers come from

## 🎯 Application Guide

### **1. Simple Chat (`streamlit_app.py`)**
General purpose chatbot with dual AI providers
```bash
streamlit run streamlit_app.py
```
**Features:**
- OpenAI GPT-3.5-turbo as primary
- Cohere Command-R+ as automatic fallback
- Real-time provider switching
- Visual status indicators

### **2. Smart Document Chat (`smart_document_chat.py`)**
Document-focused RAG with web fallback
```bash
streamlit run smart_document_chat.py
```
**Features:**
- PDF upload and processing
- Local FAISS vector storage
- DuckDuckGo web search fallback
- Source file tracking

### **3. Hybrid Chat (`cohere_hybrid_chat.py`)**
Most advanced - combines all data sources
```bash
streamlit run cohere_hybrid_chat.py
```
**Features:**
- Local PDF processing
- URL content extraction
- Cohere platform datasets
- Flexible search strategies
- JSONL export for Cohere dashboard

### **4. Cohere Playground (`cohere_playground.py`)**
Experiment with Cohere's full API suite
```bash
streamlit run cohere_playground.py
```
**Experiments Available:**
- 💬 Chat - Interactive conversation
- 🔍 Embed & Search - Semantic search
- 📊 Classify - Text classification
- 📝 Summarize - Text summarization
- 🔄 Rerank - Search result improvement
- ✨ Generate - Creative text generation

## 🔧 Setup Instructions

### **1. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **2. Configure API Keys**

**Option A: Environment Variables**
```bash
export OPENAI_API_KEY="your-openai-key"
export COHERE_API_KEY="your-cohere-key"
```

**Option B: Streamlit Secrets**
Create `.streamlit/secrets.toml`:
```toml
[openai]
OPENAI_API_KEY = "your-openai-key"

[cohere]
COHERE_API_KEY = "your-cohere-key"
```

**Option C: In-App Configuration**
Enter keys directly in the app interface (session-based)

### **3. Get API Keys**

**OpenAI API Key:**
1. Visit [OpenAI Platform](https://platform.openai.com/api-keys)
2. Create account or sign in
3. Generate new API key

**Cohere API Key:**
1. Visit [Cohere Dashboard](https://dashboard.cohere.ai/api-keys)
2. Create account or sign in
3. Generate new API key

## 📊 Data Sources & Processing

### **Local Files**
- **Supported Formats:** PDF
- **Processing:** PyPDF text extraction → RecursiveCharacterTextSplitter (1000 chars, 200 overlap)
- **Storage:** FAISS vector store (local, no cloud required)
- **Embeddings:** Cohere embed-english-v3.0

### **URLs**
- **Supported:** Any web URL
- **Processing:** WebBaseLoader → HTML content extraction
- **Integration:** Combined with local files in same vector store

### **Cohere Platform Datasets**
- **Upload:** .csv or .jsonl files to Cohere Dashboard
- **Access:** Direct API integration
- **Export:** Auto-generate JSONL from processed files

### **Web Search Fallback**
- **Engine:** DuckDuckGo Search
- **Trigger:** When no relevant documents found
- **Integration:** Seamless fallback chain

## 🔍 RAG Pipeline Architecture

```
📁 Input Sources
├── PDF Files → PyPDFLoader
├── URLs → WebBaseLoader  
├── Cohere Datasets → API
└── Web Search → DuckDuckGo

⬇️

✂️ Text Processing
├── RecursiveCharacterTextSplitter
├── Chunk Size: 1000 chars
├── Overlap: 200 chars
└── Metadata Preservation

⬇️

🧮 Embedding & Storage
├── Cohere embed-english-v3.0
├── 1024-dimensional vectors
├── FAISS local storage
└── Source tracking

⬇️

🔍 Query Processing
├── Query → Cohere embeddings
├── Similarity search (top 10)
├── Cohere rerank-english-v3.0
└── Top 5 relevant chunks

⬇️

🤖 Response Generation
├── OpenAI GPT-3.5-turbo (primary)
├── Cohere Command-R+ (fallback)
├── Context-aware prompting
└── Source attribution
```

## 🛠 Technical Dependencies

### **Core AI Libraries**
- `openai>=1.0.0` - Primary chat completions
- `cohere==5.11.4` - Embeddings, reranking, fallback chat
- `langchain==0.3.12` - RAG framework and document processing

### **Vector Storage & Search**
- `faiss-cpu>=1.7.0` - Local similarity search
- `numpy>=1.24.0` - Vector operations

### **Document Processing**
- `pypdf>=3.0.0` - PDF text extraction
- `beautifulsoup4>=4.12.0` - HTML parsing for URLs
- `requests>=2.31.0` - HTTP requests

### **Search & Tools**
- `duckduckgo-search>=7.0.0` - Web search fallback
- `streamlit==1.40.2` - Web interface

## 🎛 Search Strategies

### **Local First**
1. Search uploaded PDFs/URLs
2. Fallback to Cohere datasets
3. Final fallback to web search

### **Cohere Dataset First**
1. Search Cohere platform datasets
2. Fallback to local files
3. Final fallback to web search

### **Both + Combine**
1. Search both local and platform datasets
2. Combine and rank results
3. Web search if no results

## 🔐 Security & Privacy

- **Local Processing:** Documents processed locally with FAISS
- **API Keys:** Secure environment variable or secrets management
- **No Data Persistence:** Chat history cleared on session end
- **Source Attribution:** Always shows data source for transparency

## 🚀 Deployment

### **Local Development**
```bash
streamlit run [app_name].py
```

### **Streamlit Cloud**
1. Push to GitHub repository
2. Connect to Streamlit Cloud
3. Add API keys to secrets
4. Deploy automatically

## 📝 Usage Examples

### **Document Q&A**
1. Upload PDFs or add URLs
2. Process documents
3. Ask questions about content
4. Get contextual answers with sources

### **Multi-source Research**
1. Upload local documents
2. Connect Cohere datasets
3. Enable web search fallback
4. Ask complex research questions

### **AI Experimentation**
1. Use Cohere Playground
2. Test different models and parameters
3. Compare embedding techniques
4. Experiment with classification and summarization

## 🔄 Recent Updates

- ✅ Added OpenAI + Cohere dual provider system
- ✅ Implemented local FAISS vector storage (removed Qdrant dependency)
- ✅ Added URL content processing
- ✅ Integrated Cohere platform datasets
- ✅ Created comprehensive playground for Cohere APIs
- ✅ Enhanced RAG pipeline with reranking
- ✅ Added flexible search strategies
- ✅ Implemented graceful fallback chains

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Make changes
4. Test thoroughly
5. Submit pull request

## 📄 License

This project is licensed under the terms specified in the LICENSE file.

---

**Made with ❤️ using Streamlit, OpenAI, and Cohere**



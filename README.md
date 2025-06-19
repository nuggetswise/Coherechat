# Cohere RAG Chat & Compensation Intelligence Platform

## Overview
This application is an AI-powered chat and analytics platform for compensation data. It combines Retrieval-Augmented Generation (RAG), semantic search, document upload, and web search fallback to answer compensation-related questions with transparency and evaluation.

- **Audience:**
  - **Developers:** Learn about the architecture, tech stack, and logic for extending or maintaining the system.
  - **Non-technical users/Stakeholders:** Understand how the system works, what happens when you ask a question, and how AI is used responsibly.

---

## End-to-End Flow (Non-Technical)

1. **You ask a question** (e.g., "What's the typical salary for a Senior Engineer?") in the chat interface.
2. **The system searches its internal database** of real compensation records for relevant matches using advanced AI-powered search.
3. **If you upload documents** (like salary surveys), those are also searched for relevant information.
4. **If nothing relevant is found**, the system can search the web for up-to-date market data (if enabled).
5. **The AI reviews all the information found** and writes a clear, well-cited answer, explaining where the data came from.
6. **The AI also evaluates its own answer** for relevance, accuracy, and clarity, and shows you a transparent scorecard.
7. **You see the answer, the sources used, and an AI-powered evaluation**—all in one place.

---

## End-to-End Flow (Technical)

1. **User Input:**
   - User submits a question via the Streamlit chat UI (`pages/5_RAG_Chat.py`).

2. **Retrieval Pipeline:**
   - **Semantic Search:**
     - Query is embedded using Cohere Embeddings (`langchain_cohere.CohereEmbeddings`).
     - Internal DB (`data/Compensation Data.csv`) is loaded and indexed as a FAISS vectorstore.
     - Similarity search retrieves top-k relevant records.
   - **Document Upload:**
     - Uploaded files are parsed (PDF, CSV, DOCX, TXT, images with OCR) and indexed similarly.
   - **Reranking:**
     - All candidate results are reranked using Cohere's rerank model for best relevance.
   - **Web Search Fallback:**
     - If no relevant results or low relevance, triggers DuckDuckGo web search and reranks web snippets.

3. **Prompt Construction:**
   - Context from all sources is stitched together.
   - A cross-source prompt is generated using templates in `prompts_and_logic.py` (see `RAGConfiguration.get_cross_source_prompt`).

4. **LLM Response Generation:**
   - The prompt and context are sent to Cohere's `command-r-plus` model (or OpenAI as fallback).
   - The LLM generates a response, citing sources and synthesizing data.

5. **AI-Powered Evaluation:**
   - After each response, an evaluation prompt (from `RAGConfiguration.get_simple_rag_evaluation_prompt`) is sent to the LLM.
   - The LLM returns a JSON with scores and feedback for relevance, factual accuracy, groundedness, and more.
   - Results are displayed in the UI with full feedback (no hardcoding).

6. **UI & Transparency:**
   - The chat UI shows the answer, sources used, and the AI evaluation summary.
   - Users can see strengths and areas for improvement for each answer.

---

## Tech Stack
- **Frontend/UI:** Streamlit
- **Vector Search:** FAISS (with Cohere Embeddings)
- **LLM:** Cohere (command-r-plus), OpenAI (fallback)
- **Reranking:** Cohere rerank model
- **Web Search:** DuckDuckGo (via LangChain tool)
- **Document Parsing:** LangChain loaders, OCR (pytesseract, pdf2image, PIL)
- **Configuration & Prompts:** Centralized in `prompts_and_logic.py`
- **Evaluation:** LLM-based, using custom evaluation prompts
- **OCR for Documents:** Using pytesseract, pdf2image, and PIL for extracting text from images and PDFs
- **Database Compatibility:** CSV for straightforward data management
- **Patches:** Custom patches for Cohere types via `patches/cohere_types_patch.py`

---

## Key Files & Structure
- `pages/5_RAG_Chat.py` — Main chat app logic and UI
- `data/Compensation Data.csv` — Internal compensation database
- `prompts_and_logic.py` — All prompt templates, config, and logic for RAG and evaluation
- `embedding_utils.py` — Utilities for managing embeddings and vector operations
- `rag_evaluator.py` — Custom evaluation logic for RAG responses
- `chroma_db/` — Vectorstore index files (can be used as alternative to FAISS)
- `patches/` — Contains patches for third-party libraries to enhance functionality
- `dashboard.py` — Main dashboard for navigation between different app components

---

## Prompts Used
- **Cross-Source Answer Prompt:**
  - Synthesizes info from DB, uploads, and web, with clear instructions for citation and comparison.
- **Evaluation Prompt:**
  - Asks the LLM to rate the answer on relevance, factual accuracy, groundedness, and provide strengths/areas for improvement, returning results as JSON.
- **Web Search Prompt:**
  - Used to generate and rerank web search queries and snippets.
- **Detailed examples can be found in `prompts_and_logic.py`**

---

## Related Components
- **Compensation Planner (`pages/4_Compensation_Planner.py`):**
  - An agentic approach to compensation planning using the same database
  - Employs query decomposition and multi-tool analysis for more complex recommendations
- **Agent Evaluation Dashboard (`pages/6_Agent_Evaluation_Dashboard.py`):**
  - Evaluates the performance of AI recommendations
  - Provides insights into model performance and areas for improvement
- **Universal AI Evaluator (`pages/8_Universal_AI_Evaluator.py`):**
  - A more general-purpose evaluation tool that can assess various AI outputs

---

## How to Run Locally
1. Clone the repo and ensure `data/Compensation Data.csv` is present.
2. Install requirements: `pip install -r requirements.txt`
3. Set your Cohere and OpenAI API keys in your environment or Streamlit secrets.
4. Run: `streamlit run pages/5_RAG_Chat.py --server.port 8505`
5. Open [http://localhost:8505](http://localhost:8505) in your browser.

---

## Path Resolution
The system uses a robust path resolution mechanism to find the compensation database:
```python
# Get the absolute path to the current file (5_RAG_Chat.py)
current_file = os.path.abspath(__file__)
# Get the directory containing the current file (pages/)
current_dir = os.path.dirname(current_file)
# Get the project root directory (one level up from pages/)
project_root = os.path.dirname(current_dir)
# Construct the absolute path to the compensation data file
data_path = os.path.join(project_root, "data", "Compensation Data.csv")
```
This ensures the app works in both local and production environments.

---

## How to Deploy
- Make sure `data/Compensation Data.csv` is included in your deployment.
- Deploy to Streamlit Community Cloud or your own server.
- The app will use the internal DB for all retrieval and RAG operations.
- For multiple apps from the same repo, point each deployment to a different script (e.g., `pages/5_RAG_Chat.py` vs `pages/4_Compensation_Planner.py`).

---

## FAQ
- **What if the database has no relevant data?**
  - The app will fall back to web search and clearly indicate the source.
- **Are evaluation scores hardcoded?**
  - No, all evaluation scores and feedback are generated by the LLM in real time.
- **Can I add more data?**
  - Yes, update `Compensation Data.csv` and restart the app, or upload documents via the UI.
- **How does OCR work in the app?**
  - OCR is conditionally available if pytesseract and other dependencies are installed. It extracts text from images and PDFs.
- **Why does the system use a mix of FAISS and Chroma?**
  - FAISS is used for in-memory vector search, while Chroma provides persistence. The system is designed to work with either.

---

## Contact & Support
For questions or contributions, please contact the development team or open an issue in this repository.

---

## 🧠 System Logic, Thresholds, and Configuration (No Code Access Needed)

### What Happens When You Ask a Question?
1. **Your question is converted to an AI-friendly format** using advanced language models (embeddings).
2. **The system searches its internal compensation database** for the most relevant records using semantic similarity (not just keyword matching).
3. **If you upload documents** (PDFs, Word, CSV, images), those are also searched using the same AI-powered approach.
4. **If the system can't find a strong match internally,** it automatically searches the web for up-to-date information and uses AI to filter and summarize the best results.
5. **The AI combines all relevant information** (from the database, your uploads, and the web) and writes a clear, well-cited answer.
6. **Every answer is evaluated by the AI itself** for relevance, factual accuracy, and clarity, and you see a transparent scorecard for each response.

---

### 🔍 How Search & Retrieval Works
- **Embeddings:**
  - The system uses Cohere's state-of-the-art language models to turn your question and all documents into mathematical vectors that capture meaning.
  - This allows the system to find relevant answers even if the wording is different.
- **Semantic Search:**
  - Your question is compared to all records in the internal database and any uploaded documents.
  - Only results with a similarity score above a certain threshold (default: **0.1**) are considered relevant.
- **Reranking:**
  - The most relevant results are further reranked by another AI model to ensure the best matches are shown first.
  - Only results with a rerank score above **0.1** are included in the answer.
  - If the best result's score is below **0.3**, the system will also search the web for more information.
- **Web Search Fallback:**
  - If the internal sources are weak or missing, the system uses DuckDuckGo to search the web, then applies the same AI reranking to web results.

---

### 🤖 How the AI Answers Your Question
- **Prompt Engineering:**
  - The system uses carefully crafted instructions (prompts) to tell the AI how to combine, compare, and cite information from all sources.
  - The AI is instructed to:
    - Synthesize information from the database, uploads, and web
    - Compare and contrast data
    - Highlight agreements and discrepancies
    - Clearly cite where each piece of information comes from
    - Provide actionable, easy-to-understand insights
- **LLM Models Used:**
  - Cohere's `command-r-plus` is the primary model for generating answers and evaluations.
  - OpenAI's GPT-3.5-turbo is used as a fallback if needed.

---

### 📊 How AI Evaluation Works
- **After every answer,** the system asks the AI to rate its own response on:
  - **Relevance:** How well did the answer match your question?
  - **Factual Accuracy:** Is the answer supported by the retrieved information?
  - **Groundedness:** Does the answer stick to the facts, or does it make things up?
  - **Clarity:** Is the answer easy to understand and well-structured?
- **The AI returns a score (1-10) and detailed feedback for each dimension.**
- **No scores or feedback are hardcoded**—everything is generated live by the AI for each answer.
- **You see a transparent evaluation table and can expand to read the full feedback.**

---

### ⚙️ What Can Be Tweaked (Configuration)
- **All thresholds, model choices, and prompts are configurable by the system owner.**
- **You can adjust:**
  - The minimum similarity score for search results to be considered relevant
  - The rerank score required for a result to be included
  - The threshold for when to trigger a web search fallback
  - The number of top results to retrieve and rerank
  - The exact instructions (prompts) given to the AI for both answering and evaluation
  - The choice of AI models (Cohere, OpenAI, etc.)
- **These settings are typically managed in a configuration file by your technical team.**

---

### 🛡️ Privacy & Data Handling
- **Your uploaded documents and questions are only used for your session.**
- **The internal compensation database is never exposed directly—only summarized, relevant information is shown.**
- **Web search results are filtered and summarized by the AI before being included in any answer.**

---

### 📝 Example Prompts (Instructions Given to the AI)
- **Answer Generation Prompt:**
  - "You are a knowledgeable compensation assistant. Use the following information from the internal database, uploaded documents, and web search to answer the user's question. Compare, contrast, and cite your sources."
- **Evaluation Prompt:**
  - "Evaluate the following answer for relevance, factual accuracy, groundedness, and clarity. Provide a score (1-10) and a brief explanation for each."

---

### 💡 Summary for All Users
- **You ask a question.**
- **The system searches everywhere it can (internal DB, uploads, web) using advanced AI.**
- **The AI writes a clear, well-cited answer.**
- **The AI then evaluates its own answer and shows you a transparent scorecard.**
- **All thresholds and logic can be tuned by your technical team to fit your needs.**

---

## Integration with Other Modules

The RAG Chat system is part of a larger ecosystem of AI tools in this workspace:

1. **Multi-User Chat Support** - The system can be extended to support multiple users with the framework in `multi_user_chat.py`.

2. **Cohere Cloud RAG** - Direct integration with Cohere's cloud RAG capabilities through `cohere_cloud_rag.py` for alternative deployment options.

3. **Agent-Based Reasoning** - The `ragagentcohere.py` module provides agent-based reasoning that can be integrated with the RAG pipeline.

4. **Universal AI Evaluation** - The `universal_ai_evaluator.py` offers more extensive evaluation capabilities beyond the built-in evaluator.

5. **Customizable Patches** - The system includes patches in the `patches/` directory to enhance third-party libraries' functionality.

The modular design allows you to use RAG Chat standalone or as part of a larger compensation intelligence platform.

---

## Detailed Thresholds & Configuration Values

This section explains all the specific thresholds and configuration values used in the RAG system, and how they can be adjusted.

### Semantic Search (Cohere Embeddings)
- **Model**: `embed-english-v3.0` (Cohere's embedding model)
- **Dimensions**: 1024 (default for the model)
- **Top-k Records**: 5 (default) - This means the system retrieves the 5 most similar records from the database
  - Why 5 and not 10? We've found that 5 records typically provide sufficient context without overwhelming the LLM with too much information
  - This can be adjusted in `search_multi_source()` function by changing the `top_k=5` parameter
- **Similarity Threshold**: 0.1 - Records with a similarity score below this threshold are discarded
  - Defined in `prompts_and_logic.py` as `SEMANTIC_SIMILARITY_THRESHOLD = 0.1`

### Reranking (Cohere Rerank)
- **Model**: `rerank-english-v3.0`
- **Relevance Threshold**: 0.1 - Results with a rerank score below this are excluded from the final context
  - Configured in the reranking logic: `if result.relevance_score > 0.1:`
- **Web Search Fallback Threshold**: 0.3 - If the top reranked result has a score below 0.3, web search is triggered
  - Configured in search logic: `if reranked_results and reranked_results[0].metadata.get("rerank_score", 0) < 0.3:`

### Web Search (DuckDuckGo)
- **Number of Results**: 5 search results from DuckDuckGo
  - Configured as `num_results=5` in the DuckDuckGoSearchRun initialization
- **Web Chunk Relevance Threshold**: 0.15 - Web chunks with a rerank score below this are excluded
  - Found in web_search_fallback function: `if result.relevance_score > 0.15:`

### LLM Response Generation
- **LLM Model (Primary)**: Cohere `command-r-plus`
- **LLM Model (Fallback)**: OpenAI `gpt-3.5-turbo`
- **Temperature**: 0.3 (for both models) - Lower temperature for more focused, deterministic responses
- **Max Tokens**: 1200 for Cohere, 1000 for OpenAI
  - Configured in the response generation function

### Content Processing
- **Chunk Size**: 1000 characters - For splitting documents during processing
- **Chunk Overlap**: 200 characters - Overlap between consecutive chunks
  - Configured in the document processing logic

### How to Adjust These Thresholds
All these thresholds can be modified in the following locations:
1. For core RAG configuration: `prompts_and_logic.py` contains the central configuration
2. For search and retrieval: `search_multi_source()` function in `pages/5_RAG_Chat.py`
3. For web search: `web_search_fallback()` function in `pages/5_RAG_Chat.py`

**Example: Changing Top-k Records**
```python
# In search_multi_source() function
def search_multi_source(query, db_vectorstore, uploads_vectorstore, cohere_client, top_k=10):  # Change from 5 to 10
    # ...existing code...
```

**Example: Changing Reranking Threshold**
```python
# In search_multi_source() function
if result.relevance_score > 0.2:  # Change from 0.1 to 0.2
    # ...existing code...
```

**Example: Disabling Web Search Fallback**
Simply uncheck the "🌐 Enable web search fallback" option in the sidebar.

---

## Threshold Justifications & Impact Analysis

Understanding why specific threshold values were chosen and what happens when you adjust them is crucial for optimizing system performance.

### Semantic Search Threshold (0.1)
- **Justification:** The 0.1 similarity threshold was chosen to be intentionally permissive. Cohere embeddings typically produce higher similarity scores for truly relevant content, but we don't want to exclude potentially useful but slightly tangential information at this early stage.
- **If increased:** More stringent filtering would occur, potentially excluding relevant information that uses different terminology from the query.
- **If decreased:** More irrelevant results would be included, potentially introducing noise into the context and diluting the LLM's focus.

### Top-k Records (5)
- **Justification:** Five records balances context richness with LLM token limitations. In compensation data, we found that more than 5 records often introduced redundancy without adding significant value.
- **If increased:** More comprehensive context but higher token consumption and potentially slower responses. May introduce redundant information.
- **If decreased:** Faster and more focused responses but potentially missing important context or alternative perspectives.

### Reranking Relevance Threshold (0.1)
- **Justification:** This threshold was chosen to be permissive at first to allow for multiple relevant passages, with the understanding that reranking provides a more sophisticated relevance assessment than initial embedding similarity.
- **If increased:** The system would become more selective, potentially excluding relevant but unusual passages. This can be beneficial in domains requiring high precision.
- **If decreased:** More marginally relevant passages would be included, which can introduce noise but might capture edge cases.

### Web Search Fallback Threshold (0.3)
- **Justification:** The 0.3 threshold represents a "confidence boundary" - below this, the internal data is not considered sufficiently relevant to answer the query alone. This threshold is higher than the basic relevance threshold because web search is a more expensive operation that should only be triggered when truly necessary.
- **If increased:** Web search would trigger more frequently, potentially providing more up-to-date information but increasing API calls and response time.
- **If decreased:** Web search would be used more sparingly, leading to more reliance on internal data and potentially outdated information for time-sensitive queries.

### Web Chunk Relevance Threshold (0.15)
- **Justification:** This is set higher than the basic relevance threshold (0.1) because web content tends to be noisier and less structured than internal database entries. The higher threshold helps filter out irrelevant web content.
- **If increased:** Web search results would be more focused but potentially miss useful tangential information.
- **If decreased:** More diverse web content would be included, potentially introducing noise but capturing more perspectives.

### LLM Temperature (0.3)
- **Justification:** A relatively low temperature was chosen to prioritize factual accuracy and consistency in compensation data, where precision is important. This reduces creativity in favor of reliability.
- **If increased:** Responses would be more creative and diverse but potentially less factual or consistent.
- **If decreased:** Responses would be more deterministic and focused but potentially more rigid or formulaic.

### Document Chunk Size (1000) and Overlap (200)
- **Justification:** These values balance granularity with context preservation. 1000 characters is typically enough to capture a complete compensation record without splitting it unnaturally, while 200 characters of overlap ensures concepts that span chunk boundaries aren't lost.
- **If increased:** Larger chunks would preserve more context but reduce retrieval granularity. Larger overlap would improve context preservation but increase storage requirements.
- **If decreased:** Smaller chunks would improve retrieval precision but might lose important context. Smaller overlap would reduce storage requirements but could lose cross-chunk concepts.

### Performance Optimization Tips

These thresholds can be adjusted based on specific use cases:

- **For more precision:** Increase the relevance thresholds (0.1 → 0.2) and reduce the top-k records (5 → 3)
- **For more recall:** Decrease the relevance thresholds (0.1 → 0.05) and increase the top-k records (5 → 8)
- **For faster responses:** Reduce the top-k records (5 → 3) and increase all relevance thresholds slightly
- **For more comprehensive answers:** Increase the top-k records (5 → 8) and reduce the web search fallback threshold (0.3 → 0.25)

The system is designed to be robust to reasonable adjustments in these thresholds, so experimentation is encouraged based on your specific compensation data characteristics and user needs.

---



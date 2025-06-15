import streamlit as st
import cohere
import os
import json
import pandas as pd
import tempfile
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Cohere Dataset Lab", page_icon="🧪", layout="wide")

st.title("🧪 Cohere Dataset Laboratory")
st.caption("Create, manage, and experiment with all Cohere dataset types")

# API Key Setup
cohere_api_key = os.getenv("COHERE_API_KEY")
if not cohere_api_key and hasattr(st, "secrets"):
    if "COHERE_API_KEY" in st.secrets:
        cohere_api_key = st.secrets["COHERE_API_KEY"]
    elif "cohere" in st.secrets and "COHERE_API_KEY" in st.secrets.cohere:
        cohere_api_key = st.secrets.cohere["COHERE_API_KEY"]

if not cohere_api_key:
    cohere_api_key = st.text_input("Cohere API Key", type="password")

if not cohere_api_key:
    st.warning("Please provide your Cohere API key")
    st.stop()

co = cohere.Client(api_key=cohere_api_key)

# Initialize session state
if 'created_datasets' not in st.session_state:
    st.session_state.created_datasets = []

# Sidebar for experiment selection
with st.sidebar:
    st.header("🎯 Dataset Types")
    
    dataset_mode = st.radio(
        "Choose Your Experiment:",
        [
            "📊 Dataset Creator",
            "💬 Chat Datasets", 
            "🏷️ Classify Datasets",
            "🔄 Rerank Datasets", 
            "🔮 Embed Datasets",
            "🧪 Live API Testing"
        ]
    )
    
    st.markdown("---")
    st.header("📚 Dataset Info")
    st.markdown("""
    **Chat**: Conversation datasets for fine-tuning
    **Classify**: Text + labels for classification  
    **Rerank**: Query + relevant/irrelevant passages
    **Embed**: Text for embedding/search
    """)

# Helper functions for dataset creation
def create_chat_dataset_sample():
    """Generate sample chat dataset"""
    return [
        {
            "messages": [
                {"role": "user", "content": "What is machine learning?"},
                {"role": "assistant", "content": "Machine learning is a subset of AI that enables computers to learn and improve from data without being explicitly programmed."}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "How does neural network work?"},
                {"role": "assistant", "content": "Neural networks process information through interconnected nodes (neurons) that transform input data through weighted connections and activation functions."}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "What's the difference between supervised and unsupervised learning?"},
                {"role": "assistant", "content": "Supervised learning uses labeled data to train models, while unsupervised learning finds patterns in unlabeled data."}
            ]
        }
    ]

def create_classify_dataset_sample():
    """Generate sample classification dataset"""
    return [
        {"text": "I love this product! It's amazing and works perfectly.", "label": "positive"},
        {"text": "This is terrible. Worst purchase ever.", "label": "negative"},
        {"text": "The item is okay, nothing special.", "label": "neutral"},
        {"text": "Fantastic quality and great customer service!", "label": "positive"},
        {"text": "Poor quality, broke after one day.", "label": "negative"},
        {"text": "Average product, does what it says.", "label": "neutral"},
        {"text": "Outstanding! Exceeded my expectations completely.", "label": "positive"},
        {"text": "Waste of money, doesn't work at all.", "label": "negative"}
    ]

def create_rerank_dataset_sample():
    """Generate sample rerank dataset"""
    return [
        {
            "query": "machine learning algorithms",
            "passages": [
                {"text": "Linear regression is a fundamental statistical method", "relevant": True},
                {"text": "Random forests combine multiple decision trees", "relevant": True}, 
                {"text": "The weather today is sunny and warm", "relevant": False},
                {"text": "Neural networks mimic brain structure", "relevant": True},
                {"text": "My favorite pizza topping is pepperoni", "relevant": False}
            ]
        },
        {
            "query": "python programming",
            "passages": [
                {"text": "Python is a high-level programming language", "relevant": True},
                {"text": "Pandas is a data manipulation library", "relevant": True},
                {"text": "Elephants are large mammals", "relevant": False},
                {"text": "Flask is a web framework for Python", "relevant": True},
                {"text": "The ocean is very deep", "relevant": False}
            ]
        }
    ]

def create_embed_dataset_sample():
    """Generate sample embedding dataset"""
    return [
        {"text": "Artificial intelligence is transforming industries"},
        {"text": "Machine learning enables computers to learn from data"},
        {"text": "Deep learning uses neural networks with multiple layers"},
        {"text": "Natural language processing helps computers understand text"},
        {"text": "Computer vision allows machines to interpret images"},
        {"text": "Data science combines statistics and programming"},
        {"text": "Big data requires specialized tools for processing"},
        {"text": "Cloud computing provides scalable infrastructure"}
    ]

def save_dataset_file(data, filename, format_type="jsonl"):
    """Save dataset in appropriate format"""
    if format_type == "jsonl":
        content = "\n".join([json.dumps(item) for item in data])
        return content, "application/json"
    elif format_type == "csv":
        df = pd.DataFrame(data)
        return df.to_csv(index=False), "text/csv"

# Main content area
if dataset_mode == "📊 Dataset Creator":
    st.header("🛠️ Dataset Creator")
    st.write("Create datasets for all Cohere use cases")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📋 Choose Dataset Type")
        
        dataset_type = st.selectbox(
            "Dataset Type:",
            ["Chat", "Classify", "Rerank", "Embed input"]
        )
        
        st.subheader("⚡ Quick Start")
        if st.button(f"Generate Sample {dataset_type} Dataset", type="primary"):
            if dataset_type == "Chat":
                sample_data = create_chat_dataset_sample()
                st.session_state[f'current_{dataset_type.lower()}_data'] = sample_data
            elif dataset_type == "Classify":
                sample_data = create_classify_dataset_sample()
                st.session_state[f'current_{dataset_type.lower()}_data'] = sample_data
            elif dataset_type == "Rerank":
                sample_data = create_rerank_dataset_sample()
                st.session_state[f'current_{dataset_type.lower()}_data'] = sample_data
            elif dataset_type == "Embed input":
                sample_data = create_embed_dataset_sample()
                st.session_state[f'current_{dataset_type.lower()}_data'] = sample_data
            
            st.success(f"✅ Generated sample {dataset_type} dataset!")
            st.rerun()
    
    with col2:
        st.subheader("📁 Manual Dataset Creation")
        
        if dataset_type == "Chat":
            st.write("**Chat Dataset Format:** Conversations between user and assistant")
            
            user_msg = st.text_input("User message:")
            assistant_msg = st.text_area("Assistant response:")
            
            if st.button("Add Conversation"):
                if 'current_chat_data' not in st.session_state:
                    st.session_state.current_chat_data = []
                
                conversation = {
                    "messages": [
                        {"role": "user", "content": user_msg},
                        {"role": "assistant", "content": assistant_msg}
                    ]
                }
                st.session_state.current_chat_data.append(conversation)
                st.success("Added conversation!")
                st.rerun()
        
        elif dataset_type == "Classify":
            st.write("**Classify Dataset Format:** Text with labels")
            
            text = st.text_area("Text to classify:")
            label = st.text_input("Label:")
            
            if st.button("Add Classification Example"):
                if 'current_classify_data' not in st.session_state:
                    st.session_state.current_classify_data = []
                
                example = {"text": text, "label": label}
                st.session_state.current_classify_data.append(example)
                st.success("Added classification example!")
                st.rerun()
        
        elif dataset_type == "Rerank":
            st.write("**Rerank Dataset Format:** Query with relevant/irrelevant passages")
            
            query = st.text_input("Query:")
            passage = st.text_area("Passage:")
            relevant = st.checkbox("Is this passage relevant?")
            
            if st.button("Add to Rerank Dataset"):
                if 'current_rerank_data' not in st.session_state:
                    st.session_state.current_rerank_data = []
                
                # Find existing query or create new one
                existing_query = None
                for item in st.session_state.current_rerank_data:
                    if item["query"] == query:
                        existing_query = item
                        break
                
                if existing_query:
                    existing_query["passages"].append({"text": passage, "relevant": relevant})
                else:
                    new_query = {
                        "query": query,
                        "passages": [{"text": passage, "relevant": relevant}]
                    }
                    st.session_state.current_rerank_data.append(new_query)
                
                st.success("Added passage to rerank dataset!")
                st.rerun()
        
        elif dataset_type == "Embed input":
            st.write("**Embed Dataset Format:** Text to be embedded")
            
            text = st.text_area("Text to embed:")
            
            if st.button("Add Text"):
                if 'current_embed input_data' not in st.session_state:
                    st.session_state['current_embed input_data'] = []
                
                example = {"text": text}
                st.session_state['current_embed input_data'].append(example)
                st.success("Added text to embed dataset!")
                st.rerun()
    
    # Display current dataset
    dataset_key = f'current_{dataset_type.lower().replace(" ", "_")}_data'
    if dataset_key in st.session_state and st.session_state[dataset_key]:
        st.subheader("📋 Current Dataset")
        
        data = st.session_state[dataset_key]
        st.write(f"**Items:** {len(data)}")
        
        # Show preview
        with st.expander("Preview Dataset"):
            for i, item in enumerate(data[:3]):  # Show first 3 items
                st.write(f"**Item {i+1}:**")
                st.json(item)
            if len(data) > 3:
                st.write(f"... and {len(data) - 3} more items")
        
        # Download options
        col1, col2 = st.columns(2)
        
        with col1:
            # JSONL download
            jsonl_content, mime_type = save_dataset_file(data, f"{dataset_type.lower()}_dataset", "jsonl")
            st.download_button(
                label="📥 Download as JSONL",
                data=jsonl_content,
                file_name=f"{dataset_type.lower()}_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl",
                mime=mime_type
            )
        
        with col2:
            # Clear dataset
            if st.button("🗑️ Clear Dataset"):
                del st.session_state[dataset_key]
                st.rerun()

elif dataset_mode == "💬 Chat Datasets":
    st.header("💬 Chat Dataset Experiments")
    
    # File upload for existing datasets
    uploaded_file = st.file_uploader("Upload Chat Dataset (JSONL)", type=['jsonl', 'json'])
    
    if uploaded_file:
        try:
            content = uploaded_file.read().decode('utf-8')
            if uploaded_file.name.endswith('.jsonl'):
                data = [json.loads(line) for line in content.strip().split('\n')]
            else:
                data = json.loads(content)
            
            st.success(f"✅ Loaded {len(data)} conversations")
            
            # Display sample conversation
            with st.expander("Sample Conversation"):
                if data:
                    st.json(data[0])
            
            # Chat simulation with dataset
            st.subheader("🎭 Simulate Chat with Dataset")
            
            user_input = st.text_input("Your message:")
            if user_input and st.button("Send"):
                # Find similar conversation in dataset
                best_match = None
                best_score = 0
                
                for conv in data:
                    for msg in conv.get('messages', []):
                        if msg.get('role') == 'user':
                            # Simple similarity check
                            user_words = set(user_input.lower().split())
                            msg_words = set(msg.get('content', '').lower().split())
                            score = len(user_words.intersection(msg_words)) / len(user_words.union(msg_words))
                            if score > best_score:
                                best_score = score
                                best_match = conv
                
                if best_match:
                    st.write("**Best matching conversation from dataset:**")
                    for msg in best_match['messages']:
                        role = "🧑" if msg['role'] == 'user' else "🤖"
                        st.write(f"{role} **{msg['role']}:** {msg['content']}")
                else:
                    st.write("No similar conversation found in dataset")
        
        except Exception as e:
            st.error(f"Error loading dataset: {e}")

elif dataset_mode == "🏷️ Classify Datasets":
    st.header("🏷️ Classification Dataset Experiments")
    
    # Sample classification experiments
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Dataset Analysis")
        
        # Create sample or upload
        if st.button("Load Sample Classification Dataset"):
            sample_data = create_classify_dataset_sample()
            
            # Analyze labels
            labels = [item['label'] for item in sample_data]
            label_counts = pd.Series(labels).value_counts()
            
            st.write("**Label Distribution:**")
            st.bar_chart(label_counts)
            
            # Show examples per label
            st.write("**Examples by Label:**")
            for label in label_counts.index:
                with st.expander(f"{label.title()} Examples"):
                    examples = [item['text'] for item in sample_data if item['label'] == label]
                    for example in examples[:3]:
                        st.write(f"• {example}")
    
    with col2:
        st.subheader("🧪 Live Classification Test")
        
        test_text = st.text_area("Test your classification:", 
                                value="This product is absolutely wonderful!")
        
        if st.button("Classify Text"):
            try:
                # Use sample data as examples for classification
                sample_data = create_classify_dataset_sample()
                
                examples = [{"text": item["text"], "label": item["label"]} for item in sample_data]
                
                response = co.classify(
                    inputs=[test_text],
                    examples=examples
                )
                
                prediction = response.classifications[0]
                st.write(f"**Prediction:** {prediction.prediction}")
                st.write(f"**Confidence:** {prediction.confidence:.3f}")
                
                # Show confidence for all labels
                st.write("**All Label Confidences:**")
                for label_pred in prediction.labels:
                    st.write(f"• {label_pred.label_name}: {label_pred.confidence:.3f}")
            
            except Exception as e:
                st.error(f"Classification failed: {e}")

elif dataset_mode == "🔄 Rerank Datasets":
    st.header("🔄 Rerank Dataset Experiments")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Rerank Testing")
        
        query = st.text_input("Search Query:", value="machine learning")
        
        documents = st.text_area(
            "Documents to rank (one per line):",
            value="""Machine learning algorithms learn patterns from data
Natural language processing helps computers understand text  
The weather is sunny today
Deep learning uses neural networks
Pizza is my favorite food
Data science combines statistics and programming
Birds can fly in the sky
Artificial intelligence mimics human intelligence"""
        ).strip().split('\n')
        
        if st.button("Rerank Documents"):
            try:
                response = co.rerank(
                    model="rerank-english-v3.0",
                    query=query,
                    documents=documents,
                    top_k=len(documents)
                )
                
                st.subheader("🎯 Reranked Results:")
                for i, result in enumerate(response.results):
                    relevance_color = "🟢" if result.relevance_score > 0.5 else "🟡" if result.relevance_score > 0.2 else "🔴"
                    st.write(f"{relevance_color} **{i+1}.** (Score: {result.relevance_score:.3f})")
                    st.write(f"   {result.document.text}")
            
            except Exception as e:
                st.error(f"Reranking failed: {e}")
    
    with col2:
        st.subheader("📊 Rerank Dataset Creator")
        
        if st.button("Generate Rerank Training Data"):
            sample_data = create_rerank_dataset_sample()
            
            st.write("**Sample Rerank Dataset:**")
            for i, item in enumerate(sample_data):
                with st.expander(f"Query {i+1}: {item['query']}"):
                    for passage in item['passages']:
                        relevance = "✅ Relevant" if passage['relevant'] else "❌ Not Relevant"
                        st.write(f"{relevance}: {passage['text']}")

elif dataset_mode == "🔮 Embed Datasets":
    st.header("🔮 Embedding Dataset Experiments")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 Text Embedding")
        
        texts = st.text_area(
            "Texts to embed (one per line):",
            value="""Artificial intelligence is the future
Machine learning transforms industries
Data science drives decisions
Programming creates solutions
Technology advances rapidly"""
        ).strip().split('\n')
        
        if st.button("Generate Embeddings"):
            try:
                response = co.embed(
                    texts=texts,
                    model="embed-english-v3.0",
                    input_type="search_document"
                )
                
                st.success(f"✅ Generated embeddings for {len(texts)} texts")
                st.write(f"**Embedding Dimension:** {len(response.embeddings[0])}")
                
                # Show embedding similarity matrix
                import numpy as np
                embeddings = np.array(response.embeddings)
                similarity_matrix = np.dot(embeddings, embeddings.T)
                
                st.write("**Similarity Matrix:**")
                similarity_df = pd.DataFrame(
                    similarity_matrix,
                    index=[f"Text {i+1}" for i in range(len(texts))],
                    columns=[f"Text {i+1}" for i in range(len(texts))]
                )
                st.dataframe(similarity_df, use_container_width=True)
            
            except Exception as e:
                st.error(f"Embedding failed: {e}")
    
    with col2:
        st.subheader("🔍 Semantic Search")
        
        search_query = st.text_input("Search Query:", value="artificial intelligence")
        
        if st.button("Search Embeddings") and 'embeddings' in locals():
            try:
                query_response = co.embed(
                    texts=[search_query],
                    model="embed-english-v3.0", 
                    input_type="search_query"
                )
                
                query_embedding = np.array(query_response.embeddings[0])
                
                # Calculate similarities
                similarities = np.dot(embeddings, query_embedding)
                
                # Rank results
                results = list(zip(texts, similarities))
                results.sort(key=lambda x: x[1], reverse=True)
                
                st.write("**Search Results:**")
                for i, (text, sim) in enumerate(results):
                    st.write(f"**{i+1}.** (Score: {sim:.3f}) {text}")
            
            except Exception as e:
                st.error(f"Search failed: {e}")

elif dataset_mode == "🧪 Live API Testing":
    st.header("🧪 Live API Testing")
    st.write("Test all Cohere APIs with your datasets")
    
    # Tab-based interface for different API tests
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "🏷️ Classify", "🔄 Rerank", "🔮 Embed"])
    
    with tab1:
        st.subheader("Chat API Testing")
        
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []
        
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
        
        if prompt := st.chat_input("Chat with Cohere"):
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.write(prompt)
            
            with st.chat_message("assistant"):
                response = co.chat(
                    model="command-r-plus",
                    message=prompt,
                    temperature=0.7
                )
                st.write(response.text)
                st.session_state.chat_messages.append({"role": "assistant", "content": response.text})
    
    with tab2:
        st.subheader("Classify API Testing")
        # Add classification testing interface
        
    with tab3:
        st.subheader("Rerank API Testing") 
        # Add rerank testing interface
        
    with tab4:
        st.subheader("Embed API Testing")
        # Add embedding testing interface

# Footer
st.markdown("---")
st.caption("🧪 Cohere Dataset Laboratory • Create • Experiment • Deploy")
import streamlit as st
import cohere
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Cohere Playground", page_icon="🧪")

st.title("🧪 Cohere Playground")
st.caption("Experiment with Cohere's full API capabilities")

# API Key Setup
cohere_api_key = os.getenv("COHERE_API_KEY") or st.text_input("Cohere API Key", type="password")

if not cohere_api_key:
    st.warning("Please provide your Cohere API key")
    st.stop()

co = cohere.Client(api_key=cohere_api_key)

# Sidebar for different experiments
experiment = st.sidebar.selectbox(
    "Choose Experiment",
    ["💬 Chat", "🔍 Embed & Search", "📊 Classify", "📝 Summarize", "🔄 Rerank", "✨ Generate"]
)

if experiment == "💬 Chat":
    st.header("Chat with Command-R+")
    
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

elif experiment == "🔍 Embed & Search":
    st.header("Embeddings & Semantic Search")
    
    # Document input
    documents = st.text_area(
        "Enter documents (one per line):",
        value="The sky is blue\nPython is a programming language\nMachine learning is fascinating\nCoffee helps me code\nStreamlit makes apps easy"
    ).split('\n')
    
    query = st.text_input("Search query:", value="programming")
    
    if st.button("Search") and query:
        # Get embeddings
        doc_embeddings = co.embed(
            texts=documents,
            model="embed-english-v3.0",
            input_type="search_document"
        ).embeddings
        
        query_embedding = co.embed(
            texts=[query],
            model="embed-english-v3.0",
            input_type="search_query"
        ).embeddings[0]
        
        # Calculate similarities
        import numpy as np
        similarities = []
        for doc_emb in doc_embeddings:
            similarity = np.dot(query_embedding, doc_emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb)
            )
            similarities.append(similarity)
        
        # Display results
        results = list(zip(documents, similarities))
        results.sort(key=lambda x: x[1], reverse=True)
        
        st.subheader("Search Results:")
        for doc, sim in results:
            st.write(f"**{sim:.3f}** - {doc}")

elif experiment == "📊 Classify":
    st.header("Text Classification")
    
    text = st.text_area(
        "Text to classify:",
        value="I love this new feature! It's amazing and works perfectly."
    )
    
    examples = [
        {"text": "This is terrible", "label": "negative"},
        {"text": "This is amazing", "label": "positive"},
        {"text": "I hate this", "label": "negative"},
        {"text": "I love this", "label": "positive"},
        {"text": "This is okay", "label": "neutral"},
        {"text": "Not bad", "label": "neutral"}
    ]
    
    if st.button("Classify"):
        response = co.classify(
            inputs=[text],
            examples=examples
        )
        
        prediction = response.classifications[0]
        st.write(f"**Prediction:** {prediction.prediction}")
        st.write(f"**Confidence:** {prediction.confidence:.3f}")

elif experiment == "📝 Summarize":
    st.header("Text Summarization")
    
    text = st.text_area(
        "Text to summarize:",
        value="""Artificial intelligence (AI) refers to the simulation of human intelligence in machines 
        that are programmed to think like humans and mimic their actions. The term may also be applied 
        to any machine that exhibits traits associated with a human mind such as learning and problem-solving. 
        The ideal characteristic of artificial intelligence is its ability to rationalize and take actions 
        that have the best chance of achieving a specific goal."""
    )
    
    length = st.selectbox("Summary length:", ["short", "medium", "long"])
    
    if st.button("Summarize"):
        response = co.summarize(
            text=text,
            length=length,
            format="paragraph"
        )
        st.write("**Summary:**")
        st.write(response.summary)

elif experiment == "🔄 Rerank":
    st.header("Document Reranking")
    
    query = st.text_input("Query:", value="machine learning")
    
    documents = st.text_area(
        "Documents to rerank:",
        value="""Deep learning is a subset of machine learning
Natural language processing helps computers understand text
Computer vision enables machines to see
Data science involves analyzing large datasets
Neural networks mimic brain functions
Statistics is fundamental to data analysis"""
    ).split('\n')
    
    if st.button("Rerank"):
        response = co.rerank(
            model="rerank-english-v3.0",
            query=query,
            documents=documents,
            top_k=len(documents)
        )
        
        st.subheader("Reranked Results:")
        for i, result in enumerate(response.results):
            st.write(f"**{i+1}.** (Score: {result.relevance_score:.3f}) {result.document.text}")

elif experiment == "✨ Generate":
    st.header("Text Generation")
    
    prompt = st.text_area(
        "Prompt:",
        value="Write a creative story about a robot learning to paint:"
    )
    
    temperature = st.slider("Temperature (creativity):", 0.0, 2.0, 0.8)
    max_tokens = st.slider("Max tokens:", 50, 1000, 200)
    
    if st.button("Generate"):
        response = co.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            model="command"
        )
        
        st.write("**Generated text:**")
        st.write(response.generations[0].text)

# Sidebar info
with st.sidebar:
    st.markdown("---")
    st.markdown("### 🧪 Experiments Available:")
    st.markdown("- **Chat**: Interactive conversation")
    st.markdown("- **Embed & Search**: Semantic search")
    st.markdown("- **Classify**: Text classification")
    st.markdown("- **Summarize**: Text summarization")
    st.markdown("- **Rerank**: Improve search results")
    st.markdown("- **Generate**: Creative text generation")
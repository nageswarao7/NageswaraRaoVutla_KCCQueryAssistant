import streamlit as st
from utils import (
    preprocess_kcc, 
    build_vector_store, 
    retrieve_context_with_score, 
    ask_gemma, 
    comprehensive_fallback_search
)

st.set_page_config(page_title="KCC Query Assistant", layout="wide")
st.title("🌾 KCC Query Assistant (Offline + Fallback)")
st.markdown("Ask natural language questions from the Kisan Call Center dataset using Gemma 2B.")

# Sidebar for configuration
st.sidebar.header("Configuration")
st.sidebar.markdown("### Similarity Threshold")
similarity_threshold = st.sidebar.slider(
    "Relevance threshold (lower = more strict)", 
    min_value=0.5, 
    max_value=2.0, 
    value=1.0, 
    step=0.1
)
st.sidebar.markdown("*If no local context meets this threshold, fallback search will be used.*")

# Data preprocessing section
st.header("📊 Data Setup")
col1, col2 = st.columns(2)

with col1:
    if st.button("🔄 Preprocess KCC Data"):
        with st.spinner("Processing KCC dataset..."):
            try:
                preprocess_kcc()
                st.success("✅ KCC data preprocessed successfully!")
            except Exception as e:
                st.error(f"❌ Error preprocessing data: {str(e)}")

with col2:
    if st.button("🗂️ Build Vector Store"):
        with st.spinner("Building vector embeddings..."):
            try:
                build_vector_store()
                st.success("✅ Vector store built successfully!")
            except Exception as e:
                st.error(f"❌ Error building vector store: {str(e)}")

# Query section
st.header("💬 Ask Your Agricultural Question")

# Sample queries for demonstration
sample_queries = [
    "How to manage drought stress in groundnut cultivation?",
    "What issues do sugarcane farmers in Maharashtra commonly face?",
    "Best fertilizers for wheat crop in Punjab",
    "How to prevent fungal diseases in tomato plants?",
    "Organic farming techniques for cotton",
    "Water management for rice cultivation",
    "Soil preparation for potato farming",
    "How to control aphids in mustard crop?",
    "Post-harvest storage techniques for grains"
]

# Query input
query = st.text_input(
    "Enter your agricultural query:", 
    placeholder="e.g., How to control pests in rice farming?"
)

# Add sample query buttons
st.markdown("**Try these sample queries:**")
cols = st.columns(3)
for i, sample in enumerate(sample_queries[:6]):  # Show first 6
    col_idx = i % 3
    with cols[col_idx]:
        if st.button(f"📝 {sample[:]}...", key=f"sample_{i}"):
            query = sample

# Process query
if query:
    st.markdown("---")
    
    with st.spinner("🔍 Searching for relevant information..."):
        # First, try to get context from local KCC data
        context, score = retrieve_context_with_score(
            query, 
            score_threshold=similarity_threshold
        )
        
        if context:
            # Found relevant local context
            st.success("✅ **Found relevant information in local KCC dataset**")
            
            # Display context quality
            quality = "High" if score < 0.7 else "Medium" if score < 1.2 else "Low"
            st.info(f"📊 Context relevance: **{quality}** (distance: {score:.3f})")
            
            # Get answer from local LLM
            with st.spinner("🤖 Generating answer using local Gemma model..."):
                try:
                    response = ask_gemma(query, context)
                    
                    st.subheader("🎯 Answer from KCC Dataset:")
                    st.markdown(response)
                    
                    # Show source context in expandable section
                    with st.expander("📖 View source context"):
                        st.text(context)
                        
                except Exception as e:
                    st.error(f"❌ Error getting response from local model: {str(e)}")
                    st.info("💡 Make sure Ollama is running with Gemma 2B model")
        
        else:
            # No relevant local context found - use fallback
            st.warning("⚠️ **No relevant local context found in KCC dataset**")
            st.info(f"Similarity threshold: {similarity_threshold} | Best match distance: {score:.3f}")
            
            # Try live internet search
            st.markdown("🌐 **Searching the internet for agricultural advice...**")
            
            with st.spinner("Performing live internet search..."):
                # Get API key from secrets or environment
                api_key = st.secrets.get("SERPAPI_API_KEY") or None
                
                try:
                    live_results = comprehensive_fallback_search(query, api_key)
                    
                    st.subheader("🌐 Live Internet Search Results:")
                    st.markdown(live_results)
                    
                    # Add disclaimer
                    st.caption(
                        "⚠️ **Disclaimer:** These results are from live internet search "
                        "and may not be verified agricultural advice. Please consult "
                        "local agricultural experts for specific recommendations."
                    )
                    
                except Exception as e:
                    st.error(f"❌ Fallback search failed: {str(e)}")
                    st.info(
                        "💡 **Suggestions:**\n"
                        "- Check your internet connection\n"
                        "- Verify SerpAPI key in secrets\n"
                        "- Try a different query\n"
                        "- Ask about topics covered in the KCC dataset"
                    )

# Instructions section
st.markdown("---")
with st.expander("ℹ️ How to use this assistant"):
    st.markdown("""
    ### 🚀 Getting Started:
    1. **Preprocess Data**: Click "Preprocess KCC Data" to prepare the dataset
    2. **Build Vector Store**: Click "Build Vector Store" to create searchable embeddings
    3. **Ask Questions**: Enter your agricultural query in natural language
    
    ### 🎯 How it works:
    - **Local Search**: First searches the KCC dataset for relevant answers
    - **Relevance Check**: Uses similarity threshold to ensure quality matches
    - **Fallback Search**: If no good local match, searches the internet
    - **Clear Indicators**: Shows whether answer is from KCC data or live search
    
    ### 🔧 Configuration:
    - Adjust similarity threshold in sidebar (lower = more strict matching)
    - Add SerpAPI key in Streamlit secrets for internet search fallback
    
    ### 📝 Sample Topics:
    - Pest control methods for specific crops
    - Disease management in agriculture  
    - Fertilizer recommendations
    - Drought and water management
    - Soil preparation techniques
    - Post-harvest storage methods
    """)

# Status indicators
st.sidebar.markdown("---")
st.sidebar.header("System Status")

# Check if files exist
import os
if os.path.exists('processed_docs.json'):
    st.sidebar.success("✅ Processed data available")
else:
    st.sidebar.warning("⚠️ Run data preprocessing first")

if os.path.exists('faiss_index.pkl'):
    st.sidebar.success("✅ Vector store ready")
else:
    st.sidebar.warning("⚠️ Build vector store first")

# API key status
api_key_status = st.secrets.get("SERPAPI_API_KEY") or None
if api_key_status:
    st.sidebar.success("✅ SerpAPI key configured")
else:
    st.sidebar.info("ℹ️ Add SerpAPI key for fallback search")
import streamlit as st
import sys
import os

# Configure page
st.set_page_config(
    page_title="Energy Data Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Add path to core modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
core_dir = os.path.join(parent_dir, '1_core')
sys.path.insert(0, core_dir)
sys.path.insert(0, parent_dir)

# Import core modules
from embedding_store import EmbeddingStore
from llm_reasoning import LLMReasoning

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Header
st.title("🤖 Energy Data Chatbot")
st.markdown("Ask questions about your energy consumption data")

# Sidebar
st.sidebar.title("Controls")
if st.sidebar.button("💥 Hard Reset (Fix All Errors)"):
    st.cache_resource.clear()
    if 'messages' in st.session_state:
        st.session_state.messages = []
    st.rerun()

# Initialize components (cached)
@st.cache_resource
def get_components():
    embedding_store = EmbeddingStore()
    llm = LLMReasoning()
    return embedding_store, llm

embedding_store, llm = get_components()

# Chat interface
st.markdown("### 💬 Chat")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about energy consumption..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Search for relevant data
                results = embedding_store.search(prompt, k=5)
                
                # --- GLOBAL CONTEXT INJECTION (Fix for "how many feeders" hallucinations) ---
                # Always fetch the plant summary to give the LLM the "big picture"
                try:
                    import sqlite3
                    import json
                    conn = sqlite3.connect(os.path.join(parent_dir, '4_data', 'data_prototype', 'metadata.db'))
                    cursor = conn.cursor()
                    cursor.execute("SELECT content, metadata FROM documents WHERE doc_type = 'plant_summary'")
                    summary_row = cursor.fetchone()
                    conn.close()
                    
                    if summary_row:
                        summary_content, summary_meta_str = summary_row
                        # Prepend summary to results so LLM sees it first
                        results.insert(0, {
                            "doc_id": "global_plant_summary",  # Added doc_id to prevent TypeError
                            "content": f"GLOBAL PLANT SUMMARY (ALWAYS USE THIS FOR TOTAL COUNTS):\n{summary_content}",
                            "metadata": json.loads(summary_meta_str),
                            "doc_type": "plant_summary",
                            "score": 1.0
                        })
                except Exception as e:
                    print(f"Failed to inject global context: {e}")
                # -----------------------------------------------------------------------

                if not results:
                    response = "No data found. Please upload files using the script: `python 2_scripts/1_ingest_data.py`"
                else:
                    # Get LLM response
                    llm_result = llm.perform_reasoning(prompt, results)
                    response = llm_result.get("answer", "Unable to generate response")
                
                st.markdown(response)
                
            except Exception as e:
                response = f"Error: {str(e)}"
                st.error(response)
    
    # Add assistant response
    st.session_state.messages.append({"role": "assistant", "content": response})

# Sidebar
with st.sidebar:
    st.markdown("### ℹ️ Info")
    st.info("This is a simplified chatbot interface for testing.")
    
    st.markdown("### 📁 Data Status")
    try:
        # Check if data exists
        import sqlite3
        conn = sqlite3.connect(os.path.join(parent_dir, '4_data', 'data_prototype', 'metadata.db'))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM documents")
        doc_count = cursor.fetchone()[0]
        conn.close()
        
        st.success(f"✅ {doc_count} documents loaded")
    except:
        st.warning("⚠️ No data loaded. Run: `python 2_scripts/1_ingest_data.py`")
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()
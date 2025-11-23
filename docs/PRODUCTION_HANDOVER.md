# Production Handover - Energy Data Chatbot

## 📦 What This Is

A chatbot system that answers questions about energy consumption data. Built for integration into your website.

---

## 🏗️ Core Components (For Web Developers)

### Essential Backend Files

#### 1. **Data Processing**
- `smart_preprocessor.py` - Extracts clean data from Excel files
- `ingest_smart.py` - Script to ingest new data
- `ingestion_pipeline.py` - Low-level file processing (legacy, can simplify)

#### 2. **AI/LLM Components**
- `llm_reasoning.py` - Handles query answering with LLM
- `embedding_store.py` - Stores and searches data using embeddings
- `config.py` - Configuration (model names, paths)

#### 3. **Supporting Modules**
- `router.py` - Routes queries to appropriate handlers
- `agent_tools.py` - Numeric calculations
- `utils.py` - Logging utilities
- `user_profiles.py` - User management (optional)

#### 4. **Data Storage**
- `data_prototype/` - Contains all databases and indexes
  - `metadata.db` - Document metadata
  - `indexes/` - FAISS vector embeddings
  - `user_profiles.db` - User data (optional)

#### 5. **Testing Interface**
- `streamlit_app.py` - Demo/testing UI (keep for testing)

---

## 🔌 Integration Guide for Web Developers

### Option 1: REST API (Recommended)

Create a simple Flask/FastAPI wrapper:

```python
from fastapi import FastAPI
from llm_reasoning import LLMReasoning
from embedding_store import EmbeddingStore

app = FastAPI()
llm = LLMReasoning()
store = EmbeddingStore()

@app.post("/api/query")
def answer_query(query: str):
    # Search for relevant data
    results = store.search(query, k=5)
    
    # Get LLM answer
    answer = llm.perform_reasoning(query, results)
    
    return {"answer": answer["answer"]}
```

### Option 2: Direct Integration

Import and use the modules directly in your web framework:

```python
from embedding_store import EmbeddingStore
from llm_reasoning import LLMReasoning

# Initialize once
embedding_store = EmbeddingStore()
llm_reasoning = LLMReasoning()

# For each user query:
def handle_user_query(user_question):
    # 1. Search for relevant data
    search_results = embedding_store.search(user_question, k=5)
    
    # 2. Get LLM answer
    response = llm_reasoning.perform_reasoning(
        query=user_question,
        context=search_results
    )
    
    return response["answer"]
```

---

## 📝 Adding New Data Files

When you receive a new Excel file:

1. **Analyze the structure** (5 minutes):
   - What sheets contain data?
   - What are the column names?

2. **Update `smart_preprocessor.py`**:
   - Adjust `monthly_sheets` list if needed
   - Modify extraction logic if structure is different

3. **Run ingestion**:
   ```bash
   python ingest_smart.py
   ```

4. **Done!** Chatbot now knows the new data.

---

## 🔧 Configuration

Edit `config.py`:

```python
# Model selection
LLM_REASON_MODEL = "llama3.2:latest"  # Change to your preferred model
EMBED_MODEL = "nomic-embed-text"      # Embedding model

# Ollama server
OLLAMA_BASE_URL = "http://localhost:11434"  # Change if Ollama runs elsewhere
```

---

## 🧪 Testing

Run Streamlit for testing:
```bash
streamlit run streamlit_app.py
```

Access at: http://localhost:8501

---

## 📊 Current Dataset

- **File**: Energy Consumption Daily Report MHS Ele - Copy.xlsx
- **43 feeders** monitored
- **10 months** of data (Sept 2023 - July 2024)
- **18.2 million KWH** total consumption

---

## 🚀 Deployment Requirements

### Software Dependencies
```
pandas
numpy
faiss-cpu (or faiss-gpu for production)
requests
streamlit (testing only)
sqlite3 (built-in Python)
```

Install with:
```bash
pip install -r Requirements.txt
```

### External Service
- **Ollama** must be running with models:
  - `llama3.2:latest` (2GB)
  - `nomic-embed-text` (274MB)

Start Ollama:
```bash
ollama serve
```

---

## 📁 What to Deploy

**Minimum files needed:**
```
config.py
embedding_store.py
llm_reasoning.py
smart_preprocessor.py
utils.py
data_prototype/          # Entire folder
Requirements.txt
```

**Optional** (depending on integration):
- `router.py`
- `agent_tools.py`
- `user_profiles.py`
- `streamlit_app.py` (testing only)

---

## 💡 Performance Notes

- **Query response**: 2-4 seconds
- **Data ingestion**: ~5 seconds for 46 Excel sheets
- **Memory usage**: ~500MB (with FAISS index loaded)

---

## 🐛 Common Issues

**"LLM Error" or timeout:**
- Check Ollama is running: `ollama list`
- Model might be too slow - use llama3.2 instead of deepseek-r1

**Empty search results:**
- Run `python ingest_smart.py` to load data

**Module not found:**
- Install: `pip install -r Requirements.txt`

---

## 📞 Support

For questions about the codebase, refer to:
- `walkthrough.md` - Complete system explanation
- `preprocessing_analysis.md` - Data structure details (if kept)

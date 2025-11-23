# config.py

import os

# Get the root directory (parent of 1_core)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Paths ---
DATA_DIR = os.path.join(ROOT_DIR, "4_data", "data_prototype")
INDEX_DIR = os.path.join(DATA_DIR, "indexes")
METADATA_DB_PATH = os.path.join(DATA_DIR, "metadata.db")

# Ensure directories exist
os.makedirs(INDEX_DIR, exist_ok=True)

# --- Ollama Models ---
EMBED_MODEL = "nomic-embed-text"
SLM_PARSE_MODEL = "llama3.2:latest"  # Fast parser
LLM_REASON_MODEL = "llama3.2:latest"  # Fast model - preprocessed data is clean
LLM_EXPLAIN_MODEL = "llama3.2:latest"  # Fast answers

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# --- Ingestion Settings ---
TEXT_CHUNK_SIZE = 300
TEXT_CHUNK_OVERLAP = 50

# --- FAISS Settings ---
FAISS_INDEX_FILE = os.path.join(INDEX_DIR, "faiss.index")
ID_MAP_FILE = os.path.join(INDEX_DIR, "id_map.json")

# --- Retrieval Settings ---
HYBRID_SEARCH_WEIGHTS = {
    "w_semantic": 0.6,
    "w_meta": 0.3,
    "w_numeric": 0.05,
    "w_recency": 0.05,
}
RETRIEVAL_K = 5 # Number of documents to retrieve from FAISS

# --- API Settings ---
API_HOST = "0.0.0.0"
API_PORT = 8000

# --- UI Settings ---
UI_UPLOAD_TIMEOUT = 300 # seconds

# --- Other ---
TRUSTED_FLAG_DEFAULT = True # Default trust level for ingested documents

from pathlib import Path

# config.py -> llm_engineering -> src -> project root
ROOT = Path(__file__).resolve().parents[2]

# Data locations (adjust if your folders are elsewhere)
DATA_DIR = ROOT / "data"
KNOWLEDGE_BASE = DATA_DIR / "knowledge-base"
DB_NAME = str(DATA_DIR / "vector_db")

# Models (keep these identical for ingest and query)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "gpt-4.1-mini"        # use the model name you actually have access to

# Retrieval settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 200
TOP_K = 10

EVALS_DIR = DATA_DIR / "evals"
TEST_FILE = EVALS_DIR / "tests.jsonl"
JUDGE_MODEL = "openai/gpt-6-luna"
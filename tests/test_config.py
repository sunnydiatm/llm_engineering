from llm_engineering.config import ROOT, KNOWLEDGE_BASE, DB_NAME, EMBEDDING_MODEL

def test_root_is_project_root():
    assert (ROOT / "pyproject.toml").exists()

def test_knowledge_base_exists():
    assert KNOWLEDGE_BASE.exists()

def test_db_is_under_data_dir():
    assert "data" in DB_NAME and DB_NAME.endswith("vector_db")

def test_embedding_model_set():
    assert EMBEDDING_MODEL == "all-MiniLM-L6-v2"
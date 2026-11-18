
from pathlib import Path

from llm_engineering.config import (
    ROOT,
    DATA_DIR,
    KNOWLEDGE_BASE,
    DB_NAME,
    EMBEDDING_MODEL,
    LLM_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K,
    EVALS_DIR,
    TEST_FILE,
    JUDGE_MODEL,
)


# ---------- Directory and path tests ----------

def test_root_is_project_root():
    assert ROOT.is_dir()
    assert (ROOT / "pyproject.toml").exists()


def test_data_dir_is_under_root():
    assert DATA_DIR == ROOT / "data"


def test_knowledge_base_is_under_data_dir():
    assert KNOWLEDGE_BASE == DATA_DIR / "knowledge-base"


def test_db_is_under_data_dir():
    assert DB_NAME == str(DATA_DIR / "vector_db")


def test_db_name_is_string():
    assert isinstance(DB_NAME, str)


def test_evals_dir_is_under_data_dir():
    assert EVALS_DIR == DATA_DIR / "evals"


def test_test_file_is_under_evals_dir():
    assert TEST_FILE == EVALS_DIR / "tests.jsonl"


def test_test_file_has_jsonl_extension():
    assert TEST_FILE.suffix == ".jsonl"


# ---------- Model configuration tests ----------

def test_embedding_model_set():
    assert EMBEDDING_MODEL == "all-MiniLM-L6-v2"


def test_llm_model_set():
    assert LLM_MODEL == "gpt-4.1-mini"


def test_judge_model_set():
    assert JUDGE_MODEL == "openai/gpt-6-luna"


def test_model_names_are_non_empty_strings():
    assert isinstance(EMBEDDING_MODEL, str) and EMBEDDING_MODEL
    assert isinstance(LLM_MODEL, str) and LLM_MODEL
    assert isinstance(JUDGE_MODEL, str) and JUDGE_MODEL


# ---------- Retrieval configuration tests ----------

def test_chunk_size_is_positive():
    assert CHUNK_SIZE > 0


def test_chunk_overlap_is_non_negative():
    assert CHUNK_OVERLAP >= 0


def test_chunk_overlap_is_less_than_chunk_size():
    assert CHUNK_OVERLAP < CHUNK_SIZE


def test_top_k_is_positive():
    assert TOP_K > 0


def test_retrieval_settings_are_integers():
    assert isinstance(CHUNK_SIZE, int)
    assert isinstance(CHUNK_OVERLAP, int)
    assert isinstance(TOP_K, int)


# ---------- Configuration consistency tests ----------

def test_all_data_paths_are_path_objects():
    assert isinstance(DATA_DIR, Path)
    assert isinstance(KNOWLEDGE_BASE, Path)
    assert isinstance(EVALS_DIR, Path)
    assert isinstance(TEST_FILE, Path)


def test_knowledge_base_and_evals_are_under_data_dir():
    assert KNOWLEDGE_BASE.is_relative_to(DATA_DIR)
    assert EVALS_DIR.is_relative_to(DATA_DIR)


def test_database_path_is_under_data_dir():
    assert Path(DB_NAME).is_relative_to(DATA_DIR)

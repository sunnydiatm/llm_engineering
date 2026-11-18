from llm_engineering.rag.ingest import fetch_documents, create_chunks

def test_fetch_documents_loads_all_doc_types():
    docs = fetch_documents()
    assert len(docs) > 0
    types = {d.metadata["doc_type"] for d in docs}
    assert {"company", "contracts", "employees", "products"} <= types

def test_create_chunks_not_empty():
    chunks = create_chunks(fetch_documents())
    assert len(chunks) > 0
    assert all(c.page_content.strip() for c in chunks)
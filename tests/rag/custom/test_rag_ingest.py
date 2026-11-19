"""Tests for ingest.py (LLM chunking with Ollama, embeddings, and the Chroma collection).

The LLM, the embeddings server, the worker pool and Chroma are replaced with fakes from
conftest.py, so nothing needs Ollama or touches a real database. The real-Chroma checks live
in test_rag_integration.py.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError


@pytest.fixture(autouse=True)
def quiet_progress_bars(ingest_module, monkeypatch):
    monkeypatch.setattr(ingest_module, "tqdm", lambda iterable=None, **kwargs: iterable)


def make_document(
    doc_type="products", source="kb/products/Carllm.md", text="Carllm covers vehicles."
):
    return {"type": doc_type, "source": source, "text": text}


def chunks_reply(*headlines):
    """What the chunking model is asked to send back."""
    return json.dumps(
        {
            "chunks": [
                {
                    "headline": headline,
                    "summary": f"Summary of {headline}",
                    "original_text": f"Original text of {headline}",
                }
                for headline in headlines
            ]
        }
    )


def make_results(ingest, count):
    """`count` chunks whose texts are 'x', 'xx', 'xxx', ... so each has a distinct length."""
    return [
        ingest.Result(page_content="x" * (n + 1), metadata={"source": f"kb/{n}.md", "type": "t"})
        for n in range(count)
    ]


def install_fake_chroma(monkeypatch, ingest, fakes, existing=()):
    """Replace ingest's PersistentClient. Returns the fake client and the paths it was given."""
    client = fakes.ChromaClient(existing=existing)
    paths = []

    def factory(path):
        paths.append(path)
        return client

    monkeypatch.setattr(ingest, "PersistentClient", factory)
    return client, paths


def install_fake_embeddings(monkeypatch, ingest, embeddings):
    monkeypatch.setattr(ingest, "openai", SimpleNamespace(embeddings=embeddings))


# --- models -------------------------------------------------------------------------


def test_chunk_as_result_joins_headline_summary_and_text(ingest_module):
    chunk = ingest_module.Chunk(
        headline="Carllm overview",
        summary="What Carllm is.",
        original_text="Carllm is vehicle insurance.",
    )

    result = chunk.as_result(make_document())

    assert result.page_content == (
        "Carllm overview\n\nWhat Carllm is.\n\nCarllm is vehicle insurance."
    )
    assert result.metadata == {"source": "kb/products/Carllm.md", "type": "products"}


def test_chunk_requires_every_field(ingest_module):
    with pytest.raises(ValidationError):
        ingest_module.Chunk(headline="Only a headline")


def test_chunks_parses_the_models_json(ingest_module):
    parsed = ingest_module.Chunks.model_validate_json(chunks_reply("One", "Two"))
    assert [chunk.headline for chunk in parsed.chunks] == ["One", "Two"]


# --- fetch_documents ----------------------------------------------------------------


@pytest.fixture
def knowledge_base(tmp_path):
    root = tmp_path / "knowledge-base"
    for folder in ("products/nested", "employees", "company"):
        (root / folder).mkdir(parents=True)
    (root / "products" / "Carllm.md").write_text("Carllm text", encoding="utf-8")
    (root / "products" / "nested" / "Deep.md").write_text("Deep text", encoding="utf-8")
    (root / "employees" / "Maxine.md").write_text("Maxine – IIOTY ✓", encoding="utf-8")
    (root / "company" / "notes.txt").write_text("not markdown", encoding="utf-8")
    return root


def test_fetch_documents_loads_every_markdown_file(ingest_module, knowledge_base, monkeypatch):
    monkeypatch.setattr(ingest_module, "KNOWLEDGE_BASE_PATH", knowledge_base)

    documents = ingest_module.fetch_documents()

    assert len(documents) == 3
    assert {d["text"] for d in documents} == {"Carllm text", "Deep text", "Maxine – IIOTY ✓"}


def test_fetch_documents_types_documents_by_their_top_level_folder(
    ingest_module, knowledge_base, monkeypatch
):
    monkeypatch.setattr(ingest_module, "KNOWLEDGE_BASE_PATH", knowledge_base)

    documents = ingest_module.fetch_documents()

    by_text = {d["text"]: d["type"] for d in documents}
    assert by_text["Carllm text"] == "products"
    assert by_text["Deep text"] == "products"  # nested folders keep the top-level type
    assert by_text["Maxine – IIOTY ✓"] == "employees"


def test_fetch_documents_records_the_source_path(ingest_module, knowledge_base, monkeypatch):
    monkeypatch.setattr(ingest_module, "KNOWLEDGE_BASE_PATH", knowledge_base)

    documents = ingest_module.fetch_documents()

    sources = {d["source"] for d in documents}
    assert (knowledge_base / "products" / "Carllm.md").as_posix() in sources
    assert (knowledge_base / "products" / "nested" / "Deep.md").as_posix() in sources


def test_fetch_documents_ignores_files_that_are_not_markdown(
    ingest_module, knowledge_base, monkeypatch
):
    monkeypatch.setattr(ingest_module, "KNOWLEDGE_BASE_PATH", knowledge_base)

    documents = ingest_module.fetch_documents()

    assert "company" not in {d["type"] for d in documents}  # that folder only has a .txt file
    assert not any(d["source"].endswith(".txt") for d in documents)


def test_fetch_documents_reports_how_many_it_loaded(
    ingest_module, knowledge_base, monkeypatch, capsys
):
    monkeypatch.setattr(ingest_module, "KNOWLEDGE_BASE_PATH", knowledge_base)

    ingest_module.fetch_documents()

    assert "Loaded 3 documents" in capsys.readouterr().out


def test_fetch_documents_with_an_empty_knowledge_base(ingest_module, tmp_path, monkeypatch):
    empty = tmp_path / "empty-kb"
    empty.mkdir()
    monkeypatch.setattr(ingest_module, "KNOWLEDGE_BASE_PATH", empty)

    assert ingest_module.fetch_documents() == []


# --- make_prompt / make_messages ----------------------------------------------------


def test_make_prompt_includes_the_document_details(ingest_module):
    document = make_document(text="Carllm covers vehicles.")

    prompt = ingest_module.make_prompt(document)

    assert "products" in prompt
    assert "kb/products/Carllm.md" in prompt
    assert "Carllm covers vehicles." in prompt


def test_make_prompt_asks_for_more_chunks_from_longer_documents(ingest_module, monkeypatch):
    monkeypatch.setattr(ingest_module, "AVERAGE_CHUNK_SIZE", 100)

    short = ingest_module.make_prompt(make_document(text="x" * 50))
    long = ingest_module.make_prompt(make_document(text="x" * 250))

    assert "at least 1 chunks" in short  # 50 // 100 + 1
    assert "at least 3 chunks" in long  # 250 // 100 + 1


def test_make_messages_is_a_single_user_message(ingest_module):
    document = make_document()

    messages = ingest_module.make_messages(document)

    assert messages == [{"role": "user", "content": ingest_module.make_prompt(document)}]


# --- process_document ---------------------------------------------------------------


def test_process_document_turns_the_models_chunks_into_results(
    ingest_module, fakes, monkeypatch
):
    completion = fakes.completion(chunks_reply("Overview", "Pricing"))
    monkeypatch.setattr(ingest_module, "completion", completion)
    document = make_document()

    results = ingest_module.process_document(document)

    assert [r.page_content.split("\n\n")[0] for r in results] == ["Overview", "Pricing"]
    assert all(r.metadata == {"source": document["source"], "type": "products"} for r in results)


def test_process_document_calls_the_local_model_with_a_structured_reply(
    ingest_module, fakes, monkeypatch
):
    completion = fakes.completion(chunks_reply("Overview"))
    monkeypatch.setattr(ingest_module, "completion", completion)
    document = make_document()

    ingest_module.process_document(document)

    (call,) = completion.calls
    assert call["model"] == ingest_module.MODEL
    assert call["api_base"] == ingest_module.OLLAMA_API_BASE
    assert call["response_format"] is ingest_module.Chunks
    assert call["messages"] == ingest_module.make_messages(document)


def test_process_document_retries_after_a_failed_call(ingest_module, fakes, monkeypatch):
    completion = fakes.completion(RuntimeError("ollama busy"), chunks_reply("Overview"))
    monkeypatch.setattr(ingest_module, "completion", completion)

    results = ingest_module.process_document(make_document())

    assert len(results) == 1
    assert len(completion.calls) == 2


def test_process_document_retries_after_an_unusable_reply(ingest_module, fakes, monkeypatch):
    completion = fakes.completion("this is not json", chunks_reply("Overview"))
    monkeypatch.setattr(ingest_module, "completion", completion)

    results = ingest_module.process_document(make_document())

    assert len(results) == 1
    assert len(completion.calls) == 2


# --- create_chunks ------------------------------------------------------------------


@pytest.fixture
def in_process_pool(ingest_module, monkeypatch):
    """
    Run the 'worker pool' inside the test process. A real pool starts new processes, which
    would not see the fakes and would call the real Ollama. Returns the process counts the
    pool was created with.
    """
    created_with = []

    class InProcessPool:
        def __init__(self, processes):
            created_with.append(processes)

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

        def imap_unordered(self, function, iterable):
            return map(function, iterable)

    monkeypatch.setattr(ingest_module, "Pool", InProcessPool)
    return created_with


def test_create_chunks_collects_the_chunks_of_every_document(
    ingest_module, in_process_pool, monkeypatch
):
    def fake_process_document(document):
        return [
            ingest_module.Result(
                page_content=f"{document['source']} part {part}", metadata={"source": document["source"]}
            )
            for part in (1, 2)
        ]

    monkeypatch.setattr(ingest_module, "process_document", fake_process_document)
    documents = [make_document(source=f"kb/{n}.md") for n in range(3)]

    chunks = ingest_module.create_chunks(documents)

    assert len(chunks) == 6
    assert {c.page_content for c in chunks} == {
        f"kb/{n}.md part {part}" for n in range(3) for part in (1, 2)
    }


def test_create_chunks_uses_the_configured_number_of_workers(
    ingest_module, in_process_pool, monkeypatch
):
    monkeypatch.setattr(ingest_module, "process_document", lambda document: [])

    ingest_module.create_chunks([make_document()])

    assert in_process_pool == [ingest_module.WORKERS]


def test_create_chunks_with_no_documents(ingest_module, in_process_pool):
    assert ingest_module.create_chunks([]) == []


# --- create_embeddings (one request for everything) ---------------------------------


def test_create_embeddings_stores_vectors_documents_and_metadata(
    ingest_module, fakes, monkeypatch, tmp_path, capsys
):
    embeddings = fakes.Embeddings()
    install_fake_embeddings(monkeypatch, ingest_module, embeddings)
    client, paths = install_fake_chroma(monkeypatch, ingest_module, fakes)
    db_path = str(tmp_path / "db")
    monkeypatch.setattr(ingest_module, "DB_NAME", db_path)
    chunks = make_results(ingest_module, 3)

    ingest_module.create_embeddings(chunks)

    assert paths == [db_path]
    assert embeddings.calls == [
        {"model": ingest_module.embedding_model, "input": ["x", "xx", "xxx"]}
    ]
    (added,) = client.collections[ingest_module.collection_name].add_calls
    assert added["ids"] == ["0", "1", "2"]
    assert added["documents"] == ["x", "xx", "xxx"]
    assert added["embeddings"] == [[1.0, 1.0, 0.0], [2.0, 1.0, 0.0], [3.0, 1.0, 0.0]]
    assert added["metadatas"] == [chunk.metadata for chunk in chunks]
    assert "Vectorstore created with 3 documents" in capsys.readouterr().out


def test_create_embeddings_replaces_an_existing_collection(ingest_module, fakes, monkeypatch):
    install_fake_embeddings(monkeypatch, ingest_module, fakes.Embeddings())
    client, _ = install_fake_chroma(
        monkeypatch, ingest_module, fakes, existing=[ingest_module.collection_name]
    )

    ingest_module.create_embeddings(make_results(ingest_module, 2))

    assert client.deleted == [ingest_module.collection_name]
    assert client.collections[ingest_module.collection_name].count() == 2


def test_create_embeddings_does_not_delete_when_there_is_no_collection(
    ingest_module, fakes, monkeypatch
):
    install_fake_embeddings(monkeypatch, ingest_module, fakes.Embeddings())
    client, _ = install_fake_chroma(monkeypatch, ingest_module, fakes)

    ingest_module.create_embeddings(make_results(ingest_module, 2))

    assert client.deleted == []


# --- create_embeddings_batch --------------------------------------------------------


def test_batch_with_no_chunks_does_nothing(ingest_module, fakes, monkeypatch, capsys):
    embeddings = fakes.Embeddings()
    install_fake_embeddings(monkeypatch, ingest_module, embeddings)
    client, paths = install_fake_chroma(monkeypatch, ingest_module, fakes)

    ingest_module.create_embeddings_batch([])

    assert "No chunks to embed." in capsys.readouterr().out
    assert embeddings.calls == []
    assert paths == []  # the database was not even opened


@pytest.mark.parametrize("batch_size", [0, -5, 2049])
def test_batch_size_must_be_between_1_and_2048(ingest_module, fakes, monkeypatch, batch_size):
    embeddings = fakes.Embeddings()
    install_fake_embeddings(monkeypatch, ingest_module, embeddings)
    install_fake_chroma(monkeypatch, ingest_module, fakes)

    with pytest.raises(ValueError, match="batch_size"):
        ingest_module.create_embeddings_batch(
            make_results(ingest_module, 3), batch_size=batch_size
        )

    assert embeddings.calls == []


@pytest.mark.parametrize("batch_size, expected_calls", [(1, 3), (2048, 1)])
def test_batch_size_limits_are_allowed(
    ingest_module, fakes, monkeypatch, batch_size, expected_calls
):
    embeddings = fakes.Embeddings()
    install_fake_embeddings(monkeypatch, ingest_module, embeddings)
    install_fake_chroma(monkeypatch, ingest_module, fakes)

    ingest_module.create_embeddings_batch(make_results(ingest_module, 3), batch_size=batch_size)

    assert len(embeddings.calls) == expected_calls


def assert_every_vector_belongs_to_its_document(add_calls, total):
    ids = [doc_id for call in add_calls for doc_id in call["ids"]]
    assert ids == [str(n) for n in range(total)]
    for call in add_calls:
        for document, vector in zip(call["documents"], call["embeddings"]):
            assert vector[0] == float(len(document))


def test_batch_embeds_in_batches_and_keeps_everything_in_order(
    ingest_module, fakes, monkeypatch
):
    embeddings = fakes.Embeddings()
    install_fake_embeddings(monkeypatch, ingest_module, embeddings)
    client, _ = install_fake_chroma(monkeypatch, ingest_module, fakes)

    ingest_module.create_embeddings_batch(make_results(ingest_module, 250), batch_size=100)

    assert [len(call["input"]) for call in embeddings.calls] == [100, 100, 50]
    add_calls = client.collections[ingest_module.collection_name].add_calls
    assert len(add_calls) == 3
    assert_every_vector_belongs_to_its_document(add_calls, total=250)


def test_batch_restores_the_input_order_when_the_server_answers_out_of_order(
    ingest_module, fakes, monkeypatch
):
    install_fake_embeddings(monkeypatch, ingest_module, fakes.Embeddings(reverse=True))
    client, _ = install_fake_chroma(monkeypatch, ingest_module, fakes)

    ingest_module.create_embeddings_batch(make_results(ingest_module, 7), batch_size=3)

    add_calls = client.collections[ingest_module.collection_name].add_calls
    assert_every_vector_belongs_to_its_document(add_calls, total=7)


def test_batch_stores_the_metadata_with_each_chunk(ingest_module, fakes, monkeypatch):
    install_fake_embeddings(monkeypatch, ingest_module, fakes.Embeddings())
    client, _ = install_fake_chroma(monkeypatch, ingest_module, fakes)
    chunks = make_results(ingest_module, 5)

    ingest_module.create_embeddings_batch(chunks, batch_size=2)

    add_calls = client.collections[ingest_module.collection_name].add_calls
    stored = [meta for call in add_calls for meta in call["metadatas"]]
    assert stored == [chunk.metadata for chunk in chunks]


def test_batch_keeps_the_existing_collection_if_embedding_fails(
    ingest_module, fakes, monkeypatch
):
    install_fake_embeddings(monkeypatch, ingest_module, fakes.Embeddings(fail_on_call=2))
    client, _ = install_fake_chroma(
        monkeypatch, ingest_module, fakes, existing=[ingest_module.collection_name]
    )

    with pytest.raises(RuntimeError, match="embedding server unavailable"):
        ingest_module.create_embeddings_batch(make_results(ingest_module, 5), batch_size=2)

    assert client.deleted == []  # the old data is still there


def test_batch_replaces_the_existing_collection_after_success(ingest_module, fakes, monkeypatch):
    install_fake_embeddings(monkeypatch, ingest_module, fakes.Embeddings())
    client, _ = install_fake_chroma(
        monkeypatch, ingest_module, fakes, existing=[ingest_module.collection_name]
    )

    ingest_module.create_embeddings_batch(make_results(ingest_module, 3), batch_size=2)

    assert client.deleted == [ingest_module.collection_name]
    assert client.collections[ingest_module.collection_name].count() == 3


def test_batch_reports_its_progress(ingest_module, fakes, monkeypatch, capsys):
    install_fake_embeddings(monkeypatch, ingest_module, fakes.Embeddings())
    install_fake_chroma(monkeypatch, ingest_module, fakes)

    ingest_module.create_embeddings_batch(make_results(ingest_module, 3), batch_size=2)

    output = capsys.readouterr().out
    assert "Total chunks to embed: 3" in output
    assert "Processing in batches of 2" in output
    assert "Generated 3 embeddings" in output
    assert "Vectorstore created with 3 documents" in output


def test_batch_uses_the_database_path_from_config(
    ingest_module, fakes, monkeypatch, tmp_path
):
    install_fake_embeddings(monkeypatch, ingest_module, fakes.Embeddings())
    client, paths = install_fake_chroma(monkeypatch, ingest_module, fakes)
    db_path = str(tmp_path / "db")
    monkeypatch.setattr(ingest_module, "DB_NAME", db_path)

    ingest_module.create_embeddings_batch(make_results(ingest_module, 2))

    assert paths == [db_path]
    assert isinstance(Path(paths[0]), Path)
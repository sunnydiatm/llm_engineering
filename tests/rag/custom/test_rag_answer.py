"""Tests for llm_engineering/rag/custom/answer.py.

Covers the query rewrite, the two lookups, merging, reranking, prompt building and the final
answer. The LLM (litellm.completion), the embeddings server and the Chroma collection are all
replaced with fakes from conftest.py, so nothing needs Ollama or a database.

A few rerank tests (ids out of range, repeated ids, forgotten ids) need the tolerant version
of rerank() from the earlier fix. Without it they fail with the same IndexError you saw in
the app log.
"""

import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError


def make_chunk(answer, text, source="knowledge-base/products/Carllm.md"):
    return answer.Result(page_content=text, metadata={"source": source, "type": "products"})


def rank_reply(order):
    """What the reranking model is asked to send back."""
    return json.dumps({"order": order})


def contents(chunks):
    return [chunk.page_content for chunk in chunks]


# --- models -------------------------------------------------------------------------


def test_result_requires_content_and_metadata(custom_answer):
    with pytest.raises(ValidationError):
        custom_answer.Result(page_content="only text")


def test_rank_order_parses_json(custom_answer):
    parsed = custom_answer.RankOrder.model_validate_json('{"order": [3, 1, 2]}')
    assert parsed.order == [3, 1, 2]


def test_rank_order_rejects_non_integer_ids(custom_answer):
    with pytest.raises(ValidationError):
        custom_answer.RankOrder.model_validate_json('{"order": ["first", "second"]}')


# --- merge_chunks -------------------------------------------------------------------


def test_merge_chunks_adds_only_content_it_has_not_seen(custom_answer):
    a, b, c = (make_chunk(custom_answer, text) for text in ("A", "B", "C"))
    merged = custom_answer.merge_chunks([a, b], [b, c])
    assert contents(merged) == ["A", "B", "C"]


def test_merge_chunks_keeps_the_first_list_in_front(custom_answer):
    a, b, c = (make_chunk(custom_answer, text) for text in ("A", "B", "C"))
    merged = custom_answer.merge_chunks([c, a], [b])
    assert contents(merged) == ["C", "A", "B"]


def test_merge_chunks_with_an_empty_first_list(custom_answer):
    a, b = (make_chunk(custom_answer, text) for text in ("A", "B"))
    assert contents(custom_answer.merge_chunks([], [a, b])) == ["A", "B"]


def test_merge_chunks_with_an_empty_second_list(custom_answer):
    a, b = (make_chunk(custom_answer, text) for text in ("A", "B"))
    assert contents(custom_answer.merge_chunks([a, b], [])) == ["A", "B"]


def test_merge_chunks_does_not_modify_its_inputs(custom_answer):
    a, b, c = (make_chunk(custom_answer, text) for text in ("A", "B", "C"))
    first, second = [a, b], [b, c]
    custom_answer.merge_chunks(first, second)
    assert contents(first) == ["A", "B"]
    assert contents(second) == ["B", "C"]


def test_merge_chunks_compares_page_content_not_source(custom_answer):
    same_text_a = make_chunk(custom_answer, "Same text", source="kb/a.md")
    same_text_b = make_chunk(custom_answer, "Same text", source="kb/b.md")
    merged = custom_answer.merge_chunks([same_text_a], [same_text_b])
    assert len(merged) == 1


# --- make_rag_messages --------------------------------------------------------------


def test_make_rag_messages_builds_system_history_then_question(custom_answer):
    chunks = [
        make_chunk(custom_answer, "Carllm covers vehicles.", "kb/products/Carllm.md"),
        make_chunk(custom_answer, "Homellm covers homes.", "kb/products/Homellm.md"),
    ]
    history = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]

    messages = custom_answer.make_rag_messages("What is Carllm?", history, chunks)

    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[1:3] == history
    assert messages[-1] == {"role": "user", "content": "What is Carllm?"}

    system = messages[0]["content"]
    assert "Insurellm" in system
    assert "Extract from kb/products/Carllm.md:\nCarllm covers vehicles." in system
    assert "Extract from kb/products/Homellm.md:\nHomellm covers homes." in system
    assert system.index("Carllm covers") < system.index("Homellm covers")


def test_make_rag_messages_without_history_or_chunks(custom_answer):
    messages = custom_answer.make_rag_messages("Anything?", [], [])

    assert [m["role"] for m in messages] == ["system", "user"]
    assert "{context}" not in messages[0]["content"]  # the placeholder was filled in


def test_make_rag_messages_does_not_modify_the_history(custom_answer):
    history = [{"role": "user", "content": "Hi"}]
    custom_answer.make_rag_messages("Next?", history, [])
    assert history == [{"role": "user", "content": "Hi"}]


def test_make_rag_messages_copes_with_braces_in_the_context(custom_answer):
    chunk = make_chunk(custom_answer, "Price table: {basic} and {premium}")
    messages = custom_answer.make_rag_messages("Prices?", [], [chunk])
    assert "{basic} and {premium}" in messages[0]["content"]


# --- fetch_context_unranked ---------------------------------------------------------


def test_fetch_context_unranked_returns_results_with_their_metadata(
    custom_answer, fakes, monkeypatch
):
    collection = fakes.Collection(
        documents=["doc one", "doc two"],
        metadatas=[
            {"source": "kb/a.md", "type": "products"},
            {"source": "kb/b.md", "type": "company"},
        ],
    )
    monkeypatch.setattr(custom_answer, "collection", collection)
    monkeypatch.setattr(custom_answer, "openai", SimpleNamespace(embeddings=fakes.Embeddings()))

    chunks = custom_answer.fetch_context_unranked("What is Carllm?")

    assert all(isinstance(chunk, custom_answer.Result) for chunk in chunks)
    assert contents(chunks) == ["doc one", "doc two"]
    assert [chunk.metadata["source"] for chunk in chunks] == ["kb/a.md", "kb/b.md"]
    assert [chunk.metadata["type"] for chunk in chunks] == ["products", "company"]


def test_fetch_context_unranked_embeds_the_question_and_queries_with_it(
    custom_answer, fakes, monkeypatch
):
    embeddings = fakes.Embeddings()
    collection = fakes.Collection(documents=["doc"], metadatas=[{"source": "s", "type": "t"}])
    monkeypatch.setattr(custom_answer, "collection", collection)
    monkeypatch.setattr(custom_answer, "openai", SimpleNamespace(embeddings=embeddings))
    question = "What is Carllm?"

    custom_answer.fetch_context_unranked(question)

    assert embeddings.calls == [{"model": custom_answer.embedding_model, "input": [question]}]
    (query,) = collection.query_calls
    assert query["n_results"] == custom_answer.RETRIEVAL_K
    assert query["query_embeddings"] == [[float(len(question)), 1.0, 0.0]]


def test_fetch_context_unranked_returns_at_most_retrieval_k(custom_answer, fakes, monkeypatch):
    total = custom_answer.RETRIEVAL_K + 10
    collection = fakes.Collection(
        documents=[f"doc {n}" for n in range(total)],
        metadatas=[{"source": f"s{n}", "type": "t"} for n in range(total)],
    )
    monkeypatch.setattr(custom_answer, "collection", collection)
    monkeypatch.setattr(custom_answer, "openai", SimpleNamespace(embeddings=fakes.Embeddings()))

    chunks = custom_answer.fetch_context_unranked("anything")

    assert len(chunks) == custom_answer.RETRIEVAL_K


def test_fetch_context_unranked_with_an_empty_collection(custom_answer, fakes, monkeypatch):
    monkeypatch.setattr(custom_answer, "collection", fakes.Collection())
    monkeypatch.setattr(custom_answer, "openai", SimpleNamespace(embeddings=fakes.Embeddings()))

    assert custom_answer.fetch_context_unranked("anything") == []


# --- rerank -------------------------------------------------------------------------


def test_rerank_orders_chunks_by_the_models_ranking(custom_answer, fakes, monkeypatch):
    chunks = [make_chunk(custom_answer, text) for text in ("first", "second", "third")]
    monkeypatch.setattr(custom_answer, "completion", fakes.completion(rank_reply([3, 1, 2])))

    ranked = custom_answer.rerank("Which is best?", chunks)

    assert contents(ranked) == ["third", "first", "second"]


def test_rerank_prompt_lists_every_chunk_with_its_id(custom_answer, fakes, monkeypatch):
    chunks = [make_chunk(custom_answer, text) for text in ("first", "second")]
    completion = fakes.completion(rank_reply([1, 2]))
    monkeypatch.setattr(custom_answer, "completion", completion)

    custom_answer.rerank("Which is best?", chunks)

    (call,) = completion.calls
    assert call["model"] == custom_answer.MODEL
    assert call["response_format"] is custom_answer.RankOrder
    system_message, user_message = call["messages"]
    assert system_message["role"] == "system"
    assert user_message["role"] == "user"
    assert "Which is best?" in user_message["content"]
    assert "# CHUNK ID: 1:\n\nfirst" in user_message["content"]
    assert "# CHUNK ID: 2:\n\nsecond" in user_message["content"]


def test_rerank_ignores_ids_outside_the_chunk_range(custom_answer, fakes, monkeypatch):
    chunks = [make_chunk(custom_answer, text) for text in ("A", "B")]
    monkeypatch.setattr(custom_answer, "completion", fakes.completion(rank_reply([2, 7, 0, 1])))

    assert contents(custom_answer.rerank("q", chunks)) == ["B", "A"]


def test_rerank_ignores_repeated_ids(custom_answer, fakes, monkeypatch):
    chunks = [make_chunk(custom_answer, text) for text in ("A", "B")]
    monkeypatch.setattr(custom_answer, "completion", fakes.completion(rank_reply([2, 2, 1])))

    assert contents(custom_answer.rerank("q", chunks)) == ["B", "A"]


def test_rerank_keeps_chunks_the_model_forgot(custom_answer, fakes, monkeypatch):
    chunks = [make_chunk(custom_answer, text) for text in ("A", "B", "C")]
    monkeypatch.setattr(custom_answer, "completion", fakes.completion(rank_reply([3])))

    assert contents(custom_answer.rerank("q", chunks)) == ["C", "A", "B"]


def test_rerank_of_no_chunks_returns_no_chunks(custom_answer, fakes, monkeypatch):
    monkeypatch.setattr(custom_answer, "completion", fakes.completion(rank_reply([1, 2])))

    assert custom_answer.rerank("q", []) == []


def test_rerank_retries_after_an_unusable_reply(custom_answer, fakes, monkeypatch):
    chunks = [make_chunk(custom_answer, text) for text in ("A", "B")]
    completion = fakes.completion("this is not json", rank_reply([2, 1]))
    monkeypatch.setattr(custom_answer, "completion", completion)

    ranked = custom_answer.rerank("q", chunks)

    assert contents(ranked) == ["B", "A"]
    assert len(completion.calls) == 2


def test_rerank_gives_up_after_three_attempts(custom_answer, fakes, monkeypatch):
    chunks = [make_chunk(custom_answer, "A")]
    completion = fakes.completion("bad", "bad", "bad")
    monkeypatch.setattr(custom_answer, "completion", completion)

    with pytest.raises(ValueError):  # pydantic's ValidationError is a ValueError
        custom_answer.rerank("q", chunks)

    assert len(completion.calls) == 3


# --- rewrite_query ------------------------------------------------------------------


def test_rewrite_query_returns_the_models_reply(custom_answer, fakes, monkeypatch):
    completion = fakes.completion("Who is the founder of Insurellm?")
    monkeypatch.setattr(custom_answer, "completion", completion)
    history = [{"role": "user", "content": "Tell me about Insurellm"}]

    rewritten = custom_answer.rewrite_query("Who founded it?", history)

    assert rewritten == "Who is the founder of Insurellm?"
    (call,) = completion.calls
    assert call["model"] == custom_answer.MODEL
    prompt = call["messages"][0]["content"]
    assert "Who founded it?" in prompt
    assert "Tell me about Insurellm" in prompt


def test_rewrite_query_without_history_uses_an_empty_one(custom_answer, fakes, monkeypatch):
    completion = fakes.completion("rewritten")
    monkeypatch.setattr(custom_answer, "completion", completion)

    custom_answer.rewrite_query("What is Carllm?")

    assert "[]" in completion.calls[0]["messages"][0]["content"]


def test_rewrite_query_retries_after_a_failed_call(custom_answer, fakes, monkeypatch):
    completion = fakes.completion(RuntimeError("ollama busy"), "rewritten")
    monkeypatch.setattr(custom_answer, "completion", completion)

    assert custom_answer.rewrite_query("What is Carllm?") == "rewritten"
    assert len(completion.calls) == 2


# --- fetch_context ------------------------------------------------------------------


def test_fetch_context_searches_with_the_original_and_the_rewritten_question(
    custom_answer, monkeypatch
):
    searched = []

    def fake_unranked(question):
        searched.append(question)
        return [make_chunk(custom_answer, f"result for {question}")]

    monkeypatch.setattr(custom_answer, "rewrite_query", lambda question, history=None: "rewritten")
    monkeypatch.setattr(custom_answer, "fetch_context_unranked", fake_unranked)
    monkeypatch.setattr(custom_answer, "rerank", lambda question, chunks: chunks)

    result = custom_answer.fetch_context("original", [])

    assert searched == ["original", "rewritten"]
    assert contents(result) == ["result for original", "result for rewritten"]


def test_fetch_context_reranks_the_merged_chunks_for_the_original_question(
    custom_answer, monkeypatch
):
    shared = [make_chunk(custom_answer, "A"), make_chunk(custom_answer, "B")]
    seen = {}

    def fake_rerank(question, chunks):
        seen["question"] = question
        seen["chunks"] = chunks
        return list(reversed(chunks))

    monkeypatch.setattr(custom_answer, "rewrite_query", lambda question, history=None: "rewritten")
    monkeypatch.setattr(custom_answer, "fetch_context_unranked", lambda question: shared)
    monkeypatch.setattr(custom_answer, "rerank", fake_rerank)

    result = custom_answer.fetch_context("original", [])

    assert seen["question"] == "original"
    assert contents(seen["chunks"]) == ["A", "B"]  # both lookups found the same chunks once
    assert contents(result) == ["B", "A"]


def test_fetch_context_returns_at_most_final_k_chunks(custom_answer, monkeypatch):
    many = [make_chunk(custom_answer, f"chunk {n}") for n in range(5)]
    monkeypatch.setattr(custom_answer, "FINAL_K", 3)
    monkeypatch.setattr(custom_answer, "rewrite_query", lambda question, history=None: question)
    monkeypatch.setattr(custom_answer, "fetch_context_unranked", lambda question: many)
    monkeypatch.setattr(custom_answer, "rerank", lambda question, chunks: chunks)

    result = custom_answer.fetch_context("question", [])

    assert contents(result) == ["chunk 0", "chunk 1", "chunk 2"]


# --- answer_question ----------------------------------------------------------------


def test_answer_question_returns_the_answer_and_the_chunks_it_used(
    custom_answer, fakes, monkeypatch
):
    chunks = [make_chunk(custom_answer, "Carllm covers vehicles.")]
    monkeypatch.setattr(custom_answer, "fetch_context", lambda question, history=None: chunks)
    monkeypatch.setattr(custom_answer, "completion", fakes.completion("Carllm is vehicle cover."))

    answer, used = custom_answer.answer_question("What is Carllm?")

    assert answer == "Carllm is vehicle cover."
    assert used is chunks


def test_answer_question_sends_context_history_and_question_to_the_model(
    custom_answer, fakes, monkeypatch
):
    chunks = [make_chunk(custom_answer, "Carllm covers vehicles.", "kb/products/Carllm.md")]
    history = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]
    completion = fakes.completion("an answer")
    monkeypatch.setattr(custom_answer, "fetch_context", lambda question, history=None: chunks)
    monkeypatch.setattr(custom_answer, "completion", completion)

    custom_answer.answer_question("What is Carllm?", history)

    (call,) = completion.calls
    assert call["model"] == custom_answer.MODEL
    messages = call["messages"]
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert "Extract from kb/products/Carllm.md:\nCarllm covers vehicles." in messages[0]["content"]
    assert messages[-1] == {"role": "user", "content": "What is Carllm?"}


def test_answer_question_passes_the_history_to_fetch_context(custom_answer, fakes, monkeypatch):
    seen = {}

    def fake_fetch_context(question, history=None):
        seen["question"] = question
        seen["history"] = history
        return []

    history = [{"role": "user", "content": "Hi"}]
    monkeypatch.setattr(custom_answer, "fetch_context", fake_fetch_context)
    monkeypatch.setattr(custom_answer, "completion", fakes.completion("ok"))

    custom_answer.answer_question("What is Carllm?", history)

    assert seen == {"question": "What is Carllm?", "history": history}


def test_answer_question_does_not_remember_history_between_calls(
    custom_answer, fakes, monkeypatch
):
    completion = fakes.completion("first", "second")
    monkeypatch.setattr(custom_answer, "fetch_context", lambda question, history=None: [])
    monkeypatch.setattr(custom_answer, "completion", completion)

    custom_answer.answer_question("One?")
    custom_answer.answer_question("Two?")

    assert [len(call["messages"]) for call in completion.calls] == [2, 2]  # system + question


def test_answer_question_retries_after_a_failed_call(custom_answer, fakes, monkeypatch):
    completion = fakes.completion(RuntimeError("ollama busy"), "an answer")
    monkeypatch.setattr(custom_answer, "fetch_context", lambda question, history=None: [])
    monkeypatch.setattr(custom_answer, "completion", completion)

    answer, _ = custom_answer.answer_question("What is Carllm?")

    assert answer == "an answer"
    assert len(completion.calls) == 2


def test_answer_question_gives_up_after_three_attempts(custom_answer, fakes, monkeypatch):
    error = RuntimeError("ollama is down")
    completion = fakes.completion(error, error, error)
    monkeypatch.setattr(custom_answer, "fetch_context", lambda question, history=None: [])
    monkeypatch.setattr(custom_answer, "completion", completion)

    with pytest.raises(RuntimeError, match="ollama is down"):
        custom_answer.answer_question("What is Carllm?")

    assert len(completion.calls) == 3


# --- the whole pipeline, with only the outside services faked ------------------------


def test_full_pipeline_rewrites_searches_reranks_then_answers(custom_answer, fakes, monkeypatch):
    documents = [
        "Maxine Thompson won the IIOTY award in 2023.",
        "Carllm covers vehicles.",
        "Homellm covers homes.",
    ]
    sources = ["kb/employees/Maxine.md", "kb/products/Carllm.md", "kb/products/Homellm.md"]
    collection = fakes.Collection(
        documents=documents,
        metadatas=[{"source": source, "type": "kb"} for source in sources],
    )
    # calls in order: query rewrite, rerank, final answer
    completion = fakes.completion("Who won IIOTY in 2023?", rank_reply([3, 1, 2]), "Maxine won.")
    monkeypatch.setattr(custom_answer, "collection", collection)
    monkeypatch.setattr(custom_answer, "openai", SimpleNamespace(embeddings=fakes.Embeddings()))
    monkeypatch.setattr(custom_answer, "completion", completion)

    answer, used = custom_answer.answer_question("Who won the award?")

    assert answer == "Maxine won."
    assert contents(used) == [documents[2], documents[0], documents[1]]  # the reranked order

    rewrite_call, rerank_call, answer_call = completion.calls
    assert "response_format" not in rewrite_call
    assert rerank_call["response_format"] is custom_answer.RankOrder
    system = answer_call["messages"][0]["content"]
    assert system.index("Homellm covers homes.") < system.index("Maxine Thompson won")
    assert system.index("Maxine Thompson won") < system.index("Carllm covers vehicles.")
    assert answer_call["messages"][-1] == {"role": "user", "content": "Who won the award?"}
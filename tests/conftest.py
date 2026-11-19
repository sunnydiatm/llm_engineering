"""Shared pytest setup for the whole test suite."""

import importlib
import os
import sys
import types
from types import SimpleNamespace

import pytest

# answer.py builds a ChatOpenAI client when it is imported, which needs a key to exist.
# This dummy value is never used to call the API.
os.environ.setdefault("OPENAI_API_KEY", "test-key")


@pytest.fixture
def stub_answer(monkeypatch):
    """
    Replace llm_engineering.rag.answer with a lightweight stub.

    The real module loads the embedding model and opens the Chroma database as soon as it
    is imported. The eval modules import it, so without this stub their unit tests would
    be slow and would depend on the vector database existing.
    """
    stub = types.ModuleType("llm_engineering.rag.answer")
    stub.RETRIEVAL_K = 10
    stub.fetch_context = lambda question: []
    stub.answer_question = lambda question, history=None: ("stub answer", [])
    # answer.py now lives in rag/custom/. Register the stub under both locations so the eval
    # code works whichever import path it uses.
    monkeypatch.setitem(sys.modules, "llm_engineering.rag.answer", stub)
    monkeypatch.setitem(sys.modules, "llm_engineering.rag.custom.answer", stub)
    return stub


@pytest.fixture
def evaluate_module(stub_answer, monkeypatch):
    """Import llm_engineering.evals.evaluate against the stubbed answer module."""
    monkeypatch.delitem(sys.modules, "llm_engineering.evals.evaluate", raising=False)
    module = importlib.import_module("llm_engineering.evals.evaluate")
    yield module
    sys.modules.pop("llm_engineering.evals.evaluate", None)


@pytest.fixture
def dashboard_module(evaluate_module, monkeypatch):
    """Import llm_engineering.evals.dashboard (it imports the evaluate module)."""
    monkeypatch.delitem(sys.modules, "llm_engineering.evals.dashboard", raising=False)
    module = importlib.import_module("llm_engineering.evals.dashboard")
    yield module
    sys.modules.pop("llm_engineering.evals.dashboard", None)


# ======================================================================================
# Fakes and fixtures for the rag/custom pipeline (answer.py, ingest.py, app.py)
#
# Nothing here talks to Ollama, OpenAI or a real vector database. The modules are
# imported fresh for every test, with .env loading and retry waiting switched off.
# ======================================================================================

ANSWER_MODULES = ("llm_engineering.rag.custom.answer",)
# ingest.py sits in rag/ in your tree; the second name covers it being moved into custom/.
INGEST_MODULES = ("llm_engineering.rag.custom.ingest", "llm_engineering.rag.ingest")
APP_MODULES = ("llm_engineering.rag.custom.app",)


class TooManyCalls(BaseException):
    """
    Raised when a fake is called more often than the test planned for.

    It derives from BaseException on purpose: tenacity retries ordinary exceptions, and
    ingest.py's retry has no attempt limit, so an ordinary exception would hang the test.
    """


class FakeCollection:
    """Just enough of a Chroma collection for answer.py and ingest.py."""

    def __init__(self, documents=None, metadatas=None):
        self.documents = list(documents or [])
        self.metadatas = list(metadatas or [])
        self.query_calls = []
        self.add_calls = []

    def query(self, query_embeddings, n_results):
        self.query_calls.append({"query_embeddings": query_embeddings, "n_results": n_results})
        return {
            "documents": [self.documents[:n_results]],
            "metadatas": [self.metadatas[:n_results]],
        }

    def add(self, **kwargs):
        self.add_calls.append(kwargs)

    def count(self):
        return sum(len(call["ids"]) for call in self.add_calls)


class FakeChromaClient:
    """Stands in for chromadb.PersistentClient. Nothing touches the disk."""

    instances = []

    def __init__(self, path=None, existing=()):
        self.path = path
        self.collections = {name: FakeCollection() for name in existing}
        self.deleted = []
        FakeChromaClient.instances.append(self)

    def list_collections(self):
        return [SimpleNamespace(name=name) for name in self.collections]

    def delete_collection(self, name):
        self.deleted.append(name)
        self.collections.pop(name, None)

    def get_or_create_collection(self, name):
        return self.collections.setdefault(name, FakeCollection())


class FakeEmbeddings:
    """
    Stands in for openai.embeddings. The vector depends only on the text's length, so a test
    can check that every stored vector sits next to the document it was made from.
    """

    def __init__(self, reverse=False, fail_on_call=None):
        self.reverse = reverse  # answer in the wrong order, like a server might
        self.fail_on_call = fail_on_call  # 1-based number of the call that should fail
        self.calls = []

    def create(self, model, input):
        self.calls.append({"model": model, "input": list(input)})
        if self.fail_on_call == len(self.calls):
            raise RuntimeError("embedding server unavailable")
        data = [
            SimpleNamespace(index=position, embedding=[float(len(text)), 1.0, 0.0])
            for position, text in enumerate(input)
        ]
        if self.reverse:
            data.reverse()
        return SimpleNamespace(data=data)


def make_fake_completion(*replies):
    """
    Build a stand-in for litellm.completion that returns the given replies in order.
    A reply that is an Exception is raised instead. Every call's kwargs are kept in .calls.
    """
    queue = list(replies)
    calls = []

    def completion(**kwargs):
        calls.append(kwargs)
        if not queue:
            raise TooManyCalls(f"completion called {len(calls)} times, expected {len(replies)}")
        reply = queue.pop(0)
        if isinstance(reply, Exception):
            raise reply
        message = SimpleNamespace(content=reply)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    completion.calls = calls
    return completion


def _import_fresh(*names):
    """Import the first of `names` that exists, running its module code again from scratch."""
    for name in names:
        sys.modules.pop(name, None)
    for name in names:
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError as error:
            missing = error.name or ""
            if name != missing and not name.startswith(missing + "."):
                raise  # a dependency of the module is missing, not the module itself
    raise ModuleNotFoundError(f"none of these modules could be imported: {', '.join(names)}")


def _forget(*names):
    for name in names:
        sys.modules.pop(name, None)


@pytest.fixture
def fakes():
    """The fake services, bundled: fakes.Collection, .ChromaClient, .Embeddings, .completion."""
    return SimpleNamespace(
        Collection=FakeCollection,
        ChromaClient=FakeChromaClient,
        Embeddings=FakeEmbeddings,
        completion=make_fake_completion,
    )


@pytest.fixture
def safe_imports(monkeypatch):
    """Importing answer.py / ingest.py / app.py must not read .env or sleep between retries."""
    import dotenv
    import tenacity

    monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: False)
    monkeypatch.setattr(tenacity, "wait_exponential", lambda *args, **kwargs: tenacity.wait_none())


@pytest.fixture
def custom_answer(safe_imports, monkeypatch):
    """rag/custom/answer.py, imported with a fake Chroma client so no database is opened."""
    import chromadb

    FakeChromaClient.instances.clear()
    with monkeypatch.context() as patch:
        # answer.py opens its database while it is being imported. The patch only lasts for
        # the import; the module keeps the fake client it grabbed.
        patch.setattr(chromadb, "PersistentClient", FakeChromaClient)
        module = _import_fresh(*ANSWER_MODULES)
    yield module
    _forget(*ANSWER_MODULES)


@pytest.fixture
def ingest_module(safe_imports):
    """ingest.py (the version that chunks with an LLM and embeds with Ollama)."""
    module = _import_fresh(*INGEST_MODULES)
    if not hasattr(module, "create_embeddings_batch"):
        pytest.fail(
            f"{module.__name__} has no create_embeddings_batch, so it looks like a different "
            "ingest.py. Update INGEST_MODULES in tests/conftest.py."
        )
    yield module
    _forget(*INGEST_MODULES)


@pytest.fixture
def custom_app(safe_imports, monkeypatch):
    """rag/custom/app.py, with answer.py replaced by a stub so nothing heavy is loaded."""
    stub = types.ModuleType("answer")
    stub.answer_question = lambda question, history=None: ("stub answer", [])
    # app.py says `from answer import ...` when run as a script; cover the package path too.
    monkeypatch.setitem(sys.modules, "answer", stub)
    monkeypatch.setitem(sys.modules, "llm_engineering.rag.custom.answer", stub)
    module = _import_fresh(*APP_MODULES)
    yield module
    _forget(*APP_MODULES)
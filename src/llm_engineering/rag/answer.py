from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, convert_to_messages
from langchain_core.documents import Document
from dotenv import load_dotenv

from llm_engineering.config import DB_NAME, EMBEDDING_MODEL

load_dotenv(override=True)

MODEL = "gpt-6-luna"   # check this is valid for your account; see note below
RETRIEVAL_K = 3

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant representing the company Insurellm.
You are chatting with a user about Insurellm.
If relevant, use the given context to answer any question.
If you don't know the answer, say so.
Context:
{context}
"""

vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})
llm = ChatOpenAI(temperature=0, model_name=MODEL, reasoning_effort="none")


def fetch_context(question: str) -> list[Document]:
    return retriever.invoke(question)


def combined_question(question: str, history: list[dict] | None = None) -> str:
    history = history or []
    prior = "\n".join(m["content"] for m in history if m["role"] == "user")
    return prior + "\n" + question


def answer_question(question: str, history: list[dict] | None = None) -> tuple[str, list[Document]]:
    history = history or []
    combined = combined_question(question, history)
    docs = fetch_context(combined)
    context = "\n\n".join(doc.page_content for doc in docs)
    system_prompt = SYSTEM_PROMPT.format(context=context)
    messages = [SystemMessage(content=system_prompt)]
    messages.extend(convert_to_messages(history))
    messages.append(HumanMessage(content=question))
    response = llm.invoke(messages)
    return response.content, docs
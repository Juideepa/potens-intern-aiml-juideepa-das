import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag_core import (
    ask_question,
    build_vector_store,
    contradict,
    create_embeddings,
    create_llm,
    list_sample_document_paths,
    load_documents_from_paths,
)


load_dotenv()

app = FastAPI(
    title="DocMind AI Document API",
    description="API for asking questions about documents and comparing document details.",
    version="1.0.0",
)

state = {
    "llm": None,
    "embeddings": None,
    "vector_store": None,
    "chunks": [],
    "doc_ids": [],
}


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=10)


class ContradictRequest(BaseModel):
    document_id_a: str = Field(..., min_length=1)
    document_id_b: str = Field(..., min_length=1)
    topic: str = Field(..., min_length=1)
    top_k: int = Field(default=8, ge=1, le=12)


def require_env(name):
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def ensure_corpus():
    if state["vector_store"] is None:
        raise HTTPException(status_code=503, detail="Documents are not ready yet.")


@app.on_event("startup")
def startup():
    groq_api_key = require_env("groq_api_key")
    google_api_key = require_env("google_api_key")

    state["llm"] = create_llm(groq_api_key)
    state["embeddings"] = create_embeddings(google_api_key)

    sample_paths = list_sample_document_paths(os.getcwd())
    if len(sample_paths) < 5:
        raise RuntimeError("The sample documents folder must contain at least five documents.")

    docs = load_documents_from_paths(sample_paths)
    vector_store, chunks = build_vector_store(docs, state["embeddings"])
    state["vector_store"] = vector_store
    state["chunks"] = chunks
    state["doc_ids"] = sorted(
        {chunk.metadata.get("doc_id") for chunk in chunks if chunk.metadata.get("doc_id")}
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "documents": len(state["doc_ids"]),
        "chunks": len(state["chunks"]),
        "document_ids": state["doc_ids"],
    }


@app.post("/ask")
def ask(payload: AskRequest):
    ensure_corpus()
    return ask_question(
        state["vector_store"],
        state["llm"],
        payload.question,
        top_k=payload.top_k,
    )


@app.post("/contradict")
def check_contradiction(payload: ContradictRequest):
    ensure_corpus()
    if payload.document_id_a == payload.document_id_b:
        raise HTTPException(
            status_code=400,
            detail="Choose two different document IDs.",
        )

    return contradict(
        state["vector_store"],
        state["llm"],
        payload.document_id_a,
        payload.document_id_b,
        payload.topic,
        top_k=payload.top_k,
    )

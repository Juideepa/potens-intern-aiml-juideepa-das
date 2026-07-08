import os
import tempfile
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter


CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
DEFAULT_TOP_K = 5
LOW_CONFIDENCE_THRESHOLD = 0.18
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def create_llm(groq_api_key):
    return ChatGroq(
        groq_api_key=groq_api_key,
        model_name="openai/gpt-oss-120b",
        temperature=0,
    )


def create_embeddings(google_api_key):
    os.environ["GOOGLE_API_KEY"] = google_api_key
    return GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")


def load_documents_from_paths(paths):
    docs = []

    for path in paths:
        file_path = Path(path)
        suffix = file_path.suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            continue

        if suffix == ".pdf":
            loaded_docs = PyPDFLoader(str(file_path)).load()
            for page_number, doc in enumerate(loaded_docs, start=1):
                doc.metadata.update(
                    {
                        "source": file_path.name,
                        "doc_id": file_path.stem,
                        "page": page_number,
                    }
                )
                docs.append(doc)
        else:
            text = file_path.read_text(encoding="utf-8")
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": file_path.name,
                        "doc_id": file_path.stem,
                        "page": "n/a",
                    },
                )
            )

    return docs


def save_uploaded_files(uploaded_files):
    paths = []

    for uploaded_file in uploaded_files:
        suffix = Path(uploaded_file.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            paths.append((tmp.name, uploaded_file.name))

    return paths


def load_documents_from_uploads(uploaded_files):
    temp_paths = save_uploaded_files(uploaded_files)
    docs = []

    for temp_path, original_name in temp_paths:
        file_path = Path(temp_path)
        original_path = Path(original_name)
        suffix = original_path.suffix.lower()

        if suffix == ".pdf":
            loaded_docs = PyPDFLoader(str(file_path)).load()
            for page_number, doc in enumerate(loaded_docs, start=1):
                doc.metadata.update(
                    {
                        "source": original_name,
                        "doc_id": original_path.stem,
                        "page": page_number,
                    }
                )
                docs.append(doc)
        elif suffix in {".txt", ".md"}:
            text = file_path.read_text(encoding="utf-8")
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": original_name,
                        "doc_id": original_path.stem,
                        "page": "n/a",
                    },
                )
            )

    return docs


def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )
    chunks = splitter.split_documents(docs)

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index
        chunk.metadata["chunk_ref"] = f"chunk-{index}"

    return chunks


def build_vector_store(docs, embeddings):
    chunks = chunk_documents(docs)
    if not chunks:
        raise ValueError("No readable content found in the selected documents.")

    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store, chunks


def list_sample_document_paths(base_dir):
    sample_dir = Path(base_dir) / "sample_documents"
    return sorted(
        path
        for path in sample_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def format_context(search_results):
    context_blocks = []

    for rank, (doc, score) in enumerate(search_results, start=1):
        metadata = doc.metadata
        context_blocks.append(
            "\n".join(
                [
                    f"[Citation {rank}]",
                    f"Source: {metadata.get('source')}",
                    f"Document ID: {metadata.get('doc_id')}",
                    f"Page: {metadata.get('page')}",
                    f"Chunk: {metadata.get('chunk_ref')}",
                    f"Snippet: {doc.page_content}",
                ]
            )
        )

    return "\n\n".join(context_blocks)


def confidence_from_scores(search_results):
    if not search_results:
        return 0.0

    avg_distance = sum(float(score) for _, score in search_results) / len(search_results)
    return round(max(0.0, min(1.0, 1.0 / (1.0 + avg_distance))), 2)


def make_citations(search_results):
    citations = []

    for doc, score in search_results:
        metadata = doc.metadata
        citations.append(
            {
                "source_file": metadata.get("source"),
                "document_id": metadata.get("doc_id"),
                "page": metadata.get("page"),
                "chunk": metadata.get("chunk_ref"),
                "snippet": doc.page_content[:650].strip(),
                "retrieval_score": round(float(score), 4),
            }
        )

    return citations


ASK_PROMPT = ChatPromptTemplate.from_template(
    """
You are a careful multilingual RAG assistant.
Answer the question using only the cited context.
Return the answer in the same language as the user's question.
If the context does not contain the answer, say exactly:
"The provided documents do not cover this question."
Include short inline citation markers like [Citation 1] when using evidence.

Context:
{context}

Question:
{question}
"""
)


CONTRADICTION_PROMPT = ChatPromptTemplate.from_template(
    """
You compare two documents for conflict on one topic.
Use only the provided excerpts.
If the excerpts do not cover the topic well enough, say that there is not enough evidence.

Topic:
{topic}

Document A excerpts:
{doc_a_context}

Document B excerpts:
{doc_b_context}

Return:
- conflict: yes, no, or insufficient evidence
- reasoning: concise explanation with citation markers
"""
)


def ask_question(vector_store, llm, question, top_k=DEFAULT_TOP_K):
    search_results = vector_store.similarity_search_with_score(question, k=top_k)
    confidence = confidence_from_scores(search_results)
    citations = make_citations(search_results)

    if not search_results or confidence < LOW_CONFIDENCE_THRESHOLD:
        return {
            "answer": "The provided documents do not cover this question.",
            "confidence": confidence,
            "needs_human_review": True,
            "citations": citations,
        }

    context = format_context(search_results)
    response = llm.invoke(ASK_PROMPT.format_messages(context=context, question=question))

    return {
        "answer": response.content,
        "confidence": confidence,
        "needs_human_review": confidence < 0.45,
        "citations": citations,
    }


def filter_results_for_doc(search_results, doc_id):
    return [
        (doc, score)
        for doc, score in search_results
        if doc.metadata.get("doc_id") == doc_id
    ]


def contradict(vector_store, llm, document_id_a, document_id_b, topic, top_k=8):
    search_results = vector_store.similarity_search_with_score(topic, k=top_k * 3)
    doc_a_results = filter_results_for_doc(search_results, document_id_a)[:top_k]
    doc_b_results = filter_results_for_doc(search_results, document_id_b)[:top_k]

    if not doc_a_results or not doc_b_results:
        return {
            "conflict": "insufficient evidence",
            "reasoning": "One or both documents did not retrieve enough topic-specific evidence.",
            "citations": {
                document_id_a: make_citations(doc_a_results),
                document_id_b: make_citations(doc_b_results),
            },
        }

    response = llm.invoke(
        CONTRADICTION_PROMPT.format_messages(
            topic=topic,
            doc_a_context=format_context(doc_a_results),
            doc_b_context=format_context(doc_b_results),
        )
    )

    return {
        "conflict": "see reasoning",
        "reasoning": response.content,
        "citations": {
            document_id_a: make_citations(doc_a_results),
            document_id_b: make_citations(doc_b_results),
        },
    }

# DocMind AI - Multilingual RAG With Citations

DocMind AI is a Retrieval-Augmented Generation project built over five substantive documents. It ingests documents, chunks them, embeds them with Google Gemini embeddings, stores them in FAISS, and answers questions with source citations using a Groq-hosted LLM.

The project includes:

- A purple and white Streamlit UI for document ingestion, asking questions, and contradiction checks.
- A FastAPI API with `/ask` and `/contradict` endpoints.
- A bundled five-document sample corpus in `sample_documents/`.
- A small 10-question evaluation set in `eval_set.json`.
- No silent hallucination: if retrieved evidence is weak or missing, the app says the documents do not cover the question.

## Sample Corpus

The included corpus contains five markdown documents:

- `support_policy.md`
- `billing_policy.md`
- `data_retention_policy.md`
- `security_runbook.md`
- `enterprise_sla.md`

You can also upload five or more PDF, TXT, or Markdown documents through the Streamlit UI.

## Chunking Strategy

Documents are split with `RecursiveCharacterTextSplitter` using:

- Chunk size: 900 characters
- Overlap: 150 characters
- Separators: paragraph, line, sentence, word, and character fallbacks
- Metadata: source file, document ID, page reference, chunk ID, and chunk reference

The chunk size is intentionally moderate so each retrieved passage usually contains enough surrounding context for the LLM to answer accurately. The 150-character overlap reduces boundary loss when an answer spans the end of one chunk and the start of the next. PDF pages keep their page numbers; TXT and Markdown files use `page: n/a`.

## Hallucination Control

The answer prompt instructs the model to use only retrieved context and to answer in the same language as the query. If the evidence does not cover the question, the response must be:

```text
The provided documents do not cover this question.
```

The app also computes a simple confidence score from FAISS retrieval distances. Low-confidence retrieval is gated before generation and flagged for human review.

## Requirements Mapping

| Requirement | Implementation |
| --- | --- |
| Ingest, chunk, embed, and store documents | `rag_core.py` loads PDF/TXT/MD, chunks with metadata, embeds with Gemini, stores in FAISS |
| Explain chunking strategy | See "Chunking Strategy" above |
| `/ask` endpoint with citations | `api.py` exposes `POST /ask`; citations include source file, page, chunk, snippet |
| `/contradict` endpoint | `api.py` exposes `POST /contradict` with two document IDs and a topic |
| Multilingual flow | The prompt answers in the same language as the user query |
| Simple UI | `app.py` provides the Streamlit UI |
| No silent hallucination | Prompt and confidence gate force an explicit not-covered answer |
| Vector store and LLM | FAISS, Gemini embeddings, Groq LLM |
| Stretch: confidence score | Included in `/ask` and UI |
| Stretch: eval set | `eval_set.json` contains 10 Q&A pairs |

## Setup

Create a `.env` file in the project folder:

```env
groq_api_key=your_groq_key
google_api_key=your_google_key
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Run The Streamlit UI

```powershell
streamlit run app.py
```

In the sidebar, choose either:

- `Sample corpus` and click `Build sample corpus`
- `Upload documents` and upload at least five supported files

## Run The API

```powershell
uvicorn api:app --reload
```

Health check:

```http
GET http://127.0.0.1:8000/health
```

Ask endpoint:

```http
POST http://127.0.0.1:8000/ask
Content-Type: application/json

{
  "question": "What is the refund window?",
  "top_k": 5
}
```

Contradiction endpoint:

```http
POST http://127.0.0.1:8000/contradict
Content-Type: application/json

{
  "document_id_a": "support_policy",
  "document_id_b": "enterprise_sla",
  "topic": "Severity 1 initial response time",
  "top_k": 8
}
```

## API Response Shape

`/ask` returns:

```json
{
  "answer": "Answer with inline citation markers.",
  "confidence": 0.72,
  "needs_human_review": false,
  "citations": [
    {
      "source_file": "billing_policy.md",
      "document_id": "billing_policy",
      "page": "n/a",
      "chunk": "chunk-1",
      "snippet": "Customers may request a refund within 14 calendar days...",
      "retrieval_score": 0.4312
    }
  ]
}
```

`/contradict` returns reasoning plus citations from both documents.

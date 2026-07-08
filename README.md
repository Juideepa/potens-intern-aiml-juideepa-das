# DocMind AI - Multilingual RAG With Citations

DocMind AI is a Retrieval-Augmented Generation project built over five substantive documents. It ingests documents, chunks them, embeds them with Google Gemini embeddings, stores them in FAISS, and answers questions with source citations using a Groq-hosted LLM.

The project includes:

- A purple and white Streamlit UI for document ingestion, asking questions, and contradiction checks.
- A FastAPI API with `/ask` and `/contradict` endpoints.
- A bundled five-document sample set in `sample_documents/`.
- A small 10-question evaluation set in `eval_set.json`.
- No silent hallucination: if retrieved evidence is weak or missing, the app says the documents do not cover the question.

## Sample Documents

The included sample document set contains five markdown documents:

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

## Design Decisions

- Streamlit is used for the main interface because it keeps the document upload, retrieval controls, and answer/citation display simple to run locally.
- FastAPI is included separately so the same RAG pipeline can be called from scripts, tests, or other applications.
- FAISS is used as the local vector store because it is lightweight and does not require a hosted database for this project.
- Google Gemini embeddings are used for semantic retrieval, while the Groq-hosted chat model is used for answer generation.
- The app requires at least five uploaded documents so retrieval and contradiction checks are tested against a meaningful multi-document set.
- Citations are attached to every answer using source file, document ID, page, chunk ID, snippet, and retrieval score so responses can be audited.

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

Create a `.env` file in the project folder. You can copy the variable names from `.env.example`:

```env
groq_api_key=your_groq_key
google_api_key=your_google_key
```

Do not commit your real `.env` file. It contains local secrets and is ignored by Git.

Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

```powershell
pip install -r requirements.txt
```

## Run The Streamlit UI

```powershell
streamlit run app.py
```

In the sidebar, choose either:

- `Sample documents` and click `Build sample documents`
- `Upload documents` and upload at least five supported files

## Run The API

```powershell
uvicorn api:app --reload
```

After the server starts, open the interactive API documentation at:

```text
http://127.0.0.1:8000/docs
```

The API includes:

- `GET /health` to check whether the server is running.
- `POST /ask` to ask questions about the loaded documents.
- `POST /contradict` to compare two documents on a specific topic.

## Next Steps

- Add automated tests for document loading, chunking, confidence gating, and API response shapes.
- Add a small evaluation runner that scores answers against `eval_set.json`.
- Persist vector indexes between runs so large uploaded document sets do not need to be rebuilt every session.
- Add authentication or rate limiting before deploying the FastAPI service publicly.
- Improve contradiction detection with a structured claim extraction step.

## AI Usage

AI tools were used during development and documentation:

- VS Code Codex was used to help write and complete the code for this project.
- It helped build and polish the Streamlit app, FastAPI endpoints, RAG pipeline structure, document loading/chunking flow, citation handling, and README documentation.
- VS Code Codex was also used to prepare the repository for GitHub by adding a safer `.gitignore`, creating a non-secret `.env.example` template, and removing generated/local files such as logs and `__pycache__` files from Git tracking.

import os
import time

import streamlit as st
from dotenv import load_dotenv

from rag_core import (
    ask_question,
    build_vector_store,
    contradict,
    create_embeddings,
    create_llm,
    list_sample_document_paths,
    load_documents_from_paths,
    load_documents_from_uploads,
)


st.set_page_config(
    page_title="DocMind AI",
    page_icon="DM",
    layout="wide",
)

load_dotenv()


def get_config_value(name):
    try:
        value = st.secrets.get(name)
    except Exception:
        value = None

    value = value or os.getenv(name)
    if not value:
        st.error(
            f"Missing `{name}`. Add it to `.env` for local development "
            "or `.streamlit/secrets.toml` for Streamlit Cloud."
        )
        st.stop()

    return value


def inject_styles():
    st.markdown(
        """
        <style>
        :root {
            --purple: #7c3aed;
            --purple-dark: #24113f;
            --violet: #a855f7;
            --violet-soft: #f4ecff;
            --ink: #1b1528;
            --muted: #6d6577;
            --line: #ddd6e8;
            --grey: #f5f3f7;
            --white: #ffffff;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(168, 85, 247, 0.24), transparent 26rem),
                radial-gradient(circle at bottom right, rgba(76, 29, 149, 0.16), transparent 24rem),
                linear-gradient(180deg, #fbfaff 0%, #f4f1f7 48%, #ffffff 100%);
            color: var(--ink);
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #ffffff 0%, #f7f3fb 100%);
            border-right: 1px solid var(--line);
        }

        .hero {
            background: linear-gradient(135deg, var(--purple-dark) 0%, var(--purple) 55%, var(--violet) 100%);
            color: #ffffff;
            padding: 2.25rem;
            border-radius: 8px;
            margin-bottom: 1.25rem;
            box-shadow: 0 22px 60px rgba(76, 29, 149, 0.20);
        }

        .hero h1 {
            margin: 0 0 0.6rem 0;
            font-size: 2.4rem;
            line-height: 1.05;
            letter-spacing: 0;
        }

        .hero p {
            margin: 0;
            color: rgba(255,255,255,0.88);
            font-size: 1.02rem;
            max-width: 760px;
        }

        .metric-row {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.75rem;
            margin: 1rem 0 1.25rem 0;
        }

        .metric-box {
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 1rem;
            box-shadow: 0 10px 28px rgba(36, 17, 63, 0.06);
        }

        .metric-box span {
            display: block;
            color: var(--muted);
            font-size: 0.82rem;
            margin-bottom: 0.3rem;
        }

        .metric-box strong {
            color: var(--purple-dark);
            font-size: 1.45rem;
        }

        .citation {
            background: #ffffff;
            border: 1px solid var(--line);
            border-left: 4px solid var(--violet);
            border-radius: 8px;
            padding: 0.85rem 1rem;
            margin-bottom: 0.75rem;
        }

        .citation small {
            color: var(--purple-dark);
            font-weight: 700;
        }

        .stButton > button {
            background: linear-gradient(135deg, var(--purple) 0%, var(--violet) 100%);
            border: 1px solid var(--purple);
            color: white;
            border-radius: 8px;
            font-weight: 700;
        }

        .stButton > button:hover {
            background: var(--purple-dark);
            border-color: var(--purple-dark);
            color: white;
        }

        div[data-testid="stTextInput"] input,
        div[data-testid="stTextArea"] textarea,
        div[data-testid="stSelectbox"] div {
            border-radius: 8px;
        }

        div[data-testid="stAlert"] {
            border-radius: 8px;
        }

        @media (max-width: 760px) {
            .hero {
                padding: 1.5rem;
            }

            .hero h1 {
                font-size: 1.8rem;
            }

            .metric-row {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def initialize_models():
    groq_api_key = get_config_value("groq_api_key")
    google_api_key = get_config_value("google_api_key")

    if "llm" not in st.session_state:
        st.session_state.llm = create_llm(groq_api_key)

    if "embeddings" not in st.session_state:
        st.session_state.embeddings = create_embeddings(google_api_key)


def set_corpus(docs):
    vector_store, chunks = build_vector_store(docs, st.session_state.embeddings)
    st.session_state.vector_store = vector_store
    st.session_state.chunks = chunks
    st.session_state.doc_ids = sorted(
        {chunk.metadata.get("doc_id") for chunk in chunks if chunk.metadata.get("doc_id")}
    )


def format_section_label(section_id):
    return str(section_id or "section").replace("chunk-", "section ")


inject_styles()
initialize_models()

with st.sidebar:
    st.markdown("### Documents")
    corpus_mode = st.radio(
        "Choose documents",
        ["Use sample documents", "Upload my documents"],
        label_visibility="collapsed",
    )

    if corpus_mode == "Use sample documents":
        sample_paths = list_sample_document_paths(os.getcwd())
        st.caption(f"{len(sample_paths)} sample documents are ready to use.")
        if st.button("Process sample documents", use_container_width=True):
            with st.spinner("Reading and preparing the sample documents..."):
                set_corpus(load_documents_from_paths(sample_paths))
            st.success("Sample documents are ready.")
    else:
        uploaded_files = st.file_uploader(
            "Upload at least five PDF, TXT, or MD files",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
        )
        if uploaded_files and len(uploaded_files) < 5:
            st.warning("Please upload five or more documents.")
        if uploaded_files and len(uploaded_files) >= 5:
            if st.button("Process uploaded documents", use_container_width=True):
                with st.spinner("Reading and preparing your documents..."):
                    set_corpus(load_documents_from_uploads(uploaded_files))
                st.success("Your documents are ready.")

    st.markdown("### Document Status")
    st.caption("After processing, you can ask questions and compare documents.")

st.markdown(
    """
    <div class="hero">
        <h1>DocMind AI</h1>
        <p>Ask questions about your documents, get clear answers with source references, and compare files for conflicting details.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

doc_count = len(st.session_state.get("doc_ids", []))
chunk_count = len(st.session_state.get("chunks", []))
ready_label = "Ready" if "vector_store" in st.session_state else "Not ready"
st.markdown(
    f"""
    <div class="metric-row">
        <div class="metric-box"><span>Document status</span><strong>{ready_label}</strong></div>
        <div class="metric-box"><span>Documents ready</span><strong>{doc_count}</strong></div>
        <div class="metric-box"><span>Text sections found</span><strong>{chunk_count}</strong></div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_ask, tab_contradict = st.tabs(["Ask questions", "Compare documents"])

with tab_ask:
    if "vector_store" not in st.session_state:
        st.info("Process the sample documents or upload five or more documents to begin.")
    else:
        question = st.text_area(
            "Question",
            placeholder="Ask in English, Hindi, Bengali, Spanish, or another language. The answer will match your language.",
            height=110,
        )
        if st.button("Ask", use_container_width=True):
            if not question.strip():
                st.warning("Enter a question first.")
            else:
                with st.spinner("Finding the best document details and writing an answer..."):
                    start = time.time()
                    result = ask_question(
                        st.session_state.vector_store,
                        st.session_state.llm,
                        question.strip(),
                    )
                    elapsed = time.time() - start

                st.markdown("### Answer")
                st.write(result["answer"])

                col_a, col_b = st.columns(2)
                col_a.metric("Confidence", f"{result['confidence']:.2f}")
                col_b.metric("Needs review", "Yes" if result["needs_human_review"] else "No")
                st.caption(f"Response time: {elapsed:.2f} seconds")

                st.markdown("### Sources")
                for citation in result["citations"]:
                    section_label = format_section_label(citation["chunk"])
                    st.markdown(
                        f"""
                        <div class="citation">
                            <small>{citation["source_file"]} | page {citation["page"]} | {section_label}</small>
                            <p>{citation["snippet"]}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

with tab_contradict:
    if "vector_store" not in st.session_state:
        st.info("Process documents before comparing them.")
    elif len(st.session_state.doc_ids) < 2:
        st.warning("At least two documents are needed for comparison.")
    else:
        col_a, col_b = st.columns(2)
        doc_a = col_a.selectbox("Document A", st.session_state.doc_ids, key="doc_a")
        doc_b = col_b.selectbox("Document B", st.session_state.doc_ids, key="doc_b")
        topic = st.text_input("Topic", placeholder="Example: refund window, escalation timing, data retention")

        if st.button("Compare documents", use_container_width=True):
            if doc_a == doc_b:
                st.warning("Choose two different documents.")
            elif not topic.strip():
                st.warning("Enter a topic to compare.")
            else:
                with st.spinner("Finding related details and checking for differences..."):
                    result = contradict(
                        st.session_state.vector_store,
                        st.session_state.llm,
                        doc_a,
                        doc_b,
                        topic.strip(),
                    )

                st.markdown("### Result")
                st.write(result["reasoning"])

                st.markdown("### Evidence")
                for doc_id, citations in result["citations"].items():
                    st.markdown(f"**{doc_id}**")
                    for citation in citations:
                        section_label = format_section_label(citation["chunk"])
                        st.markdown(
                            f"""
                            <div class="citation">
                                <small>{citation["source_file"]} | page {citation["page"]} | {section_label}</small>
                                <p>{citation["snippet"]}</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

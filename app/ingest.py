import os
import warnings
from pathlib import Path

from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

VECTORSTORE_PATH = "./vectorstore"
PDF_PATH = "./data/promtior.pdf"
WEB_URL = "https://promtior.ai"


def load_documents():
    docs = []

    # Load web content
    try:
        loader = WebBaseLoader(WEB_URL)
        web_docs = loader.load()
        docs.extend(web_docs)
        print(f"[ingest] Loaded {len(web_docs)} document(s) from {WEB_URL}")
    except Exception as e:
        warnings.warn(f"[ingest] Failed to load web content: {e}")

    # Load PDF
    if Path(PDF_PATH).exists():
        try:
            loader = PyPDFLoader(PDF_PATH)
            pdf_docs = loader.load()
            docs.extend(pdf_docs)
            print(f"[ingest] Loaded {len(pdf_docs)} page(s) from {PDF_PATH}")
        except Exception as e:
            warnings.warn(f"[ingest] Failed to load PDF: {e}")
    else:
        warnings.warn(
            f"[ingest] PDF not found at {PDF_PATH} — skipping. "
            "Place promtior.pdf in the data/ directory to include it."
        )

    return docs


def create_vectorstore():
    docs = load_documents()
    if not docs:
        raise RuntimeError("[ingest] No documents loaded. Cannot build vector store.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    print(f"[ingest] Split into {len(chunks)} chunks")

    embeddings = OpenAIEmbeddings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)

    os.makedirs(VECTORSTORE_PATH, exist_ok=True)
    vectorstore.save_local(VECTORSTORE_PATH)
    print(f"[ingest] Vector store saved to {VECTORSTORE_PATH}")

    return vectorstore


def get_or_create_vectorstore():
    embeddings = OpenAIEmbeddings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )

    index_file = Path(VECTORSTORE_PATH) / "index.faiss"
    if index_file.exists():
        print(f"[ingest] Loading existing vector store from {VECTORSTORE_PATH}")
        return FAISS.load_local(
            VECTORSTORE_PATH,
            embeddings,
            allow_dangerous_deserialization=True,
        )

    print("[ingest] No existing vector store found — creating new one")
    return create_vectorstore()

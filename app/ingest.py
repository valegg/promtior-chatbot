import os
import urllib.request
import warnings
from pathlib import Path

from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

VECTORSTORE_PATH = "./vectorstore"
PDF_PATH = "./data/promtior.pdf"
WEB_URL = "https://promtior.ai"


def download_pdf_if_needed():
    """Download the PDF from PDF_URL env var if it isn't already on disk."""
    pdf_url = os.getenv("PDF_URL")
    if not pdf_url:
        return

    pdf_path = Path(PDF_PATH)
    if pdf_path.exists():
        print("[ingest] PDF already exists locally — skipping download.")
        return

    print(f"[ingest] Downloading PDF from {pdf_url} ...")
    try:
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(pdf_url, pdf_path)
        print(f"[ingest] PDF saved to {PDF_PATH}")
    except Exception as e:
        warnings.warn(f"[ingest] Failed to download PDF: {e}")


def load_documents():
    docs = []

    # Download PDF from remote URL if PDF_URL is set and file is missing
    download_pdf_if_needed()

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
            "Set PDF_URL env var or place promtior.pdf in the data/ directory."
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

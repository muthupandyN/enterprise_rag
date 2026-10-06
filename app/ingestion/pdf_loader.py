from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def load_pdf(pdf_path: Path) -> list[Document]:
    """Load a single PDF and return its pages as Documents."""

    loader = PyPDFLoader(str(pdf_path))

    documents = loader.load()

    for document in documents:
        document.metadata["source_file"] = pdf_path.name
        document.metadata["file_type"] = "pdf"

    return documents


def load_all_pdfs(pdf_directory: Path) -> list[Document]:
    """Load all PDF files recursively from a directory."""

    all_documents: list[Document] = []

    pdf_files = list(pdf_directory.glob("**/*.pdf"))

    print(f"Found {len(pdf_files)} PDF files")

    for pdf_file in pdf_files:
        print(f"Processing: {pdf_file}")

        documents = load_pdf(pdf_file)

        # print(f"documents print:{documents}")

        all_documents.extend(documents)

        print(f"Loaded {len(documents)} pages")

    return all_documents
# from pathlib import Path

# from app.config import settings
# from app.ingestion.pdf_loader import load_all_pdfs
# from app.ingestion.chunker import chunk_documents


# def main() -> None:
#     pdf_directory = Path("data/raw")

#     documents = load_all_pdfs(pdf_directory)

#     print(f"\nTotal documents: {len(documents)}")

#     chunks = chunk_documents(
#         documents,
#         chunk_size=settings.chunk_size,
#         chunk_overlap=settings.chunk_overlap,
#     )

#     print(f"Total chunks: {len(chunks)}")

#     if chunks:
#         print("\nFirst chunk:")
#         print(chunks[0].page_content)

#         print("\nMetadata:")
#         print(chunks[0].metadata)


# if __name__ == "__main__":
#     main()





from app.config import settings
from app.embeddings.service import EmbeddingService
from app.vectorstore.chroma_store import ChromaStore


def main() -> None:
    embedding_service = EmbeddingService(
        settings.embedding_model
    )

    texts = [
    "The application uses FastAPI.",
    "The API is implemented using FastAPI.",
    "The company provides annual leave to employees.",
]

    # vector = embedding_service.embed_text(text)
    vectors = embedding_service.embed_documents(texts)

    # print(f"Embedding model: {settings.embedding_model}")
    # print(f"Vector dimensions: {len(vectors)}")
    # print(f"First 5 values: {vectors[:5]}")

    similarity_ab = cosine_similarity(vectors[0], vectors[1])
    similarity_ac = cosine_similarity(vectors[0], vectors[2])


    vector_store = ChromaStore()
    vector_store.add_documents(
        ids=["doc1","doc2","doc3"],
        texts=texts,
        embeddings=vectors,
        metadatas=[{"source": "example"}] * len(texts),
    )

    print(f"A vs B: {similarity_ab:.4f}")
    print(f"A vs C: {similarity_ac:.4f}")


    print("Documents in collection:", vector_store.collection.count())


    # THEN ADD THE RETRIEVAL TEST HERE
    results = vector_store.collection.query(
        query_embeddings=[vectors[0]],
        n_results=2,
    )

    print("Documents:")
    print(results["documents"])

    print("\nIDs:")
    print(results["ids"])

    print("\nDistances:")
    print(results["distances"])




import numpy as np


def cosine_similarity(a: list[float], b: list[float]) -> float:
    a = np.array(a)
    b = np.array(b)

    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )


if __name__ == "__main__":
    main()
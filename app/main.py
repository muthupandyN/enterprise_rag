from pathlib import Path

from app.config import settings
from app.ingestion.pdf_loader import load_all_pdfs
from app.ingestion.chunker import chunk_documents
from app.embeddings.service import EmbeddingService
from app.vectorstore.chroma_store import ChromaStore

def main() -> None:
    pdf_directory = Path("data/raw")

    documents = load_all_pdfs(pdf_directory)

    print(f"\nTotal documents: {len(documents)}")

    chunks = chunk_documents(
        documents,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    metadatas = [chunk.metadata for chunk in chunks]


    chunk_ids = []

    for i, chunk in enumerate(chunks):
        source_file = chunk.metadata["source_file"]
        chunk_id = f"{source_file}_{i}"

        chunk_ids.append(chunk_id)

    print("Number of chunks:", len(chunks))
    print("First 5 chunk IDs:", chunk_ids[:5])

    texts = [chunk.page_content for chunk in chunks]


    embedding_service = EmbeddingService(
        settings.embedding_model
    )



    embeddings = embedding_service.embed_documents(texts)

    # print("Number of embeddings:", len(embeddings))
    # print("Embedding dimension:", len(embeddings[0]))

    # print("IDs:", len(chunk_ids))
    # print("Texts:", len(texts))
    # print("Embeddings:", len(embeddings))

    # assert len(chunk_ids) == len(texts) == len(embeddings)
    
    vector_store = ChromaStore()

    vector_store.reset_collection()
    vector_store.add_documents(
        ids=chunk_ids,
        texts=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    print("Documents in collection:", vector_store.collection.count())


    query = "Which framework is used to build the API?"

    query_embedding = embedding_service.embed_text(query)

    results = vector_store.search(
        query_embedding=query_embedding,
        top_k=3,
    )

    print("\nQuery:", query)

    print("\nRetrieved documents:")
    for i, document in enumerate(results["documents"][0]):
        print(f"\nRank {i + 1}:")
        print(document)

    print("\nIDs:")
    print(results["ids"][0])

    print("\nDistances:")
    print(results["distances"][0])

    print("\nMetadata:")
    print(results["metadatas"][0])

if __name__ == "__main__":
    main()





# from app.config import settings
# from app.embeddings.service import EmbeddingService
# from app.vectorstore.chroma_store import ChromaStore


# def main() -> None:
#     embedding_service = EmbeddingService(
#         settings.embedding_model
#     )

#     texts = [
#     "The application uses FastAPI.",
#     "The API is implemented using FastAPI.",
#     "The company provides annual leave to employees.",
# ]

#     # vector = embedding_service.embed_text(text)
#     vectors = embedding_service.embed_documents(texts)

#     # print(f"Embedding model: {settings.embedding_model}")
#     # print(f"Vector dimensions: {len(vectors)}")
#     # print(f"First 5 values: {vectors[:5]}")

#     similarity_ab = cosine_similarity(vectors[0], vectors[1])
#     similarity_ac = cosine_similarity(vectors[0], vectors[2])


#     vector_store = ChromaStore()
#     vector_store.add_documents(
#         ids=["doc1","doc2","doc3"],
#         texts=texts,
#         embeddings=vectors,
#         metadatas=[{"source": "example"}] * len(texts),
#     )


#     print(f"A vs B: {similarity_ab:.4f}")
#     print(f"A vs C: {similarity_ac:.4f}")


#     print("Documents in collection:", vector_store.collection.count())


#     query = "Which framework is used to build the API?"

#     query_embedding = embedding_service.embed_text(query)



#     # THEN ADD THE RETRIEVAL TEST HERE
#     results = vector_store.search(query_embedding=query_embedding, top_k=2)

#     print("Search results:")

#     results = vector_store.search(
#     query_embedding=query_embedding,
#     top_k=2,
#     )

#     print("\nQuery:", query)
#     print("Retrieved documents:")
#     print(results["documents"])

#     print("\nRetrieved IDs:")
#     print(results["ids"])

#     print("\nDistances:")
#     print(results["distances"])

   




# import numpy as np


# def cosine_similarity(a: list[float], b: list[float]) -> float:
#     a = np.array(a)
#     b = np.array(b)

#     return np.dot(a, b) / (
#         np.linalg.norm(a) * np.linalg.norm(b)
#     )


# if __name__ == "__main__":
#     main()
import chromadb


class ChromaStore:
    def __init__(self, collection_name: str = "enterprise_rag"):
        self.client = chromadb.PersistentClient(
            path="data/chroma"
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )

    def add_documents(
        self,
        ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
    ):
        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )


    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ):
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )


    def reset_collection(self):
        self.client.delete_collection(
            name=self.collection.name
        )

        self.collection = self.client.get_or_create_collection(
            name=self.collection.name
        )
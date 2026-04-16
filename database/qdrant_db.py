import uuid
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from config import QDRANT_HOST, QDRANT_PORT

class QdrantDB:
    def __init__(self, collection_name="messages", vector_size=768): # deepvk/USER-bge-m3 outputs 768 for base model
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self.collection_name = collection_name
        self.vector_size = vector_size
        self._ensure_collection()

    def _ensure_collection(self):
        """Creates collection if it doesn't exist."""
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)
        
        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
            )

    def add_message(self, user_id: str, text: str, embedding: list, timestamp: float, role: str):
        """Adds a message embedding to Qdrant."""
        point_id = str(uuid.uuid4())
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "user_id": user_id,
                        "text": text,
                        "timestamp": timestamp,
                        "role": role # "user" or "bot"
                    }
                )
            ]
        )

    def search_similar(self, user_id: str, query_embedding: list, limit: int = 5):
        """Searches for similar past messages in the user's history."""
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="user_id",
                        match=MatchValue(value=user_id)
                    )
                ]
            ),
            limit=limit
        )
        return results.points

qdrant_db = QdrantDB()

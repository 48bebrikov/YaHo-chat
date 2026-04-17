import uuid
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from config import QDRANT_HOST, QDRANT_PORT

# Payload: user_id, text, role, kind ("fact" | "dialogue_snippet"), sort_ts (float), event_utc_iso (str)


class QdrantDB:
    def __init__(self, collection_name="messages", vector_size=768):  # deepvk/USER2-base outputs 768 for base model
        self._client = None
        self.collection_name = collection_name
        self.vector_size = vector_size

    @property
    def client(self):
        """Connect lazily so importing this module does not require a running Qdrant (e.g. CI)."""
        if self._client is None:
            self._client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
            self._ensure_collection()
        return self._client

    def _ensure_collection(self):
        """Creates collection if it doesn't exist."""
        collections = self._client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)

        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
            )

    def upsert_memory_point(
        self,
        user_id: str,
        text: str,
        embedding: list,
        role: str,
        kind: str,
        sort_ts: float,
        event_utc_iso: str,
    ):
        """Одна точка памяти: fact или dialogue_snippet."""
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
                        "role": role,
                        "kind": kind,
                        "sort_ts": sort_ts,
                        "event_utc_iso": event_utc_iso,
                    },
                )
            ],
        )

    def search_similar_by_kind(
        self,
        user_id: str,
        query_embedding: list,
        kind: str,
        limit: int,
    ):
        """Поиск по сходству внутри одного kind и user_id."""
        return self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            query_filter=Filter(
                must=[
                    FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                    FieldCondition(key="kind", match=MatchValue(value=kind)),
                ]
            ),
            limit=limit,
        ).points


qdrant_db = QdrantDB()

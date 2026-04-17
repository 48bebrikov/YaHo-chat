from database.qdrant_db import QdrantDB


def test_qdrant_client_not_connected_until_used():
    db = QdrantDB(collection_name="ci_lazy")
    assert db._client is None
    assert db.collection_name == "ci_lazy"
    assert db.vector_size == 768

from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name="deepvk/USER2-base"):
        self.model = SentenceTransformer(model_name)
    
    def get_embedding(self, text: str) -> list[float]:
        """Returns the embedding for a single string."""
        embedding = self.model.encode(text)
        return embedding.tolist()
        
    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Returns embeddings for a list of strings."""
        embeddings = self.model.encode(texts)
        return [emb.tolist() for emb in embeddings]

embedder = Embedder()

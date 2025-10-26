import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


# ===== CONFIG =====
INDEX_DIR = "index"
MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5


class Retriever:
    def __init__(self, index_dir=INDEX_DIR, model_name=MODEL_NAME):
        self.index_dir = index_dir
        self.model = SentenceTransformer(model_name)
        self.index = faiss.read_index(os.path.join(index_dir, "index.faiss"))
        with open(os.path.join(index_dir, "metadata.json"), "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

    def retrieve(self, query, top_k=TOP_K):
        """Return top-k most similar chunks with cosine similarity."""
        query_emb = self.model.encode([query])
        D, I = self.index.search(np.array(query_emb, dtype="float32"), top_k)
        results = []
        for idx, score in zip(I[0], D[0]):
            if idx < len(self.metadata):
                results.append({
                    "chunk_id": self.metadata[idx]["chunk_id"],
                    "source": self.metadata[idx]["source"],
                    "text": self.metadata[idx]["text"],
                    "score": float(np.exp(-score))  # Convert L2 distance to pseudo-similarity
                })
        return results


if __name__ == "__main__":
    retriever = Retriever()
    query = input("Enter your query: ")
    results = retriever.retrieve(query)
    for r in results:
        print(f"\nScore: {r['score']:.4f} | Source: {r['source']}\n{r['text'][:250]}...")

import os
import json
import fitz  # PyMuPDF
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss

# ===== CONFIG =====
DATA_DIR = "data/docs"
INDEX_DIR = "index"
MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
OVERLAP = 50

# ===== UTILS =====
def get_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text")
    return text


def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ===== MAIN INGESTION =====
def ingest_documents(data_dir=DATA_DIR, index_dir=INDEX_DIR):
    os.makedirs(index_dir, exist_ok=True)

    model = SentenceTransformer(MODEL_NAME)
    documents = []
    embeddings = []

    for file in os.listdir(data_dir):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(data_dir, file)
            print(f"Processing {file} ...")
            text = get_text_from_pdf(pdf_path)
            chunks = chunk_text(text, CHUNK_SIZE, OVERLAP)

            for idx, chunk in enumerate(tqdm(chunks)):
                emb = model.encode(chunk)
                documents.append({
                    "source": file,
                    "chunk_id": f"{file}#{idx}",
                    "text": chunk
                })
                embeddings.append(emb)

    embeddings = np.array(embeddings, dtype="float32")

    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Save index and metadata
    faiss.write_index(index, os.path.join(index_dir, "index.faiss"))
    save_json(documents, os.path.join(index_dir, "metadata.json"))
    print(f"✅ Ingestion complete. Indexed {len(documents)} chunks.")


if __name__ == "__main__":
    ingest_documents()

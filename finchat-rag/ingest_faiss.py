# finchat-rag/ingest_faiss.py
import os
import json
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
import openai
import faiss
import numpy as np
import pdfplumber

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in .env")

openai.api_key = OPENAI_API_KEY

DATA_DIR = Path("finchat-rag/data")
INDEX_DIR = Path("finchat-rag/faiss_index")
INDEX_DIR.mkdir(parents=True, exist_ok=True)
META_FILE = INDEX_DIR / "metadata.json"
INDEX_FILE = INDEX_DIR / "faiss.index"

# simple tokenizer-ish chunker (by words)
def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50):
    words = text.split()
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i+chunk_size])
        yield chunk
        i += chunk_size - overlap

def load_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        pages = []
        with pdfplumber.open(path) as pdf:
            for p in pdf.pages:
                pages.append(p.extract_text() or "")
        return "\n".join(pages)
    else:
        return path.read_text(encoding="utf-8", errors="ignore")

def get_embedding(texts):
    # OpenAI embeddings accepts batched inputs
    resp = openai.Embedding.create(model=EMBED_MODEL, input=texts)
    # resp['data'] is list of embeddings in same order
    return [d["embedding"] for d in resp["data"]]

def main():
    # gather chunks & metadata
    all_texts = []
    metadatas = []
    files = list(DATA_DIR.glob("**/*.*"))
    print(f"Found {len(files)} files to ingest.")
    for f in files:
        txt = load_text(f)
        if not txt.strip():
            continue
        for i, chunk in enumerate(chunk_text(txt)):
            all_texts.append(chunk)
            metadatas.append({"source": str(f), "chunk_id": i})

    if not all_texts:
        print("No documents found in data directory. Put .txt or .pdf files under finchat-rag/data/")
        return

    # embed in batches (to avoid huge requests)
    BATCH = 32
    embeddings = []
    for i in tqdm(range(0, len(all_texts), BATCH), desc="Embedding batches"):
        batch_texts = all_texts[i:i+BATCH]
        batch_emb = get_embedding(batch_texts)
        embeddings.extend(batch_emb)

    # convert to numpy array
    emb_dim = len(embeddings[0])
    xb = np.array(embeddings).astype("float32")

    # build FAISS index (IndexFlatL2 is simplest; swap for IVF/PQ for large corpora)
    index = faiss.IndexFlatL2(emb_dim)
    index.add(xb)

    # persist index and metadata
    faiss.write_index(index, str(INDEX_FILE))
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump({"texts": all_texts, "metadatas": metadatas}, f, ensure_ascii=False, indent=2)

    print(f"Ingested {len(all_texts)} chunks. Index saved to {INDEX_FILE}. Metadata saved to {META_FILE}.")

if __name__ == "__main__":
    main()

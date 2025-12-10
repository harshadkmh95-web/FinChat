# finchat-rag/app_faiss.py
import os
import json
from dotenv import load_dotenv
import openai
import faiss
import numpy as np
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion  # SK's OpenAI connector
from semantic_kernel.connectors.ai.open_ai import OpenAITextEmbedding   # SK embedding helper (not required if using openai directly)

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")  # change as needed

if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in .env")

openai.api_key = OPENAI_API_KEY

INDEX_DIR = "finchat-rag/faiss_index"
INDEX_FILE = os.path.join(INDEX_DIR, "faiss.index")
META_FILE = os.path.join(INDEX_DIR, "metadata.json")

# load index & metadata
index = faiss.read_index(INDEX_FILE)
with open(META_FILE, "r", encoding="utf-8") as f:
    meta = json.load(f)
texts = meta["texts"]
metadatas = meta["metadatas"]

# helper: embed a query (using openai lib)
def embed_query(q):
    resp = openai.Embedding.create(model=EMBED_MODEL, input=[q])
    return np.array(resp["data"][0]["embedding"], dtype="float32")

def retrieve(query, top_k=4):
    v = embed_query(query)
    v = v.reshape(1, -1)
    D, I = index.search(v, top_k)
    hits = []
    for idx in I[0]:
        if idx < 0 or idx >= len(texts):
            continue
        hits.append({"text": texts[idx], "meta": metadatas[idx]})
    return hits

# Build Semantic Kernel and register OpenAI chat completion
def build_kernel():
    kernel = Kernel()
    # Register OpenAI chat completion service with Semantic Kernel
    # The SK connector class may accept model name and api key via env or params.
    # OpenAIChatCompletion uses env vars or takes settings; this example shows simple init.
    chat = OpenAIChatCompletion(model=CHAT_MODEL, api_key=OPENAI_API_KEY)
    kernel.register_chat_service("openai", chat)
    return kernel

def make_prompt(query, retrieved):
    context = "\n\n---\n\n".join([f"Source: {r['meta']['source']} (chunk {r['meta']['chunk_id']})\n\n{r['text']}" for r in retrieved])
    prompt = f"""You are FinChat, an AI assistant specialized in finance.
Use the context below to answer the user's question. If the context does not contain the answer, say you don't know and suggest how to find the answer.

Context:
{context}

User question:
{query}

Answer concisely. At the end, list the sources you used (filename and chunk id).
"""
    return prompt

def ask(kernel, query, top_k=4):
    retrieved = retrieve(query, top_k=top_k)
    if not retrieved:
        return "No documents found in the index."

    prompt = make_prompt(query, retrieved)

    # Use kernel to run a chat completion using the registered service
    # The exact kernel API for running chat calls may vary by SK version, but typically:
    # response = kernel.run_chat("openai", prompt)  # pseudo
    # To be safe, use the connector class directly via Semantic Kernel's ChatCompletion wrapper:
    chat_service = kernel.get_chat_service("openai")  # depending on SK API name
    # Many SK versions provide a simple method to send messages; below is a conservative approach:
    # (If your SK version differs, call the SDK's chat API accordingly — see docs.)
    result = chat_service.complete(prompt)  # may be complete/chat; adjust per SDK
    # If result is object, extract text:
    if isinstance(result, str):
        return result
    else:
        # try common attribute names
        return getattr(result, "text", str(result))

def main():
    kernel = build_kernel()
    print("FinChat (FAISS+Semantic Kernel) ready. Type 'exit' to quit.")
    while True:
        q = input("You: ").strip()
        if q.lower() in ("exit", "quit"):
            break
        try:
            ans = ask(kernel, q, top_k=4)
        except Exception as e:
            ans = f"Error: {e}"
        print("\nFinChat:", ans)
        print("-" * 60)

if __name__ == "__main__":
    main()

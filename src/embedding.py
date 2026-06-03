# ╔════════════════════════════════════════════╗
# ║            ALIKOAIK
# ║  FILE      │  embadding.py
# ║  DATE      │  01/04/2026
# ║  GITHUB    │  github.com/alikoaik
# ╚════════════════════════════════════════════╝

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def embedding(chunks) -> FAISS:
    if not chunks:
        raise ValueError("No chunks provided — the document may be empty or unsupported.")

    print("\n⏳ Loading embedding model (first time may take 1-2 minutes)...")
    try:
        embeddingData = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    except Exception as e:
        raise RuntimeError(f"Failed to load embedding model: {e}") from e
    print("✓ Model loaded!")

    print(f"Creating vector store from {len(chunks)} chunks...")
    try:
        VectorStore = FAISS.from_documents(chunks, embedding=embeddingData)
    except Exception as e:
        raise RuntimeError(f"Failed to create vector store: {e}") from e
    print("✓ Vector store created!")

    return VectorStore

# HOW THIS WORKS:
#
# 1. load_dotenv() loads OPENAI_API_KEY from .env file into
#    environment variables
#
# 2. OpenAIEmbeddings() automatically reads OPENAI_API_KEY
#    from environment and creates an embedding model
#
# 3. FAISS.from_documents() does:
#    - Takes each chunk of text
#    - Converts it to a vector (numerical representation)
#      using OpenAI embeddings
#    - Stores vectors in FAISS (local vector database)
#
# 4. Returns VectorStore that can be queried for similarity
#    search (finding relevant chunks based on a question)
#
# NOTE: Code outside functions (module level) runs ONCE when
# the file is imported, NOT when calling the function.
# That's why load_dotenv() is at module level - it loads
# .env once at startup instead of every function call.

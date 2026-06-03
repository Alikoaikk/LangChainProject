# ╔════════════════════════════════════════════╗
# ║            ALIKOAIK
# ║  FILE      │  loader.py
# ║  DATE      │  03/04/2026
# ║  GITHUB    │  github.com/alikoaik
# ╚════════════════════════════════════════════╝

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

def loadFile(name) -> list[Document]:
    if not name.endswith('.pdf'):
        name += '.pdf'

    path = f"data/{name}"

    if not os.path.exists(path):
        available = [f for f in os.listdir("data") if f.endswith(".pdf")]
        hint = f" Available files: {', '.join(available)}" if available else " The data folder is empty."
        raise FileNotFoundError(f"'{name}' not found in the data folder.{hint}")

    try:
        loader = PyPDFLoader(path)
        document = loader.load()
    except Exception as e:
        raise RuntimeError(f"Failed to read '{name}': {e}") from e

    if not document:
        raise ValueError(f"'{name}' loaded but contains no readable text.")

    return document

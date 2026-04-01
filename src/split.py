# ╔════════════════════════════════════════════╗
# ║            ALIKOAIK
# ║  FILE      │  split.py
# ║  DATE      │  01/04/2026
# ║  GITHUB    │  github.com/alikoaik
# ╚════════════════════════════════════════════╝

from langchain_text_splitters import RecursiveCharacterTextSplitter

def spliting(documents) :

  text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 50
  )
  chunks = text_splitter.split_documents(documents)
  # print(f"the chunk leng = {len(chunks)}")
  # print(chunks[0].page_content)
  # print(f"the chunk leng = {len(chunks)}")
  # print(f"second chunk {chunks[1].page_content}")
  return chunks

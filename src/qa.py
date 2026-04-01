# ╔════════════════════════════════════════════╗
# ║            ALIKOAIK
# ║  FILE      │  qa.py
# ║  DATE      │  02/04/2026
# ║  GITHUB    │  github.com/alikoaik
# ╚════════════════════════════════════════════╝

from langchain_ollama import ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def ask_question(vectorstore: FAISS, question: str) -> str :

  retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
  llm = ChatOllama(model="llama3.2", temperature=0)

  # Create prompt template
  template = """Answer the question based only on the following context:

{context}

Question: {question}

Answer:"""

  prompt = ChatPromptTemplate.from_template(template)

  # Create chain
  def format_docs(docs):
      return "\n\n".join(doc.page_content for doc in docs)

  chain = (
      {"context": retriever | format_docs, "question": RunnablePassthrough()}
      | prompt
      | llm
      | StrOutputParser()
  )

  result = chain.invoke(question)
  return result

# HOW THIS WORKS:
#
# 1. search_kwargs={"k": 4}:
#    - k = number of chunks to retrieve
#    - k=4 means "return top 4 most similar chunks"
# 2. temperature=0:
#    - Controls LLM creativity/randomness (range: 0-2)
#    - temperature=0: Most accurate, deterministic, same answer every time
#    - temperature=0.7-1.0: Balanced, some variation in responses
#    - temperature=2.0: Very creative, unpredictable, can hallucinate
#    - We use 0 for Q&A because we want factual, consistent answers
#
# 3. qa_chain({"query": question}):
#    - Returns dict: {"result": "answer text", "source_documents": [chunks]}


# ╔════════════════════════════════════════════╗
# ║            ALIKOAIK
# ║  FILE      │  qa_openai.py
# ║  DATE      │  03/04/2026
# ║  GITHUB    │  github.com/alikoaik
# ╚════════════════════════════════════════════╝

import os
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def ask_question(vectorstore: FAISS, question: str) -> str :
  """Ask question using OpenAI API"""
  retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

  # Get API key from environment
  api_key = os.getenv("OPENAI_API_KEY")
  base_url = os.getenv("OPENAI_BASE_URL")

  if not api_key:
      raise ValueError("OPENAI_API_KEY not found in .env file")

  llm = ChatOpenAI(
      model="openai/gpt-4o-mini",
      temperature=0,
      api_key=api_key,
      base_url=base_url
  )

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

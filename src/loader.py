from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

def loadFile (name) -> list[Document] :

  if not name.endswith('.pdf'):
    name += '.pdf'

  loader = PyPDFLoader(f"data/{name}")
  document = loader.load()
  # print(document[0].page_content)
  return document

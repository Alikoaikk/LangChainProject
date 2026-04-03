# LangChain RAG Project - Interview Preparation Guide

## Project Overview
Built a **Retrieval-Augmented Generation (RAG)** system using LangChain that allows users to ask questions about PDF documents using either local LLMs (Ollama) or OpenAI's API.

---

## Key Concepts Learned

### 1. **Document Loading**
- Used `PyPDFLoader` from LangChain to load PDF files
- Returns `Document` objects containing page content and metadata
- **File**: `src/loader.py`

```python
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("data/file.pdf")
documents = loader.load()
```

**Why this matters**: Document loaders are the entry point for any RAG system - they extract text from various formats (PDF, CSV, HTML, etc.)

---

### 2. **Text Splitting (Chunking)**
- Used `RecursiveCharacterTextSplitter` to break documents into smaller chunks
- **Parameters**:
  - `chunk_size=500`: Each chunk has ~500 characters
  - `chunk_overlap=50`: 50 characters overlap between chunks to preserve context
- **File**: `src/split.py`

**Why this matters**:
- LLMs have token limits
- Smaller chunks = more precise retrieval
- Overlap prevents losing context at chunk boundaries

---

### 3. **Embeddings & Vector Storage**
- Used `HuggingFaceEmbeddings` with model `all-MiniLM-L6-v2` (free, local)
- Stored vectors in **FAISS** (Facebook AI Similarity Search) - a local vector database
- **File**: `src/embedding.py`

**How it works**:
1. Text chunks → Embedding model → Numerical vectors (arrays of numbers)
2. Vectors stored in FAISS for fast similarity search
3. When user asks question → Question is embedded → FAISS finds most similar chunks

**Why FAISS**: Fast, runs locally, no external dependencies

---

### 4. **Retrieval-Augmented Generation (RAG)**
- Combined retrieval (finding relevant docs) + generation (LLM answering)
- **Retriever**: Searches vectorstore for top-k most relevant chunks (k=4)
- **Files**: `src/qa.py` (Ollama), `src/qa_openai.py` (OpenAI)

**RAG Pipeline**:
```
User Question → Embed Question → FAISS Retrieval (top 4 chunks)
             → Format as Context → Prompt Template → LLM → Answer
```

---

### 5. **LangChain Expression Language (LCEL)**
- Used **LCEL chains** to create modular, reusable pipelines
- Chain components with `|` (pipe operator)

```python
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

**Benefits**:
- Declarative syntax (what to do, not how)
- Easy to debug and modify
- Supports streaming and async operations

---

### 6. **Prompt Engineering**
- Created a **prompt template** to guide LLM responses
- Instructed LLM to answer "based only on the following context"
- Prevents hallucination (making up information)

```python
template = """Answer the question based only on the following context:

{context}

Question: {question}

Answer:"""
```

---

### 7. **Multi-Provider LLM Support**
- **Ollama** (`ChatOllama`): Local LLM (llama3.2) - free, runs offline
- **OpenAI** (`ChatOpenAI`): Cloud API (gpt-4o-mini) - requires API key
- Used environment variables (`.env`) to store API keys securely

**Temperature Parameter**:
- `temperature=0`: Deterministic, factual answers (best for Q&A)
- `temperature=0.7-1.0`: Creative, varied responses
- `temperature=2.0`: Very creative, may hallucinate

---

### 8. **Retrieval Parameters**
- `search_kwargs={"k": 4}`: Returns top 4 most similar chunks
- Higher k = more context but may include irrelevant info
- Lower k = faster but may miss important details

---

## Technologies Used
| Technology | Purpose |
|------------|---------|
| LangChain | RAG framework, chains, document loaders |
| FAISS | Vector database for similarity search |
| HuggingFace Embeddings | Convert text to vectors (all-MiniLM-L6-v2) |
| Ollama | Local LLM inference (llama3.2) |
| OpenAI API | Cloud LLM (gpt-4o-mini) |
| PyPDFLoader | Extract text from PDF files |
| Python dotenv | Manage environment variables |

---

## Project Architecture
```
1. Load PDF → PyPDFLoader
2. Split into chunks → RecursiveCharacterTextSplitter (500 chars, 50 overlap)
3. Create embeddings → HuggingFaceEmbeddings (all-MiniLM-L6-v2)
4. Store in FAISS → Vector database
5. User asks question → Embed question → Retrieve top 4 chunks
6. Format context → Send to LLM (Ollama/OpenAI) → Get answer
```

---

## Interview Questions & Answers

### **Q1: What is RAG and why did you use it?**
**A**: RAG (Retrieval-Augmented Generation) combines information retrieval with LLM generation. Instead of relying solely on the LLM's training data, RAG retrieves relevant documents first, then uses them as context for the LLM to answer. This reduces hallucinations and allows answering questions about specific documents (like PDFs) that the LLM wasn't trained on.

---

### **Q2: Why did you use FAISS instead of other vector databases?**
**A**: FAISS is lightweight, runs locally without external services, and is extremely fast for similarity search. For this learning project, I didn't need features of production databases like Pinecone or Weaviate (persistence, cloud hosting, multi-user access). FAISS was perfect for understanding core vector search concepts.

---

### **Q3: Explain the difference between chunk_size and chunk_overlap**
**A**:
- `chunk_size=500` means each text chunk has ~500 characters
- `chunk_overlap=50` means consecutive chunks share 50 characters
- Overlap prevents losing context at boundaries. Example: If a sentence is split across two chunks, overlap ensures both chunks contain the full sentence.

---

### **Q4: What happens if you set temperature=2 in your Q&A system?**
**A**: The LLM would generate very creative but unpredictable answers, potentially hallucinating information not present in the documents. For factual Q&A, `temperature=0` is best because it produces deterministic, accurate answers based strictly on the provided context.

---

### **Q5: How does the retriever find relevant chunks?**
**A**:
1. User's question is converted to a vector (embedding)
2. FAISS computes similarity (cosine similarity) between question vector and all chunk vectors
3. Returns top k=4 chunks with highest similarity scores
4. These chunks are formatted as context for the LLM

---

### **Q6: Why did you support both Ollama and OpenAI?**
**A**:
- **Ollama**: Free, runs offline, good for learning/testing, no API costs
- **OpenAI**: More powerful models, faster responses, but requires API key and costs money
- Supporting both shows flexibility and understanding of different deployment scenarios (local vs cloud).

---

### **Q7: What is LCEL and why is it useful?**
**A**: LangChain Expression Language (LCEL) is a declarative way to build chains using the pipe operator (`|`). It's useful because:
- Makes code more readable and maintainable
- Easy to add/remove steps in the pipeline
- Supports streaming and async operations out of the box
- Built-in error handling and retry logic

---

### **Q8: How would you improve this project?**
**A**:
1. **Add persistence**: Save FAISS index to disk to avoid re-embedding on every run
2. **Add conversation memory**: Store chat history for follow-up questions
3. **Multi-document support**: Load multiple PDFs at once
4. **Add web UI**: Use Streamlit or Gradio for better UX
5. **Add source citations**: Show which PDF page/chunk the answer came from
6. **Better error handling**: Handle malformed PDFs, network errors, etc.
7. **Streaming responses**: Show LLM output word-by-word as it generates

---

### **Q9: What's the difference between embeddings and LLMs?**
**A**:
- **Embeddings**: Convert text to numerical vectors (arrays of numbers) that represent meaning. Used for search/similarity.
- **LLMs**: Generate human-like text based on prompts. Used for reasoning/answering.
- In this project: HuggingFace embeddings find relevant chunks, then LLM (Ollama/OpenAI) generates the answer.

---

### **Q10: How does RunnablePassthrough work in your chain?**
**A**: `RunnablePassthrough()` passes the input (question) directly through without modification. In the chain:
```python
{"context": retriever | format_docs, "question": RunnablePassthrough()}
```
- `context` is computed from retriever
- `question` is passed as-is
- Both are combined and sent to the prompt template

---

### **Q11: What challenges did you face?**
**A**:
1. **First-time embedding model load**: Takes 1-2 minutes to download (added user feedback)
2. **Environment setup**: Managing Ollama server + Python dependencies
3. **API key management**: Learning to use `.env` files securely
4. **Chunk size tuning**: Balancing context size vs retrieval precision

---

### **Q12: Can you explain your project in 30 seconds?**
**A**: "I built a RAG system that lets users ask questions about PDF documents. It loads PDFs, splits them into chunks, converts chunks to vectors using HuggingFace embeddings, stores them in FAISS, and when a user asks a question, it retrieves the most relevant chunks and sends them to an LLM (Ollama or OpenAI) to generate an answer. This prevents hallucinations because the LLM only answers based on the actual document content."

---

## Key Files Reference
- `app.py` - Main entry point, user interaction loop
- `src/loader.py` - PDF loading with PyPDFLoader
- `src/split.py` - Text chunking with RecursiveCharacterTextSplitter
- `src/embedding.py` - HuggingFace embeddings + FAISS vector storage
- `src/qa.py` - Ollama-based Q&A chain
- `src/qa_openai.py` - OpenAI-based Q&A chain
- `.env` - Stores OPENAI_API_KEY and OPENAI_BASE_URL

---

## Quick Recap Before Interview
1. **What**: RAG system for PDF Q&A
2. **Why**: Learn LangChain, vector databases, and RAG architecture
3. **How**: Load → Chunk → Embed → Store → Retrieve → Generate
4. **Tech**: LangChain, FAISS, HuggingFace, Ollama/OpenAI
5. **Key Concepts**: Embeddings, vector search, prompt engineering, LCEL chains, temperature tuning

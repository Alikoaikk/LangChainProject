# 📄 LangChain RAG PDF Q&A System

A production-ready **Retrieval-Augmented Generation (RAG)** system that intelligently answers questions about PDF documents. Uses LangChain's modern LCEL chains, vector embeddings, and supports both local (Ollama) and cloud-based (OpenAI) LLMs.

## 🎯 What This Project Demonstrates

This project showcases a complete implementation of the **RAG pattern** - combining intelligent document retrieval with generative AI to answer questions grounded in actual document content, eliminating hallucinations.

### Architecture Overview

```
PDF Input
   ↓
[PyPDFLoader] → Extract text & metadata
   ↓
[RecursiveCharacterTextSplitter] → Chunk (500 chars, 50 overlap)
   ↓
[HuggingFace Embeddings] → Convert chunks to vectors (all-MiniLM-L6-v2)
   ↓
[FAISS Vector DB] → Store embeddings for fast similarity search
   ↓
User Question → [Retriever] → Top 4 relevant chunks
   ↓
[LCEL Chain] → Format context + question
   ↓
[LLM] → Ollama (local llama3.2) OR OpenAI (gpt-4o-mini)
   ↓
Grounded Answer (from document only)
```

## ✨ Key Features

- **🔗 LCEL Chains**: Modern LangChain Expression Language for composable, production-ready pipelines
- **🧠 Intelligent Retrieval**: FAISS vector database with configurable similarity search (k=4 top chunks)
- **📊 Smart Chunking**: Recursive text splitting with 50-character overlap to preserve context boundaries
- **🌐 Multi-Provider LLMs**: 
  - **Ollama** - Local inference (llama3.2), free, offline-capable
  - **OpenAI** - Cloud API (gpt-4o-mini), faster, more capable
- **🔍 Embeddings**: HuggingFace's `all-MiniLM-L6-v2` (free, local, no API keys)
- **⚙️ Production Error Handling**: Connection checks, API validation, graceful fallbacks
- **❄️ Temperature Control**: Deterministic answers (temperature=0) to prevent hallucination
- **📝 Grounded Answers**: Prompts LLM to use only document context

## 🛠 Tech Stack

| Component | Technology | Why? |
|-----------|-----------|------|
| **Framework** | LangChain + LCEL | Modern, composable, production-ready |
| **Vector Storage** | FAISS | Fast similarity search, local, no infrastructure |
| **Embeddings** | HuggingFace (all-MiniLM-L6-v2) | Free, high-quality, runs locally |
| **Local LLM** | Ollama (llama3.2) | Free inference, offline, learning-friendly |
| **Cloud LLM** | OpenAI API (gpt-4o-mini) | Powerful, fast, cloud-based option |
| **PDF Loading** | PyPDFLoader | Built into LangChain, handles metadata |
| **Text Splitting** | RecursiveCharacterTextSplitter | Preserves semantic boundaries |

## 📋 Prerequisites

- Python 3.8+
- Ollama (for local LLM option) OR OpenAI API key
- ~2GB disk space (for first embedding model download)

## 🚀 Quick Start

### 1. Clone & Setup

```bash
git clone <repo-url>
cd LangChain
make setup  # Creates venv + installs all dependencies
```

### 2. Configure LLM Provider

**Option A: Local (Ollama)**
```bash
# Install Ollama: https://ollama.ai
ollama serve                    # Start server (keep running)
ollama pull llama3.2           # Download model (~2GB)
```

**Option B: OpenAI/OpenRouter**
```bash
echo "OPENAI_API_KEY=your-key-here" > .env
echo "OPENAI_BASE_URL=https://openrouter.ai/api/v1" >> .env
```

### 3. Prepare Documents

```bash
mkdir -p data
cp your-document.pdf data/
```

### 4. Run

```bash
make run
```

## 🔨 Makefile Commands

All setup and execution is handled by make:

| Command | What it does |
|---------|------------|
| `make help` | Show all available commands |
| `make setup` | Create venv + install dependencies (one-time) |
| `make install` | Install dependencies into existing venv |
| `make run` | Run the application |
| `make freeze` | Update requirements.txt from current environment |
| `make lint` | Check syntax on all Python files |
| `make clean` | Remove venv and cache files |

## 📖 Usage

```
Choose your LLM provider:
1. Ollama (Local)
2. OpenAI (API Key)
Enter your choice (1 or 2): 1

Enter the name of the file in the data folder: your-document.pdf

✓ Document loaded and ready for questions!

Ask a question (or 'quit' to exit): What are the main points?

⏳ Loading embedding model (first time may take 1-2 minutes)...
✓ Model loaded!
Creating vector store from 45 chunks...
✓ Vector store created!

Question: What are the main points?
Answer: Based on the document... [AI-generated answer based only on document content]

Ask a question (or 'quit' to exit): quit
Goodbye!
```

## 🏗 Project Structure

```
LangChain/
├── app.py                    # Main orchestrator & user interaction
├── src/
│   ├── loader.py            # PDF loading with error handling
│   ├── split.py             # Recursive text chunking (500 chars, 50 overlap)
│   ├── embedding.py         # HuggingFace embeddings + FAISS vector store
│   ├── qa.py                # LCEL chain for Ollama-based Q&A
│   └── qa_openai.py         # LCEL chain for OpenAI-based Q&A
├── data/                    # Place PDFs here
├── requirements.txt         # All dependencies
├── .env                     # API keys (not in git)
├── .env.example             # Template
├── INTERVIEW_PREP.md        # Learning guide & concepts
└── README.md               # This file
```

## 🔑 Core Concepts

### Retrieval-Augmented Generation (RAG)
Instead of relying solely on an LLM's training data, RAG:
1. **Retrieves** relevant documents/chunks first
2. **Augments** the prompt with this retrieved context
3. **Generates** answers using only this context

✅ Prevents hallucination  
✅ Answers about documents LLM never saw  
✅ Grounded in actual source material  

### Vector Embeddings & Similarity Search
- Text chunks → 384-dimensional vectors (HuggingFace)
- User question → Same embedding space
- FAISS finds top-4 chunks with highest cosine similarity
- These become context for LLM

### LCEL Chains (LangChain Expression Language)
```python
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```
- Declarative, composable pipeline
- Easy to debug, test, and modify
- Supports streaming and async out-of-the-box

### Temperature Control
- `temperature=0` → Deterministic, factual (our choice for Q&A)
- `temperature=0.7` → Balanced creativity
- `temperature=2.0` → Very creative, may hallucinate

## 🎓 Learning Value

This project demonstrates:
- ✅ Modern RAG pattern implementation
- ✅ LangChain LCEL for production pipelines
- ✅ Vector databases and semantic search
- ✅ Multi-provider LLM abstraction
- ✅ Production-grade error handling
- ✅ Environment management and security
- ✅ Prompt engineering for grounded responses

Perfect for portfolios, interviews, or understanding how modern AI systems work.

## 🔧 Customization

**Adjust retrieval behavior** in `src/qa.py` / `src/qa_openai.py`:
```python
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})  # Top 4 chunks
```
- Higher `k` = more context, slower, may include noise
- Lower `k` = faster, less context

**Modify chunking strategy** in `src/split.py`:
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,        # Larger = more context per chunk
    chunk_overlap=50       # Larger = more redundancy, slower
)
```

**Change embedding model** in `src/embedding.py`:
```python
HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# Other options: "all-mpnet-base-v2" (larger, slower, more accurate)
```

## 🚨 Error Handling

The project includes robust error handling for:
- Missing PDF files (shows available files)
- Ollama connection failures
- OpenAI API errors (invalid keys, rate limits)
- Empty/unreadable PDFs
- First-time embedding model download

## 📚 Interview Prep

See `INTERVIEW_PREP.md` for:
- 12 common interview questions about RAG
- Concept explanations
- Architecture deep-dives
- 30-second elevator pitch

## 🤝 Contributing

Built by **alikoaik** | [GitHub](https://github.com/alikoaik)

---

**What makes RAG special**: Traditional LLMs answer from their training data. RAG systems answer from your data. That's the difference between a chatbot and a document understanding system.

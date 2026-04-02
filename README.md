# LangChain RAG PDF Q&A System

A Retrieval-Augmented Generation (RAG) application that allows you to ask questions about PDF documents using LangChain with Ollama or OpenAI.

## What This Project Does

This is a **smart PDF question-answering system** that lets you ask questions about any PDF document in natural language.

### How It Works (Step by Step)

1. **📄 Load PDF**: You provide a PDF filename - the app reads it using PyPDFLoader
2. **✂️ Split Text**: The PDF is broken into small chunks (500 characters each with 50 character overlap for context)
3. **🧠 Create Embeddings**: Each chunk is converted into numerical vectors (embeddings) using HuggingFace's AI model
4. **💾 Store in Vector Database**: All embeddings are stored in FAISS (a fast similarity search database)
5. **❓ Ask Questions**: You type a question - the system finds the top 4 most relevant chunks
6. **✨ Generate Answer**: Your chosen LLM (Ollama or OpenAI) reads those chunks and generates a precise answer

**Example**: If you upload a medical research paper and ask "What are the side effects?", the system finds relevant sections and gives you an accurate answer based only on that document.

## Features

- 📄 Process PDF documents of any size
- ✂️ Intelligent text chunking with overlap for better context
- 🧠 Local embeddings using HuggingFace (no API keys needed)
- 🔄 Choose between Ollama (local) or OpenAI (API) for question answering
- 💾 FAISS vector store for lightning-fast similarity search
- 🔁 Interactive Q&A loop (ask multiple questions about the same document)
- 🧹 Clean terminal interface showing only current question and answer

## Prerequisites

- Python 3.8 or higher
- Ollama installed on your system (for local option)
- OpenAI API key (for OpenAI option)

## Ollama Installation & Setup

Ollama is an open-source tool that lets you run large language models locally on your machine.

### macOS Installation

```bash
# Option 1: Using the install script
curl -fsSL https://ollama.com/install.sh | sh

# Option 2: Using Homebrew
brew install ollama
```

### Linux Installation

```bash
# Using the official install script
curl -fsSL https://ollama.com/install.sh | sh
```

### Start Ollama Server

**IMPORTANT**: Ollama must be running before you use this application with the local option.

```bash
# Start Ollama server (keep this terminal open)
ollama serve
```

The server will run on `http://localhost:11434`

### Download the Llama 3.2 Model

```bash
# Pull and run the Llama 3.2 model
ollama run llama3.2
```

This downloads the model (~2GB) and starts an interactive chat. Type `/bye` to exit.

### Verify Ollama is Working

```bash
# List installed models
ollama list

# Test the model
ollama run llama3.2 "What is Python?"
```

### Common Ollama Commands

```bash
ollama serve              # Start Ollama server
ollama list               # Show installed models
ollama pull llama3.2      # Download a model
ollama run llama3.2       # Run a model interactively
ollama rm llama3.2        # Remove a model
```

## OpenAI API Setup

If you want to use OpenAI instead of Ollama:

1. Get an API key from [OpenRouter](https://openrouter.ai/) or [OpenAI](https://platform.openai.com/)
2. Create a `.env` file in the project root
3. Add your API key:

```bash
OPENAI_API_KEY=your-api-key-here
OPENAI_BASE_URL=https://openrouter.ai/api/v1
```

## Python Project Setup

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd LangChain
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # macOS/Linux
```

### 3. Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# Install additional packages
pip install langchain-community langchain-huggingface langchain-ollama langchain-openai pypdf faiss-cpu python-dotenv
```

### 4. Set Up Data Folder

```bash
# Create data folder for your PDFs
mkdir -p data

# Copy your PDF files to the data folder
cp /path/to/your/document.pdf data/
```

### 5. Create .env File

```bash
# Create .env file for OpenAI API key (optional, only if using OpenAI)
touch .env
```

Add your credentials:
```
OPENAI_API_KEY=your-api-key-here
OPENAI_BASE_URL=https://openrouter.ai/api/v1
```

## Usage

### Step 1: Start Ollama Server (If Using Ollama)

Open a terminal and run:

```bash
ollama serve
```

Keep this terminal open while using the application.

### Step 2: Run the Application

In a new terminal:

```bash
# Activate virtual environment
source venv/bin/activate

# Run the app
python app.py
```

### Step 3: Choose Your LLM Provider

```
Choose your LLM provider:
1. Ollama (Local)
2. OpenAI (API Key)
Enter your choice (1 or 2): 1
```

### Step 4: Load Document and Ask Questions

```
Enter the name of the file exist in data folder
> mydocument.pdf

✓ Document loaded and ready for questions!

Ask a question (or 'quit' to exit): What is this document about?

Question: What is this document about?

Answer: This document discusses machine learning algorithms...

Ask a question (or 'quit' to exit): quit
Goodbye!
```

## Project Structure

```
LangChain/
├── app.py                 # Main entry point - orchestrates the entire workflow
├── src/
│   ├── loader.py          # Loads PDF files from data/ folder
│   ├── split.py           # Splits documents into chunks
│   ├── embedding.py       # Creates vector embeddings using HuggingFace
│   ├── qa.py              # Handles question answering using Ollama
│   └── qa_openai.py       # Handles question answering using OpenAI
├── data/                  # Place your PDF files here
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (API keys)
└── README.md             # Documentation
```

## Contact

**alikoaik**
GitHub: [github.com/alikoaik](https://github.com/alikoaik)

---

**Made with ❤️ using LangChain, Ollama, and OpenAI**

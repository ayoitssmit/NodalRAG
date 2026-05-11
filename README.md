# NodalRAG

NodalRAG is an interactive Retrieval-Augmented Generation (RAG) application specifically architected for academic research and personal knowledge management. It enables users to upload document datasets, query them using natural language, and explore the underlying mathematical relationships between data points through a dynamic, human-readable 2D knowledge map. 

The application is optimized for complete local execution on consumer-grade hardware, prioritizing privacy, performance, and accuracy by eliminating semantic confusion between distinct document topics.

## System Architecture

NodalRAG is built on a modern, decoupled architecture:

- **Frontend (Next.js 14 & React):** Provides a high-performance, responsive user interface. It manages document uploads, the chat interface, and the interactive visualization of the high-dimensional latent space.
- **Backend (FastAPI):** A high-throughput Python API that orchestrates the data pipeline, handling routing, CORS, and endpoint management.
- **Inference Engine (Ollama):** Executes large language models locally. We utilize `llama3:8b` for generation and summarization, and `nomic-embed-text` for generating high-quality vector embeddings.
- **RAG Framework (LlamaIndex):** Manages the ingestion pipeline, smart chunking algorithms, contextual prepending, and retrieval logic.
- **Vector Database (ChromaDB):** An embedded, persistence-enabled vector database used to store and query the generated document embeddings.

## Prerequisites

Before installing NodalRAG, ensure your system meets the following requirements:

1. **Python 3.10+**: Required for the FastAPI backend and LlamaIndex.
2. **Node.js 18+ & npm**: Required for the Next.js frontend.
3. **Ollama**: Must be installed and running as a background service.
   - Install from [ollama.com](https://ollama.com/).
   - Pull the required models by executing the following commands in your terminal:
     ```bash
     ollama pull llama3:8b
     ollama pull nomic-embed-text
     ```

## Installation Guide

Follow these steps to set up the environment on your local machine.

### 1. Clone the Repository

```bash
git clone https://github.com/ayoitssmit/NodalRAG.git
cd NodalRAG
```

### 2. Backend Setup

The backend utilizes a Python virtual environment to isolate dependencies.

```bash
cd backend
python -m venv venv

# For Windows:
.\venv\Scripts\activate
# For macOS/Linux:
# source venv/bin/activate

pip install -r requirements.txt 
# Note: If requirements.txt is not yet generated, install the core dependencies:
# pip install fastapi uvicorn chromadb llama-index llama-index-llms-ollama llama-index-embeddings-ollama python-multipart llama-index-vector-stores-chroma
```

### 3. Frontend Setup

The frontend relies on standard Node package management. Open a new terminal window to keep the backend environment separate.

```bash
cd frontend
npm install
```

## Execution Instructions

To run NodalRAG, you must start both the backend server and the frontend development server concurrently.

### Starting the Backend

In your backend terminal with the virtual environment activated:

```bash
cd backend
python main.py
```
The FastAPI server will initialize and listen on `http://localhost:8000`. You can view the interactive API documentation at `http://localhost:8000/docs`.

### Starting the Frontend

In your frontend terminal:

```bash
cd frontend
npm run dev
```
The Next.js application will compile and be available at `http://localhost:3000`. 

## System Verification and Smoke Tests

To guarantee that your local AI stack is correctly configured and that LlamaIndex can communicate with both Ollama and ChromaDB, the repository includes two smoke tests.

Run these scripts from the `backend` directory with your virtual environment activated:

1. **Ollama Connectivity Test**
   Verifies that the Python client can successfully trigger generation and embedding tasks.
   ```bash
   python test_ollama.py
   ```

2. **Database and Indexing Integration Test**
   Validates the ChromaDB persistence layer and ensures LlamaIndex can successfully index and retrieve a dummy document.
   ```bash
   python test_chroma_llamaindex.py
   ```

## Directory Structure

```text
NodalRAG/
├── backend/
│   ├── chroma_db/                  # Persistent local vector storage
│   ├── main.py                     # FastAPI application entry point
│   ├── test_chroma_llamaindex.py   # Integration test script
│   └── test_ollama.py              # Inference test script
├── frontend/
│   ├── src/
│   │   └── app/
│   │       ├── chat/               # Chat interface routes
│   │       ├── upload/             # Document ingestion routes
│   │       ├── layout.tsx          # Next.js root layout
│   │       └── page.tsx            # Next.js root page
│   ├── tailwind.config.ts          # Styling configuration
│   └── package.json                # Node dependencies
└── README.md
```

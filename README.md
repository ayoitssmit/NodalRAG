# NodalRAG

Interactive RAG application for academic research and personal knowledge management.

## Prerequisites

- [Ollama](https://ollama.com/) installed and running.
- Models pulled:
  - `ollama pull llama3:8b`
  - `ollama pull nomic-embed-text`

## Getting Started

### 1. Backend (FastAPI)

The backend handles document ingestion, vector storage (ChromaDB), and retrieval.

```powershell
cd backend
.\venv\Scripts\activate
python main.py
```
The API will be available at `http://localhost:8000`.

### 2. Frontend (Next.js)

The frontend provides the interactive 2D knowledge map and chat interface.

```powershell
cd frontend
npm run dev
```
The UI will be available at `http://localhost:3000`.

## Smoke Tests

To verify that your local AI stack (Ollama + LlamaIndex + ChromaDB) is working correctly:

1. **Ollama Test**: Checks if Llama 3 and Nomic embeddings are responsive.
   ```powershell
   cd backend
   .\venv\Scripts\python test_ollama.py
   ```

2. **Integration Test**: Verifies ChromaDB persistence and LlamaIndex retrieval.
   ```powershell
   cd backend
   .\venv\Scripts\python test_chroma_llamaindex.py
   ```

## Project Structure

- `backend/`: FastAPI server and RAG logic.
- `frontend/`: Next.js web application.
- `backend/chroma_db/`: Local vector database storage.

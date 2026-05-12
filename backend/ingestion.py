import hashlib
import sqlite3
import json
import asyncio
import os
import uuid
import numpy as np
from functools import lru_cache
from typing import Dict

import chromadb
from sklearn.manifold import TSNE

from fastapi import APIRouter, UploadFile, BackgroundTasks, Request
from fastapi.responses import StreamingResponse

from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

router = APIRouter()

DB_PATH = "./doc_cache.db"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "nodalrag_docs"

def init_dbs():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS uploads (hash TEXT PRIMARY KEY, doc_id TEXT)")

init_dbs()

# Settings
Settings.llm = Ollama(model="gemma3:latest", request_timeout=120.0)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
chroma_collection = chroma_client.get_or_create_collection(COLLECTION_NAME)

class ProgressStream:
    def __init__(self):
        self.queue = asyncio.Queue()
        self.is_done = False

# Global state for SSE
progress_streams: Dict[str, ProgressStream] = {}

def get_file_hash(file_path: str) -> str:
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()

def get_cached_doc_id(file_hash: str) -> str | None:
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute("SELECT doc_id FROM uploads WHERE hash = ?", (file_hash,))
        row = cursor.fetchone()
        return row[0] if row else None

def set_cached_doc_id(file_hash: str, doc_id: str):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("INSERT INTO uploads (hash, doc_id) VALUES (?, ?)", (file_hash, doc_id))

def run_tsne_for_doc(doc_id: str):
    try:
        results = chroma_collection.get(where={"document_id": doc_id}, include=["embeddings", "metadatas"])
        embeddings = results["embeddings"]
        ids = results["ids"]
        metadatas = results["metadatas"]
        
        if not embeddings or len(embeddings) < 2:
            if doc_id in progress_streams:
                asyncio.run_coroutine_threadsafe(
                    progress_streams[doc_id].queue.put('{"status": "done"}'),
                    asyncio.get_running_loop()
                )
                progress_streams[doc_id].is_done = True
            return
            
        n_samples = len(embeddings)
        perplexity = min(30, n_samples - 1)
        
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        coords = tsne.fit_transform(np.array(embeddings))
        
        for i, _id in enumerate(ids):
            metadatas[i]["tsne_x"] = float(coords[i][0])
            metadatas[i]["tsne_y"] = float(coords[i][1])
            
        chroma_collection.update(ids=ids, metadatas=metadatas)
        
        # We can try to send 'done' to the SSE stream if it's still open
        if doc_id in progress_streams:
            asyncio.run_coroutine_threadsafe(
                progress_streams[doc_id].queue.put('{"status": "done"}'),
                asyncio.get_running_loop()
            )
            progress_streams[doc_id].is_done = True
    except Exception as e:
        print(f"TSNE Error: {e}")
        if doc_id in progress_streams:
            asyncio.run_coroutine_threadsafe(
                progress_streams[doc_id].queue.put(f'{{"error": "{str(e)}" }}'),
                asyncio.get_running_loop()
            )
            progress_streams[doc_id].is_done = True

async def ingest_pipeline(file_path: str, doc_id: str, file_hash: str, background_tasks: BackgroundTasks):
    stream = progress_streams[doc_id]
    
    try:
        await stream.queue.put('{"status": "file received"}')
        
        await stream.queue.put('{"status": "summarising"}')
        reader = SimpleDirectoryReader(input_files=[file_path])
        documents = await asyncio.to_thread(reader.load_data)
        
        full_text = "\n".join([d.text for d in documents])
        prompt = f"Summarize the following document in exactly 2 sentences. Do not include any other text.\n\n{full_text[:10000]}"
        
        llm = Ollama(model="gemma3:latest", request_timeout=120.0)
        summary_response = await asyncio.to_thread(llm.complete, prompt)
        summary = str(summary_response).strip()
        
        await stream.queue.put('{"status": "chunking"}')
        splitter = SentenceSplitter(chunk_size=768, chunk_overlap=76)
        all_nodes = []
        for doc in documents:
            doc.doc_id = doc_id
            nodes = splitter.get_nodes_from_documents([doc])
            for node in nodes:
                node.metadata["document_summary"] = summary
                # Extract page number (LlamaIndex adds it to metadata)
                node.metadata["raw_text"] = node.text
                node.text = f"Summary: {summary}\n\nContent: {node.text}"
                all_nodes.append(node)
                
        await stream.queue.put('{"status": "embedding"}')
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        embed_model = OllamaEmbedding(model_name="nomic-embed-text")
        index = await asyncio.to_thread(
            VectorStoreIndex, nodes=all_nodes, storage_context=storage_context, embed_model=embed_model
        )
        
        set_cached_doc_id(file_hash, doc_id)
        chunk_ids = [node.node_id for node in all_nodes]
        
        # Schedule t-SNE in background task
        background_tasks.add_task(run_tsne_for_doc, doc_id)
        
        await stream.queue.put(json.dumps({"status": "t-SNE", "chunk_ids": chunk_ids}))
        
    except Exception as e:
        await stream.queue.put(json.dumps({"status": "error", "message": str(e)}))
        stream.is_done = True
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@router.post("/ingest")
async def ingest_document(file: UploadFile, background_tasks: BackgroundTasks):
    doc_id = str(uuid.uuid4())
    file_path = f"./temp_{doc_id}_{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())
        
    file_hash = get_file_hash(file_path)
    cached_doc_id = get_cached_doc_id(file_hash)
    
    if cached_doc_id:
        results = chroma_collection.get(where={"document_id": cached_doc_id}, include=[])
        os.remove(file_path)
        return {"doc_id": cached_doc_id, "chunk_ids": results["ids"], "status": "deduplicated"}

    # Start ingestion pipeline
    progress_streams[doc_id] = ProgressStream()
    background_tasks.add_task(ingest_pipeline, file_path, doc_id, file_hash, background_tasks)
    
    # We return immediately, frontend will subscribe to SSE
    return {"doc_id": doc_id, "status": "processing"}

@router.get("/progress/{doc_id}")
async def get_progress(doc_id: str, request: Request):
    async def event_generator():
        stream = progress_streams.get(doc_id)
        if not stream:
            yield "data: {\"status\": \"not found\"}\n\n"
            return
            
        while True:
            if await request.is_disconnected():
                break
                
            try:
                # Wait for next event with a timeout for heartbeat
                event = await asyncio.wait_for(stream.queue.get(), timeout=1.0)
                yield f"data: {event}\n\n"
                
                # Check if it's the done signal
                try:
                    event_data = json.loads(event)
                    if event_data.get("status") in ["done", "error"]:
                        break
                except:
                    pass
            except asyncio.TimeoutError:
                # Heartbeat
                yield ": heartbeat\n\n"
                
    return StreamingResponse(event_generator(), media_type="text/event-stream")

# LRU response cache (cache layer 3)
@lru_cache(maxsize=128)
def cached_query(query_str: str, doc_id: str) -> str:
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    embed_model = OllamaEmbedding(model_name="nomic-embed-text")
    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
    
    from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
    filters = MetadataFilters(filters=[ExactMatchFilter(key="document_id", value=doc_id)])
    
    llm = Ollama(model="gemma3:latest", request_timeout=120.0)
    query_engine = index.as_query_engine(filters=filters, llm=llm, similarity_top_k=5)
    response = query_engine.query(query_str)
    return str(response)



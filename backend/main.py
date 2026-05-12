from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import ingestion
app = FastAPI(title="NodalRAG API", description="Backend for NodalRAG")

# Add CORS for localhost:3000
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    doc_id: str

app.include_router(ingestion.router)

@app.get("/")
def read_root():
    return {"message": "Welcome to NodalRAG API. Visit /docs for the interactive API documentation."}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/query")
def query_documents(req: QueryRequest):
    # Use Cache Layer 3
    response_text = ingestion.cached_query(req.query, req.doc_id)
    return {"status": "success", "response": response_text}

@app.get("/graph")
def get_graph():
    # Stub for Phase 3
    return {"status": "not implemented", "message": "501 Stub"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

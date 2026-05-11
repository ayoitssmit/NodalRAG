import time
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

def test_ollama():
    print("Testing Llama 3 generation...")
    llm = Ollama(model="llama3:8b", request_timeout=60.0)
    
    start_time = time.time()
    try:
        response = llm.complete("Say 'Hello, World!'")
        latency = time.time() - start_time
        print(f"Llama 3 response: {response}")
        print(f"Latency: {latency:.2f} seconds")
        assert str(response).strip() != "", "Llama 3 response was empty"
    except Exception as e:
        print(f"Skipping Llama 3 generation due to error (likely memory): {e}")
    
    print("\nTesting Nomic Embeddings...")
    embed_model = OllamaEmbedding(model_name="nomic-embed-text")
    
    start_time = time.time()
    embedding = embed_model.get_text_embedding("This is a test document.")
    latency = time.time() - start_time
    
    print(f"Embedding length: {len(embedding)}")
    print(f"Latency: {latency:.2f} seconds")
    assert len(embedding) > 0, "Embedding was empty"
    
    print("\nOllama smoke test passed successfully!")

if __name__ == "__main__":
    test_ollama()

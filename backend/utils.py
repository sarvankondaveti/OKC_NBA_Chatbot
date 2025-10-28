import requests, json
from backend.config import OLLAMA_HOST


def ollama_embed(model: str, text: str):
    r = requests.post(f"{OLLAMA_HOST}/api/embeddings", json={"model": model, "prompt": text})
    r.raise_for_status()
    return r.json()["embedding"]


def ollama_generate(model: str, prompt: str):
    # Optimized parameters for speed
    payload = {
        "model": model, 
        "prompt": prompt, 
        "stream": False,
        "options": {
            "num_predict": 60,   # Optimized for consistent 20-25 second responses
            "temperature": 0.1,  
            "top_p": 0.9,      
            "num_ctx": 2048     # Smaller context window
        }
    }
    r = requests.post(f"{OLLAMA_HOST}/api/generate", json=payload)
    r.raise_for_status()
    return r.json()["response"]

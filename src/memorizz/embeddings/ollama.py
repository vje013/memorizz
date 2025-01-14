# ./embeddings/ollama.py

import logging
from langchain_ollama import OllamaEmbeddings    
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_embedding(text: str, model: str = "nomic-embed-text") -> List[float]:
    embeddings = OllamaEmbeddings(model=model)   
    text = text.replace("\n", " ")
    try:
        return embeddings.embed_query(text)
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        raise
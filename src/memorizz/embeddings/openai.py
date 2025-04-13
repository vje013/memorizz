# ./embeddings/openai.py

import logging
import openai
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_embedding(text: str, model: str = "text-embedding-3-small", dimensions: int = 256) -> List[float]:
    text = text.replace("\n", " ")
    try:
        return openai.OpenAI().embeddings.create(input=[text], model=model, dimensions=dimensions).data[0].embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        raise

def get_embedding_dimensions(model: str = "text-embedding-3-small") -> int:
    if model == "text-embedding-3-small":
        return 256
    elif model == "text-embedding-3-large":
        return 1024
    else:
        raise ValueError(f"Unsupported model: {model}")
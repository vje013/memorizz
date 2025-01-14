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
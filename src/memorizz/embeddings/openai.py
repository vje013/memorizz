# ./embeddings/openai.py

import logging
import openai
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

openai_client = openai.OpenAI()

def get_embedding(text: str, model: str = "text-embedding-3-small", dimensions: int = 256) -> List[float]:

    """
    Get the embedding of a text using OpenAI's API.

    Parameters:
        text (str): The text to get the embedding of.
        model (str): The model to use for the embedding.
        dimensions (int): The dimensions of the embedding.

    Returns:
        List[float]: The embedding of the text.
    """

    text = text.replace("\n", " ")
    try:
        return openai_client.embeddings.create(input=[text], model=model, dimensions=dimensions).data[0].embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        raise

def get_embedding_dimensions(model: str = "text-embedding-3-small") -> int:
    """
    Get the dimensions of the embedding for a given model.

    Parameters:
        model (str): The model to get the dimensions of.

    Returns:
        int: The dimensions of the embedding.
    """
    if model == "text-embedding-3-small":
        return 256
    else:
        raise ValueError(f"Unsupported model: {model}")



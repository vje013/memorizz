from .conversational_memory_component import ConversationMemoryComponent
from ..memory_provider import MemoryProvider
from ..memory_provider.memory_type import MemoryType
from .memory_mode import MemoryMode
from ..embeddings.openai import get_embedding
from typing import TYPE_CHECKING, Dict, Any, List, Optional
import time
import numpy as np
import pprint

# Use lazy initialization for OpenAI
def get_openai_llm():
    from ..llms.openai import OpenAI
    return OpenAI()

class MemoryComponent:
    def __init__(self, memory_mode: str, memory_provider: MemoryProvider = None):
        self.memory_mode = memory_mode
        self.memory_provider = memory_provider
        self.query_embedding = None

    def generate_memory_component(self, content: dict):
        """
        Generate the memory component based on the memory mode.
        """

        # Generate the embedding of the memory component
        content["embedding"] = get_embedding(content["content"])

        # Generate the memory component based on the memory mode
        if self.memory_mode == MemoryMode.Conversational or self.memory_mode == MemoryMode.General:
            return self._generate_conversational_memory_component(content)
        elif self.memory_mode == MemoryMode.Task:
            return self._generate_task_memory_component()
        elif self.memory_mode == MemoryMode.Workflow:
            return self._generate_workflow_memory_component()
        else:
            raise ValueError(f"Invalid memory mode: {self.memory_mode}")

    def _generate_conversational_memory_component(self, content: dict) -> ConversationMemoryComponent:
        """
        Generate the conversational memory component.
        
        Parameters:
            content (dict): The content of the memory component.

        Returns:
            ConversationMemoryComponent: The conversational memory component.
        """
        memory_component = ConversationMemoryComponent(
            role=content["role"],
            content=content["content"],
            timestamp=content["timestamp"],
            conversation_id=content["conversation_id"],
            memory_id=content["memory_id"],
            embedding=content["embedding"]
        )

        # Save the memory component to the memory provider
        self._save_memory_component(memory_component)

        return memory_component

    def _generate_task_memory_component(self):
        pass

    def _generate_workflow_memory_component(self):
        pass
    
    def _save_memory_component(self, memory_component: any):
        """
        Save the memory component to the memory provider.
        """

        # Remove the score(vector similarity score calculated by the vector search of the memory provider) from the memory component if it exists
        if "score" in memory_component:
            memory_component.pop("score", None)

        # Convert Pydantic model to dictionary if needed
        if hasattr(memory_component, 'model_dump'):
            memory_component_dict = memory_component.model_dump()
        elif hasattr(memory_component, 'dict'):
            memory_component_dict = memory_component.dict()
        else:
            # If it's already a dictionary, use it as is
            memory_component_dict = memory_component

        if self.memory_mode == MemoryMode.Conversational or self.memory_mode == MemoryMode.General:
            self.memory_provider.store(memory_component_dict, MemoryType.CONVERSATION_MEMORY)
        elif self.memory_mode == MemoryMode.Task:
            self.memory_provider.store(memory_component_dict, MemoryType.TASK_MEMORY)
        elif self.memory_mode == MemoryMode.Workflow:
            self.memory_provider.store(memory_component_dict, MemoryType.WORKFLOW_MEMORY)
        else:
            raise ValueError(f"Invalid memory mode: {self.memory_mode}")

    def retrieve_memory_components_by_memory_id(self, memory_id: str, memory_type: MemoryType):
        """
        Retrieve the memory components by memory id.

        Parameters:
            memory_id (str): The id of the memory to retrieve the memory components for.
            memory_type (MemoryType): The type of the memory to retrieve the memory components for.

        Returns:
            List[MemoryComponent]: The memory components.
        """
        if memory_type == MemoryType.CONVERSATION_MEMORY:
            return self.memory_provider.retrieve_conversation_history_ordered_by_timestamp(memory_id)
        elif memory_type == MemoryType.TASK_MEMORY:
            return self.memory_provider.retrieve_task_history_ordered_by_timestamp(memory_id)
        elif memory_type == MemoryType.WORKFLOW_MEMORY:
            return self.memory_provider.retrieve_workflow_history_ordered_by_timestamp(memory_id)
        else:
            raise ValueError(f"Invalid memory type: {memory_type}")

    def retrieve_memory_components_by_conversation_id(self, conversation_id: str):
        pass

    def retrieve_memory_components_by_query(self, query: str, memory_id: str, memory_type: MemoryType, limit: int = 5):
        """
        Retrieve the memory components by query.

        Parameters:
            query (str): The query to use for retrieval.
            memory_id (str): The id of the memory to retrieve the memory components for.
            memory_type (MemoryType): The type of the memory to retrieve the memory components for.
            limit (int): The limit of the memory components to return.

        Returns:
            List[MemoryComponent]: The memory components.
        """

        # Create the query embedding here so that it is not created for each memory component
        self.query_embedding = get_embedding(query)

        # Get the memory components by query
        memory_components = self.memory_provider.retrieve_memory_components_by_query(query, self.query_embedding, memory_id, memory_type, limit)

        # Get the surronding conversation ids from each of the memory components
        surrounding_conversation_ids = [memory_component["conversation_id"] for memory_component in memory_components]

        # Before returning the memory components, we need to update the memory signals within the memory components
        for memory_component in memory_components:
            self.update_memory_signals_within_memory_component(memory_component, memory_type, surrounding_conversation_ids)

        # Calculate the memory signal for each of the memory components
        for memory_component in memory_components:
            memory_component["memory_signal"] = self.calculate_memory_signal(memory_component, query)

        # Sort the memory components by the memory signal
        memory_components.sort(key=lambda x: x["memory_signal"], reverse=True)

        # Return the memory components
        return memory_components
    

    def update_memory_signals_within_memory_component(self, memory_component: any, memory_type: MemoryType, surrounding_conversation_ids: list[str]):
        """
        Update the memory signal within the memory component.

        Parameters:
            memory_component (dict): The memory component to update the memory signal within.
            memory_type (MemoryType): The type of the memory to update the memory signal within.
            surrounding_conversation_ids (list[str]): The list of surrounding conversation ids.
        """

        # Update the recall_recency field (how recently the memory component was recalled), this is the current timestamp
        memory_component["recall_recency"] = time.time()

        # Update the importance field with a list of calling ID and surronding ID's
        memory_component["associated_conversation_ids"] = surrounding_conversation_ids

        # Save the memory component to the memory provider
        self._save_memory_component(memory_component)

    def calculate_memory_signal(self, memory_component: any, query: str):
        """
        Calculate the memory signal within the memory component.

        Parameters:
            memory_component (any): The memory component to calculate the memory signal within.
            query (str): The query to use for calculation.

        Returns:
            float: The memory signal between 0 and 1.
        """
        # Detect the gap between the current timestamp and the recall_recency field
        recency = time.time() - memory_component["recall_recency"]

        # Get the number of associated memory ids (this is used to calcualte the importance of the memory component)
        number_of_associated_conversation_ids = len(memory_component["associated_conversation_ids"])

        # If the score exists, use it as the relevance score (this is the vector similarity score calculated by the vector search of the memory provider)
        if "score" in memory_component:
            relevance = memory_component["score"]
        else:
            # Calculate the relevance of the memory component which is a vector score between the memory component and the query
            relevance = self.calculate_relevance(query, memory_component)

        # Calulate importance of the memory component
        importance = self.calculate_importance(memory_component["content"], query)

        # Calculate the normalized memory signal
        memory_signal = recency * number_of_associated_conversation_ids * relevance * importance

        # Normalize the memory signal between 0 and 1
        memory_signal = memory_signal / 100

        # Return the memory signal
        return memory_signal

    def calculate_relevance(self, query: str, memory_component: any) -> float:
        """
        Calculate the relevance of the query with the memory component.

        Parameters:
            query (str): The query to use for calculation.
            memory_component (any): The memory component to calculate the relevance within.

        Returns:
            float: The relevance between 0 and 1.
        """
        # Get embedding of the query
        if self.query_embedding is None:
            self.query_embedding = get_embedding(query)

        # Get embedding of the memory component if it is not already embedded
        if memory_component["embedding"] is None:
            memory_component_embedding = get_embedding(memory_component["content"])
        else:
            memory_component_embedding = memory_component["embedding"]

        # Calculate the cosine similarity between the query embedding and the memory component embedding
        relevance = self.cosine_similarity(self.query_embedding, memory_component_embedding)

        # Return the relevance
        return relevance
        

    # We might not need this as the memory compoennt should have a score from retrieval
    def cosine_similarity(self, query_embedding: list[float], memory_component_embedding: list[float]) -> float:
        """
        Calculate the cosine similarity between two embeddings.

        Parameters:
            query_embedding (list[float]): The query embedding.
            memory_component_embedding (list[float]): The memory component embedding.

        Returns:
            float: The cosine similarity between the two embeddings.
        """
        # Calculate the dot product of the two embeddings
        dot_product = np.dot(query_embedding, memory_component_embedding)

        # Calculate the magnitude of the two embeddings
        magnitude_query_embedding = np.linalg.norm(query_embedding)
        magnitude_memory_component_embedding = np.linalg.norm(memory_component_embedding)

        # Calculate the cosine similarity
        cosine_similarity = dot_product / (magnitude_query_embedding * magnitude_memory_component_embedding)

        # Return the cosine similarity
        return cosine_similarity


    def calculate_importance(self, memory_component_content: str, query: str) -> float:
        """
        Calculate the importance of the memory component.
        Using an LLM to calculate the importance of the memory component.

        Parameters:
            memory_component_content (str): The content of the memory component to calculate the importance within.
            query (str): The query to use for calculation.

        Returns:
            float: The importance between 0 and 1.
        """
   

        importance_prompt = f"""
        Calculate the importance of the following memory component:
        {memory_component_content}
        in relation to the following query and rate the likely poignancy of the memory component:
        {query}
        Return the importance of the memory component as a number between 0 and 1.
        """

        # Get the importance of the memory component
        importance = get_openai_llm().generate_text(importance_prompt, instructions="Return the importance of the memory component as a number between 0 and 1. No other text or comments, just the number. For example: 0.5")

        # Return the importance
        return float(importance)


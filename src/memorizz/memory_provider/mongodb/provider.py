from pymongo import MongoClient
from pymongo.operations import SearchIndexModel
from ..base import MemoryProvider
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from src.memorizz.memory_provider.memory_type import MemoryType
from src.memorizz.embeddings.openai import get_embedding
from src.memorizz.embeddings.openai import get_embedding_dimensions


@dataclass
class MongoDBConfig():
    """Configuration for the MongoDB provider."""

    def __init__(self, uri: str, db_name: str = "memorizz"):
        """
        Initialize the MongoDB provider with configuration settings.
        
        Parameters:
        -----------
        uri : str
            The MongoDB URI.
        db_name : str
            The database name.
        """
        self.uri = uri
        self.db_name = db_name


class MongoDBProvider(MemoryProvider):
    """MongoDB implementation of the MemoryProvider interface."""
    
    def __init__(self, config: MongoDBConfig):
        """
        Initialize the MongoDB provider with configuration settings.
        
        Parameters:
        -----------
        config : MongoDBConfig
            Configuration dictionary containing:
            - 'uri': MongoDB URI
            - 'db_name': Database name
        """
        self.config = config
        self.client = MongoClient(config.uri)
        self.db = self.client[config.db_name]
        self.persona_collection = self.db[MemoryType.PERSONAS.value]
        self.toolbox_collection = self.db[MemoryType.TOOLBOX.value]
        self.short_term_memory_collection = self.db[MemoryType.SHORT_TERM_MEMORY.value]
        self.long_term_memory_collection = self.db[MemoryType.LONG_TERM_MEMORY.value]
        self.conversation_memory_collection = self.db[MemoryType.CONVERSATION_MEMORY.value]
        self.workflow_memory_collection = self.db[MemoryType.WORKFLOW_MEMORY.value]

        # Ensure the vector index for the toolbox collection exists
        self._ensure_vector_index(self.toolbox_collection, "vector_index")
        # Ensure the vector index for the persona collection exists
        self._ensure_vector_index(self.persona_collection, "vector_index")
        

    def store(self, data: Dict[str, Any], memory_store_type: MemoryType) -> str:
        """
        Store data in MongoDB.
        
        Parameters:
        -----------
        data : Dict[str, Any]
            The document to be stored.
        memory_store_type : MemoryType
            The type of memory store (e.g., "persona", "toolbox", etc.)
        
        Returns:
        --------
        str
            The ID of the inserted document.
        """

        if memory_store_type == MemoryType.PERSONAS:
            result = self.persona_collection.insert_one(data)
        elif memory_store_type == MemoryType.TOOLBOX:         
            result = self.toolbox_collection.insert_one(data)
        elif memory_store_type == MemoryType.SHORT_TERM_MEMORY:
            result = self.short_term_memory_collection.insert_one(data)
        elif memory_store_type == MemoryType.LONG_TERM_MEMORY:
            result = self.long_term_memory_collection.insert_one(data)
        elif memory_store_type == MemoryType.CONVERSATION_MEMORY:
            result = self.conversation_memory_collection.insert_one(data)
        elif memory_store_type == MemoryType.WORKFLOW_MEMORY:
            result = self.workflow_memory_collection.insert_one(data)
            
        return str(result.inserted_id)

    def retrieve_by_query(self, query: Dict[str, Any], memory_store_type: MemoryType, limit: int = 1) -> Optional[Dict[str, Any]]:
        """
        Retrieve a document from MongoDB.
        
        Parameters:
        -----------
        query : Dict[str, Any]
            The query to use for retrieval.
        
        Returns:
        --------
        Optional[Dict[str, Any]]
            The retrieved document, or None if not found.
        """
        if memory_store_type == MemoryType.PERSONAS:
            return self.persona_collection.find(query, limit=limit)
        elif memory_store_type == MemoryType.TOOLBOX:
            return self.retrieve_toolbox_item(query, limit)
        elif memory_store_type == MemoryType.SHORT_TERM_MEMORY:
            return self.short_term_memory_collection.find(query, limit=limit)
        elif memory_store_type == MemoryType.LONG_TERM_MEMORY:
            return self.long_term_memory_collection.find(query, limit=limit)
        elif memory_store_type == MemoryType.CONVERSATION_MEMORY:
            return self.conversation_memory_collection.find(query, limit=limit)
        elif memory_store_type == MemoryType.WORKFLOW_MEMORY:
            return self.workflow_memory_collection.find(query, limit=limit)
        
    def retrieve_by_id(self, id: str, memory_store_type: MemoryType) -> Optional[Dict[str, Any]]:
        """
        Retrieve a document from MongoDB by id.

        Parameters:
        -----------
        id : str
            The id of the document to retrieve.
        memory_store_type : MemoryType
            The type of memory store (e.g., "persona", "toolbox", etc.)§

        Returns:
        --------
        Optional[Dict[str, Any]]
            The retrieved document, or None if not found.
        """
        if memory_store_type == MemoryType.PERSONAS:
            return self.persona_collection.find_one({"persona_id": id}, {"embedding": 0})
        elif memory_store_type == MemoryType.TOOLBOX:
            return self.toolbox_collection.find_one({"tool_id": id}, {"embedding": 0})
        elif memory_store_type == MemoryType.SHORT_TERM_MEMORY:
            return self.short_term_memory_collection.find_one({"short_term_memory_id": id})
        elif memory_store_type == MemoryType.LONG_TERM_MEMORY:
            return self.long_term_memory_collection.find_one({"long_term_memory_id": id})
        elif memory_store_type == MemoryType.CONVERSATION_MEMORY:
            return self.conversation_memory_collection.find_one({"conversation_id": id})
        elif memory_store_type == MemoryType.WORKFLOW_MEMORY:
            return self.workflow_memory_collection.find_one({"workflow_id": id})
    
    def retrieve_by_name(self, name: str, memory_store_type: MemoryType) -> Optional[Dict[str, Any]]:
        """
        Retrieve a document from MongoDB by name.
        
        Parameters:
        -----------
        name : str
            The name of the document to retrieve.
        memory_store_type : MemoryType
            The type of memory store to retrieve from.
        
        Returns:
        --------
        Optional[Dict[str, Any]]
            The retrieved document, or None if not found.
        """
        if memory_store_type == MemoryType.TOOLBOX:
            # Use projection in find_one directly
            return self.toolbox_collection.find_one(
                {"name": name},
                {"embedding": 0}  # Exclude embedding field
            )
        elif memory_store_type == MemoryType.PERSONAS:
            return self.persona_collection.find_one({"name": name}, {"embedding": 0})
        elif memory_store_type == MemoryType.SHORT_TERM_MEMORY:
            return self.short_term_memory_collection.find_one({"name": name})
        elif memory_store_type == MemoryType.LONG_TERM_MEMORY:
            return self.long_term_memory_collection.find_one({"name": name})
        elif memory_store_type == MemoryType.CONVERSATION_MEMORY:
            return self.conversation_memory_collection.find_one({"name": name})
        elif memory_store_type == MemoryType.WORKFLOW_MEMORY:
            return self.workflow_memory_collection.find_one({"name": name})

    def retrieve_toolbox_item(self, query: Dict[str, Any], limit: int = 1) -> Optional[Dict[str, Any]]:
        """
        Retrieve a toolbox item or several items from MongoDB.
        This function uses a vector search to retrieve the most similar toolbox items.
        Parameters:
        -----------
        query : Dict[str, Any]
            The query to use for retrieval.
        limit : int
            The maximum number of toolbox items to return.
        
        Returns:
        --------
        Optional[List[Dict[str, Any]]]
            The retrieved toolbox items, or None if not found.
        """

        # Get the embedding for the query
        embedding = get_embedding(query)

        # Create the vector search pipeline
        pipeline = [
            {
                "$vectorSearch": {
                    "queryVector": embedding,
                    "path": "embedding",
                    "numCandidates": 100,
                    "limit": limit,
                    "index": "vector_index"
                }
            },
            {
                "$project": {
                    "_id": 1,
                    "embedding": 0,
                    "score": { "$meta": "vectorSearchScore" }
                }
            }
        ]

        # Execute the vector search
        results = list(self.toolbox_collection.aggregate(pipeline))

        # Return the results
        return results if results else None

    def delete_by_id(self, id: str, memory_store_type: MemoryType) -> bool:
        """
        Delete a document from MongoDB by id.
        
        Parameters:
        -----------
        id : str
            The id of the document to delete.
        memory_store_type : MemoryType
            The type of memory store (e.g., "persona", "toolbox", etc.)
        
        Returns:
        --------
        bool
            True if deletion was successful, False otherwise.
        """
        if memory_store_type == MemoryType.PERSONAS:
            result = self.persona_collection.delete_one({"persona_id": id})
        elif memory_store_type == MemoryType.TOOLBOX:
            result = self.toolbox_collection.delete_one({"tool_id": id})
        elif memory_store_type == MemoryType.SHORT_TERM_MEMORY:
            result = self.short_term_memory_collection.delete_one({"short_term_memory_id": id})
        elif memory_store_type == MemoryType.LONG_TERM_MEMORY:
            result = self.long_term_memory_collection.delete_one({"long_term_memory_id": id})
        elif memory_store_type == MemoryType.CONVERSATION_MEMORY:
            result = self.conversation_memory_collection.delete_one({"conversation_id": id})
        elif memory_store_type == MemoryType.WORKFLOW_MEMORY:
            result = self.workflow_memory_collection.delete_one({"workflow_id": id})
            
        return result.deleted_count > 0
    
    def delete_by_name(self, name: str, memory_store_type: MemoryType) -> bool:
        """
        Delete a document from MongoDB by name.

        Parameters:
        -----------
        name : str
            The name of the document to delete.
        memory_store_type : MemoryType
            The type of memory store (e.g., "persona", "toolbox", etc.)
        
        Returns:
        --------
        bool
            True if deletion was successful, False otherwise.
        """
        if memory_store_type == MemoryType.TOOLBOX:
            result = self.toolbox_collection.delete_one({"name": name})
        elif memory_store_type == MemoryType.PERSONAS:
            result = self.persona_collection.delete_one({"name": name})
        elif memory_store_type == MemoryType.SHORT_TERM_MEMORY:
            result = self.short_term_memory_collection.delete_one({"name": name})
        elif memory_store_type == MemoryType.LONG_TERM_MEMORY:
            result = self.long_term_memory_collection.delete_one({"name": name}) 
        elif memory_store_type == MemoryType.CONVERSATION_MEMORY:
            result = self.conversation_memory_collection.delete_one({"name": name})
        elif memory_store_type == MemoryType.WORKFLOW_MEMORY:
            result = self.workflow_memory_collection.delete_one({"name": name})
        
        return result.deleted_count > 0

    def delete_all(self, memory_store_type: MemoryType) -> bool:
        """
        Delete all documents within a memory store type in MongoDB.
        
        Parameters:
        -----------
        memory_store_type : MemoryType
            The type of memory store (e.g., "persona", "toolbox", etc.)
        
        Returns:
        --------
        bool
            True if deletion was successful, False otherwise.
        """
        if memory_store_type == MemoryType.PERSONAS:
            result = self.persona_collection.delete_many({})
        elif memory_store_type == MemoryType.TOOLBOX:
            result = self.toolbox_collection.delete_many({})
        elif memory_store_type == MemoryType.SHORT_TERM_MEMORY:
            result = self.short_term_memory_collection.delete_many({})
        elif memory_store_type == MemoryType.LONG_TERM_MEMORY:
            result = self.long_term_memory_collection.delete_many({})
        elif memory_store_type == MemoryType.CONVERSATION_MEMORY:
            result = self.conversation_memory_collection.delete_many({})
        elif memory_store_type == MemoryType.WORKFLOW_MEMORY:
            result = self.workflow_memory_collection.delete_many({})
            
        return result.deleted_count > 0
            
    
    def list_all(self, memory_store_type: MemoryType) -> List[Dict[str, Any]]:
        """
        List all documents within a memory store type in MongoDB.

        Parameters:
        -----------
        memory_store_type : MemoryType
            The type of memory store (e.g., "persona", "toolbox", etc.)
        
        Returns:
        --------
        List[Dict[str, Any]]
            The list of all documents from MongoDB.
        """

        if memory_store_type == MemoryType.PERSONAS:
            return list(self.persona_collection.find({}, {"embedding": 0}))
        elif memory_store_type == MemoryType.TOOLBOX:
            return list(self.toolbox_collection.find({}, {"embedding": 0}))
        elif memory_store_type == MemoryType.SHORT_TERM_MEMORY:
            return list(self.short_term_memory_collection.find())
        elif memory_store_type == MemoryType.LONG_TERM_MEMORY:
            return list(self.long_term_memory_collection.find())
        elif memory_store_type == MemoryType.CONVERSATION_MEMORY:
            return list(self.conversation_memory_collection.find())
        elif memory_store_type == MemoryType.WORKFLOW_MEMORY:
            return list(self.workflow_memory_collection.find())
        
    def update_by_id(self, id: str, data: Dict[str, Any], memory_store_type: MemoryType) -> bool:
        """
        Update a document in a memory store type in MongoDB by id.

        Parameters:
        -----------
        id : str
            The id of the document to update.
        data : Dict[str, Any]
            The data to update the document with.
        memory_store_type : MemoryType
            The type of memory store (e.g., "persona", "toolbox", etc.)
        
        Returns:
        --------
        bool
            True if update was successful, False otherwise.
        """
        if memory_store_type == MemoryType.PERSONAS:
            result = self.persona_collection.update_one({"persona_id": id}, {"$set": data})
        elif memory_store_type == MemoryType.TOOLBOX:


            result = self.toolbox_collection.update_one({"tool_id": id}, {"$set": data})
        elif memory_store_type == MemoryType.SHORT_TERM_MEMORY:
            result = self.short_term_memory_collection.update_one({"short_term_memory_id": id}, {"$set": data})
        elif memory_store_type == MemoryType.LONG_TERM_MEMORY:
            result = self.long_term_memory_collection.update_one({"long_term_memory_id": id}, {"$set": data})
        elif memory_store_type == MemoryType.CONVERSATION_MEMORY:
            result = self.conversation_memory_collection.update_one({"conversation_id": id}, {"$set": data})
        elif memory_store_type == MemoryType.WORKFLOW_MEMORY:
            result = self.workflow_memory_collection.update_one({"workflow_id": id}, {"$set": data})
            
        return result.modified_count > 0
            
            
    def update_toolbox_item(self, id: str, data: Dict[str, Any]) -> bool:
        """
        Update a toolbox item in MongoDB by id.
        """

        # Update the emebdding if the name, docstring or signature has changed

        # Get the old data
        old_data = self.retrieve_by_id(id, MemoryType.TOOLBOX)

        # Concatenate the name, docstring and signature if any of them have changed
        if old_data["name"] != data["name"]:
            data["name"] = data["name"]
        if old_data["docstring"] != data["docstring"]:
            data["docstring"] = data["docstring"]
        if old_data["signature"] != data["signature"]:
            data["signature"] = data["signature"]

        # Update the embedding
        data["embedding"] = get_embedding(data["name"] + " " + data["docstring"] + " " + data["signature"])

        result = self.toolbox_collection.update_one({"tool_id": id}, {"$set": data})
        return result.modified_count > 0
    
    def _setup_vector_search_index(self, collection, index_name="vector_index"):
        """
        Setup a vector search index for a MongoDB collection and wait for 30 seconds.

        Args:
        collection: MongoDB collection object
        index_name: Name of the index (default: "vector_index")
        """

        # Define the index definition
        vector_index_definition = {
            "fields": [
                {
                    "type": "vector",
                    "path": "embedding",
                    "numDimensions": get_embedding_dimensions("text-embedding-3-small"),
                    "similarity": "cosine",
                }
            ]
        }

        new_vector_search_index_model = SearchIndexModel(
            definition=vector_index_definition, name=index_name, type="vectorSearch"
        )

        # Create the new index
        try:
            result = collection.create_search_index(model=new_vector_search_index_model)
            print(f"Creating index '{index_name}'... for collection {collection.name}")
            return result

        except Exception as e:
            print(f"Error creating new vector search index '{index_name}': {e!s}")
            return None
        
    def _ensure_vector_index(self, collection, index_name="vector_index"):
        search_indexes = list(collection.list_search_indexes())
        has_vector_index = any(index.get("name") == index_name and index.get("type") == "vectorSearch" for index in search_indexes)
        if not has_vector_index:
            self._setup_vector_search_index(collection, index_name)

    def close(self) -> None:
        """Close the connection to MongoDB."""
        self.client.close()
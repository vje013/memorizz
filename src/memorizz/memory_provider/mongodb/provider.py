from pymongo import MongoClient
from pymongo.operations import SearchIndexModel
from ..base import MemoryProvider
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, TYPE_CHECKING
from src.memorizz.memory_provider.memory_type import MemoryType
from src.memorizz.embeddings.openai import get_embedding
from src.memorizz.embeddings.openai import get_embedding_dimensions
from bson import ObjectId
from pymongo.collection import Collection
from pymongo.database import Database
from datetime import datetime
import pprint
from src.memorizz.memory_component.memory_mode import MemoryMode
from src.memorizz.memagent import MemAgentModel
from src.memorizz.persona.persona import Persona
from src.memorizz.persona.role_type import RoleType

# Use TYPE_CHECKING for forward references to avoid circular imports
# if TYPE_CHECKING:
#     from src.memorizz.memagent import MemAgentModel

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
        self.memagent_collection = self.db[MemoryType.MEMAGENT.value]

        # Create all memory stores in MongoDB.
        self._create_memory_stores()

        # Create a vector index for each memory store in MongoDB.
        self._create_vector_indexes_for_memory_stores()

    def _create_memory_stores(self) -> None:
        """
        Create all memory stores in MongoDB.
        """
        self._create_memory_store(MemoryType.MEMAGENT)
        self._create_memory_store(MemoryType.PERSONAS)
        self._create_memory_store(MemoryType.TOOLBOX)
        self._create_memory_store(MemoryType.SHORT_TERM_MEMORY)
        self._create_memory_store(MemoryType.LONG_TERM_MEMORY)
        self._create_memory_store(MemoryType.CONVERSATION_MEMORY)
        self._create_memory_store(MemoryType.WORKFLOW_MEMORY)
    
    def _create_memory_store(self, memory_store_type: MemoryType) -> None:
        """
        Create a new memory store in MongoDB.

        Parameters:
        -----------
        memory_store_type : MemoryType
            The type of memory store to create.

        Returns:
        --------
        None
        """

        # Create collection if it doesn't exist within the database/memory provider
        # Check if the collection exists within the database and if it doesn't, create an empty collection
        for memory_store_type in MemoryType:
            if memory_store_type.value not in self.db.list_collection_names():
                self.db.create_collection(memory_store_type.value)
                print(f"Created collection: {memory_store_type.value} successfully.")
        

    def _create_vector_indexes_for_memory_stores(self) -> None:
        """
        Create a vector index for each memory store in MongoDB.

        Returns:
        --------
        None
        """

        self._ensure_vector_index(self.persona_collection, "vector_index", memory_store=False)
        self._ensure_vector_index(self.toolbox_collection, "vector_index", memory_store=True)
        self._ensure_vector_index(self.short_term_memory_collection, "vector_index", memory_store=True)
        self._ensure_vector_index(self.long_term_memory_collection, "vector_index", memory_store=True)
        self._ensure_vector_index(self.conversation_memory_collection, "vector_index", memory_store=True)
        self._ensure_vector_index(self.workflow_memory_collection, "vector_index", memory_store=True)
        self._ensure_vector_index(self.memagent_collection, "vector_index", memory_store=True)


        for memory_store_type in MemoryType:
            if memory_store_type == MemoryType.PERSONAS:
                memory_store_present = False
            else:
                memory_store_present = True

            self._ensure_vector_index(
                collection=self.db[memory_store_type.value],
                index_name="vector_index",
                memory_store=memory_store_present,
            )
            
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
            The ID of the inserted/updated document.
        """
        # Get the appropriate collection based on memory type
        collection = None
        if memory_store_type == MemoryType.PERSONAS:
            collection = self.persona_collection
        elif memory_store_type == MemoryType.TOOLBOX:
            collection = self.toolbox_collection
        elif memory_store_type == MemoryType.SHORT_TERM_MEMORY:
            collection = self.short_term_memory_collection
        elif memory_store_type == MemoryType.LONG_TERM_MEMORY:
            collection = self.long_term_memory_collection
        elif memory_store_type == MemoryType.CONVERSATION_MEMORY:
            collection = self.conversation_memory_collection
        elif memory_store_type == MemoryType.WORKFLOW_MEMORY:
            collection = self.workflow_memory_collection

        if collection is None:
            raise ValueError(f"Invalid memory store type: {memory_store_type}")

        # If the document has an _id, try to update it
        if "_id" in data:
            result = collection.update_one(
                {"_id": data["_id"]},
                {"$set": data},
                upsert=True
            )
            return str(data["_id"])
        else:
            # If no _id, insert as new document
            result = collection.insert_one(data)
            return str(result.inserted_id)

    def retrieve_by_query(self, query: Dict[str, Any], memory_store_type: MemoryType, limit: int = 1) -> Optional[Dict[str, Any]]:
        """
        Retrieve a document from MongoDB.
        
        Parameters:
        -----------
        query : Dict[str, Any]
            The query to use for retrieval.
        limit : int
            The maximum number of documents to return.
        
        Returns:
        --------
        Optional[Dict[str, Any]]
            The retrieved document, or None if not found.
        """
        
        if memory_store_type == MemoryType.PERSONAS:
            return self.retrieve_persona_by_query(query, limit=limit)
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


    def retrieve_persona_by_query(self, query: Dict[str, Any], limit: int = 1) -> Optional[Dict[str, Any]]:
        """
        Retrieve a persona or several personas from MongoDB.
        This function uses a vector search to retrieve the most similar personas.

        Parameters:
        -----------
        query : Dict[str, Any]

        Returns:
        --------
        Optional[List[Dict[str, Any]]]
            The retrieved personas, or None if not found.
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
        results = list(self.persona_collection.aggregate(pipeline))

        # Return the results
        return results if results else None
        

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
    

    def retrieve_conversation_history_ordered_by_timestamp(self, memory_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve the conversation history ordered by timestamp.

        Parameters:
        -----------
        memory_id : str
            The id of the memory to retrieve the conversation history for.

        Returns:
        --------
        List[Dict[str, Any]]
            The conversation history ordered by timestamp.
        """
        return list(self.conversation_memory_collection.find({"memory_id": memory_id}, {"embedding": 0}).sort("timestamp", 1))
    
    def retrieve_memory_components_by_query(self, query: str = None, query_embedding: list[float] = None, memory_id: str = None, memory_type: MemoryType = None, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve memory components by query.

        Parameters:
        -----------
        query : str
            The query to use for retrieval.
        query_embedding : list[float]
            The embedding of the query.
        memory_id : str
            The id of the memory to retrieve the memory components for.
        memory_type : MemoryType
            The type of memory to retrieve the memory components for.
        limit : int
            The maximum number of memory components to return.

        Returns:
        --------
        List[Dict[str, Any]]
            The memory components ordered by timestamp.
        """

        # Detect the memory type
        if memory_type == MemoryType.CONVERSATION_MEMORY:
            return self.get_conversation_memory_components(query, query_embedding, memory_id, limit)
        elif memory_type == MemoryType.TASK_MEMORY:
            pass
            # TODO: return self.get_task_memory_components(query, memory_id, limit)
        elif memory_type == MemoryType.WORKFLOW_MEMORY:
            pass
            # TODO: return self.get_workflow_memory_components(query, memory_id, limit)


    def get_conversation_memory_components(self, query: str = None, query_embedding: list[float] = None, memory_id: str = None, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get the conversation memory components.

        Parameters:
        -----------
        query : str
            The query to use for retrieval.
        query_embedding : list[float]
            The embedding of the query.
        memory_id : str
            The id of the memory to retrieve the memory components for.
        limit : int
            The maximum number of memory components to return.

        Returns:
        --------
        List[Dict[str, Any]]
            The memory components ordered by timestamp.
        """

        # If the query embedding is not provided, then we create it
        if query_embedding is None and query is not None:
            query_embedding = get_embedding(query)

        vector_stage = {
            "$vectorSearch": {
                "index": "vector_index",
                "queryVector": query_embedding,
                "path": "embedding",
                "numCandidates": 100,
                "limit": limit,
                "filter": {"memory_id": memory_id}
            }
        }

        # Add the vector stage to the pipeline
        pipeline = [
            vector_stage,
            {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
            {"$sort": {"score": -1, "timestamp": 1}},
            {"$unset": ["_id"]},
        ]

        # Execute the pipeline
        results = list(self.conversation_memory_collection.aggregate(pipeline))

        # Return the results
        return results
    
    def store_memagent(self, memagent: "MemAgentModel") -> "MemAgentModel":
        """
        Store a memagent in the MongoDB database.
        
        Parameters:
        -----------
        memagent : MemAgentModel
            The memagent to be stored.
        
        Returns:
        --------
        MemAgentModel
            The stored memagent.
        """

        # Check to ensure the memagent agent_id doesn't already exist
        if self.memagent_collection.find_one({"agent_id": memagent.agent_id}):
            raise ValueError(f"MemAgent with id {memagent.agent_id} already exists")
        
        # Convert the MemAgentModel to a dictionary
        memagent_dict = memagent.model_dump()
        
        # Convert persona to a serializable format if it exists
        if memagent.persona:
            # Store the entire persona object as a serializable dictionary
            memagent_dict["persona"] = memagent.persona.to_dict()
        
        # Remove any function objects from tools that could cause serialization issues
        if memagent_dict.get("tools") and isinstance(memagent_dict["tools"], list):
            for tool in memagent_dict["tools"]:
                if "function" in tool and callable(tool["function"]):
                    del tool["function"]
        
        self.memagent_collection.insert_one(memagent_dict)

        return memagent_dict
    
    def update_memagent(self, memagent: "MemAgentModel") -> "MemAgentModel":
        """
        Update a memagent in the MongoDB database.
        """
        # Convert the MemAgentModel to a dictionary
        memagent_dict = memagent.model_dump()

        # Convert persona to a serializable format if it exists
        if memagent.persona:
            memagent_dict["persona"] = memagent.persona.to_dict()
        
        # Remove any function objects from tools that could cause serialization issues
        if memagent_dict.get("tools") and isinstance(memagent_dict["tools"], list):
            for tool in memagent_dict["tools"]:
                if "function" in tool and callable(tool["function"]):
                    del tool["function"]
        
        # Update the memagent in the MongoDB database
        self.memagent_collection.update_one({"agent_id": memagent.agent_id}, {"$set": memagent_dict})

        return memagent_dict

    
    def retrieve_memagent(self, agent_id: str) -> "MemAgentModel":
        """
        Retrieve a memagent from the MongoDB database.
        
        Parameters:
        -----------
        agent_id : str
            The agent ID to retrieve.
        
        Returns:
        --------
        MemAgentModel
            The retrieved memagent.
        """
        from src.memorizz.memory_component.memory_mode import MemoryMode
        
        # Get the document from MongoDB
        document = self.memagent_collection.find_one({"agent_id": agent_id})
        if not document:
            return None
        
        # Create a new MemAgent with data from the document
        memagent = MemAgentModel(
            instruction=document.get("instruction"),
            memory_mode=document.get("memory_mode"),
            max_steps=document.get("max_steps"),
            memory_ids=document.get("memory_ids") or [],
            agent_id=document.get("agent_id"),
            tools=document.get("tools"),
            memory_provider=self
        )
        
        # Construct persona if present in the document
        if document.get("persona"):
            persona_data = document.get("persona")
            # Handle role as a string by matching it to a RoleType enum
            role_str = persona_data.get("role")
            role = None
            
            # Match the string role to a RoleType enum
            for role_type in RoleType:
                if role_type.value == role_str:
                    role = role_type
                    break
            
            # If no matching enum is found, default to GENERAL
            if role is None:
                role = RoleType.GENERAL
                
            memagent.persona = Persona(
                name=persona_data.get("name"),
                role=role,  # Pass the RoleType enum instead of string
                goals=persona_data.get("goals"),
                background=persona_data.get("background"),
                persona_id=persona_data.get("persona_id")
            )

        return memagent
    
    def delete_memagent(self, agent_id: str) -> bool:
        """
        Delete a memagent from the memory provider.

        Parameters:
        -----------
        agent_id : str
            The id of the memagent to delete.

        Returns:
        --------
        bool
            True if deletion was successful, False otherwise.
        """
        result = self.memagent_collection.delete_one({"agent_id": agent_id})

        return result.deleted_count > 0
    
    def list_memagents(self) -> List["MemAgentModel"]:
        """
        List all memagents in the MongoDB database.
        
        Returns:
        --------
        List[MemAgentModel]
            The list of memagents.
        """
        
        documents = list(self.memagent_collection.find())
        agents = []
        
        for doc in documents:
            agent = MemAgentModel(
                instruction=doc.get("instruction"),
                memory_mode=doc.get("memory_mode"),
                max_steps=doc.get("max_steps"),
                memory_ids=doc.get("memory_ids") or [],
                agent_id=doc.get("agent_id"),
                tools=doc.get("tools"),  # Include tools from document
                memory_provider=self
            )
            
            # Construct persona if present in the document
            if doc.get("persona"):
                persona_data = doc.get("persona")
                # Handle role as a string by matching it to a RoleType enum
                role_str = persona_data.get("role")
                role = None
                
                # Match the string role to a RoleType enum
                for role_type in RoleType:
                    if role_type.value == role_str:
                        role = role_type
                        break
                
                # If no matching enum is found, default to GENERAL
                if role is None:
                    role = RoleType.GENERAL
                    
                agent.persona = Persona(
                    name=persona_data.get("name"),
                    role=role,  # Pass the RoleType enum instead of string
                    goals=persona_data.get("goals"),
                    background=persona_data.get("background"),
                    persona_id=persona_data.get("persona_id")
                )
                
            agents.append(agent)
            
        return agents

    
    def update_memagent_memory_ids(self, agent_id: str, memory_ids: List[str]) -> bool:
        """
        Update the memory_ids of a memagent in the memory provider.

        Parameters:
        -----------
        agent_id : str
            The id of the memagent to update.
        memory_ids : List[str]
            The list of memory_ids to update.

        Returns:
        --------
        bool
            True if update was successful, False otherwise.
        """
        result = self.memagent_collection.update_one(
            {"agent_id": agent_id}, 
            {"$set": {"memory_ids": memory_ids}}
        )

        return result.modified_count > 0
    
    def delete_memagent_memory_ids(self, agent_id: str) -> bool:
        """
        Delete the memory_ids of a memagent in the memory provider.

        Parameters:
        -----------
        agent_id : str
            The id of the memagent to update.

        Returns:
        --------
        bool
            True if deletion was successful, False otherwise.
        """
        result = self.memagent_collection.update_one({"agent_id": agent_id}, {"$unset": {"memory_ids": []}})

        return result.modified_count > 0
    
    def delete_memagent(self, agent_id: str, cascade: bool = False) -> bool:
        """
        Delete a memagent from the memory provider by id.

        Parameters:
        -----------
        agent_id : str
            The id of the memagent to delete.
        cascade : bool
            Whether to cascade the deletion of the memagent. This deletes all the memory components associated with the memagent by deleting the memory_ids and their corresponding memory store in the memory provider.

        Returns:
        --------
        bool
            True if deletion was successful, False otherwise.
        """
        if cascade:
            # Retrieve the memagent
            memagent = self.retrieve_memagent(agent_id)

            if memagent is None:
                raise ValueError(f"MemAgent with id {agent_id} not found")
            
            # Delete all the memory components associated with the memagent by deleting the memory_ids and their corresponding memory store in the memory provider.
            for memory_id in memagent.memory_ids:
                # Loop through all the memory stores and delete records with the memory_ids
                for memory_type in MemoryType:
                    self._delete_memory_components_by_memory_id(memory_id, memory_type)
        else:
            result = self.memagent_collection.delete_one({"agent_id": agent_id})

        return result.deleted_count > 0
    
    def _delete_memory_components_by_memory_id(self, memory_id: str, memory_type: MemoryType):
        """
        Delete all the memory components associated with the memory_id.

        Parameters:
        -----------
        memory_id : str
            The id of the memory to delete.
        memory_type : MemoryType
            The type of memory to delete.

        Returns:
        --------
        bool
            True if deletion was successful, False otherwise.
        """
        if memory_type == MemoryType.CONVERSATION_MEMORY:
            self.conversation_memory_collection.delete_many({"memory_id": memory_id})
        elif memory_type == MemoryType.TASK_MEMORY:
            self.task_memory_collection.delete_many({"memory_id": memory_id})
        elif memory_type == MemoryType.WORKFLOW_MEMORY:
            self.workflow_memory_collection.delete_many({"memory_id": memory_id})
        elif memory_type == MemoryType.SHORT_TERM_MEMORY:
            self.short_term_memory_collection.delete_many({"memory_id": memory_id})
        elif memory_type == MemoryType.LONG_TERM_MEMORY:
            self.long_term_memory_collection.delete_many({"memory_id": memory_id})
        elif memory_type == MemoryType.PERSONAS:
            self.persona_collection.delete_many({"memory_id": memory_id})
        elif memory_type == MemoryType.TOOLBOX:
            self.toolbox_collection.delete_many({"memory_id": memory_id})
        elif memory_type == MemoryType.MEMAGENT:
            self.memagent_collection.delete_many({"memory_id": memory_id})
                

    def _setup_vector_search_index(self, collection, index_name="vector_index", memory_store: bool = False):
        """
        Setup a vector search index for a MongoDB collection and wait for 30 seconds.

        Args:
        collection: MongoDB collection object
        index_name: Name of the index (default: "vector_index")
        memory_store: Whether to add the memory_id field to the index (default: False)
        """

        # Define the index definition
        vector_index_definition = {
            "fields": [
                {
                    "type": "vector",
                    "path": "embedding",
                    # TODO: Make this dynamic based on the embedding model
                    "numDimensions": get_embedding_dimensions("text-embedding-3-small"),
                    "similarity": "cosine",
                }
            ]
        }

        # If the memory store is true, then we add the memory_id field to the index
        # This is used to prefilter the memory components by memory_id
        # useful to narrow the scope of your semantic search and ensure that not all vectors are considered for comparison. 
        # It reduces the number of documents against which to run similarity comparisons, which can decrease query latency and increase the accuracy of search results.
        if memory_store:
            vector_index_definition["fields"].append({
                "type": "filter",
                "path": "memory_id",
            })

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
        
    def _ensure_vector_index(self, collection, index_name="vector_index", memory_store: bool = False):
        search_indexes = list(collection.list_search_indexes())
        has_vector_index = any(index.get("name") == index_name and index.get("type") == "vectorSearch" for index in search_indexes)
        if not has_vector_index:
            self._setup_vector_search_index(collection, index_name, memory_store)
            print(f"Created vector index for {collection.name} collection successfully.")

    def close(self) -> None:
        """Close the connection to MongoDB."""
        self.client.close()
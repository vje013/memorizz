from typing import Dict, Any, List, Callable, Optional
from ..memory_provider import MemoryProvider
from ..memory_provider.memory_type import MemoryType
from ..embeddings.openai import get_embedding
import inspect
import uuid

class Toolbox:
    """A toolbox for managing and retrieving tools using a memory provider."""
    
    def __init__(self, provider: MemoryProvider):
        """
        Initialize the toolbox with a memory provider.
        
        Parameters:
        -----------
        provider : MemoryProvider
            The memory provider to use for storing and retrieving tools.
        """
        self.provider = provider
        self._tools: Dict[str, Callable] = {}  # In-memory storage of functions

    def register_tool(self, func: Callable) -> str:
        """
        Register a function as a tool in the toolbox.
        
        Parameters:
        -----------
        func : Callable
            The function to register as a tool.
        
        Returns:
        --------
        str
            The ID of the registered tool.
        """
        # Get the function's docstring and signature
        docstring = func.__doc__ or ""
        signature = str(inspect.signature(func))
        
        # Generate embedding for the tool
        embedding = get_embedding(f"{func.__name__} {docstring} {signature}")
        
        # Generate a unique tool ID
        tool_id = str(uuid.uuid4())
        
        # Create tool data (without the function)
        tool_data = {
            "tool_id": tool_id,
            "type": "function",
            "name": func.__name__,
            "docstring": docstring,
            "signature": signature,
            "embedding": embedding
        }
        
        # Store the tool metadata in the memory provider
        self.provider.store(tool_data, memory_store_type=MemoryType.TOOLBOX)
        
        # Store the actual function in memory
        self._tools[tool_id] = func
        
        return tool_id

    def get_tool_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get a single tool by its name.
        
        Parameters:
        -----------
        name : str
            The name of the tool to retrieve.
        
        Returns:
        --------
        Dict[str, Any]
            The tool data, or None if not found.
        
        """
        # First check if we have the function in memory
        if name in self._tools:
            return self._tools[name]
        
        # If not, try to get it from the provider
        # One thing to note is that the name is not unique and we get a single tool matching the name
        tool_data = self.provider.retrieve_by_name(name, memory_store_type=MemoryType.TOOLBOX)

        if tool_data:
            return tool_data
        
        return None
    
    def get_tool_by_id(self, id: str) -> Optional[Dict[str, Any]]:
        """
        Get a tool by its id.

        Parameters:
        -----------
        id : str
            The id of the tool to retrieve.
        
        Returns:
        --------
        Dict[str, Any]
            The tool data, or None if not found.
        """
        return self.provider.retrieve_by_id(id, memory_store_type=MemoryType.TOOLBOX)

    def get_most_similar_tools(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get the most similar tools to a query.
        
        Parameters:
        -----------
        query : str
            The query to search for.
        limit : int, optional
            The maximum number of tools to return.
        
        Returns:
        --------
        List[Dict[str, Any]]
            A list of the most similar tools.
        """
        similar_tools = self.provider.retrieve_by_query(
            query,
            memory_store_type=MemoryType.TOOLBOX,
            limit=limit
        )
        
        # Add the actual functions to the results
        for tool in similar_tools:
            tool_id = tool.get("tool_id")
            if tool_id in self._tools:
                tool["function"] = self._tools[tool_id]
        
        return similar_tools

    def delete_tool_by_name(self, name: str) -> bool:
        """
        Delete a tool from the toolbox by name.
        
        Parameters:
        -----------
        name : str
            The name of the tool to delete.
        
        Returns:
        --------
        bool
            True if deletion was successful, False otherwise.
        """
        # Delete from memory
        if name in self._tools:
            del self._tools[name]
        
        # Delete from provider
        return self.provider.delete_by_name(name, memory_store_type=MemoryType.TOOLBOX)
    
    def delete_tool_by_id(self, id: str) -> bool:
        """
        Delete a tool from the toolbox by id.

        Parameters:
        -----------
        id : str
            The id of the tool to delete.
        
        Returns:
        --------
        bool
            True if deletion was successful, False otherwise.
        """
        return self.provider.delete_by_id(id, memory_store_type=MemoryType.TOOLBOX)
    
    def delete_all(self) -> bool:
        """
        Delete all tools in the toolbox.

        Returns:
        --------
        bool
            True if deletion was successful, False otherwise.
        """
        return self.provider.delete_all(memory_store_type=MemoryType.TOOLBOX)
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """
        List all tools in the toolbox.
        
        Returns:
        --------
        List[Dict[str, Any]]
            A list of all tools in the toolbox.
        """
        tools = self.provider.list_all(memory_store_type=MemoryType.TOOLBOX)
        
        # Add the actual functions to the results
        for tool in tools:
            tool_id = tool.get("tool_id")
            if tool_id in self._tools:
                tool["function"] = self._tools[tool_id]
        
        return tools
    
    def update_tool_by_id(self, id: str, data: Dict[str, Any]) -> bool:
        """
        Update a tool in the toolbox by id.

        Parameters:
        -----------
        id : str
            The id of the tool to update.
        data : Dict[str, Any]
            The data to update the tool with.
        
        Returns:
        --------
        bool
            True if update was successful, False otherwise.
        """
        return self.provider.update_by_id(id, data, memory_store_type=MemoryType.TOOLBOX)
            
    
    
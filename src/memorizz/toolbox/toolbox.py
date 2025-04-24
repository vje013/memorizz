from typing import Dict, Any, List, Callable, Optional, Union, TYPE_CHECKING
from ..memory_provider import MemoryProvider
from ..memory_provider.memory_type import MemoryType
from ..embeddings.openai import get_embedding
# Use TYPE_CHECKING for forward references to avoid circular imports
if TYPE_CHECKING:
    from ..llms.openai import OpenAI
import inspect
import uuid
from .tool_schema import ToolSchemaType

# Initialize OpenAI lazily to avoid circular imports
def get_openai():
    from ..llms.openai import OpenAI
    return OpenAI()

class Toolbox:
    """A toolbox for managing and retrieving tools using a memory provider."""
    
    def __init__(self, memory_provider: MemoryProvider):
        """
        Initialize the toolbox with a memory provider.
        
        Parameters:
        -----------
        memory_provider : MemoryProvider
            The memory provider to use for storing and retrieving tools.
        """
        self.memory_provider = memory_provider
        
        # In-memory storage of functions
        self._tools: Dict[str, Callable] = {}

    def register_tool(self, func: Optional[Callable] = None, augment: bool = False) -> Union[str, Callable]:
        """
        Register a function as a tool in the toolbox.
        
        Parameters:
        -----------
        func : Callable, optional
            The function to register as a tool. If None, returns a decorator.
        augment : bool, optional
            Whether to augment the tool docstring with an LLM generated description.
            And also include to the metadata synthecially generated queries that are used in the embedding generation process and used to seach the tool.
        Returns:
        --------
        Union[str, Callable]
            If func is provided, returns the tool ID. Otherwise returns a decorator.
        """
        def decorator(f: Callable) -> str:
            # Get the function's docstring and signature
            docstring = f.__doc__ or ""
            signature = str(inspect.signature(f))
            tool_id = str(uuid.uuid4())

            if augment:
                # Extend the docstring with an LLM generated description
                docstring = self._augment_docstring(docstring)

                # Generate synthecially generated queries
                queries = self._generate_queries(docstring)

                # Generate embedding for the tool using the augmented docstring, function name, signature and queries
                embedding = get_embedding(f"{f.__name__} {docstring} {signature} {queries}")

                # Get the tool metadata
                tool_data = self._get_tool_metadata(f)
                
                # Create a dictionary with the tool data and embedding
                tool_dict = {
                    "tool_id": tool_id,
                    "embedding": embedding,
                    "queries": queries,
                    **tool_data.model_dump()
                }
            else:
                # Generate embedding for the tool using the function name, docstring and signature
                embedding = get_embedding(f"{f.__name__} {docstring} {signature}")

                # Get the tool metadata
                tool_data = self._get_tool_metadata(f)
                
                # Create a dictionary with the tool data and embedding
                tool_dict = {
                    "tool_id": tool_id,
                    "embedding": embedding,
                    **tool_data.model_dump()
                }
            
            # Store the tool metadata in the memory provider
            self.memory_provider.store(tool_dict, memory_store_type=MemoryType.TOOLBOX)
            
            # Store the actual function in memory
            self._tools[tool_id] = f
            
            return tool_id

        if func is None:
            return decorator
        return decorator(func)

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
        tool_data = self.memory_provider.retrieve_by_name(name, memory_store_type=MemoryType.TOOLBOX)

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
        return self.memory_provider.retrieve_by_id(id, memory_store_type=MemoryType.TOOLBOX)

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
        similar_tools = self.memory_provider.retrieve_by_query(
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
        return self.memory_provider.delete_by_name(name, memory_store_type=MemoryType.TOOLBOX)
    
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
        return self.memory_provider.delete_by_id(id, memory_store_type=MemoryType.TOOLBOX)
    
    def delete_all(self) -> bool:
        """
        Delete all tools in the toolbox.

        Returns:
        --------
        bool
            True if deletion was successful, False otherwise.
        """
        return self.memory_provider.delete_all(memory_store_type=MemoryType.TOOLBOX)
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """
        List all tools in the toolbox.
        
        Returns:
        --------
        List[Dict[str, Any]]
            A list of all tools in the toolbox.
        """
        tools = self.memory_provider.list_all(memory_store_type=MemoryType.TOOLBOX)
        
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
        return self.memory_provider.update_by_id(id, data, memory_store_type=MemoryType.TOOLBOX)
    
    def _get_tool_metadata(self, func: Callable) -> ToolSchemaType:
        """
        Get the metadata for a tool.
        """
        return get_openai().get_tool_metadata(func)
    
    def _augment_docstring(self, docstring: str) -> str:
        """
        Augment the docstring with an LLM generated description.
        """
        return get_openai().augment_docstring(docstring)
    
    def _generate_queries(self, docstring: str) -> List[str]:
        """
        Generate queries for the tool.
        """
        return get_openai().generate_queries(docstring)
            
    
    
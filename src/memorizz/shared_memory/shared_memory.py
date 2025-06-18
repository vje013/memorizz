from typing import Dict, Any, List, Optional
from datetime import datetime
from ..memory_provider import MemoryProvider
from ..memory_provider.memory_type import MemoryType
from bson import ObjectId
import json
import logging

logger = logging.getLogger(__name__)

class BlackboardEntry:
    """Individual entry in the shared memory blackboard."""
    
    def __init__(self, 
                 agent_id: str, 
                 content: Any, 
                 entry_type: str, 
                 created_at: datetime = None):
        self.agent_id = agent_id
        self.content = content
        self.entry_type = entry_type  # "tool_call", "conversation", "task_assignment", "result"
        self.created_at = created_at or datetime.now()
        self.memory_id = str(ObjectId())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "memory_id": self.memory_id,
            "agent_id": self.agent_id,
            "content": self.content,
            "entry_type": self.entry_type,
            "created_at": self.created_at.isoformat()
        }

class SharedMemory:
    """Shared memory system for multi-agent coordination."""
    
    def __init__(self, memory_provider: MemoryProvider):
        self.memory_provider = memory_provider
        
    def create_shared_session(self, 
                             root_agent_id: str, 
                             delegate_agent_ids: List[str] = None) -> str:
        """
        Create a new shared memory session for multi-agent coordination.
        
        Parameters:
            root_agent_id (str): The ID of the root/orchestrating agent
            delegate_agent_ids (List[str]): List of delegate agent IDs
            
        Returns:
            str: The shared session ID
        """
        shared_session = {
            "session_id": str(ObjectId()),
            "root_agent_id": root_agent_id,
            "delegate_agent_ids": delegate_agent_ids or [],
            "sub_agent_ids": [],  # Will be populated recursively
            "blackboard": [],
            "created_at": datetime.now().isoformat(),
            "status": "active"  # active, completed, failed
        }
        
        # Store in memory provider
        session_id = self.memory_provider.store(shared_session, MemoryType.SHARED_MEMORY)
        return str(session_id)
    
    def add_blackboard_entry(self, 
                           session_id: str, 
                           agent_id: str, 
                           content: Any, 
                           entry_type: str) -> bool:
        """
        Add an entry to the shared blackboard.
        
        Parameters:
            session_id (str): The shared session ID
            agent_id (str): The ID of the agent adding the entry
            content (Any): The content to add
            entry_type (str): Type of entry (tool_call, conversation, etc.)
            
        Returns:
            bool: Success status
        """
        try:
            # Get the shared session
            session = self.memory_provider.retrieve_by_id(session_id, MemoryType.SHARED_MEMORY)
            if not session:
                return False
            
            # Create blackboard entry
            entry = BlackboardEntry(agent_id, content, entry_type)
            
            # Add to blackboard
            session["blackboard"].append(entry.to_dict())
            
            # Update in storage
            return self.memory_provider.update_by_id(session_id, session, MemoryType.SHARED_MEMORY)
            
        except Exception as e:
            logger.error(f"Error adding blackboard entry: {e}")
            return False
    
    def get_blackboard_entries(self, 
                              session_id: str, 
                              agent_id: str = None, 
                              entry_type: str = None) -> List[Dict[str, Any]]:
        """
        Retrieve blackboard entries with optional filtering.
        
        Parameters:
            session_id (str): The shared session ID
            agent_id (str, optional): Filter by agent ID
            entry_type (str, optional): Filter by entry type
            
        Returns:
            List[Dict[str, Any]]: List of blackboard entries
        """
        try:
            session = self.memory_provider.retrieve_by_id(session_id, MemoryType.SHARED_MEMORY)
            if not session:
                return []
            
            entries = session.get("blackboard", [])
            
            # Apply filters
            if agent_id:
                entries = [e for e in entries if e.get("agent_id") == agent_id]
            if entry_type:
                entries = [e for e in entries if e.get("entry_type") == entry_type]
            
            return entries
            
        except Exception as e:
            logger.error(f"Error retrieving blackboard entries: {e}")
            return []
    
    def update_session_status(self, session_id: str, status: str) -> bool:
        """Update the status of a shared session."""
        try:
            session = self.memory_provider.retrieve_by_id(session_id, MemoryType.SHARED_MEMORY)
            if session:
                session["status"] = status
                return self.memory_provider.update_by_id(session_id, session, MemoryType.SHARED_MEMORY)
            return False
        except Exception as e:
            logger.error(f"Error updating session status: {e}")
            return False
    
    def is_root_agent(self, session_id: str, agent_id: str) -> bool:
        """Check if an agent is the root agent for a session."""
        try:
            session = self.memory_provider.retrieve_by_id(session_id, MemoryType.SHARED_MEMORY)
            return session and session.get("root_agent_id") == agent_id
        except Exception as e:
            logger.error(f"Error checking root agent status: {e}")
            return False
    
    def get_session_by_root_agent(self, root_agent_id: str) -> Optional[Dict[str, Any]]:
        """Get active shared session by root agent ID."""
        try:
            # This would need to be implemented in the memory provider
            # For now, we'll need to search through sessions
            return None
        except Exception as e:
            logger.error(f"Error getting session by root agent: {e}")
            return None 
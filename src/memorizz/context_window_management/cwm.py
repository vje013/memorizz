from typing import List
# from ..memagent import MemAgent
from ..memory_provider.memory_type import MemoryType

# Can take in an agent and then return a prompt that informs the agent on how to manage the context window
class CWM:
    # def __init__(self, agent: MemAgent):
    #     self.agent = agent
    
    @staticmethod
    def get_prompt_from_memory_types(memory_types: List[MemoryType]):
        prompt = "You are an AI Agent endowed with a powerful, multi-tiered memory augmentation system. Your mission is to use all available memory modalities to deliver consistent, accurate, and context-rich responses. The aim is to esure that through augmented memory, you become belivable, capable, and reliable."

        for memory_type in memory_types:
            prompt += CWM._generate_prompt_for_memory_type(memory_type)

        return prompt

    @staticmethod
    def _generate_prompt_for_memory_type(memory_type: MemoryType):
        prompt = ""
        
        if memory_type.value == MemoryType.CONVERSATION_MEMORY.value:
            # Construct a prompt that informs the agent on what the memory type means and how to use it
            prompt += f"\n\nMemory Type: {MemoryType.CONVERSATION_MEMORY.value}\n"
            prompt += f"Memory Type Description: This is a memory type that stores the conversation history between the agent and the user.\n"
            prompt += f"Memory Type Usage: Use this to provide continuity, avoid repeating yourself, and reference prior turns.\n"
        elif memory_type.value == MemoryType.WORKFLOW_MEMORY.value:
            # Construct a prompt that informs the agent on what the memory type means and how to use it
            prompt += f"\n\nMemory Type: {MemoryType.WORKFLOW_MEMORY.value}\n"
            prompt += f"Memory Type Description: This is a memory type that stores the workflow history between the agent and the user.\n"
            prompt += f"Memory Type Usage: Use this to provide continuity, avoid repeating yourself, and reference prior turns.\n"
        elif memory_type.value == MemoryType.SHARED_MEMORY.value:
            # Construct a prompt that informs the agent on what the memory type means and how to use it
            prompt += f"\n\nMemory Type: {MemoryType.SHARED_MEMORY.value}\n"
            prompt += f"Memory Type Description: This is a memory type that stores shared blackboard information for multi-agent coordination.\n"
            prompt += f"Memory Type Usage: Use this to coordinate with other agents, understand your role in the agent hierarchy, and access shared coordination activities and context.\n"
        
        return prompt

# Can take in an array of memory stores and then return a prompt that informs the agent on how to manage the context window
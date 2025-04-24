from ..memorizz.toolbox import Toolbox
from ..memorizz.llms.openai import OpenAI
from ..memorizz.persona import Persona
from typing import Optional, Union, List, Dict, Any
import json
from ..memorizz.memory_component import MemoryComponent, ConversationMemoryComponent
from datetime import datetime
import uuid
from ..memorizz.memory_provider import MemoryProvider
from ..memorizz.memory_component.memory_mode import MemoryMode
from ..memorizz.memory_provider.memory_type import MemoryType
import logging
from pydantic import BaseModel
from ..memorizz.toolbox.tool_schema import ToolSchemaType
from typing import Callable
logger = logging.getLogger(__name__)

class Role:
    User = "user"
    Assistant = "assistant"
    Developer = "developer"
    Tool = "tool"

class MemAgentModel(BaseModel):
    model: Optional[OpenAI] = None
    agent_id: Optional[str] = None
    tools: Optional[Union[List, Toolbox]] = None
    persona: Optional[Persona] = None
    instruction: Optional[str] = None
    memory_mode: Optional[str] = MemoryMode.Default
    max_steps: int = 20
    memory_ids: Optional[List[str]] = None
    tool_access: Optional[str] = "private"
    
    model_config = {
        "arbitrary_types_allowed": True  # Allow arbitrary types like Toolbox
    }


class MemAgent:
    def __init__(
        self,
        model: Optional[OpenAI] = None, # LLM to use
        tools: Optional[Union[List, Toolbox]] = None, # List of tools to use or toolbox
        persona: Optional[Persona] = None, # Persona of the agent
        instruction: Optional[str] = None, # Instruction of the agent
        memory_mode: Optional[str] = MemoryMode.Default, # Memory mode of the agent
        max_steps: int = 20, # Maximum steps of the agent
        memory_provider: Optional[MemoryProvider] = None, # Memory provider of the agent
        memory_ids: Optional[Union[str, List[str]]] = None, # Memory id(s) of the agent
        agent_id: Optional[str] = None, # Agent id of the agent
        tool_access: Optional[str] = "private" # Tool access of the agent
    ):
        # If the memory provider is not provided, then we use the default memory provider
        self.memory_provider = memory_provider or MemoryProvider()

        # Initialize the memory component based on the memory mode
        self.memory_component = MemoryComponent(memory_mode, self.memory_provider)

        # Initialize the model if not provided
        self.model = model or OpenAI(model="gpt-4.1")
        
        # If the memory provider is provided and the agent id is provided, then we load the memagent from the memory provider
        if memory_provider and agent_id:
            try:
                # Load the memagent from the memory provider
                loaded_agent = memory_provider.retrieve_memagent(agent_id)
                if loaded_agent:
                    print(f"Loaded agent: {loaded_agent}")
                    print(f"Loaded agent tools: {loaded_agent.tools}")
                    # Copy all the attributes from the loaded agent to self
                    for key, value in vars(loaded_agent).items():
                        setattr(self, key, value)
                    
                    # If the model is not provided, then we use the default model
                    if loaded_agent.model is None:
                        self.model = OpenAI(model="gpt-4.1")
                    return
                else:
                    print(f"No agent found with id {agent_id}, creating a new one")
            except Exception as e:
                print(f"Error loading agent from memory provider: {e}")
                print("Creating a new agent instead")

        # Initialize the memagent
        self.tools = tools
        self.persona = persona
        self.instruction = instruction or "You are a helpful assistant."
        self.memory_mode = memory_mode
        self.max_steps = max_steps
        self.user_input = ""
        self.tool_access = tool_access
        # Initialize memory_ids as a list, converting single string if needed
        if memory_ids is None:
            self.memory_ids = []
        elif isinstance(memory_ids, str):
            self.memory_ids = [memory_ids]
        else:
            self.memory_ids = memory_ids
            
        self.agent_id = agent_id

    def _generate_system_prompt(self):
        """
        Generate the system prompt for the agent.

        This method generates the system prompt for the agent based on the persona and instruction.
        If both are provided, the persona prompt is prepended to the instruction.
        If only the persona is provided, the persona prompt is used.
        If only the instruction is provided, the instruction is used.
        
        Returns:
            str: The system prompt for the agent.
        """

        # Generate the system prompt or message from the persona and instruction if provided
        if self.persona and self.instruction:

            # Generate the system prompt from the persona and instruction
            persona_prompt = self.persona.generate_system_prompt_input()

            return f"{persona_prompt}\n\n{self.instruction}"
        
        elif self.persona:
            return f"{self.persona.generate_system_prompt_input()}"
        else:
            return f"{self.instruction}"
        

    def _format_tool(tool_meta: Dict[str, Any]) -> Dict[str,Any]:
        """
        Format the tool.

        This method formats the tool.

        Parameters:
            tool_meta (Dict[str, Any]): The tool meta.

        Returns:
            Dict[str, Any]: The formatted tool.
        """

        # Initialize the properties and required parameters
        props, req = {}, []

        # Format the tool
        for p in tool_meta["parameters"]:
            # Convert 'float' type to 'number' for OpenAI API compatibility
            param_type = p.get("type", "string")
            if param_type == "float":
                param_type = "number"
                
            props[p["name"]] = {
                "type": param_type,
                "description": p.get("description", "")
            }
            if p.get("required", False):
                req.append(p["name"])

        # Return the formatted tool
        return {
            "type": "function",
            "name": tool_meta["name"],
            "description": tool_meta["description"],
            "parameters": {
                "type": "object",
                "properties": props,
                "required": req
            }
        }
        

    def _load_tools_from_toolbox(self, query: str):
        """
        Load the tools from the toolbox.

        This method loads the tools from the toolbox based on the user input.

        Parameters:
            query (str): The user input.

        Returns:
            List[Dict[str, Any]]: The list of tools.
        """

        # Load the tools from the toolbox
        tools = self.tools.get_most_similar_tools(query)
        formatted_tools = []
        
        for t in tools:
            formatted_tool = self._format_tool(t)
            # Preserve the tool_id for reference
            if "tool_id" in t:
                formatted_tool["tool_id"] = t["tool_id"]
            formatted_tools.append(formatted_tool)
            
        return formatted_tools

    def _load_tools_from_memagent(self) -> List[Dict[str, Any]]:
        """
        Load the tools from the memagent and format them
        for OpenAI function‐calling (flat schema with name/description/parameters).
        """
        if not self.tools:
            return []

        # Ensure each tool has the required 'type' field and properly formatted parameters
        if isinstance(self.tools, list):
            formatted_tools = []
            for tool in self.tools:
                # Create a properly formatted copy of the tool
                formatted_tool = {
                    "type": "function",
                    "name": tool["name"],
                    "description": tool["description"]
                }
                
                # Format parameters according to OpenAI's function calling format
                if "parameters" in tool and isinstance(tool["parameters"], list):
                    properties = {}
                    required = []
                    
                    for param in tool["parameters"]:
                        # Convert 'float' type to 'number' for OpenAI API compatibility
                        param_type = param.get("type", "string")
                        if param_type == "float":
                            param_type = "number"
                            
                        properties[param["name"]] = {
                            "type": param_type,
                            "description": param.get("description", "")
                        }
                        if param.get("required", False):
                            required.append(param["name"])
                    
                    formatted_tool["parameters"] = {
                        "type": "object",
                        "properties": properties,
                        "required": required
                    }
                else:
                    # If parameters is already in the correct format, keep it
                    formatted_tool["parameters"] = tool.get("parameters", {"type": "object", "properties": {}})
                
                # Preserve tool_id for reference
                if "tool_id" in tool:
                    formatted_tool["tool_id"] = tool["tool_id"]
                
                formatted_tools.append(formatted_tool)
            
            return formatted_tools

        return self.tools


    def run(self, query: str, memory_id: str = None, conversation_id: str = None) -> str:
        """
        Run the agent: load history & memory, call LLM with optional functions,
        execute any function_call, loop until final text answer.
        """

        # 1) Ensure memory_id
        if memory_id is None:
            if self.memory_ids and len(self.memory_ids) > 0:
                # Use the most recent memory_id if none specified
                memory_id = self.memory_ids[-1]
            else:
                # Create a new memory_id if none exist
                memory_id = str(uuid.uuid4())
        elif memory_id not in (self.memory_ids or []):
            # Create a new memory_id if specified one doesn't exist
            memory_id = str(uuid.uuid4())

        # persist to agent if needed
        if self.agent_id and memory_id not in (self.memory_ids or []):
            self.memory_ids.append(memory_id)
            if hasattr(self.memory_provider, "update_memagent_memory_ids"):
                self.memory_provider.update_memagent_memory_ids(self.agent_id, self.memory_ids)

        # 2) Ensure conversation_id
        if conversation_id is None:
            conversation_id = str(uuid.uuid4())

        # 3) Build system + user prompt
        system_prompt = self._generate_system_prompt()
        augmented_query = f"This is the query to be answered or key objective to be achieved: {query}"

        # 4) Append past conversation history
        for conv in self.load_conversation_history(memory_id):
            augmented_query += (
                f"\n\nPrevious: {conv['role']}: {conv['content']}. "
                "Use this context to answer coherently."
            )

        # 5) Append relevant memory components
        if self.memory_mode in (MemoryMode.Conversational, MemoryMode.Default):
            for mem in self._load_relevant_memory_components(
                query, MemoryType.CONVERSATION_MEMORY, memory_id, limit=5
            ):
                augmented_query += (
                    f"\n\nRelevant memory: {mem['role']}: {mem['content']}. "
                    "Use this to inform your answer."
                )

        # 6) Record the user's turn in memory
        if self.memory_mode in (MemoryMode.Conversational, MemoryMode.Default):
            self._generate_conversational_memory_component({
                "role": Role.User,
                "content": query,
                "timestamp": datetime.now().isoformat(),
                "conversation_id": conversation_id,
                "memory_id": memory_id,
            })

        # 7) Seed the chat
        messages = [
            {"role": "system",    "content": system_prompt},
            {"role": "user",      "content": augmented_query},
        ]
        tool_choice = "auto"

        # 8) Main loop
        for _ in range(self.max_steps):
            # a) Build function schema list
            if self.tools:
                if isinstance(self.tools, Toolbox) and self.tool_access != "global":
                    tool_metas = self._load_tools_from_memagent()
                elif isinstance(self.tools, Toolbox):
                    tool_metas = self._load_tools_from_toolbox(query)
                else:
                    tool_metas = self._load_tools_from_memagent()
                if not tool_metas:
                    tool_choice = "none"
            else:
                tool_metas, tool_choice = [], "none"

            # b) Call the Responses API
            response = self.model.client.responses.create(
                model="gpt-4.1",
                input=messages,
                tools=tool_metas,
                tool_choice=tool_choice
            )

            # c) See if model called a function
            tool_calls = [
                o for o in response.output
                if getattr(o, "type", None) == "function_call"
            ]

            if tool_calls:
                call = tool_calls[0]
                name = call.name
                try:
                    args = json.loads(call.arguments)
                    messages.append({
                        "type":      "function_call",
                        "call_id":   call.call_id,
                        "name":      call.name,
                        "arguments": call.arguments,
                    })
                except Exception:
                    args = {}

                # d) Lookup the Python function backing this call
                fn = None
                entry = None
                if isinstance(self.tools, Toolbox):
                    for meta in (self._load_tools_from_toolbox(query) if self.tool_access=="global"
                                else self._load_tools_from_memagent()):
                        if meta["name"] == name:
                            entry = meta
                            fn = getattr(self, "_tool_functions", {}).get(meta.get("tool_id"))
                            break
                elif isinstance(self.tools, list):
                    formatted = self._load_tools_from_memagent()
                    for t in formatted:
                        if t["name"] == name:
                            entry = t
                            for orig in self.tools:
                                if orig["name"] == name:
                                    fn = getattr(self, "_tool_functions", {}).get(orig.get("tool_id"))
                                    break
                            break

                if not entry:
                    result = f"Error: Tool '{name}' not found in available tools."
                elif not callable(fn):
                    logger.warning(f"Tool '{name}' found but function is not callable. Tool ID: {entry.get('tool_id')}")
                    result = f"Sorry, the tool '{name}' is currently unavailable. It exists in the system but its implementation function is not properly registered."
                else:
                    # e) Execute and append the function's result
                    result = fn(**args)

                # Append the result (either actual function output or error message)
                messages.append({
                    "type":    "function_call_output",
                    "call_id": call.call_id,
                    "output":  str(result),  # Convert result to string to ensure compatibility
                })

                # f) Record into memory
                # TODO: This is an area to store workflow memory (e.g. tool calls, tool call outputs, etc.)
                # Commenting out for now as we don't have a workflow memory component yet
                # self._generate_conversational_memory_component({
                #     "role": Role.Tool,
                #     "tool_call_id": call.call_id,
                #     "tool_id": entry.get("tool_id"),
                #     "name": name,
                #     "content": result,
                #     "memory_id": memory_id,
                #     "conversation_id": conversation_id,
                #     "timestamp": datetime.now().isoformat(),
                # })

                # g) Loop back so model ingests the tool output
                continue

            # h) No function calls → final answer
            if response.output_text:
                # Record into memory
                if self.memory_mode in (MemoryMode.Conversational, MemoryMode.General):
                    self._generate_conversational_memory_component({
                        "role": Role.Assistant,
                        "content": response.output_text,
                        "timestamp": datetime.now().isoformat(),
                        "conversation_id": conversation_id,
                        "memory_id": memory_id,
                    })

                # Append as assistant turn and return
                messages.append({
                    "role":    "assistant",
                    "content": response.output_text
                })
                return response.output_text

        # 9) If we never returned…
        raise RuntimeError("Max steps exceeded without reaching a final answer.")


    def load_conversation_history(self, memory_id: str = None):
        """
        Load the conversation history.

        This method loads the conversation history based on the memory id.

        Parameters:
            memory_id (str): The memory id.

        Returns:
            List[ConversationMemoryComponent]: The conversation history.
        """

        # If the memory id is not provided and we have memory_ids, use the most recent one
        if memory_id is None and self.memory_ids:
            memory_id = self.memory_ids[-1]

        return self.memory_component.retrieve_memory_components_by_memory_id(memory_id, MemoryType.CONVERSATION_MEMORY)

    def _load_relevant_memory_components(self, query: str, memory_type: MemoryType, memory_id: str = None, limit: int = 5):
        """
        Load the relevant memory components.

        This method loads the relevant memory components based on the query.

        Parameters:
            query (str): The user input.
            memory_id (str): The memory id.
            limit (int): The limit of the memory components to return.

        Returns:
            List[ConversationMemoryComponent]: The conversation history.
        """

        # If the memory id is not provided and we have memory_ids, use the most recent one
        if memory_id is None and self.memory_ids:
            memory_id = self.memory_ids[-1]

        # Load the relevant memory components from the memory provider
        relevant_memory_components = self.memory_component.retrieve_memory_components_by_query(query, memory_id=memory_id, memory_type=memory_type, limit=limit)

        # Return the relevant memory components
        return relevant_memory_components


    def _generate_conversational_memory_component(self, content: dict) -> ConversationMemoryComponent:
        """
        Generate the conversational memory component.

        This method generates the conversational memory component based on the content.

        Parameters:
            content (dict): The content of the memory component.

        Returns:
            str: The conversational memory component.
        """

        # Generate the conversational memory component
        memory_component = self.memory_component.generate_memory_component(content)
        return memory_component
    
    def save(self):
        """
        Store the memagent in the memory provider.

        This method stores the memagent in the memory provider.
        """
        if self.agent_id is None:
            self.agent_id = str(uuid.uuid4())

        # Create a new MemAgentModel with the current object's attributes
        memagent_to_save = MemAgentModel(
            instruction=self.instruction,
            memory_mode=self.memory_mode,
            max_steps=self.max_steps,
            memory_ids=self.memory_ids,
            agent_id=self.agent_id,
            persona=self.persona,
            tools=self.tools
        )

        # Store the memagent in the memory provider
        saved_memagent = self.memory_provider.store_memagent(memagent_to_save)

        # Create a new MemAgentModel with the current object's attributes
        # and handle memory_mode conversion properly
        memory_mode_value = self.memory_mode
        if isinstance(memory_mode_value, str):
            # Map the string to the corresponding MemoryMode attribute
            if memory_mode_value == "general":
                memory_mode_value = MemoryMode.General
            elif memory_mode_value == "conversational":
                memory_mode_value = MemoryMode.Conversational
            elif memory_mode_value == "task":
                memory_mode_value = MemoryMode.Task
            elif memory_mode_value == "workflow":
                memory_mode_value = MemoryMode.Workflow
            else:
                # Default to General if string doesn't match
                memory_mode_value = MemoryMode.General

        # Create a new MemAgent object with our current attributes
        saved_memagent = MemAgentModel(
            instruction=saved_memagent["instruction"],
            memory_mode=memory_mode_value,
            max_steps=saved_memagent["max_steps"],
            memory_ids=saved_memagent["memory_ids"],
            agent_id=saved_memagent["agent_id"],
        )

        # Log the saved memagent
        logger.info(f"Memagent {self.agent_id} saved in the memory provider")
        # Log the details and attributes of the saved memagent
        # Show the logs as a json object
        logger.info(saved_memagent.model_dump_json(indent=4))

        return saved_memagent


    def update(self, 
               instruction: Optional[str] = None,
               memory_mode: Optional[str] = None,
               max_steps: Optional[int] = None,
               memory_ids: Optional[List[str]] = None,
               persona: Optional[Persona] = None,
               tools: Optional[List[Dict[str, Any]]] = None):
        """
        Update the memagent in the memory provider.

        This method updates various parts of the memagent in the memory provider.

        Parameters:
            instruction (str): The instruction of the memagent.
            memory_mode (str): The memory mode of the memagent.
            max_steps (int): The maximum steps of the memagent.
            memory_ids (List[str]): The memory ids of the memagent.
            persona (Persona): The persona of the memagent.
            tools (List[Dict[str, Any]]): The tools of the memagent.

        Returns:
            MemAgent: The updated memagent.
        """

        # Update the memagent in the memory provider
        if tools:
            self.tools = tools

        if persona:
            self.persona = persona

        if instruction:
            self.instruction = instruction

        if memory_mode:
            self.memory_mode = memory_mode

        if max_steps:
            self.max_steps = max_steps

        if memory_ids:
            self.memory_ids = memory_ids

        memagent_to_update = MemAgentModel(
            instruction=self.instruction,
            memory_mode=self.memory_mode,
            max_steps=self.max_steps,
            memory_ids=self.memory_ids,
            agent_id=self.agent_id,
            persona=self.persona,
            tools=self.tools
        )

        # Update the memagent in the memory provider
        updated_memagent_dict = self.memory_provider.update_memagent(memagent_to_update)
        logger.info(f"Memagent {self.agent_id} updated in the memory provider")

        return self

    @classmethod
    def load(cls,
             agent_id: str,
             memory_provider: Optional[MemoryProvider] = None,
             **overrides
             ):
        """
        Retrieve the memagent from the memory provider.

        This method retrieves the memagent from the memory provider.

        Parameters:
            agent_id (str): The agent id.

        Returns:
            MemAgent: The memagent.
        """
        logger.info(f"Loading MemAgent with agent id {agent_id}...")

        # If the memory provider is not provided, then we use the default memory provider
        provider = memory_provider or MemoryProvider()

        # Retrieve the memagent from the memory provider
        memagent = provider.retrieve_memagent(agent_id)

        if not memagent:
            raise ValueError(f"MemAgent with agent id {agent_id} not found in the memory provider")
        
        # Convert memory_mode string to MemoryMode class attribute if needed
        if memagent and isinstance(memagent.memory_mode, str):
            if memagent.memory_mode == "general":
                memagent.memory_mode = MemoryMode.General
            elif memagent.memory_mode == "conversational":
                memagent.memory_mode = MemoryMode.Conversational
            elif memagent.memory_mode == "task":
                memagent.memory_mode = MemoryMode.Task
            elif memagent.memory_mode == "workflow":
                memagent.memory_mode = MemoryMode.Workflow
            else:
                # Default to General if string doesn't match
                memagent.memory_mode = MemoryMode.General

        # Instantiate with saved parameters (and allow callers to override e.g. model)
        memagent = cls(
            model=overrides.get("model", getattr(memagent, "model", None)),
            tools=overrides.get("tools", getattr(memagent, "tools", None)),
            persona=overrides.get("persona", getattr(memagent, "persona", None)),
            instruction=overrides.get("instruction", getattr(memagent, "instruction", None)),
            memory_mode=overrides.get("memory_mode", getattr(memagent, "memory_mode", None)),
            max_steps=overrides.get("max_steps", getattr(memagent, "max_steps", None)),
            memory_ids=overrides.get("memory_ids", getattr(memagent, "memory_ids", [])),
            agent_id=agent_id,
            memory_provider=provider
        )

        # Show the logs as a json object
        logger.info(f"MemAgent loaded with agent_id: {agent_id}")

        return memagent
    
    @classmethod
    def _do_delete(cls, 
                   agent_id: str, 
                   cascade: bool, 
                   memory_provider: MemoryProvider):
        """
        Delete the memagent from the memory provider.

        This method deletes the memagent from the memory provider.

        Parameters:
            agent_id (str): The agent id.
            cascade (bool): Whether to cascade the deletion of the memagent. This deletes all the memory components associated with the memagent by deleting the memory_ids and their corresponding memory store in the memory provider.
            memory_provider (MemoryProvider): The memory provider to use.

        Returns:
            bool: True if the memagent was deleted successfully, False otherwise.
        """

        try:
            result = memory_provider.delete_memagent(agent_id, cascade)
            logger.info(f"MemAgent {agent_id} deleted from the memory provider")
            return result
        except Exception as e:
            logger.error(f"Error deleting MemAgent {agent_id} from the memory provider: {e}")
            return False
    
    @classmethod
    def delete(cls, 
               agent_id: str,
               cascade: bool = False,
               memory_provider: Optional[MemoryProvider] = None
    ):
        """
        Delete the memagent from the memory provider.

        This method deletes the memagent from the memory provider.

        Parameters:
            agent_id (str): The agent id.
            memory_provider (Optional[MemoryProvider]): The memory provider to use.
            cascade (bool): Whether to cascade the deletion of the memagent. This deletes all the memory components associated with the memagent by deleting the memory_ids and their corresponding memory store in the memory provider.

        Returns:
            bool: True if the memagent was deleted successfully, False otherwise.
        """
        
        # If the memory provider is not provided, then use the default memory provider
        provider = memory_provider or MemoryProvider()
        
        try:
            result = cls._do_delete(agent_id, cascade, provider)
            logger.info(f"MemAgent {agent_id} deleted from the memory provider")
            return result
        except Exception as e:
            logger.error(f"Error deleting MemAgent {agent_id} from the memory provider: {e}")
            return False

    def delete(self, cascade: bool = False): 
        """
        Delete the memagent from the memory provider.

        This method deletes the memagent from the memory provider.

        Parameters:
            cascade (bool): Whether to cascade the deletion of the memagent. This deletes all the memory components associated with the memagent by deleting the memory_ids and their corresponding memory store in the memory provider.

        Returns:
            bool: True if the memagent was deleted successfully, False otherwise.
        """

        if self.agent_id is None:
            raise ValueError("MemAgent agent_id is not set. Please set the agent_id before deleting the memagent.")

        return type(self)._do_delete(self.agent_id, cascade, self.memory_provider)
    
    # Memory ID Management
    
    def update_memory_ids(self, memory_ids: List[str]):
        """
        Update the memory_ids of the memagent.

        This method updates the memory_ids of the memagent.

        Parameters:
            memory_ids (List[str]): The memory_ids to update.

        Returns:
            bool: True if the memory_ids were updated successfully, False otherwise.
        """
        self.memory_ids = memory_ids

        if hasattr(self.memory_provider, 'update_memagent_memory_ids'):
            # Update the memory_ids of the memagent in the memory provider
            self.memory_provider.update_memagent_memory_ids(self.agent_id, self.memory_ids)
        else:
            raise ValueError("Memory provider does not have the update_memagent_memory_ids method.")
        
        return True
    
    def delete_memory_ids(self, memory_ids: List[str]):
        """
        Delete the memory_ids of the memagent.

        This method deletes the memory_ids of the memagent.

        Parameters:
            memory_ids (List[str]): The memory_ids to delete.

        Returns:
            bool: True if the memory_ids were deleted successfully, False otherwise.
        """
        self.memory_ids = [memory_id for memory_id in self.memory_ids if memory_id not in memory_ids]

        if hasattr(self.memory_provider, 'delete_memagent_memory_ids'):
            # Delete the memory_ids of the memagent in the memory provider
            self.memory_provider.delete_memagent_memory_ids(self.agent_id)
        else:
            raise ValueError("Memory provider does not have the delete_memagent_memory_ids method.")
        
        return True

    # Persona Management
    
    def set_persona(self, persona: Persona, save: bool = True):
        """
        Set the persona of the memagent.

        Parameters:
            persona (Persona): The persona to set.
            save (bool): Whether to save the memagent after setting the persona.

        Returns:
            bool: True if the persona was set successfully, False otherwise.
        """
        self.persona = persona
        if save:
            self.update()

    def set_persona_from_memory_provider(self, persona_id: str, save: bool = True):
        """
        Set the persona of the memagent from the persona memory store within the memory provider.

        Parameters:
            persona_id (str): The persona id.
            save (bool): Whether to save the memagent after setting the persona.

        Returns:
            bool: True if the persona was set successfully, False otherwise.
        """
        self.persona = self.memory_provider.retrieve_persona(persona_id)

        if self.persona:
            if save:
                self.update()
            return True
        else:
            raise ValueError("Persona is not set. Please set the persona before setting it from the memory provider.")

    def export_persona(self):
        """
        Export the persona of the memagent to the persona memory store within the memory provider.

        Returns:
            bool: True if the persona was exported successfully, False otherwise.
        """
        if self.persona:
            self.persona.store_persona(self.memory_provider)
        else:
            raise ValueError("Persona is not set. Please set the persona before exporting it.")
        
        return True
    
    def delete_persona(self, save: bool = True):
        """
        Delete the persona of the memagent.

        Parameters:
            save (bool): Whether to save the memagent after deleting the persona.

        Returns:
            bool: True if the persona was deleted successfully, False otherwise.
        """

        if self.persona:
            self.persona = None
            if save:
                self.update()
        else:
            raise ValueError("Persona is not set. Please set the persona before deleting it.")
        
        return True
    
    # Toolbox Management Functions

    def add_tool(self, tool_id: str = None, toolbox: Toolbox = None) -> bool:
        """
        Add or update a single tool to this agent from a Toolbox.

        You must supply either:
          • tool_id: the UUID of an existing toolbox entry, or
          • toolbox: a Toolbox instance (for batch import).

        If the tool is already in self.tools, its entry will be overwritten
        with the latest metadata & function reference. Otherwise it will be appended.

        Parameters:
            tool_id (str):    The tool id in the memory-provider's toolbox.
            toolbox (Toolbox): The Toolbox instance to pull the Python function from.

        Returns:
            bool: True if the tool was added or updated successfully.
        """
        def _build_entry(meta: dict, python_fn: Callable) -> dict:
            """
            Construct the flat OpenAI‐style schema entry and
            register the python_fn in our internal lookup.
            """
            entry = {
                "tool_id":    meta["tool_id"],
                "name":       meta["function"]["name"],
                "description":meta["function"]["description"],
                "parameters": meta["function"]["parameters"],
                # preserve strict‐mode flag if present:
                **({"strict": meta.get("strict", True)}),
            }
            # keep a private map of tool_id → python function,
            # but don't include it when serializing the agent's 'tools' list
            self._tool_functions = getattr(self, "_tool_functions", {})
            self._tool_functions[meta["tool_id"]] = python_fn

            return entry

        # --- single‐tool branch ---
        if tool_id:
            # 1) fetch the metadata from memory-provider
            meta = self.memory_provider.retrieve_by_id(tool_id, MemoryType.TOOLBOX)
            if not meta:
                raise ValueError(f"No such tool in the toolbox: {tool_id}")

            # 2) resolve the Python callable from the provided Toolbox
            if not isinstance(toolbox, Toolbox):
                raise ValueError("Need a Toolbox instance to resolve the Python callable")

            python_fn = toolbox._tools.get(tool_id)
            if not callable(python_fn):
                # fallback: perhaps the stored metadata itself packs a .function field?
                tb_meta = next(
                    (m for m in toolbox.list_tools() if m.get("tool_id") == tool_id),
                    None
                )
                if tb_meta and callable(tb_meta.get("function")):
                    python_fn = tb_meta["function"]

            if not callable(python_fn):
                # Silently skip tools without functions instead of raising error
                return False

            # 3) build the new entry
            new_entry = _build_entry(meta, python_fn)

            # 4) if already present in self.tools, overwrite in-place
            existing_idx = None
            for idx, t in enumerate(self.tools or []):
                if t.get("tool_id") == tool_id:
                    existing_idx = idx
                    break

            if existing_idx is not None:
                # replace the old entry
                self.tools[existing_idx] = new_entry
            else:
                # append a brand-new entry
                self.tools = self.tools or []
                self.tools.append(new_entry)

            # 5) persist the agent's updated tools list
            self.update()
            return True

        # --- batch‐import branch ---
        if toolbox:
            if not isinstance(toolbox, Toolbox):
                raise TypeError(f"Expected a Toolbox, got {type(toolbox)}")
            success = False
            for meta in toolbox.list_tools():
                tid = meta.get("tool_id")
                if tid:
                    # recursively leverage the single‐tool logic
                    if self.add_tool(tool_id=tid, toolbox=toolbox):
                        success = True
            return success

        # --- neither provided: error ---
        raise ValueError("Must supply either a tool_id or a Toolbox instance.")

    def export_tools(self):
        """
        Export the tools of the memagent to the toolbox within the memory provider.
        Only tools with a tool_id are exported.

        Returns:
            bool: True if the tools were exported successfully, False otherwise.
        """
        # If tools is set and not empty, export the tools to the toolbox within the memory provider
        if self.tools and len(self.tools) > 0:
            for tool in self.tools:
                if tool.get("tool_id") is None:
                    # Export the tool to the toolbox within the memory provider
                    new_tool_id = self.memory_provider.store(tool, MemoryType.TOOLBOX)
                    self.tools.append({"tool_id": new_tool_id, **tool})
            return True
        else:
            raise ValueError("No tools to export. Please add a tool to the memagent before exporting it.")
    
    def refresh_tools(self, tool_id: str):
        """
        Refresh the tools of the memagent from the toolbox within the memory provider.
        This method refreshes the tools of the memagent from the toolbox within the memory provider.

        Parameters:
            tool_id (str): The tool id.

        Returns:
            bool: True if the tools were refreshed successfully, False otherwise.
        """
        
        # TODO In this method there is a retriveal of the tool from the memory provider (1 within this method and 2 in the add_tool method), this can be optimized by retrieving the tool from the memory provider once and then adding it to the memagent. 
        if tool_id:
            # Retrieve the tool from the toolbox within the memory provider
            tool_meta = self.memory_provider.retrieve_by_id(tool_id, MemoryType.TOOLBOX)

            # If the tool is not found, raise an error
            if not tool_meta:
                raise ValueError(f"No such tool: {tool_id} in the toolbox within memory provider")
            
            # If the tool is found, add it to the memagent
            self.add_tool(tool_id=tool_id)
            return True
        else:
            raise ValueError("Tool id is not set. Please set the tool id before refreshing the tool.")


    def delete_tool(self, tool_id: str):
        """
        Delete a tool from the memagent.

        This method deletes a tool from the memagent.

        Parameters:
            tool_id (str): The tool id.

        Returns:
            bool: True if the tool was deleted successfully, False otherwise.
        """
        if self.tools:
            self.tools = [tool for tool in self.tools if tool["tool_id"] != tool_id]
            self.save()
        else:
            raise ValueError("No tools to delete. Please add a tool to the memagent before deleting it.")

    def __str__(self):
        return f"MemAgent(agent_id={self.agent_id}, memory_ids={self.memory_ids}, memory_mode={self.memory_mode}, max_steps={self.max_steps}, instruction={self.instruction}, model={self.model}, tools={self.tools}, persona={self.persona})"
    
    def __repr__(self):
        return self.__str__()

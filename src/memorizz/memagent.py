from .toolbox import Toolbox
from .llms.openai import OpenAI
from .persona import Persona
from typing import Optional, Union, List, Dict, Any
import json
from .memory_component import MemoryComponent, ConversationMemoryComponent
from datetime import datetime
import uuid
from .memory_provider import MemoryProvider
from .memory_component.memory_mode import MemoryMode
from .memory_provider.memory_type import MemoryType
import logging
from pydantic import BaseModel
from .toolbox.tool_schema import ToolSchemaType
from typing import Callable
from .workflow.workflow import Workflow, WorkflowOutcome
from .context_window_management.cwm import CWM
from .long_term_memory import KnowledgeBase
from bson import ObjectId

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
    long_term_memory_ids: Optional[List[str]] = None
    delegates: Optional[List[str]] = None  # Store delegate agent IDs
    
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
        tool_access: Optional[str] = "private", # Tool access of the agent
        delegates: Optional[List['MemAgent']] = None # Delegate agents for multi-agent mode
    ):
        # If the memory provider is not provided, then we use the default memory provider
        self.memory_provider = memory_provider or MemoryProvider()

        # Initialize the memory component based on the memory mode
        self.memory_component = MemoryComponent(memory_mode, self.memory_provider)

        # Initialize the model if not provided
        # TODO: Remove this once we have a way to pass the model to the agent
        # TODO: Below needs to be configured globally and not per agent
        self.model = OpenAI(model="gpt-4.1")
        
        # Multi-agent setup
        self.delegates = delegates or []
        self.is_multi_agent_mode = len(self.delegates) > 0
        self._multi_agent_orchestrator = None
        
        # If the memory provider is provided and the agent id is provided, then we load the memagent from the memory provider
        if memory_provider and agent_id:
            try:
                # Load the memagent from the memory provider
                loaded_agent = memory_provider.retrieve_memagent(agent_id)
                if loaded_agent:
                    # Copy all the attributes from the loaded agent to self
                    for key, value in vars(loaded_agent).items():
                        setattr(self, key, value)
                    
                    # If the model is not provided, then we use the default model
                    if loaded_agent.model is None:
                        self.model = OpenAI(model="gpt-4.1")
                    
                    # Load delegate agents if they exist
                    if hasattr(loaded_agent, 'delegates') and loaded_agent.delegates:
                        self.delegates = []
                        for delegate_id in loaded_agent.delegates:
                            try:
                                delegate_agent = MemAgent.load(delegate_id, memory_provider)
                                self.delegates.append(delegate_agent)
                            except Exception as e:
                                logger.warning(f"Could not load delegate agent {delegate_id}: {e}")
                        self.is_multi_agent_mode = len(self.delegates) > 0
                    
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

        # If tools is a Toolbox, properly initialize it using the same logic as add_tool
        if isinstance(tools, Toolbox):
            self._initialize_tools_from_toolbox(tools)

    def _initialize_multi_agent_orchestrator(self):
        """Initialize the multi-agent orchestrator if in multi-agent mode."""
        if self.is_multi_agent_mode and not self._multi_agent_orchestrator:
            # Import here to avoid circular imports
            from .multi_agent_orchestrator import MultiAgentOrchestrator
            self._multi_agent_orchestrator = MultiAgentOrchestrator(self, self.delegates)

    def _check_for_shared_memory_context(self) -> str:
        """Check if this agent is part of a shared memory session and return context."""
        try:
            # Import here to avoid circular imports
            from .shared_memory import SharedMemory
            
            shared_memory = SharedMemory(self.memory_provider)
            
            # TODO This would need to be implemented in the memory provider
            # For now, we'll check if there's an active session where this agent is the root
            # session = shared_memory.get_session_by_root_agent(self.agent_id)
            # if session:
            #     return shared_memory.get_blackboard_entries(session["session_id"])
            
            return ""
        except Exception as e:
            logger.error(f"Error checking shared memory context: {e}")
            return ""

    def _initialize_tools_from_toolbox(self, toolbox: Toolbox):
        """
        Initialize tools from a Toolbox using the same logic as add_tool.
        This ensures proper function reference management and tool metadata handling.
        
        Parameters:
            toolbox (Toolbox): The Toolbox instance to initialize tools from.
        """
        if not isinstance(toolbox, Toolbox):
            raise TypeError(f"Expected a Toolbox, got {type(toolbox)}")
        
        # Convert to list format for agent use
        self.tools = []
        
        for meta in toolbox.list_tools():
            # Use _id for function lookup
            tid = str(meta.get("_id"))
            if tid:
                # Resolve the Python callable from the provided Toolbox
                python_fn = toolbox._tools.get(tid)
                if not callable(python_fn):
                    # fallback: perhaps the stored metadata itself packs a .function field?
                    if meta and callable(meta.get("function")):
                        python_fn = meta["function"]

                if callable(python_fn):
                    # build the new entry
                    new_entry = self._build_entry(meta, python_fn)
                    self.tools.append(new_entry)
                else:
                    # Silently skip tools without functions
                    logger.warning(f"Skipping tool with _id {tid} - no callable function found")
                    continue

    def _build_entry(self, meta: dict, python_fn: Callable) -> dict:
        """
        Construct the flat OpenAI‐style schema entry and
        register the python_fn in our internal lookup.
        
        Parameters:
            meta (dict): Tool metadata containing function information
            python_fn (Callable): The Python function to register
            
        Returns:
            dict: Formatted tool entry for OpenAI API
        """
        entry = {
            "_id":        meta["_id"],  # Use _id as primary identifier
            "name":       meta["function"]["name"],
            "description":meta["function"]["description"],
            "parameters": meta["function"]["parameters"],
            # preserve strict‐mode flag if present:
            **({"strict": meta.get("strict", True)}),
        }
        # keep a private map of _id → python function,
        # but don't include it when serializing the agent's 'tools' list
        self._tool_functions = getattr(self, "_tool_functions", {})
        self._tool_functions[str(meta["_id"])] = python_fn

        return entry

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
        

    @staticmethod
    def _format_tool(tool_meta: Dict[str, Any]) -> Dict[str,Any]:
        """
        Format the tool.

        This method formats the tool.

        Parameters:
            tool_meta (Dict[str, Any]): The tool meta.

        Returns:
            Dict[str, Any]: The formatted tool.
        """

        # Handle different tool metadata structures
        # Case 1: Tool has proper 'function' metadata structure
        if "function" in tool_meta and isinstance(tool_meta["function"], dict):
            function_data = tool_meta["function"]
            name = function_data.get("name", "unknown_tool")
            description = function_data.get("description", "No description available")
            parameters = function_data.get("parameters", [])
        # Case 2: Tool has flat structure (name, description, parameters at top level)
        elif "description" in tool_meta:
            name = tool_meta.get("name", "unknown_tool")
            description = tool_meta.get("description", "No description available")
            parameters = tool_meta.get("parameters", [])
        # Case 3: Tool has corrupted structure with raw function - skip it
        elif "function" in tool_meta and callable(tool_meta["function"]):
            logger.warning(f"Skipping tool with _id {tool_meta.get('_id')} - contains raw function instead of metadata")
            return None
        else:
            # Fallback for unknown structure
            logger.warning(f"Unknown tool structure for tool with _id {tool_meta.get('_id')}, using fallback")
            name = "unknown_tool"
            description = "Tool metadata corrupted"
            parameters = []

        # Initialize the properties and required parameters
        props, req = {}, []

        # Format the tool parameters
        if isinstance(parameters, list):
            for p in parameters:
                if not isinstance(p, dict):
                    continue
                    
                # Normalize the parameter type for OpenAI API compatibility
                param_type = p.get("type", "string")
                
                # Clean up type string - remove any extra text like "(required)"
                if isinstance(param_type, str):
                    param_type = param_type.lower().strip()
                    # Remove any parenthetical content
                    if "(" in param_type:
                        param_type = param_type.split("(")[0].strip()
                    
                    # Normalize numeric types to 'number'
                    if param_type in ["float", "decimal", "double", "numeric", "number"]:
                        param_type = "number"
                    elif param_type in ["int", "integer"]:
                        param_type = "integer"
                    elif param_type in ["bool", "boolean"]:
                        param_type = "boolean"
                    elif param_type in ["str", "text"]:
                        param_type = "string"
                    # Default to string if unrecognized
                    elif param_type not in ["string", "number", "integer", "boolean", "array", "object"]:
                        param_type = "string"
                    
                props[p["name"]] = {
                    "type": param_type,
                    "description": p.get("description", "")
                }
                if p.get("required", False):
                    req.append(p["name"])

        # Return the formatted tool
        return {
            "type": "function",
            "name": name,
            "description": description,
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
            # Skip tools that couldn't be formatted (corrupted metadata)
            if formatted_tool is None:
                continue
            # Preserve the _id for reference (convert ObjectId to string for JSON serialization)
            if "_id" in t:
                formatted_tool["_id"] = str(t["_id"])
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
                # Handle different tool metadata structures
                # Case 1: Tool has proper 'function' metadata structure
                if "function" in tool and isinstance(tool["function"], dict):
                    function_data = tool["function"]
                    name = function_data.get("name", "unknown_tool")
                    description = function_data.get("description", "No description available")
                    parameters = function_data.get("parameters", [])
                # Case 2: Tool has flat structure (name, description, parameters at top level)
                elif "name" in tool:
                    name = tool.get("name", "unknown_tool")
                    description = tool.get("description", "No description available")
                    parameters = tool.get("parameters", [])
                # Case 3: Tool has corrupted structure - skip it
                else:
                    logger.warning(f"Skipping tool with _id {tool.get('_id')} - missing name and function structure")
                    continue

                # Create a properly formatted copy of the tool
                formatted_tool = {
                    "type": "function",
                    "name": name,
                    "description": description
                }
                
                # Format parameters according to OpenAI's function calling format
                if isinstance(parameters, list) and len(parameters) > 0:
                    properties = {}
                    required = []
                    
                    for param in parameters:
                        if not isinstance(param, dict):
                            continue
                            
                        param_name = param.get("name")
                        if not param_name:
                            continue
                            
                        # Normalize the parameter type for OpenAI API compatibility
                        param_type = param.get("type", "string")
                        
                        # Clean up type string - remove any extra text like "(required)"
                        if isinstance(param_type, str):
                            param_type = param_type.lower().strip()
                            # Remove any parenthetical content
                            if "(" in param_type:
                                param_type = param_type.split("(")[0].strip()
                            
                            # Normalize numeric types to 'number'
                            if param_type in ["float", "decimal", "double", "numeric", "number"]:
                                param_type = "number"
                            elif param_type in ["int", "integer"]:
                                param_type = "integer"
                            elif param_type in ["bool", "boolean"]:
                                param_type = "boolean"
                            elif param_type in ["str", "text"]:
                                param_type = "string"
                            # Default to string if unrecognized
                            elif param_type not in ["string", "number", "integer", "boolean", "array", "object"]:
                                param_type = "string"
                            
                        properties[param_name] = {
                            "type": param_type,
                            "description": param.get("description", "")
                        }
                        if param.get("required", False):
                            required.append(param_name)
                    
                    formatted_tool["parameters"] = {
                        "type": "object",
                        "properties": properties,
                        "required": required
                    }
                else:
                    # If no parameters or parameters is already in the correct format, provide empty schema
                    formatted_tool["parameters"] = {
                        "type": "object", 
                        "properties": {},
                        "required": []
                    }
                
                # Preserve _id for reference (convert ObjectId to string for JSON serialization)
                if "_id" in tool:
                    formatted_tool["_id"] = str(tool["_id"])
                
                formatted_tools.append(formatted_tool)
            
            return formatted_tools

        return self.tools


    def run(self, query: str, memory_id: str = None, conversation_id: str = None) -> str:
        """
        Run the agent: load history & memory, call LLM with optional functions,
        execute any function_call, loop until final text answer.
        
        If in multi-agent mode, orchestrate task decomposition and delegation.
        """
        
        # Check if we're in multi-agent mode
        if self.is_multi_agent_mode:
            self._initialize_multi_agent_orchestrator()
            return self._multi_agent_orchestrator.execute_multi_agent_workflow(
                query, memory_id, conversation_id
            )

        # TODO: Remove this once we have a way to pass the model to the agent
        # TODO: Below needs to be configured globally and not per agent
        self.model = OpenAI(model="gpt-4.1")

        # 1) Ensure memory_id
        if memory_id is None:
            if self.memory_ids and len(self.memory_ids) > 0:
                # Use the most recent memory_id if none specified
                memory_id = self.memory_ids[-1]
            else:
                # Create a new memory_id if none exist using MongoDB ObjectId
                memory_id = str(ObjectId())
        elif memory_id not in (self.memory_ids or []):
            # Create a new memory_id if specified one doesn't exist using MongoDB ObjectId
            memory_id = str(ObjectId())

        # persist to agent if needed
        if self.agent_id and memory_id not in (self.memory_ids or []):
            self.memory_ids.append(memory_id)
            if hasattr(self.memory_provider, "update_memagent_memory_ids"):
                self.memory_provider.update_memagent_memory_ids(self.agent_id, self.memory_ids)

        # 2) Ensure conversation_id
        if conversation_id is None:
            # Use MongoDB ObjectId for better performance
            conversation_id = str(ObjectId())

        # 3) Augment the user query with the query
        augmented_query = f"This is the query to be answered or key objective to be achieved: {query}"

        # Get the prompt for the memory types
        # TODO: Need to get the memory types the agent is using or infer from the memory_mode
        memory_types = [MemoryType.WORKFLOW_MEMORY, MemoryType.CONVERSATION_MEMORY, MemoryType.PERSONAS]
        cwm_prompt = CWM.get_prompt_from_memory_types(memory_types)

        augmented_query += f"\n\n{cwm_prompt}"

        # Check for shared memory context (multi-agent coordination)
        shared_memory_context = self._check_for_shared_memory_context()
        if shared_memory_context:
            augmented_query += shared_memory_context

        # 4) Load and integrate workflow memory
        if self.memory_mode in (MemoryMode.Workflow): #MemoryMode.Default is not used here because it is not a valid memory mode
            try:
                # Retrieve relevant workflows based on the query
                relevant_workflows = Workflow.retrieve_workflows_by_query(query, self.memory_provider)
                
                if relevant_workflows and len(relevant_workflows) > 0:
                    workflow_context = "\n\n---------THIS IS YOUR WORFLOW MEMORY---------\n"
                    workflow_context += "\n\nPrevious workflow executions that may be relevant to ensure you are on the right track, use this information to guide your execution:\n"
                    for workflow in relevant_workflows:
                        # Add workflow details including outcome to guide execution
                        workflow_context += f"- Workflow '{workflow.name}': {workflow.description}\n"
                        workflow_context += f"  Outcome: {workflow.outcome.value}\n"
                        if workflow.outcome == WorkflowOutcome.FAILURE:
                            workflow_context += f"  Error: {workflow.steps.get('error', 'Unknown error')}\n"
                        
                        # Add detailed step information
                        workflow_context += f"  Steps taken: {len(workflow.steps)}\n"
                        for step_name, step_data in workflow.steps.items():
                            workflow_context += f"    Step: {step_name}\n"
                            workflow_context += f"      Function: {step_data.get('_id', 'Unknown')}\n"
                            workflow_context += f"      Arguments: {step_data.get('arguments', {})}\n"
                            workflow_context += f"      Result: {step_data.get('result', 'No result')}\n"
                            if step_data.get('error'):
                                workflow_context += f"      Error: {step_data.get('error')}\n"
                            workflow_context += f"      Timestamp: {step_data.get('timestamp', 'Unknown')}\n"
                        workflow_context += "\n"
                    
                    augmented_query += workflow_context

            except Exception as e:
                logger.error(f"Error loading workflow memory: {str(e)}")
                # Continue execution even if workflow memory loading fails

        # 5) Build system + user prompt
        system_prompt = self._generate_system_prompt()

        # Write a conversational history prompt
        conversational_history_prompt = "---------THIS IS YOUR CONVERSATIONAL HISTORY MEMORY---------\n"
        conversational_history_prompt += "\n\nPrevious conversations that may be relevant to ensure you are on the right track, use this information to guide your execution:\n"
        augmented_query += conversational_history_prompt

        # 6) Append past conversation history
        for conv in self.load_conversation_history(memory_id):
            augmented_query += (
                f"\n\n{conv['role']}: {conv['content']}. "
            )

        # Write relevant memory components prompt
        relevant_memory_components_prompt = "---------THIS IS YOUR RELEVANT MEMORY COMPONENTS---------\n"
        relevant_memory_components_prompt += "\n\nRelevant memory components that may be relevant to ensure you are on the right track, use this information to guide your execution:\n"
        augmented_query += relevant_memory_components_prompt

        # 7) Append relevant memory components
        if self.memory_mode in (MemoryMode.Conversational, MemoryMode.Default):
            for mem in self._load_relevant_memory_components(
                query, MemoryType.CONVERSATION_MEMORY, memory_id, limit=5
            ):
                augmented_query += (
                    f"\n\n{mem['role']}: {mem['content']}. "
                )

        # Add long-term knowledge to the prompt if agent has long_term_memory_ids
        if hasattr(self, "long_term_memory_ids") and self.long_term_memory_ids:
            # Write long term memory prompt
            long_term_memory_prompt = "---------THIS IS YOUR LONG-TERM KNOWLEDGE---------\n"
            long_term_memory_prompt += "\n\nRelevant knowledge from your long-term memory that may help answer the query:\n"
            augmented_query += long_term_memory_prompt
            
            # Import knowledge base and load relevant knowledge
            kb = KnowledgeBase(self.memory_provider)
            
            # First, try to retrieve knowledge semantically similar to the query
            semantic_entries = kb.retrieve_knowledge_by_query(query, limit=3)
            
            # Add semantic matches first
            if semantic_entries:
                augmented_query += "\n\n--- Semantically Relevant Knowledge ---\n"
                for entry in semantic_entries:
                    augmented_query += f"\n\nKnowledge: {entry.get('content', '')}\n"
                    augmented_query += f"Namespace: {entry.get('namespace', 'general')}\n"
            
            # For each memory ID, retrieve and add relevant knowledge
            augmented_query += "\n\n--- Agent's Associated Knowledge ---\n"
            for memory_id in self.long_term_memory_ids:
                knowledge_entries = kb.retrieve_knowledge(memory_id)
                for entry in knowledge_entries:
                    # Skip entries already included in semantic search
                    if entry in semantic_entries:
                        continue
                    augmented_query += f"\n\nKnowledge: {entry.get('content', '')}\n"
                    augmented_query += f"Namespace: {entry.get('namespace', 'general')}\n"
        
        # Reinforce the user query and the objective to end the prompt construction
        augmented_query += f"\n\nRemember the user query to address and objective is: {query}"

        # 8) Record the user's turn in memory
        if self.memory_mode in (MemoryMode.Conversational, MemoryMode.Default):
            self._generate_conversational_memory_component({
                "role": Role.User,
                "content": query,
                "timestamp": datetime.now().isoformat(),
                "conversation_id": conversation_id,
                "memory_id": memory_id,
            })

        # 9) Seed the chat
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": augmented_query},
        ]

        tool_choice = "auto"

        # Export system prompt to file
        with open("monitoring_system_prompt.txt", "w") as f:
            f.write(system_prompt)

        # Export augmented query to file
        with open("monitoring_augmented_query.txt", "w") as f:
            f.write(augmented_query)

        # 10) Main loop
        for _ in range(self.max_steps):
            # a) Build function schema list
            if self.tools:
                if isinstance(self.tools, Toolbox):
                    if self.tool_access == "global":
                        tool_metas = self._load_tools_from_toolbox(query)
                    else:
                        # For private access, convert Toolbox tools to the expected format
                        tool_metas = []
                        for tool_meta in self.tools.list_tools():
                            formatted_tool = self._format_tool(tool_meta)
                            if formatted_tool is not None:
                                tool_metas.append(formatted_tool)
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
                # Create a workflow to track all tool calls
                workflow = Workflow(
                    name=f"Tool Execution: {len(tool_calls)} steps",
                    description=f"Execution of {len(tool_calls)} tools",
                    memory_id=memory_id,
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                    outcome=WorkflowOutcome.SUCCESS,
                    user_query=query
                )

                for call in tool_calls:
                    name = call.name
                    error_message = None  # Initialize error message for this tool call
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
                        # For Toolbox, search in the formatted tools we just created
                        for meta in tool_metas:
                            if meta["name"] == name:
                                entry = meta
                                # Use the Toolbox's get_function_by_id method
                                fn = self.tools.get_function_by_id(str(meta.get("_id")))
                                break
                    elif isinstance(self.tools, list):
                        formatted = self._load_tools_from_memagent()
                        for t in formatted:
                            if t["name"] == name:
                                entry = t
                                for orig in self.tools:
                                    # Handle different tool structures
                                    orig_name = None
                                    if "name" in orig:
                                        orig_name = orig["name"]
                                    elif "function" in orig and isinstance(orig["function"], dict):
                                        orig_name = orig["function"].get("name")
                                    
                                    if orig_name == name:
                                        fn = getattr(self, "_tool_functions", {}).get(str(orig.get("_id")))
                                        break
                                break

                    if not entry:
                        result = f"Error: Tool '{name}' not found in available tools."
                        error_message = result
                        workflow_outcome = WorkflowOutcome.FAILURE
                    elif not callable(fn):
                        logger.warning(f"Tool '{name}' found but function is not callable. Tool ID: {entry.get('_id')}")
                        result = f"Sorry, the tool '{name}' is currently unavailable. It exists in the system but its implementation function is not properly registered."
                        error_message = result
                        workflow_outcome = WorkflowOutcome.FAILURE
                    else:
                        try:
                            # e) Execute and append the function's result
                            result = fn(**args)
                            workflow_outcome = WorkflowOutcome.SUCCESS
                        except Exception as e:
                            logger.error(f"Error executing tool {name}: {str(e)}")
                            result = f"Error executing tool {name}: {str(e)}"
                            error_message = str(e)
                            workflow_outcome = WorkflowOutcome.FAILURE

                    # Append the result (either actual function output or error message)
                    messages.append({
                        "type":    "function_call_output",
                        "call_id": call.call_id,
                        "output":  str(result),
                    })

                    # Add step to workflow
                    workflow.add_step(f"Step {len(workflow.steps) + 1}: {name}", {
                        "_id": str(entry.get("_id")) if entry else None,
                        "arguments": args,
                        "result": result,
                        "timestamp": datetime.now().isoformat(),
                        "error": error_message
                    })

                    # Update workflow outcome if any step failed
                    if workflow_outcome == WorkflowOutcome.FAILURE:
                        workflow.outcome = WorkflowOutcome.FAILURE

                # Store the complete workflow with all steps
                if self.memory_mode in (MemoryMode.Workflow, MemoryMode.Default):
                    try:
                        workflow.store_workflow(self.memory_provider)
                    except Exception as e:
                        logger.error(f"Error storing workflow: {str(e)}")
                        # Continue execution even if workflow storage fails

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

        # 11) If we never returned…
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
        # Convert tools to serializable format if it's a Toolbox object
        tools_to_save = self.tools
        if isinstance(self.tools, Toolbox):
            # Convert Toolbox to list of tool metadata for serialization
            tools_to_save = []
            for tool_meta in self.tools.list_tools():
                # Check if 'function' field contains metadata (dict) or raw function (callable)
                function_field = tool_meta.get("function", {})
                
                if callable(function_field):
                    # If it's a raw function, skip this tool as it's improperly stored
                    logger.warning(f"Skipping tool with _id {tool_meta.get('_id')} - contains raw function instead of metadata")
                    continue
                elif isinstance(function_field, dict):
                    # If it's proper metadata, extract serializable tool information
                    serializable_tool = {
                        "_id": tool_meta.get("_id"),
                        "function": {
                            "name": function_field.get("name"),
                            "description": function_field.get("description"),
                            "parameters": function_field.get("parameters", [])
                        },
                        "type": tool_meta.get("type", "function")
                    }
                    tools_to_save.append(serializable_tool)
                else:
                    # If it's neither dict nor callable, skip with warning
                    logger.warning(f"Skipping tool with _id {tool_meta.get('_id')} - unknown function field type: {type(function_field)}")
                    continue
        
        # Create a new MemAgentModel with the current object's attributes
        memagent_to_save = MemAgentModel(
            instruction=self.instruction,
            memory_mode=self.memory_mode,
            max_steps=self.max_steps,
            memory_ids=self.memory_ids,
            agent_id=self.agent_id,  # This will be removed in the provider
            persona=self.persona,
            tools=tools_to_save,
            long_term_memory_ids=getattr(self, "long_term_memory_ids", None)
        )

        # Store the memagent in the memory provider
        saved_memagent = self.memory_provider.store_memagent(memagent_to_save)

        # Update the agent_id to the MongoDB _id that was generated
        self.agent_id = str(saved_memagent["_id"])

        # Log the saved memagent
        logger.info(f"Memagent {self.agent_id} saved in the memory provider")
        # Log the details and attributes of the saved memagent
        # Show the logs as a json object
        logger.info(json.dumps(saved_memagent, indent=4, default=str))

        return self


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

        # Convert tools to serializable format if it's a Toolbox object
        tools_to_update = self.tools
        if isinstance(self.tools, Toolbox):
            # Convert Toolbox to list of tool metadata for serialization
            tools_to_update = []
            for tool_meta in self.tools.list_tools():
                # Extract serializable tool information
                # Check if 'function' field contains metadata (dict) or raw function (callable)
                function_field = tool_meta.get("function", {})
                
                if callable(function_field):
                    # If it's a raw function, skip this tool as it's improperly stored
                    logger.warning(f"Skipping tool with _id {tool_meta.get('_id')} - contains raw function instead of metadata")
                    continue
                elif isinstance(function_field, dict):
                    # If it's proper metadata, extract serializable tool information
                    serializable_tool = {
                        "_id": tool_meta.get("_id"),
                        "function": {
                            "name": function_field.get("name"),
                            "description": function_field.get("description"),
                            "parameters": function_field.get("parameters", [])
                        },
                        "type": tool_meta.get("type", "function")
                    }
                    tools_to_update.append(serializable_tool)
                else:
                    # If it's neither dict nor callable, skip with warning
                    logger.warning(f"Skipping tool with _id {tool_meta.get('_id')} - unknown function field type: {type(function_field)}")
                    continue

        memagent_to_update = MemAgentModel(
            instruction=self.instruction,
            memory_mode=self.memory_mode,
            max_steps=self.max_steps,
            memory_ids=self.memory_ids,
            agent_id=self.agent_id,
            persona=self.persona,
            tools=tools_to_update
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
        
        # Set long_term_memory_ids if they exist
        if hasattr(memagent, "long_term_memory_ids") and memagent.long_term_memory_ids:
            memagent.long_term_memory_ids = memagent.long_term_memory_ids

        # Show the logs as a json object
        logger.info(f"MemAgent loaded with agent_id: {agent_id}")

        return memagent
    
    def refresh(self):
        """
        Refresh the memagent from the memory provider.

        This method refreshes the memagent from the memory provider.

        Returns:
            MemAgent: The refreshed memagent.
        """
        try:
            # Get a fresh copy of the memagent from the memory provider
            memagent = self.memory_provider.retrieve_memagent(self.agent_id)

            # Update the memagent with the fresh copy
            self.__dict__.update(memagent.__dict__)

            return self
        except Exception as e:
            logger.error(f"Error refreshing memagent {self.agent_id}: {e}")
            return False
    
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
    
    # Memory Management Methods

    def download_memory(self, memagent: MemAgentModel):
        """
        Download the memory of the memagent.

        This method downloads the memory of the memagent.
        It takes in a memagent and then adds the memory_ids of the memagent to the memory_ids attribute of the memagent.
        It then updates the memory_ids of the memagent in the memory provider.

        Parameters:
            memagent (MemAgent): The memagent to download the memory from.

        Returns:
            bool: True if the memory was downloaded successfully, False otherwise.
        """

        try:
            # Add the list of the memory_ids to the memory_ids attribute of the memagent
            self.memory_ids = self.memory_ids + memagent.memory_ids

            # Update the memory_ids of the memagent in the memory provider
            if hasattr(self.memory_provider, 'update_memagent_memory_ids'):
                self.memory_provider.update_memagent_memory_ids(self.agent_id, self.memory_ids)
            else:
                raise ValueError("Memory provider does not have the update_memagent_memory_ids method.")
            return True
        except Exception as e:
            logger.error(f"Error downloading memory from memagent {memagent.agent_id}: {e}")
            return False
    
    def update_memory(self, memory_ids: List[str]):
        """
        Update the memory_ids of the memagent.

        This method updates the memory_ids of the memagent.
        It takes in a list of memory_ids and then adds the list to the memory_ids attribute of the memagent.
        It then updates the memory_ids of the memagent in the memory provider.

        Parameters:
            memory_ids (List[str]): The memory_ids to update.

        Returns:
            bool: True if the memory_ids were updated successfully, False otherwise.
        """

        try:           

            # Update the memory_ids of the memagent in the memory provider
            if hasattr(self.memory_provider, 'update_memagent_memory_ids'):
                 # Add the list of memory_ids to the memory_ids attribute of the memagent
                memories_to_add = self.memory_ids + memory_ids
                self.memory_provider.update_memagent_memory_ids(self.agent_id, memories_to_add)
                
                # Update the memory_ids of the memagent in the memagent
                self.memory_ids = memories_to_add
            else:
                raise ValueError("Memory provider does not have the update_memagent_memory_ids method.")
            
            return True
        except Exception as e:
            logger.error(f"Error updating memory_ids of memagent {self.agent_id}: {e}")
            return False
    
    def delete_memory(self):
        """
        Delete the memory_ids of the memagent.

        It deletes the memory_ids of the memagent in the memory provider.

        Returns:
            bool: True if the memory_ids were deleted successfully, False otherwise.
        """

        try:
            if hasattr(self.memory_provider, 'delete_memagent_memory_ids'):
                # Delete the memory_ids of the memagent in the memory provider
                self.memory_provider.delete_memagent_memory_ids(self.agent_id)

                # Delete the memory_ids of the memagent in the memagent
                self.memory_ids = []
                return True
            else:
                raise ValueError("Memory provider does not have the delete_memagent_memory_ids method.")
        except Exception as e:
            logger.error(f"Error deleting memory_ids of memagent {self.agent_id}: {e}")
            return False

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

        # Check if the memory provider has the retrieve_persona method
        if hasattr(self.memory_provider, 'retrieve_persona'):
            self.persona = self.memory_provider.retrieve_persona(persona_id)
        else:
            raise ValueError("Memory provider does not have the retrieve_persona method.")

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
        if tool_id:
            # 1) fetch the metadata from memory-provider
            meta = self.memory_provider.retrieve_by_id(tool_id, MemoryType.TOOLBOX)
            if not meta:
                raise ValueError(f"No such tool in the toolbox: {tool_id}")

            # 2) resolve the Python callable from the provided Toolbox
            if not isinstance(toolbox, Toolbox):
                raise ValueError("Need a Toolbox instance to resolve the Python callable")

            # Use _id (which is the same as tool_id now) for function lookup
            python_fn = toolbox._tools.get(tool_id)
            if not callable(python_fn):
                # fallback: perhaps the stored metadata itself packs a .function field?
                tb_meta = next(
                    (m for m in toolbox.list_tools() if str(m.get("_id")) == tool_id),
                    None
                )
                if tb_meta and callable(tb_meta.get("function")):
                    python_fn = tb_meta["function"]

            if not callable(python_fn):
                # Silently skip tools without functions instead of raising error
                return False

            # 3) build the new entry
            new_entry = self._build_entry(meta, python_fn)

            # 4) Handle different types of self.tools (Toolbox vs list)
            if isinstance(self.tools, Toolbox):
                # If self.tools is a Toolbox, convert to list format for agent use
                self.tools = []
            
            # Now self.tools should be a list (or None)
            existing_idx = None
            for idx, t in enumerate(self.tools or []):
                if str(t.get("_id")) == tool_id:
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
            self.update(tools=self.tools)
            return True

        # --- batch‐import branch ---
        if toolbox:
            if not isinstance(toolbox, Toolbox):
                raise TypeError(f"Expected a Toolbox, got {type(toolbox)}")
            
            # If self.tools is already the same Toolbox, no need to re-add
            if self.tools is toolbox:
                return True
                
            # Use the shared initialization logic for better performance
            # Convert existing tools to list format if needed
            if isinstance(self.tools, Toolbox):
                self.tools = []
            
            # Store current tools count to check if any were added
            initial_count = len(self.tools or [])
            
            # Add all tools from the toolbox using shared logic
            for meta in toolbox.list_tools():
                tid = str(meta.get("_id"))
                if tid:
                    # Resolve the Python callable from the provided Toolbox
                    python_fn = toolbox._tools.get(tid)
                    if not callable(python_fn):
                        # fallback: perhaps the stored metadata itself packs a .function field?
                        if meta and callable(meta.get("function")):
                            python_fn = meta["function"]

                    if callable(python_fn):
                        # Check if tool already exists to avoid duplicates
                        existing_idx = None
                        for idx, t in enumerate(self.tools or []):
                            if str(t.get("_id")) == tid:
                                existing_idx = idx
                                break

                        # build the new entry
                        new_entry = self._build_entry(meta, python_fn)
                        
                        if existing_idx is not None:
                            # replace the old entry
                            self.tools[existing_idx] = new_entry
                        else:
                            # append a brand-new entry
                            self.tools = self.tools or []
                            self.tools.append(new_entry)
                    else:
                        # Silently skip tools without functions
                        logger.warning(f"Skipping tool with _id {tid} - no callable function found")
                        continue
            
            # Save the updated tools if any were added
            final_count = len(self.tools or [])
            if final_count > initial_count:
                self.update(tools=self.tools)
                return True
            
            return final_count > 0  # Return True if we have tools, even if none were newly added

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
                if tool.get("_id") is None:
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
            tool_id (str): The tool id (_id).

        Returns:
            bool: True if the tool was deleted successfully, False otherwise.
        """
        if self.tools:
            self.tools = [tool for tool in self.tools if str(tool.get("_id")) != tool_id]
            self.save()
        else:
            raise ValueError("No tools to delete. Please add a tool to the memagent before deleting it.")

    def __str__(self):
        """
        Return a string representation of the memagent.
        """

        # Get a fresh copy of the memagent from the memory provider
        self.refresh()

        return f"MemAgent(agent_id={self.agent_id}, memory_ids={self.memory_ids}, memory_mode={self.memory_mode}, max_steps={self.max_steps}, instruction={self.instruction}, model={self.model}, tools={self.tools}, persona={self.persona})"
    
    def __repr__(self):
        """
        Return a string representation of the memagent that can be used to recreate the object.
        """
        return f"MemAgent(agent_id={self.agent_id}, memory_provider={self.memory_provider})"

    # Long-term Memory Management
    
    def add_long_term_memory(self, corpus: str, namespace: str = "general") -> Optional[str]:
        """
        Add long-term memory to the agent.
        
        Parameters:
            corpus (str): The text content to store in long-term memory.
            namespace (str): A namespace to categorize the knowledge.
            
        Returns:
            Optional[str]: The ID of the created long-term memory, or None if unsuccessful.
        """
        try:
            kb = KnowledgeBase(self.memory_provider)
            
            # Ingest the knowledge
            long_term_memory_id = kb.ingest_knowledge(corpus, namespace)
            
            # Initialize the long_term_memory_ids attribute if it doesn't exist
            if not hasattr(self, "long_term_memory_ids"):
                self.long_term_memory_ids = []
                
            # Add the ID to the agent's long-term memory IDs
            self.long_term_memory_ids.append(long_term_memory_id)
            
            # Save the updated agent
            self.save()
            
            return long_term_memory_id
        except Exception as e:
            logger.error(f"Error adding long-term memory to agent {self.agent_id}: {e}")
            return None
    
    def retrieve_long_term_memory(self, long_term_memory_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve long-term memory associated with the agent.
        
        Parameters:
            long_term_memory_id (Optional[str]): The ID of a specific long-term memory to retrieve.
                                                If None, retrieves all long-term memories associated with the agent.
                                                
        Returns:
            List[Dict[str, Any]]: List of knowledge entries.
        """
        try:
            kb = KnowledgeBase(self.memory_provider)
            
            # Initialize result list
            all_entries = []
            
            # If a specific ID is provided, retrieve just that one
            if long_term_memory_id:
                return kb.retrieve_knowledge(long_term_memory_id)
                
            # Otherwise, retrieve all long-term memories associated with the agent
            if hasattr(self, "long_term_memory_ids") and self.long_term_memory_ids:
                for memory_id in self.long_term_memory_ids:
                    entries = kb.retrieve_knowledge(memory_id)
                    all_entries.extend(entries)
                    
            return all_entries
        except Exception as e:
            logger.error(f"Error retrieving long-term memory for agent {self.agent_id}: {e}")
            return []
    
    def delete_long_term_memory(self, long_term_memory_id: str) -> bool:
        """
        Delete a specific long-term memory and remove it from the agent.
        
        Parameters:
            long_term_memory_id (str): The ID of the long-term memory to delete.
            
        Returns:
            bool: True if deletion was successful, False otherwise.
        """
        try:
            # Remove the ID from the agent's list
            if hasattr(self, "long_term_memory_ids") and self.long_term_memory_ids:
                if long_term_memory_id in self.long_term_memory_ids:
                    self.long_term_memory_ids.remove(long_term_memory_id)
                    self.update()
            
            # Delete the knowledge from the memory provider
            kb = KnowledgeBase(self.memory_provider)
            return kb.delete_knowledge(long_term_memory_id)
        except Exception as e:
            logger.error(f"Error deleting long-term memory {long_term_memory_id}: {e}")
            return False
    
    def update_long_term_memory(self, long_term_memory_id: str, corpus: str) -> bool:
        """
        Update the content of a specific long-term memory.
        
        Parameters:
            long_term_memory_id (str): The ID of the long-term memory to update.
            corpus (str): The new text content.
            
        Returns:
            bool: True if update was successful, False otherwise.
        """
        try:
            # Check if this memory is associated with the agent
            if not hasattr(self, "long_term_memory_ids") or long_term_memory_id not in self.long_term_memory_ids:
                logger.warning(f"Long-term memory {long_term_memory_id} is not associated with agent {self.agent_id}")
                return False
            
            # Update the knowledge
            kb = KnowledgeBase(self.memory_provider)
            return kb.update_knowledge(long_term_memory_id, corpus)
        except Exception as e:
            logger.error(f"Error updating long-term memory {long_term_memory_id}: {e}")
            return False

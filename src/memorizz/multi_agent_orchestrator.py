import asyncio
import concurrent.futures
from typing import List, Dict, Any, Optional
from .memagent import MemAgent
from .shared_memory import SharedMemory
from .task_decomposition import TaskDecomposer, SubTask
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class MultiAgentOrchestrator:
    """Orchestrates multi-agent execution with parallel processing."""
    
    def __init__(self, root_agent: MemAgent, delegates: List[MemAgent]):
        self.root_agent = root_agent
        self.delegates = delegates
        self.shared_memory = SharedMemory(root_agent.memory_provider)
        self.task_decomposer = TaskDecomposer(root_agent)
        self.shared_session_id = None
        
    def execute_multi_agent_workflow(self, 
                                   user_query: str, 
                                   memory_id: str = None,
                                   conversation_id: str = None) -> str:
        """
        Execute a multi-agent workflow with task decomposition and parallel execution.
        
        Parameters:
            user_query (str): The original user query
            memory_id (str): Memory ID for context
            conversation_id (str): Conversation ID for tracking
            
        Returns:
            str: Consolidated response from all agents
        """
        try:
            # 1. Create shared memory session
            delegate_ids = [agent.agent_id for agent in self.delegates]
            self.shared_session_id = self.shared_memory.create_shared_session(
                root_agent_id=self.root_agent.agent_id,
                delegate_agent_ids=delegate_ids
            )
            
            # Log the start of multi-agent execution
            self.shared_memory.add_blackboard_entry(
                session_id=self.shared_session_id,
                agent_id=self.root_agent.agent_id,
                content={"original_query": user_query, "started_at": datetime.now().isoformat()},
                entry_type="workflow_start"
            )
            
            # 2. Decompose task into sub-tasks
            sub_tasks = self.task_decomposer.decompose_task(user_query, self.delegates)
            
            if not sub_tasks:
                # Fallback to single agent execution
                logger.warning("Task decomposition failed, falling back to root agent")
                return self.root_agent.run(user_query, memory_id, conversation_id)
            
            # Log task decomposition
            self.shared_memory.add_blackboard_entry(
                session_id=self.shared_session_id,
                agent_id=self.root_agent.agent_id,
                content={"sub_tasks": [task.to_dict() for task in sub_tasks]},
                entry_type="task_decomposition"
            )
            
            # 3. Execute sub-tasks in parallel
            sub_task_results = self._execute_sub_tasks_parallel(sub_tasks, memory_id, conversation_id)
            
            # 4. Consolidate results
            consolidated_response = self._consolidate_results(user_query, sub_task_results)
            
            # 5. Update shared memory with final result
            self.shared_memory.add_blackboard_entry(
                session_id=self.shared_session_id,
                agent_id=self.root_agent.agent_id,
                content={"consolidated_response": consolidated_response, "completed_at": datetime.now().isoformat()},
                entry_type="workflow_complete"
            )
            
            # Mark session as completed
            self.shared_memory.update_session_status(self.shared_session_id, "completed")
            
            return consolidated_response
            
        except Exception as e:
            logger.error(f"Error in multi-agent workflow: {e}")
            # Mark session as failed
            if self.shared_session_id:
                self.shared_memory.update_session_status(self.shared_session_id, "failed")
            
            # Fallback to single agent execution
            return self.root_agent.run(user_query, memory_id, conversation_id)
    
    def _execute_sub_tasks_parallel(self, 
                                  sub_tasks: List[SubTask], 
                                  memory_id: str,
                                  conversation_id: str) -> List[Dict[str, Any]]:
        """Execute sub-tasks in parallel using ThreadPoolExecutor."""
        
        # Create agent mapping for quick lookup
        agent_map = {agent.agent_id: agent for agent in self.delegates}
        
        # Sort tasks by priority and handle dependencies
        sorted_tasks = sorted(sub_tasks, key=lambda x: x.priority)
        
        results = []
        completed_tasks = set()
        
        # Use ThreadPoolExecutor for parallel execution
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.delegates)) as executor:
            # Submit tasks that have no dependencies first
            future_to_task = {}
            
            for task in sorted_tasks:
                if not task.dependencies:  # No dependencies, can execute immediately
                    agent = agent_map.get(task.assigned_agent_id)
                    if agent:
                        future = executor.submit(self._execute_single_task, task, agent, memory_id, conversation_id)
                        future_to_task[future] = task
            
            # Process completed tasks and submit dependent tasks
            while future_to_task:
                # Wait for at least one task to complete
                done_futures = concurrent.futures.wait(future_to_task.keys(), return_when=concurrent.futures.FIRST_COMPLETED)
                
                for future in done_futures.done:
                    task = future_to_task[future]
                    try:
                        result = future.result()
                        task.result = result
                        task.status = "completed"
                        completed_tasks.add(task.task_id)
                        results.append(task.to_dict())
                        
                        # Log task completion
                        self.shared_memory.add_blackboard_entry(
                            session_id=self.shared_session_id,
                            agent_id=task.assigned_agent_id,
                            content={"task_id": task.task_id, "result": result},
                            entry_type="task_completion"
                        )
                        
                    except Exception as e:
                        logger.error(f"Error executing task {task.task_id}: {e}")
                        task.status = "failed"
                        task.result = f"Error: {str(e)}"
                        results.append(task.to_dict())
                    
                    del future_to_task[future]
                
                # Check for new tasks that can be executed (dependencies met)
                for task in sorted_tasks:
                    if (task.status == "pending" and 
                        task.task_id not in [t.task_id for t in future_to_task.values()] and
                        all(dep in completed_tasks for dep in task.dependencies)):
                        
                        agent = agent_map.get(task.assigned_agent_id)
                        if agent:
                            future = executor.submit(self._execute_single_task, task, agent, memory_id, conversation_id)
                            future_to_task[future] = task
        
        return results
    
    def _execute_single_task(self, 
                           task: SubTask, 
                           agent: MemAgent, 
                           memory_id: str,
                           conversation_id: str) -> str:
        """Execute a single sub-task with an agent."""
        
        try:
            # Log task start
            self.shared_memory.add_blackboard_entry(
                session_id=self.shared_session_id,
                agent_id=agent.agent_id,
                content={"task_id": task.task_id, "description": task.description, "started_at": datetime.now().isoformat()},
                entry_type="task_start"
            )
            
            # Execute the task
            result = agent.run(task.description, memory_id, conversation_id)
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing task {task.task_id} with agent {agent.agent_id}: {e}")
            raise
    
    def _consolidate_results(self, 
                           original_query: str, 
                           sub_task_results: List[Dict[str, Any]]) -> str:
        """Consolidate results from all sub-tasks into a final response."""
        
        try:
            # Create consolidation prompt
            consolidation_prompt = self.task_decomposer.create_consolidation_prompt(
                original_query, sub_task_results
            )
            
            # Use root agent to consolidate
            messages = [
                {"role": "system", "content": consolidation_prompt},
                {"role": "user", "content": "Please provide a consolidated response based on the sub-task results."}
            ]
            
            response = self.root_agent.model.client.responses.create(
                model="gpt-4.1",
                input=messages
            )
            
            return response.output_text
            
        except Exception as e:
            logger.error(f"Error consolidating results: {e}")
            
            # Fallback: simple concatenation
            consolidated = f"Results for: {original_query}\n\n"
            for result in sub_task_results:
                consolidated += f"Task: {result.get('description', 'Unknown')}\n"
                consolidated += f"Result: {result.get('result', 'No result')}\n\n"
            
            return consolidated
    
    def get_shared_memory_context(self) -> str:
        """Get shared memory context for inclusion in agent prompts."""
        
        if not self.shared_session_id:
            return ""
        
        try:
            # Get blackboard entries
            entries = self.shared_memory.get_blackboard_entries(self.shared_session_id)
            
            if not entries:
                return ""
            
            context = "\n\n---------SHARED MEMORY CONTEXT---------\n"
            context += "Multi-agent coordination information:\n\n"
            
            for entry in entries[-10:]:  # Last 10 entries
                context += f"Agent: {entry.get('agent_id', 'Unknown')}\n"
                context += f"Type: {entry.get('entry_type', 'Unknown')}\n"
                context += f"Content: {entry.get('content', 'No content')}\n"
                context += f"Time: {entry.get('created_at', 'Unknown')}\n"
                context += "---\n"
            
            return context
            
        except Exception as e:
            logger.error(f"Error getting shared memory context: {e}")
            return "" 
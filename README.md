# Memorizz 🧠

[![PyPI version](https://badge.fury.io/py/memorizz.svg)](https://badge.fury.io/py/memorizz)
[![PyPI downloads](https://img.shields.io/pypi/dm/memorizz.svg)](https://pypistats.org/packages/memorizz)

> **⚠️ IMPORTANT WARNING ⚠️**
> 
> **MemoRizz is an EXPERIMENTAL library intended for EDUCATIONAL PURPOSES ONLY.**
> 
> **Do NOT use in production environments or with sensitive data.**
> 
> This library is under active development, has not undergone security audits, and may contain bugs or breaking changes in future releases.

## Overview

MemoRizz is an advanced memory management framework designed for AI agents, enabling persistent, context-aware, and semantically searchable information storage. It seamlessly integrates MongoDB with vector embedding capabilities, empowering agents with sophisticated cognitive functions such as conversation history tracking, tool usage management, and consistent persona maintenance. 

**Why MemoRizz?**
- 🧠 **Persistent Memory**: Your AI agents remember conversations across sessions
- 🔍 **Semantic Search**: Find relevant information using natural language
- 🛠️ **Tool Integration**: Automatically discover and execute functions
- 👤 **Persona System**: Create consistent, specialized agent personalities
- 📊 **Vector Search**: MongoDB Atlas Vector Search for efficient retrieval

## Key Features

- **Persistent Memory Management**: Long-term memory storage with semantic retrieval
- **MemAgent System**: Complete AI agents with memory, personas, and tools
- **MongoDB Integration**: Built on MongoDB Atlas with vector search capabilities
- **Tool Registration**: Automatically convert Python functions into LLM-callable tools
- **Persona Framework**: Create specialized agent personalities and behaviors
- **Vector Embeddings**: Semantic similarity search across all stored information

## Installation

```bash
pip install memorizz
```

### Prerequisites
- Python 3.7+
- MongoDB Atlas account (or local MongoDB with vector search)
- OpenAI API key (for embeddings and LLM functionality)

## Quick Start

### 1. Basic MemAgent Setup

```python
import os
from memorizz.memory_provider.mongodb.provider import MongoDBConfig, MongoDBProvider
from memorizz.memagent import MemAgent
from memorizz.llms.openai import OpenAI

# Set up your API keys
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

# Configure MongoDB memory provider
mongodb_config = MongoDBConfig(uri="your-mongodb-atlas-uri")
memory_provider = MongoDBProvider(mongodb_config)

# Create a MemAgent
agent = MemAgent(
    model=OpenAI(model="gpt-4"),
    instruction="You are a helpful assistant with persistent memory.",
    memory_provider=memory_provider
)

# Start conversing - the agent will remember across sessions
response = agent.run("Hello! My name is John and I'm a software engineer.")
print(response)

# Later in another session...
response = agent.run("What did I tell you about myself?")
print(response)  # Agent remembers John is a software engineer
```

### 2. Creating Specialized Agents with Personas

```python
from memorizz.persona import Persona
from memorizz.persona import Persona

# Create a technical expert persona
tech_expert = Persona(
    name="TechExpert",
    role="Senior Software Engineer",
    goals="Help developers solve complex technical problems with detailed explanations.",
    background="10+ years experience in Python, AI/ML, and distributed systems."
)

# Apply persona to agent
agent.set_persona(tech_expert)
agent.save()

# Now the agent will respond as a technical expert
response = agent.run("How should I design a scalable microservices architecture?")
```

### 3. Tool Registration and Function Calling

```python
from memorizz.database.mongodb import MongoDBTools, MongoDBToolsConfig

# Configure tools database
tools_config = MongoDBToolsConfig(
    uri="your-mongodb-atlas-uri",
    database_name="my_tools_db"
)

# Register tools using decorator
with MongoDBTools(tools_config) as tools:
    toolbox = tools.mongodb_toolbox()
    
    @toolbox
    def calculate_compound_interest(principal: float, rate: float, time: int) -> float:
        """Calculate compound interest for financial planning."""
        return principal * (1 + rate) ** time
    
    @toolbox
    def get_weather(city: str) -> str:
        """Get current weather for a city."""
        # Your weather API integration here
        return f"Weather in {city}: 72°F, sunny"
    
    # Add tools to your agent
    agent.add_tool(toolbox=tools)
    
    # Agent can now discover and use these tools automatically
    response = agent.run("What's the weather in San Francisco and calculate interest on $1000 at 5% for 3 years?")
```

## Core Concepts

### Memory Types

MemoRizz supports different memory categories for organizing information:

- **Conversation**: Chat history and dialogue context
- **Task**: Goal-oriented information and progress tracking  
- **Workflow**: Multi-step process information
- **General**: Factual knowledge and declarative information
- **Working**: Temporary processing space (LLM context)

### Vector Search

All stored information is automatically embedded and indexed for semantic search:

```python
# Store information with automatic embedding
agent.store_memory("I prefer Python for backend development", memory_type="general")

# Later, semantically related queries will retrieve this info
response = agent.run("What programming languages do I like?")
# Agent will find and use the stored preference
```

### Tool Discovery

Tools are semantically indexed, allowing natural language discovery:

```python
# Tools are automatically found based on intent
agent.run("I need to check the weather")  # Finds and uses get_weather tool
agent.run("Help me calculate some financial returns")  # Finds compound_interest tool
```

## Advanced Usage

### Custom Memory Providers

Extend the memory provider interface for custom storage backends:

```python
from memorizz.memory_provider.base import MemoryProvider

class CustomMemoryProvider(MemoryProvider):
    def store_memory(self, content, memory_type, **kwargs):
        # Your custom storage logic
        pass
    
    def retrieve_memories(self, query, memory_type=None, limit=10):
        # Your custom retrieval logic
        pass
```

### Memory Management

Control how memories are stored and retrieved:

```python
# Store with metadata
agent.store_memory(
    content="Completed project X with React and Node.js",
    memory_type="task",
    metadata={"project": "X", "technologies": ["React", "Node.js"]}
)

# Retrieve specific memories
memories = agent.retrieve_memories(
    query="projects with React",
    memory_type="task",
    limit=5
)
```

## Architecture

```
┌─────────────────┐
│   MemAgent      │  ← High-level agent interface
├─────────────────┤
│   Persona       │  ← Agent personality & behavior
├─────────────────┤
│   Toolbox       │  ← Function registration & discovery
├─────────────────┤
│ Memory Provider │  ← Storage abstraction layer
├─────────────────┤
│ Vector Search   │  ← Semantic similarity & retrieval
├─────────────────┤
│   MongoDB       │  ← Persistent storage backend
└─────────────────┘
```

## Examples

Check out the `examples/` directory for complete working examples:

- **Basic Agent**: Simple conversational agent with memory
- **Specialized Agent**: Technical expert with persona
- **Tool Integration**: Agent with custom function calling
- **Memory Management**: Advanced memory storage and retrieval

## Configuration

### MongoDB Atlas Setup

1. Create a MongoDB Atlas cluster
2. Enable Vector Search on your cluster
3. Create a database and collection for your agent
4. Get your connection string

### Environment Variables

```bash
# Required
export OPENAI_API_KEY="your-openai-api-key"
export MONGODB_URI="your-mongodb-atlas-uri"

# Optional
export MONGODB_DB_NAME="memorizz"  # Default database name
```

## Troubleshooting

**Common Issues:**

1. **MongoDB Connection**: Ensure your IP is whitelisted in Atlas
2. **Vector Search**: Verify vector search is enabled on your cluster
3. **API Keys**: Check OpenAI API key is valid and has credits

## Contributing

This is an educational project. Contributions for learning purposes are welcome:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality  
4. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Educational Resources

This library demonstrates key concepts in:
- **AI Agent Architecture**: Memory, reasoning, and tool use
- **Vector Databases**: Semantic search and retrieval
- **LLM Integration**: Function calling and context management
- **Software Design**: Clean abstractions and extensible architecture

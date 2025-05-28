# MemoRizz

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

MemoRizz is an advanced memory management framework designed for AI agents, enabling persistent, context-aware, and semantically searchable information storage. It seamlessly integrates a Memory Provider (any storage solution or database) with vector embedding storage and retrieval capabilities, empowering agents with sophisticated cognitive functions such as conversation history tracking, tool usage management, and consistent persona maintenance. Agents powered by this extensive and reliable memory system are known as **memagents**. 

MemoRizz provides an abstraction for creating these types of agents through the MemAgent constructor.

## Memory Providers

MemoRizz supports multiple memory providers for different use cases:

### MongoDB (Cloud/Production)
- **MongoDB Atlas** with Vector Search for production deployments
- Requires cloud setup and paid service
- Excellent for scalable, production applications

### PostgreSQL + pgvector (Local/Development) 🆕
- **Local PostgreSQL** with pgvector extension for development
- Easy Docker setup for local development
- Free and open-source
- Perfect for learning and experimentation

See [PostgreSQL Setup Guide](README_postgresql.md) for detailed instructions.

## Core Components

### 1. Memory Management

MemoRizz implements a multi-layered memory architecture modeled after human cognition:

- **Memory Provider**: Abstraction layer for persistent storage (MongoDB, PostgreSQL, etc.)
- **Memory Types**: Specialized stores for different information categories:
  - Conversation: Historical exchanges between agents and users
  - Task: Goal-oriented information for tracking objectives
  - Workflow: Multi-step process information
  - General: Factual, declarative knowledge
  - Working: Temporary processing space (LLM context window)
  - Entity
  - Summarisation

- **Memory Modes**: Different memory management strategies:
  - Default: Balanced approach for general use
  - Conversational: Optimized for dialogue coherence
  - Task: Focused on goal completion
  - Workflow: Specialized for multi-step processes

### 2. Persona System

Create AI agents with consistent identity and personality:

- **Persona Creation**: Define agents with specific roles, goals, and backgrounds
- **Semantic Retrieval**: Find relevant personas based on natural language queries
- **System Prompt Generation**: Automatically create effective prompts based on persona attributes
- **Persistence**: Store and retrieve personas across application sessions

### 3. Toolbox Functionality

Register and discover functions as AI-callable tools:

- **Function Registration**: Convert Python functions to LLM-callable tools
- **Semantic Tool Discovery**: Find relevant tools based on natural language queries
- **Tool Management**: Store, update, and delete tools with database persistence
- **Parameter Handling**: Automatically extract and validate function parameters

### 4. Database Tools

MemoRizz provides database-specific tools for function registration and vector search:

#### MongoDB Tools
```python
from memorizz.database.mongodb import MongoDBTools, MongoDBToolsConfig

config = MongoDBToolsConfig(
    mongo_uri="your-mongodb-uri",
    db_name="function_calling_db",
    get_embedding=your_embedding_function
)

with MongoDBTools(config) as mongo_tools:
    mongodb_toolbox = mongo_tools.mongodb_toolbox()
    
    @mongodb_toolbox
    def example_function(param: str) -> str:
        """Example function description."""
        return f"Result: {param}"
    
    # Find relevant tools
    tools = mongo_tools.populate_tools("search query", num_tools=2)
```

#### PostgreSQL Tools
```python
from memorizz.database.postgresql import PostgreSQLTools, PostgreSQLToolsConfig

config = PostgreSQLToolsConfig(
    host="localhost",
    database="function_calling_db",
    user="postgres",
    password="password",
    get_embedding=your_embedding_function
)

with PostgreSQLTools(config) as pg_tools:
    postgresql_toolbox = pg_tools.postgresql_toolbox()
    
    @postgresql_toolbox
    def example_function(param: str) -> str:
        """Example function description."""
        return f"Result: {param}"
    
    # Find relevant tools
    tools = pg_tools.populate_tools("search query", num_tools=2)
    
    # Additional PostgreSQL-specific operations
    all_tools = pg_tools.list_tools()
    specific_tool = pg_tools.get_tool("function_name")
    pg_tools.delete_tool("function_name")
```

**Key Features:**
- **Identical APIs**: Both MongoDB and PostgreSQL tools provide the same interface
- **Vector Search**: Semantic similarity search for tool discovery
- **Auto-registration**: Functions automatically converted to LLM-callable tools
- **Parameter Extraction**: Automatic function signature analysis
- **Multiple Toolboxes**: Support for organizing tools in different collections/tables

See [Database Tools Comparison](docs/database_tools_comparison.md) for detailed feature comparison.

### 5. MemAgent

Unified agent implementation with memory, persona, and tool capabilities:

- **Stateful Conversations**: Maintain context across multiple interactions
- **Tool Integration**: Seamlessly execute external functions based on natural language requests
- **Persona Incorporation**: Consistent behavior guided by persona attributes
- **Memory Persistence**: Recall relevant information across sessions
- **Access Control**: Configure private or global tool access patterns

## Installation

```bash
pip install memorizz
```

## Quick Start

### Option 1: PostgreSQL (Recommended for Local Development)

```bash
# 1. Start PostgreSQL with pgvector
docker-compose up -d

# 2. Install dependencies
pip install psycopg2-binary numpy

# 3. Run the example
python examples/postgresql_example.py
```

```python
from memorizz.memory_provider.postgresql import PostgreSQLProvider, PostgreSQLConfig
from memorizz.memagent import MemAgent
from memorizz.llms.openai import OpenAI

# Create PostgreSQL memory provider
config = PostgreSQLConfig(
    host="localhost",
    database="memorizz",
    user="postgres",
    password="password"
)
memory_provider = PostgreSQLProvider(config)

# Create agent
agent = MemAgent(
    model=OpenAI(model="gpt-4"),
    instruction="You are a helpful assistant.",
    memory_provider=memory_provider
)

response = agent.run("Hello, can you help me with a question?")
print(response)
```

### Option 2: MongoDB (Production)

```python
from memorizz.memory_provider.mongodb.provider import MongoDBConfig, MongoDBProvider
from memorizz.memagent import MemAgent
from memorizz.llms.openai import OpenAI

# Create MongoDB memory provider
mongodb_config = MongoDBConfig(uri="your-mongodb-uri")
memory_provider = MongoDBProvider(mongodb_config)

# Create agent
agent = MemAgent(
    model=OpenAI(model="gpt-4"),
    instruction="You are a helpful assistant.",
    memory_provider=memory_provider
)

response = agent.run("Hello, can you help me with a question?")
print(response)
```

### Adding a Persona

```python
from memorizz.persona import Persona

# Create a persona
tech_expert = Persona(
    name="TechExpert",
    role="Technical Specialist",
    goals="Help users solve technical problems clearly and accurately.",
    background="An experienced engineer with deep knowledge of software systems."
)

# Add the persona to the agent
agent.set_persona(tech_expert)
agent.save()
```

### Creating a Toolbox

```python
from memorizz.toolbox import Toolbox

# Create a toolbox
toolbox = Toolbox(memory_provider)

# Register a tool
@toolbox.register_tool
def get_weather(latitude: float, longitude: float) -> float:
    """Get the current temperature at the specified coordinates."""
    # Implementation here
    return temperature

# Add tools to the agent
agent.add_tool(toolbox=toolbox)
```

## Architecture

MemoRizz implements a layered architecture:

1. **Agent Layer**: MemAgent instances with personas and tools
2. **Memory Layer**: Memory components and provider abstractions
3. **Embedding & Retrieval Layer**: Vector search and semantic matching
4. **Storage Layer**: Memory Provider (MongoDB/PostgreSQL) with vector capabilities

## Implementation Notes

- Vector embeddings enable semantic search across all memory types
- MongoDB Atlas Vector Search or PostgreSQL pgvector provide efficient similarity-based retrieval
- Memory IDs and conversation IDs maintain relational context
- Each MemAgent maintains its own memory context and tool references

## Feature Roadmap

- [x] Core memory management framework
- [x] Persona system for agent identity
- [x] Toolbox for function registration and discovery
- [x] MongoDB integration with vector search
- [x] PostgreSQL + pgvector integration for local development
- [x] MemAgent implementation with persona and tool support
- [ ] Memory pruning and consolidation strategies
- [ ] Improved cross-agent memory sharing
- [ ] Performance optimizations for large memory stores
- [ ] Expanded embedding model support
- [ ] Enhanced security features
- [ ] Comprehensive test suite and documentation

## Educational Purpose

This library is intended for:
- Learning about memory management in AI systems
- Experimenting with persistent agent architectures
- Understanding vector embeddings for semantic retrieval
- Exploring persona-based agent design
- Studying tool integration in language models

## License

This project is licensed under the MIT License.

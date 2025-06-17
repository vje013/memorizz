# PostgreSQL Memory Provider for Memorizz

This document explains how to set up and use the PostgreSQL memory provider with pgvector for local development and production use.

## 🚀 Quick Start

### Prerequisites

- Python 3.7+
- Docker and Docker Compose (for easy setup)
- PostgreSQL with pgvector extension (or use our Docker setup)

### 1. Start PostgreSQL with pgvector

The easiest way to get started is using our provided Docker Compose setup:

```bash
# Start PostgreSQL with pgvector
docker-compose up -d

# Verify it's running
docker-compose ps
```

This will start:
- PostgreSQL 16 with pgvector extension
- Database: `memorizz`
- User: `postgres`
- Password: `password`
- Port: `5432`

### 2. Install Dependencies

```bash
# Install the updated package with PostgreSQL support
pip install -e .

# Or install dependencies manually
pip install psycopg2-binary numpy
```

### 3. Basic Usage

```python
from memorizz.memory_provider.postgresql import PostgreSQLProvider, PostgreSQLConfig
from memorizz.memory_provider.memory_type import MemoryType

# Configure connection
config = PostgreSQLConfig(
    host="localhost",
    port=5432,
    database="memorizz",
    user="postgres",
    password="password"
)

# Initialize provider
provider = PostgreSQLProvider(config)

# Store some data
data = {
    "name": "My Memory",
    "content": "This is important information",
    "embedding": [0.1, 0.2, 0.3, ...]  # Your embedding vector
}

memory_id = provider.store(data, MemoryType.LONG_TERM_MEMORY)
print(f"Stored memory with ID: {memory_id}")

# Retrieve by ID
retrieved = provider.retrieve_by_id(memory_id, MemoryType.LONG_TERM_MEMORY)
print(f"Retrieved: {retrieved}")

# Vector similarity search
similar = provider.retrieve_by_query(
    {"embedding": search_embedding}, 
    MemoryType.LONG_TERM_MEMORY,
    limit=5
)
```

## 🏗️ Architecture

### Database Schema

The PostgreSQL provider creates the following tables:

#### Core Tables
- `personas` - AI agent personas and roles
- `toolbox` - Available tools and functions
- `short_term_memory` - Temporary memory storage
- `long_term_memory` - Persistent memory storage
- `conversation_memory` - Chat history and interactions
- `workflow_memory` - Process and workflow data
- `agents` - MemAgent configurations

#### Common Columns
All tables include:
- `id` (UUID) - Primary key
- `name` (VARCHAR) - Human-readable name
- `created_at` (TIMESTAMP) - Creation time
- `updated_at` (TIMESTAMP) - Last modification
- `metadata` (JSONB) - Flexible metadata storage
- `embedding` (VECTOR) - Vector embeddings for similarity search

### Vector Search

The provider uses pgvector's HNSW (Hierarchical Navigable Small World) indexes for efficient similarity search:

```sql
-- Example index creation
CREATE INDEX conversation_memory_embedding_idx 
ON conversation_memory 
USING hnsw (embedding vector_cosine_ops);
```

## 🔧 Configuration Options

### PostgreSQLConfig Parameters

```python
config = PostgreSQLConfig(
    host="localhost",        # PostgreSQL host
    port=5432,              # PostgreSQL port
    database="memorizz",    # Database name
    user="postgres",        # Database user
    password="password"     # Database password
)
```

### Environment Variables

You can also use environment variables:

```bash
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
export POSTGRES_DB=memorizz
export POSTGRES_USER=postgres
export POSTGRES_PASSWORD=password
```

## 📊 Performance Considerations

### Vector Search Performance

1. **Index Types**: Uses HNSW indexes for fast approximate nearest neighbor search
2. **Embedding Dimensions**: Optimized for OpenAI's text-embedding-3-small (1536 dimensions)
3. **Distance Metrics**: Supports cosine distance (`<=>` operator)

### Query Optimization

```python
# Efficient: Use vector search with filters
results = provider.retrieve_by_query({
    "embedding": query_embedding,
    "memory_id": "specific_memory"
}, MemoryType.CONVERSATION_MEMORY, limit=10)

# Less efficient: Retrieve all then filter
all_results = provider.list_all(MemoryType.CONVERSATION_MEMORY)
```

## 🔄 Migration from MongoDB

The PostgreSQL provider is designed as a drop-in replacement for the MongoDB provider:

```python
# Before (MongoDB)
from memorizz.memory_provider.mongodb import MongoDBProvider, MongoDBConfig
config = MongoDBConfig(uri="mongodb://localhost:27017", db_name="memorizz")
provider = MongoDBProvider(config)

# After (PostgreSQL)
from memorizz.memory_provider.postgresql import PostgreSQLProvider, PostgreSQLConfig
config = PostgreSQLConfig(host="localhost", database="memorizz")
provider = PostgreSQLProvider(config)

# Same interface for all operations
memory_id = provider.store(data, MemoryType.LONG_TERM_MEMORY)
```

## 🐳 Production Deployment

### Docker Production Setup

```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_DB: memorizz
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    restart: always

volumes:
  postgres_data:
```

### Connection Pooling

For production, consider using connection pooling:

```python
import psycopg2.pool

# In your application
class PooledPostgreSQLProvider(PostgreSQLProvider):
    def __init__(self, config, min_conn=1, max_conn=20):
        self.pool = psycopg2.pool.ThreadedConnectionPool(
            min_conn, max_conn,
            host=config.host,
            port=config.port,
            database=config.database,
            user=config.user,
            password=config.password
        )
```

## 🧪 Testing

Run the example to test your setup:

```bash
# Make sure PostgreSQL is running
docker-compose up -d

# Run the example
python examples/postgresql_example.py
```

Expected output:
```
🚀 PostgreSQL Memory Provider Example
==================================================
📡 Connecting to PostgreSQL...
✅ Connected successfully!
📝 Example 1: Storing a Persona
✅ Stored persona with ID: 123e4567-e89b-12d3-a456-426614174000
...
🎉 All examples completed successfully!
```

## 🔍 Troubleshooting

### Common Issues

1. **Connection Failed**
   ```
   ConnectionError: Failed to connect to PostgreSQL
   ```
   - Check if PostgreSQL is running: `docker-compose ps`
   - Verify connection details in config
   - Check firewall settings

2. **pgvector Extension Missing**
   ```
   ERROR: extension "vector" does not exist
   ```
   - Use the pgvector Docker image: `pgvector/pgvector:pg16`
   - Or install pgvector manually on your PostgreSQL instance

3. **Embedding Dimension Mismatch**
   ```
   ERROR: vector dimension mismatch
   ```
   - Ensure all embeddings have the same dimensions
   - Check your embedding model configuration

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Your provider code here
```

## 🆚 PostgreSQL vs MongoDB Comparison

| Feature | PostgreSQL + pgvector | MongoDB Atlas |
|---------|----------------------|---------------|
| **Local Development** | ✅ Easy Docker setup | ❌ Requires cloud setup |
| **Vector Search** | ✅ Native pgvector | ✅ Atlas Vector Search |
| **ACID Compliance** | ✅ Full ACID | ⚠️ Limited |
| **SQL Support** | ✅ Full SQL | ❌ MongoDB Query Language |
| **Scalability** | ✅ Excellent | ✅ Excellent |
| **Cost** | ✅ Free for local | 💰 Paid service |
| **Ecosystem** | ✅ Mature | ✅ Growing |

## 📚 Additional Resources

- [pgvector Documentation](https://github.com/pgvector/pgvector)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [Vector Similarity Search Guide](https://github.com/pgvector/pgvector#vector-similarity-search)
- [HNSW Algorithm](https://arxiv.org/abs/1603.09320)

## 🤝 Contributing

To contribute to the PostgreSQL provider:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

This PostgreSQL provider is part of the Memorizz project and follows the same MIT license. 
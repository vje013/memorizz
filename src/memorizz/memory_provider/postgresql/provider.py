import psycopg2
import psycopg2.extras
from psycopg2.extensions import register_adapter, AsIs
from psycopg2.errors import DuplicateTable, DuplicateObject
from ..base import MemoryProvider
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, TYPE_CHECKING
from ..memory_type import MemoryType
import json
import uuid
import numpy as np
from datetime import datetime, timezone
from ...memory_component.memory_mode import MemoryMode
from ...embeddings.openai import get_embedding, get_embedding_dimensions

# Use TYPE_CHECKING for forward references to avoid circular imports
if TYPE_CHECKING:
    from ...memagent import MemAgentModel

# Register numpy array adapter for PostgreSQL
def adapt_numpy_array(numpy_array):
    return AsIs(f"'{numpy_array.tolist()}'::vector")

register_adapter(np.ndarray, adapt_numpy_array)

@dataclass
class PostgreSQLConfig:
    """Configuration for the PostgreSQL provider."""
    
    def __init__(self, 
                 host: str = "localhost", 
                 port: int = 5432,
                 database: str = "memorizz", 
                 user: str = "postgres", 
                 password: str = "password"):
        """
        Initialize the PostgreSQL provider with configuration settings.
        
        Parameters:
        -----------
        host : str
            The PostgreSQL host.
        port : int
            The PostgreSQL port.
        database : str
            The database name.
        user : str
            The database user.
        password : str
            The database password.
        """
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password


class PostgreSQLProvider(MemoryProvider):
    """PostgreSQL implementation of the MemoryProvider interface with pgvector support."""
    
    def __init__(self, config: PostgreSQLConfig):
        """
        Initialize the PostgreSQL provider with configuration settings.
        
        Parameters:
        -----------
        config : PostgreSQLConfig
            Configuration object containing database connection details.
        """
        self.config = config
        self.connection = None
        self.embedding_dimensions = get_embedding_dimensions("text-embedding-3-small")
        
        # Connect to PostgreSQL
        self._connect()
        
        # Create all memory stores and indexes
        self._create_memory_stores()
        self._create_vector_indexes()

    def _connect(self):
        """Establish connection to PostgreSQL database."""
        try:
            self.connection = psycopg2.connect(
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.user,
                password=self.config.password
            )
            self.connection.autocommit = True
            
            # Enable pgvector extension
            with self.connection.cursor() as cursor:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                
        except Exception as e:
            raise ConnectionError(f"Failed to connect to PostgreSQL: {e}")

    def _create_memory_stores(self):
        """Create all memory store tables in PostgreSQL."""
        with self.connection.cursor() as cursor:
            # Create tables for each memory type
            for memory_type in MemoryType:
                table_name = memory_type.value
                
                # Base columns for all tables
                base_columns = f"""
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    name VARCHAR(255),
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW(),
                    metadata JSONB DEFAULT '{{}}'::jsonb
                """
                
                # Add memory_id for memory stores (not for personas)
                if memory_type != MemoryType.PERSONAS:
                    base_columns += ",\n                    memory_id VARCHAR(255)"
                
                # Add embedding column for vector search
                base_columns += f",\n                    embedding vector({self.embedding_dimensions})"
                
                # Add specific columns based on memory type
                if memory_type == MemoryType.PERSONAS:
                    specific_columns = """
                        role VARCHAR(100),
                        goals TEXT,
                        background TEXT,
                        persona_id VARCHAR(255)
                    """
                elif memory_type == MemoryType.TOOLBOX:
                    specific_columns = """
                        description TEXT,
                        parameters JSONB,
                        tool_id VARCHAR(255)
                    """
                elif memory_type == MemoryType.CONVERSATION_MEMORY:
                    specific_columns = """
                        content TEXT,
                        role VARCHAR(50),
                        timestamp TIMESTAMP DEFAULT NOW(),
                        conversation_id VARCHAR(255)
                    """
                elif memory_type == MemoryType.WORKFLOW_MEMORY:
                    specific_columns = """
                        description TEXT,
                        steps JSONB,
                        workflow_id VARCHAR(255)
                    """
                elif memory_type == MemoryType.MEMAGENT:
                    specific_columns = """
                        instruction TEXT,
                        memory_mode VARCHAR(50),
                        max_steps INTEGER,
                        memory_ids JSONB DEFAULT '[]'::jsonb,
                        agent_id VARCHAR(255),
                        tools JSONB DEFAULT '[]'::jsonb,
                        persona JSONB
                    """
                else:  # SHORT_TERM_MEMORY, LONG_TERM_MEMORY
                    specific_columns = """
                        content TEXT,
                        importance_score FLOAT DEFAULT 0.0,
                        access_count INTEGER DEFAULT 0,
                        last_accessed TIMESTAMP DEFAULT NOW()
                    """
                
                # Combine columns
                all_columns = base_columns
                if specific_columns:
                    all_columns += ",\n                    " + specific_columns
                
                # Create table
                create_table_sql = f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    {all_columns}
                );
                """
                
                cursor.execute(create_table_sql)
                print(f"Created table: {table_name}")

    def _create_vector_indexes(self):
        """Create vector indexes for similarity search."""
        with self.connection.cursor() as cursor:
            for memory_type in MemoryType:
                table_name = memory_type.value
                
                # Create HNSW index for vector similarity search
                index_name = f"{table_name}_embedding_idx"
                cursor.execute(f"""
                    CREATE INDEX IF NOT EXISTS {index_name} 
                    ON {table_name} 
                    USING hnsw (embedding vector_cosine_ops);
                """)
                
                # Create indexes on commonly queried fields
                cursor.execute(f"CREATE INDEX IF NOT EXISTS {table_name}_name_idx ON {table_name} (name);")
                
                if memory_type != MemoryType.PERSONAS:
                    cursor.execute(f"CREATE INDEX IF NOT EXISTS {table_name}_memory_id_idx ON {table_name} (memory_id);")
                
                if memory_type == MemoryType.CONVERSATION_MEMORY:
                    cursor.execute(f"CREATE INDEX IF NOT EXISTS {table_name}_timestamp_idx ON {table_name} (timestamp);")
                    cursor.execute(f"CREATE INDEX IF NOT EXISTS {table_name}_conversation_id_idx ON {table_name} (conversation_id);")
                
                print(f"Created indexes for table: {table_name}")

    def store(self, data: Dict[str, Any], memory_store_type: MemoryType) -> str:
        """
        Store data in PostgreSQL.
        
        Parameters:
        -----------
        data : Dict[str, Any]
            The document to be stored.
        memory_store_type : MemoryType
            The type of memory store.
        
        Returns:
        --------
        str
            The ID of the inserted/updated document.
        """
        table_name = memory_store_type.value
        data_copy = data.copy()
        
        # Handle embedding
        if 'embedding' in data_copy and isinstance(data_copy['embedding'], list):
            data_copy['embedding'] = np.array(data_copy['embedding'])
        
        # Set updated_at timestamp
        data_copy['updated_at'] = datetime.now()
        
        with self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            # Check if document exists (by id or name)
            existing_id = None
            if 'id' in data_copy:
                cursor.execute(f"SELECT id FROM {table_name} WHERE id = %s", (data_copy['id'],))
                result = cursor.fetchone()
                if result:
                    existing_id = result['id']
            elif 'name' in data_copy:
                cursor.execute(f"SELECT id FROM {table_name} WHERE name = %s", (data_copy['name'],))
                result = cursor.fetchone()
                if result:
                    existing_id = result['id']
            
            if existing_id:
                # Update existing record
                set_clauses = []
                values = []
                for key, value in data_copy.items():
                    if key != 'id':
                        set_clauses.append(f"{key} = %s")
                        if isinstance(value, (dict, list)):
                            values.append(json.dumps(value))
                        else:
                            values.append(value)
                
                values.append(existing_id)
                update_sql = f"UPDATE {table_name} SET {', '.join(set_clauses)} WHERE id = %s RETURNING id"
                cursor.execute(update_sql, values)
                return str(existing_id)
            else:
                # Insert new record
                if 'id' not in data_copy:
                    data_copy['id'] = str(uuid.uuid4())
                
                columns = list(data_copy.keys())
                placeholders = ['%s'] * len(columns)
                values = []
                
                for key in columns:
                    value = data_copy[key]
                    if isinstance(value, (dict, list)) and key != 'embedding':
                        values.append(json.dumps(value))
                    else:
                        values.append(value)
                
                insert_sql = f"""
                    INSERT INTO {table_name} ({', '.join(columns)}) 
                    VALUES ({', '.join(placeholders)}) 
                    RETURNING id
                """
                cursor.execute(insert_sql, values)
                result = cursor.fetchone()
                return str(result['id'])

    def retrieve_by_query(self, query: Dict[str, Any], memory_store_type: MemoryType, limit: int = 1) -> Optional[Dict[str, Any]]:
        """
        Retrieve documents from PostgreSQL by query.
        
        Parameters:
        -----------
        query : Dict[str, Any]
            The query to use for retrieval.
        memory_store_type : MemoryType
            The type of memory store.
        limit : int
            The maximum number of documents to return.
        
        Returns:
        --------
        Optional[Dict[str, Any]]
            The retrieved document(s), or None if not found.
        """
        table_name = memory_store_type.value
        
        with self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            # Handle vector search if embedding is provided
            if 'embedding' in query:
                embedding = np.array(query['embedding'])
                
                # Build filter conditions
                where_conditions = []
                values = [embedding]
                
                for key, value in query.items():
                    if key != 'embedding':
                        where_conditions.append(f"{key} = %s")
                        values.append(value)
                
                where_clause = ""
                if where_conditions:
                    where_clause = f"WHERE {' AND '.join(where_conditions)}"
                
                vector_sql = f"""
                    SELECT *, (embedding <=> %s) as distance 
                    FROM {table_name} 
                    {where_clause}
                    ORDER BY embedding <=> %s 
                    LIMIT %s
                """
                values.append(embedding)
                values.append(limit)
                cursor.execute(vector_sql, values)
            else:
                # Regular query
                where_conditions = []
                values = []
                
                for key, value in query.items():
                    where_conditions.append(f"{key} = %s")
                    values.append(value)
                
                where_clause = f"WHERE {' AND '.join(where_conditions)}" if where_conditions else ""
                values.append(limit)
                
                regular_sql = f"SELECT * FROM {table_name} {where_clause} LIMIT %s"
                cursor.execute(regular_sql, values)
            
            results = cursor.fetchall()
            
            if not results:
                return None
            
            # Convert to list of dicts and handle JSON fields
            processed_results = []
            for row in results:
                row_dict = dict(row)
                # Convert embedding back to list if present
                if 'embedding' in row_dict and row_dict['embedding'] is not None:
                    row_dict['embedding'] = row_dict['embedding'].tolist()
                processed_results.append(row_dict)
            
            return processed_results if limit > 1 else processed_results[0]

    def retrieve_by_id(self, id: str, memory_store_type: MemoryType) -> Optional[Dict[str, Any]]:
        """Retrieve a document from PostgreSQL by id."""
        table_name = memory_store_type.value
        
        with self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            cursor.execute(f"SELECT * FROM {table_name} WHERE id = %s", (id,))
            result = cursor.fetchone()
            
            if result:
                row_dict = dict(result)
                # Convert embedding back to list if present
                if 'embedding' in row_dict and row_dict['embedding'] is not None:
                    row_dict['embedding'] = row_dict['embedding'].tolist()
                return row_dict
            return None

    def retrieve_by_name(self, name: str, memory_store_type: MemoryType) -> Optional[Dict[str, Any]]:
        """Retrieve a document from PostgreSQL by name."""
        table_name = memory_store_type.value
        
        with self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            # Exclude embedding from results for performance
            cursor.execute(f"SELECT * FROM {table_name} WHERE name = %s", (name,))
            result = cursor.fetchone()
            
            if result:
                row_dict = dict(result)
                # Remove embedding for performance (as done in MongoDB version)
                row_dict.pop('embedding', None)
                return row_dict
            return None

    def delete_by_id(self, id: str, memory_store_type: MemoryType) -> bool:
        """Delete a document from PostgreSQL by id."""
        table_name = memory_store_type.value
        
        with self.connection.cursor() as cursor:
            cursor.execute(f"DELETE FROM {table_name} WHERE id = %s", (id,))
            return cursor.rowcount > 0

    def delete_by_name(self, name: str, memory_store_type: MemoryType) -> bool:
        """Delete a document from PostgreSQL by name."""
        table_name = memory_store_type.value
        
        with self.connection.cursor() as cursor:
            cursor.execute(f"DELETE FROM {table_name} WHERE name = %s", (name,))
            return cursor.rowcount > 0

    def delete_all(self, memory_store_type: MemoryType) -> bool:
        """Delete all documents within a memory store type."""
        table_name = memory_store_type.value
        
        with self.connection.cursor() as cursor:
            cursor.execute(f"DELETE FROM {table_name}")
            return True

    def list_all(self, memory_store_type: MemoryType) -> List[Dict[str, Any]]:
        """List all documents within a memory store type."""
        table_name = memory_store_type.value
        
        with self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            # Exclude embedding for performance
            cursor.execute(f"SELECT * FROM {table_name} ORDER BY created_at DESC")
            results = cursor.fetchall()
            
            processed_results = []
            for row in results:
                row_dict = dict(row)
                # Remove embedding for performance
                row_dict.pop('embedding', None)
                processed_results.append(row_dict)
            
            return processed_results

    def retrieve_conversation_history_ordered_by_timestamp(self, memory_id: str) -> List[Dict[str, Any]]:
        """Retrieve conversation history ordered by timestamp."""
        with self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            cursor.execute("""
                SELECT * FROM conversation_memory 
                WHERE memory_id = %s 
                ORDER BY timestamp ASC
            """, (memory_id,))
            
            results = cursor.fetchall()
            return [dict(row) for row in results]

    def update_by_id(self, id: str, data: Dict[str, Any], memory_store_type: MemoryType) -> bool:
        """Update a document in PostgreSQL by id."""
        table_name = memory_store_type.value
        data_copy = data.copy()
        data_copy['updated_at'] = datetime.now()
        
        # Handle embedding
        if 'embedding' in data_copy and isinstance(data_copy['embedding'], list):
            data_copy['embedding'] = np.array(data_copy['embedding'])
        
        with self.connection.cursor() as cursor:
            set_clauses = []
            values = []
            
            for key, value in data_copy.items():
                set_clauses.append(f"{key} = %s")
                if isinstance(value, (dict, list)) and key != 'embedding':
                    values.append(json.dumps(value))
                else:
                    values.append(value)
            
            values.append(id)
            update_sql = f"UPDATE {table_name} SET {', '.join(set_clauses)} WHERE id = %s"
            cursor.execute(update_sql, values)
            return cursor.rowcount > 0

    def close(self) -> None:
        """Close the connection to PostgreSQL."""
        if self.connection:
            self.connection.close()

    # MemAgent specific methods
    def store_memagent(self, memagent: "MemAgentModel") -> str:
        """Store a memagent in PostgreSQL."""
        memagent_dict = memagent.model_dump()
        
        # Convert persona to a serializable format if it exists
        if memagent.persona:
            memagent_dict["persona"] = memagent.persona.to_dict()
        
        # Remove any function objects from tools
        if memagent_dict.get("tools") and isinstance(memagent_dict["tools"], list):
            for tool in memagent_dict["tools"]:
                if "function" in tool and callable(tool["function"]):
                    del tool["function"]
        
        return self.store(memagent_dict, MemoryType.MEMAGENT)

    def delete_memagent(self, agent_id: str, cascade: bool = False) -> bool:
        """Delete a memagent from PostgreSQL."""
        if cascade:
            # Get the memagent first
            memagent_data = self.retrieve_by_id(agent_id, MemoryType.MEMAGENT)
            if memagent_data and 'memory_ids' in memagent_data:
                # Delete all associated memory components
                for memory_id in memagent_data['memory_ids']:
                    for memory_type in MemoryType:
                        if memory_type != MemoryType.MEMAGENT:
                            with self.connection.cursor() as cursor:
                                cursor.execute(f"DELETE FROM {memory_type.value} WHERE memory_id = %s", (memory_id,))
        
        return self.delete_by_id(agent_id, MemoryType.MEMAGENT)

    def update_memagent_memory_ids(self, agent_id: str, memory_ids: List[str]) -> bool:
        """Update the memory_ids of a memagent."""
        return self.update_by_id(agent_id, {"memory_ids": memory_ids}, MemoryType.MEMAGENT)

    def delete_memagent_memory_ids(self, agent_id: str) -> bool:
        """Delete the memory_ids of a memagent."""
        return self.update_by_id(agent_id, {"memory_ids": []}, MemoryType.MEMAGENT)

    def list_memagents(self) -> List[Dict[str, Any]]:
        """List all memagents."""
        return self.list_all(MemoryType.MEMAGENT)

    def retrieve_memory_components_by_query(self, query: str = None, query_embedding: list[float] = None, 
                                          memory_id: str = None, memory_type: MemoryType = None, 
                                          limit: int = 5) -> List[Dict[str, Any]]:
        """Retrieve memory components by query with vector search."""
        if query_embedding is None and query is not None:
            query_embedding = get_embedding(query)
        
        search_query = {"embedding": query_embedding}
        if memory_id:
            search_query["memory_id"] = memory_id
        
        return self.retrieve_by_query(search_query, memory_type, limit=limit) 
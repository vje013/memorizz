import os
import getpass
import inspect
import json
import uuid
from functools import wraps
from typing import get_type_hints, List, Dict, Any, Optional
import psycopg2
import psycopg2.extras
from psycopg2.extensions import register_adapter, AsIs
import numpy as np
import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress httpx logs to reduce noise from API requests
logging.getLogger("httpx").setLevel(logging.WARNING)

# Register numpy array adapter for PostgreSQL
def adapt_numpy_array(numpy_array):
    return AsIs(f"'{numpy_array.tolist()}'::vector")

register_adapter(np.ndarray, adapt_numpy_array)

@dataclass
class PostgreSQLToolsConfig:
    host: str = "localhost"
    port: int = 5432
    database: str = "function_calling_db"
    user: str = "postgres"
    password: Optional[str] = None
    table_name: str = "tools"
    vector_search_candidates: int = 150
    vector_index_name: str = "vector_index"
    get_embedding: callable = None


class PostgreSQLTools:
    def __init__(self, config: PostgreSQLToolsConfig = PostgreSQLToolsConfig()):
        self.config = config
        if not self.config.get_embedding:
            raise ValueError("get_embedding function is not provided")
        
        if self.config.password is None:
            self.config.password = os.getenv('POSTGRES_PASSWORD') or getpass.getpass("Enter PostgreSQL password: ")
        
        self.connection = None
        self.embedding_dimensions = None
        
        try:
            # Connect to PostgreSQL
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
            
            # Get embedding dimensions
            test_embedding = self.config.get_embedding("test")
            self.embedding_dimensions = len(test_embedding)
            
            # Ensure tools table exists
            self._ensure_tools_table()
            
            # Ensure vector index exists
            self._ensure_vector_index()
            
            logger.info("PostgreSQLTools initialized successfully.")
            
        except psycopg2.Error as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during PostgreSQL initialization: {str(e)}")
            raise
        
        if self.connection is None:
            logger.warning("PostgreSQLTools initialization failed. Some features may not work.")

    def _ensure_tools_table(self):
        """Create the tools table if it doesn't exist."""
        with self.connection.cursor() as cursor:
            create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS {self.config.table_name} (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                name VARCHAR(255) UNIQUE NOT NULL,
                description TEXT NOT NULL,
                parameters JSONB NOT NULL,
                embedding vector({self.embedding_dimensions}),
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            );
            """
            cursor.execute(create_table_sql)
            logger.info(f"Table '{self.config.table_name}' ensured.")

    def _ensure_vector_index(self):
        """Create vector index for similarity search if it doesn't exist."""
        with self.connection.cursor() as cursor:
            index_name = f"{self.config.table_name}_{self.config.vector_index_name}"
            
            # Check if index exists
            cursor.execute("""
                SELECT indexname FROM pg_indexes 
                WHERE tablename = %s AND indexname = %s
            """, (self.config.table_name, index_name))
            
            if not cursor.fetchone():
                # Create HNSW index for vector similarity search
                cursor.execute(f"""
                    CREATE INDEX {index_name} 
                    ON {self.config.table_name} 
                    USING hnsw (embedding vector_cosine_ops);
                """)
                logger.info(f"Vector index '{index_name}' created.")
            else:
                logger.info(f"Vector index '{index_name}' already exists.")

    def postgresql_toolbox(self, table_name: Optional[str] = None):
        """
        Decorator to register functions as tools in PostgreSQL.
        
        Args:
            table_name: Optional table name to store tools (defaults to config table_name)
        """
        if table_name is None:
            table_name = self.config.table_name

        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            signature = inspect.signature(func)
            docstring = inspect.getdoc(func) or ""

            if not docstring:
                raise ValueError(f"Error registering tool {func.__name__}: Docstring is missing. Please provide a docstring for the function.")

            type_hints = get_type_hints(func)

            tool_def = {
                "name": func.__name__,
                "description": docstring.strip(),
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }

            # Extract parameters from function signature
            for param_name, param in signature.parameters.items():
                if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                    continue

                param_type = type_hints.get(param_name, type(None))
                json_type = "string"
                if param_type in (int, float):
                    json_type = "number"
                elif param_type == bool:
                    json_type = "boolean"

                tool_def["parameters"]["properties"][param_name] = {
                    "type": json_type,
                    "description": f"Parameter {param_name}"
                }

                if param.default == inspect.Parameter.empty:
                    tool_def["parameters"]["required"].append(param_name)

            tool_def["parameters"]["additionalProperties"] = False

            try:
                # Generate embedding for the tool description
                vector = self.config.get_embedding(tool_def["description"])
                vector_array = np.array(vector)
                
                # Store tool in PostgreSQL
                with self.connection.cursor() as cursor:
                    cursor.execute(f"""
                        INSERT INTO {table_name} (name, description, parameters, embedding, updated_at)
                        VALUES (%s, %s, %s, %s, NOW())
                        ON CONFLICT (name) 
                        DO UPDATE SET 
                            description = EXCLUDED.description,
                            parameters = EXCLUDED.parameters,
                            embedding = EXCLUDED.embedding,
                            updated_at = NOW()
                    """, (
                        tool_def["name"],
                        tool_def["description"],
                        json.dumps(tool_def["parameters"]),
                        vector_array
                    ))
                
                logger.info(f"Successfully registered tool: {func.__name__}")
            except Exception as e:
                logger.error(f"Error registering tool {func.__name__}: {str(e)}")
                raise

            return wrapper
        return decorator

    def _vector_search(self, user_query: str, table_name: Optional[str] = None, limit: int = 2) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search for tools.
        
        Args:
            user_query: Query string to search for
            table_name: Optional table name (defaults to config table_name)
            limit: Maximum number of results to return
            
        Returns:
            List of matching tools
        """
        if table_name is None:
            table_name = self.config.table_name

        try:
            query_embedding = self.config.get_embedding(user_query)
            query_vector = np.array(query_embedding)
        except Exception as e:
            logger.error(f"Error generating embedding for query: {str(e)}")
            raise

        try:
            with self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute(f"""
                    SELECT 
                        id,
                        name,
                        description,
                        parameters,
                        (embedding <=> %s) as distance
                    FROM {table_name}
                    ORDER BY embedding <=> %s
                    LIMIT %s
                """, (query_vector, query_vector, limit))
                
                results = cursor.fetchall()
                
                # Convert to list of dicts and parse JSON parameters
                processed_results = []
                for row in results:
                    row_dict = dict(row)
                    # Parse JSON parameters
                    if isinstance(row_dict['parameters'], str):
                        row_dict['parameters'] = json.loads(row_dict['parameters'])
                    processed_results.append(row_dict)
                
                return processed_results
                
        except Exception as e:
            logger.error(f"Error performing vector search: {str(e)}")
            raise

    def populate_tools(self, user_query: str, num_tools: int = 2) -> List[Dict[str, Any]]:
        """
        Find and format tools based on a user query.
        
        Args:
            user_query: Query string to search for relevant tools
            num_tools: Number of tools to return
            
        Returns:
            List of formatted tools ready for LLM function calling
        """
        try:
            search_results = self._vector_search(user_query, limit=num_tools)
            tools = []
            for result in search_results:
                print(result)
                tool = {
                    "type": "function",
                    "function": {
                        "name": result["name"],
                        "description": result["description"],
                        "parameters": result["parameters"]
                    }
                }
                tools.append(tool)
            logger.info(f"Successfully populated {len(tools)} tools")
            return tools
        except Exception as e:
            logger.error(f"Error populating tools: {str(e)}")
            raise

    def create_toolbox(self, table_name: str, embedding_dimensions: Optional[int] = None, index_name: str = "vector_index"):
        """
        Create a new tools table in PostgreSQL and set up vector search index.

        Args:
            table_name: Name of the table to create
            embedding_dimensions: Dimensions for the vector column (defaults to current config)
            index_name: Name of the vector index (default: "vector_index")

        Returns:
            bool: True if the toolbox was created successfully, False otherwise
        """
        if embedding_dimensions is None:
            embedding_dimensions = self.embedding_dimensions

        try:
            with self.connection.cursor() as cursor:
                # Create the table
                create_table_sql = f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    name VARCHAR(255) UNIQUE NOT NULL,
                    description TEXT NOT NULL,
                    parameters JSONB NOT NULL,
                    embedding vector({embedding_dimensions}),
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                );
                """
                cursor.execute(create_table_sql)
                logger.info(f"Table '{table_name}' created successfully.")

                # Create the vector search index
                full_index_name = f"{table_name}_{index_name}"
                cursor.execute(f"""
                    CREATE INDEX IF NOT EXISTS {full_index_name}
                    ON {table_name} 
                    USING hnsw (embedding vector_cosine_ops);
                """)
                logger.info(f"Vector search index '{full_index_name}' created successfully for table '{table_name}'.")

                # Update the config to use the new table
                self.config.table_name = table_name
                self.config.vector_index_name = index_name

                return True

        except psycopg2.Error as e:
            logger.error(f"PostgreSQL error creating toolbox: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Error creating toolbox: {str(e)}")
            return False

    def list_tools(self, table_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all tools in the specified table.
        
        Args:
            table_name: Optional table name (defaults to config table_name)
            
        Returns:
            List of all tools
        """
        if table_name is None:
            table_name = self.config.table_name

        try:
            with self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute(f"""
                    SELECT id, name, description, parameters, created_at, updated_at
                    FROM {table_name}
                    ORDER BY created_at DESC
                """)
                
                results = cursor.fetchall()
                
                # Convert to list of dicts and parse JSON parameters
                processed_results = []
                for row in results:
                    row_dict = dict(row)
                    if isinstance(row_dict['parameters'], str):
                        row_dict['parameters'] = json.loads(row_dict['parameters'])
                    processed_results.append(row_dict)
                
                return processed_results
                
        except Exception as e:
            logger.error(f"Error listing tools: {str(e)}")
            raise

    def delete_tool(self, tool_name: str, table_name: Optional[str] = None) -> bool:
        """
        Delete a tool by name.
        
        Args:
            tool_name: Name of the tool to delete
            table_name: Optional table name (defaults to config table_name)
            
        Returns:
            bool: True if tool was deleted, False otherwise
        """
        if table_name is None:
            table_name = self.config.table_name

        try:
            with self.connection.cursor() as cursor:
                cursor.execute(f"DELETE FROM {table_name} WHERE name = %s", (tool_name,))
                deleted = cursor.rowcount > 0
                if deleted:
                    logger.info(f"Tool '{tool_name}' deleted successfully.")
                else:
                    logger.warning(f"Tool '{tool_name}' not found.")
                return deleted
        except Exception as e:
            logger.error(f"Error deleting tool {tool_name}: {str(e)}")
            return False

    def get_tool(self, tool_name: str, table_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get a specific tool by name.
        
        Args:
            tool_name: Name of the tool to retrieve
            table_name: Optional table name (defaults to config table_name)
            
        Returns:
            Tool dictionary or None if not found
        """
        if table_name is None:
            table_name = self.config.table_name

        try:
            with self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute(f"""
                    SELECT id, name, description, parameters, created_at, updated_at
                    FROM {table_name}
                    WHERE name = %s
                """, (tool_name,))
                
                result = cursor.fetchone()
                if result:
                    row_dict = dict(result)
                    if isinstance(row_dict['parameters'], str):
                        row_dict['parameters'] = json.loads(row_dict['parameters'])
                    return row_dict
                return None
                
        except Exception as e:
            logger.error(f"Error getting tool {tool_name}: {str(e)}")
            return None

    def close(self):
        """Close the PostgreSQL connection."""
        if self.connection:
            self.connection.close()
            logger.info("PostgreSQL connection closed.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


__all__ = ['PostgreSQLTools', 'PostgreSQLToolsConfig']

# You can create a function to get the postgresql_toolbox decorator:
def get_postgresql_toolbox(config: PostgreSQLToolsConfig = PostgreSQLToolsConfig()):
    """
    Get a PostgreSQL toolbox decorator for registering functions as tools.
    
    Args:
        config: PostgreSQL tools configuration
        
    Returns:
        Decorator function for registering tools
    """
    return PostgreSQLTools(config).postgresql_toolbox 
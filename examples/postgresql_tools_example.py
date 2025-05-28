#!/usr/bin/env python3
"""
PostgreSQL Tools Example

This example demonstrates how to use PostgreSQL tools for function registration,
vector search, and toolbox management - equivalent to MongoDB tools functionality.

Prerequisites:
1. PostgreSQL with pgvector extension running
2. Database 'function_calling_db' created
3. Environment variable POSTGRES_PASSWORD set or will prompt for password
4. OpenAI API key for embeddings

Usage:
    python examples/postgresql_tools_example.py
"""

import os
import sys
from typing import List, Dict, Any

# Add the src directory to the path so we can import memorizz
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from memorizz.database.postgresql import PostgreSQLTools, PostgreSQLToolsConfig

# Mock embedding function for demonstration
# In real usage, you'd use OpenAI or another embedding service
def mock_get_embedding(text: str) -> List[float]:
    """Mock embedding function that returns a simple hash-based vector."""
    import hashlib
    # Create a simple deterministic embedding based on text hash
    hash_obj = hashlib.md5(text.encode())
    hash_hex = hash_obj.hexdigest()
    
    # Convert hex to numbers and normalize to create a 1536-dimension vector
    embedding = []
    for i in range(1536):
        # Use different parts of the hash to create variation
        char_index = i % len(hash_hex)
        embedding.append((ord(hash_hex[char_index]) - 48) / 255.0)  # Normalize to 0-1
    
    return embedding

def real_openai_embedding(text: str) -> List[float]:
    """Real OpenAI embedding function (requires OpenAI API key)."""
    try:
        from openai import OpenAI
        client = OpenAI()  # Uses OPENAI_API_KEY environment variable
        
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except ImportError:
        print("OpenAI library not installed. Using mock embeddings.")
        return mock_get_embedding(text)
    except Exception as e:
        print(f"Error with OpenAI embeddings: {e}. Using mock embeddings.")
        return mock_get_embedding(text)

def main():
    print("🐘 PostgreSQL Tools Example")
    print("=" * 50)
    
    # Choose embedding function
    use_real_embeddings = os.getenv('OPENAI_API_KEY') is not None
    embedding_func = real_openai_embedding if use_real_embeddings else mock_get_embedding
    
    if use_real_embeddings:
        print("✅ Using OpenAI embeddings")
    else:
        print("⚠️  Using mock embeddings (set OPENAI_API_KEY for real embeddings)")
    
    # Configure PostgreSQL tools
    config = PostgreSQLToolsConfig(
        host="localhost",
        port=5432,
        database="function_calling_db",
        user="postgres",
        # password will be prompted or read from POSTGRES_PASSWORD env var
        table_name="example_tools",
        get_embedding=embedding_func
    )
    
    # Initialize PostgreSQL tools
    print("\n📊 Initializing PostgreSQL Tools...")
    try:
        with PostgreSQLTools(config) as pg_tools:
            print("✅ Connected to PostgreSQL successfully!")
            
            # Get the decorator for registering tools
            postgresql_toolbox = pg_tools.postgresql_toolbox()
            
            # Register some example tools
            print("\n🔧 Registering example tools...")
            
            @postgresql_toolbox
            def calculate_area(length: float, width: float) -> float:
                """Calculate the area of a rectangle given length and width."""
                return length * width
            
            @postgresql_toolbox
            def get_weather(city: str, country: str = "US") -> str:
                """Get current weather information for a specific city and country."""
                return f"Weather in {city}, {country}: Sunny, 72°F"
            
            @postgresql_toolbox
            def send_email(to: str, subject: str, body: str) -> bool:
                """Send an email to a recipient with the specified subject and body."""
                print(f"Email sent to {to}: {subject}")
                return True
            
            @postgresql_toolbox
            def search_database(query: str, table: str, limit: int = 10) -> List[Dict]:
                """Search a database table with a query and return limited results."""
                return [{"id": 1, "data": f"Result for {query} in {table}"}]
            
            print("✅ Tools registered successfully!")
            
            # List all tools
            print("\n📋 Listing all registered tools:")
            tools = pg_tools.list_tools()
            for tool in tools:
                print(f"  • {tool['name']}: {tool['description'][:60]}...")
            
            # Test vector search functionality
            print("\n🔍 Testing vector search...")
            
            test_queries = [
                "I need to calculate the size of a room",
                "What's the weather like today?",
                "Send a message to someone",
                "Find information in a database"
            ]
            
            for query in test_queries:
                print(f"\n🔎 Query: '{query}'")
                relevant_tools = pg_tools.populate_tools(query, num_tools=2)
                
                print("   📦 Relevant tools:")
                for tool in relevant_tools:
                    func_info = tool['function']
                    print(f"     • {func_info['name']}: {func_info['description'][:50]}...")
            
            # Test getting a specific tool
            print("\n🎯 Getting specific tool...")
            weather_tool = pg_tools.get_tool("get_weather")
            if weather_tool:
                print(f"   Found tool: {weather_tool['name']}")
                print(f"   Description: {weather_tool['description']}")
                print(f"   Parameters: {weather_tool['parameters']}")
            
            # Test creating a new toolbox
            print("\n🏗️  Creating a new toolbox...")
            success = pg_tools.create_toolbox("math_tools", index_name="math_vector_index")
            if success:
                print("✅ New toolbox 'math_tools' created successfully!")
                
                # Register a tool in the new toolbox
                math_toolbox = pg_tools.postgresql_toolbox("math_tools")
                
                @math_toolbox
                def calculate_circle_area(radius: float) -> float:
                    """Calculate the area of a circle given its radius."""
                    import math
                    return math.pi * radius ** 2
                
                print("✅ Math tool registered in new toolbox!")
                
                # List tools in the new toolbox
                math_tools = pg_tools.list_tools("math_tools")
                print(f"   Math toolbox contains {len(math_tools)} tools")
            
            # Test tool deletion
            print("\n🗑️  Testing tool deletion...")
            deleted = pg_tools.delete_tool("send_email")
            if deleted:
                print("✅ Tool 'send_email' deleted successfully!")
            else:
                print("❌ Failed to delete tool 'send_email'")
            
            print("\n🎉 PostgreSQL Tools example completed successfully!")
            print("\n💡 Key features demonstrated:")
            print("   • Function registration with automatic parameter extraction")
            print("   • Vector similarity search for tool discovery")
            print("   • Multiple toolboxes (tables) support")
            print("   • CRUD operations on tools")
            print("   • Automatic vector indexing with pgvector")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n🔧 Troubleshooting:")
        print("   • Ensure PostgreSQL is running with pgvector extension")
        print("   • Check database connection parameters")
        print("   • Verify database 'function_calling_db' exists")
        print("   • Set POSTGRES_PASSWORD environment variable")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 
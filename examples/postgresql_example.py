#!/usr/bin/env python3
"""
Example demonstrating the PostgreSQL memory provider with pgvector.

This example shows how to:
1. Set up a PostgreSQL memory provider
2. Store and retrieve different types of memory
3. Perform vector similarity searches
4. Manage MemAgents with PostgreSQL backend

Prerequisites:
- PostgreSQL with pgvector extension running
- Run: docker-compose up -d to start the local PostgreSQL instance
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from memorizz.memory_provider.postgresql import PostgreSQLProvider, PostgreSQLConfig
from memorizz.memory_provider.memory_type import MemoryType
from memorizz.embeddings.openai import get_embedding
import json

def main():
    print("🚀 PostgreSQL Memory Provider Example")
    print("=" * 50)
    
    # Configure PostgreSQL connection
    config = PostgreSQLConfig(
        host="localhost",
        port=5432,
        database="memorizz",
        user="postgres",
        password="password"
    )
    
    try:
        # Initialize the PostgreSQL provider
        print("📡 Connecting to PostgreSQL...")
        provider = PostgreSQLProvider(config)
        print("✅ Connected successfully!")
        
        # Example 1: Store a persona
        print("\n📝 Example 1: Storing a Persona")
        persona_data = {
            "name": "AI Assistant",
            "role": "helpful_assistant",
            "goals": "Help users with their questions and tasks",
            "background": "I am an AI assistant designed to be helpful, harmless, and honest.",
            "embedding": get_embedding("AI assistant helpful harmless honest")
        }
        
        persona_id = provider.store(persona_data, MemoryType.PERSONAS)
        print(f"✅ Stored persona with ID: {persona_id}")
        
        # Example 2: Store conversation memory
        print("\n💬 Example 2: Storing Conversation Memory")
        conversation_data = {
            "content": "Hello, how can I help you today?",
            "role": "assistant",
            "memory_id": "conv_001",
            "conversation_id": "chat_123",
            "embedding": get_embedding("Hello, how can I help you today?")
        }
        
        conv_id = provider.store(conversation_data, MemoryType.CONVERSATION_MEMORY)
        print(f"✅ Stored conversation with ID: {conv_id}")
        
        # Example 3: Vector similarity search
        print("\n🔍 Example 3: Vector Similarity Search")
        search_query = "greeting and assistance"
        search_embedding = get_embedding(search_query)
        
        # Search in conversation memory
        results = provider.retrieve_by_query(
            {"embedding": search_embedding, "memory_id": "conv_001"}, 
            MemoryType.CONVERSATION_MEMORY,
            limit=5
        )
        
        if results:
            print(f"✅ Found similar conversation:")
            print(f"   Content: {results['content']}")
            print(f"   Distance: {results.get('distance', 'N/A')}")
        else:
            print("❌ No similar conversations found")
        
        # Example 4: Store and retrieve by name
        print("\n🏷️  Example 4: Retrieve by Name")
        retrieved_persona = provider.retrieve_by_name("AI Assistant", MemoryType.PERSONAS)
        if retrieved_persona:
            print(f"✅ Retrieved persona: {retrieved_persona['name']}")
            print(f"   Role: {retrieved_persona['role']}")
        else:
            print("❌ Persona not found")
        
        # Example 5: List all items in a memory store
        print("\n📋 Example 5: List All Personas")
        all_personas = provider.list_all(MemoryType.PERSONAS)
        print(f"✅ Found {len(all_personas)} personas:")
        for persona in all_personas:
            print(f"   - {persona['name']} ({persona['role']})")
        
        # Example 6: Store long-term memory
        print("\n🧠 Example 6: Storing Long-term Memory")
        ltm_data = {
            "content": "User prefers concise explanations and examples",
            "importance_score": 0.8,
            "memory_id": "user_001",
            "embedding": get_embedding("User prefers concise explanations and examples")
        }
        
        ltm_id = provider.store(ltm_data, MemoryType.LONG_TERM_MEMORY)
        print(f"✅ Stored long-term memory with ID: {ltm_id}")
        
        # Example 7: Update memory
        print("\n✏️  Example 7: Updating Memory")
        update_success = provider.update_by_id(
            ltm_id, 
            {"importance_score": 0.9, "access_count": 1}, 
            MemoryType.LONG_TERM_MEMORY
        )
        print(f"✅ Update successful: {update_success}")
        
        # Example 8: Conversation history
        print("\n📚 Example 8: Conversation History")
        # Add more conversation entries
        for i, message in enumerate([
            "What's the weather like?",
            "I can help you check the weather. What's your location?",
            "I'm in San Francisco",
            "The weather in San Francisco is currently sunny and 72°F"
        ]):
            role = "user" if i % 2 == 0 else "assistant"
            conv_data = {
                "content": message,
                "role": role,
                "memory_id": "conv_001",
                "conversation_id": "chat_123",
                "embedding": get_embedding(message)
            }
            provider.store(conv_data, MemoryType.CONVERSATION_MEMORY)
        
        # Retrieve conversation history
        history = provider.retrieve_conversation_history_ordered_by_timestamp("conv_001")
        print(f"✅ Retrieved {len(history)} conversation entries:")
        for entry in history:
            print(f"   {entry['role']}: {entry['content']}")
        
        print("\n🎉 All examples completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        if 'provider' in locals():
            provider.close()
            print("\n🔌 Connection closed")

if __name__ == "__main__":
    main() 
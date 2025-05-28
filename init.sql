-- Initialize pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create a test to verify pgvector is working
SELECT vector_dims('[1,2,3]'::vector); 
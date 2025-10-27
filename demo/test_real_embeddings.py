#!/usr/bin/env python3
"""
Test script using actual LLM embeddings with sentence-transformers
"""

import os
import sys
import pandas as pd

# Add src to path
sys.path.insert(0, 'src')

from pgvector_db.utils import DBConfigLocal, download_llm
from pgvector_db.generate_embeddings import EmbeddingGenerator
from pgvector_db.pg_insert import pg_insert

def test_real_embeddings():
    """Test with actual sentence-transformers embeddings"""
    
    print("🤖 Testing with real LLM embeddings...")
    
    # Step 1: Download/setup the model
    print("📥 Setting up sentence-transformers model...")
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    
    try:
        # Download model if not exists
        os.makedirs("./models", exist_ok=True)
        model_path = download_llm(
            model_name=model_name,
            save_path="./models",
            model_type="sentence-transformers"
        )
        print(f"✅ Model ready at: {model_path}")
    except Exception as e:
        print(f"❌ Model setup failed: {e}")
        return False
    
    # Step 2: Initialize embedding generator
    print("🔧 Initializing embedding generator...")
    try:
        generator = EmbeddingGenerator(
            model_path=model_path,  # Use the actual path returned from download_llm
            model_type="sentence-transformers",
            batch_size=32,
            device="mps"  # Use Apple Silicon GPU if available
        )
        print("✅ Embedding generator ready")
    except Exception as e:
        print(f"❌ Generator initialization failed: {e}")
        return False
    
    # Step 3: Generate embeddings for test texts
    test_texts = [
        "PostgreSQL is a powerful, open source object-relational database system.",
        "Machine learning models can generate vector embeddings from text.",
        "pgvector extension enables similarity search in PostgreSQL.",
        "Python and uv make package management simple and fast.",
        "Vector databases are essential for AI applications."
    ]
    
    print(f"📝 Generating embeddings for {len(test_texts)} texts...")
    try:
        embeddings = generator.generate_embeddings(test_texts)
        print(f"✅ Generated embeddings with shape: {embeddings.shape}")
        print(f"🎯 Embedding dimension: {embeddings.shape[1]}")
    except Exception as e:
        print(f"❌ Embedding generation failed: {e}")
        return False
    
    # Step 4: Prepare data for PostgreSQL
    dataset = pd.DataFrame({
        'text': test_texts,
        'embedding': [emb.tolist() for emb in embeddings]  # Convert numpy arrays to lists
    })
    
    # Step 5: Configure database and insert
    password = os.environ.get('PG_PASSWORD')
    if not password:
        raise ValueError("PG_PASSWORD environment variable must be set")
    db_config = DBConfigLocal(
        db_name="vector_db",
        db_user="py_pg_user",
        db_password=password,
        db_host="localhost",
        schema_name="public",
        table_name="documents"
    )
    
    print("💾 Inserting real embeddings into PostgreSQL...")
    try:
        pg_insert(dataset, db_config, batch_size=100)
        print("✅ SUCCESS: Real LLM embeddings inserted successfully!")
        return True
    except Exception as e:
        print(f"❌ Database insert failed: {e}")
        return False

if __name__ == "__main__":
    success = test_real_embeddings()
    sys.exit(0 if success else 1)
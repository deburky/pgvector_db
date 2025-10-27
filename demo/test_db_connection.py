#!/usr/bin/env python3
"""
Simple test script to verify PostgreSQL connection and pgvector_db package functionality
"""

import os
import sys
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, 'src')

from pgvector_db.utils import DBConfigLocal
from pgvector_db.pg_insert import pg_insert

def test_connection():
    """Test basic database connection and insert functionality"""
    
    # Get password from environment
    password = os.environ.get('PG_PASSWORD')
    if not password:
        raise ValueError("PG_PASSWORD environment variable must be set")
    
    # Configure database connection
    db_config = DBConfigLocal(
        db_name="vector_db",
        db_user="py_pg_user", 
        db_password=password,
        db_host="localhost",
        schema_name="public",
        table_name="documents"
    )
    
    # Create test data with simple embeddings
    test_data = pd.DataFrame({
        'text': [
            'This is a test document',
            'Another test document for vector storage',
            'PostgreSQL with pgvector is awesome'
        ],
        'embedding': [
            np.random.rand(384).tolist(),  # Random 384-dimensional vector
            np.random.rand(384).tolist(),
            np.random.rand(384).tolist()
        ]
    })
    
    print("🔍 Testing database connection and insert...")
    print(f"📊 Test data shape: {test_data.shape}")
    print(f"🎯 Embedding dimension: {len(test_data['embedding'][0])}")
    
    try:
        # Test the insert functionality
        pg_insert(test_data, db_config, batch_size=10)
        print("✅ SUCCESS: Data inserted successfully!")
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_connection()
    sys.exit(0 if success else 1)
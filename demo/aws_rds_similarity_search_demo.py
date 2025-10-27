#!/usr/bin/env python3
"""
Demonstration of advanced vector similarity search with AWS RDS PostgreSQL
"""

import os
import sys
from time import time

import pandas as pd
import psycopg2

from pgvector_db.generate_embeddings import EmbeddingGenerator
from pgvector_db.pg_insert import pg_insert
from pgvector_db.utils import DBConfigRDS, download_llm


def demo_rds_similarity_search():  # sourcery skip: extract-duplicate-method
    """Demonstrate advanced semantic similarity search on AWS RDS PostgreSQL with pgvector."""

    print("AWS RDS Vector Similarity Search Demo")
    print("=" * 60)

    # Database configuration
    password = os.environ.get("ADMIN_PASS")
    if not password:
        raise ValueError("ADMIN_PASS environment variable must be set")
    db_config = DBConfigRDS(
        db_name="postgres",
        db_user="admin_user",
        db_password=password,
        db_host="pgvector-db.cqalgqrrs8us.us-east-1.rds.amazonaws.com",
        db_port="5432",
        schema_name="public",
        table_name="documents",
    )

    # Step 1: Download/setup the model (if needed)
    print("\nSetting up sentence-transformers model...")
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    try:
        # Download model if not exists
        os.makedirs("./models", exist_ok=True)
        model_path = download_llm(
            model_name=model_name,
            save_path="./models",
            model_type="sentence-transformers",
        )
        print(f"Model ready at: {model_path}")
    except Exception as e:
        print(f"Model setup failed: {e}")
        # Fall back to direct initialization
        model_path = model_name
        print(f"Falling back to direct model initialization: {model_path}")

    # Step 2: Initialize embedding generator
    print("\nInitializing embedding generator...")
    try:
        generator = EmbeddingGenerator(
            model_path=model_path,
            model_type="sentence-transformers",
            batch_size=32,
            device="cpu",  # Use CPU for compatibility
        )
        print("Embedding generator ready")
    except Exception as e:
        print(f"Generator initialization failed: {e}")
        return False

    # Step 3: Connect to RDS
    try:
        # Connect to database
        conn = psycopg2.connect(
            dbname=db_config.db_name,
            user=db_config.db_user,
            password=db_config.db_password,
            host=db_config.db_host,
            port=db_config.db_port,
        )
        cursor = conn.cursor()

        # Check if pgvector extension is installed
        cursor.execute("SELECT extname FROM pg_extension WHERE extname = 'vector';")
        if cursor.fetchone():
            print("pgvector extension is installed")
        else:
            print("pgvector extension is NOT installed")
            try:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                conn.commit()
                print("Successfully installed pgvector extension")
            except Exception as e:
                print(f"Could not install pgvector extension: {str(e)}")
                return False

        # Step 4: Generate embeddings for test texts - using the same texts as in test_real_embeddings.py
        test_texts = [
            "PostgreSQL is a powerful, open source object-relational database system.",
            "Machine learning models can generate vector embeddings from text.",
            "pgvector extension enables similarity search in PostgreSQL.",
            "Python and uv make package management simple and fast.",
            "Vector databases are essential for AI applications.",
        ]

        print(f"\n📝 Generating embeddings for {len(test_texts)} texts...")
        embeddings = generator.generate_embeddings(test_texts)
        print(f"Generated embeddings with shape: {embeddings.shape}")
        print(f"Embedding dimension: {embeddings.shape[1]}")

        # Prepare data for insertion
        df_to_insert = pd.DataFrame(
            {"text": test_texts, "embedding": [emb.tolist() for emb in embeddings]}
        )

        # Check if table exists, create if not
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id SERIAL PRIMARY KEY,
            text TEXT,
            embedding vector(384)
        );
        """)
        conn.commit()

        # Step 5: Insert embeddings into database
        print("\nInserting embeddings into PostgreSQL...")
        try:
            pg_insert(df_to_insert, db_config, batch_size=100)
            print("SUCCESS: LLM embeddings inserted successfully!")

            # Create index for better performance (if it doesn't exist)
            try:
                print("\nCreating vector index for better performance...")
                cursor.execute("""
                CREATE INDEX IF NOT EXISTS documents_embedding_idx 
                ON documents USING ivfflat (embedding vector_cosine_ops) 
                WITH (lists = 100);
                """)
                conn.commit()
                print("Index created successfully")
            except Exception as e:
                print(f"Index creation warning (can be ignored if index exists): {e}")
                conn.rollback()
        except Exception as e:
            print(f"Database insert failed: {e}")
            return False

        # Step 6: Demonstrate similarity search
        print("\n🔍 Performing similarity searches")

        # Use same queries as in test_real_embeddings.py
        query_texts = [
            "database management systems",
            "artificial intelligence and machine learning",
            "open source software development",
        ]

        for i, query_text in enumerate(query_texts, 1):
            print(f"\n{i}. Query: '{query_text}'")
            print("-" * 60)

            # Generate query embedding
            query_embedding = generator.generate_embeddings([query_text])[0].tolist()

            # Convert array to string representation for SQL
            vector_str = f"'[{','.join(map(str, query_embedding))}]'"

            # Execute search
            query = f"""
            SELECT 
                id, 
                text, 
                1 - (embedding <=> {vector_str}::vector) AS similarity
            FROM 
                documents
            ORDER BY 
                embedding <=> {vector_str}::vector
            LIMIT 3;
            """
            cursor.execute(query)

            results = cursor.fetchall()
            for j, (doc_id, text, similarity) in enumerate(results, 1):
                print(f"{j}. [ID:{doc_id}] {text}")
                print(f"Similarity: {similarity:.4f}")

        # Step 7: Output performance metrics
        print("\n📊 Vector Database Performance")
        print("-" * 60)

        # Count documents
        cursor.execute("SELECT COUNT(*) FROM documents;")
        doc_count = cursor.fetchone()[0]

        # Test search performance
        start_time = time()
        # Convert array to string representation for SQL
        vector_str = f"'[{','.join(map(str, query_embedding))}]'"
        query = f"""
        SELECT id FROM documents ORDER BY embedding <=> {vector_str}::vector LIMIT 10;
        """
        cursor.execute(query)
        search_time = time() - start_time

        print(f"Total documents: {doc_count}")
        print(f"Search time: {search_time:.4f} seconds")

        if doc_count > 0:
            print(f"Average time per document: {search_time / doc_count * 1000:.4f} ms")

        # Show table size
        cursor.execute("""
        SELECT pg_size_pretty(pg_total_relation_size('documents'));
        """)
        table_size = cursor.fetchone()[0]
        print(f"Table size: {table_size}")

        conn.close()
        print("\nAWS RDS Vector similarity search demo completed successfully!")
        return True

    except Exception as e:
        print(f"Demo failed: {str(e)}")
        return False


if __name__ == "__main__":
    # Prompt for password if not set in environment
    if not os.environ.get("ADMIN_PASS"):
        import getpass

        os.environ["ADMIN_PASS"] = getpass.getpass(
            "Enter RDS password for admin_user: "
        )

    success = demo_rds_similarity_search()
    sys.exit(0 if success else 1)

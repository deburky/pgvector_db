#!/usr/bin/env python3
"""
Test script to verify PostgreSQL RDS connection with pgvector on AWS
"""

import os
from time import time

import pandas as pd
import psycopg2
from sentence_transformers import SentenceTransformer

from pgvector_db.pg_insert import pg_insert
from pgvector_db.utils import DBConfigRDS


def test_aws_rds_connection():  # sourcery skip: extract-method, inline-variable
    """Test RDS database connection and pgvector functionality."""

    db_config = DBConfigRDS(
        db_name="postgres",
        db_user="admin_user",
        db_password=os.environ.get("ADMIN_PASS"),
        db_host="pgvector-db.cqalgqrrs8us.us-east-1.rds.amazonaws.com",
        db_port="5432",
        schema_name="public",
        table_name="documents",
    )

    print("📥 Loading embedding model...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    print("✅ Model loaded")

    test_texts = [
        "PostgreSQL is a powerful, open source object-relational database system.",
        "Machine learning models can generate vector embeddings from text.",
        "pgvector extension enables similarity search in PostgreSQL.",
        "Python and uv make package management simple and fast.",
        "Vector databases are essential for AI applications.",
    ]

    print(f"📝 Generating embeddings for {len(test_texts)} texts...")
    start = time()
    embeddings = model.encode(test_texts)
    print(f"✅ Generated in {time() - start:.2f} seconds")

    test_data = pd.DataFrame(
        {"text": test_texts, "embedding": [e.tolist() for e in embeddings]}
    )

    print("🔌 Testing database connection and insert...")
    try:
        pg_insert(test_data, db_config, batch_size=10)
        print("✅ Inserted data successfully into RDS!")
    except Exception as e:
        print(f"❌ Insert failed: {str(e)}")
        return False

    # Test pgvector directly via psycopg2
    print("🔍 Checking pgvector extension...")
    conn_str = f"dbname={db_config.db_name} user={db_config.db_user} password={db_config.db_password} host={db_config.db_host} port={db_config.db_port}"
    with psycopg2.connect(conn_str) as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT extname FROM pg_extension WHERE extname = 'vector';")
            # sourcery skip: no-conditionals-in-tests
            if cursor.fetchone():
                print("✅ pgvector is installed")
            else:
                print("⚠️ Installing pgvector...")
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                conn.commit()

            query_text = "database systems"
            query_emb = model.encode([query_text])[0].tolist()

            print("🔎 Running similarity search...")
            # Convert array to string representation for SQL
            vector_str = f"'[{','.join(map(str, query_emb))}]'"

            query = f"""
                SELECT id, text, 1 - (embedding <=> {vector_str}::vector) AS similarity
                FROM documents
                ORDER BY embedding <=> {vector_str}::vector
                LIMIT 3;
            """
            cursor.execute(query)

            results = cursor.fetchall()
            print("\n📊 Similarity Results:")
            # sourcery skip: no-loop-in-tests
            for r in results:
                print(f"ID={r[0]} | Score={r[2]:.4f} | {r[1]}")

    print("🎯 All tests successful!")
    return True


if __name__ == "__main__":
    if not os.environ.get("ADMIN_PASS"):
        os.environ["ADMIN_PASS"] = input("Enter admin password: ")
    test_aws_rds_connection()

#!/usr/bin/env python3
"""
Vector similarity search demo on AWS RDS PostgreSQL with pgvector
"""

import os

import pandas as pd
import psycopg2
from sentence_transformers import SentenceTransformer

from pgvector_db.pg_insert import pg_insert
from pgvector_db.utils import DBConfigRDS


def main():
    password = os.environ.get("ADMIN_PASS")
    if not password:
        raise ValueError("ADMIN_PASS environment variable must be set")
    db_config = DBConfigRDS(
        db_name="postgres",
        db_user="admin_user",
        db_password=password,
        db_host="pgvector-db.cqalgqrrs8eu.us-east-1.rds.amazonaws.com",
        db_port="5432",
        schema_name="public",
        table_name="documents",
    )

    print("📥 Loading model...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    texts = [
        "PostgreSQL is a powerful, open source object-relational database system.",
        "Machine learning models can generate vector embeddings from text.",
        "pgvector extension enables similarity search in PostgreSQL.",
        "Python and uv make package management simple and fast.",
        "Vector databases are essential for AI applications.",
    ]

    embeddings = model.encode(texts)
    df = pd.DataFrame({"text": texts, "embedding": [e.tolist() for e in embeddings]})

    print("🛠 Ensuring table exists...")
    conn = psycopg2.connect(
        host=db_config.db_host,
        dbname=db_config.db_name,
        user=db_config.db_user,
        password=db_config.db_password,
        port=db_config.db_port,
    )
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id SERIAL PRIMARY KEY,
            text TEXT,
            embedding VECTOR(384)
        );
    """)
    conn.commit()

    print("💾 Inserting embeddings...")
    pg_insert(df, db_config, batch_size=100)

    query_text = "database systems"
    query_emb = model.encode([query_text])[0].tolist()
    vector_str = f"'[{','.join(map(str, query_emb))}]'"

    print("\n🔍 Running similarity search...")
    cur.execute(f"""
        SELECT id, text,
               1 - (embedding <=> {vector_str}::vector) AS sim
        FROM documents
        ORDER BY embedding <=> {vector_str}::vector
        LIMIT 3;
    """)
    results = cur.fetchall()

    for rank, (id, text, sim) in enumerate(results, 1):
        print(f"{rank}. ID={id} | sim={sim:.4f} | {text}")

    cur.close()
    conn.close()
    print("\n🎯 Done! Semantic search is working ✅")


if __name__ == "__main__":
    main()

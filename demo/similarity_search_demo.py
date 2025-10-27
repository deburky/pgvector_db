#!/usr/bin/env python3
"""
Demonstration of vector similarity search with real examples
"""

import os
import sys

# Add src to path
sys.path.insert(0, "src")

import psycopg2

from pgvector_db.utils import DBConfigLocal


def demo_similarity_search():
    """Demonstrate semantic similarity search capabilities"""

    print("🔍 Vector Similarity Search Demo")
    print("=" * 50)

    # Database configuration
    password = os.environ.get("PG_PASSWORD")
    if not password:
        raise ValueError("PG_PASSWORD environment variable must be set")

    db_config = DBConfigLocal(
        db_name="vector_db",
        db_user="py_pg_user",
        db_password=password,
        db_host="localhost",
        schema_name="public",
        table_name="documents",
    )

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

        # Check available documents
        cursor.execute("SELECT COUNT(*) FROM documents;")
        doc_count = cursor.fetchone()[0]
        print(f"📊 Documents in database: {doc_count}")

        if doc_count == 0:
            print("❌ No documents found. Run test_real_embeddings.py first!")
            return False

        print("\n📝 Available documents:")
        cursor.execute("SELECT id, text FROM documents ORDER BY id;")
        documents = cursor.fetchall()
        for doc_id, text in documents:
            print(f"  {doc_id}. {text}")

        # Demonstrate different similarity searches
        queries = [
            ("database systems", "🗄️  Query: Database-related content"),
            ("machine learning", "🤖 Query: AI/ML-related content"),
            ("programming", "💻 Query: Programming-related content"),
        ]

        for query_term, description in queries:
            print(f"\n{description}")
            print("-" * 40)

            # Find documents containing the query term
            cursor.execute(
                "SELECT embedding FROM documents WHERE text ILIKE %s LIMIT 1;",
                (f"%{query_term}%",),
            )
            result = cursor.fetchone()

            if result:
                # Use existing document as query
                similarity_query = """
                WITH query_embedding AS (
                    SELECT embedding FROM documents WHERE text ILIKE %s LIMIT 1
                )
                SELECT 
                    d.id,
                    d.text,
                    1 - (d.embedding <=> qe.embedding) as similarity
                FROM documents d, query_embedding qe
                WHERE d.text NOT ILIKE %s
                ORDER BY d.embedding <=> qe.embedding
                LIMIT 3;
                """
                cursor.execute(similarity_query, (f"%{query_term}%", f"%{query_term}%"))
            else:
                # If no exact match, find most similar to all documents
                print(
                    f"  No documents contain '{query_term}', showing top similar documents:"
                )
                cursor.execute("""
                    SELECT id, text, 'N/A' as similarity 
                    FROM documents 
                    ORDER BY id 
                    LIMIT 3;
                """)

            results = cursor.fetchall()
            if results:
                for i, (doc_id, text, similarity) in enumerate(results, 1):
                    if similarity != "N/A":
                        print(f"  {i}. [ID:{doc_id}] {text[:60]}...")
                        print(f"     Similarity: {float(similarity):.4f}")
                    else:
                        print(f"  {i}. [ID:{doc_id}] {text}")
            else:
                print("  No similar documents found.")

        # Show vector operations
        print("\n🔧 Vector Operations Demo")
        print("-" * 40)

        # Show average similarity between all documents
        cursor.execute("""
            SELECT 
                AVG(1 - (d1.embedding <=> d2.embedding)) as avg_similarity
            FROM documents d1, documents d2 
            WHERE d1.id != d2.id;
        """)
        avg_sim = cursor.fetchone()[0]
        if avg_sim:
            print(f"📊 Average similarity between all documents: {float(avg_sim):.4f}")

        # Show document with highest self-similarity (should be 1.0)
        cursor.execute("""
            SELECT 
                id, 
                text,
                1 - (embedding <=> embedding) as self_similarity
            FROM documents 
            LIMIT 1;
        """)
        result = cursor.fetchone()
        if result:
            doc_id, text, self_sim = result
            print(f"✅ Self-similarity check: {float(self_sim):.4f} (should be 1.0)")

        conn.close()
        print("\n🎉 Similarity search demo completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        return False


if __name__ == "__main__":
    success = demo_similarity_search()
    sys.exit(0 if success else 1)

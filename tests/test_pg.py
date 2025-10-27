"""test_pg.py module."""

import numpy as np
import pandas as pd
import psycopg2
import pytest
from pgvector_db.pg_copy import pg_copy
from pgvector_db.pg_insert import pg_insert
from pgvector_db.utils import DBConfigLocal

# Define local test database configuration
TEST_DB_CONFIG = DBConfigLocal(
    db_name="vector_db",
    db_user="py_pg_user",
    db_password="test_password",
    db_host="localhost",
    schema_name="public",
    table_name="test_embeddings",
)


@pytest.fixture(scope="function")
def setup_test_db():
    """Set up & tear down a real local PostgreSQL test database."""
    conn = psycopg2.connect(
        dbname=TEST_DB_CONFIG.db_name,
        user=TEST_DB_CONFIG.db_user,
        password=TEST_DB_CONFIG.db_password,
        host=TEST_DB_CONFIG.db_host,
        port="5432",
    )
    cur = conn.cursor()

    # Create test table
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS public.test_embeddings (
            id SERIAL PRIMARY KEY,
            text TEXT,
            embedding VECTOR(3)
        )
    """
    )
    conn.commit()
    yield conn  # Provide database to test

    # Cleanup after test
    cur.execute("DROP TABLE IF EXISTS public.test_embeddings")
    conn.commit()
    conn.close()


def test_pg_insert(setup_test_db):
    """Test inserting data into PostgreSQL using `pg_insert`."""
    df = pd.DataFrame({"text": ["Hello world"], "embedding": [[0.1, 0.2, 0.3]]})

    pg_insert(df, TEST_DB_CONFIG, batch_size=1)

    # Verify data is inserted
    cur = setup_test_db.cursor()
    cur.execute("SELECT text, embedding::vector FROM public.test_embeddings")
    result = cur.fetchone()

    assert result is not None
    assert result[0] == "Hello world"
    assert isinstance(result[1], str)

    # Convert string to NumPy array & validate
    embedding_array = np.fromstring(result[1].strip("[]"), sep=",")
    assert isinstance(embedding_array, np.ndarray)
    assert embedding_array.shape == (3,)  # Ensure correct shape


def test_pg_copy(setup_test_db):
    """Test copying data to PostgreSQL using `pg_copy`."""
    df = pd.DataFrame(
        {
            "text": ["Sentence A", "Sentence B"],
            "embedding": [[0.5, 0.6, 0.7], [0.8, 0.9, 1.0]],
        }
    )

    pg_copy(df, TEST_DB_CONFIG)

    # Verify data exists
    cur = setup_test_db.cursor()
    cur.execute("SELECT COUNT(*) FROM public.test_embeddings")
    count = cur.fetchone()[0]

    assert count == 2  # Ensure two records were inserted

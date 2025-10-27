"""test_generate_embeddings.py module."""

import os
import tempfile

import awswrangler as wr
import boto3
import numpy as np
import pyarrow.parquet as pq
import pytest
from moto import mock_s3
from pgvector_db.generate_embeddings import EmbeddingGenerator

# Setup test variables
MODEL_PATH = "./models/all-MiniLM-L6-v2"
S3_BUCKET = "test-vector-bucket"
S3_PATH = f"s3://{S3_BUCKET}/vector_data/"


@pytest.fixture(scope="function")
def s3_mock():
    """Set up a mocked S3 environment for testing."""
    with mock_s3():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=S3_BUCKET)
        yield s3


@pytest.fixture(scope="module")
def embedding_generator():
    """Fixture to initialize the EmbeddingGenerator."""
    return EmbeddingGenerator(model_path=MODEL_PATH, batch_size=2, device="cpu")


@pytest.fixture(scope="function")
def temp_parquet_path():
    """Fixture to create a temporary directory for Parquet files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield os.path.join(temp_dir, "test_embeddings.parquet")


def test_generate_embeddings(embedding_generator):
    """Test embedding generation correctness."""
    texts = ["Hello world", "Testing embeddings"]

    embeddings = embedding_generator.generate_embeddings(texts)

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (2, embeddings.shape[1])  # (2, embedding_dim)
    assert len(embeddings) == len(texts)


def test_save_to_local_parquet(embedding_generator, temp_parquet_path):
    """Test saving embeddings locally as a Parquet file."""
    texts = ["Sample text 1", "Sample text 2"]
    embeddings = embedding_generator.generate_embeddings(texts)

    # Save to local Parquet
    embedding_generator.save_to_parquet(texts, embeddings, temp_parquet_path)

    # Check if file exists
    assert os.path.exists(temp_parquet_path)

    # Validate Parquet content
    df = pq.read_table(temp_parquet_path).to_pandas()
    assert df.shape[0] == len(texts)
    assert "text" in df.columns
    assert "embedding" in df.columns


def test_save_to_s3_parquet(embedding_generator, s3_mock):
    """Test saving embeddings to S3."""
    texts = ["AWS is great", "S3 is scalable"]
    embeddings = embedding_generator.generate_embeddings(texts)

    # Save to S3
    embedding_generator.save_to_parquet(texts, embeddings, S3_PATH)

    # Validate S3 file existence
    objects = s3_mock.list_objects_v2(Bucket=S3_BUCKET)
    assert "Contents" in objects, f"S3 Objects: {objects}"
    assert len(objects["Contents"]) > 0

    # Validate content
    df_loaded = wr.s3.read_parquet(S3_PATH)
    assert df_loaded.shape[0] == len(texts)
    assert "text" in df_loaded.columns
    assert "embedding" in df_loaded.columns

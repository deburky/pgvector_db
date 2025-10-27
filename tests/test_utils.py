"""test_utils.py."""

import os

import pytest
from pgvector_db.utils import download_llm

# Define test model names & paths
TEST_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TEST_HF_MODEL_NAME = "bert-base-uncased"
TEST_SAVE_PATH = "./test_models"


@pytest.fixture(scope="function", autouse=True)
def cleanup():
    """Fixture to clean up test directories after each test."""
    yield
    if os.path.exists(TEST_SAVE_PATH):
        os.system(f"rm -rf {TEST_SAVE_PATH}")


def test_download_llm_sentence_transformers():
    """Test `download_llm` for SentenceTransformers models (without mocks)."""

    # Run the function (actually downloads model)
    model_path = download_llm(TEST_MODEL_NAME, TEST_SAVE_PATH, "sentence-transformers")

    # Assertions: Check that the directory exists
    expected_path = f"{TEST_SAVE_PATH}/{TEST_MODEL_NAME.replace('/', '-')}"
    assert os.path.exists(expected_path)
    assert model_path == expected_path


def test_download_llm_huggingface():
    """Test `download_llm` for Hugging Face models (without mocks)."""

    # Run the function (actually downloads model)
    model_path = download_llm(TEST_HF_MODEL_NAME, TEST_SAVE_PATH, "huggingface")

    # Assertions: Check that the directory exists
    expected_path = f"{TEST_SAVE_PATH}/{TEST_HF_MODEL_NAME.replace('/', '-')}"
    assert os.path.exists(expected_path)
    assert model_path == expected_path

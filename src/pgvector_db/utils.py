"""utils.py module."""

import logging
import time
from dataclasses import dataclass
from functools import wraps

import boto3

# Configure Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# Time Execution Decorator
def time_it(func):
    """Decorate to measure execution time of a function."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        logging.info("Starting execution of %s", func.__name__)
        start_time = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start_time
        logging.info(
            "Completed %s in %d min %.2f sec",
            func.__name__,
            int(duration // 60),
            duration % 60,
        )
        return result

    return wrapper


# Database Configuration
@dataclass(slots=True)
class DBConfigLocal:
    """Configure Local PostgreSQL."""

    db_name: str
    db_user: str
    db_password: str
    db_host: str
    db_port: str = "5432"
    schema_name: str = "public"
    table_name: str = "documents"


@dataclass(slots=True)
class DBConfigRDS:
    """Configure Amazon RDS with optional IAM authentication."""

    db_name: str
    db_user: str
    db_host: str
    db_password: str = None
    db_port: str = "5432"
    schema_name: str = "public"
    table_name: str = "documents"
    use_iam: bool = False
    aws_region: str = "us-east-1"

    def get_iam_rds_token(self) -> str:
        """Generate an AWS IAM authentication token for Amazon RDS."""
        if not self.use_iam:
            return self.db_password  # Use password if IAM is disabled

        rds_client = boto3.client("rds", region_name=self.aws_region)
        try:
            token = rds_client.generate_db_auth_token(
                DBHostname=self.db_host, Port=int(self.db_port), DBUsername=self.db_user
            )
            logging.info("Generated IAM authentication token for RDS")
            return token
        except Exception as e:
            logging.error("Error generating IAM token: %s", e)
            raise


# LLM Download Function
@time_it
def download_llm(model_name: str, save_path: str, model_type: str) -> str:
    """
    Download a model (SentenceTransformers or Hugging Face) and save locally.

    :param model_name: Name of the model (e.g., "sentence-transformers/all-MiniLM-L6-v2").
    :param save_path: Directory where the model will be saved.
    :param model_type: "sentence-transformers" or "huggingface".
    :return: Local path where the model is saved.
    """
    model_dir = f"{save_path}/{model_name.replace('/', '-')}"

    if model_type == "sentence-transformers":
        import torch
        from sentence_transformers import SentenceTransformer

        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        model = SentenceTransformer(model_name, device=device)
        model.save(model_dir)

    elif model_type == "huggingface":
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)

        tokenizer.save_pretrained(model_dir)
        model.save_pretrained(model_dir)

    else:
        raise ValueError(
            "Invalid model_type. Choose 'sentence-transformers' or 'huggingface'."
        )

    return model_dir

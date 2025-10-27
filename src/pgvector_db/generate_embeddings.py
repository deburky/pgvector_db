"""generate_embeddings.py module."""

import datetime
import logging
from typing import List, Literal

import awswrangler as wr
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class EmbeddingGenerator:
    def __init__(
        self,
        model_path: str,
        model_type="sentence-transformers",
        batch_size=500,
        device=None,
    ):
        """
        Initialize the embedding generator with a locally stored model.
        :param model_path: Path to the locally stored model.
        :param model_type: Type of model ("sentence-transformers" or "huggingface").
        :param batch_size: Number of texts to process in a single batch.
        :param device: Device to run on ("cpu", "cuda", "mps").
        """
        self.model_path = model_path
        self.model_type = model_type
        self.batch_size = batch_size

        # Allow user to specify a device, otherwise automatically detect
        if device:
            self.device = device
        else:
            self.device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )

        logging.info(f"✅ Using device: {self.device}")

        if model_type == "sentence-transformers":
            self.model = SentenceTransformer(
                model_path, device=self.device
            )  # Assign device
        elif model_type == "huggingface":
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModel.from_pretrained(model_path).to(self.device)
        else:
            raise ValueError(
                "Invalid model_type. Choose 'sentence-transformers' or 'huggingface'."
            )

        logging.info(f"✅ Loaded model from {model_path} using {model_type}")

    def generate_embeddings(
        self,
        texts: List[str],
        embedding_type: Literal["sentence", "token"] = "sentence",
    ) -> np.ndarray:
        """
        Generate embeddings for a list of texts using batch processing.

        :param texts: List of input texts.
        :param embedding_type: "sentence" for full sentence embeddings, "token" for token embeddings.
        :return: NumPy array of embeddings.
        """
        # Limit MPS memory usage
        if self.device == "mps":
            torch.mps.set_per_process_memory_fraction(0.8)
        if self.model_type == "sentence-transformers":
            if embedding_type == "sentence":
                with torch.no_grad():
                    embeddings = self.model.encode(
                        texts,
                        batch_size=self.batch_size,
                        convert_to_numpy=True,
                        show_progress_bar=True,
                    )
            elif embedding_type == "token":
                embeddings = [self.model.tokenize(text)["input_ids"] for text in texts]
            else:
                raise ValueError(
                    "Invalid embedding_type. Choose 'sentence' or 'token'."
                )

        elif self.model_type == "huggingface":
            with torch.no_grad():
                inputs = self.tokenizer(
                    texts, return_tensors="pt", padding=True, truncation=True
                ).to(self.device)

                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

        return embeddings

    def save_to_parquet(
        self, texts: List[str], embeddings: List[np.ndarray], output_path: str
    ):
        """
        Save text and embeddings as a Parquet file. Supports both local and S3 storage.

        :param texts: List of input texts.
        :param embeddings: Corresponding embeddings.
        :param output_path: File path to save the Parquet file (supports local & S3).
        """

        # Create a timestamp for partitioning
        generation_date = datetime.date.today().isoformat()
        df = pd.DataFrame(
            {
                "generation_date": generation_date,
                "text": texts,
                "embedding": [e.tolist() for e in embeddings],
            }
        )

        if output_path.startswith("s3://"):
            # Write directly to S3 using awswrangler
            logging.info(f"✅ Saving embeddings to S3: {output_path}")
            wr.s3.to_parquet(
                df=df,
                path=output_path,
                dataset=True,
                partition_cols=["generation_date"],
                mode="append",
            )
        else:
            # Save locally as Parquet
            table = pa.Table.from_pandas(df.drop(columns=["generation_date"]))
            pq.write_table(table, output_path)
            logging.info(f"✅ Saved {len(texts)} embeddings locally to {output_path}")

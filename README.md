# LLM Embedding Generation and PostgreSQL Vector Database

> Production-ready Python package for generating embeddings with 🤗 Hugging Face models and storing them in PostgreSQL with `pgvector` extension.

**Author**: https://www.github.com/deburky

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PostgreSQL 17](https://img.shields.io/badge/postgresql-17-blue.svg)](https://www.postgresql.org/)
[![pgvector](https://img.shields.io/badge/pgvector-0.8.1-green.svg)](https://github.com/pgvector/pgvector)

## 🚀 Quick Start

```bash
# 1. Install dependencies
uv sync

# 2. Set up PostgreSQL with pgvector (see setup guide below)
brew install postgresql@17 pgvector
brew services start postgresql@17

# 3. Run the demo
export PG_PASSWORD=your_actual_password
uv run python demo/test_real_embeddings.py
```

## 📖 Table of Contents

1. [Features](#features)
2. [Installation](#installation)
3. [Quick Demo](#quick-demo)
4. [PostgreSQL Setup](#postgresql-setup)
5. [Usage Examples](#usage-examples)
6. [AWS Integration](#aws-integration)
7. [Performance](#performance)

## ✨ Features

✅ **Real LLM Embeddings** - Generate 384-1536 dimensional vectors using sentence-transformers  
✅ **PostgreSQL Integration** - Store embeddings with enterprise-grade reliability  
✅ **Vector Similarity Search** - Fast semantic search with IVFFLAT/HNSW indexing  
✅ **Batch Processing** - Efficient bulk operations with `pg_copy` and `pg_insert`  
✅ **Apple Silicon Support** - Optimized for MPS (Metal Performance Shaders)  
✅ **AWS Integration** - S3 storage and RDS support  
✅ **Production Ready** - Comprehensive error handling and logging  

## 🔧 Installation

This project uses [uv](https://github.com/astral-sh/uv) for fast Python package management:

```bash
# Clone the repository
git clone https://github.com/deburky/pgvector_db.git
cd pgvector_db

# Install dependencies
uv sync

# Optional: Install with development dependencies
uv sync --extra dev --extra test
```

## 🎯 Quick Demo

Try our interactive demos to see the package in action:

```bash
# 1. Basic connectivity test
uv run python demo/test_db_connection.py

# 2. Real LLM embeddings generation and storage
uv run python demo/test_real_embeddings.py

# 3. Interactive similarity search demo
uv run python demo/similarity_search_demo.py
```

**📝 For detailed instructions, see [`demo/README.md`](./demo/README.md)**

## 📊 Embeddings

An embedding is a mapping from discrete objects (words, sentences, documents) to points in a continuous vector space. This enables neural networks and machine learning models to process text semantically.

We use open-source models from 🤗 Hugging Face and sentence-transformers to generate high-quality embeddings for downstream tasks like similarity search, classification, and retrieval-augmented generation (RAG).

### Download and Cache Models

```python
from pgvector_db.utils import download_llm

# Sentence Transformers model
model_path = download_llm(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    save_path="./models",
    model_type="sentence-transformers"
)

# Hugging Face model
hf_model_path = download_llm(
    model_name="bert-base-uncased",
    save_path="./models",
    model_type="huggingface"
)
```

### Generate Embeddings and Store to S3

```python
from pgvector_db.generate_embeddings import EmbeddingGenerator

# Initialize with local model
generator = EmbeddingGenerator(
    model_path="./models/sentence-transformers-all-MiniLM-L6-v2",
    model_type="sentence-transformers",
    batch_size=32,
    device="mps"  # Use Apple Silicon GPU
)

texts = ["Hello world", "Vector databases are powerful"]
embeddings = generator.generate_embeddings(texts)

# Store to S3 with partitioning for Athena
generator.save_to_parquet(
    texts, 
    embeddings, 
    "s3://my-bucket/embeddings/",
    partition_cols=["generation_date"]
)
```

## 🗄️ PostgreSQL Setup

### Prerequisites

- **macOS**: Homebrew installed
- **Windows**: Use Docker, WSL, or native PostgreSQL installer
- **Linux**: Use your distribution's package manager

### Install PostgreSQL 17 and pgvector

```bash
# Install PostgreSQL 17 (latest with pgvector support)
brew install postgresql@17

# Install pgvector extension  
brew install pgvector

# Verify installation
/opt/homebrew/opt/postgresql@17/bin/pg_config --sharedir
```

### Start PostgreSQL and Create Database

```bash
# Start PostgreSQL 17
brew services start postgresql@17

# Add to PATH for easier access
export PATH="/opt/homebrew/opt/postgresql@17/bin:$PATH"

# Create database and enable vector extension
psql postgres -c "CREATE DATABASE vector_db;"
psql vector_db -c "CREATE EXTENSION vector;"
```

### Create Database Schema

```sql
-- Connect to vector_db
\c vector_db

-- Create documents table (384 dims for all-MiniLM-L6-v2)
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    text TEXT,
    embedding VECTOR(384)
);

-- Create application user
CREATE USER py_pg_user WITH PASSWORD 'your_secure_password';
GRANT ALL PRIVILEGES ON DATABASE vector_db TO py_pg_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO py_pg_user;
GRANT USAGE, SELECT ON SEQUENCE documents_id_seq TO py_pg_user;

-- Create index for fast similarity search
CREATE INDEX ON documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
```

### Environment Variables

```bash
export PG_PASSWORD=your_secure_password
export PGUSER=py_pg_user  # Optional
```

## 💻 Usage Examples

### Basic Database Configuration

```python
from pgvector_db.utils import DBConfigLocal

db_config = DBConfigLocal(
    db_name="vector_db",
    db_user="py_pg_user", 
    db_password=os.getenv('PG_PASSWORD'),
    db_host="localhost",
    schema_name="public",
    table_name="documents"
)
```

### Store Embeddings in PostgreSQL

```python
import pandas as pd
from pgvector_db.pg_insert import pg_insert

# Prepare data with embeddings as lists
dataset = pd.DataFrame({
    'text': ["Sample document", "Another document"],
    'embedding': [embedding1.tolist(), embedding2.tolist()]
})

# Insert into PostgreSQL
pg_insert(dataset, db_config, batch_size=1000)
```

### Vector Similarity Search

```sql
-- Find most similar documents to a query
WITH query_embedding AS (
    SELECT embedding FROM documents WHERE text LIKE '%database%' LIMIT 1
)
SELECT 
    d.text,
    1 - (d.embedding <=> qe.embedding) as similarity
FROM documents d, query_embedding qe
WHERE d.text NOT LIKE '%database%'
ORDER BY d.embedding <=> qe.embedding
LIMIT 5;
```

### AWS RDS PostgreSQL Integration

Our package seamlessly works with AWS RDS PostgreSQL instances. Key components for RDS integration include:

1. **Connection Setup**
   ```python
   # Standard password authentication (used in our examples)
   db_config = DBConfigRDS(
       db_name="postgres",
       db_user="admin_user",
       db_password=os.environ.get('ADMIN_PASS'),
       db_host="your-pgvector-rds.region.rds.amazonaws.com",
       schema_name="public",
       table_name="documents"
   )
   
   # Alternative: IAM authentication (more secure for production)
   # db_config = DBConfigRDS(
   #     db_name="postgres",
   #     db_user="admin_user", 
   #     db_host="your-pgvector-rds.region.rds.amazonaws.com",
   #     use_iam=True,  # Enable IAM authentication
   #     aws_region="us-east-1"
   # )
   ```

2. **Vector Operations**
   ```python
   # Create table with vector column if needed
   cursor.execute("""
   CREATE TABLE IF NOT EXISTS documents (
       id SERIAL PRIMARY KEY,
       text TEXT,
       embedding vector(384)
   );
   """)

   # Create vector index for fast similarity search
   cursor.execute("""
   CREATE INDEX IF NOT EXISTS documents_embedding_idx 
   ON documents USING ivfflat (embedding vector_cosine_ops) 
   WITH (lists = 100);
   """)
   ```

3. **Vector Similarity Search**
   ```python
   # Important: Use string representation with explicit type casting
   vector_str = f"'[{','.join(map(str, query_embedding))}]'"
   
   # Execute search with proper vector operator
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
   ```

4. **Using the AWS RDS Examples**
   ```bash
   # Set password for RDS connection
   export ADMIN_PASS=your_actual_rds_password
   
   # Run the AWS RDS examples
   python demo/test_aws_rds_connection.py
   python demo/aws_rds_similarity_search_demo.py
   ```

For more details, explore the demo examples in the `demo/` directory.

## ⚡ Performance Options

### Bulk Operations (Recommended)

For large datasets, use the COPY-based workflow:

```python
from pgvector_db.pg_copy import pg_copy

# Efficient bulk insert using PostgreSQL COPY
pg_copy(dataset, db_config)
```

### Real-time Operations  

For single records or small batches:

```python
from pgvector_db.pg_insert import pg_insert

# Row-by-row insertion for real-time applications
pg_insert(dataset, db_config, batch_size=100)
```

## 🛠️ Development

### Project Structure

```
pgvector_db/
├── src/pgvector_db/           # Main package
│   ├── generate_embeddings.py # LLM embedding generation
│   ├── pg_copy.py             # Bulk COPY operations  
│   ├── pg_insert.py           # INSERT operations
│   └── utils.py               # Database configs & utilities
├── demo/                      # Working examples
│   ├── README.md              # Demo instructions
│   ├── test_real_embeddings.py # Complete pipeline test
│   └── similarity_search_demo.py # Interactive demo
├── notebooks/                 # Jupyter examples
└── tests/                     # Unit tests
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=pgvector_db

# Run specific test
uv run python demo/test_real_embeddings.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📚 Resources

- [📖 Complete Tutorial](./demo/medium_article.md) - Step-by-step guide
- [🔧 Demo Examples](./demo/) - Working code samples  
- [📝 Jupyter Notebooks](./notebooks/) - Interactive examples
- [🐘 pgvector Documentation](https://github.com/pgvector/pgvector) - Vector extension
- [🤗 Sentence Transformers](https://www.sbert.net/) - Embedding models

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

**⭐ Star this repo if you found it helpful!**

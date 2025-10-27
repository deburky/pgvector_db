# pgvector Demo Examples

This directory contains practical examples and tutorials for using PostgreSQL with pgvector for vector database applications.

## 📁 Contents

- **`medium_article.md`** - Complete tutorial for building a production-ready vector database
- **`test_db_connection.py`** - Basic connectivity test with random vectors
- **`test_real_embeddings.py`** - Real LLM embedding generation and storage test

## 🚀 Quick Start

### Prerequisites

1. **PostgreSQL 17 + pgvector installed:**
   ```bash
   brew install postgresql@17 pgvector
   brew services start postgresql@17
   ```

2. **Database setup:**
   ```bash
   # Create database and enable vector extension
   psql postgres -c "CREATE DATABASE vector_db;"
   psql vector_db -c "CREATE EXTENSION vector;"
   
   # Create table and user (see medium_article.md for full SQL)
   psql vector_db -c "CREATE TABLE documents (id SERIAL PRIMARY KEY, text TEXT, embedding VECTOR(384));"
   psql vector_db -c "CREATE USER py_pg_user WITH PASSWORD 'your_secure_password';"
   # ... (additional permissions)
   ```

3. **Environment setup:**
   ```bash
      export PG_PASSWORD=your_secure_password
   export PATH="/opt/homebrew/opt/postgresql@17/bin:$PATH"
   ```

### Running the Examples

#### Test 1: Basic Connectivity
```bash
# From project root
cd /path/to/pgvector_db
uv run python demo/test_db_connection.py
```

**What it does:**
- Tests basic PostgreSQL connection
- Inserts test documents with random 384-dimensional vectors
- Verifies the pgvector_db package works

**Expected output:**
```
🔍 Testing database connection and insert...
📊 Test data shape: (3, 2)
🎯 Embedding dimension: 384
✅ SUCCESS: Data inserted successfully!
```

#### Test 2: Real LLM Embeddings
```bash
# From project root
uv run python demo/test_real_embeddings.py
```

**What it does:**
- Downloads sentence-transformers model (`all-MiniLM-L6-v2`)
- Generates real embeddings for 5 test sentences
- Stores embeddings in PostgreSQL
- Demonstrates the complete ML → Database pipeline

**Expected output:**
```
🤖 Testing with real LLM embeddings...
📥 Setting up sentence-transformers model...
✅ Model ready at: ./models/sentence-transformers-all-MiniLM-L6-v2
🔧 Initializing embedding generator...
✅ Embedding generator ready
📝 Generating embeddings for 5 texts...
✅ Generated embeddings with shape: (5, 384)
🎯 Embedding dimension: 384
💾 Inserting real embeddings into PostgreSQL...
✅ SUCCESS: Real LLM embeddings inserted successfully!
```

### Verify Data & Test Similarity Search

After running the tests, verify your data:

```sql
-- Connect to database
psql vector_db

-- Check inserted data
SELECT id, text, vector_dims(embedding) as dim FROM documents ORDER BY id DESC LIMIT 5;

-- Test semantic similarity search
WITH query_embedding AS (
    SELECT embedding FROM documents WHERE text LIKE '%database%' LIMIT 1
)
SELECT 
    d.id,
    d.text,
    1 - (d.embedding <=> qe.embedding) as similarity
FROM documents d, query_embedding qe
WHERE d.text NOT LIKE '%database%'
ORDER BY d.embedding <=> qe.embedding
LIMIT 3;
```

## 🎯 Understanding the Examples

### Database Configuration (`DBConfigLocal`)

```python
db_config = DBConfigLocal(
    db_name="vector_db",        # Database name
    db_user="py_pg_user",       # Application user
    db_password=password,       # From environment variable
    db_host="localhost",        # Local PostgreSQL
    schema_name="public",       # Default schema
    table_name="documents"      # Target table
)
```

### Embedding Generation Pipeline

1. **Model Download**: Cache sentence-transformers model locally
2. **Generator Setup**: Initialize with device optimization (MPS for Apple Silicon)
3. **Batch Processing**: Generate embeddings efficiently in batches
4. **Format Conversion**: Convert numpy arrays to PostgreSQL-compatible lists

### Vector Storage Options

- **`pg_insert`**: Row-by-row insertion, good for real-time applications
- **`pg_copy`**: Bulk insertion using PostgreSQL COPY, faster for large datasets

## 🔧 Customization

### Different Models

Change the model in `test_real_embeddings.py`:

```python
# Different embedding models
model_options = {
    "mini": "sentence-transformers/all-MiniLM-L6-v2",      # 384 dims, fast
    "base": "sentence-transformers/all-mpnet-base-v2",     # 768 dims, balanced  
    "large": "sentence-transformers/all-roberta-large-v1"  # 1024 dims, accurate
}
```

**Note**: Update table schema to match embedding dimensions:
```sql
-- For 768-dimensional embeddings
ALTER TABLE documents ALTER COLUMN embedding TYPE VECTOR(768);
```

### Custom Test Data

Modify the test texts in `test_real_embeddings.py`:

```python
test_texts = [
    "Your custom text here",
    "Add domain-specific content",
    "Test with your actual use case"
]
```

## 🐛 Troubleshooting

### Common Issues

1. **"could not access file $libdir/vector"**
   - pgvector not properly installed for your PostgreSQL version
   - Solution: Reinstall with matching versions

2. **"role py_pg_user does not exist"**
   - Database user not created
   - Solution: Run the user creation SQL commands

3. **"relation documents does not exist"**
   - Table not created
   - Solution: Run the table creation SQL

4. **Connection refused**
   - PostgreSQL not running
   - Solution: `brew services start postgresql@17`

### Checking Setup

```bash
# Verify PostgreSQL is running
brew services list | grep postgresql

# Check pgvector installation
/opt/homebrew/opt/postgresql@17/bin/pg_config --sharedir

# Test database connection
psql vector_db -c "SELECT version();"
```

## 📊 Performance Notes

- **Model Loading**: ~2-3 seconds (cached after first run)
- **Embedding Generation**: ~0.3 seconds per text on Apple Silicon
- **Database Insert**: ~0.01 seconds per record
- **Memory Usage**: ~500MB for model + embeddings

## 🚀 Next Steps

1. **Scale Testing**: Try with larger datasets (1K, 10K, 100K+ documents)
2. **Production Setup**: Configure connection pooling, replication, backups
3. **Integration**: Connect with your web framework (FastAPI, Flask, Django)
4. **Optimization**: Tune indexes and batch sizes for your workload

## 📚 Additional Resources

- [Project Documentation](../README.md) - Package overview
- [pgvector Docs](https://github.com/pgvector/pgvector) - Extension reference
- [Sentence Transformers](https://www.sbert.net/) - Embedding models
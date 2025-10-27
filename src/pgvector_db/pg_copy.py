"""pg_copy.py module."""

import logging
import tempfile

import pandas as pd
import psycopg
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from pgpq import ArrowToPostgresBinaryEncoder
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from pgvector_db.utils import DBConfigLocal, DBConfigRDS, time_it

# Configure logging
logging.basicConfig(
    filename="rds_ingestion.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

console = Console()


@time_it
def pg_copy(data, config: DBConfigLocal):
    """Copy embeddings from Parquet/Arrow/Pandas to PostgreSQL."""

    db_password = (
        config.get_iam_rds_token()
        if isinstance(config, DBConfigRDS)
        else config.db_password
    )
    db_url = f"postgresql://{config.db_user}:{db_password}@{config.db_host}:{config.db_port}/{config.db_name}"

    # Check if `data` is a Pandas DataFrame or a Parquet file path
    if isinstance(data, pd.DataFrame):
        console.print("[bold cyan1]⚡ Converting DataFrame to Parquet...[/bold cyan1]")
        temp_file = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False)
        parquet_path = temp_file.name
        pq.write_table(pa.Table.from_pandas(data), parquet_path)
    elif isinstance(data, str):
        parquet_path = data
    else:
        raise ValueError(
            "Invalid input: must be a Pandas DataFrame or a Parquet file path."
        )

    # Load Arrow dataset from Parquet (handles S3/local paths)
    dataset = ds.dataset(parquet_path, format="parquet")

    # Create encoder object
    encoder = ArrowToPostgresBinaryEncoder(dataset.schema)

    # Get expected PostgreSQL schema
    pg_schema = encoder.schema()
    cols = [
        f'"{col_name}" {col.data_type.ddl()}' for col_name, col in pg_schema.columns
    ]
    ddl = f"CREATE TEMP TABLE temp_data ({','.join(cols)})"

    try:
        with psycopg.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute(ddl)  # Create temporary table

                with Progress(
                    SpinnerColumn("earth", style="bright_magenta"),
                    TextColumn(
                        "[dodger_blue1]Copying Parquet Data to PostgreSQL (BINARY)...[/dodger_blue1]"
                    ),
                    console=console,
                    transient=True,
                ) as progress:
                    task = progress.add_task("", total=1)  # Single COPY operation

                    with cur.copy(
                        "COPY temp_data FROM STDIN WITH (FORMAT BINARY)"
                    ) as copy:
                        copy.write(encoder.write_header())
                        for batch in dataset.to_batches():
                            copy.write(encoder.write_batch(batch))
                        copy.write(encoder.finish())

                    progress.advance(task, 1)
                cur.execute(
                    f"INSERT INTO {config.schema_name}.{config.table_name} "
                    "(text, embedding) SELECT text, embedding FROM temp_data;"
                )
                conn.commit()

        logging.info(
            "Inserted data from Parquet into %s.%s using COPY!",
            config.schema_name,
            config.table_name,
        )
        console.print(
            f"[bold spring_green3]✔ Success[/bold spring_green3]: "
            f"copied Parquet data into [bold]{config.schema_name}.{config.table_name}[/bold]"
        )

    except psycopg.DatabaseError as e:
        logging.error("Database error: %s", e)
        console.print(f"[bold deep_pink2]Database error:[/bold deep_pink2] {e}")

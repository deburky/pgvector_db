"""pg_insert.py module."""

import logging

import pandas as pd
import psycopg2
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

from pgvector_db.utils import DBConfigLocal, DBConfigRDS, time_it

# Configure logging
logging.basicConfig(
    filename="rds_ingestion.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

console = Console()


@time_it
def pg_insert(df: pd.DataFrame, config: DBConfigLocal, batch_size=1000):
    """Batch insert embeddings into PostgreSQL."""

    required_columns = {"text", "embedding"}
    if missing_columns := required_columns - set(df.columns):
        raise ValueError(f"DataFrame is missing required columns: {missing_columns}")

    db_password = (
        config.get_iam_rds_token()
        if isinstance(config, DBConfigRDS)
        else config.db_password
    )
    insert_query = f"INSERT INTO {config.schema_name}.{config.table_name} (text, embedding) VALUES (%s, %s)"
    data_to_insert = list(zip(df["text"], df["embedding"]))
    total_batches = (len(data_to_insert) + batch_size - 1) // batch_size
    try:
        with psycopg2.connect(
            dbname=config.db_name,
            user=config.db_user,
            password=db_password,
            host=config.db_host,
            port=config.db_port,
            sslmode=(
                "require"
                if isinstance(config, DBConfigRDS) and config.use_iam
                else "prefer"
            ),
        ) as conn:
            with conn.cursor() as cur:
                with Progress(
                    SpinnerColumn("earth", style="bright_magenta"),
                    TextColumn("[dodger_blue1]Writing to PostgreSQL...[/dodger_blue1]"),
                    BarColumn(),
                    TextColumn("[progress.percentage]{task.percentage:.0f}%"),
                    console=console,
                ) as progress:
                    task = progress.add_task("", total=total_batches)

                    for i in range(0, len(data_to_insert), batch_size):
                        batch = data_to_insert[i : i + batch_size]
                        cur.executemany(insert_query, batch)
                        conn.commit()
                        progress.advance(task, 1)
                        progress.refresh()

        logging.info(
            "Inserted %d rows into %s.%s!",
            len(data_to_insert),
            config.schema_name,
            config.table_name,
        )
        console.print(
            f"[bold spring_green3]✔ Success[/bold spring_green3]: inserted [bold]{len(data_to_insert)} rows[/bold] into [bold]{config.schema_name}.{config.table_name}[/bold]"
        )

    except psycopg2.DatabaseError as e:
        logging.error("Database error: %s", e)
        console.print(f"[bold deep_pink2]Database error:[/bold deep_pink2] {e}")

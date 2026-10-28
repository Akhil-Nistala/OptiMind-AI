import sqlite3
from contextlib import contextmanager
from pathlib import Path
import pandas as pd


@contextmanager
def readonly_connection(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Production database missing: {path}; run python -m scripts.prepare_data"
        )
    conn = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=ON")
    try:
        yield conn
    finally:
        conn.close()


def read_period(path, start, end):
    with readonly_connection(path) as conn:
        return pd.read_sql_query(
            "SELECT * FROM production WHERE production_date BETWEEN ? AND ? ORDER BY production_date,machine_id",
            conn,
            params=(start, end),
        )

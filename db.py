from __future__ import annotations

from pathlib import Path
import duckdb

PROJECT_ROOT = Path(__file__).resolve().parents[0]
DATABASE_PATH = PROJECT_ROOT/"data"/"silver"/"events.duckdb"
SCHEMA_PATH = PROJECT_ROOT / "data" /"silver"/"create_tables.sql"
def connect(read_only:bool=False)-> duckdb.DuckDBPyConnection:
    DATABASE_PATH.parent.mkdir(parents=True,exist_ok=True)

    return duckdb.connect(
        str(DATABASE_PATH),
        read_only = read_only
    )

def initialize_database()->None:
    ddl = SCHEMA_PATH.read_text(encoding="utf-8")
    with connect() as connection:
        connection.execute(ddl)

if __name__ == "__main__":
    initialize_database()

from __future__ import annotations
import os
import wrds
user_name=os.getenv("WRDS_USERNAME")
print(user_name)

connection = wrds.Connection(wrds_username=f"{user_name}")
libraries=connection.list_libraries()
preferred = (
    "crsp_a_stock",
    "crsp_q_stock",
    "crsp_m_stock",
    "crsp",
)

library = next(name for name in preferred if name in libraries)
tables = sorted(connection.list_tables(library=library))
print(tables)

candidates = [
    table
    for table in tables
    if any(
        word in table.lower()
        for word in ("dsf", "stockname", "securityinfo")
        )
]
print("\nCandidate tables:")
for table in candidates:
    print(f"  {table}")
for table in ("dsf_v2", "stocknames_v2", "stksecurityinfohist"):
    if table in tables:
        print(f"\nColumns: {library}.{table}")
        description = connection.describe_table(
            library=library,
            table=table,
        )
        print(description.to_string(index=False))

connection.close()
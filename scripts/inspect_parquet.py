from __future__ import annotations

import os
import pandas as pd
from pathlib import Path


OUTPUT_PATH = Path(
    "data/raw/wrds/crsp/daily_bars_2024_h1.parquet"
)
data = pd.read_parquet(OUTPUT_PATH)
print(data.columns)
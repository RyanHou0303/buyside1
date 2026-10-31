from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option('display.max_columns', None)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SILVER_DIR = PROJECT_ROOT / "data/silver"

UNIVERSE_PATH = SILVER_DIR / "universe_2010_2024.parquet"
FEATURES_PATH = SILVER_DIR / "features.parquet"

FEATURE_COLUMNS = [
    "reversal_1m",
    "mom_12m",
    "vol_1m",
    "log_adv_1m",
]

d1=pd.read_parquet(UNIVERSE_PATH)
d2=pd.read_parquet(FEATURES_PATH)
print(d1)
print(d2)
def build_research_panel():
        features=pd.read_parquet(FEATURES_PATH)
        universe=pd.read_parquet(UNIVERSE_PATH)
        panel = universe.merge(
        features,
        on=["permno", "holding_month"],
        how="left",
        validate="one_to_one",
        indicator="feature_match",
    )
        return panel
research_panel=build_research_panel()
print(research_panel["feature_match"])



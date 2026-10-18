from pathlib import Path
import pandas as pd
from backtest.engine import backtest
pd.set_option("display.max_columns", None)


DATA_PATH =(
    Path(__file__).resolve().parents[2]/"data/raw/wrds/crsp/daily_bars_2024_h1.parquet"
)
bars = pd.read_parquet(DATA_PATH)
returns =(
    bars.pivot(
        index="session_date",
        columns="permno",
        values="total_return"
    )
    .sort_index()
)
print(returns)
weights = pd.DataFrame(
    1.0 / returns.shape[1],
    index=returns.index,
    columns=returns.columns,
)
result = backtest(
    returns=returns,
    weights=weights,
    initial_nav=1.0,
    transaction_cost_bps=10,
    plot=True,
)

from pathlib import Path
import pandas as pd
import pandas as pd
data=pd.read_parquet("/Users/bohanhou/Desktop/COLUMBIA/buyside1/data/raw/wrds/crsp/yearly_v2/crsp_20000101_20001231.parquet")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR =PROJECT_ROOT/"data/raw/wrds/crsp/yearly_v2"
OUTPUT_PATH = PROJECT_ROOT / "data/silver/universe_2010.parquet"

"""   permno       date       ret     prc   vol primaryexch tradingstatusflg
0   10001 2000-01-03  0.007353  8.5625  1721           Q                A
1   10001 2000-01-04 -0.014599  8.4375  1080           Q                A
2   10001 2000-01-05  0.014815  8.5625  1711           Q                A
3   10001 2000-01-06 -0.007299     8.5   580           Q                A
4   10001 2000-01-07 -0.007353  8.4375  1406           Q                A
"""
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data/raw/wrds/crsp/yearly_v2"
OUTPUT_PATH = PROJECT_ROOT / "data/silver/universe_2010.parquet"

OUTPUT_COLUMNS = [
    "formation_date",
    "holding_month",
    "permno",
    "adv60",
    "liquidity_rank",
    "prc"

]
"""
for each stock, calculate price*volume
recent 60 days avg doller vol
top 1000 stock,
use it for one month
"""

def build_monthly_universe(
    bars: pd.DataFrame,
    n_stocks: int = 1000,
    min_price: float = 5.0,
) -> pd.DataFrame:
    data = bars.copy()
    data["date"] = pd.to_datetime(data["date"])

    if data.duplicated(["permno", "date"]).any():
        raise ValueError("Duplicate permno/date records")


    valid = data["prc"].abs().gt(0) & data["vol"].ge(0)
    data["dollar_volume"] = (
        data["prc"].abs() * data["vol"]
    ).where(valid)


    dates = pd.DatetimeIndex(data["date"].unique()).sort_values()
    month_ends = (
        dates.to_series()
        .groupby(dates.to_period("M"))
        .max()
    )

    outputs = []

    for formation_date in month_ends:
        position = dates.get_loc(formation_date)


        if position < 59:
            continue

        window_start = dates[position - 59]

        history = data.loc[
            data["date"].between(window_start, formation_date)
        ]

        liquidity = (
            history.groupby("permno")
            .agg(
                adv60=("dollar_volume", "mean"),
                valid_days=("dollar_volume", "count"),
            )
            .reset_index()
        )

        print(f"liquidity {liquidity}")




        snapshot = data.loc[
            data["date"].eq(formation_date),
            ["permno", "prc", "tradingstatusflg"],
        ]
        print(f"snapshot {snapshot}")

        candidates = snapshot.merge(liquidity, on="permno")

        eligible = candidates.loc[
            candidates["tradingstatusflg"].eq("A")
            & candidates["prc"].abs().ge(min_price)
            & candidates["valid_days"].eq(60)
            & candidates["adv60"].gt(0)
        ]


        selected = (
            eligible.sort_values(
                ["adv60", "permno"],
                ascending=[False, True],
            )
            .head(n_stocks)
            .copy()
        )

        selected["formation_date"] = formation_date

        selected["holding_month"] = (
            formation_date.to_period("M") + 1
        ).to_timestamp()

        selected["liquidity_rank"] = range(1, len(selected) + 1)

        outputs.append(selected[OUTPUT_COLUMNS])

    if not outputs:
        raise ValueError("Need at least 60 trading dates")

    return pd.concat(outputs, ignore_index=True)


def main() -> None:
    columns = ["permno", "date", "prc", "vol", "tradingstatusflg"]

    frames = [
        pd.read_parquet(
            DATA_DIR / f"crsp_{year}0101_{year}1231.parquet",
            columns=columns,
        )
        for year in range(2009, 2026)
    ]

    bars = pd.concat(frames, ignore_index=True)
    universe = build_monthly_universe(bars)

    universe = universe.loc[
        universe["holding_month"].dt.year.eq(2010)
    ].reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    universe.to_parquet(OUTPUT_PATH, index=False)

    print(universe.groupby("holding_month").size())
    print(f"Saved {len(universe):,} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

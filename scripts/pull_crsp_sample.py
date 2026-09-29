from __future__ import annotations

import os
from pathlib import Path

import wrds


TICKERS = (
    "AAPL",
    "MSFT",
    "AMZN",
    "NVDA",
    "META",
    "GOOGL",
    "JPM",
    "XOM",
    "JNJ",
    "WMT",
)

START_DATE = "2024-01-01"
END_DATE = "2024-06-30"

OUTPUT_PATH = Path(
    "data/raw/wrds/crsp/daily_bars_2024_h1.parquet"
)
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


SQL = """
WITH selected_names AS (
    SELECT
        permno,
        permco,
        ticker,
        namedt,
        nameenddt
    FROM crsp_a_stock.stocknames_v2
    WHERE ticker IN %(tickers)s
)
SELECT
    d.permno,
    d.permco,
    n.ticker,
    d.dlycaldt AS session_date,
    d.primaryexch,

    CAST(d.dlyopen AS DOUBLE PRECISION) AS open_price_raw,
    CAST(d.dlyhigh AS DOUBLE PRECISION) AS high_price_raw,
    CAST(d.dlylow AS DOUBLE PRECISION) AS low_price_raw,
    CAST(d.dlyclose AS DOUBLE PRECISION) AS close_price_raw,
    CAST(d.dlyvol AS BIGINT) AS volume_raw,

    CAST(d.dlyret AS DOUBLE PRECISION) AS total_return,
    CAST(d.dlyretx AS DOUBLE PRECISION) AS price_return,

    CAST(d.dlycumfacpr AS DOUBLE PRECISION)
        AS cumulative_price_factor,
    CAST(d.dlycumfacshr AS DOUBLE PRECISION)
        AS cumulative_share_factor,

    CAST(d.dlycap AS DOUBLE PRECISION) AS market_cap,
    CAST(d.shrout AS BIGINT) AS shares_outstanding

FROM crsp_a_stock.dsf_v2 AS d

INNER JOIN selected_names AS n
    ON d.permno = n.permno
    AND d.dlycaldt >= n.namedt
    AND d.dlycaldt <= COALESCE(
        n.nameenddt,
        DATE '9999-12-31'
    )

WHERE d.dlycaldt BETWEEN %(start_date)s AND %(end_date)s
  AND d.sharetype = 'NS'
  AND d.securitytype = 'EQTY'
  AND d.securitysubtype = 'COM'
  AND d.usincflg = 'Y'
  AND d.issuertype IN ('ACOR', 'CORP')
  AND d.tradingstatusflg = 'A'

ORDER BY
    d.permno,
    d.dlycaldt
"""


def main() -> None:
    username = os.getenv("WRDS_USERNAME")
    if not username:
        raise RuntimeError(
            "WRDS_USERNAME environment variable is not set"
        )

    connection = wrds.Connection(wrds_username=username)

    try:
        bars = connection.raw_sql(
            SQL,
            params={
                "tickers": TICKERS,
                "start_date": START_DATE,
                "end_date": END_DATE,
            },
            date_cols=["session_date"],
            chunksize=None,
        )

        bars.to_parquet(OUTPUT_PATH, index=False)

        print(bars.head())
        print(
            f"Saved {len(bars):,} rows to "
            f"{OUTPUT_PATH.resolve()}"
        )
    finally:
        connection.close()


if __name__ == "__main__":
    main()
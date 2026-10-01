from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd
import wrds


OUTPUT_DIR = Path("data/raw/wrds/crsp/yearly")


class CRSPDataLoader:
    def __init__(self, wrds_username: str = "bohanhou") -> None:
        self.db = wrds.Connection(wrds_username=wrds_username)

    def close(self) -> None:
        self.db.close()

    def fetch_data(
        self,
        start_date: str = "2000-01-01",
        end_date: str = "2025-12-31",
        min_price: float = 5.0,
    ) -> None:
        start = date.fromisoformat(start_date)
        end = date.fromisoformat(end_date)

        if start > end:
            raise ValueError("start_date must be on or before end_date")

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        for year in range(start.year, end.year + 1):
            chunk_start = max(start, date(year, 1, 1))
            chunk_end = min(end, date(year, 12, 31))

            print(f"Downloading {chunk_start} through {chunk_end}...")

            frame = self.db.raw_sql(
                """
                SELECT
                    permno,
                    dlycaldt AS date,
                    CAST(dlyret AS DOUBLE PRECISION) AS ret,
                    CAST(dlyprc AS DOUBLE PRECISION) AS prc,
                    CAST(dlyvol AS BIGINT) AS vol
                FROM crsp_a_stock.dsf_v2
                WHERE dlycaldt BETWEEN %(chunk_start)s AND %(chunk_end)s
                  AND sharetype = 'NS'
                  AND securitytype = 'EQTY'
                  AND securitysubtype = 'COM'
                  AND primaryexch IN ('N', 'A', 'Q')
                  AND tradingstatusflg = 'A'
                ORDER BY permno, dlycaldt
                """,
                params={
                    "chunk_start": chunk_start,
                    "chunk_end": chunk_end,
                },
                date_cols=["date"],
                chunksize=None,
            )

            frame = self._clean_data(frame, min_price=min_price)
            output_path = OUTPUT_DIR / f"crsp_{year}.parquet"
            frame.to_parquet(output_path, index=False)

            print(f"Saved {len(frame):,} rows to {output_path}")

    @staticmethod
    def _clean_data(frame: pd.DataFrame, min_price: float) -> pd.DataFrame:
        frame = frame.copy()
        frame["prc"] = frame["prc"].abs()

        if min_price > 0:
            frame = frame.loc[frame["prc"] >= min_price].copy()

        frame["ret"] = frame["ret"].fillna(0.0)
        frame = frame.sort_values(["permno", "date"]).reset_index(drop=True)
        frame["permno"] = frame["permno"].astype(int)

        return frame[["permno", "date", "ret", "prc", "vol"]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download CRSP CIZ daily common-stock data by year."
    )
    parser.add_argument("--username", default="bohanhou")
    parser.add_argument("--start-date", default="2000-01-01")
    parser.add_argument("--end-date", default="2025-12-31")
    parser.add_argument("--min-price", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    loader = CRSPDataLoader(wrds_username=args.username)

    try:
        loader.fetch_data(
            start_date=args.start_date,
            end_date=args.end_date,
            min_price=args.min_price,
        )
    finally:
        loader.close()


if __name__ == "__main__":
    main()

"""Build month-end features and calendar-aligned next-month return labels."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data/raw/wrds/crsp/yearly_v2"
OUTPUT_PATH = PROJECT_ROOT / "data/silver/features.parquet"


FEATURE_COLUMNS = [
    "reversal_1m",
    "mom_12m",
    "vol_1m",
    "log_adv_1m",
]



#build monthly data based on daily data we have
def aggregate_monthly(daily):
    daily["date"]=pd.to_datetime(daily["date"])

    invalid_return = daily["ret"].notna()

    daily["month"]=(
        daily["date"].dt.to_period("M").dt.to_timestamp()
    )
    daily["gross_return"]=1+daily["ret"]
    valid_volume = (
        np.isfinite(daily["prc"])
        & (daily["prc"].abs() > 0)
        & np.isfinite(daily["vol"])
        & (daily["vol"] >= 0)
    )
    daily["dollar_volume"]=(daily["prc"].abs()*daily["vol"]).where(valid_volume)
    calendar = daily.groupby("month").agg(
        market_days=("date", "nunique"),
        formation_date=("date", "max"),
    )

    grouped = daily.groupby(["permno", "month"])
    monthly=grouped.agg(
        observed_vol_1m=("ret","std"),
        observed_adv_1m=("dollar_volume", "mean"),
        n_days=("date", "nunique"),
        n_valid_returns=("ret", "count"),
        n_valid_dollar_volume=("dollar_volume", "count"),
    )

    monthly["observed_return"]=grouped["gross_return"].prod()-1
    monthly = monthly.reset_index().merge(
        calendar,
        on="month",
        validate="many_to_one",
    )
    
    return monthly


#add_feature(based on monthly results)
def add_features(monthly):
    panel = (monthly.sort_values(["permno","month"])).reset_index(drop=True).copy()
    panel["ret_1m"]=panel["observed_return"]
    panel["reversal_1m"] = -panel["ret_1m"]
    panel["vol_1m"] = panel["observed_vol_1m"]
    panel["adv_1m"] = panel["observed_adv_1m"]
    panel["log_adv_1m"] = np.log1p(panel["adv_1m"])


    months = pd.date_range(panel["month"].min(),panel["month"].max(),freq="MS")

    returns = panel.pivot(index="month",columns="permno",values="ret_1m").reindex(months)

    momentum = (
        (1 + returns)
        .shift(1)
        .rolling(window=11, min_periods=11)
        .apply(np.prod, raw=True)
        - 1
    )
    print(momentum)
    row_index = pd.MultiIndex.from_frame(
        panel[["month", "permno"]]
    )
    panel["mom_12m"]=momentum.stack().reindex(row_index).to_numpy()
    print(panel)
    panel["holding_month"]=panel["month"]+pd.offsets.MonthBegin(1)
    targets = panel[[
        "permno",
        "month",
        "ret_1m",
        "observed_return",
        "formation_date",
    ]].rename(columns={
        "month": "holding_month",
        "ret_1m": "target_ret_1m",
        "observed_return": "next_observed_return",
        "formation_date": "label_end_date",
    })

    print(panel)
    panel = panel.merge(
        targets,
        on=["permno", "holding_month"],
        how="left",
        validate="one_to_one",
    )
    print(panel)
    return panel 


def build_month_data(
        start_year=2009,
        end_year=2024,
        data_dir=DATA_DIR,
        output_path=OUTPUT_PATH
):
    frames = []
    for year in range(start_year, end_year + 1):
        path = (
            Path(data_dir)
            / f"crsp_{year}0101_{year}1231.parquet"
        )
        daily = pd.read_parquet(path)
        monthly=aggregate_monthly(daily)
        frames.append(monthly)

    monthly_panel = pd.concat(frames, ignore_index=True)
    panel=add_features(monthly_panel)
    panel.to_parquet(output_path,index=False)

if __name__ =="__main__":
    data = build_month_data(
        start_year=2009,
        end_year=2024,
        output_path=OUTPUT_PATH
    )

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from build_panel import build_research_panel
from sklearn.linear_model import RidgeCV

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
SILVER_DIR = PROJECT_ROOT / "data/silver"
FEATURES_PATH = SILVER_DIR / "features.parquet"

features= pd.read_parquet(FEATURES_PATH)


FEATURE_COLUMNS = [
    "reversal_1m",
    "mom_12m",
    "vol_1m",
    "log_adv_1m",
    "mom_6_2",
    "maxret_1m",
    "amihud_1m",
]

panel = build_research_panel()
panel["holding_month"]=pd.to_datetime(panel["holding_month"])
panel[FEATURE_COLUMNS] = panel[FEATURE_COLUMNS].replace([np.inf,-np.inf],np.nan)
has_target=np.isfinite(panel["target_ret_1m"])
print(panel)

TRAIN_YEARS = 5
FIRST_DATA_YEAR= int(panel["holding_month"].dt.year.min())
FIRST_TEST_YEAR = FIRST_DATA_YEAR+TRAIN_YEARS
LAST_TEST_YEAR =int(panel["holding_month"].dt.year.max())
prediction_frames=[]
for test_year in range(FIRST_TEST_YEAR,LAST_TEST_YEAR+1):
    train_start = pd.Timestamp(
        year=test_year-TRAIN_YEARS,
        month=1,
        day=1
    )
    test_start = pd.Timestamp(
        year=test_year,
        month=1,
        day=1
    )
    test_end = pd.Timestamp(
            year=test_year+1,
            month=1,
            day=1
        )
    train=panel.loc[
        (panel["holding_month"]>=train_start)&(panel["holding_month"]<test_start)&has_target
    ].copy()

    current_test=panel.loc[
        (panel["holding_month"]>=test_start)
        &(panel["holding_month"]<test_end)
    ].copy()

    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        RidgeCV(
            alphas=[0.01, 0.1, 1.0, 10.0, 100.0],
            scoring="neg_mean_squared_error",
        ),
    )

    model.fit(train[FEATURE_COLUMNS],
              train["target_ret_1m"])
    best_alpha = model.named_steps["ridgecv"].alpha_

    print(
        f"预测年份：{test_year}，"
        f"选择的 alpha：{best_alpha}"
    )
    current_test["prediction"] = model.predict(
        current_test[FEATURE_COLUMNS]
    )

    #benchmark
    current_test["momentum_score"]= current_test["mom_12m"].fillna(train["mom_12m"].median())
    prediction_frames.append(current_test)

    print(
    f"train：{test_year - TRAIN_YEARS}–{test_year - 1}，"
    f"predict：{test_year}，"
    f"train sample size：{len(train):,}，"
    f"test sample size：{len(current_test):,}"
)

test = pd.concat(prediction_frames).sort_values(["holding_month","permno"])


months = pd.date_range(f"{FIRST_TEST_YEAR}-01-01",f"{LAST_TEST_YEAR}-12-01",freq="MS",name="holding_month")

missing_months = months.difference(
    pd.DatetimeIndex(test["holding_month"].unique())
)

if len(missing_months) > 0:
    raise ValueError(f"doesn't have pred data：{missing_months}")
#note train the model based on feature columns and the target return, so each prediction corresponding predicted next
#month return, which means we can rank current month(holding month) based on prediction and trade current month(no look ahead bias)

ranked=test.sort_values(["holding_month","prediction","permno"],ascending=[True,False,True])

#rank the stocks for each month based on ridge prediction, note the 
ranked["rank"]=(
    ranked.groupby("holding_month").cumcount()+1
)

ranked["n_stocks"]=(
    ranked.groupby("holding_month")["permno"].transform("size")

)

ranked["n_each"]=np.floor(
    ranked["n_stocks"]*0.10
)

longs = ranked.loc[
    ranked["rank"]<=ranked["n_each"]
].copy()

longs["weight"]=(
    0.5/longs.groupby("holding_month")["permno"].transform("size")
)

shorts = ranked.loc[
    ranked["rank"]>ranked["n_stocks"]-ranked["n_each"]
].copy()

shorts["weight"]=(
    -0.5/shorts.groupby("holding_month")["permno"].transform("size")
)

#we select the top 20% stocked for those whose prediction are positive
selected = pd.concat([longs,shorts])
print(selected)



realized= features[["permno","month","observed_return","n_valid_returns"]].copy()
realized["observed_return"] = realized["observed_return"].where(
    realized["n_valid_returns"] > 0
)
returns = realized.pivot(
    index="month",
    columns="permno",
    values="observed_return"
)



weights=selected.pivot(
    index="holding_month",
    columns="permno",
    values = "weight",
).reindex(index=months).fillna(0.0)



returns = returns.reindex(
    index= weights.index,
    columns=weights.columns
).fillna(0)
invalid_held = (weights > 0) & (
    returns.isna() | ~np.isfinite(returns)
)






from backtest import engine

result = engine.backtest(
    returns=returns,
    weights=weights,
    transaction_cost_bps=15,
    initial_nav=1.0,
    plot=False,
)

benchmark =test[[
    "holding_month",
    "permno",
    "momentum_score"
]].sort_values(
    ["holding_month","momentum_score","permno"],ascending=[True,False,True].copy()
)
grouped = benchmark.groupby("holding_month")
benchmark["rank"]=grouped.cumcount()+1
benchmark["n_stocks"]=grouped["permno"].transform("size")

benchmark["n_each"]=np.floor(benchmark["n_stocks"]*0.1).astype(int)
benchmark["weight"]=np.select(
    [
        benchmark["rank"]<=benchmark["n_each"],
        benchmark["rank"]>(benchmark["n_stocks"]-benchmark["n_each"]),

    ],[0.5/benchmark["n_each"],
       -0.5/benchmark["n_each"]
    ],
    default=0.0
)

benchmark_weights = benchmark.pivot(
    index="holding_month",
    columns="permno",
    values="weight"
).reindex(index=months).fillna(0.0)
benchmark_returns = realized.pivot(
    index="month",
    columns="permno",
    values="observed_return",
).reindex(
    index=benchmark_weights.index,
    columns=benchmark_weights.columns,
).fillna(0.0)

benchmark_result = engine.backtest(
    returns=benchmark_returns,
    weights=benchmark_weights,
    transaction_cost_bps=15,
    initial_nav=1.0,
    plot=False,
)

import matplotlib.pyplot as plt

nav_comparison = pd.DataFrame({
    "Ridge":result["nav"],
    "Momentum Benchmark": benchmark_result["nav"]
})

nav_comparison.plot(
    figsize=(12,6),
    title="Ridge vs Momentum(benchmark)",
    ylabel="NAV",
    xlabel="Holding month",
    grid=True,
)
plt.tight_layout()
plt.show()

def summarize(bt):
    monthly_returns = bt["portfolio_return"].astype(float)
    monthly_std = monthly_returns.std(ddof=1)

    return pd.Series({
        "CAGR": (
            bt["nav"].iloc[-1] ** (12 / len(bt)) - 1
        ),
        "Sharpe": (
            monthly_returns.mean() / monthly_std * np.sqrt(12)
            if monthly_std > 0 else np.nan
        ),
        "max_drawdown": bt["drawdown"].min(),
        "avg_monthly_turnover": bt["turnover"].mean(),
    })


comparison = pd.DataFrame({
    "Ridge": summarize(result),
    "Momentum": summarize(benchmark_result),
}).T

print(comparison.round(4))

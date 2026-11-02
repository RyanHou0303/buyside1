import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from build_panel import build_research_panel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
SILVER_DIR = PROJECT_ROOT / "data/silver"
FEATURES_PATH = SILVER_DIR / "features.parquet"

features= pd.read_parquet(FEATURES_PATH)
print(features.columns)




FEATURE_COLUMNS = [
    "reversal_1m",
    "mom_12m",
    "vol_1m",
    "log_adv_1m",
]


panel = build_research_panel()

panel["holding_month"]=pd.to_datetime(panel["holding_month"])

panel[FEATURE_COLUMNS] = panel[FEATURE_COLUMNS].replace([np.inf,-np.inf],np.nan)
has_target=np.isfinite(panel["target_ret_1m"])


train = panel.loc[
    panel["holding_month"].between("2010-01-01","2014-12-01")
    &has_target].copy()

#check whether you have the data or not in test
test = panel.loc[
    panel["holding_month"].between("2015-01-01", "2015-12-01")
   
].copy()

#missing month check

months = pd.date_range("2015-01-01","2015-12-01",freq="MS",name="holding_month")

missing_months = months.difference(
    pd.DatetimeIndex(test["holding_month"].unique())
)

if len(missing_months) > 0:
    raise ValueError(f"doesn't have pred data：{missing_months}")

model = make_pipeline(SimpleImputer(strategy="median"),
                      StandardScaler(),
                      Ridge(alpha=0.1))


model.fit(
    train[FEATURE_COLUMNS],
    train["target_ret_1m"],
)
test["prediction"] = model.predict(test[FEATURE_COLUMNS])
#note train the model based on feature columns and the target return, so each prediction corresponding predicted next
#month return, which means we can rank current month(holding month) based on prediction and trade current month(no look ahead bias)

positive = test.loc[
    test["prediction"]>0
].copy()

positive=positive.sort_values(["holding_month","prediction","permno"],ascending=[True,False,True],)
#rank the stocks for each month based on ridge prediction, note the 
positive["rank"]=(
    positive.groupby("holding_month").cumcount()+1
)

positive["n_positive"]=(
    positive.groupby("holding_month")["permno"].transform("size")

)
positive["n_select"]=np.ceil(
    positive["n_positive"]*0.2
)
#we select the top 20% stocked for those whose prediction are positive
selected = positive.loc[
    positive["rank"]<=positive["n_select"]
].copy()

#equal weights
selected["weight"]=1.0/selected.groupby("holding_month")["permno"].transform("size")

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


returns = returns.where(weights > 0, 0.0)



from backtest import engine

result = engine.backtest(
    returns=returns,
    weights=weights,
    transaction_cost_bps=0,
    initial_nav=1.0,
    plot=True,
)




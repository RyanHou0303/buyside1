import pandas as pd
import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from build_panel import build_research_panel


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
    & has_target
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
                      Ridge(alpha=0))


model.fit(
    train[FEATURE_COLUMNS],
    train["target_ret_1m"],
)
test["prediction"] = model.predict(test[FEATURE_COLUMNS])


positive = test.loc[
    test["prediction"]>0
].copy()

positive=positive.sort_values(["holding_month","prediction","permno"],ascending=[True,False,True],)
positive["rank"]=(
    positive.groupby("holding_month").cumcount()+1
)

positive["n_positive"] = (
    positive.groupby("holding_month")["permno"]
    .transform("size")
)
print(positive.loc[positive["holding_month"]=="2015-02"])





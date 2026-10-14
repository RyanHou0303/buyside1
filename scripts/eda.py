import pandas as pd
data=pd.read_parquet("/Users/bohanhou/Desktop/COLUMBIA/buyside1/data/raw/wrds/crsp/yearly_v2/crsp_20000101_20001231.parquet")
data=pd.read_parquet("/Users/bohanhou/Desktop/COLUMBIA/buyside1/data/silver/universe_2010.parquet")
data_equities=pd.read_parquet("/Users/bohanhou/Desktop/COLUMBIA/buyside1/data/raw/wrds/crsp/yearly_v2/crsp_20000101_20001231.parquet")
print(data_equities["primaryexch"])
print(data.columns)
print(data.loc[data["permno"]==14593])

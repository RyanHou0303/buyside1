import pandas as pd
data=pd.read_parquet("/Users/bohanhou/Desktop/COLUMBIA/buyside1/data/silver/features.parquet")

pd.set_option('display.max_columns', None)


print(data)



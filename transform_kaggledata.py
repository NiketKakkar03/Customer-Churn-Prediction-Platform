import pandas as pd
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
df.to_parquet("data/raw/telco_churn.parquet", index=False)
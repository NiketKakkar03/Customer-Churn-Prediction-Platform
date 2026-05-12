import joblib, pandas as pd
pipe = joblib.load("models/champion.joblib")
# Should print something like: Pipeline(steps=[('pre', ColumnTransformer(...)), ('clf', LogisticRegression(...))])
print(pipe)

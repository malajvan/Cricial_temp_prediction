import pandas as pd

train_set=pd.read_csv("data/test.csv").dropna(axis=0)
print(train_set['index'])

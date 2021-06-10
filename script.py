import pandas as pd
df = pd.read_csv("unlabeled.csv")

for index, row in df.iterrows():
    if row.any() == None:
        print("found it ")
import pandas as pd
df = pd.read_csv("unlabeled.csv")

for index, row in df.iterrows():
    label, text = row
    print("found it ")
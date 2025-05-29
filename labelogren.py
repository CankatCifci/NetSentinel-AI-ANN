import pandas as pd

df = pd.read_csv("Darknet.CSV", nrows=5)
print(df.columns.tolist())

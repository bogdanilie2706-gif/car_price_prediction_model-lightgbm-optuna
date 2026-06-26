import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT_DIR / "data" / "processed" / "dataframe.csv", low_memory=False)

df["body_type"].map(type).value_counts()
print(df["body_type"].value_counts(dropna=False).head(30))

import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"

ge = pd.read_csv(PROCESSED_DATA_DIR / "germany.csv")
mc = pd.read_csv(PROCESSED_DATA_DIR / "maroc.csv")
us = pd.read_csv(PROCESSED_DATA_DIR / "us.csv")
uk = pd.read_csv(PROCESSED_DATA_DIR / "uk.csv")

temp = [ge, mc, us, uk]

df = pd.concat(temp, ignore_index = True)

df['brand'] = df['brand'].str.lower()


df.to_csv(PROCESSED_DATA_DIR / "dataframe.csv", index=False)

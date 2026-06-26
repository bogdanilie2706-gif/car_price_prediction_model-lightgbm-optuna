import pandas as pd
import numpy as np
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"

df = pd.read_csv(PROCESSED_DATA_DIR / "dataframe.csv", low_memory=False)

# elimin branduri cu mai putin de 100 aparitii
brand_counts = df['brand'].value_counts()
df = df[df['brand'].isin(brand_counts[brand_counts >= 50].index)]

# fac la fel si pentru modele
model_counts = df['model'].value_counts()
df = df[df['model'].isin(model_counts[model_counts >= 50].index)]

df['model'] = df['model'].str.replace('_', ' ').str.replace('-', ' ').str.lower()
df['brand'] = df['brand'].str.replace('_', ' ').str.replace('-', ' ').str.lower()


df.loc[(df['engine_power'] < 20) | (df['engine_power'] > 1000), 'engine_power'] = np.nan
df.loc[(df['engine_size'] <= 0) & (df['fuel'] != 'electric'), 'engine_size'] = np.nan


def impute_condition(dataframe):
    dataframe['condition_was_missing'] = dataframe['condition'].isna().astype(int)

    dataframe['age_group'] = pd.cut(
        dataframe['age'],
        bins=[-1, 2, 5, 10, 15, 20, np.inf],
        labels=['0-2', '3-5', '6-10', '11-15', '16-20', '21+']
    )
    dataframe['km_group'] = pd.cut(
        dataframe['km'],
        bins=[-1, 50000, 100000, 150000, 250000, np.inf],
        labels=['0-50k', '50k-100k', '100k-150k', '150k-250k', '250k+']
    )

    dataframe['condition'] = dataframe['condition'].fillna(
        dataframe.groupby(['age_group', 'km_group'], observed=True)['condition'].transform('mean')
    )
    dataframe['condition'] = dataframe['condition'].fillna(
        dataframe.groupby('age_group', observed=True)['condition'].transform('mean')
    )
    dataframe['condition'] = dataframe['condition'].fillna(dataframe['condition'].mean())
    dataframe['condition'] = dataframe['condition'].clip(0, 1)

    return dataframe.drop(columns=['age_group', 'km_group'])


def impute_numeric_by_groups(dataframe, column):
    dataframe[f'{column}_was_missing'] = dataframe[column].isna().astype(int)

    group_columns = [
        ['brand', 'model', 'fuel'],
        ['brand', 'model'],
        ['model'],
        ['brand'],
    ]

    for columns in group_columns:
        dataframe[column] = dataframe[column].fillna(
            dataframe.groupby(columns)[column].transform('median')
        )

    dataframe[column] = dataframe[column].fillna(dataframe[column].median())

    return dataframe


df = impute_condition(df)
df = impute_numeric_by_groups(df, 'engine_power')
df = impute_numeric_by_groups(df, 'engine_size')

df.to_csv(PROCESSED_DATA_DIR / "dataframe_modified.csv", index=False)


print(df['fuel'].value_counts())
print(df['brand'].value_counts())
print(df['age'].value_counts())
print(df['model'].value_counts())
print(df['condition'].value_counts())
print(df['transmission'].value_counts())
print(df['body_type'].value_counts())

print(df.shape)
print(df.sample(20).to_string())

print(df.isna().sum())
print(df.describe().to_string())

print(df.shape)

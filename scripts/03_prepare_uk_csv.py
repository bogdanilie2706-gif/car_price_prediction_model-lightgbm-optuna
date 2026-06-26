import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

uk_files = {
    "audi.csv": "Audi",
    "bmw.csv": "BMW",
    "ford.csv": "Ford",
    "hyundai.csv": "Hyundai",
    "merc.csv": "Mercedes",
    "skoda.csv": "Skoda",
    "toyota.csv": "Toyota",
    "vw.csv": "Volkswagen",
    "vauxhall.csv": "Opel"
}

uk_temp = []
for file_name, brand_name in uk_files.items():
    temp = pd.read_csv(RAW_DATA_DIR / "UK" / file_name)
    temp.columns = temp.columns.str.strip()
    temp = temp.filter(items =
        ['model', 'year', 'price', 'transmission', 'mileage', 'fuelType', 'engineSize'])
    temp['brand'] = brand_name
    uk_temp.append(temp)

uk = pd.concat(uk_temp, ignore_index=True)

uk = uk.rename(columns={
    'mileage': 'km',
    'fuelType': 'fuel',
    'engineSize': 'engine_size'
})

print(uk.shape)

uk['age'] = 2020 - uk['year']
uk = uk[(uk['age'] >= 0) & (uk['age'] <= 30)]
uk = uk.drop(columns = ['year'])

uk['brand'] = uk['brand'].str.lower()

uk['model'] = uk['model'].str.strip().str.lower()
uk['model'] = uk['model'].fillna('unknown')

uk['km'] = (uk['km'] * 1.6).astype(int) # din mile in km

uk['price'] = (uk['price'] * 1.16).astype(int) # din lire in euro
uk = uk[(uk['price'] >= 500) & (uk['price'] <= 350000)]

uk['fuel'] = uk['fuel'].str.strip().str.lower().map({
    'petrol': 'petrol',
    'diesel': 'diesel',
    'electric': 'electric',
    'hybrid': 'hybrid',
    'other': None
})
uk['fuel'] = uk['fuel'].fillna('unknown')

uk['transmission'] = uk['transmission'].str.strip().str.lower().map({
    'automatic': 'automatic',
    'manual': 'manual',
    'semi-auto': 'automatic', # considerat tot automata
    'other': None
})
uk['transmission'] = uk['transmission'].fillna('unknown')

# coloane lipsa
uk['condition'] = None
uk['engine_power'] = None
uk['body_type'] = 'unknown'

cols = ['brand', 'model', 'age', 'km', 'fuel', 'transmission',
        'price', 'condition', 'engine_power', 'engine_size', 'body_type']
uk = uk[cols]

uk.to_csv(PROCESSED_DATA_DIR / "uk.csv", index=False)

print(uk.shape)
print(uk.sample(20).to_string())

print(uk.isna().sum())

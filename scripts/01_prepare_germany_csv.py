import pandas as pd
import numpy as np
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

germany = pd.read_csv(RAW_DATA_DIR / "germany.csv", usecols =
    ['brand', 'model', 'yearOfRegistration', 'kilometer', 'fuelType', 'gearbox', 'powerPS', 'vehicleType', 'price', 'notRepairedDamage'],
    encoding= 'latin-1')

germany = germany.rename(columns={
    'yearOfRegistration': 'year',
    'kilometer': 'km',
    'fuelType': 'fuel',
    'gearbox': 'transmission',
    'powerPS': 'engine_power',
    'vehicleType': 'body_type',
    'notRepairedDamage': 'damage',
})

germany = germany.dropna(subset = ['year', 'brand', 'km'])

germany['age'] = 2018 - germany['year']
germany = germany.drop(columns = ['year'])
germany = germany[(germany['age'] >= 0) & (germany['age'] <= 30)]

germany['engine_power'] = germany['engine_power'].replace(0, None) # in cazul in care nu este specificat

germany['brand'] = germany['brand'].str.title()

germany['fuel'] = germany['fuel'].str.lower().map({ # traducere
    'benzin': 'petrol',
    'diesel': 'diesel',
    'elektro': 'electric',
    'hybrid': 'hybrid',
    'andere': None,
    'lpg': 'lpg',
    'cng': 'lpg' #ambele sunt o varianta de gaz
})
germany['fuel'] = germany['fuel'].fillna('unknown')

germany['transmission'] = germany['transmission'].str.lower().map({ # traducere
    'manuell': 'manual',
    'automatik': 'automatic'
})
germany['transmission'] = germany['transmission'].fillna('unknown')

germany['body_type'] = germany['body_type'].str.lower().map({ # traducere
    'limousine': 'sedan',
    'kombi': 'estate',
    'kleinwagen': 'hatchback',
    'suv': 'suv',
    'coupe': 'coupe',
    'cabrio': 'convertible',
    'bus': 'van',
    'andere': None
})
germany['body_type'] = germany['body_type'].fillna('unknown')

# pretul este deja in euro, elimin preturi nefavorabile
germany = germany[(germany['price'] >= 500) & (germany['price'] <= 350000)]

# coloane lipsa
germany['condition'] = None
germany['engine_size'] = None

germany['model'] = germany['model'].str.replace('_', ' ').str.replace('-', ' ').str.lower()

germany['model'] = germany['model'].replace({
    'andere': None,

    # BMW - format "Xer"
    '1er': '1 series',
    '2er': '2 series',
    '3er': '3 series',
    '4er': '4 series',
    '5er': '5 series',
    '6er': '6 series',
    '7er': '7 series',

    # BMW - format "X reihe"
    '1 reihe': '1 series',
    '2 reihe': '2 series',
    '3 reihe': '3 series',
    '4 reihe': '4 series',
    '5 reihe': '5 series',
    '6 reihe': '6 series',
    'x reihe': 'x series',
    'i reihe': 'i series',
    'm reihe': 'm series',
    'z reihe': 'z series',

    # Mercedes - format "X klasse"
    'a klasse': 'a class',
    'b klasse': 'b class',
    'c klasse': 'c class',
    'e klasse': 'e class',
    'g klasse': 'g class',
    'm klasse': 'm class',
    's klasse': 's class',
    'v klasse': 'v class',

    # Mazda - format "X reihe"
    'mx reihe': 'mx series',
    'cx reihe': 'cx series',

    # Volvo - format "X reihe"
    'xc reihe': 'xc series',
    'rx reihe': 'rx series',
    'c reihe': 'c series',

    # Land Rover
    'rangerover': 'range rover',

    'kaefer': 'beetle',  # "Käfer" = numele german pentru VW Beetle
    'serie 2': '2 series',  # variantă greșit scrisă

    # Honda - format "X reihe"
    'cr reihe': 'cr series',
})
germany['model'] = germany['model'].fillna('unknown')

def damage_to_condition(damage):
    if damage == 'ja':
        return np.clip(np.random.normal(0.2, 0.08), 0, 1)
    elif damage == 'nein':
        return np.clip(np.random.normal(0.8, 0.08), 0, 1)
    else:
        return np.nan

germany['condition'] = germany['damage'].apply(damage_to_condition)

germany = germany.drop(columns = ['damage'])

cols = ['brand', 'model', 'age', 'km', 'fuel', 'transmission',
        'price', 'condition', 'engine_power', 'engine_size', 'body_type']
germany = germany[cols]


print(germany['fuel'].value_counts())
print(germany['age'].value_counts())
print(germany['brand'].value_counts())
print(germany['model'].value_counts().to_string())
print(germany['condition'].value_counts())
print(germany['transmission'].value_counts())
print(germany['body_type'].value_counts())

germany.to_csv(PROCESSED_DATA_DIR / "germany.csv", index=False)

print(germany.shape)
print(germany.sample(20).to_string())

print(germany.isna().sum())
print(germany.describe())

print(germany.shape)

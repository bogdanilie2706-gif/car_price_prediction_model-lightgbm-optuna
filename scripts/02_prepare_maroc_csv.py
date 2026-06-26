import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)

ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

maroc = pd.read_csv(RAW_DATA_DIR / "maroc.csv", usecols = [
    'Brand', 'Model', 'Year', 'Mileage', 'Gearbox', 'Fuel', 'Condition', 'Price'])

maroc = maroc.rename(columns={
    'Brand': 'brand',
    'Model': 'model',
    'Year': 'year',
    'Mileage': 'km',
    'Gearbox': 'transmission',
    'Fuel': 'fuel',
    'Condition': 'condition',
    'Price': 'price'
})

print(maroc.shape)

maroc['year'] = maroc['year'].replace('1980 ou plus ancien', 1970).astype(int)
maroc['age'] = 2024 - maroc['year']
maroc = maroc[(maroc['age'] >= 0) & (maroc['age'] <= 30)].drop(columns = ['year'])

maroc = maroc[maroc['km'].str.contains('-')] # scot liniile cu km care provoaca erori
km_range = maroc['km'].str.replace(' ', '', regex=False).str.split('-', expand=True).astype(int)
maroc['km'] = ((km_range[0] + km_range[1]) // 2)
maroc['km'] = (maroc['km'] * 1.6).astype(int) # trecere in km

maroc['price'] = (maroc['price'] * 0.090).round(2) # trece din dihram marocan in euro
maroc = maroc[(maroc['price'] >= 500) & (maroc['price'] <= 350000)]

maroc['brand'] = maroc['brand'].str.title()
maroc['brand'] = maroc['brand'].replace({'Autres': None})
maroc = maroc.dropna(subset=['brand'])

INITIAL_MODEL_REPLACEMENTS = {
    'autres': None,
    'c elysee': 'c elysee',
    'qq': 'qashqai',
    'qq6': 'qashqai',
    'country man': 'countryman',
    'range rover vogue': 'range rover',
    'land cruiser prado': 'prado',
    'rangerover': 'range rover',
    'slk klasse amg': 'slk amg',
}

FINAL_MODEL_REPLACEMENTS = {
    'a1 sportback': 'a1',
    'a3 sportback': 'a3',
    'a5 sportback': 'a5',
    'q3 sportback': 'q3',
    'q5 sportback': 'q5',

    'class a berline': 'a class',
    'class c cabriolet': 'c class',
    'class c coupe': 'c class',
    'class e cabriolet': 'e class',
    'class e coupe': 'e class',
    'class glc coupe': 'glc',
    'class gle coupe': 'gle',
    'class s coupe': 's class',
    'amg gle coupe': 'gle',

    'megane cabriolet': 'megane',
    'megane cc': 'megane',
    'megane coupe': 'megane',
    'megane estate': 'megane',
    'megane sedan': 'megane',

    'dokker van': 'dokker',
    'doblo classic': 'doblo',
    'doblo panorama': 'doblo',
    'express van': 'express',
    'kangoo express': 'kangoo',
    'logan mcv': 'logan',

    'sandero stepway': 'sandero',
    'sandero streetway': 'sandero',
    'passat cc': 'passat',
    'golf plus': 'golf',

    'c3 aircross': 'c3',
    'c3 cactus': 'c3',
    'c3 picasso': 'c3',
    'c4 aircross': 'c4',
    'c4 cactus': 'c4',
    'c4 picasso': 'c4',
    'c4 spacetourer': 'c4',
    'grand c4 picasso': 'c4',
    'c5 aircross': 'c5',
    'c5x': 'c5',
    'xsara picasso': 'xsara',

    'range rover sport': 'range rover',
    'range rover velar': 'range rover',
    'range rover evoque cabriolet': 'range rover evoque',
    'discovery sport': 'discovery',
    'pajero sport': 'pajero',
    'cayenne coupe': 'cayenne',
    'corolla cross': 'corolla',
    'yaris cross': 'yaris',

    '206 plus': '206',
    '206 cc': '206',
    '206 sw': '206',
    '207 cc': '207',
    '207 sw': '207',
    '307 cc': '307',
    '307 sw': '307',
    '308 cc': '308',
    '308 sw': '308',
    '406 coupe': '406',
    '407 coupe': '407',
    '407 sw': '407',
}

BODY_SUFFIXES = {
    'berline',
    'cabriolet',
    'cargo',
    'coupe',
    'estate',
    'sedan',
    'sportback',
    'sw',
    'tourer',
    'van',
}

MERCEDES_NAMED_MODELS = {
    'cl',
    'cla',
    'clc',
    'clk',
    'cls',
    'gl',
    'gla',
    'glb',
    'glc',
    'gle',
    'glk',
    'gls',
    'ml',
    'sl',
    'slk',
}
MERCEDES_LETTER_CLASSES = {'a', 'b', 'c', 'e', 'g', 'r', 's', 'v'}

def normalize_model(model):
    if pd.isna(model):
        return None

    model = str(model).strip().lower()
    model = (
        model
        .replace('_', ' ')
        .replace('-', ' ')
        .replace('+', ' plus')
        .replace('\u00e3\u00a9', 'e')
        .replace('\u00c3\u00a9', 'e')
        .replace('\u00e9', 'e')
    )
    model = ' '.join(model.split())
    model = INITIAL_MODEL_REPLACEMENTS.get(model, model)

    if model is None:
        return None

    words = model.split()

    if words and words[0] == 'classe':
        words[0] = 'class'

    if len(words) >= 2 and words[0] == 'serie':
        words = [words[1], 'series'] + words[2:]

    model = ' '.join(words)
    return model

def infer_body_type(model, brand):
    if pd.isna(model):
        return None

    words = str(model).split()

    if 'cabriolet' in words:
        return 'convertible'

    if words and words[-1] == 'cc':
        if brand == 'Volkswagen':
            return 'coupe'
        return 'convertible'

    if 'coupe' in words:
        return 'coupe'

    if 'sedan' in words or 'berline' in words:
        return 'sedan'

    if any(word in words for word in ['estate', 'mcv', 'sw', 'tourer']):
        return 'estate'

    if (
        model in ['dokker van', 'express van', 'kangoo express']
        or any(word in words for word in ['cargo', 'minibus', 'van'])
    ):
        return 'van'

    if any(word in words for word in ['aircross', 'cross', 'crossback']):
        return 'suv'

    if 'sportback' in words:
        if words[0].startswith('q') or words[0] == 'e':
            return 'suv'
        return 'hatchback'

    if any(word in words for word in ['stepway', 'streetway']):
        return 'hatchback'

    if any(word in words for word in ['life', 'panorama', 'picasso', 'spacetourer']):
        return 'mpv'

    return None

def simplify_model(model):
    if pd.isna(model):
        return None

    words = str(model).split()
    if len(words) >= 2 and words[1] == 'series':
        return f'{words[0]} series'

    if len(words) > 1 and words[-1] in BODY_SUFFIXES:
        model = ' '.join(words[:-1])
        words = model.split()

    if len(words) > 1 and words[-1] == 'cc':
        model = ' '.join(words[:-1])
        words = model.split()

    model = FINAL_MODEL_REPLACEMENTS.get(model, model)
    words = str(model).split()

    if len(words) >= 2 and words[0] == 'class':
        if words[1] in MERCEDES_NAMED_MODELS:
            return words[1]
        if words[1] in MERCEDES_LETTER_CLASSES:
            return f'{words[1]} class'

    if len(words) >= 2 and words[0] == 'amg' and words[1] in MERCEDES_NAMED_MODELS:
        return words[1]

    return model

maroc['model'] = maroc['model'].apply(normalize_model)
maroc['body_type'] = [
    infer_body_type(model, brand)
    for model, brand in zip(maroc['model'], maroc['brand'])
]
maroc['model'] = maroc['model'].apply(simplify_model)
maroc['model'] = maroc['model'].fillna('unknown')

maroc['fuel'] = maroc['fuel'].str.lower().map({
    'petrol': 'petrol',
    'diesel': 'diesel',
    'electrique': 'electric',
    'hybrid': 'hybrid',
    'lpg' : 'lpg'
})
maroc['fuel'] = maroc['fuel'].fillna('unknown')

maroc['transmission'] = maroc['transmission'].str.lower().map({
    'automatic': 'automatic',
    'manual': 'manual'
})
maroc['transmission'] = maroc['transmission'].fillna('unknown')
maroc['body_type'] = maroc['body_type'].fillna('unknown')

maroc['condition'] = maroc['condition'].str.title()

# coloane lipsa
maroc['engine_power'] = None
maroc['engine_size'] = None

condition_map_maroc = {
    'For Parts': 0.05,
    'Damaged': 0.15,
    'Fair': 0.3,
    'Good': 0.5,
    'Very Good': 0.7,
    'Excellent': 0.85,
    'New': 1.0,
}

def condition_to_float(condition, noise=0.05):
    base = condition_map_maroc.get(condition, np.nan)
    if pd.isna(base):
        return np.nan
    return np.clip(base + np.random.uniform(-noise, noise), 0, 1)

maroc['condition'] = maroc['condition'].apply(condition_to_float)


cols = ['brand', 'model', 'age', 'km', 'fuel', 'transmission',
        'price', 'condition', 'engine_power', 'engine_size', 'body_type']
maroc = maroc[cols]

maroc.to_csv(PROCESSED_DATA_DIR / "maroc.csv", index=False)

print(maroc.shape)
print(maroc.sample(20).to_string())

print(maroc.isna().sum())
print(maroc.describe())

print(maroc['fuel'].value_counts())
print(maroc['age'].value_counts())
print(maroc['brand'].value_counts())
print(maroc['model'].value_counts().to_string())
print(maroc['condition'].value_counts())
print(maroc['transmission'].value_counts())
print(maroc['body_type'].value_counts())

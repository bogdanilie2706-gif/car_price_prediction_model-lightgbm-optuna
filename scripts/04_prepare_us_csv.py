import pandas as pd
import numpy as np
from difflib import get_close_matches
import re
from pathlib import Path

np.random.seed(42)

ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

us = pd.read_csv(RAW_DATA_DIR / "craigslist.csv", usecols = [
    'price', 'year', 'manufacturer', 'model', 'condition', 'fuel', 'odometer', 'transmission', 'type'])

us = us.rename(columns={
    'manufacturer': 'brand',
    'odometer': 'km',
    'type': 'body_type'
}) # redenumite dupa cum va fi standardul

us = us.dropna(subset = ['year', 'brand', 'km'])

us['age'] = 2022 - us['year'].astype(int) # era float
us = us.drop(columns = ['year'])
us = us[(us['age'] >= 0) & (us['age'] <= 30)]

us['price'] = (us['price'] * 0.92).round(2) # din dolari in euro
us = us[(us['price'] >= 500) & (us['price'] <= 350000)] # iau preturile mai valide

us['km'] = us['km'].astype(int)  * 1.6# si el era float
us = us[(us['km'] >= 500) & (us['km'] <= 900000)]

us['brand'] = us['brand'].str.strip().str.lower()
us['model'] = us['model'].str.strip().str.lower()

def clean_text(value):
    if pd.isna(value):
        return None

    value = str(value).lower().strip()
    value = value.replace('&', ' and ')
    value = value.replace('-', ' ')
    value = re.sub(r'[^a-z0-9]+', ' ', value)
    value = re.sub(r'\s+', ' ', value).strip()
    return value

def normalize_common_patterns(text):
    replacements = {
        'f150': 'f 150',
        'f250': 'f 250',
        'f350': 'f 350',
        'f450': 'f 450',
        'f550': 'f 550',
        'f 250sd': 'f 250',
        'rav4': 'rav 4',
        'crv': 'cr v',
        'hrv': 'hr v',
        '3 series': '3 series',
        '3 series 330i': '3 series',
        '3 series 340i': '3 series',
        '3 series 328i': '3 series',
        '3 series 335i': '3 series',
        '4 series 440i': '4 series',
        '1 series 128i': '1 series',
    }

    for old, new in replacements.items():
        text = re.sub(rf'\b{re.escape(old)}\b', new, text)

    return text

KNOWN_MODELS = {
    'ford': [
        'super duty f 550', 'super duty f 450', 'super duty f 350', 'super duty f 250',
        'f 550', 'f 450', 'f 350', 'f 250', 'f 150',
        'c max', 'transit connect', 'crown victoria', 'explorer sport trac',
        'escape', 'explorer', 'expedition', 'edge', 'focus', 'fiesta', 'fusion',
        'mustang', 'ranger', 'taurus', 'transit', 'ecosport', 'flex',
        'econoline', 'excursion', 'thunderbird', 'bronco'
    ],
    'chevrolet': [
        'silverado 3500hd', 'silverado 2500hd', 'silverado 1500',
        'silverado 3500', 'silverado 2500', 'silverado',
        'malibu', 'impala', 'cruze', 'camaro', 'corvette', 'tahoe',
        'suburban', 'colorado', 'traverse', 'equinox', 'express',
        'spark', 'trax', 'avalanche', 'trailblazer', 'cobalt',
        'sonic', 'blazer', 'hhr', 'volt', 'el camino', 'aveo'
    ],
    'gmc': [
        'sierra 3500hd', 'sierra 2500hd', 'sierra 1500',
        'sierra 3500', 'sierra 2500', 'sierra',
        'yukon xl', 'yukon', 'acadia', 'terrain', 'canyon',
        'envoy'
    ],
    'ram': [
        'pickup 2500', 'pickup 1500',
        '1500 classic', '1500', '2500', '3500', '5500'
    ],
    'toyota': [
        'land cruiser', 'fj cruiser', '4runner', 'highlander',
        'camry', 'corolla', 'prius', 'tacoma', 'tundra',
        'rav 4', 'sienna', 'avalon', 'sequoia', 'venza',
        'yaris', 'matrix'
    ],
    'honda': [
        'cr v', 'hr v', 'civic', 'accord', 'pilot', 'odyssey',
        'fit', 'element', 'ridgeline'
    ],
    'jeep': [
        'grand cherokee', 'wrangler unlimited', 'wrangler',
        'cherokee', 'liberty', 'compass', 'patriot',
        'gladiator', 'renegade', 'commander'
    ],
    'bmw': [
        '1 series', '2 series', '3 series', '4 series',
        '5 series', '7 series', 'x1', 'x3', 'x5',
        '328i', '330i', '335i', '340i', '528i', '535i'
    ],
    'mercedes-benz': [
        'c class', 'e class', 's class', 'm class',
        'gl class', 'gla class', 'glc', 'cla', 'ml'
    ],
    'mazda': [
        'mx 5 miata', 'mazda3', 'mazda6', 'cx 5', 'cx 9', '3', '6'
    ],
    'lexus': [
        'rx 350', 'rx 330', 'rx', 'es 350', 'es 300', 'es',
        'ls 460', 'is 250', 'is 350', 'is', 'gs 350', 'gx 460',
        'nx 300'
    ],
    'hyundai': [
        'santa fe', 'sonata', 'elantra', 'tucson', 'accent',
        'veloster', 'genesis'
    ],
    'kia': [
        'soul', 'sportage', 'optima', 'forte', 'sorento',
        'rio', 'sedona'
    ],
    'subaru': [
        'xv crosstrek', 'crosstrek', 'outback', 'forester',
        'impreza', 'legacy', 'wrx'
    ],
    'dodge': [
        'grand caravan', 'charger', 'challenger', 'durango', 'journey',
        'dart', 'dakota', 'avenger', 'caliber'
    ],
    'cadillac': [
        'escalade', 'cts', 'srx', 'xt4', 'xts', 'ats', 'dts', 'deville',
        'ct5'
    ],
    'acura': ['mdx', 'rdx', 'tl', 'tsx', 'tlx', 'ilx'],
    'chrysler': ['town country', 'pacifica', 'pt cruiser', '300', '200', 'sebring', 'aspen'],
    'buick': ['enclave', 'encore', 'lacrosse', 'lucerne', 'lesabre', 'verano', 'regal', 'century'],
    'infiniti': ['qx80', 'qx60', 'qx56', 'qx50', 'q50', 'g37', 'g35'],
    'lincoln': ['navigator', 'town car', 'continental', 'mkz', 'mkx', 'mks', 'mkc'],
    'volvo': ['xc90', 'xc60', 's60'],
    'mitsubishi': ['outlander sport', 'outlander', 'lancer', 'eclipse'],
    'mini': ['cooper countryman', 'countryman', 'cooper hardtop', 'cooper'],
    'pontiac': ['grand prix', 'vibe', 'g6'],
    'jaguar': ['e pace', 'xf'],
    'rover': ['range rover sport', 'range rover evoque', 'discovery sport', 'sport', 'evoque'],
    'porsche': ['cayenne'],
    'saturn': ['vue', 'ion'],
    'tesla': ['model 3', 'model s', 'model x', 'model y'],
    'fiat': ['500'],
    'alfa-romeo': ['stelvio'],
}

DIRECT_MODEL_REPLACEMENTS = {
    'mercedes-benz': {
        'c-class': 'c class',
        'c class': 'c class',
        'c-class c 300': 'c class',
        'benz c300': 'c class',
        'e-class': 'e class',
        'benz e350': 'e class',
        's-class': 's class',
        'm-class': 'ml',
        'gl-class': 'gl',
        'gla-class gla 45': 'gla',
        'benz': None,
    },
    'bmw': {
        '328i': '3 series',
        '330i': '3 series',
        '335i': '3 series',
        '340i': '3 series',
        '528i': '5 series',
        '535i': '5 series',
    },
    'mazda': {
        'mazda3': '3',
        'mazda6': '6',
        'mx-5 miata grand touring': 'mx 5 miata',
        'mx-5 miata club': 'mx 5 miata',
    },
    'chrysler': {
        'town & country': 'town country',
        'town and country': 'town country',
        'pt cruiser': 'pt cruiser',
    },
    'rover': {
        'sport': 'range rover sport',
        'evoque': 'range rover evoque',
        'discovery sport': 'discovery sport',
    },
}

KNOWN_MODELS = {
    brand: sorted({clean_text(model) for model in models}, key=len, reverse=True)
    for brand, models in KNOWN_MODELS.items()
}

DIRECT_MODEL_REPLACEMENTS = {
    brand: {clean_text(model): replacement for model, replacement in rules.items()}
    for brand, rules in DIRECT_MODEL_REPLACEMENTS.items()
}

def infer_us_body_type(raw_model):
    model = clean_text(raw_model)
    if model is None:
        return None

    if any(x in model for x in ['sport utility', 'suv']):
        return 'suv'
    if any(x in model for x in ['pickup', 'crew cab', 'double cab', 'regular cab', 'supercrew', 'super cab', 'quad cab']):
        return 'pickup'
    if any(x in model for x in ['cargo van', 'passenger van']):
        return 'van'
    if 'sedan' in model:
        return 'sedan'
    if 'coupe' in model:
        return 'coupe'
    if 'convertible' in model:
        return 'convertible'
    if 'hatchback' in model:
        return 'hatchback'
    if 'wagon' in model:
        return 'estate'

    return None


def normalize_us_model(row):
    brand = clean_text(row['brand'])
    model = clean_text(row['model'])

    if brand is None or model is None:
        return None

    model = normalize_common_patterns(model)

    if brand == 'bmw':
        if re.search(r'\b(128i)\b', model):
            return '1 series'
        if re.search(r'\b(328i|330i|335i|340i|328 i|330 i|335 i|340 i)\b', model):
            return '3 series'
        if re.search(r'\b(440i|440 i)\b', model):
            return '4 series'
        if re.search(r'\b(528i|535i|528 i|535 i)\b', model):
            return '5 series'

    if brand == 'mercedes-benz':
        if re.search(r'\b(c class|c300|c 300|c 350|c class c 300)\b', model):
            return 'c class'
        if re.search(r'\b(e class|e350|e 350)\b', model):
            return 'e class'
        if re.search(r'\b(s class|s550|s 550)\b', model):
            return 's class'
        if 'm class' in model:
            return 'ml'

    direct_rules = DIRECT_MODEL_REPLACEMENTS.get(brand, {})
    if model in direct_rules:
        return direct_rules[model]

    known = KNOWN_MODELS.get(brand, [])

    for known_model in known:
        if re.search(rf'\b{re.escape(known_model)}\b', model):
            normalized = direct_rules.get(known_model, known_model)

            # BMW numeric trims
            if brand == 'bmw':
                if normalized in {'328i', '330i', '335i', '340i'}:
                    return '3 series'
                if normalized in {'528i', '535i'}:
                    return '5 series'

            # Ford super duty variants
            if brand == 'ford' and normalized.startswith('super duty f '):
                return normalized.replace('super duty ', '')

            # Chevy/GMC heavy-duty variants
            if normalized.startswith('silverado '):
                return 'silverado'
            if normalized.startswith('sierra '):
                return 'sierra'

            return normalized

    matches = get_close_matches(model, known, n=1, cutoff=0.88)
    if matches:
        match = direct_rules.get(matches[0], matches[0])
        if brand == 'bmw':
            if match in {'328i', '330i', '335i', '340i'}:
                return '3 series'
            if match in {'528i', '535i'}:
                return '5 series'
        return match

    return None


us['body_type_from_model'] = us['model'].apply(infer_us_body_type)
us['model'] = us.apply(normalize_us_model, axis=1)

us['body_type'] = us['body_type'].str.lower().map({
    'sedan': 'sedan',
    'coupe': 'coupe',
    'convertible': 'convertible',
    'hatchback': 'hatchback',
    'wagon': 'estate',
    'van': 'van',
    'mini-van': 'van',
    'bus': 'van',
    'suv': 'suv',
    'offroad': 'suv',
    'pickup': 'pickup',
    'truck': 'pickup',
    'other': None,
})
us['body_type'] = us['body_type'].fillna(us['body_type_from_model'])
us = us.drop(columns=['body_type_from_model'])
us['body_type'] = us['body_type'].fillna('unknown')



us['transmission'] = us['transmission'].str.lower().map({
    'automatic': 'automatic',
    'manual': 'manual',
    'other': 'automatic'
})
us['transmission'] = us['transmission'].fillna('unknown')

us['fuel'] = us['fuel'].str.lower().map({
    'gas': 'petrol',
    'diesel': 'diesel',
    'hybrid': 'hybrid',
    'electric': 'electric',
    'other': None
})
us['fuel'] = us['fuel'].fillna('unknown')

us['engine_power'] = None
us['engine_size'] = None # nu exista datele in acest dataset

np.random.seed(42)

condition_map = {
    'salvage': 0.1,
    'fair': 0.3,
    'good': 0.5,
    'like new': 0.7,
    'excellent': 0.85,
    'new': 1.0,
}

def condition_to_float(condition, noise=0.05):
    base = condition_map.get(condition, np.nan)
    if pd.isna(base):
        return np.nan
    return np.clip(base + np.random.uniform(-noise, noise), 0, 1)

us['condition'] = us['condition'].apply(condition_to_float)
us['model'] = us['model'].fillna('unknown')

cols = ['brand', 'model', 'age', 'km', 'fuel', 'transmission',
        'price', 'condition', 'engine_power', 'engine_size', 'body_type']
us = us[cols]


print(us['fuel'].value_counts())
print(us['age'].value_counts())
print(us['brand'].value_counts())
print(us['model'].value_counts())
print(us['condition'].value_counts())
print(us['transmission'].value_counts())
print(us['body_type'].value_counts())

us.to_csv(PROCESSED_DATA_DIR / "us.csv", index=False)

print(us.shape)
print(us.sample(20).to_string())

print(us.isna().sum())
print(us.describe())

print(us.shape)

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[1]

FEATURES = [
    "brand", "model", "age", "km", "fuel", "transmission",
    "condition", "engine_power", "engine_size", "body_type",
]
CAT_COLS = ["brand", "model", "fuel", "transmission", "body_type"]

MODEL_PATH = ROOT_DIR / "models" / "model_lightgbm_optuna_preturi_auto.pkl"
DATA_PATH = ROOT_DIR / "data" / "processed" / "dataframe_modified.csv"


st.set_page_config(page_title="Predictie Pret Auto", page_icon="car", layout="centered")
st.title("Predictie Pret Auto")
st.caption("Completeaza datele masinii pentru a obtine o estimare de pret.")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_reference_data():
    return pd.read_csv(DATA_PATH)


try:
    model = load_model()
except FileNotFoundError:
    st.error(
        f"Nu am gasit modelul la '{MODEL_PATH}'. "
        "Ruleaza scriptul de training sau copiaza modelul in folderul models."
    )
    st.stop()

try:
    df_ref = load_reference_data()
except FileNotFoundError:
    st.error(
        f"Nu am gasit fisierul de date la '{DATA_PATH}'. "
        "Ruleaza scripturile de procesare a datelor."
    )
    st.stop()


brand_options = sorted(df_ref["brand"].dropna().astype(str).unique())
fuel_options = sorted(df_ref["fuel"].dropna().astype(str).unique())
transmission_options = sorted(df_ref["transmission"].dropna().astype(str).unique())
body_type_options = sorted(df_ref["body_type"].dropna().astype(str).unique())

age_max = int(df_ref["age"].max())
km_max = int(df_ref["km"].max())
power_min, power_max = float(df_ref["engine_power"].min()), float(df_ref["engine_power"].max())
size_min, size_max = float(df_ref["engine_size"].min()), float(df_ref["engine_size"].max())


st.subheader("Caracteristici masina")
brand = st.selectbox("Brand", brand_options)

model_options = sorted(
    df_ref.loc[df_ref["brand"].astype(str) == brand, "model"]
    .dropna()
    .astype(str)
    .unique()
)
if not model_options:
    model_options = sorted(df_ref["model"].dropna().astype(str).unique())

with st.form("car_form"):
    col1, col2 = st.columns(2)

    with col1:
        st.text_input("Brand selectat", value=brand, disabled=True)
        model_input = st.selectbox("Model", model_options)
        age = st.number_input(
            "Vechime (ani)",
            min_value=0,
            max_value=max(age_max, 50),
            value=min(5, age_max),
            step=1,
        )
        km = st.number_input(
            "Kilometri parcursi",
            min_value=0,
            max_value=max(km_max, 500_000),
            value=min(80_000, km_max),
            step=1000,
        )
        fuel = st.selectbox("Combustibil", fuel_options)

    with col2:
        transmission = st.selectbox("Transmisie", transmission_options)
        body_type = st.selectbox("Caroserie", body_type_options)
        condition = st.slider(
            "Conditie (0 = foarte uzata, 1 = ca noua)",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.01,
        )
        engine_power = st.number_input(
            "Putere motor (CP)",
            min_value=0.0,
            max_value=max(power_max, 600.0),
            value=float(round((power_min + power_max) / 2)),
            step=5.0,
        )
        engine_size = st.number_input(
            "Capacitate cilindrica (L)",
            min_value=0.0,
            max_value=max(size_max, 8.0),
            value=float(round((size_min + size_max) / 2, 1)),
            step=0.1,
        )

    submitted = st.form_submit_button("Estimeaza pretul", use_container_width=True)


if submitted:
    input_dict = {
        "brand": brand,
        "model": model_input,
        "age": age,
        "km": km,
        "fuel": fuel,
        "transmission": transmission,
        "condition": condition,
        "engine_power": engine_power,
        "engine_size": engine_size,
        "body_type": body_type,
    }
    X_input = pd.DataFrame([input_dict])[FEATURES]

    for col in CAT_COLS:
        train_categories = sorted(df_ref[col].dropna().astype(str).unique())
        X_input[col] = pd.Categorical(X_input[col].astype(str), categories=train_categories)

    pred_log = model.predict(X_input)
    pred_price = np.clip(np.expm1(pred_log), 0, None)[0]

    st.success(f"Pret estimat: **{pred_price:,.0f} EUR**")

    with st.expander("Detalii predictie"):
        st.write("Date introduse:")
        st.json(input_dict)
        st.caption(
            "Aceasta este o estimare statistica bazata pe modelul antrenat. "
            "Pretul real poate varia in functie de factori care nu sunt capturati "
            "in datele de antrenare."
        )

st.divider()
st.caption("Model: LightGBM, target log-transformat, hiperparametri optimizati cu Optuna. R2 aprox. 0.877.")

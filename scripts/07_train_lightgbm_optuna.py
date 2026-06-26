import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

import lightgbm as lgb
import optuna

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
FIGURES_DIR = ROOT_DIR / "figures"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 1. INCARCARE DATE
# ============================================================
df = pd.read_csv(PROCESSED_DATA_DIR / "dataframe_modified.csv")

features = ['brand', 'model', 'age', 'km', 'fuel', 'transmission',
            'condition', 'engine_power', 'engine_size', 'body_type']
target = 'price'

cat_cols = ['brand', 'model', 'fuel', 'transmission', 'body_type']

# split in 3: train (pentru tuning, cu cv intern), test (evaluare finala)
train, test = train_test_split(df, test_size=0.2, random_state=42)
# din train mai scoatem un val set, folosit de optuna ca sa nu "vada" test-ul deloc
train_opt, val_opt = train_test_split(train, test_size=0.2, random_state=42)

def prep_categoricals(X_fit, *others):
    """Seteaza dtype category pe X_fit si aplica aceleasi categorii pe others."""
    X_fit = X_fit.copy()
    for col in cat_cols:
        X_fit[col] = X_fit[col].astype(str).astype('category')
    result = [X_fit]
    for X in others:
        X = X.copy()
        for col in cat_cols:
            X[col] = X[col].astype(str).astype(
                pd.CategoricalDtype(categories=X_fit[col].cat.categories)
            )
        result.append(X)
    return result

X_train_opt = train_opt[features]
X_val_opt = val_opt[features]
X_train_full = train[features]
X_test = test[features]

X_train_opt, X_val_opt = prep_categoricals(X_train_opt, X_val_opt)
X_train_full, X_test = prep_categoricals(X_train_full, X_test)

y_train_opt = np.log1p(train_opt[target])
y_val_opt = np.log1p(val_opt[target])
y_train_full = np.log1p(train[target])
y_test = test[target]  # pastram originalul pentru evaluarea finala (scala reala)

# ============================================================
# 2. OBIECTIV OPTUNA
# ============================================================
def objective(trial):
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'random_state': 42,
        'n_jobs': -1,
        'n_estimators': 5000,
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 16, 256),
        'max_depth': trial.suggest_int('max_depth', 4, 16),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
    }

    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train_opt, y_train_opt,
        eval_set=[(X_val_opt, y_val_opt)],
        eval_metric='rmse',
        categorical_feature=cat_cols,
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
        ],
    )

    pred = model.predict(X_val_opt, num_iteration=model.best_iteration_)
    rmse = np.sqrt(mean_squared_error(y_val_opt, pred))
    return rmse

# ============================================================
# 3. RULARE STUDIU OPTUNA
# ============================================================
# N_TRIALS: 50 e un punct de start rezonabil pe 840k randuri (fiecare trial
# antreneaza un model cu early stopping, deci dureaza). Creste daca ai timp.
N_TRIALS = 50

study = optuna.create_study(direction='minimize', study_name='lgbm_preturi_auto')
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

print("\nCei mai buni hiperparametri gasiti:")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")
print(f"Best RMSE (log scale, pe val set): {study.best_value:.5f}")

# ============================================================
# 4. ANTRENARE MODEL FINAL CU CEI MAI BUNI PARAMETRI
# ============================================================
# Antrenam pe tot train-ul (train_opt + val_opt), evaluam pe test (nevazut de optuna)
best_params = study.best_params.copy()
best_params.update({
    'objective': 'regression',
    'metric': 'rmse',
    'verbosity': -1,
    'random_state': 42,
    'n_jobs': -1,
    'n_estimators': 8000,
})

final_model = lgb.LGBMRegressor(**best_params)
final_model.fit(
    X_train_full, y_train_full,
    eval_set=[(X_test, np.log1p(y_test))],
    eval_metric='rmse',
    categorical_feature=cat_cols,
    callbacks=[
        lgb.early_stopping(stopping_rounds=100),
        lgb.log_evaluation(period=200),
    ],
)

y_pred_log = final_model.predict(X_test, num_iteration=final_model.best_iteration_)
y_pred = np.clip(np.expm1(y_pred_log), 0, None)

# ============================================================
# 5. EVALUARE FINALA
# ============================================================
print("\nLightGBM + Optuna - model final")
print(f"MAE:  {mean_absolute_error(y_test, y_pred):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"R2:   {r2_score(y_test, y_pred):.4f}")

importance = pd.DataFrame({
    'feature': features,
    'importance': final_model.booster_.feature_importance(importance_type='gain')
}).sort_values('importance', ascending=False)
print("\nImportanta feature-urilor (gain):")
print(importance.to_string(index=False))

# ============================================================
# 6. GRAFICE
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(y_test, y_pred, alpha=0.1, s=1, color='steelblue')
lim = max(y_test.max(), y_pred.max())
axes[0].plot([0, lim], [0, lim], 'r--')
axes[0].set_xlim(0, lim)
axes[0].set_ylim(0, lim)
axes[0].set_xlabel('Pret real')
axes[0].set_ylabel('Pret prezis')
axes[0].set_title('Real vs Prezis - LightGBM + Optuna')

residuals = y_test - y_pred
axes[1].hist(residuals, bins=50, color='steelblue', edgecolor='white')
axes[1].axvline(0, color='red', linestyle='--')
axes[1].set_xlabel('Reziduu')
axes[1].set_ylabel('Frecventa')
axes[1].set_title('Distributia reziduurilor')

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'lightgbm_optuna_rezultate.png', dpi=120)
plt.show()

# ============================================================
# 7. SALVARE
# ============================================================
joblib.dump(final_model, MODELS_DIR / 'model_lightgbm_optuna_preturi_auto.pkl')
joblib.dump(study.best_params, MODELS_DIR / 'best_params_optuna.pkl')
print(f"\nModel salvat: {MODELS_DIR / 'model_lightgbm_optuna_preturi_auto.pkl'}")
print(f"Parametri salvati: {MODELS_DIR / 'best_params_optuna.pkl'}")

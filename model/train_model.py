# ── TRAIN & SERIALIZE MODEL ───────────────────────────────────
# Run this once locally to generate rf_model.joblib + scaler.joblib
# These files are loaded by the Streamlit app at runtime

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# ── CONFIG ────────────────────────────────────────────────────
FEATURES = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]
TARGET   = "NObeyesdad"
SEED     = 12345

OBESITY_ORDER = [
    'Insufficient_Weight', 'Normal_Weight',
    'Overweight_Level_I',  'Overweight_Level_II',
    'Obesity_Type_I',      'Obesity_Type_II',      'Obesity_Type_III'
]

# ── LOAD DATA ─────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv("/Users/juliaangladalomaeva/Desktop/customer-health-risk-app/data/obesity.csv")
X  = df[FEATURES].values
y  = df[TARGET].values

# ── SPLIT ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=SEED, stratify=y
)

# ── SCALE ─────────────────────────────────────────────────────
# Note: RF doesn't require scaling, but we save the scaler
# so the predictor page can show standardized feature contributions
scaler   = StandardScaler()
X_tr_sc  = scaler.fit_transform(X_train)
X_te_sc  = scaler.transform(X_test)

# ── TRAIN ─────────────────────────────────────────────────────
print("Training Random Forest...")
rf = RandomForestClassifier(
    n_estimators = 100,
    random_state = SEED,
    n_jobs       = -1
)
rf.fit(X_train, y_train)   # RF on raw (matches Repo 7)

# ── EVALUATE ──────────────────────────────────────────────────
y_pred = rf.predict(X_test)
acc    = accuracy_score(y_test, y_pred)
print(f"\n✅ Model accuracy: {acc*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ── SERIALIZE ─────────────────────────────────────────────────
os.makedirs(".", exist_ok=True)
joblib.dump(rf,     "rf_model.joblib")
joblib.dump(scaler, "scaler.joblib")
joblib.dump({
    'features':      FEATURES,
    'target':        TARGET,
    'obesity_order': OBESITY_ORDER,
    'accuracy':      acc,
    'X_test':        X_test,
    'y_test':        y_test,
    'y_pred':        y_pred
}, "model_metadata.joblib")

print("\n✅ Saved: rf_model.joblib · scaler.joblib · model_metadata.joblib")
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from feature_extraction import extract_features_from_path, FEATURE_COLS

# ─────────────────────────────────────────
# BUILD DATASET
# ─────────────────────────────────────────
print("Reading ground_truth.csv ...")
ground_truth = pd.read_csv('ground_truth.csv')  # columns: fruit_id, juice_ml

records = []
for _, row in ground_truth.iterrows():
    img_path = os.path.join('images', f"{row['fruit_id']}.jpg")
    if not os.path.exists(img_path):
        print(f"  [SKIP] Image not found: {img_path}")
        continue
    features = extract_features_from_path(img_path)
    if features is None:
        print(f"  [SKIP] Could not extract features: {img_path}")
        continue
    features['fruit_id'] = row['fruit_id']
    features['juice_ml'] = row['juice_ml']
    records.append(features)

if len(records) == 0:
    print("\n[ERROR] No samples were processed. Check your images/ folder and ground_truth.csv.")
    exit()

df = pd.DataFrame(records)
df.to_csv('calamansi_dataset.csv', index=False)
print(f"\nDataset built: {len(df)} samples saved to calamansi_dataset.csv")
print(df[FEATURE_COLS + ['juice_ml']].head())


# ─────────────────────────────────────────
# TRAIN LINEAR REGRESSION MODEL
# ─────────────────────────────────────────
X = df[FEATURE_COLS]
y = df['juice_ml']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_sc, y_train)


# ─────────────────────────────────────────
# EVALUATE
# ─────────────────────────────────────────
y_pred = model.predict(X_test_sc)
mae    = mean_absolute_error(y_test, y_pred)
rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
r2     = r2_score(y_test, y_pred)

print("\n══════════════════════════════════")
print("       TRAINING RESULTS           ")
print("══════════════════════════════════")
print(f"  R² Score : {r2:.4f}")
print(f"  MAE      : {mae:.4f} mL")
print(f"  RMSE     : {rmse:.4f} mL")
print("══════════════════════════════════")


# ─────────────────────────────────────────
# SAVE MODEL
# ─────────────────────────────────────────
joblib.dump(model,  'juice_yield_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("\n✅ Model saved:  juice_yield_model.pkl")
print("✅ Scaler saved: scaler.pkl")
print("\nNext step: run  python evaluate_model.py")
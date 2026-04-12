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
# LOAD & VALIDATE GROUND TRUTH
# ─────────────────────────────────────────
GT_FILE     = 'ground_truth.csv'
IMG_DIR     = 'images'
MODEL_FILE  = 'juice_yield_model.pkl'
SCALER_FILE = 'scaler.pkl'
DATASET_OUT = 'calamansi_dataset.csv'

print("Reading ground_truth.csv ...")

if not os.path.exists(GT_FILE):
    raise FileNotFoundError(
        f"❌ '{GT_FILE}' not found. "
        "Create a CSV with columns: fruit_id, juice_ml"
    )

ground_truth = pd.read_csv(GT_FILE)

# Validate required columns
required_cols = {'fruit_id', 'juice_ml'}
missing_cols  = required_cols - set(ground_truth.columns)
if missing_cols:
    raise ValueError(
        f"❌ ground_truth.csv is missing columns: {missing_cols}. "
        f"Found columns: {list(ground_truth.columns)}"
    )

# Drop rows with null fruit_id or juice_ml
before = len(ground_truth)
ground_truth = ground_truth.dropna(subset=['fruit_id', 'juice_ml'])
if len(ground_truth) < before:
    print(f"  ⚠️  Dropped {before - len(ground_truth)} rows with missing values.")

if len(ground_truth) == 0:
    raise ValueError("❌ ground_truth.csv has no valid rows after removing nulls.")

print(f"  Found {len(ground_truth)} entries in ground_truth.csv")


# ─────────────────────────────────────────
# BUILD DATASET
# ─────────────────────────────────────────
records      = []
skipped_img  = []
skipped_feat = []

for _, row in ground_truth.iterrows():
    fruit_id = str(row['fruit_id']).strip()
    img_path = os.path.join(IMG_DIR, f"{fruit_id}.jpg")

    if not os.path.exists(img_path):
        skipped_img.append(img_path)
        continue

    features = extract_features_from_path(img_path)
    if features is None:
        skipped_feat.append(img_path)
        continue

    features['fruit_id'] = fruit_id
    features['juice_ml'] = float(row['juice_ml'])
    records.append(features)

# Report skipped files
if skipped_img:
    print(f"\n  ⚠️  {len(skipped_img)} image(s) not found — skipped:")
    for p in skipped_img[:10]:
        print(f"       {p}")
    if len(skipped_img) > 10:
        print(f"       ... and {len(skipped_img) - 10} more.")

if skipped_feat:
    print(f"\n  ⚠️  {len(skipped_feat)} image(s) had no detectable features — skipped:")
    for p in skipped_feat[:10]:
        print(f"       {p}")
    if len(skipped_feat) > 10:
        print(f"       ... and {len(skipped_feat) - 10} more.")

# Guard: need at least 5 usable samples to train
if len(records) == 0:
    raise RuntimeError(
        "❌ No usable samples found. "
        "Check that your images/ folder exists and images match fruit_id values in ground_truth.csv."
    )

if len(records) < 5:
    raise RuntimeError(
        f"❌ Only {len(records)} usable sample(s) found — need at least 5 to train. "
        "Add more labelled images to ground_truth.csv."
    )

df = pd.DataFrame(records)

# Validate all feature columns are present
missing_feat_cols = [c for c in FEATURE_COLS if c not in df.columns]
if missing_feat_cols:
    raise RuntimeError(
        f"❌ Feature columns missing from extracted data: {missing_feat_cols}. "
        "Check feature_extraction.py FEATURE_COLS matches compute_features() output."
    )

# Drop any rows where feature values are NaN
rows_before = len(df)
df = df.dropna(subset=FEATURE_COLS + ['juice_ml'])
if len(df) < rows_before:
    print(f"\n  ⚠️  Dropped {rows_before - len(df)} row(s) with NaN feature values.")

if len(df) < 5:
    raise RuntimeError(
        f"❌ Only {len(df)} clean sample(s) after dropping NaN rows — need at least 5."
    )

df.to_csv(DATASET_OUT, index=False)
print(f"\n✅ Dataset built: {len(df)} samples saved to {DATASET_OUT}")
print(df[FEATURE_COLS + ['juice_ml']].head())
print(f"\n  juice_ml stats:\n{df['juice_ml'].describe().to_string()}")


# ─────────────────────────────────────────
# PREPARE DATA
# ─────────────────────────────────────────
X = df[FEATURE_COLS]
y = df['juice_ml']

# Use stratified-like split only when dataset is large enough
# For small datasets (< 20), use a smaller test split to keep more training data
test_size = 0.2 if len(df) >= 20 else max(1, int(len(df) * 0.15))

# train_test_split needs at least 2 samples in each split
if len(df) < 10:
    print(f"\n  ⚠️  Small dataset ({len(df)} samples) — using 1 test sample, rest for training.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1, random_state=42
    )
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

print(f"\n  Train samples : {len(X_train)}")
print(f"  Test samples  : {len(X_test)}")

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)


# ─────────────────────────────────────────
# TRAIN LINEAR REGRESSION MODEL
# ─────────────────────────────────────────
model = LinearRegression()
model.fit(X_train_sc, y_train)

print("\n─── Linear Regression Coefficients ───")
for feat, coef in zip(FEATURE_COLS, model.coef_):
    print(f"  {feat:30s}: {coef:+.4f}")
print(f"  {'Intercept':30s}: {model.intercept_:+.4f}")


# ─────────────────────────────────────────
# EVALUATE
# ─────────────────────────────────────────
y_pred = model.predict(X_test_sc)
mae    = mean_absolute_error(y_test, y_pred)
rmse   = np.sqrt(mean_squared_error(y_test, y_pred))

# r2_score is undefined with 1 test sample — handle gracefully
if len(y_test) >= 2:
    r2 = r2_score(y_test, y_pred)
    r2_str = f"{r2:.4f}"
else:
    r2     = float('nan')
    r2_str = "N/A (need ≥2 test samples)"

print("\n─── Model Evaluation (Test Set) ───")
print(f"  Samples: {len(y_test)}")
print(f"  MAE    : {mae:.4f} mL")
print(f"  RMSE   : {rmse:.4f} mL")
print(f"  R²     : {r2_str}")

# Also report full-dataset fit quality
y_all_pred = model.predict(scaler.transform(X))
r2_full    = r2_score(y, y_all_pred)
mae_full   = mean_absolute_error(y, y_all_pred)
print(f"\n─── Full Dataset Fit ───")
print(f"  MAE    : {mae_full:.4f} mL")
print(f"  R²     : {r2_full:.4f}")

if r2_full < 0.5:
    print(
        "\n  ⚠️  Low R² — model fit is poor. Consider:\n"
        "     • Adding more labelled samples (aim for 30+)\n"
        "     • Checking ground_truth.csv juice_ml values are accurate\n"
        "     • Ensuring training images are top-down, well-lit, one fruit each"
    )


# ─────────────────────────────────────────
# SAVE MODEL AND SCALER
# ─────────────────────────────────────────
joblib.dump(model,  MODEL_FILE)
joblib.dump(scaler, SCALER_FILE)
print(f"\n✅ Model saved  : {MODEL_FILE}")
print(f"✅ Scaler saved : {SCALER_FILE}")
print("\nYou can now run:  streamlit run app.py")
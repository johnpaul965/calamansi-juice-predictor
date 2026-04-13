import pandas as pd
import numpy as np
import os
import joblib
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
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

required_cols = {'fruit_id', 'juice_ml'}
missing_cols  = required_cols - set(ground_truth.columns)
if missing_cols:
    raise ValueError(
        f"❌ ground_truth.csv is missing columns: {missing_cols}. "
        f"Found columns: {list(ground_truth.columns)}"
    )

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

if len(records) == 0:
    raise RuntimeError(
        "❌ No usable samples found. "
        "Check that your images/ folder exists and images match fruit_id values in ground_truth.csv."
    )
if len(records) < 5:
    raise RuntimeError(
        f"❌ Only {len(records)} usable sample(s) found — need at least 5 to train."
    )

df = pd.DataFrame(records)

# ── FILTER 1: Remove extreme juice_ml labels ──
before = len(df)
df = df[(df['juice_ml'] > 1.5) & (df['juice_ml'] < 7.0)]
if len(df) < before:
    print(f"\n  ⚠️  Dropped {before - len(df)} row(s) with extreme juice_ml (outside 1.5–7.0 mL).")

# ── FILTER 2: Tighter volume/juice mismatch (raised threshold from 50 to 200) ──
before = len(df)
df = df[~((df['estimated_volume_cm3'] < 200) & (df['juice_ml'] > 4.5))]
if len(df) < before:
    print(f"  ⚠️  Dropped {before - len(df)} row(s) with volume/juice mismatch.")

# ── FILTER 3: IQR outlier removal on juice_ml ──
Q1, Q3 = df['juice_ml'].quantile(0.25), df['juice_ml'].quantile(0.75)
IQR = Q3 - Q1
before = len(df)
df = df[(df['juice_ml'] >= Q1 - 2.5 * IQR) & (df['juice_ml'] <= Q3 + 2.5 * IQR)]
if len(df) < before:
    print(f"  ⚠️  Dropped {before - len(df)} row(s) as juice_ml outliers (IQR method).")

missing_feat_cols = [c for c in FEATURE_COLS if c not in df.columns]
if missing_feat_cols:
    raise RuntimeError(
        f"❌ Feature columns missing from extracted data: {missing_feat_cols}."
    )

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
# FEATURE CORRELATION ANALYSIS
# Show which features are highly correlated (helps understand Ridge behavior)
# ─────────────────────────────────────────
print("\n─── Feature Correlation Matrix (top correlated pairs) ───")
corr_matrix = df[FEATURE_COLS].corr()
# Get pairs with |correlation| > 0.7
high_corr_pairs = []
for i in range(len(FEATURE_COLS)):
    for j in range(i+1, len(FEATURE_COLS)):
        corr_val = corr_matrix.iloc[i, j]
        if abs(corr_val) > 0.7:
            high_corr_pairs.append((FEATURE_COLS[i], FEATURE_COLS[j], corr_val))

if high_corr_pairs:
    print("  Highly correlated features (|r| > 0.7):")
    for feat1, feat2, corr in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True):
        print(f"    {feat1:30s} ↔ {feat2:30s}: {corr:+.3f}")
    print("  → Ridge regression will handle this multicollinearity")
else:
    print("  No highly correlated features found")


# ─────────────────────────────────────────
# USE ALL FEATURES (as stated in paper)
# Ridge regression handles multicollinearity through regularization
# ─────────────────────────────────────────
MODEL_FEATURES = FEATURE_COLS  # Use ALL features from paper
print(f"\n  Using all {len(MODEL_FEATURES)} features: {MODEL_FEATURES}")

X = df[MODEL_FEATURES]
y = df['juice_ml']


# ─────────────────────────────────────────
# PREPARE DATA
# ─────────────────────────────────────────
test_size = 0.2 if len(df) >= 20 else max(1, int(len(df) * 0.15))

if len(df) < 10:
    print(f"\n  ⚠️  Small dataset ({len(df)} samples) — using 1 test sample.")
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
# HYPERPARAMETER TUNING — Find optimal alpha
# Use cross-validation to select best regularization strength
# ─────────────────────────────────────────
print("\n─── Tuning Ridge Alpha (regularization strength) ───")

# Test a range of alpha values
if len(df) >= 10:
    # Use GridSearchCV for datasets with enough samples
    param_grid = {'alpha': [0.1, 0.5, 1.0, 5.0, 10.0, 20.0, 50.0]}
    grid_search = GridSearchCV(
        Ridge(), 
        param_grid, 
        cv=min(5, len(X_train)),  # 5-fold or less if small dataset
        scoring='r2',
        n_jobs=-1
    )
    grid_search.fit(X_train_sc, y_train)
    best_alpha = grid_search.best_params_['alpha']
    print(f"  Best alpha: {best_alpha:.1f} (R² = {grid_search.best_score_:.4f})")
else:
    # For very small datasets, just use a sensible default
    best_alpha = 10.0
    print(f"  Small dataset — using alpha = {best_alpha}")


# ─────────────────────────────────────────
# TRAIN FINAL MODEL with best alpha
# ─────────────────────────────────────────
model = Ridge(alpha=best_alpha)
model.fit(X_train_sc, y_train)

print("\n─── Ridge Regression Coefficients ───")
for feat, coef in zip(MODEL_FEATURES, model.coef_):
    print(f"  {feat:30s}: {coef:+.4f}")
print(f"  {'Intercept':30s}: {model.intercept_:+.4f}")

# Coefficient interpretation
vol_idx = MODEL_FEATURES.index('estimated_volume_cm3')
vol_coef = model.coef_[vol_idx]
if vol_coef > 0:
    print("\n  ✅ Volume coefficient is positive (physically correct)")
else:
    print("\n  ⚠️  Volume coefficient is negative — data may contain labeling errors")


# ─────────────────────────────────────────
# CROSS-VALIDATION (more reliable than single split)
# ─────────────────────────────────────────
if len(df) >= 10:
    X_all_sc = scaler.transform(X)
    cv_folds = min(5, len(df))
    cv_scores = cross_val_score(
        Ridge(alpha=best_alpha), 
        X_all_sc, 
        y, 
        cv=cv_folds, 
        scoring='r2'
    )
    print(f"\n─── {cv_folds}-Fold Cross-Validation R² ───")
    print(f"  Scores : {[f'{s:.4f}' for s in cv_scores]}")
    print(f"  Mean R²: {cv_scores.mean():.4f}  (±{cv_scores.std():.4f})")
    
    # Also get MAE from CV
    cv_mae_scores = -cross_val_score(
        Ridge(alpha=best_alpha), 
        X_all_sc, 
        y, 
        cv=cv_folds, 
        scoring='neg_mean_absolute_error'
    )
    print(f"  Mean MAE: {cv_mae_scores.mean():.4f} mL (±{cv_mae_scores.std():.4f})")


# ─────────────────────────────────────────
# EVALUATE ON TEST SET
# ─────────────────────────────────────────
y_pred = model.predict(X_test_sc).clip(1.4, 7.2)

mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

if len(y_test) >= 2:
    r2     = r2_score(y_test, y_pred)
    r2_str = f"{r2:.4f}"
else:
    r2     = float('nan')
    r2_str = "N/A (need ≥2 test samples)"

print("\n─── Model Evaluation (Test Set) ───")
print(f"  Samples: {len(y_test)}")
print(f"  MAE    : {mae:.4f} mL")
print(f"  RMSE   : {rmse:.4f} mL")
print(f"  R²     : {r2_str}")


# ─────────────────────────────────────────
# FULL DATASET FIT QUALITY
# ─────────────────────────────────────────
X_all_sc   = scaler.transform(X)
y_all_pred = model.predict(X_all_sc).clip(1.4, 7.2)
r2_full    = r2_score(y, y_all_pred)
mae_full   = mean_absolute_error(y, y_all_pred)

print(f"\n─── Full Dataset Fit ───")
print(f"  MAE    : {mae_full:.4f} mL")
print(f"  R²     : {r2_full:.4f}")


# ─────────────────────────────────────────
# PERFORMANCE DIAGNOSTICS
# ─────────────────────────────────────────
if r2_full < 0.5:
    print(
        "\n  ⚠️  Low R² — model fit is poor. Consider:\n"
        "     • Checking ground_truth.csv juice_ml values are accurate\n"
        "     • Ensuring images are top-down, well-lit, one fruit each\n"
        "     • Adding more diverse samples (aim for 30+ fruits)\n"
        "     • The relationship may be non-linear — consider Random Forest"
    )
elif r2_full < 0.7:
    print(
        "\n  ℹ️  Moderate fit. To improve:\n"
        "     • Add more training samples for better generalization\n"
        "     • Verify measurement consistency in ground_truth.csv"
    )
else:
    print("\n  ✅ Good model fit!")

# Show prediction errors
print("\n─── Prediction Analysis (Test Set) ───")
if len(y_test) > 0:
    errors = y_pred - y_test
    print(f"  Mean error      : {errors.mean():+.4f} mL")
    print(f"  Std dev of error: {errors.std():.4f} mL")
    print(f"  Max overpredict : {errors.max():+.4f} mL")
    print(f"  Max underpredict: {errors.min():+.4f} mL")


# ─────────────────────────────────────────
# SAVE MODEL AND ARTIFACTS
# ─────────────────────────────────────────
joblib.dump(model,          MODEL_FILE)
joblib.dump(scaler,         SCALER_FILE)
joblib.dump(MODEL_FEATURES, 'model_features.pkl')

print(f"\n✅ Model saved         : {MODEL_FILE}")
print(f"✅ Scaler saved        : {SCALER_FILE}")
print(f"✅ Feature list saved  : model_features.pkl")
print(f"\n📊 Model summary:")
print(f"   • Algorithm      : Ridge Regression (alpha={best_alpha})")
print(f"   • Features       : {len(MODEL_FEATURES)} (all from paper)")
print(f"   • Training samples: {len(X_train)}")
print(f"   • R² score       : {r2_full:.4f}")
print(f"   • MAE            : {mae_full:.4f} mL")
print("\nYou can now run:  streamlit run app.py")
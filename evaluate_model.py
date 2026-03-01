import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import joblib
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from feature_extraction import FEATURE_COLS

# ─────────────────────────────────────────
# LOAD DATA AND MODEL
# ─────────────────────────────────────────
print("Loading dataset and model...")

df     = pd.read_csv('calamansi_dataset.csv')
model  = joblib.load('juice_yield_model.pkl')
scaler = joblib.load('scaler.pkl')

X = df[FEATURE_COLS]
y = df['juice_ml']

X_sc  = scaler.transform(X)
y_pred = model.predict(X_sc)

# Output folder for all graphs
os.makedirs('evaluation_results', exist_ok=True)


# ─────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────
mae  = mean_absolute_error(y, y_pred)
mse  = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y, y_pred)

print("\n══════════════════════════════════")
print("       MODEL EVALUATION RESULTS   ")
print("══════════════════════════════════")
print(f"  R² Score : {r2:.4f}")
print(f"  MAE      : {mae:.4f} mL")
print(f"  MSE      : {mse:.4f}")
print(f"  RMSE     : {rmse:.4f} mL")
print("══════════════════════════════════")

# 5-Fold Cross Validation
cv_scores = cross_val_score(model, X_sc, y, cv=5, scoring='r2')
print(f"\n  5-Fold Cross-Validation R²:")
for i, score in enumerate(cv_scores, 1):
    print(f"    Fold {i}: {score:.4f}")
print(f"  Mean R² : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print("══════════════════════════════════\n")


# ─────────────────────────────────────────
# GRAPH 1: Actual vs Predicted
# ─────────────────────────────────────────
plt.figure(figsize=(7, 6))
plt.scatter(y, y_pred, color='steelblue', edgecolors='white', s=80, alpha=0.85, label='Samples')
min_val = min(y.min(), y_pred.min()) - 0.5
max_val = max(y.max(), y_pred.max()) + 0.5
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Juice Yield (mL)', fontsize=12)
plt.ylabel('Predicted Juice Yield (mL)', fontsize=12)
plt.title('Actual vs Predicted Juice Yield', fontsize=14, fontweight='bold')
plt.legend()
plt.annotate(f'R² = {r2:.4f}\nMAE = {mae:.4f} mL\nRMSE = {rmse:.4f} mL',
             xy=(0.05, 0.75), xycoords='axes fraction',
             fontsize=11, bbox=dict(boxstyle='round,pad=0.4', fc='lightyellow', ec='gray'))
plt.tight_layout()
plt.savefig('evaluation_results/01_actual_vs_predicted.png', dpi=150)
plt.close()
print("✅ Saved: 01_actual_vs_predicted.png")


# ─────────────────────────────────────────
# GRAPH 2: Residuals vs Predicted
# ─────────────────────────────────────────
residuals = y - y_pred

plt.figure(figsize=(7, 5))
plt.scatter(y_pred, residuals, color='tomato', edgecolors='white', s=80, alpha=0.85)
plt.axhline(0, color='black', linewidth=1.5, linestyle='--')
plt.xlabel('Predicted Juice Yield (mL)', fontsize=12)
plt.ylabel('Residuals (mL)', fontsize=12)
plt.title('Residual Plot', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('evaluation_results/02_residual_plot.png', dpi=150)
plt.close()
print("✅ Saved: 02_residual_plot.png")


# ─────────────────────────────────────────
# GRAPH 3: Residual Distribution
# ─────────────────────────────────────────
plt.figure(figsize=(7, 5))
plt.hist(residuals, bins=20, color='steelblue', edgecolor='white')
plt.axvline(0, color='red', linestyle='--', linewidth=1.5)
plt.xlabel('Residual (mL)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Distribution of Residuals', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('evaluation_results/03_residual_distribution.png', dpi=150)
plt.close()
print("✅ Saved: 03_residual_distribution.png")


# ─────────────────────────────────────────
# GRAPH 4: Feature Importance (Coefficients)
# ─────────────────────────────────────────
coef_df = pd.DataFrame({
    'Feature'    : FEATURE_COLS,
    'Coefficient': model.coef_
}).sort_values('Coefficient', ascending=True)

colors = ['tomato' if c < 0 else 'steelblue' for c in coef_df['Coefficient']]

plt.figure(figsize=(8, 5))
plt.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors, edgecolor='white')
plt.axvline(0, color='black', linewidth=1)
plt.xlabel('Coefficient Value', fontsize=12)
plt.title('Linear Regression Feature Coefficients', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('evaluation_results/04_feature_coefficients.png', dpi=150)
plt.close()
print("✅ Saved: 04_feature_coefficients.png")


# ─────────────────────────────────────────
# GRAPH 5: Cross-Validation R² Scores
# ─────────────────────────────────────────
plt.figure(figsize=(7, 5))
folds = [f'Fold {i}' for i in range(1, 6)]
bars  = plt.bar(folds, cv_scores, color='mediumseagreen', edgecolor='white')
plt.axhline(cv_scores.mean(), color='red', linestyle='--', linewidth=1.5, label=f'Mean R² = {cv_scores.mean():.4f}')
plt.ylim(0, 1.05)
plt.ylabel('R² Score', fontsize=12)
plt.title('5-Fold Cross-Validation R² Scores', fontsize=14, fontweight='bold')
plt.legend()
for bar, score in zip(bars, cv_scores):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
             f'{score:.4f}', ha='center', fontsize=10)
plt.tight_layout()
plt.savefig('evaluation_results/05_cross_validation.png', dpi=150)
plt.close()
print("✅ Saved: 05_cross_validation.png")


# ─────────────────────────────────────────
# GRAPH 6: Feature Correlation with Juice Yield
# ─────────────────────────────────────────
correlations = df[FEATURE_COLS + ['juice_ml']].corr()['juice_ml'].drop('juice_ml').sort_values()

colors = ['tomato' if c < 0 else 'steelblue' for c in correlations]

plt.figure(figsize=(8, 5))
plt.barh(correlations.index, correlations.values, color=colors, edgecolor='white')
plt.axvline(0, color='black', linewidth=1)
plt.xlabel('Pearson Correlation Coefficient', fontsize=12)
plt.title('Feature Correlation with Juice Yield', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('evaluation_results/06_feature_correlation.png', dpi=150)
plt.close()
print("✅ Saved: 06_feature_correlation.png")


# ─────────────────────────────────────────
# SAVE METRICS SUMMARY TO CSV
# ─────────────────────────────────────────
summary = pd.DataFrame({
    'Metric': ['R² Score', 'MAE (mL)', 'MSE', 'RMSE (mL)',
                'CV Mean R²', 'CV Std R²'],
    'Value' : [round(r2, 4), round(mae, 4), round(mse, 4),
                round(rmse, 4), round(cv_scores.mean(), 4), round(cv_scores.std(), 4)]
})
summary.to_csv('evaluation_results/metrics_summary.csv', index=False)
print("✅ Saved: metrics_summary.csv")

print("\n🎓 All evaluation files saved inside: evaluation_results/")

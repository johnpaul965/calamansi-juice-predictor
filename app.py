import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, KFold

from feature_extraction import extract_features_from_array, FEATURE_COLS

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Calamansi Juice Yield Prediction System",
    page_icon="🍋",
    layout="wide"
)

# ─────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 36px; font-weight: bold;
        color: #166534; text-align: center;
    }
    .sub-title {
        font-size: 16px; color: #6b7280;
        text-align: center; margin-bottom: 10px;
    }
    .section-header {
        font-size: 20px; font-weight: bold; color: #166534;
        border-bottom: 2px solid #16a34a;
        padding-bottom: 6px; margin-top: 20px; margin-bottom: 10px;
    }
    .result-box {
        background-color: #f0fdf4;
        border-left: 5px solid #16a34a;
        border-radius: 8px; padding: 20px; margin-top: 10px;
    }
    .step-box {
        background-color: #fefce8;
        border: 1px solid #fde68a;
        border-radius: 10px; padding: 14px 18px; margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────
@st.cache_resource
def load_model():
    model  = joblib.load('juice_yield_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model_loaded = os.path.exists('juice_yield_model.pkl') and os.path.exists('scaler.pkl')
if model_loaded:
    model, scaler = load_model()

# ─────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e5/Calamansi.jpg/320px-Calamansi.jpg",
                 use_column_width=True)
st.sidebar.markdown("## 🍋 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "🔍 Predict Juice Yield", "📊 Model Performance"])
st.sidebar.markdown("---")
st.sidebar.markdown("**Capstone Project**")
st.sidebar.markdown("Image-Based System for Predicting Calamansi Juice Yield Using Linear Regression")
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Model Status:** {'✅ Loaded' if model_loaded else '❌ Not found — run train_model.py'}")


# ══════════════════════════════════════════════════════
# PAGE 1: HOME
# ══════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown('<div class="main-title">🍋 Calamansi Juice Yield Prediction System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">An Image-Based System Using Linear Regression</div>', unsafe_allow_html=True)
    st.divider()

    col1, col2 = st.columns([1.5, 1])
    with col1:
        st.markdown('<div class="section-header">📌 About This System</div>', unsafe_allow_html=True)
        st.write("""
        This system predicts the **juice yield (mL)** of a Calamansi fruit
        *(Citrus microcarpa)* using image processing and linear regression.
        Instead of manually squeezing each fruit to measure juice,
        this system analyzes a photo and automatically predicts how much
        juice the fruit will produce.
        """)
        st.markdown('<div class="section-header">🎯 Objectives</div>', unsafe_allow_html=True)
        st.write("**1.** Collect and process calamansi fruit images.")
        st.write("**2.** Extract image features such as size, shape, and color.")
        st.write("**3.** Train a Linear Regression model to predict juice yield.")
        st.write("**4.** Deploy the model as a web-based prediction system.")

    with col2:
        st.markdown('<div class="section-header">📐 System Scope</div>', unsafe_allow_html=True)
        st.markdown("""
        | Item | Detail |
        |------|--------|
        | Fruit | Calamansi *(Citrus microcarpa)* |
        | Input | Fruit image (JPG/PNG) |
        | Output | Juice yield in mL |
        | Model | Linear Regression |
        | Features | Shape + Color |
        """)

    st.divider()
    st.markdown('<div class="section-header">⚙️ How the System Works</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown('<div class="step-box">📷 <b>Step 1</b><br>Upload calamansi image</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="step-box">🔧 <b>Step 2</b><br>Preprocess & segment fruit</div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="step-box">📐 <b>Step 3</b><br>Extract shape & color features</div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="step-box">📈 <b>Step 4</b><br>Apply linear regression model</div>', unsafe_allow_html=True)
    with c5:
        st.markdown('<div class="step-box">🍋 <b>Step 5</b><br>Display predicted juice yield</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="section-header">🔬 Image Features Extracted</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Shape Features**")
        st.write("- Area (cm²)")
        st.write("- Diameter (cm)")
        st.write("- Perimeter (cm)")
        st.write("- Circularity")
        st.write("- Estimated Volume (cm³)")
    with col2:
        st.markdown("**Color Features**")
        st.write("- Mean Hue")
        st.write("- Mean Saturation")
        st.write("- Mean Brightness (Value)")


# ══════════════════════════════════════════════════════
# PAGE 2: PREDICT
# ══════════════════════════════════════════════════════
elif page == "🔍 Predict Juice Yield":
    st.markdown('<div class="main-title">🔍 Predict Juice Yield</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Upload a calamansi image to get the predicted juice yield</div>', unsafe_allow_html=True)
    st.divider()

    if not model_loaded:
        st.error("❌ Model not found. Please run **train_model.py** first before using this page.")
        st.stop()

    uploaded_file = st.file_uploader(
        "Upload a calamansi image (JPG or PNG)",
        type=['jpg', 'jpeg', 'png']
    )

    if uploaded_file:
        pil_image = Image.open(uploaded_file).convert('RGB')
        img_array = np.array(pil_image)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="section-header">📷 Original Image</div>', unsafe_allow_html=True)
            st.image(pil_image, use_column_width=True)

        with st.spinner("🔍 Processing image..."):
            features, mask = extract_features_from_array(img_array)

        if features is None:
            st.error("❌ Fruit not detected. Please use a clearer image with a plain background and good lighting.")
            st.stop()

        segmented = cv2.bitwise_and(img_array, img_array, mask=mask)
        with col2:
            st.markdown('<div class="section-header">🔬 Segmented Fruit</div>', unsafe_allow_html=True)
            st.image(segmented, use_column_width=True)

        st.divider()

        X_new           = np.array([[features[f] for f in FEATURE_COLS]])
        X_new_sc        = scaler.transform(X_new)
        predicted_yield = model.predict(X_new_sc)[0]

        st.markdown('<div class="section-header">📊 Prediction Result</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="result-box">
            <h2 style="color:#166534; margin:0;">Predicted Juice Yield</h2>
            <h1 style="color:#16a34a; font-size:52px; margin:8px 0;">{predicted_yield:.2f} mL</h1>
            <p style="color:#6b7280; margin:0;">Based on image features extracted using OpenCV</p>
        </div>
        """, unsafe_allow_html=True)

        st.divider()
        st.markdown('<div class="section-header">🧪 Extracted Image Features</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Shape Features**")
            st.table(pd.DataFrame({
                "Feature": ["Area (cm²)", "Diameter (cm)", "Perimeter (cm)", "Circularity", "Estimated Volume (cm³)"],
                "Value": [
                    f"{features['area_cm2']:.4f}",
                    f"{features['diameter_cm']:.4f}",
                    f"{features['perimeter_cm']:.4f}",
                    f"{features['circularity']:.4f}",
                    f"{features['estimated_volume_cm3']:.4f}",
                ]
            }))
        with col2:
            st.markdown("**Color Features**")
            st.table(pd.DataFrame({
                "Feature": ["Mean Hue", "Mean Saturation", "Mean Brightness"],
                "Value": [
                    f"{features['mean_hue']:.2f}",
                    f"{features['mean_saturation']:.2f}",
                    f"{features['mean_value']:.2f}",
                ]
            }))


# ══════════════════════════════════════════════════════
# PAGE 3: MODEL PERFORMANCE
# ══════════════════════════════════════════════════════
elif page == "📊 Model Performance":
    st.markdown('<div class="main-title">📊 Model Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Evaluation metrics and graphs of the Linear Regression model</div>', unsafe_allow_html=True)
    st.divider()

    if not model_loaded:
        st.error("❌ Model not found. Please run **train_model.py** first.")
        st.stop()

    if not os.path.exists('calamansi_dataset.csv'):
        st.error("❌ Dataset not found. Please run **train_model.py** first.")
        st.stop()

    df     = pd.read_csv('calamansi_dataset.csv')
    X      = df[FEATURE_COLS]
    y      = df['juice_ml']
    X_sc   = scaler.transform(X)
    y_pred = model.predict(X_sc)

    mae  = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2   = r2_score(y, y_pred)

    st.markdown('<div class="section-header">📐 Evaluation Metrics</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Samples", f"{len(df)}")
    c2.metric("R² Score", f"{r2:.4f}", help="Closer to 1.0 is better")
    c3.metric("MAE", f"{mae:.4f} mL", help="Mean Absolute Error")
    c4.metric("RMSE", f"{rmse:.4f} mL", help="Root Mean Squared Error")

    st.divider()

    # Actual vs Predicted + Residuals
    st.markdown('<div class="section-header">📈 Actual vs Predicted & Residual Analysis</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        fig1, ax1 = plt.subplots(figsize=(6, 5))
        ax1.scatter(y, y_pred, color='steelblue', alpha=0.7, edgecolors='white', s=60)
        min_val = min(y.min(), y_pred.min()) - 0.5
        max_val = max(y.max(), y_pred.max()) + 0.5
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        ax1.set_xlabel('Actual Juice Yield (mL)')
        ax1.set_ylabel('Predicted Juice Yield (mL)')
        ax1.set_title('Actual vs Predicted')
        ax1.legend()
        ax1.text(0.05, 0.92, f'R² = {r2:.4f}', transform=ax1.transAxes,
                 fontsize=10, color='darkred',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        plt.tight_layout()
        st.pyplot(fig1)

    with col2:
        residuals = y - y_pred
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        ax2.scatter(y_pred, residuals, color='darkorange', alpha=0.7, edgecolors='white', s=60)
        ax2.axhline(0, color='red', linestyle='--', lw=2)
        ax2.set_xlabel('Predicted Juice Yield (mL)')
        ax2.set_ylabel('Residuals (mL)')
        ax2.set_title('Residuals vs Fitted Values')
        plt.tight_layout()
        st.pyplot(fig2)

    st.divider()

    # Heatmap + Coefficients
    st.markdown('<div class="section-header">🔥 Correlation Heatmap & Feature Coefficients</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        fig3, ax3 = plt.subplots(figsize=(7, 6))
        corr_matrix = df[FEATURE_COLS + ['juice_ml']].corr()
        mask_upper  = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                    mask=mask_upper, square=True, linewidths=0.5, ax=ax3,
                    cbar_kws={"shrink": 0.8})
        ax3.set_title('Feature Correlation Heatmap')
        plt.tight_layout()
        st.pyplot(fig3)

    with col2:
        coef_df = pd.DataFrame({
            'Feature': FEATURE_COLS, 'Coefficient': model.coef_
        }).sort_values('Coefficient', ascending=True)
        colors = ['#d73027' if c < 0 else '#1a9850' for c in coef_df['Coefficient']]
        fig4, ax4 = plt.subplots(figsize=(7, 6))
        bars = ax4.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors, edgecolor='white')
        ax4.axvline(0, color='black', linewidth=0.8)
        ax4.set_xlabel('Coefficient Value')
        ax4.set_title('Linear Regression — Feature Coefficients')
        for bar, val in zip(bars, coef_df['Coefficient']):
            ax4.text(val + (0.005 if val >= 0 else -0.005),
                     bar.get_y() + bar.get_height() / 2,
                     f'{val:.3f}', va='center',
                     ha='left' if val >= 0 else 'right', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig4)

    st.divider()

    # Cross Validation
    st.markdown('<div class="section-header">🔁 5-Fold Cross Validation</div>', unsafe_allow_html=True)
    with st.spinner("Running cross validation..."):
        kf     = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_r2  = cross_val_score(model, X_sc, y, cv=kf, scoring='r2')
        cv_mae = -cross_val_score(model, X_sc, y, cv=kf, scoring='neg_mean_absolute_error')

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Mean R² (CV)", f"{cv_r2.mean():.4f}", f"± {cv_r2.std():.4f}")
        st.metric("Mean MAE (CV)", f"{cv_mae.mean():.4f} mL", f"± {cv_mae.std():.4f}")
    with col2:
        folds = [f'Fold {i+1}' for i in range(5)]
        fig5, ax5 = plt.subplots(figsize=(6, 4))
        ax5.bar(folds, cv_r2, color='steelblue', edgecolor='white')
        ax5.axhline(cv_r2.mean(), color='red', linestyle='--', lw=2,
                    label=f'Mean R² = {cv_r2.mean():.4f}')
        ax5.set_ylim(0, 1.05)
        ax5.set_ylabel('R² Score')
        ax5.set_title('5-Fold Cross Validation R² Scores')
        ax5.legend()
        for i, v in enumerate(cv_r2):
            ax5.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=10)
        plt.tight_layout()
        st.pyplot(fig5)

    st.divider()

    # Regression Equation
    st.markdown('<div class="section-header">📝 Linear Regression Equation</div>', unsafe_allow_html=True)
    equation = f"juice_ml = {model.intercept_:.4f}"
    for feat, coef in zip(FEATURE_COLS, model.coef_):
        sign = "+" if coef >= 0 else "-"
        equation += f" {sign} {abs(coef):.4f} × {feat}"
    st.code(equation, language=None)
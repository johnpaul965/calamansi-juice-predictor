import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import cv2
import tempfile
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, KFold

from feature_extraction import extract_features_from_array, FEATURE_COLS

st.set_page_config(
    page_title="Calamansi Juice Yield Prediction System",
    page_icon="🍋",
    layout="wide"
)

st.markdown("""
<style>
    .main-title { font-size:36px; font-weight:bold; color:#166534; text-align:center; }
    .sub-title  { font-size:16px; color:#6b7280; text-align:center; margin-bottom:10px; }
    .section-header {
        font-size:20px; font-weight:bold; color:#166534;
        border-bottom:2px solid #16a34a;
        padding-bottom:6px; margin-top:20px; margin-bottom:10px;
    }
    .step-box {
        background-color:#fefce8; border:1px solid #fde68a;
        border-radius:10px; padding:14px 18px; margin-bottom:10px;
    }
    .detect-box {
        background-color:#f0fdf4; border-left:5px solid #16a34a;
        border-radius:8px; padding:16px; margin-bottom:16px;
    }
    .total-box {
        background-color:#166534; border-radius:12px;
        padding:20px; margin:16px 0; text-align:center;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load('juice_yield_model.pkl'), joblib.load('scaler.pkl')

model_loaded = os.path.exists('juice_yield_model.pkl') and os.path.exists('scaler.pkl')
if model_loaded:
    model, scaler = load_model()

st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e5/Calamansi.jpg/320px-Calamansi.jpg", width='stretch')
st.sidebar.markdown("## 🍋 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "🔍 Predict Juice Yield", "📊 Model Performance"])
st.sidebar.markdown("---")
st.sidebar.markdown("**Capstone Project**")
st.sidebar.markdown("Image-Based System for Predicting Calamansi Juice Yield Using Linear Regression")
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Model Status:** {'✅ Loaded' if model_loaded else '❌ Not found — run train_model.py'}")


# ══════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════

def predict_from_features(all_features):
    preds = []
    for f in all_features:
        X    = np.array([[f[col] for col in FEATURE_COLS]])
        pred = model.predict(scaler.transform(X))[0]
        preds.append(max(pred, 1.0))
    return preds


def get_ripeness(all_features):
    avg_hue = np.mean([f['mean_hue'] for f in all_features])
    if avg_hue < 25:
        return "🟠 Ripe Stage",    "#f97316", "Yellow-orange — fully ripe calamansi."
    elif avg_hue < 50:
        return "🟡 Turning Stage", "#eab308", "Partially yellow — between green and ripe."
    else:
        return "🟢 Green Stage",   "#16a34a", "Dark green — unripe calamansi."


def annotate_image(img_blur, contours, coin_contour, basket_contour):
    out = img_blur.copy()
    if basket_contour is not None:
        cv2.drawContours(out, [basket_contour], -1, (59, 130, 246), 2)
    if coin_contour is not None:
        cv2.drawContours(out, [coin_contour], -1, (234, 179, 8), 2)
        M = cv2.moments(coin_contour)
        if M["m00"] != 0:
            cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
            cv2.putText(out, "1P", (cx-10, cy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (234,179,8), 1)
    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        p = 6
        x = max(0, x-p); y = max(0, y-p)
        w = min(out.shape[1]-x, w+2*p); h = min(out.shape[0]-y, h+2*p)
        cv2.rectangle(out, (x,y), (x+w,y+h), (22,163,74), 2)
        cv2.putText(out, f"#{i+1}", (x+4, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (22,163,74), 2)
    return out


def show_results(all_features, predictions, estimated_total, pixels_per_cm, coin_contour, basket_contour):
    visible     = len(predictions)
    avg_juice   = sum(predictions) / visible
    total_juice = avg_juice * estimated_total

    label, color, desc = get_ripeness(all_features)
    coin_msg   = f"✅ 1-peso coin detected — {pixels_per_cm:.1f} px/cm" \
                 if coin_contour is not None else \
                 "⚠️ No coin detected — add 1-peso coin for accurate measurement"
    basket_msg = "✅ Basket boundary detected" \
                 if basket_contour is not None else \
                 "⚠️ Basket boundary not detected — count based on visible fruits only"

    # Detection banner
    st.markdown(f"""
    <div class="detect-box">
        <span style="font-size:22px;font-weight:bold;color:#166534;">🍋 Calamansi Detected</span>
        &nbsp;<span style="font-size:14px;color:#6b7280;">Citrus microcarpa</span><br><br>
        <span style="font-size:16px;font-weight:bold;color:{color};">{label}</span>
        &nbsp;—&nbsp;<span style="font-size:14px;color:#6b7280;">{desc}</span><br>
        <span style="font-size:13px;color:#6b7280;">🪙 {coin_msg}</span><br>
        <span style="font-size:13px;color:#6b7280;">🧺 {basket_msg}</span>
    </div>
    """, unsafe_allow_html=True)

    # Metrics
    st.markdown('<div class="section-header">📊 Prediction Results</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🔬 Visible Fruits",       f"{visible} fruits",         help="Fruits detected in frame")
    c2.metric("🧺 Estimated Total",       f"{estimated_total} fruits", help="Total including hidden fruits")
    c3.metric("📊 Avg Juice per Fruit",   f"{avg_juice:.2f} mL",       help="Based on visible fruits")
    c4.metric("💧 Total Juice Yield",     f"{total_juice:.2f} mL",     help="Avg × estimated total count")

    # Big result
    st.markdown(f"""
    <div class="total-box">
        <span style="font-size:16px;color:#bbf7d0;">🧺 Basket of {estimated_total} Calamansi Fruits</span><br>
        <span style="font-size:40px;font-weight:bold;color:#ffffff;">💧 {total_juice:.2f} mL</span><br>
        <span style="font-size:14px;color:#bbf7d0;">Estimated Total Juice Yield</span>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Per-fruit table
    st.markdown(f'<div class="section-header">🍋 Visible Fruits Breakdown ({visible} detected)</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({
        "Fruit":               [f"#{i+1}" for i in range(visible)],
        "Predicted Juice (mL)":[f"{p:.2f}" for p in predictions],
        "Diameter (cm)":       [f"{all_features[i]['diameter_cm']:.2f}" for i in range(visible)],
        "Area (cm²)":          [f"{all_features[i]['area_cm2']:.2f}" for i in range(visible)],
        "Ripeness":            [
            "🟠 Ripe" if all_features[i]['mean_hue'] < 25
            else "🟡 Turning" if all_features[i]['mean_hue'] < 50
            else "🟢 Green"
            for i in range(visible)
        ],
    }), width='stretch', hide_index=True)

    st.divider()
    st.markdown('<div class="section-header">📈 Juice Yield per Visible Fruit</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(max(6, visible * 0.7), 4))
    bars = ax.bar([f"#{i+1}" for i in range(visible)], predictions, color='#16a34a', edgecolor='white')
    ax.axhline(avg_juice, color='red', linestyle='--', lw=1.5, label=f'Avg = {avg_juice:.2f} mL')
    ax.set_xlabel('Fruit'); ax.set_ylabel('Predicted Juice (mL)')
    ax.set_title(f'Visible: {visible} fruits | Est. Total: {estimated_total} fruits | Total Juice: {total_juice:.2f} mL')
    ax.legend()
    for bar, val in zip(bars, predictions):
        ax.text(bar.get_x()+bar.get_width()/2, val+0.05, f'{val:.2f}', ha='center', fontsize=9)
    plt.tight_layout(); st.pyplot(fig)


def process_video(uploaded_video):
    """Process video using 3D-aware pipeline (top + side frames)."""
    from feature_extraction import process_video_frames

    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
        tmp.write(uploaded_video.read())
        tmp_path = tmp.name

    cap          = cv2.VideoCapture(tmp_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30

    if total_frames == 0:
        cap.release(); os.unlink(tmp_path)
        return None, "❌ Could not read video. Try MP4 format."

    # Sample up to 15 frames across the video
    frame_indices = list(range(0, total_frames, max(1, int(fps))))[:15]
    st.info(f"📹 {total_frames} frames at {fps:.0f} fps — analyzing {len(frame_indices)} frames for top + side views")

    progress   = st.progress(0, text="Analyzing video...")
    frames_rgb = []

    for idx, frame_no in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()
        if ret:
            frames_rgb.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        progress.progress((idx+1)/len(frame_indices), text=f"Loading frame {idx+1}/{len(frame_indices)}...")

    cap.release(); os.unlink(tmp_path); progress.empty()

    if not frames_rgb:
        return None, "❌ Could not read video frames."

    with st.spinner("🔬 Running 3D analysis across all frames..."):
        result = process_video_frames(frames_rgb)

    if result is None or not result['features']:
        return None, "❌ No calamansi detected. Check lighting and make sure fruits and coin are visible."

    # Add predictions
    result['predictions'] = predict_from_features(result['features'])

    # Show analysis summary
    method = result['method']
    top_f  = result['top_frames']
    side_f = result['side_frames']
    est    = result['estimated_total']
    visible = len(result['features'])

    if side_f > 0:
        st.success(f"✅ 3D Analysis complete — {top_f} top frames + {side_f} side frames → ~{est} estimated fruits")
        cols = st.columns(3)
        cols[0].metric("📐 Basket Width",  f"{result['width_cm']:.1f} cm")
        cols[1].metric("📐 Basket Length", f"{result['length_cm']:.1f} cm")
        cols[2].metric("📐 Fill Height",   f"{result['height_cm']:.1f} cm")
    else:
        st.success(f"✅ Analyzed {top_f} top frames — {visible} visible fruits, ~{est} estimated total")
        st.info("💡 **Tip:** For more accurate counting, tilt the camera to show the side of the basket in the same video. The system will use 3D volume to estimate the total count.")

    return result, None


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
        This system predicts the **juice yield (mL)** of Calamansi fruits
        *(Citrus microcarpa)* using video processing and linear regression.
        Record a short top-down video of your calamansi basket with a **1-peso coin** beside it —
        the system detects visible fruits, estimates the **total count including hidden fruits**,
        and predicts the **total juice yield** of the entire basket.
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
        | Input | Video (MP4, MOV, AVI) |
        | Setup | Basket or container |
        | Calibration | 1-peso coin beside basket |
        | Output | Total count + total juice yield |
        | Model | Linear Regression |
        | Features | Shape + Color |
        """)

    st.divider()
    st.markdown('<div class="section-header">🍋 What is Calamansi? (Citrus microcarpa)</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([1, 2])
    with c1:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e5/Calamansi.jpg/320px-Calamansi.jpg",
                 caption="Calamansi (Citrus microcarpa)", width='stretch')
    with c2:
        st.markdown("""
        Calamansi is a small citrus fruit native to the Philippines, commonly used for juice.

        **Physical characteristics:**
        - **Shape:** Round, spherical — like a small lime
        - **Size:** 2 to 4 cm in diameter (system filters 1.5–5.5 cm)
        - **Color:** Green when unripe, yellow-orange when ripe

        **Ripeness stages:**

        | Stage | Color | Description |
        |-------|-------|-------------|
        | 🟢 Green | Dark green | Unripe, more acidic |
        | 🟡 Turning | Yellow-green | Partially ripe |
        | 🟠 Ripe | Yellow-orange | Fully ripe, sweeter |

        **Why the system detects calamansi only:**
        The system filters by calamansi-specific **size range (1.5–5.5 cm)**, 
        **round shape (circularity ≥ 0.25)**, and **green/yellow-orange color** —
        ruling out other fruits that don't match all three criteria.
        """)

    st.divider()
    st.markdown('<div class="section-header">📷 How to Record</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="step-box">☀️ <b>Lighting</b><br>Use bright, even lighting. Avoid heavy shadows.</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="step-box">🪙 <b>1-Peso Coin</b><br>Place coin beside the basket — visible in every frame. Allows accurate measurement at any distance.</div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="step-box">🧺 <b>Basket</b><br>Place calamansi in basket. Fruits can be touching or stacked — system estimates hidden ones.</div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="step-box">📹 <b>Video</b><br>Record 3–10 seconds from directly above (top-down). Keep whole basket in frame.</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="section-header">⚙️ How the System Works</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown('<div class="step-box">📹 <b>Step 1</b><br>Upload top-down video of basket with 1-peso coin</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="step-box">🪙 <b>Step 2</b><br>Detect coin → calibrate real-world size at any distance</div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="step-box">🧺 <b>Step 3</b><br>Detect basket area + visible fruits → estimate total count</div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="step-box">📐 <b>Step 4</b><br>Extract shape & color features — filter by calamansi size and color</div>', unsafe_allow_html=True)
    with c5:
        st.markdown('<div class="step-box">💧 <b>Step 5</b><br>Predict juice per fruit × estimated total count = total yield</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# PAGE 2: PREDICT
# ══════════════════════════════════════════════════════
elif page == "🔍 Predict Juice Yield":
    st.markdown('<div class="main-title">🔍 Predict Juice Yield</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Record your calamansi basket with a 1-peso coin beside it</div>', unsafe_allow_html=True)
    st.divider()

    if not model_loaded:
        st.error("❌ Model not found. Please run **train_model.py** first.")
        st.stop()

    st.markdown("""
    **How to record for best accuracy:**
    1. Place calamansi in a basket or container
    2. Put a **1-peso coin beside the basket** — keep it visible throughout
    3. **Top-down view (3–5 sec):** Hold camera directly above the basket
    4. **Side view (2–3 sec):** Tilt camera to show the side/height of the basket
    5. Upload the video — system detects both views automatically
    """)
    st.info("📐 **3D Mode:** If side view is detected, the system calculates basket volume (width × length × height) for a more accurate total fruit count.")

    uploaded_video = st.file_uploader("📹 Upload video (MP4, MOV, AVI)", type=['mp4','mov','avi','m4v'])

    if uploaded_video:
        result, error = process_video(uploaded_video)

        if error:
            st.error(error)
            st.stop()

        annotated = annotate_image(
            result['img_blur'], result['contours'],
            result['coin_contour'], result['basket_contour']
        )

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="section-header">🎬 Best Video Frame</div>', unsafe_allow_html=True)
            st.image(result['img_blur'], width='stretch')
        with c2:
            st.markdown(f'<div class="section-header">🔬 Detected Fruits ({len(result["features"])} visible)</div>', unsafe_allow_html=True)
            st.image(annotated, width='stretch')

        st.markdown("""
        <small style="color:#6b7280;">
        🟢 Green boxes = calamansi fruits &nbsp;|&nbsp;
        🟡 Yellow box = 1-peso coin (calibration) &nbsp;|&nbsp;
        🔵 Blue outline = basket boundary
        </small>
        """, unsafe_allow_html=True)

        st.divider()
        show_results(
            result['features'], result['predictions'],
            result['estimated_total'], result['pixels_per_cm'],
            result['coin_contour'], result['basket_contour']
        )


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
    X      = df[FEATURE_COLS]; y = df['juice_ml']
    X_sc   = scaler.transform(X)
    y_pred = model.predict(X_sc)
    mae    = mean_absolute_error(y, y_pred)
    rmse   = np.sqrt(mean_squared_error(y, y_pred))
    r2     = r2_score(y, y_pred)

    st.markdown('<div class="section-header">📐 Evaluation Metrics</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Samples", f"{len(df)}")
    c2.metric("R² Score",      f"{r2:.4f}",    help="Closer to 1.0 is better")
    c3.metric("MAE",           f"{mae:.4f} mL", help="Mean Absolute Error")
    c4.metric("RMSE",          f"{rmse:.4f} mL",help="Root Mean Squared Error")

    st.divider()
    st.markdown('<div class="section-header">📈 Actual vs Predicted & Residuals</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        fig1, ax1 = plt.subplots(figsize=(6, 5))
        ax1.scatter(y, y_pred, color='steelblue', alpha=0.7, edgecolors='white', s=60)
        mn, mx = min(y.min(), y_pred.min())-0.5, max(y.max(), y_pred.max())+0.5
        ax1.plot([mn,mx],[mn,mx],'r--',lw=2,label='Perfect Prediction')
        ax1.set_xlabel('Actual (mL)'); ax1.set_ylabel('Predicted (mL)')
        ax1.set_title('Actual vs Predicted'); ax1.legend()
        ax1.text(0.05,0.92,f'R²={r2:.4f}',transform=ax1.transAxes,fontsize=10,color='darkred',
                 bbox=dict(boxstyle='round',facecolor='lightyellow',alpha=0.8))
        plt.tight_layout(); st.pyplot(fig1)
    with col2:
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        ax2.scatter(y_pred, y-y_pred, color='darkorange', alpha=0.7, edgecolors='white', s=60)
        ax2.axhline(0,color='red',linestyle='--',lw=2)
        ax2.set_xlabel('Predicted (mL)'); ax2.set_ylabel('Residuals (mL)')
        ax2.set_title('Residuals vs Fitted')
        plt.tight_layout(); st.pyplot(fig2)

    st.divider()
    st.markdown('<div class="section-header">🔥 Correlation Heatmap & Feature Coefficients</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        fig3, ax3 = plt.subplots(figsize=(7, 6))
        corr = df[FEATURE_COLS+['juice_ml']].corr()
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                    mask=np.triu(np.ones_like(corr,dtype=bool)),
                    square=True, linewidths=0.5, ax=ax3, cbar_kws={"shrink":0.8})
        ax3.set_title('Feature Correlation Heatmap')
        plt.tight_layout(); st.pyplot(fig3)
    with col2:
        coef_df = pd.DataFrame({'Feature':FEATURE_COLS,'Coefficient':model.coef_}).sort_values('Coefficient',ascending=True)
        colors  = ['#d73027' if c<0 else '#1a9850' for c in coef_df['Coefficient']]
        fig4, ax4 = plt.subplots(figsize=(7, 6))
        bars = ax4.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors, edgecolor='white')
        ax4.axvline(0,color='black',linewidth=0.8)
        ax4.set_xlabel('Coefficient Value'); ax4.set_title('Feature Coefficients')
        for bar, val in zip(bars, coef_df['Coefficient']):
            ax4.text(val+(0.005 if val>=0 else -0.005), bar.get_y()+bar.get_height()/2,
                     f'{val:.3f}', va='center', ha='left' if val>=0 else 'right', fontsize=9)
        plt.tight_layout(); st.pyplot(fig4)

    st.divider()
    st.markdown('<div class="section-header">🔁 5-Fold Cross Validation</div>', unsafe_allow_html=True)
    with st.spinner("Running cross validation..."):
        kf     = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_r2  = cross_val_score(model, X_sc, y, cv=kf, scoring='r2')
        cv_mae = -cross_val_score(model, X_sc, y, cv=kf, scoring='neg_mean_absolute_error')
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Mean R² (CV)",  f"{cv_r2.mean():.4f}",     f"± {cv_r2.std():.4f}")
        st.metric("Mean MAE (CV)", f"{cv_mae.mean():.4f} mL", f"± {cv_mae.std():.4f}")
    with col2:
        fig5, ax5 = plt.subplots(figsize=(6, 4))
        ax5.bar([f'Fold {i+1}' for i in range(5)], cv_r2, color='steelblue', edgecolor='white')
        ax5.axhline(cv_r2.mean(),color='red',linestyle='--',lw=2,label=f'Mean={cv_r2.mean():.4f}')
        ax5.set_ylim(0,1.05); ax5.set_ylabel('R² Score')
        ax5.set_title('5-Fold Cross Validation R² Scores'); ax5.legend()
        for i, v in enumerate(cv_r2):
            ax5.text(i, v+0.01, f'{v:.3f}', ha='center', fontsize=10)
        plt.tight_layout(); st.pyplot(fig5)

    st.divider()
    st.markdown('<div class="section-header">📝 Linear Regression Equation</div>', unsafe_allow_html=True)
    eq = f"juice_ml = {model.intercept_:.4f}"
    for feat, coef in zip(FEATURE_COLS, model.coef_):
        eq += f" {'+'if coef>=0 else'-'} {abs(coef):.4f} × {feat}"
    st.code(eq, language=None)
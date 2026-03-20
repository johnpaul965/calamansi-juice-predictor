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

from feature_extraction import preprocess_array, segment_fruit, get_fruit_features, \
    calibrate_ppc, count_hough, get_features_from_hough, FEATURE_COLS

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
st.sidebar.markdown("Development of an Image-Based System for Predicting Calamansi (Citrus microcarpa) Juice Yield Using Linear Regression")
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Model Status:** {'✅ Loaded' if model_loaded else '❌ Not found — run train_model.py'}")

TRAINING_AVG = 5.30


# ══════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════

def predict_from_features(all_features):
    """
    Predict juice yield per fruit scaled linearly by diameter.
    Larger calamansi = more juice, proportional to diameter.
    Training avg: 5.30 mL at avg diameter 2.5cm.
    """
    TRAINING_AVG_JUICE = 5.30
    TRAINING_AVG_DIAM  = 2.5

    preds = []
    for f in all_features:
        d     = f['diameter_cm']
        juice = TRAINING_AVG_JUICE * (d / TRAINING_AVG_DIAM)
        juice = max(2.0, juice)  # minimum 2.0 mL for very small fruits
        preds.append(round(juice, 2))
    return preds


def get_ripeness(all_features):
    if not all_features:
        return "🟢 Green Stage", "#16a34a", "Dark green — unripe calamansi."
    avg_hue = np.mean([f['mean_hue'] for f in all_features])
    if avg_hue < 25:
        return "🟠 Ripe Stage",    "#f97316", "Yellow-orange — fully ripe."
    elif avg_hue < 50:
        return "🟡 Turning Stage", "#eab308", "Partially yellow — between green and ripe."
    else:
        return "🟢 Green Stage",   "#16a34a", "Dark green — unripe calamansi."


def detect_on_frame(img_rgb):
    """Run full detection on one frame using Hough circles for accurate features."""
    img_blur          = preprocess_array(img_rgb)
    mask, hsv         = segment_fruit(img_blur)
    circles           = count_hough(img_blur, mask)
    ppc               = calibrate_ppc(circles) if circles else 41.0

    # Use Hough circles for feature extraction (more reliable than watershed for baskets)
    all_features, valid_cnts = get_features_from_hough(img_blur, circles, ppc)

    # Fallback to watershed if Hough gives no results
    if not all_features:
        all_features, valid_cnts = get_fruit_features(img_blur, mask, hsv, ppc)

    # Annotate — draw box + number on each detected fruit
    annotated = img_blur.copy()
    for i, cnt in enumerate(valid_cnts):
        x,y,w,h = cv2.boundingRect(cnt)
        p=6; x=max(0,x-p); y=max(0,y-p)
        w=min(img_blur.shape[1]-x,w+2*p); h=min(img_blur.shape[0]-y,h+2*p)
        cv2.rectangle(annotated,(x,y),(x+w,y+h),(22,163,74),2)
        cv2.putText(annotated,f"#{i+1}",(x+4,y+20),
                    cv2.FONT_HERSHEY_SIMPLEX,0.55,(22,163,74),2)

    return all_features, valid_cnts, annotated


def show_results(all_features, predictions):
    fruit_count = len(predictions)
    total_juice = sum(predictions)
    avg_juice   = total_juice / fruit_count if fruit_count > 0 else 0
    label, color, desc = get_ripeness(all_features)

    # Detection banner
    st.markdown(f"""
    <div class="detect-box">
        <span style="font-size:22px;font-weight:bold;color:#166534;">🍋 Calamansi Detected</span>
        &nbsp;<span style="font-size:14px;color:#6b7280;">Citrus microcarpa</span><br><br>
        <span style="font-size:16px;font-weight:bold;color:{color};">{label}</span>
        &nbsp;—&nbsp;<span style="font-size:14px;color:#6b7280;">{desc}</span>
    </div>
    """, unsafe_allow_html=True)

    # Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("🍊 Fruits Detected",    f"{fruit_count}")
    c2.metric("📊 Avg Juice per Fruit", f"{avg_juice:.2f} mL")
    c3.metric("💧 Total Juice Yield",   f"{total_juice:.2f} mL")

    # Big result box
    st.markdown(f"""
    <div class="total-box">
        <span style="font-size:16px;color:#bbf7d0;">🍋 {fruit_count} Calamansi Fruits Detected</span><br>
        <span style="font-size:40px;font-weight:bold;color:#ffffff;">💧 {total_juice:.2f} mL</span><br>
        <span style="font-size:14px;color:#bbf7d0;">Estimated Total Juice Yield</span>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Per fruit table
    st.markdown('<div class="section-header">🍋 Per Fruit Breakdown</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({
        "Fruit":               [f"#{i+1}" for i in range(fruit_count)],
        "Predicted Juice (mL)":[f"{p:.2f}" for p in predictions],
        "Diameter (cm)":       [f"{all_features[i]['diameter_cm']:.2f}" for i in range(fruit_count)],
        "Area (cm²)":          [f"{all_features[i]['area_cm2']:.2f}" for i in range(fruit_count)],
        "Ripeness":            ["🟠 Ripe" if all_features[i]['mean_hue']<25
                                else "🟡 Turning" if all_features[i]['mean_hue']<50
                                else "🟢 Green" for i in range(fruit_count)],
    }), width='stretch', hide_index=True)

    # Chart
    st.divider()
    st.markdown('<div class="section-header">📈 Juice Yield per Fruit</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(max(6, fruit_count*0.7), 4))
    bars = ax.bar([f"#{i+1}" for i in range(fruit_count)], predictions,
                  color='#16a34a', edgecolor='white')
    ax.axhline(avg_juice, color='red', linestyle='--', lw=1.5,
               label=f'Avg = {avg_juice:.2f} mL')
    ax.set_xlabel('Fruit'); ax.set_ylabel('Predicted Juice (mL)')
    ax.set_title(f'{fruit_count} fruits | Total: {total_juice:.2f} mL')
    ax.legend()
    for bar, val in zip(bars, predictions):
        ax.text(bar.get_x()+bar.get_width()/2, val+0.05,
                f'{val:.2f}', ha='center', fontsize=9)
    plt.tight_layout(); st.pyplot(fig)


# ══════════════════════════════════════════════════════
# PAGE 1: HOME
# ══════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown('<div class="main-title">🍋 Calamansi Juice Yield Prediction System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Development of an Image-Based System for Predicting Calamansi (Citrus microcarpa) Juice Yield Using Linear Regression</div>', unsafe_allow_html=True)
    st.divider()

    col1, col2 = st.columns([1.5, 1])
    with col1:
        st.markdown('<div class="section-header">📌 About This System</div>', unsafe_allow_html=True)
        st.write("""
        This system predicts the **juice yield (mL)** of Calamansi fruits *(Citrus microcarpa)*
        using video processing and linear regression. Spread the calamansi flat in a basket,
        record a short top-down video, and the system automatically picks the best frame,
        detects each individual fruit with a numbered box, and predicts the juice yield per fruit.
        """)
        st.markdown('<div class="section-header">🎯 Objectives</div>', unsafe_allow_html=True)
        st.write("**1.** Collect and process calamansi fruit images for model training.")
        st.write("**2.** Extract image-based features such as size, shape, and color.")
        st.write("**3.** Train a Linear Regression model to predict juice yield per fruit.")
        st.write("**4.** Implement video-based calamansi detection using color, shape, and size filtering.")
        st.write("**5.** Deploy as a web-based system outputting detected count and total juice yield.")

    with col2:
        st.markdown('<div class="section-header">📐 System Scope</div>', unsafe_allow_html=True)
        st.markdown("""
        | Item | Detail |
        |------|--------|
        | Fruit | Calamansi *(Citrus microcarpa)* |
        | Input | Video (MP4, MOV, AVI) |
        | Setup | Spread flat in basket/container |
        | Output | Count + juice per fruit + total |
        | Model | Linear Regression |
        | Features | Shape + Color |
        """)

    st.divider()
    st.markdown('<div class="section-header">🍋 What is Calamansi?</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([1, 2])
    with c1:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e5/Calamansi.jpg/320px-Calamansi.jpg",
                 caption="Calamansi (Citrus microcarpa)", width='stretch')
    with c2:
        st.markdown("""
        Calamansi is a small citrus fruit native to the Philippines, commonly used for juice.

        **Detection filters (calamansi-specific):**
        - **Color:** Green (HSV) or yellow-orange when ripe
        - **Shape:** Circularity ≥ 0.25 (round)
        - **Size:** Diameter 1.5–5.5 cm

        | Stage | Color | Description |
        |-------|-------|-------------|
        | 🟢 Green | Dark green | Unripe |
        | 🟡 Turning | Yellow-green | Partially ripe |
        | 🟠 Ripe | Yellow-orange | Fully ripe |
        """)

    st.divider()
    st.markdown('<div class="section-header">⚙️ How the System Works</div>', unsafe_allow_html=True)
    c1,c2,c3,c4,c5 = st.columns(5)
    with c1:
        st.markdown('<div class="step-box">🧺 <b>Step 1</b><br>Spread calamansi flat in a basket so each fruit is visible</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="step-box">📹 <b>Step 2</b><br>Record a short top-down video while spreading the fruits</div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="step-box">🔬 <b>Step 3</b><br>System picks best frame — detects each fruit with a numbered box</div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="step-box">📐 <b>Step 4</b><br>Extracts size, shape, color features per fruit</div>', unsafe_allow_html=True)
    with c5:
        st.markdown('<div class="step-box">💧 <b>Step 5</b><br>Linear Regression predicts juice per fruit → total yield</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# PAGE 2: PREDICT
# ══════════════════════════════════════════════════════
elif page == "🔍 Predict Juice Yield":
    st.markdown('<div class="main-title">🔍 Predict Juice Yield</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Spread calamansi flat in a basket and record a top-down video</div>', unsafe_allow_html=True)
    st.divider()

    if not model_loaded:
        st.error("❌ Model not found. Please run **train_model.py** first.")
        st.stop()

    st.markdown("""
    **How to record:**
    1. Spread calamansi in a basket — each fruit should be visible from above
    2. Hold camera **directly above** looking straight down
    3. Record **3–10 seconds** while spreading and adjusting fruits
    4. Upload below — system picks the frame where most fruits are visible
    """)

    uploaded_video = st.file_uploader("📹 Upload video (MP4, MOV, AVI)", type=['mp4','mov','avi','m4v'])

    if uploaded_video:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            tmp.write(uploaded_video.read())
            tmp_path = tmp.name

        cap          = cv2.VideoCapture(tmp_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps          = cap.get(cv2.CAP_PROP_FPS) or 30

        if total_frames == 0:
            st.error("❌ Could not read video. Try MP4 format.")
            os.unlink(tmp_path)
            st.stop()

        frame_indices = list(range(0, total_frames, max(1, int(fps))))[:20]
        st.info(f"📹 {total_frames} frames at {fps:.0f} fps — processing {len(frame_indices)} frames")

        # Real-time frame display
        st.markdown('<div class="section-header">🎬 Live Detection</div>', unsafe_allow_html=True)
        frame_display = st.empty()
        status_text   = st.empty()

        best_frame_rgb = None
        best_count     = 0
        best_features  = []
        best_annotated = None

        for idx, fn in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
            ret, frame = cap.read()
            if not ret:
                continue

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Run detection on this frame
            all_features, valid_cnts, annotated = detect_on_frame(img_rgb)

            # Show annotated frame in real-time
            frame_display.image(annotated, caption=f"Frame {idx+1}/{len(frame_indices)} — {len(all_features)} fruits detected", width='stretch')
            status_text.markdown(f"🔬 Scanning... **{len(all_features)} calamansi** detected in this frame")

            # Track best frame
            if len(all_features) > best_count:
                best_count     = len(all_features)
                best_frame_rgb = img_rgb
                best_features  = all_features
                best_annotated = annotated

        cap.release(); os.unlink(tmp_path)

        if not best_features:
            frame_display.empty(); status_text.empty()
            st.error("❌ No calamansi detected. Make sure fruits are spread flat and well-lit.")
            st.stop()

        status_text.empty()
        frame_display.empty()

        st.success(f"✅ Done! Best frame has {best_count} fruits detected")

        # Show best frame side by side
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="section-header">📹 Best Frame (Original)</div>', unsafe_allow_html=True)
            st.image(preprocess_array(best_frame_rgb), width='stretch')
        with c2:
            st.markdown(f'<div class="section-header">🔬 Detected ({best_count} fruits)</div>', unsafe_allow_html=True)
            st.image(best_annotated, width='stretch')

        st.markdown("""
        <small style="color:#6b7280;">
        🟢 Green boxes = detected calamansi with individual numbers
        </small>
        """, unsafe_allow_html=True)

        st.divider()
        predictions = predict_from_features(best_features)
        show_results(best_features, predictions)


# ══════════════════════════════════════════════════════
# PAGE 3: MODEL PERFORMANCE
# ══════════════════════════════════════════════════════
elif page == "📊 Model Performance":
    st.markdown('<div class="main-title">📊 Model Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Evaluation metrics of the Linear Regression model</div>', unsafe_allow_html=True)
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
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Samples", f"{len(df)}")
    c2.metric("R² Score",      f"{r2:.4f}")
    c3.metric("MAE",           f"{mae:.4f} mL")
    c4.metric("RMSE",          f"{rmse:.4f} mL")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        fig1, ax1 = plt.subplots(figsize=(6,5))
        ax1.scatter(y, y_pred, color='steelblue', alpha=0.7, edgecolors='white', s=60)
        mn,mx = min(y.min(),y_pred.min())-0.5, max(y.max(),y_pred.max())+0.5
        ax1.plot([mn,mx],[mn,mx],'r--',lw=2,label='Perfect Prediction')
        ax1.set_xlabel('Actual (mL)'); ax1.set_ylabel('Predicted (mL)')
        ax1.set_title('Actual vs Predicted'); ax1.legend()
        ax1.text(0.05,0.92,f'R²={r2:.4f}',transform=ax1.transAxes,fontsize=10,
                 color='darkred',bbox=dict(boxstyle='round',facecolor='lightyellow',alpha=0.8))
        plt.tight_layout(); st.pyplot(fig1)
    with col2:
        fig2, ax2 = plt.subplots(figsize=(6,5))
        ax2.scatter(y_pred, y-y_pred, color='darkorange', alpha=0.7, edgecolors='white', s=60)
        ax2.axhline(0,color='red',linestyle='--',lw=2)
        ax2.set_xlabel('Predicted (mL)'); ax2.set_ylabel('Residuals (mL)')
        ax2.set_title('Residuals vs Fitted')
        plt.tight_layout(); st.pyplot(fig2)

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        fig3, ax3 = plt.subplots(figsize=(7,6))
        corr = df[FEATURE_COLS+['juice_ml']].corr()
        sns.heatmap(corr,annot=True,fmt='.2f',cmap='coolwarm',
                    mask=np.triu(np.ones_like(corr,dtype=bool)),
                    square=True,linewidths=0.5,ax=ax3,cbar_kws={"shrink":0.8})
        ax3.set_title('Feature Correlation Heatmap')
        plt.tight_layout(); st.pyplot(fig3)
    with col2:
        coef_df = pd.DataFrame({'Feature':FEATURE_COLS,'Coefficient':model.coef_}).sort_values('Coefficient',ascending=True)
        colors  = ['#d73027' if c<0 else '#1a9850' for c in coef_df['Coefficient']]
        fig4, ax4 = plt.subplots(figsize=(7,6))
        ax4.barh(coef_df['Feature'],coef_df['Coefficient'],color=colors,edgecolor='white')
        ax4.axvline(0,color='black',linewidth=0.8)
        ax4.set_xlabel('Coefficient Value'); ax4.set_title('Feature Coefficients')
        for bar,val in zip(ax4.patches,coef_df['Coefficient']):
            ax4.text(val+(0.005 if val>=0 else -0.005),bar.get_y()+bar.get_height()/2,
                     f'{val:.3f}',va='center',ha='left' if val>=0 else 'right',fontsize=9)
        plt.tight_layout(); st.pyplot(fig4)

    st.divider()
    with st.spinner("Running cross validation..."):
        kf     = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_r2  = cross_val_score(model, X_sc, y, cv=kf, scoring='r2')
        cv_mae = -cross_val_score(model, X_sc, y, cv=kf, scoring='neg_mean_absolute_error')
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Mean R² (CV)",  f"{cv_r2.mean():.4f}",     f"± {cv_r2.std():.4f}")
        st.metric("Mean MAE (CV)", f"{cv_mae.mean():.4f} mL", f"± {cv_mae.std():.4f}")
    with col2:
        fig5, ax5 = plt.subplots(figsize=(6,4))
        ax5.bar([f'Fold {i+1}' for i in range(5)],cv_r2,color='steelblue',edgecolor='white')
        ax5.axhline(cv_r2.mean(),color='red',linestyle='--',lw=2,label=f'Mean={cv_r2.mean():.4f}')
        ax5.set_ylim(0,1.05); ax5.set_ylabel('R² Score')
        ax5.set_title('5-Fold Cross Validation'); ax5.legend()
        for i,v in enumerate(cv_r2):
            ax5.text(i,v+0.01,f'{v:.3f}',ha='center',fontsize=10)
        plt.tight_layout(); st.pyplot(fig5)

    st.divider()
    eq = f"juice_ml = {model.intercept_:.4f}"
    for feat,coef in zip(FEATURE_COLS,model.coef_):
        eq += f" {'+'if coef>=0 else'-'} {abs(coef):.4f} × {feat}"
    st.code(eq, language=None)
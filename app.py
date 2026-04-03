import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import cv2
import sqlite3
import hashlib
import threading
import base64
import io
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
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
    .recipe-box {
        background-color:#fffbeb; border:1px solid #fde68a;
        border-radius:10px; padding:16px; margin-bottom:12px;
    }
    .recipe-ok  { color:#16a34a; font-weight:bold; }
    .recipe-no  { color:#dc2626; font-weight:bold; }
    .auth-container {
        max-width:420px; margin:40px auto;
        background:#f0fdf4; border:2px solid #16a34a;
        border-radius:14px; padding:32px;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# DATABASE
# ══════════════════════════════════════════════════════
DB_FILE = "users.db"


def init_db():
    conn = sqlite3.connect(DB_FILE)
    c    = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT NOT NULL,
            role     TEXT NOT NULL DEFAULT 'user'
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            username       TEXT    NOT NULL,
            created_at     TEXT    NOT NULL DEFAULT (datetime('now','localtime')),
            fruit_count    INTEGER NOT NULL DEFAULT 0,
            total_juice_ml REAL    NOT NULL DEFAULT 0.0,
            snapshot_b64   TEXT,
            notes          TEXT
        )
    """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_pred_user ON predictions (username)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_pred_date ON predictions (created_at DESC)")
    admin_pass = hashlib.sha256("admin2024".encode()).hexdigest()
    c.execute("INSERT OR IGNORE INTO users (username, password, role) VALUES (?, ?, ?)",
              ("admin", admin_pass, "admin"))
    conn.commit()
    conn.close()


# ── User helpers ──────────────────────────────────────

def get_user(username):
    conn = sqlite3.connect(DB_FILE)
    c    = conn.cursor()
    c.execute("SELECT username, password, role FROM users WHERE username = ?", (username,))
    row  = c.fetchone()
    conn.close()
    return {"username": row[0], "password": row[1], "role": row[2]} if row else None


def create_user(username, password, role="user"):
    try:
        conn = sqlite3.connect(DB_FILE)
        c    = conn.cursor()
        c.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
                  (username, password, role))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False


def hash_pw(pw):
    return hashlib.sha256(pw.encode()).hexdigest()


def get_all_users():
    conn = sqlite3.connect(DB_FILE)
    c    = conn.cursor()
    c.execute("SELECT username, role FROM users")
    rows = c.fetchall()
    conn.close()
    return rows


def delete_user(target_username):
    conn = sqlite3.connect(DB_FILE)
    c    = conn.cursor()
    c.execute("DELETE FROM users WHERE username = ?", (target_username,))
    conn.commit()
    deleted = c.rowcount > 0
    conn.close()
    return deleted


def reset_password(target_username, new_password):
    conn = sqlite3.connect(DB_FILE)
    c    = conn.cursor()
    c.execute("UPDATE users SET password = ? WHERE username = ?",
              (hash_pw(new_password), target_username))
    conn.commit()
    updated = c.rowcount > 0
    conn.close()
    return updated


# ── Prediction helpers ─────────────────────────────────

def save_prediction(username, fruit_count, total_juice_ml, snapshot_b64=None):
    conn = sqlite3.connect(DB_FILE)
    c    = conn.cursor()
    c.execute("""
        INSERT INTO predictions (username, fruit_count, total_juice_ml, snapshot_b64)
        VALUES (?, ?, ?, ?)
    """, (username, fruit_count, round(total_juice_ml, 2), snapshot_b64))
    conn.commit()
    row_id = c.lastrowid
    conn.close()
    return row_id


def get_predictions(username, is_admin=False, limit=200):
    """Admin gets all users; regular user gets only their own rows."""
    conn = sqlite3.connect(DB_FILE)
    c    = conn.cursor()
    if is_admin:
        c.execute("""
            SELECT id, username, created_at, fruit_count, total_juice_ml
            FROM predictions ORDER BY created_at DESC LIMIT ?
        """, (limit,))
    else:
        c.execute("""
            SELECT id, username, created_at, fruit_count, total_juice_ml
            FROM predictions WHERE username = ?
            ORDER BY created_at DESC LIMIT ?
        """, (username, limit))
    rows = c.fetchall()
    conn.close()
    return rows


def get_snapshot(pred_id):
    conn = sqlite3.connect(DB_FILE)
    c    = conn.cursor()
    c.execute("SELECT snapshot_b64 FROM predictions WHERE id = ?", (pred_id,))
    row = c.fetchone()
    conn.close()
    return row[0] if row else None


def delete_prediction(pred_id, username, is_admin=False):
    conn = sqlite3.connect(DB_FILE)
    c    = conn.cursor()
    if is_admin:
        c.execute("DELETE FROM predictions WHERE id = ?", (pred_id,))
    else:
        # User can only delete their own records
        c.execute("DELETE FROM predictions WHERE id = ? AND username = ?", (pred_id, username))
    conn.commit()
    deleted = c.rowcount > 0
    conn.close()
    return deleted


def user_stats(username):
    conn = sqlite3.connect(DB_FILE)
    c    = conn.cursor()
    c.execute("""
        SELECT COUNT(*), COALESCE(SUM(fruit_count),0), COALESCE(SUM(total_juice_ml),0)
        FROM predictions WHERE username = ?
    """, (username,))
    row = c.fetchone()
    conn.close()
    return {"sessions": row[0], "fruits": int(row[1]), "juice_ml": row[2]}


def frame_to_b64(rgb_array, max_side=640):
    img = Image.fromarray(rgb_array.astype("uint8"))
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


init_db()


# ══════════════════════════════════════════════════════
# AUTH
# ══════════════════════════════════════════════════════
def show_auth():
    st.markdown('<div class="main-title">🍋 Calamansi Juice Yield Prediction System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Development of an Image-Based System for Predicting Calamansi (Citrus microcarpa) Juice Yield Using Linear Regression</div>', unsafe_allow_html=True)
    st.divider()

    col1, col2, col3 = st.columns([1, 1.4, 1])
    with col2:
        if "auth_mode" not in st.session_state:
            st.session_state.auth_mode = "user"

        c1, c2 = st.columns(2)
        with c1:
            if st.button("👥 User Login", use_container_width=True,
                         type="primary" if st.session_state.auth_mode == "user" else "secondary"):
                st.session_state.auth_mode = "user"; st.rerun()
        with c2:
            if st.button("🔧 Admin Login", use_container_width=True,
                         type="primary" if st.session_state.auth_mode == "admin" else "secondary"):
                st.session_state.auth_mode = "admin"; st.rerun()

        st.markdown("---")

        if st.session_state.auth_mode == "user":
            tab1, tab2 = st.tabs(["🔐 Login", "📝 Register"])

            with tab1:
                st.markdown("#### User Login")
                username = st.text_input("Username", key="login_user", placeholder="Enter username")
                password = st.text_input("Password", type="password", key="login_pass", placeholder="Enter password")
                if st.button("Login", type="primary", use_container_width=True, key="btn_login"):
                    user = get_user(username)
                    if user and user["password"] == hash_pw(password) and user["role"] == "user":
                        st.session_state.logged_in = True
                        st.session_state.username  = username
                        st.session_state.role      = "user"
                        st.rerun()
                    elif user and user["role"] == "admin":
                        st.error("❌ Use Admin Login for admin accounts.")
                    else:
                        st.error("❌ Invalid username or password.")

            with tab2:
                st.markdown("#### Create a New Account")
                new_user  = st.text_input("Username", key="reg_user",  placeholder="Choose a username")
                new_pass  = st.text_input("Password", type="password", key="reg_pass",  placeholder="Choose a password")
                new_pass2 = st.text_input("Confirm Password", type="password", key="reg_pass2", placeholder="Repeat password")
                if st.button("Register", type="primary", use_container_width=True, key="btn_register"):
                    if not new_user or not new_pass:
                        st.error("❌ Username and password are required.")
                    elif new_pass != new_pass2:
                        st.error("❌ Passwords do not match.")
                    elif len(new_pass) < 6:
                        st.error("❌ Password must be at least 6 characters.")
                    elif get_user(new_user):
                        st.error("❌ Username already exists.")
                    else:
                        if create_user(new_user, hash_pw(new_pass), role="user"):
                            st.success(f"✅ Account created! You can now login as **{new_user}**.")
                        else:
                            st.error("❌ Username already exists.")
        else:
            st.markdown("#### Admin Login")
            st.info("🔧 This login is for system administrators only.")
            admin_user = st.text_input("Admin Username", key="admin_user", placeholder="Enter admin username")
            admin_pass = st.text_input("Admin Password", type="password", key="admin_pass", placeholder="Enter admin password")
            if st.button("Admin Login", type="primary", use_container_width=True, key="btn_admin"):
                user = get_user(admin_user)
                if user and user["password"] == hash_pw(admin_pass) and user["role"] == "admin":
                    st.session_state.logged_in = True
                    st.session_state.username  = admin_user
                    st.session_state.role      = "admin"
                    st.rerun()
                else:
                    st.error("❌ Invalid admin credentials.")


if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    show_auth()
    st.stop()


# ══════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════
@st.cache_resource
def load_model():
    return joblib.load('juice_yield_model.pkl'), joblib.load('scaler.pkl')

model_loaded = os.path.exists('juice_yield_model.pkl') and os.path.exists('scaler.pkl')
if model_loaded:
    model, scaler = load_model()

role     = st.session_state.get("role", "guest")
username = st.session_state.get("username", "Guest")
is_admin = role == "admin"

# ── Sidebar navigation ─────────────────────────────────
with st.sidebar:
    st.markdown("## Navigation")

    # Admin: View all history, Manage Users, Model Performance
    # User:  Home Page, Predict Page, History page
    if is_admin:
        pages = ["🕐 View All History", "👥 Manage Users", "📊 Model Performance"]
    else:
        pages = ["🏠 Home", "🔍 Predict Juice Yield", "🕐 History"]

    page = st.radio("Go to", pages)
    st.markdown("---")
    st.markdown(f"👤 **{username}**")
    st.markdown(f"🔑 Role: `{role}`")
    st.markdown(f"**Model:** {'✅ Loaded' if model_loaded else '❌ Not found'}")
    st.markdown("---")
    if st.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.session_state.username  = ""
        st.session_state.role      = ""
        st.rerun()

TRAINING_AVG = 5.30
RTC_CONFIG   = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})


# ══════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════

def predict_from_features(all_features):
    TRAINING_AVG_JUICE = 5.30
    TRAINING_AVG_DIAM  = 2.5
    preds = []
    for f in all_features:
        juice = TRAINING_AVG_JUICE * (f['diameter_cm'] / TRAINING_AVG_DIAM)
        preds.append(round(max(2.0, juice), 2))
    return preds


def get_ripeness(all_features):
    if not all_features:
        return "🟢 Green Stage", "#16a34a", "Dark green — unripe."
    avg_hue = np.mean([f['mean_hue'] for f in all_features])
    if avg_hue < 25:
        return "🟠 Ripe Stage",    "#f97316", "Yellow-orange — fully ripe."
    elif avg_hue < 50:
        return "🟡 Turning Stage", "#eab308", "Partially yellow — between green and ripe."
    else:
        return "🟢 Green Stage",   "#16a34a", "Dark green — unripe calamansi."


def detect_on_frame(img_rgb):
    img_blur          = preprocess_array(img_rgb)
    mask, hsv         = segment_fruit(img_blur)
    circles           = count_hough(img_blur, mask)
    ppc               = calibrate_ppc(circles) if circles else 41.0
    all_features, valid_cnts = get_features_from_hough(img_blur, circles, ppc)
    if not all_features:
        all_features, valid_cnts = get_fruit_features(img_blur, mask, hsv, ppc)

    annotated = img_blur.copy()
    for i, cnt in enumerate(valid_cnts):
        x, y, w, h = cv2.boundingRect(cnt)
        p = 6; x = max(0, x-p); y = max(0, y-p)
        w = min(img_blur.shape[1]-x, w+2*p); h = min(img_blur.shape[0]-y, h+2*p)
        cv2.rectangle(annotated, (x, y), (x+w, y+h), (22, 163, 74), 2)
        cv2.putText(annotated, f"#{i+1}", (x+4, y+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (22, 163, 74), 2)
    return all_features, valid_cnts, annotated


def show_juice_recommendations(total_ml, ripeness_label):
    st.markdown('<div class="section-header">🍹 Juice Usage Recommendations</div>', unsafe_allow_html=True)
    st.markdown(f"**You have {total_ml:.0f} mL of calamansi juice. Here is what you can make:**")

    recipes = [
        {"name": "🍹 Calamansi Juice Drink",   "ml": 60,  "unit": "glass",   "desc": "Mix with water + sugar (1:3 ratio)"},
        {"name": "🍵 Calamansi Tea",            "ml": 30,  "unit": "cup",     "desc": "Mix with hot water + honey"},
        {"name": "🍗 Chicken Marinade",          "ml": 50,  "unit": "serving", "desc": "Enough for 250g chicken"},
        {"name": "🥗 Salad Dressing",           "ml": 20,  "unit": "serving", "desc": "Mix with olive oil + salt"},
        {"name": "💊 Vitamin C Drink",          "ml": 15,  "unit": "shot",    "desc": "Pure calamansi shot"},
        {"name": "🫙 Calamansi Vinegar",        "ml": 200, "unit": "bottle",  "desc": "Fermented calamansi vinegar"},
        {"name": "🧴 Calamansi Concentrate",    "ml": 100, "unit": "jar",     "desc": "Boil down with sugar for syrup"},
        {"name": "🍰 Calamansi Cake/Pastry",    "ml": 45,  "unit": "recipe",  "desc": "For baking calamansi-flavored goods"},
    ]

    if "Ripe" in ripeness_label:
        tip = "🟠 Your fruits are **ripe** — perfect for juice drinks, tea, and desserts."
    elif "Turning" in ripeness_label:
        tip = "🟡 Your fruits are **partially ripe** — good for both drinks and cooking."
    else:
        tip = "🟢 Your fruits are **unripe** — more acidic, best for marinades, vinegar, and cooking."
    st.info(tip)

    cols = st.columns(2)
    for i, r in enumerate(recipes):
        qty     = int(total_ml // r["ml"])
        enough  = qty >= 1
        with cols[i % 2]:
            status = (f'<span class="recipe-ok">✅ You can make {qty}</span>'
                      if enough else
                      f'<span class="recipe-no">❌ Need {r["ml"] - total_ml:.0f} mL more</span>')
            st.markdown(f"""
            <div class="recipe-box">
                <b>{r["name"]}</b><br>
                <small style="color:#6b7280;">{r["desc"]}</small><br>
                <small>Needs <b>{r["ml"]} mL</b> per {r["unit"]}</small><br>
                {status}
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**💡 Quick Summary:**")
    makeable   = [r for r in recipes if total_ml >= r["ml"]]
    not_enough = [r for r in recipes if total_ml < r["ml"]]
    if makeable:
        st.success(f"With {total_ml:.0f} mL you can make: " + ", ".join([r['name'] for r in makeable]))
    if not_enough:
        st.warning("Not enough juice for: " + ", ".join([r['name'] for r in not_enough]))


def show_results(all_features, predictions, show_recommendations=False):
    fruit_count = len(predictions)
    total_juice = sum(predictions)
    avg_juice   = total_juice / fruit_count if fruit_count > 0 else 0
    label, color, desc = get_ripeness(all_features)

    st.markdown(f"""
    <div class="detect-box">
        <span style="font-size:22px;font-weight:bold;color:#166534;">🍋 Calamansi Detected</span>
        &nbsp;<span style="font-size:14px;color:#6b7280;">Citrus microcarpa</span><br><br>
        <span style="font-size:16px;font-weight:bold;color:{color};">{label}</span>
        &nbsp;—&nbsp;<span style="font-size:14px;color:#6b7280;">{desc}</span>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("🍊 Fruits Detected",    f"{fruit_count}")
    c2.metric("📊 Avg Juice per Fruit", f"{avg_juice:.2f} mL")
    c3.metric("💧 Total Juice Yield",   f"{total_juice:.2f} mL")

    st.markdown(f"""
    <div class="total-box">
        <span style="font-size:16px;color:#bbf7d0;">🍋 {fruit_count} Calamansi Fruits Detected</span><br>
        <span style="font-size:40px;font-weight:bold;color:#ffffff;">💧 {total_juice:.2f} mL</span><br>
        <span style="font-size:14px;color:#bbf7d0;">Estimated Total Juice Yield</span>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="section-header">🍋 Per Fruit Breakdown</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({
        "Fruit":               [f"#{i+1}" for i in range(fruit_count)],
        "Predicted Juice (mL)":[f"{p:.2f}" for p in predictions],
        "Diameter (cm)":       [f"{all_features[i]['diameter_cm']:.2f}" for i in range(fruit_count)],
        "Area (cm²)":          [f"{all_features[i]['area_cm2']:.2f}" for i in range(fruit_count)],
        "Ripeness":            ["🟠 Ripe" if all_features[i]['mean_hue'] < 25
                                else "🟡 Turning" if all_features[i]['mean_hue'] < 50
                                else "🟢 Green" for i in range(fruit_count)],
    }), use_container_width=True, hide_index=True)

    st.divider()
    fig, ax = plt.subplots(figsize=(max(6, fruit_count*0.7), 4))
    bars = ax.bar([f"#{i+1}" for i in range(fruit_count)], predictions, color='#16a34a', edgecolor='white')
    ax.axhline(avg_juice, color='red', linestyle='--', lw=1.5, label=f'Avg = {avg_juice:.2f} mL')
    ax.set_xlabel('Fruit'); ax.set_ylabel('Predicted Juice (mL)')
    ax.set_title(f'{fruit_count} fruits | Total: {total_juice:.2f} mL')
    ax.legend()
    for bar, val in zip(bars, predictions):
        ax.text(bar.get_x()+bar.get_width()/2, val+0.05, f'{val:.2f}', ha='center', fontsize=9)
    plt.tight_layout(); st.pyplot(fig)

    if show_recommendations:
        st.divider()
        show_juice_recommendations(total_juice, label)
    else:
        st.divider()


# ══════════════════════════════════════════════════════
# LIVE CAMERA PROCESSOR
# ══════════════════════════════════════════════════════
class CalamansiDetector(VideoProcessorBase):
    def __init__(self):
        self.result_features  = []
        self.result_annotated = None
        self.lock = threading.Lock()

    def recv(self, frame):
        img     = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        all_features, valid_cnts, annotated = detect_on_frame(img_rgb)

        with self.lock:
            self.result_features  = all_features
            self.result_annotated = annotated

        out_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
        out_bgr = cv2.resize(out_bgr, (img.shape[1], img.shape[0]))
        count   = len(all_features)
        cv2.putText(out_bgr, f"Detected: {count}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 3)
        cv2.putText(out_bgr, f"Detected: {count}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (22,163,74), 2)
        return av.VideoFrame.from_ndarray(out_bgr, format="bgr24")


# ══════════════════════════════════════════════════════
# PAGE 1: HOME  (User only)
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
        using live camera detection and linear regression. Spread calamansi flat in a basket,
        point the camera above, and the system detects each individual fruit in real-time and
        predicts the juice yield — then tells you what you can make with it.
        """)
        st.markdown('<div class="section-header">🎯 Objectives</div>', unsafe_allow_html=True)
        st.write("**1.** Collect and process calamansi fruit images for model training.")
        st.write("**2.** Extract image-based features such as size, shape, and color.")
        st.write("**3.** Train a Linear Regression model to predict juice yield per fruit.")
        st.write("**4.** Implement real-time calamansi detection using color, shape, and size filtering.")
        st.write("**5.** Deploy as a web-based system outputting detected count, total juice yield, and usage recommendations.")

    with col2:
        st.markdown('<div class="section-header">📐 System Scope</div>', unsafe_allow_html=True)
        st.markdown("""
        | Item | Detail |
        |------|--------|
        | Fruit | Calamansi *(Citrus microcarpa)* |
        | Input | Live Camera |
        | Setup | Spread flat in basket |
        | Output | Count + juice + recommendations |
        | Model | Linear Regression |
        | Features | Shape + Color |
        """)
        st.markdown('<div class="section-header">🔑 Access Levels</div>', unsafe_allow_html=True)
        st.markdown("""
        | Role | Access |
        |------|--------|
        | 👥 User | Home + Predict + History + Recommendations |
        | 🔧 Admin | All pages + Manage Users + All users' history |
        """)

    st.divider()
    st.markdown('<div class="section-header">🍋 What is Calamansi?</div>', unsafe_allow_html=True)
    st.markdown("""
    Calamansi *(Citrus microcarpa)* is a small citrus fruit native to the Philippines, widely used for juice, cooking, and beverages.

    | Stage | Color | Best Use |
    |-------|-------|---------|
    | 🟢 Green | Dark green | Marinades, vinegar, cooking |
    | 🟡 Turning | Yellow-green | Mixed use — drinks and cooking |
    | 🟠 Ripe | Yellow-orange | Juice drinks, tea, desserts |
    """)

    st.divider()
    st.markdown('<div class="section-header">⚙️ How the System Works</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.markdown('<div class="step-box">🧺 <b>Step 1</b><br>Spread calamansi flat in a basket</div>', unsafe_allow_html=True)
    with c2: st.markdown('<div class="step-box">📷 <b>Step 2</b><br>Open live camera — point directly above</div>', unsafe_allow_html=True)
    with c3: st.markdown('<div class="step-box">🔬 <b>Step 3</b><br>System detects each fruit in real-time</div>', unsafe_allow_html=True)
    with c4: st.markdown('<div class="step-box">💧 <b>Step 4</b><br>Click Predict — get juice yield per fruit</div>', unsafe_allow_html=True)
    with c5: st.markdown('<div class="step-box">🍹 <b>Step 5</b><br>See what you can make with your juice</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# PAGE 2: PREDICT  (User only)
# ══════════════════════════════════════════════════════
elif page == "🔍 Predict Juice Yield":
    st.markdown('<div class="main-title">🔍 Predict Juice Yield</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Spread calamansi flat in a basket — point camera from above</div>', unsafe_allow_html=True)
    st.divider()

    if not model_loaded:
        st.error("❌ Model not found. Please run **train_model.py** first.")
        st.stop()

    st.markdown("""
    **How to use:**
    1. Spread calamansi in a basket so each fruit is visible
    2. Click **Start** to open the live camera
    3. Hold camera directly above the basket
    4. Green boxes will appear on each detected fruit in real-time
    5. When all fruits are clearly visible — click **📸 Predict Now**
    """)

    ctx = webrtc_streamer(
        key="calamansi-live",
        video_processor_factory=CalamansiDetector,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    st.divider()

    if ctx.video_processor:
        if st.button("📸 Predict Now — Capture Current Frame", type="primary"):
            with ctx.video_processor.lock:
                features  = list(ctx.video_processor.result_features)
                annotated = ctx.video_processor.result_annotated

            if not features:
                st.error("❌ No calamansi detected. Make sure fruits are spread flat and well-lit.")
            else:
                predictions = predict_from_features(features)
                total_juice = sum(predictions)

                snapshot_b64 = frame_to_b64(annotated) if annotated is not None else None
                pred_id = save_prediction(
                    username       = username,
                    fruit_count    = len(features),
                    total_juice_ml = total_juice,
                    snapshot_b64   = snapshot_b64,
                )
                st.success(f"✅ {len(features)} calamansi detected! Saved as record #{pred_id}.")

                if annotated is not None:
                    st.image(annotated, caption=f"Captured — {len(features)} fruits detected",
                             use_container_width=True)
                show_results(features, predictions, show_recommendations=True)
    else:
        st.info("👆 Click **Start** above to open the camera.")


# ══════════════════════════════════════════════════════
# PAGE 3: HISTORY
# User  → "🕐 History"        — own records only, own stats, delete own
# Admin → "🕐 View All History" — all users' records, full stats, user filter, delete any
# ══════════════════════════════════════════════════════
elif page in ("🕐 History", "🕐 View All History"):
    if is_admin:
        st.markdown('<div class="main-title">🕐 View All History</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-title">All users\' juice yield prediction sessions</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="main-title">🕐 Prediction History</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-title">Your past juice yield prediction sessions</div>', unsafe_allow_html=True)
    st.divider()

    rows = get_predictions(username, is_admin=is_admin)

    if not rows:
        st.info("No predictions yet. Run a session on the Predict page to see results here.")
        st.stop()

    df = pd.DataFrame(rows, columns=["id", "user", "date", "fruits", "juice_ml"])
    df["date"]     = pd.to_datetime(df["date"])
    df["juice_ml"] = df["juice_ml"].round(1)

    # ── Summary stats ──────────────────────────────────
    if is_admin:
        # Admin: aggregate stats across all users
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Sessions", len(df))
        c2.metric("Unique Users",   df["user"].nunique())
        c3.metric("Total Fruits",   int(df["fruits"].sum()))
        c4.metric("Total Juice",    f"{df['juice_ml'].sum():.1f} mL")
    else:
        # User: only their own stats
        stats = user_stats(username)
        c1, c2, c3 = st.columns(3)
        c1.metric("My Sessions",  stats["sessions"])
        c2.metric("Total Fruits", stats["fruits"])
        c3.metric("Total Juice",  f"{stats['juice_ml']:.1f} mL")

    st.divider()

    # ── Admin-only: filter by user ─────────────────────
    if is_admin:
        all_users = ["All"] + sorted(df["user"].unique().tolist())
        sel_user  = st.selectbox("Filter by user", all_users)
        if sel_user != "All":
            df = df[df["user"] == sel_user]

    # ── Table ──────────────────────────────────────────
    # Admin sees the "user" column; regular users do not
    if is_admin:
        display_cols = ["user", "date", "fruits", "juice_ml"]
        col_labels   = {"user": "User", "date": "Date & Time", "fruits": "Fruits", "juice_ml": "Juice (mL)"}
    else:
        display_cols = ["date", "fruits", "juice_ml"]
        col_labels   = {"date": "Date & Time", "fruits": "Fruits", "juice_ml": "Juice (mL)"}

    st.dataframe(
        df[display_cols].rename(columns=col_labels),
        use_container_width=True,
        hide_index=True,
    )

    # ── Session detail ─────────────────────────────────
    st.divider()
    st.markdown('<div class="section-header">🔍 View Session Detail</div>', unsafe_allow_html=True)

    pred_labels = [
        f"#{r['id']}  {r['date'].strftime('%Y-%m-%d %H:%M')}  —  "
        f"{r['fruits']} fruits  {r['juice_ml']} mL"
        + (f"  [{r['user']}]" if is_admin else "")
        for _, r in df.iterrows()
    ]
    sel_idx = st.selectbox("Select a session", range(len(df)), format_func=lambda i: pred_labels[i])
    sel_row = df.iloc[sel_idx]
    sel_id  = int(sel_row["id"])

    col_img, col_info = st.columns([2, 1])

    with col_img:
        b64 = get_snapshot(sel_id)
        if b64:
            st.image(f"data:image/png;base64,{b64}",
                     caption=f"Session #{sel_id} — detected fruits",
                     use_container_width=True)
        else:
            st.info("No snapshot stored for this session.")

    with col_info:
        st.markdown("**Session Details**")
        if is_admin:
            st.write(f"**User:** {sel_row['user']}")   # Admin sees who owns this record
        st.write(f"**Date:** {sel_row['date'].strftime('%Y-%m-%d %H:%M:%S')}")
        st.write(f"**Fruits detected:** {sel_row['fruits']}")
        st.write(f"**Juice yield:** {sel_row['juice_ml']} mL")

        st.divider()

        # Admin can delete any record; user can only delete their own
        can_delete = is_admin or sel_row["user"] == username
        if can_delete:
            if st.button("🗑️ Delete this record", type="secondary"):
                if delete_prediction(sel_id, username, is_admin):
                    st.success("Record deleted.")
                    st.rerun()
                else:
                    st.error("Could not delete — permission denied.")


# ══════════════════════════════════════════════════════
# PAGE 4: MANAGE USERS  (Admin only)
# ══════════════════════════════════════════════════════
elif page == "👥 Manage Users":
    if not is_admin:
        st.error("❌ Access denied. Admins only.")
        st.stop()

    st.markdown('<div class="main-title">👥 User Management</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Admin can manage system users</div>', unsafe_allow_html=True)
    st.divider()

    users = get_all_users()
    if not users:
        st.info("No users found.")
        st.stop()

    df_users = pd.DataFrame(users, columns=["username", "role"])

    st.markdown("### 📋 All Users")
    st.dataframe(df_users, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("### ➕ Create New User")
    with st.expander("Create a new user account", expanded=False):
        nu_user  = st.text_input("Username",         key="nu_user",  placeholder="Enter username")
        nu_pass  = st.text_input("Password",          type="password", key="nu_pass",  placeholder="Enter password")
        nu_pass2 = st.text_input("Confirm Password",  type="password", key="nu_pass2", placeholder="Repeat password")
        nu_role  = st.selectbox("Role", ["user", "admin"], key="nu_role")
        if st.button("Create User", type="primary", key="btn_create_user"):
            if not nu_user or not nu_pass:
                st.error("❌ Username and password are required.")
            elif nu_pass != nu_pass2:
                st.error("❌ Passwords do not match.")
            elif len(nu_pass) < 6:
                st.error("❌ Password must be at least 6 characters.")
            elif get_user(nu_user):
                st.error("❌ Username already exists.")
            else:
                if create_user(nu_user, hash_pw(nu_pass), role=nu_role):
                    st.success(f"✅ User **{nu_user}** created with role `{nu_role}`.")
                    st.rerun()
                else:
                    st.error("❌ Failed to create user.")

    st.divider()
    st.markdown("### ⚙️ Manage Existing User")
    manageable_users = df_users["username"].tolist()
    selected_user = st.selectbox("Select User", manageable_users)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🔑 Reset Password")
        new_pass = st.text_input("New Password", type="password", key="reset_pass")
        if st.button("Reset Password", type="primary"):
            if selected_user == "admin":
                st.warning("⚠️ Cannot reset main admin password here.")
            elif len(new_pass) < 6:
                st.error("❌ Password must be at least 6 characters.")
            else:
                if reset_password(selected_user, new_pass):
                    st.success(f"✅ Password reset for **{selected_user}**.")
                else:
                    st.error("❌ Failed to reset password.")

    with col2:
        st.markdown("#### 🗑️ Delete User")
        st.warning(f"This will permanently delete **{selected_user}** and cannot be undone.")
        if st.button("Delete User", type="secondary"):
            if selected_user == "admin":
                st.warning("⚠️ Cannot delete the main admin account.")
            elif selected_user == username:
                st.warning("⚠️ Cannot delete your own account while logged in.")
            else:
                if delete_user(selected_user):
                    st.success(f"✅ User **{selected_user}** deleted.")
                    st.rerun()
                else:
                    st.error("❌ Failed to delete user.")


# ══════════════════════════════════════════════════════
# PAGE 5: MODEL PERFORMANCE  (Admin only)
# ══════════════════════════════════════════════════════
elif page == "📊 Model Performance":
    if not is_admin:
        st.error("❌ Access denied. Admins only.")
        st.stop()

    st.markdown('<div class="main-title">📊 Model Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Evaluation metrics of the Linear Regression model</div>', unsafe_allow_html=True)
    st.divider()

    if not model_loaded:
        st.error("❌ Model not found."); st.stop()
    if not os.path.exists('calamansi_dataset.csv'):
        st.error("❌ Dataset not found."); st.stop()

    df     = pd.read_csv('calamansi_dataset.csv')
    X      = df[FEATURE_COLS]; y = df['juice_ml']
    X_sc   = scaler.transform(X)
    y_pred = model.predict(X_sc)
    mae    = mean_absolute_error(y, y_pred)
    rmse   = np.sqrt(mean_squared_error(y, y_pred))
    r2     = r2_score(y, y_pred)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Samples", f"{len(df)}")
    c2.metric("R² Score",      f"{r2:.4f}")
    c3.metric("MAE",           f"{mae:.4f} mL")
    c4.metric("RMSE",          f"{rmse:.4f} mL")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        fig1, ax1 = plt.subplots(figsize=(6, 5))
        ax1.scatter(y, y_pred, color='steelblue', alpha=0.7, edgecolors='white', s=60)
        mn, mx = min(y.min(), y_pred.min())-0.5, max(y.max(), y_pred.max())+0.5
        ax1.plot([mn, mx], [mn, mx], 'r--', lw=2, label='Perfect Prediction')
        ax1.set_xlabel('Actual (mL)'); ax1.set_ylabel('Predicted (mL)')
        ax1.set_title('Actual vs Predicted'); ax1.legend()
        plt.tight_layout(); st.pyplot(fig1)
    with col2:
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        ax2.scatter(y_pred, y-y_pred, color='darkorange', alpha=0.7, edgecolors='white', s=60)
        ax2.axhline(0, color='red', linestyle='--', lw=2)
        ax2.set_xlabel('Predicted (mL)'); ax2.set_ylabel('Residuals (mL)')
        ax2.set_title('Residuals vs Fitted')
        plt.tight_layout(); st.pyplot(fig2)

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        fig3, ax3 = plt.subplots(figsize=(7, 6))
        corr = df[FEATURE_COLS+['juice_ml']].corr()
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                    mask=np.triu(np.ones_like(corr, dtype=bool)),
                    square=True, linewidths=0.5, ax=ax3, cbar_kws={"shrink": 0.8})
        ax3.set_title('Feature Correlation Heatmap')
        plt.tight_layout(); st.pyplot(fig3)
    with col2:
        coef_df = pd.DataFrame({'Feature': FEATURE_COLS, 'Coefficient': model.coef_}).sort_values('Coefficient', ascending=True)
        colors  = ['#d73027' if c < 0 else '#1a9850' for c in coef_df['Coefficient']]
        fig4, ax4 = plt.subplots(figsize=(7, 6))
        ax4.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors, edgecolor='white')
        ax4.axvline(0, color='black', linewidth=0.8)
        ax4.set_title('Feature Coefficients')
        plt.tight_layout(); st.pyplot(fig4)

    st.divider()
    with st.spinner("Running cross validation..."):
        kf     = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_r2  = cross_val_score(model, X_sc, y, cv=kf, scoring='r2')
        cv_mae = -cross_val_score(model, X_sc, y, cv=kf, scoring='neg_mean_absolute_error')
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Mean R² (CV)",  f"{cv_r2.mean():.4f}",  f"± {cv_r2.std():.4f}")
        st.metric("Mean MAE (CV)", f"{cv_mae.mean():.4f} mL", f"± {cv_mae.std():.4f}")
    with col2:
        fig5, ax5 = plt.subplots(figsize=(6, 4))
        ax5.bar([f'Fold {i+1}' for i in range(5)], cv_r2, color='steelblue', edgecolor='white')
        ax5.axhline(cv_r2.mean(), color='red', linestyle='--', lw=2, label=f'Mean={cv_r2.mean():.4f}')
        ax5.set_ylim(0, 1.05); ax5.legend()
        ax5.set_title('5-Fold Cross Validation')
        plt.tight_layout(); st.pyplot(fig5)

    st.divider()
    eq = f"juice_ml = {model.intercept_:.4f}"
    for feat, coef in zip(FEATURE_COLS, model.coef_):
        eq += f" {'+'if coef>=0 else'-'} {abs(coef):.4f} × {feat}"
    st.code(eq, language=None)
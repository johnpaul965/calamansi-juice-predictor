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
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# DATABASE — 5 tables
# ══════════════════════════════════════════════════════
DB_FILE = "users.db"


def init_db():
    conn = sqlite3.connect(DB_FILE)
    c    = conn.cursor()

    # USERS
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id    INTEGER PRIMARY KEY AUTOINCREMENT,
            username   TEXT    NOT NULL UNIQUE,
            email      TEXT    NOT NULL UNIQUE,
            password   TEXT    NOT NULL,
            role       TEXT    NOT NULL DEFAULT 'user',
            created_at TEXT    NOT NULL DEFAULT (datetime('now','localtime'))
        )
    """)

    # PREDICTIONS
    c.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            prediction_id         INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id               INTEGER NOT NULL,
            prediction_time       TEXT    NOT NULL DEFAULT (datetime('now','localtime')),
            fruit_count           INTEGER NOT NULL DEFAULT 0,
            total_diameter_cm     REAL    DEFAULT 0.0,
            total_area_cm2        REAL    DEFAULT 0.0,
            avg_hue               REAL    DEFAULT 0.0,
            ripeness_class        TEXT,
            predicted_juice_ml    REAL    NOT NULL DEFAULT 0.0,
            recipe_recommendation TEXT,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        )
    """)

    # IMAGES
    c.execute("""
        CREATE TABLE IF NOT EXISTS images (
            image_id       INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_id  INTEGER NOT NULL,
            image_filename TEXT,
            image_width    INTEGER,
            image_height   INTEGER,
            snapshot_b64   TEXT,
            capture_time   TEXT NOT NULL DEFAULT (datetime('now','localtime')),
            FOREIGN KEY (prediction_id) REFERENCES predictions(prediction_id)
        )
    """)

    # CALAMANSI_FRUIT
    c.execute("""
        CREATE TABLE IF NOT EXISTS calamansi_fruit (
            fruit_id            INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id            INTEGER NOT NULL,
            diameter_cm         REAL,
            area_cm2            REAL,
            hue_value           REAL,
            ripeness            TEXT,
            individual_juice_ml REAL,
            FOREIGN KEY (image_id) REFERENCES images(image_id)
        )
    """)

    # MODELS
    c.execute("""
        CREATE TABLE IF NOT EXISTS models (
            model_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT,
            version    TEXT,
            r2_score   REAL,
            mae        REAL,
            rmse       REAL,
            trained_at TEXT,
            status     TEXT DEFAULT 'active' CHECK(status IN ('active','archived'))
        )
    """)

    c.execute("CREATE INDEX IF NOT EXISTS idx_pred_user ON predictions (user_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_pred_time ON predictions (prediction_time DESC)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_fruit_img ON calamansi_fruit (image_id)")

    # Default admin
    admin_pass = hashlib.sha256("admin2024".encode()).hexdigest()
    c.execute("""
        INSERT OR IGNORE INTO users (username, email, password, role)
        VALUES (?, ?, ?, ?)
    """, ("admin", "admin@calamansi.sys", admin_pass, "admin"))

    conn.commit()
    conn.close()


# ── User helpers ──────────────────────────────────────

def get_user(username):
    conn = sqlite3.connect(DB_FILE)
    c    = conn.cursor()
    c.execute("SELECT user_id, username, email, password, role FROM users WHERE username=?", (username,))
    row  = c.fetchone()
    conn.close()
    if row:
        return {"user_id": row[0], "username": row[1], "email": row[2],
                "password": row[3], "role": row[4]}
    return None


def get_user_by_email(email):
    conn = sqlite3.connect(DB_FILE)
    c    = conn.cursor()
    c.execute("SELECT username FROM users WHERE email=?", (email,))
    row  = c.fetchone()
    conn.close()
    return row[0] if row else None


def create_user(username, email, password, role="user"):
    try:
        conn = sqlite3.connect(DB_FILE)
        c    = conn.cursor()
        c.execute("INSERT INTO users (username,email,password,role) VALUES (?,?,?,?)",
                  (username, email, password, role))
        conn.commit(); conn.close()
        return True
    except sqlite3.IntegrityError:
        return False


def hash_pw(pw):
    return hashlib.sha256(pw.encode()).hexdigest()


# ── Prediction helpers ────────────────────────────────

def save_full_prediction(user_id, features, predictions, annotated_rgb,
                         ripeness_label, recipe_names):
    total_juice = sum(predictions)
    total_diam  = sum(f['diameter_cm'] for f in features)
    total_area  = sum(f['area_cm2']    for f in features)
    avg_hue     = float(np.mean([f['mean_hue'] for f in features])) if features else 0.0
    recipe_str  = ", ".join(recipe_names[:3]) if recipe_names else ""

    conn = sqlite3.connect(DB_FILE)
    c    = conn.cursor()

    # predictions row
    c.execute("""
        INSERT INTO predictions
            (user_id, fruit_count, total_diameter_cm, total_area_cm2,
             avg_hue, ripeness_class, predicted_juice_ml, recipe_recommendation)
        VALUES (?,?,?,?,?,?,?,?)
    """, (user_id, len(features), round(total_diam,4), round(total_area,4),
          round(avg_hue,4), ripeness_label, round(total_juice,2), recipe_str))
    prediction_id = c.lastrowid

    # images row
    snapshot_b64 = frame_to_b64(annotated_rgb) if annotated_rgb is not None else None
    h, w         = (annotated_rgb.shape[:2] if annotated_rgb is not None else (0,0))
    filename     = f"pred_{prediction_id}.png"
    c.execute("""
        INSERT INTO images (prediction_id, image_filename, image_width, image_height, snapshot_b64)
        VALUES (?,?,?,?,?)
    """, (prediction_id, filename, w, h, snapshot_b64))
    image_id = c.lastrowid

    # calamansi_fruit rows
    for f, juice in zip(features, predictions):
        hue = f['mean_hue']
        rip = "Ripe" if hue < 25 else ("Turning" if hue < 50 else "Green")
        c.execute("""
            INSERT INTO calamansi_fruit
                (image_id, diameter_cm, area_cm2, hue_value, ripeness, individual_juice_ml)
            VALUES (?,?,?,?,?,?)
        """, (image_id, round(f['diameter_cm'],4), round(f['area_cm2'],4),
              round(hue,4), rip, round(juice,2)))

    conn.commit(); conn.close()
    return prediction_id


def save_model_record(model_name, version, r2, mae, rmse):
    conn = sqlite3.connect(DB_FILE)
    c    = conn.cursor()
    c.execute("UPDATE models SET status='archived' WHERE status='active'")
    c.execute("""
        INSERT INTO models (model_name, version, r2_score, mae, rmse, trained_at, status)
        VALUES (?,?,?,?,?,datetime('now','localtime'),'active')
    """, (model_name, version, round(r2,6), round(mae,6), round(rmse,6)))
    conn.commit(); conn.close()


def get_predictions(user_id, is_admin=False, limit=200):
    conn = sqlite3.connect(DB_FILE)
    c    = conn.cursor()
    if is_admin:
        c.execute("""
            SELECT p.prediction_id, u.username, p.prediction_time,
                   p.fruit_count, p.predicted_juice_ml, p.ripeness_class
            FROM predictions p JOIN users u ON p.user_id=u.user_id
            ORDER BY p.prediction_time DESC LIMIT ?
        """, (limit,))
    else:
        c.execute("""
            SELECT p.prediction_id, u.username, p.prediction_time,
                   p.fruit_count, p.predicted_juice_ml, p.ripeness_class
            FROM predictions p JOIN users u ON p.user_id=u.user_id
            WHERE p.user_id=?
            ORDER BY p.prediction_time DESC LIMIT ?
        """, (user_id, limit))
    rows = c.fetchall(); conn.close()
    return rows


def get_snapshot_by_prediction(prediction_id):
    conn = sqlite3.connect(DB_FILE)
    c    = conn.cursor()
    c.execute("SELECT snapshot_b64 FROM images WHERE prediction_id=?", (prediction_id,))
    row = c.fetchone(); conn.close()
    return row[0] if row else None


def get_fruits_by_prediction(prediction_id):
    conn = sqlite3.connect(DB_FILE)
    c    = conn.cursor()
    c.execute("""
        SELECT cf.fruit_id, cf.diameter_cm, cf.area_cm2,
               cf.hue_value, cf.ripeness, cf.individual_juice_ml
        FROM calamansi_fruit cf
        JOIN images i ON cf.image_id=i.image_id
        WHERE i.prediction_id=?
    """, (prediction_id,))
    rows = c.fetchall(); conn.close()
    return rows


def delete_prediction(prediction_id, user_id, is_admin=False):
    conn = sqlite3.connect(DB_FILE)
    c    = conn.cursor()
    c.execute("DELETE FROM calamansi_fruit WHERE image_id IN (SELECT image_id FROM images WHERE prediction_id=?)", (prediction_id,))
    c.execute("DELETE FROM images WHERE prediction_id=?", (prediction_id,))
    if is_admin:
        c.execute("DELETE FROM predictions WHERE prediction_id=?", (prediction_id,))
    else:
        c.execute("DELETE FROM predictions WHERE prediction_id=? AND user_id=?", (prediction_id, user_id))
    conn.commit()
    deleted = c.rowcount > 0
    conn.close()
    return deleted


def user_stats(user_id):
    conn = sqlite3.connect(DB_FILE)
    c    = conn.cursor()
    c.execute("""
        SELECT COUNT(*), COALESCE(SUM(fruit_count),0), COALESCE(SUM(predicted_juice_ml),0)
        FROM predictions WHERE user_id=?
    """, (user_id,))
    row = c.fetchone(); conn.close()
    return {"sessions": row[0], "fruits": int(row[1]), "juice_ml": row[2]}


def frame_to_b64(rgb_array, max_side=640):
    img = Image.fromarray(rgb_array.astype("uint8"))
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


init_db()


# ══════════════════════════════════════════════════════
# MODEL LOAD + REGISTER METRICS
# ══════════════════════════════════════════════════════
@st.cache_resource
def load_model():
    return joblib.load('juice_yield_model.pkl'), joblib.load('scaler.pkl')

model_loaded = os.path.exists('juice_yield_model.pkl') and os.path.exists('scaler.pkl')
if model_loaded:
    model, scaler = load_model()
    if os.path.exists('calamansi_dataset.csv'):
        try:
            _df    = pd.read_csv('calamansi_dataset.csv')
            _X     = scaler.transform(_df[FEATURE_COLS])
            _y     = _df['juice_ml']
            _yp    = model.predict(_X)
            save_model_record("juice_yield_model", "1.0",
                              r2_score(_y,_yp),
                              mean_absolute_error(_y,_yp),
                              float(np.sqrt(mean_squared_error(_y,_yp))))
        except Exception:
            pass


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
                         type="primary" if st.session_state.auth_mode=="user" else "secondary"):
                st.session_state.auth_mode = "user"; st.rerun()
        with c2:
            if st.button("🔧 Admin Login", use_container_width=True,
                         type="primary" if st.session_state.auth_mode=="admin" else "secondary"):
                st.session_state.auth_mode = "admin"; st.rerun()

        st.markdown("---")

        if st.session_state.auth_mode == "user":
            tab1, tab2 = st.tabs(["🔐 Login", "📝 Register"])

            with tab1:
                st.markdown("#### User Login")
                username = st.text_input("Username", key="login_user", placeholder="Enter username")
                password = st.text_input("Password", type="password", key="login_pass")
                if st.button("Login", type="primary", use_container_width=True, key="btn_login"):
                    user = get_user(username)
                    if user and user["password"]==hash_pw(password) and user["role"]=="user":
                        st.session_state.logged_in = True
                        st.session_state.username  = username
                        st.session_state.user_id   = user["user_id"]
                        st.session_state.role      = "user"
                        st.rerun()
                    elif user and user["role"]=="admin":
                        st.error("❌ Use Admin Login for admin accounts.")
                    else:
                        st.error("❌ Invalid username or password.")

            with tab2:
                st.markdown("#### Create a New Account")
                new_user  = st.text_input("Username",         key="reg_user")
                new_email = st.text_input("Email",            key="reg_email", placeholder="your@email.com")
                new_pass  = st.text_input("Password",         type="password", key="reg_pass")
                new_pass2 = st.text_input("Confirm Password", type="password", key="reg_pass2")
                if st.button("Register", type="primary", use_container_width=True, key="btn_register"):
                    if not new_user or not new_email or not new_pass:
                        st.error("❌ All fields are required.")
                    elif "@" not in new_email:
                        st.error("❌ Enter a valid email address.")
                    elif new_pass != new_pass2:
                        st.error("❌ Passwords do not match.")
                    elif len(new_pass) < 6:
                        st.error("❌ Password must be at least 6 characters.")
                    elif get_user(new_user):
                        st.error("❌ Username already exists.")
                    elif get_user_by_email(new_email):
                        st.error("❌ Email already registered.")
                    else:
                        if create_user(new_user, new_email, hash_pw(new_pass)):
                            st.success(f"✅ Account created! You can now login as **{new_user}**.")
                        else:
                            st.error("❌ Registration failed.")
        else:
            st.markdown("#### Admin Login")
            st.info("🔧 This login is for system administrators only.")
            admin_user = st.text_input("Admin Username", key="admin_user")
            admin_pass = st.text_input("Admin Password", type="password", key="admin_pass")
            if st.button("Admin Login", type="primary", use_container_width=True, key="btn_admin"):
                user = get_user(admin_user)
                if user and user["password"]==hash_pw(admin_pass) and user["role"]=="admin":
                    st.session_state.logged_in = True
                    st.session_state.username  = admin_user
                    st.session_state.user_id   = user["user_id"]
                    st.session_state.role      = "admin"
                    st.rerun()
                else:
                    st.error("❌ Invalid admin credentials.")


if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    show_auth(); st.stop()


# ══════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════
role     = st.session_state.get("role", "guest")
username = st.session_state.get("username", "Guest")
user_id  = st.session_state.get("user_id", None)
is_admin = role == "admin"

with st.sidebar:
    st.markdown("## Navigation")
    pages = (["🏠 Home", "🔍 Predict Juice Yield", "🕐 History", "📊 Model Performance"]
             if is_admin else
             ["🏠 Home", "🔍 Predict Juice Yield", "🕐 History"])
    page = st.radio("Go to", pages)
    st.markdown("---")
    st.markdown(f"👤 **{username}**")
    st.markdown(f"🔑 Role: `{role}`")
    st.markdown(f"**Model:** {'✅ Loaded' if model_loaded else '❌ Not found'}")
    st.markdown("---")
    if st.button("🚪 Logout"):
        for k in ["logged_in","username","user_id","role"]:
            st.session_state[k] = False if k=="logged_in" else ("" if k!="user_id" else None)
        st.rerun()

RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})


# ══════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════

def predict_from_features(all_features):
    return [round(max(2.0, 5.30*(f['diameter_cm']/2.5)), 2) for f in all_features]


def get_ripeness(all_features):
    if not all_features:
        return "🟢 Green Stage", "#16a34a", "Dark green — unripe."
    avg_hue = np.mean([f['mean_hue'] for f in all_features])
    if avg_hue < 25:   return "🟠 Ripe Stage",    "#f97316", "Yellow-orange — fully ripe."
    elif avg_hue < 50: return "🟡 Turning Stage", "#eab308", "Partially yellow — between green and ripe."
    else:              return "🟢 Green Stage",   "#16a34a", "Dark green — unripe calamansi."


def detect_on_frame(img_rgb):
    img_blur = preprocess_array(img_rgb)
    mask, hsv = segment_fruit(img_blur)
    circles   = count_hough(img_blur, mask)
    ppc       = calibrate_ppc(circles) if circles else 41.0
    all_features, valid_cnts = get_features_from_hough(img_blur, circles, ppc)
    if not all_features:
        all_features, valid_cnts = get_fruit_features(img_blur, mask, hsv, ppc)
    annotated = img_blur.copy()
    for i, cnt in enumerate(valid_cnts):
        x,y,w,h = cv2.boundingRect(cnt)
        p=6; x=max(0,x-p); y=max(0,y-p)
        w=min(img_blur.shape[1]-x,w+2*p); h=min(img_blur.shape[0]-y,h+2*p)
        cv2.rectangle(annotated,(x,y),(x+w,y+h),(22,163,74),2)
        cv2.putText(annotated,f"#{i+1}",(x+4,y+20),cv2.FONT_HERSHEY_SIMPLEX,0.55,(22,163,74),2)
    return all_features, valid_cnts, annotated


def show_juice_recommendations(total_ml, ripeness_label):
    st.markdown('<div class="section-header">🍹 Juice Usage Recommendations</div>', unsafe_allow_html=True)
    st.markdown(f"**You have {total_ml:.0f} mL of calamansi juice. Here is what you can make:**")
    recipes = [
        {"name":"🍹 Calamansi Juice Drink","ml":60, "unit":"glass",  "desc":"Mix with water + sugar (1:3 ratio)"},
        {"name":"🍵 Calamansi Tea",         "ml":30, "unit":"cup",   "desc":"Mix with hot water + honey"},
        {"name":"🍗 Chicken Marinade",       "ml":50, "unit":"serving","desc":"Enough for 250g chicken"},
        {"name":"🥗 Salad Dressing",        "ml":20, "unit":"serving","desc":"Mix with olive oil + salt"},
        {"name":"💊 Vitamin C Drink",       "ml":15, "unit":"shot",  "desc":"Pure calamansi shot"},
        {"name":"🫙 Calamansi Vinegar",     "ml":200,"unit":"bottle","desc":"Fermented calamansi vinegar"},
        {"name":"🧴 Calamansi Concentrate", "ml":100,"unit":"jar",   "desc":"Boil down with sugar for syrup"},
        {"name":"🍰 Calamansi Cake/Pastry", "ml":45, "unit":"recipe","desc":"For baking calamansi-flavored goods"},
    ]
    tip = ("🟠 Your fruits are **ripe** — perfect for juice drinks, tea, and desserts." if "Ripe" in ripeness_label
           else "🟡 Your fruits are **partially ripe** — good for both drinks and cooking." if "Turning" in ripeness_label
           else "🟢 Your fruits are **unripe** — more acidic, best for marinades, vinegar, and cooking.")
    st.info(tip)
    cols = st.columns(2)
    for i, r in enumerate(recipes):
        qty = int(total_ml // r["ml"])
        with cols[i%2]:
            status = (f'<span class="recipe-ok">✅ You can make {qty}</span>' if qty>=1
                      else f'<span class="recipe-no">❌ Need {r["ml"]-total_ml:.0f} mL more</span>')
            st.markdown(f'<div class="recipe-box"><b>{r["name"]}</b><br>'
                        f'<small style="color:#6b7280;">{r["desc"]}</small><br>'
                        f'<small>Needs <b>{r["ml"]} mL</b> per {r["unit"]}</small><br>{status}</div>',
                        unsafe_allow_html=True)
    st.markdown("---")
    makeable   = [r for r in recipes if total_ml >= r["ml"]]
    not_enough = [r for r in recipes if total_ml <  r["ml"]]
    if makeable:   st.success("With "+f"{total_ml:.0f} mL you can make: "+", ".join(r['name'] for r in makeable))
    if not_enough: st.warning("Not enough juice for: "+", ".join(r['name'] for r in not_enough))
    return [r["name"] for r in makeable]


def show_results(all_features, predictions, show_recommendations=False):
    fruit_count = len(predictions)
    total_juice = sum(predictions)
    avg_juice   = total_juice/fruit_count if fruit_count>0 else 0
    label, color, desc = get_ripeness(all_features)

    st.markdown(f'<div class="detect-box">'
                f'<span style="font-size:22px;font-weight:bold;color:#166534;">🍋 Calamansi Detected</span>'
                f'&nbsp;<span style="font-size:14px;color:#6b7280;">Citrus microcarpa</span><br><br>'
                f'<span style="font-size:16px;font-weight:bold;color:{color};">{label}</span>'
                f'&nbsp;—&nbsp;<span style="font-size:14px;color:#6b7280;">{desc}</span></div>',
                unsafe_allow_html=True)

    c1,c2,c3 = st.columns(3)
    c1.metric("🍊 Fruits Detected",    f"{fruit_count}")
    c2.metric("📊 Avg Juice per Fruit", f"{avg_juice:.2f} mL")
    c3.metric("💧 Total Juice Yield",   f"{total_juice:.2f} mL")

    st.markdown(f'<div class="total-box">'
                f'<span style="font-size:16px;color:#bbf7d0;">🍋 {fruit_count} Calamansi Fruits Detected</span><br>'
                f'<span style="font-size:40px;font-weight:bold;color:#ffffff;">💧 {total_juice:.2f} mL</span><br>'
                f'<span style="font-size:14px;color:#bbf7d0;">Estimated Total Juice Yield</span></div>',
                unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="section-header">🍋 Per Fruit Breakdown</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({
        "Fruit":               [f"#{i+1}" for i in range(fruit_count)],
        "Predicted Juice (mL)":[f"{p:.2f}" for p in predictions],
        "Diameter (cm)":       [f"{all_features[i]['diameter_cm']:.2f}" for i in range(fruit_count)],
        "Area (cm²)":          [f"{all_features[i]['area_cm2']:.2f}"    for i in range(fruit_count)],
        "Ripeness":            ["🟠 Ripe" if all_features[i]['mean_hue']<25
                                else "🟡 Turning" if all_features[i]['mean_hue']<50
                                else "🟢 Green" for i in range(fruit_count)],
    }), use_container_width=True, hide_index=True)

    st.divider()
    fig, ax = plt.subplots(figsize=(max(6, fruit_count*0.7), 4))
    bars = ax.bar([f"#{i+1}" for i in range(fruit_count)], predictions, color='#16a34a', edgecolor='white')
    ax.axhline(avg_juice, color='red', linestyle='--', lw=1.5, label=f'Avg = {avg_juice:.2f} mL')
    ax.set_xlabel('Fruit'); ax.set_ylabel('Predicted Juice (mL)')
    ax.set_title(f'{fruit_count} fruits | Total: {total_juice:.2f} mL'); ax.legend()
    for bar, val in zip(bars, predictions):
        ax.text(bar.get_x()+bar.get_width()/2, val+0.05, f'{val:.2f}', ha='center', fontsize=9)
    plt.tight_layout(); st.pyplot(fig)

    recipe_names = []
    if show_recommendations:
        st.divider()
        recipe_names = show_juice_recommendations(total_juice, label)
    else:
        st.divider()
    return label, recipe_names


# ══════════════════════════════════════════════════════
# LIVE CAMERA
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
        cv2.putText(out_bgr,f"Detected: {count}",(10,35),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,0),3)
        cv2.putText(out_bgr,f"Detected: {count}",(10,35),cv2.FONT_HERSHEY_SIMPLEX,1.0,(22,163,74),2)
        return av.VideoFrame.from_ndarray(out_bgr, format="bgr24")


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
        st.write("This system predicts the **juice yield (mL)** of Calamansi fruits *(Citrus microcarpa)* using live camera detection and linear regression.")
        st.markdown('<div class="section-header">🎯 Objectives</div>', unsafe_allow_html=True)
        for i, o in enumerate([
            "Collect and process calamansi fruit images for model training.",
            "Extract image-based features such as size, shape, and color.",
            "Train a Linear Regression model to predict juice yield per fruit.",
            "Implement real-time calamansi detection using color, shape, and size filtering.",
            "Deploy as a web-based system outputting detected count, total juice yield, and usage recommendations."
        ], 1): st.write(f"**{i}.** {o}")
    with col2:
        st.markdown('<div class="section-header">📐 System Scope</div>', unsafe_allow_html=True)
        st.markdown("| Item | Detail |\n|------|--------|\n| Fruit | Calamansi *(Citrus microcarpa)* |\n| Input | Live Camera |\n| Setup | Spread flat in basket |\n| Output | Count + juice + recommendations |\n| Model | Linear Regression |\n| Features | Shape + Color |")
        st.markdown('<div class="section-header">🔑 Access Levels</div>', unsafe_allow_html=True)
        st.markdown("| Role | Access |\n|------|--------|\n| 👥 User | Home + Predict + History + Recommendations |\n| 🔧 Admin | All pages + All users history |")
    st.divider()
    st.markdown('<div class="section-header">🍋 What is Calamansi?</div>', unsafe_allow_html=True)
    st.markdown("| Stage | Color | Best Use |\n|-------|-------|----------|\n| 🟢 Green | Dark green | Marinades, vinegar, cooking |\n| 🟡 Turning | Yellow-green | Mixed use |\n| 🟠 Ripe | Yellow-orange | Juice drinks, tea, desserts |")
    st.divider()
    st.markdown('<div class="section-header">⚙️ How the System Works</div>', unsafe_allow_html=True)
    cols = st.columns(5)
    steps = [("🧺","Step 1","Spread calamansi flat in a basket"),
             ("📷","Step 2","Open live camera — point directly above"),
             ("🔬","Step 3","System detects each fruit in real-time"),
             ("💧","Step 4","Click Predict — get juice yield per fruit"),
             ("🍹","Step 5","See what you can make with your juice")]
    for col, (icon, step, desc) in zip(cols, steps):
        with col:
            st.markdown(f'<div class="step-box">{icon} <b>{step}</b><br>{desc}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# PAGE 2: PREDICT
# ══════════════════════════════════════════════════════
elif page == "🔍 Predict Juice Yield":
    st.markdown('<div class="main-title">🔍 Predict Juice Yield</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Spread calamansi flat in a basket — point camera from above</div>', unsafe_allow_html=True)
    st.divider()

    if not model_loaded:
        st.error("❌ Model not found. Please run **train_model.py** first."); st.stop()

    st.markdown("**How to use:**\n1. Spread calamansi in a basket\n2. Click **Start** to open the live camera\n3. Hold camera directly above\n4. Green boxes appear on detected fruits\n5. Click **📸 Predict Now** when all fruits are visible")

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
                if annotated is not None:
                    st.image(annotated, caption=f"Captured — {len(features)} fruits detected",
                             use_container_width=True)
                ripeness_label, recipe_names = show_results(features, predictions, show_recommendations=True)

                # Save to all 3 tables
                pred_id = save_full_prediction(
                    user_id=user_id, features=features, predictions=predictions,
                    annotated_rgb=annotated, ripeness_label=ripeness_label,
                    recipe_names=recipe_names)
                st.success(f"✅ {len(features)} fruits detected! Saved as prediction #{pred_id}.")
    else:
        st.info("👆 Click **Start** above to open the camera.")


# ══════════════════════════════════════════════════════
# PAGE 3: HISTORY
# ══════════════════════════════════════════════════════
elif page == "🕐 History":
    st.markdown('<div class="main-title">🕐 Prediction History</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Past juice yield prediction sessions</div>', unsafe_allow_html=True)
    st.divider()

    rows = get_predictions(user_id, is_admin=is_admin)
    if not rows:
        st.info("No predictions yet. Run a session on the Predict page to see results here."); st.stop()

    df = pd.DataFrame(rows, columns=["id","user","date","fruits","juice_ml","ripeness"])
    df["date"]     = pd.to_datetime(df["date"])
    df["juice_ml"] = df["juice_ml"].round(1)

    if is_admin:
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Total Sessions", len(df))
        c2.metric("Unique Users",   df["user"].nunique())
        c3.metric("Total Fruits",   int(df["fruits"].sum()))
        c4.metric("Total Juice",    f"{df['juice_ml'].sum():.1f} mL")
    else:
        stats = user_stats(user_id)
        c1,c2,c3 = st.columns(3)
        c1.metric("My Sessions",  stats["sessions"])
        c2.metric("Total Fruits", stats["fruits"])
        c3.metric("Total Juice",  f"{stats['juice_ml']:.1f} mL")

    st.divider()

    if is_admin:
        sel_user = st.selectbox("Filter by user", ["All"]+sorted(df["user"].unique().tolist()))
        if sel_user != "All":
            df = df[df["user"]==sel_user]

    display_cols = ["user","date","fruits","juice_ml","ripeness"] if is_admin else ["date","fruits","juice_ml","ripeness"]
    col_labels   = {"user":"User","date":"Date & Time","fruits":"Fruits","juice_ml":"Juice (mL)","ripeness":"Ripeness"}
    st.dataframe(df[display_cols].rename(columns=col_labels), use_container_width=True, hide_index=True)

    st.divider()
    st.markdown('<div class="section-header">🔍 View Session Detail</div>', unsafe_allow_html=True)
    pred_labels = [
        f"#{r['id']}  {r['date'].strftime('%Y-%m-%d %H:%M')}  —  {r['fruits']} fruits  {r['juice_ml']} mL"
        +(f"  [{r['user']}]" if is_admin else "")
        for _,r in df.iterrows()
    ]
    sel_idx = st.selectbox("Select a session", range(len(df)), format_func=lambda i: pred_labels[i])
    sel_row = df.iloc[sel_idx]
    sel_id  = int(sel_row["id"])

    col_img, col_info = st.columns([2,1])
    with col_img:
        b64 = get_snapshot_by_prediction(sel_id)
        if b64:
            st.image(f"data:image/png;base64,{b64}", caption=f"Session #{sel_id}", use_container_width=True)
        else:
            st.info("No snapshot stored for this session.")
        fruit_rows = get_fruits_by_prediction(sel_id)
        if fruit_rows:
            st.markdown("**Per-fruit details**")
            st.dataframe(pd.DataFrame(fruit_rows,
                columns=["ID","Diameter (cm)","Area (cm²)","Hue","Ripeness","Juice (mL)"]),
                use_container_width=True, hide_index=True)
    with col_info:
        st.markdown("**Session Details**")
        if is_admin: st.write(f"**User:** {sel_row['user']}")
        st.write(f"**Date:** {sel_row['date'].strftime('%Y-%m-%d %H:%M:%S')}")
        st.write(f"**Fruits detected:** {sel_row['fruits']}")
        st.write(f"**Juice yield:** {sel_row['juice_ml']} mL")
        st.write(f"**Ripeness:** {sel_row['ripeness']}")
        st.divider()
        if is_admin or sel_row["user"]==username:
            if st.button("🗑️ Delete this record", type="secondary"):
                if delete_prediction(sel_id, user_id, is_admin):
                    st.success("Record deleted."); st.rerun()
                else:
                    st.error("Could not delete.")

    st.divider()
    csv = df[display_cols].rename(columns=col_labels).to_csv(index=False)
    st.download_button("⬇️ Download CSV", data=csv,
                       file_name=f"history_{username}.csv", mime="text/csv")


# ══════════════════════════════════════════════════════
# PAGE 4: MODEL PERFORMANCE (admin only)
# ══════════════════════════════════════════════════════
elif page == "📊 Model Performance":
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
    rmse   = float(np.sqrt(mean_squared_error(y, y_pred)))
    r2     = r2_score(y, y_pred)

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
        fig5, ax5 = plt.subplots(figsize=(6,4))
        ax5.bar([f'Fold {i+1}' for i in range(5)],cv_r2,color='steelblue',edgecolor='white')
        ax5.axhline(cv_r2.mean(),color='red',linestyle='--',lw=2,label=f'Mean={cv_r2.mean():.4f}')
        ax5.set_ylim(0,1.05); ax5.legend()
        ax5.set_title('5-Fold Cross Validation')
        plt.tight_layout(); st.pyplot(fig5)

    st.divider()
    eq = f"juice_ml = {model.intercept_:.4f}"
    for feat,coef in zip(FEATURE_COLS,model.coef_):
        eq += f" {'+'if coef>=0 else'-'} {abs(coef):.4f} × {feat}"
    st.code(eq, language=None)
<<<<<<< HEAD
# 🍋 Calamansi Juice Yield Prediction System

An image-based system for predicting Calamansi (*Citrus microcarpa*) juice yield using Linear Regression.

---

## 📁 Project Structure

```
├── images/                  ← put your calamansi photos here
├── ground_truth.csv         ← fruit_id and juice_ml measurements
├── requirements.txt         ← all required libraries
├── rename_images.py         ← auto-rename photos to CAL_001.jpg format
├── feature_extraction.py    ← image processing module (shared)
├── train_model.py           ← trains and saves the model
├── evaluate_model.py        ← generates evaluation graphs
└── app.py                   ← Streamlit web application (3 pages)
```

---

## ⚙️ Installation

**1. Make sure Python is installed**
Download at https://www.python.org/downloads/

**2. Clone this repository**
```bash
git clone https://github.com/YOUR_USERNAME/calamansi-juice-predictor.git
cd calamansi-juice-predictor
```

**3. Install all required libraries**
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Step 1 — Add your images
Place all calamansi photos inside the `images/` folder.

### Step 2 — Rename images
```bash
python rename_images.py
```
This renames all photos to `CAL_001.jpg`, `CAL_002.jpg`, etc.

### Step 3 — Prepare ground_truth.csv
Create a CSV file with this format:
```
fruit_id, juice_ml
CAL_001, 7.2
CAL_002, 9.5
CAL_003, 8.1
```
- `fruit_id` — matches the photo filename (without .jpg)
- `juice_ml` — actual juice extracted from that fruit in mL

### Step 4 — Train the model
```bash
python train_model.py
```
This reads your images and CSV, extracts features, trains the Linear Regression model, and saves `juice_yield_model.pkl` and `scaler.pkl`.

### Step 5 — (Optional) Generate evaluation graphs
```bash
python evaluate_model.py
```
Saves graphs to `evaluation_results/` folder for use in your paper.

### Step 6 — Run the web app
```bash
streamlit run app.py
```
Opens the system in your browser automatically.

---

## 📊 Web App Pages

| Page | Description |
|------|-------------|
| 🏠 Home | System overview, objectives, how it works |
| 🔍 Predict Juice Yield | Upload image → get predicted juice yield in mL |
| 📊 Model Performance | R², MAE, RMSE, graphs, cross validation |

---

## 🔬 Features Extracted from Images

**Shape:** Area, Diameter, Perimeter, Circularity, Estimated Volume

**Color:** Mean Hue, Mean Saturation, Mean Brightness

---

## 📦 Requirements

```
streamlit
opencv-python
numpy
pandas
scikit-learn
matplotlib
seaborn
pillow
joblib
```

---

## 📌 Notes

- Minimum **100–200 fruit samples** recommended for good model accuracy
- Use consistent lighting and white background when taking photos
- All images must be JPG or PNG format
- The `images/` folder and `ground_truth.csv` are **not included** in this repository — you must provide your own data
=======
# calamansi-juice-predictor
>>>>>>> d6131087365df65516bf723ebb43d190bddec8b1

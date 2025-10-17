# train_model_v3.py
import pandas as pd
import numpy as np
import joblib
from scipy.sparse import hstack, csr_matrix
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import warnings

warnings.filterwarnings("ignore")
print("🚀 Training multimodal model...")

# ---------- Load data ----------
train = pd.read_csv("student_resource/dataset/train.csv")
train['catalog_content'] = train['catalog_content'].fillna("")

# ---------- Numeric features ----------
def extract_features(text):
    value_match = re.search(r"(\d+(\.\d+)?)", text)
    value = float(value_match.group(1)) if value_match else 0.0

    pack_match = re.search(r"[Pp]ack(?:\s*of|\s*[:\-])?\s*(\d+)", text)
    pack = int(pack_match.group(1)) if pack_match else 1

    units = ['ounce', 'oz', 'pound', 'lb', 'g', 'gram', 'kg', 'count', 
             'pack', 'fl oz', 'ml', 'liter', 'l']
    unit_found = None
    for u in units:
        if re.search(rf"\b{u}\b", text.lower()):
            unit_found = u
            break

    return pd.Series([value, unit_found, pack])

train[['Value', 'Unit', 'Pack']] = train['catalog_content'].apply(extract_features)
train['Unit'] = train['Unit'].astype('category').cat.codes
train = train.fillna(0)

# ---------- Split ----------
X_text = train['catalog_content']
y = train['price']
X_train_text, X_val_text, y_train, y_val = train_test_split(
    X_text, y, test_size=0.2, random_state=42
)
train_idx = X_train_text.index
val_idx = X_val_text.index

# ---------- TF-IDF ----------
print("⏳ Creating TF-IDF features...")
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = tfidf.fit_transform(X_train_text)
X_val_tfidf = tfidf.transform(X_val_text)
print("✅ TF-IDF complete.")

# ---------- Numeric ----------
num_train = csr_matrix(train.loc[train_idx, ['Value', 'Unit', 'Pack']].values)
num_val = csr_matrix(train.loc[val_idx, ['Value', 'Unit', 'Pack']].values)

# ---------- Image features ----------
try:
    ids, img_features = joblib.load("image_features.pkl")
    img_df = pd.DataFrame(img_features, index=ids)
    img_df = img_df.reindex(train['sample_id'], fill_value=0)
    img_train = csr_matrix(img_df.loc[train_idx].values)
    img_val = csr_matrix(img_df.loc[val_idx].values)
    print("🖼️ Image features loaded and merged.")
except Exception as e:
    print("⚠️ Could not load image features, proceeding without them:", e)
    img_train = csr_matrix((len(train_idx), 0))
    img_val = csr_matrix((len(val_idx), 0))

# ---------- Combine ----------
X_train_all = hstack([X_train_tfidf, num_train, img_train])
X_val_all = hstack([X_val_tfidf, num_val, img_val])

# ---------- Train ----------
print("🚀 Training LightGBM model...")
model = lgb.LGBMRegressor(
    n_estimators=600,
    learning_rate=0.05,
    num_leaves=128,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train_all, y_train)

# ---------- Evaluate ----------
def smape(y_true, y_pred):
    return 100/len(y_true) * np.sum(np.abs(y_pred - y_true) / ((np.abs(y_true) + np.abs(y_pred)) / 2))

preds = model.predict(X_val_all)
mae = np.mean(np.abs(y_val - preds))
print(f"✅ Validation MAE: {mae:.4f}")
print(f"📊 Validation SMAPE: {smape(y_val, preds):.2f}%")

# ---------- Save ----------
joblib.dump(model, "model_v3_multimodal.pkl")
joblib.dump(tfidf, "tfidf_v3.pkl")
print("🎯 Multimodal model training complete and saved successfully!")

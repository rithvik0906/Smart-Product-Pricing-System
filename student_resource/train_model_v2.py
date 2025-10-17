# train_model_v2.py
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
import joblib

print("🚀 Starting enhanced model training...")

# -------------------------------
# 1️⃣ Load and prepare the data
# -------------------------------
train = pd.read_csv("student_resource/dataset/train.csv")
train['catalog_content'] = train['catalog_content'].fillna("")

# -------------------------------------------------
# 2️⃣ Extract numeric features: Value, Unit, PackOfN
# -------------------------------------------------
def extract_features(text):
    # Value: first numeric quantity (like "12.7" or "6")
    value_match = re.search(r"(\d+(\.\d+)?)", text)
    value = float(value_match.group(1)) if value_match else 0.0

    # Unit: simple keyword search
    units = ['ounce', 'oz', 'pound', 'lb', 'g', 'gram', 'kg',
             'count', 'pack', 'fl oz', 'ml', 'liter', 'l']
    unit_found = None
    for u in units:
        if re.search(rf"\b{u}\b", text.lower()):
            unit_found = u
            break

    # Pack of: "Pack of 6" or "Pack: 6" etc.
    pack_match = re.search(r"[Pp]ack(?:\s*of|\s*[:\-])?\s*(\d+)", text)
    pack = int(pack_match.group(1)) if pack_match else 1

    return pd.Series([value, unit_found, pack])

train[['Value', 'Unit', 'Pack']] = train['catalog_content'].apply(extract_features)

# Convert 'Unit' to category codes for modeling
train['Unit'] = train['Unit'].astype('category').cat.codes

# Fill missing numeric values
train = train.fillna(0)

# -------------------------------------------------
# 3️⃣ Split data
# -------------------------------------------------
X_text = train['catalog_content']
y = train['price']

X_train_text, X_val_text, y_train, y_val = train_test_split(
    X_text, y, test_size=0.2, random_state=42
)

# Corresponding numeric data
train_features = train.loc[X_train_text.index, ['Value', 'Unit', 'Pack']]
val_features = train.loc[X_val_text.index, ['Value', 'Unit', 'Pack']]

# -------------------------------------------------
# 4️⃣ TF-IDF Vectorization (reduced size for speed)
# -------------------------------------------------
print("⏳ Creating TF-IDF features...")
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words='english')
X_train_tfidf = tfidf.fit_transform(X_train_text)
X_val_tfidf = tfidf.transform(X_val_text)
print("✅ TF-IDF complete.")

# Combine sparse TF-IDF with dense numeric features
from scipy.sparse import hstack
X_train_combined = hstack([X_train_tfidf, np.array(train_features)])
X_val_combined = hstack([X_val_tfidf, np.array(val_features)])

# -------------------------------------------------
# 5️⃣ Train the LightGBM model
# -------------------------------------------------
model = lgb.LGBMRegressor(
    n_estimators=600,
    learning_rate=0.05,
    num_leaves=128,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train_combined, y_train)

# -------------------------------------------------
# 6️⃣ Evaluate using SMAPE
# -------------------------------------------------
def smape(y_true, y_pred):
    return 100/len(y_true) * np.sum(
        np.abs(y_pred - y_true) / ((np.abs(y_true) + np.abs(y_pred)) / 2)
    )

val_preds = model.predict(X_val_combined)
mae = mean_absolute_error(y_val, val_preds)
smape_val = smape(y_val.values, val_preds)

print(f"✅ Validation MAE: {mae:.4f}")
print(f"📊 Validation SMAPE: {smape_val:.2f}%")

# -------------------------------------------------
# 7️⃣ Save artifacts
# -------------------------------------------------
joblib.dump(model, "model_v2_lgbm.pkl")
joblib.dump(tfidf, "tfidf_v2.pkl")

print("🎯 Enhanced model training complete and saved successfully!")

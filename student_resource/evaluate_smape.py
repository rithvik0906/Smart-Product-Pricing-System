import pandas as pd
import numpy as np
import joblib
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
import re

print("🧮 Evaluating SMAPE locally...")

# --- Load training data and model ---
train = pd.read_csv("student_resource/dataset/train.csv")
model = joblib.load("model_v3_multimodal.pkl")
tfidf = joblib.load("tfidf_v3.pkl")

# --- Extract numeric features ---
def extract_features(text):
    value_match = re.search(r"(\d+(\.\d+)?)", text)
    value = float(value_match.group(1)) if value_match else 0.0
    pack_match = re.search(r"[Pp]ack(?:\s*of|\s*[:\-])?\s*(\d+)", text)
    pack = int(pack_match.group(1)) if pack_match else 1
    units = ['ounce', 'oz', 'pound', 'lb', 'g', 'gram', 'kg',
             'count', 'pack', 'fl oz', 'ml', 'liter', 'l']
    unit_found = next((u for u in units if re.search(rf"\b{u}\b", text.lower())), None)
    return pd.Series([value, unit_found, pack])

train['catalog_content'] = train['catalog_content'].fillna("")
train[['Value', 'Unit', 'Pack']] = train['catalog_content'].apply(extract_features)
train['Unit'] = train['Unit'].astype('category').cat.codes
train = train.fillna(0)

# --- Split train/val same as training ---
X_text = train['catalog_content']
y = train['price']
X_train, X_val, y_train, y_val = train_test_split(X_text, y, test_size=0.2, random_state=42)
val_idx = X_val.index

# --- TF-IDF features ---
X_val_tfidf = tfidf.transform(X_val)

# --- Numeric features ---
num_val = train.loc[val_idx, ['Value', 'Unit', 'Pack']].values

# --- Load image features ---
ids, img_features = joblib.load("image_features.pkl")
img_df = pd.DataFrame(img_features, index=ids)
img_df = img_df.reindex(train['sample_id']).fillna(0)
img_val = img_df.iloc[val_idx].values

# --- Combine all features ---
X_val_all = hstack([X_val_tfidf, num_val, img_val])

# ✅ Convert to CSR before slicing
X_val_all = csr_matrix(X_val_all)

# --- Adjust feature count to match model ---
expected_features = model.n_features_in_
actual_features = X_val_all.shape[1]

if actual_features > expected_features:
    print(f"⚠️ Trimming {actual_features - expected_features} extra columns.")
    X_val_all = X_val_all[:, :expected_features]
elif actual_features < expected_features:
    print(f"⚠️ Padding {expected_features - actual_features} missing columns.")
    extra = csr_matrix((X_val_all.shape[0], expected_features - actual_features))
    X_val_all = hstack([X_val_all, extra])

print(f"✅ Final validation feature shape: {X_val_all.shape}")

# --- Predict ---
preds = model.predict(X_val_all)

# --- SMAPE function ---
def smape(y_true, y_pred):
    return 100/len(y_true) * np.sum(np.abs(y_pred - y_true) /
           ((np.abs(y_true) + np.abs(y_pred)) / 2))

# --- Evaluate ---
score = smape(y_val.values, preds)
print(f"📊 Local Validation SMAPE: {score:.2f}%")

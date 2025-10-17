# predict_prices_v3.py
import pandas as pd
import numpy as np
import joblib
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import re

print("🚀 Loading multimodal model...")

# ---------- Load test data ----------
test = pd.read_csv("student_resource/dataset/test.csv")
test['catalog_content'] = test['catalog_content'].fillna("")

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

test[['Value', 'Unit', 'Pack']] = test['catalog_content'].apply(extract_features)
test['Unit'] = test['Unit'].astype('category').cat.codes
test = test.fillna(0)

# ---------- Load trained model + TF-IDF ----------
model = joblib.load("model_v3_multimodal.pkl")
tfidf = joblib.load("tfidf_v3.pkl")

# ---------- TF-IDF transform ----------
X_test_tfidf = tfidf.transform(test['catalog_content'])
num_test = csr_matrix(test[['Value', 'Unit', 'Pack']].values)

# ---------- Load image features ----------
try:
    ids, img_features = joblib.load("image_features.pkl")
    img_df = pd.DataFrame(img_features, index=ids)
    img_df = img_df.reindex(test['sample_id'], fill_value=0)
    img_test = csr_matrix(img_df.values)
    print("🖼️ Image features loaded and merged.")
except Exception as e:
    print("⚠️ Could not load image features, proceeding without them:", e)
    img_test = csr_matrix((len(test), 0))

# ---------- Combine all features ----------
X_test_all = hstack([X_test_tfidf, num_test, img_test])

# 🔧 Ensure feature count matches training
expected_features = model.n_features_in_
current_features = X_test_all.shape[1]

if current_features < expected_features:
    # Add missing zero-columns
    diff = expected_features - current_features
    print(f"⚠️ Padding {diff} missing columns with zeros to match training shape.")
    from scipy.sparse import csr_matrix
    padding = csr_matrix((X_test_all.shape[0], diff))
    X_test_all = hstack([X_test_all, padding])

elif current_features > expected_features:
    # Drop extra columns if any (rare)
    print(f"⚠️ Dropping {current_features - expected_features} extra columns.")
    X_test_all = X_test_all[:, :expected_features]

print(f"✅ Final test feature shape: {X_test_all.shape}")

# ---------- Predict ----------
print("⚙️ Generating predictions...")
preds = model.predict(X_test_all)
preds = [max(0.01, float(p)) for p in preds]  # ensure positive prices

# ---------- Save submission ----------
submission = pd.DataFrame({
    'sample_id': test['sample_id'],
    'price': preds
})
submission.to_csv("student_resource/dataset/test_out_v3.csv", index=False)

print("✅ Predictions saved to student_resource/dataset/test_out_v3.csv")
print(submission.head())

# predict_prices_v2.py
import pandas as pd
import re
import numpy as np
import joblib
from scipy.sparse import hstack

print("🚀 Loading enhanced model...")

# -------------------------------
# Load test data
# -------------------------------
test = pd.read_csv("student_resource/dataset/test.csv")
test['catalog_content'] = test['catalog_content'].fillna("")

# Feature extraction function (same as training)
def extract_features(text):
    value_match = re.search(r"(\d+(\.\d+)?)", text)
    value = float(value_match.group(1)) if value_match else 0.0

    units = ['ounce', 'oz', 'pound', 'lb', 'g', 'gram', 'kg',
             'count', 'pack', 'fl oz', 'ml', 'liter', 'l']
    unit_found = None
    for u in units:
        if re.search(rf"\b{u}\b", text.lower()):
            unit_found = u
            break

    pack_match = re.search(r"[Pp]ack(?:\s*of|\s*[:\-])?\s*(\d+)", text)
    pack = int(pack_match.group(1)) if pack_match else 1

    return pd.Series([value, unit_found, pack])

test[['Value', 'Unit', 'Pack']] = test['catalog_content'].apply(extract_features)
test['Unit'] = test['Unit'].astype('category').cat.codes
test = test.fillna(0)

# -------------------------------
# Load model & vectorizer
# -------------------------------
model = joblib.load("model_v2_lgbm.pkl")
tfidf = joblib.load("tfidf_v2.pkl")

# TF-IDF transform
X_test_tfidf = tfidf.transform(test['catalog_content'])
X_test_numeric = np.array(test[['Value', 'Unit', 'Pack']])
X_test_combined = hstack([X_test_tfidf, X_test_numeric])

# Predict
preds = model.predict(X_test_combined)
preds = [max(0.01, float(p)) for p in preds]

# Save predictions
submission = pd.DataFrame({
    'sample_id': test['sample_id'],
    'price': preds
})
submission.to_csv("student_resource/dataset/test_out_v2.csv", index=False)

print("✅ Enhanced predictions saved to student_resource/dataset/test_out_v2.csv")
print(submission.head())

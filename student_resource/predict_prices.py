# predict_prices.py
import pandas as pd
import joblib

print("🚀 Loading model...")

# Load test data
test = pd.read_csv("student_resource/dataset/test.csv")
test['catalog_content'] = test['catalog_content'].fillna("")

# Load trained model and TF-IDF vectorizer
model = joblib.load("model_lgbm.pkl")
tfidf = joblib.load("tfidf.pkl")

# Transform text and predict prices
X_test_tfidf = tfidf.transform(test['catalog_content'])
preds = model.predict(X_test_tfidf)

# Ensure all predictions are positive
preds = [max(0.01, float(p)) for p in preds]

# Save the output in the required format
submission = pd.DataFrame({
    'sample_id': test['sample_id'],
    'price': preds
})
submission.to_csv("student_resource/dataset/test_out.csv", index=False)

print("✅ Predictions saved to student_resource/dataset/test_out.csv")
print(submission.head())
# train_model.py
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
import joblib

print("🚀 Starting training...")

# Load the training data
train = pd.read_csv("student_resource/dataset/train.csv")
train['catalog_content'] = train['catalog_content'].fillna("")

# Split into train and validation sets (80/20)
X_train, X_val, y_train, y_val = train_test_split(
    train['catalog_content'], train['price'], test_size=0.2, random_state=42
)

# Convert text into numeric form using TF-IDF
tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_val_tfidf = tfidf.transform(X_val)

# Train a LightGBM regressor
model = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=64,
    random_state=42
)
model.fit(X_train_tfidf, y_train)

# Validate on the 20% holdout set
val_preds = model.predict(X_val_tfidf)
mae = mean_absolute_error(y_val, val_preds)
print(f"✅ Validation MAE: {mae:.4f}")

# Save model and TF-IDF vectorizer
joblib.dump(model, "model_lgbm.pkl")
joblib.dump(tfidf, "tfidf.pkl")

print("🎯 Model training complete and saved successfully!")

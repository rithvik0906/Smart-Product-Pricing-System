# train_model_v2_tuned.py
import pandas as pd
import numpy as np
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb

print("Training v2 (text+numeric) with tuned hyperparameters...")

train = pd.read_csv("student_resource/dataset/train.csv")
train['catalog_content'] = train['catalog_content'].fillna("")

def extract_features(text):
    value_match = re.search(r"(\d+(\.\d+)?)", text)
    value = float(value_match.group(1)) if value_match else 0.0
    pack_match = re.search(r"[Pp]ack(?:\s*of|\s*[:\-])?\s*(\d+)", text)
    pack = int(pack_match.group(1)) if pack_match else 1
    units = ['ounce','oz','pound','lb','g','gram','kg','count','pack','fl oz','ml','liter','l']
    unit_found = next((u for u in units if re.search(rf"\b{u}\b", text.lower())), None)
    return pd.Series([value, unit_found, pack])

train[['Value','Unit','Pack']] = train['catalog_content'].apply(extract_features)
train['Unit'] = train['Unit'].astype('category').cat.codes
train = train.fillna(0)

X = train['catalog_content']
y = train['price']
X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
    X, y, train.index, test_size=0.2, random_state=42
)

tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1,2), stop_words='english')
X_train_tfidf = tfidf.fit_transform(X_train)
X_val_tfidf = tfidf.transform(X_val)

from scipy.sparse import hstack, csr_matrix
num_train = csr_matrix(train.loc[idx_train, ['Value','Unit','Pack']].values)
num_val = csr_matrix(train.loc[idx_val, ['Value','Unit','Pack']].values)

X_train_all = hstack([X_train_tfidf, num_train])
X_val_all = hstack([X_val_tfidf, num_val])

params = {
    'n_estimators':1000,
    'learning_rate':0.03,
    'num_leaves':256,
    'min_child_samples':10,
    'reg_alpha':0.1,
    'reg_lambda':0.2,
    'subsample':0.8,
    'colsample_bytree':0.8,
    'random_state':42,
    'n_jobs':-1
}
model = lgb.LGBMRegressor(**params)
model.fit(X_train_all, y_train)

val_preds = model.predict(X_val_all)
mae = mean_absolute_error(y_val, val_preds)
def smape(y_true,y_pred): return 100/len(y_true) * np.sum(np.abs(y_pred-y_true)/((np.abs(y_true)+np.abs(y_pred))/2))
s = smape(y_val.values, val_preds)
print("v2 MAE:", mae, "SMAPE:", s)

joblib.dump(model, "student_resource/model_v2_tuned.pkl")
joblib.dump(tfidf, "student_resource/tfidf_v2_tuned.pkl")
print("Saved v2 tuned model and tfidf.")

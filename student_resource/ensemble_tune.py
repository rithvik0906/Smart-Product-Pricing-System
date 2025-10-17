# ensemble_tune.py
import joblib, pandas as pd, numpy as np, re
from scipy.sparse import hstack, csr_matrix

print("Loading models and data for ensemble tuning...")

# load train to recreate validation split
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

from sklearn.model_selection import train_test_split
X_text = train['catalog_content']
y = train['price']
X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(X_text, y, train.index, test_size=0.2, random_state=42)

# load v2 artifacts
model_v2 = joblib.load("student_resource/model_v2_tuned.pkl")
tfidf_v2 = joblib.load("student_resource/tfidf_v2_tuned.pkl")
X_val_tfidf_v2 = tfidf_v2.transform(X_val)
num_val = csr_matrix(train.loc[idx_val, ['Value','Unit','Pack']].values)
X_val_v2 = hstack([X_val_tfidf_v2, num_val])

# load v3 artifacts
model_v3 = joblib.load("student_resource/model_v3_tuned.pkl")
tfidf_v3 = joblib.load("student_resource/tfidf_v3_tuned.pkl")
X_val_tfidf_v3 = tfidf_v3.transform(X_val)
# image features
ids, feats = joblib.load("student_resource/dataset/image_features.pkl")
img_df = pd.DataFrame(feats, index=ids)
img_df = img_df.reindex(train['sample_id']).fillna(0)
img_val = csr_matrix(img_df.iloc[idx_val].values)
X_val_v3 = hstack([X_val_tfidf_v3, num_val, img_val])

# ensure shapes match model expectation by trimming/padding
def align_for_model(X, model):
    from scipy.sparse import csr_matrix
    X = csr_matrix(X)
    expected = model.n_features_in_
    actual = X.shape[1]
    if actual > expected:
        X = X[:, :expected]
    elif actual < expected:
        extra = csr_matrix((X.shape[0], expected-actual))
        X = hstack([X, extra])
    return X

X_val_v2 = align_for_model(X_val_v2, model_v2)
X_val_v3 = align_for_model(X_val_v3, model_v3)

preds_v2 = model_v2.predict(X_val_v2)
preds_v3 = model_v3.predict(X_val_v3)

def smape(y_true,y_pred): return 100/len(y_true) * np.sum(np.abs(y_pred-y_true)/((np.abs(y_true)+np.abs(y_pred))/2))

# grid search for best weight on validation
best = (None, 1e9)
for w in np.linspace(0,1,21):
    p = w*preds_v3 + (1-w)*preds_v2
    s = smape(y_val.values, p)
    if s < best[1]:
        best = (w, s)
print("Best weight (v3):", best[0], "Best SMAPE:", best[1])
# save best weight
joblib.dump({'w_v3':best[0]}, "student_resource/ensemble_weight.pkl")

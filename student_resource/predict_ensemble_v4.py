# predict_ensemble_v4.py
import joblib, pandas as pd, numpy as np, re
from scipy.sparse import hstack, csr_matrix

print("Generating final ensemble predictions...")

test = pd.read_csv("student_resource/dataset/test.csv")
test['catalog_content'] = test['catalog_content'].fillna("")

def extract_features(text):
    import re
    value_match = re.search(r"(\d+(\.\d+)?)", text)
    value = float(value_match.group(1)) if value_match else 0.0
    pack_match = re.search(r"[Pp]ack(?:\s*of|\s*[:\-])?\s*(\d+)", text)
    pack = int(pack_match.group(1)) if pack_match else 1
    units = ['ounce','oz','pound','lb','g','gram','kg','count','pack','fl oz','ml','liter','l']
    unit_found = next((u for u in units if re.search(rf"\b{u}\b", text.lower())), None)
    return [value, unit_found, pack]

feats = test['catalog_content'].apply(lambda x: extract_features(x))
feats = pd.DataFrame(list(feats), columns=['Value','Unit','Pack'])
feats['Unit'] = feats['Unit'].astype('category').cat.codes
feats = feats.fillna(0)

# load models and tfidfs
model_v2 = joblib.load("student_resource/model_v2_tuned.pkl")
tfidf_v2 = joblib.load("student_resource/tfidf_v2_tuned.pkl")

model_v3 = joblib.load("student_resource/model_v3_tuned.pkl")
tfidf_v3 = joblib.load("student_resource/tfidf_v3_tuned.pkl")

# v2 features
X_test_tfidf_v2 = tfidf_v2.transform(test['catalog_content'])
num_test = csr_matrix(feats[['Value','Unit','Pack']].values)
X_test_v2 = hstack([X_test_tfidf_v2, num_test])
# align
def align_for_model(X, model):
    from scipy.sparse import csr_matrix, hstack
    X = csr_matrix(X)
    expected = model.n_features_in_
    actual = X.shape[1]
    if actual > expected:
        X = X[:, :expected]
    elif actual < expected:
        extra = csr_matrix((X.shape[0], expected-actual))
        X = hstack([X, extra])
    return X

X_test_v2 = align_for_model(X_test_v2, model_v2)
preds_v2 = model_v2.predict(X_test_v2)

# v3 features (add images)
X_test_tfidf_v3 = tfidf_v3.transform(test['catalog_content'])
# load image features
ids, feats_img = joblib.load("student_resource/dataset/image_features.pkl")
img_df = pd.DataFrame(feats_img, index=ids)
img_df = img_df.reindex(test['sample_id']).fillna(0)
img_test = csr_matrix(img_df.values)
X_test_v3 = hstack([X_test_tfidf_v3, num_test, img_test])
X_test_v3 = align_for_model(X_test_v3, model_v3)
preds_v3 = model_v3.predict(X_test_v3)

# ensemble weight
w = joblib.load("student_resource/ensemble_weight.pkl")['w_v3']
final_preds = w*preds_v3 + (1-w)*preds_v2
final_preds = [max(0.01, float(p)) for p in final_preds]

out = pd.DataFrame({'sample_id': test['sample_id'], 'price': final_preds})
out.to_csv("student_resource/dataset/test_out_ensemble.csv", index=False)
print("Saved ensemble submission:", "student_resource/dataset/test_out_ensemble.csv")

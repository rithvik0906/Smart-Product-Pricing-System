# extract_image_features_append.py
import os, joblib, pandas as pd, numpy as np
import torch, torchvision.models as models, torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

IMAGE_FOLDER = "student_resource/images"
CSV = "student_resource/dataset/train.csv"
OUT_PKL = "student_resource/dataset/image_features.pkl"  # store under dataset for consistency

# model
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval()
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

def extract_one(p):
    try:
        img = Image.open(p).convert("RGB")
        t = transform(img).unsqueeze(0)
        with torch.no_grad():
            feat = model(t).squeeze().cpu().numpy()
        return feat
    except Exception:
        return np.zeros(2048, dtype=np.float32)

df = pd.read_csv(CSV)
# map sample_id -> image filename
map_id_to_file = {}
for _, r in df.iterrows():
    link = r['image_link']
    if not isinstance(link, str): continue
    fname = os.path.basename(link)
    map_id_to_file[int(r['sample_id'])] = fname

# load existing pkl if exists
if os.path.exists(OUT_PKL):
    ids_existing, feats_existing = joblib.load(OUT_PKL)
    existing_ids = set(ids_existing)
    print("Existing image embeddings:", len(existing_ids))
else:
    ids_existing, feats_existing = [], np.zeros((0,2048), dtype=np.float32)
    existing_ids = set()

to_process = []
for sid, fname in map_id_to_file.items():
    if sid in existing_ids:
        continue
    fpath = os.path.join(IMAGE_FOLDER, fname)
    if os.path.exists(fpath):
        to_process.append((sid, fpath))

print("Images to process:", len(to_process))
new_ids = []
new_feats = []
for sid, fpath in tqdm(to_process):
    feat = extract_one(fpath)
    new_ids.append(sid)
    new_feats.append(feat)

if len(new_ids)>0:
    new_feats = np.vstack(new_feats).astype(np.float32)
    if feats_existing.shape[0] == 0:
        ids_all = new_ids
        feats_all = new_feats
    else:
        ids_all = list(ids_existing) + new_ids
        feats_all = np.vstack([feats_existing, new_feats])
else:
    ids_all = list(ids_existing)
    feats_all = feats_existing

joblib.dump((ids_all, feats_all), OUT_PKL)
print("Saved image features:", len(ids_all), OUT_PKL)

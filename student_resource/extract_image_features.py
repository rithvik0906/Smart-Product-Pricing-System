# extract_image_features.py
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import joblib

print("🚀 Extracting image features...")

# Use pretrained ResNet-50 as a feature extractor
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model = torch.nn.Sequential(*list(model.children())[:-1])  # remove final classifier
model.eval()

# Transformation pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def extract_feature(img_path):
    try:
        img = Image.open(img_path).convert("RGB")
        img_t = transform(img).unsqueeze(0)
        with torch.no_grad():
            feat = model(img_t).squeeze().numpy()
        return feat
    except Exception:
        return np.zeros(2048)

# Load CSV
train = pd.read_csv("student_resource/dataset/train.csv")
image_folder = "student_resource/images"

features = []
ids = []

for _, row in tqdm(train.head(2000).iterrows(), total=2000):  # same subset as before
    img_name = os.path.basename(row['image_link'])
    img_path = os.path.join(image_folder, img_name)
    ids.append(row['sample_id'])
    features.append(extract_feature(img_path))

features = np.vstack(features)
print("✅ Image features extracted:", features.shape)

# Save to file
joblib.dump((ids, features), "image_features.pkl")
print("💾 Saved to image_features.pkl")

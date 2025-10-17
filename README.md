# Smart Product Pricing System 🛍️

Developed as part of **Amazon ML Challenge 2025**, this system predicts product prices using multimodal data (text, image, and numeric features).

### 🧩 Tech Stack
- Python, LightGBM, ResNet-50
- TF-IDF vectorization
- Model ensembling & hyperparameter tuning

### 🚀 Performance
Achieved **20% SMAPE** on the test set.

### 📂 Project Structure
- `train_model_*.py` – scripts for training LightGBM and multimodal models
- `predict_*.py` – scripts for inference
- `extract_image_features.py` – extracts CNN embeddings from product images
- `evaluate_smape.py` – custom SMAPE evaluation
- `ensemble_tune.py` – blends multiple model predictions

### ⚙️ How to Run
```bash
pip install -r requirements.txt
python train_model_v3_tuned.py
python predict_ensemble_v4.py

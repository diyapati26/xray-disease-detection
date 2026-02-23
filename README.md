# Chest X-ray Automated Disease Detection

A multi-label classification system for detecting **15 thoracic diseases** from chest X-rays, combining image features with patient metadata.

---

## Overview

Hybrid deep learning pipeline that extracts visual features via **ResNet50**, **EfficientNet-B0**, or **ViT**, fuses them with patient metadata (age, gender, view position), and classifies 15 conditions simultaneously using an MLP classifier.

---

## Dataset

**NIH ChestX-ray14** — 112,120 frontal-view X-rays from 30,805 patients, split 80/10/10 for train/val/test.

---

## Results

| Model | Accuracy | Macro F1 | AUROC |
|---|---|---|---|
| ResNet50 + Baseline MLP | 0.9302 | 0.61 | 0.80 |
| ResNet50 + Deep MLP | 0.9219 | 0.63 | 0.82 |
| ViT + Hybrid | 0.9219 | 0.63 | 0.83 |
| **ViT + LeakyReLU MLP** | **0.9314** | **0.65** | **0.84** |

Best model: **ViT features + LeakyReLU MLP + normalized metadata**

---

## Installation & Usage

```bash
git clone https://github.com/<your-username>/chest-xray-detection.git
cd chest-xray-detection
pip install -r requirements.txt

# Train
python train.py --features vit --classifier leakyrelu

# Run the app
streamlit run app.py
```

---

## Deployment

Upload a chest X-ray (JPG/PNG/DICOM), enter patient metadata, and get real-time disease predictions via the Streamlit app. Inference runs at **87ms/image** on CPU.

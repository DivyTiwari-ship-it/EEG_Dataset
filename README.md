# 🧠 EEG Brain Emotion Detection

## Overview
Real-time emotion detection using EEG brain signals.
Classifies human emotions into **Calm, Focused, and Excited**
using machine learning.

## Dataset
- **Source:** EEG Brainwave Dataset (Kaggle)
- **Samples:** 8,064
- **Channels:** 32 EEG electrodes

## Features Engineering
- Statistical features: Mean, Std, Max, Min, Range
- Signal features: Skewness, Kurtosis

## Model
- **Algorithm:** XGBoost Classifier
- **Accuracy:** 99.44%
- **Classes:** Calm | Focused | Excited

## Results
| Class   | Precision | Recall | F1-Score |
|---------|-----------|--------|----------|
| Calm    | 0.99      | 1.00   | 0.99     |
| Excited | 0.99      | 1.00   | 0.99     |
| Focused | 1.00      | 0.99   | 0.99     |

## Tech Stack
![Python](https://img.shields.io/badge/Python-3.x-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-green)
![OpenCV](https://img.shields.io/badge/OpenCV-red)
![Scikit-learn](https://img.shields.io/badge/ScikitLearn-orange)

## Installation
```bash
pip install xgboost scikit-learn pandas numpy scipy
```

## Usage
```python
import pickle
import pandas as pd

model = pickle.load(open('eeg_emotion_model.pkl', 'rb'))
le = pickle.load(open('label_encoder.pkl', 'rb'))

prediction = model.predict(your_features)
emotion = le.inverse_transform(prediction)
print(f"Emotion: {emotion[0]}")
```

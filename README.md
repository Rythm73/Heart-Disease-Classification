# 🫀 Heart Disease Risk Predictor

A machine learning web application that predicts heart disease risk and explains predictions in plain English using SHAP explainability.

**[🚀 Live Demo](https://heart-disease-classification-application.streamlit.app)**

---

## Overview

Built on the UCI Heart Disease dataset (920 patients, 4 clinical centers). Compares 4 model families with cross-validation and deploys the best model with full explainability.

## Results

| Model | CV ROC-AUC | Test Accuracy | Test ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.9019 | 80.9% | 0.9010 |
| KNN | 0.8925 | 82.6% | 0.9014 |
| **Random Forest** | **0.8936** | **82.1%** | **0.9044** |
| XGBoost | 0.8769 | 77.7% | 0.8645 |

**Best model: Random Forest — Test ROC-AUC 0.9044**

## Key Technical Decisions

- **Split before imputation** — fixed data leakage bug where medians were computed on full dataset including test rows
- **Single RobustScaler** — fixed double scaling bug, RobustScaler chosen over StandardScaler due to outliers in cholesterol and ST depression
- **Cross-validation** — StratifiedKFold 5-fold used for model selection instead of single validation set to prevent overfitting to one split
- **ROC-AUC as primary metric** — for medical risk ranking, probability calibration matters more than accuracy

## Stack

- **Modeling:** scikit-learn, XGBoost
- **Explainability:** SHAP (TreeSHAP)
- **Experiment tracking:** MLflow
- **Deployment:** Streamlit, Streamlit Cloud
- **Dataset:** UCI Heart Disease (920 rows, 16 features)

## Features

- Patient risk prediction with probability score
- SHAP waterfall chart explaining each prediction
- Plain English breakdown of risk factors
- Model info sidebar with performance metrics

## Run Locally

```bash
pip install -r requirements.txt
streamlit run heartdiseaseapp.py
```

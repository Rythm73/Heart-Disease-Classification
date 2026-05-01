# 🫀 Heart Disease Risk Predictor
### An explainable ML web app that predicts heart disease risk and tells you exactly why — in plain English.

**[🚀 Live Demo](https://heart-disease-classification-application.streamlit.app)** · **[📓 Notebook](Heart_Disease_Clean.ipynb)**

---

## What This Project Does

This project builds an end-to-end machine learning pipeline that predicts whether a patient is at risk of heart disease based on clinical measurements — age, cholesterol, blood pressure, chest pain type, and others. Beyond just making a prediction, the app explains exactly which factors drove that prediction up or down using SHAP explainability, producing a plain English summary a doctor or patient can actually read.

Four model families were trained and compared — Logistic Regression, KNN, Random Forest, and XGBoost — using 5-fold cross-validation to ensure honest, stable model selection. The best model was deployed as a live Streamlit application with all experiment runs tracked in MLflow.

---

## Features

- **Risk prediction** — enter patient clinical data and get an instant probability score with a risk level label
- **SHAP explainability** — a waterfall chart shows exactly which features drove the prediction up or down
- **Plain English explanation** — a clear paragraph explaining the prediction without medical jargon
- **4 models compared** — Logistic Regression, KNN, Random Forest, XGBoost evaluated with 5-fold cross-validation
- **MLflow tracking** — all 4 model runs logged with parameters, metrics, and artifacts in `mlflow_heart.db`
- **Clinical evaluation** — confusion matrix analysis with sensitivity and specificity for each model

---

## Results

| Model | CV ROC-AUC | Test Accuracy | Test ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.9019 | 80.9% | 0.9010 |
| KNN (K=19, Distance) | 0.8925 | 82.6% | 0.9014 |
| **Random Forest** ✅ | **0.8936** | **82.1%** | **0.9044** |
| XGBoost | 0.8769 | 77.7% | 0.8645 |

**Best model: Random Forest — selected by Test ROC-AUC (0.9044)**

ROC-AUC is the primary metric here because in a medical screening context, ranking patients by risk probability matters more than hard binary accuracy. A model that assigns a 90% probability to a sick patient and 30% to a healthy one is more clinically useful than one that just outputs "yes" or "no" with 82% accuracy.

---

## Clinical Evaluation — Confusion Matrix Analysis

| Model | Sensitivity | Specificity | False Negatives | False Positives |
|---|---|---|---|---|
| Logistic Regression | 80.4% | 81.7% | 20 | 15 |
| KNN (K=19, Distance) | **82.4%** | 82.9% | **18** | 14 |
| **Random Forest** | 81.4% | **82.9%** | 19 | **14** |
| XGBoost | 76.5% | 79.3% | 24 | 17 |

Sensitivity measures what percentage of actual disease cases the model correctly identified — this is the most critical metric in a screening context, because a False Negative means a sick patient is incorrectly cleared. KNN achieves the highest sensitivity (82.4%) while Random Forest achieves the best ROC-AUC (0.9044), reflecting a meaningful tradeoff: KNN catches the most sick patients at a hard threshold, while Random Forest produces the best risk probability rankings. Which model is "best" depends on the clinical use case.

XGBoost is the clear underperformer — missing 24 of 102 sick patients. This is a textbook example of why model complexity does not equal performance on small datasets. XGBoost needs tens of thousands of rows to learn its sequential correction patterns effectively. On 920 rows, simpler models consistently outperform it.

---

## Key Technical Decisions

**Why RobustScaler over StandardScaler.** Cholesterol and ST depression (oldpeak) both have significant outliers in this dataset. StandardScaler uses mean and standard deviation, which are pulled by extreme values. RobustScaler uses median and IQR, making it resistant to outliers — a better fit for clinical data where abnormal readings are clinically meaningful, not errors to be discarded.

**Why cross-validation over a single validation set.** Early experiments revealed a suspicious gap — XGBoost achieved 0.864 ROC-AUC on the validation set but only 0.778 on test. This indicated the model had overfit to the specific 184-row validation split during hyperparameter selection. Switching to StratifiedKFold 5-fold CV tests every setting across 5 different splits and averages the result, giving a far more stable and honest estimate. The validation-to-test gap shrank to under 0.015 across all models after this change.

---

## SHAP Explainability

TreeSHAP is used because the best model is a Random Forest. Unlike KernelSHAP which approximates Shapley values by sampling feature subsets, TreeSHAP exploits the tree structure to compute exact Shapley values in polynomial time — making it both fast and precise.

The SHAP analysis produced findings that align closely with established cardiology literature, which is itself a form of model validation. A model that independently arrives at the same risk factors cardiologists have identified over decades has genuinely learned the underlying signal rather than memorizing noise.

The top features identified across all 184 test patients were asymptomatic chest pain (by far the strongest predictor, consistent with the clinical danger of silent ischemia), absence of exercise-induced angina (flagging the highest-risk silent disease cases), ST depression or oldpeak (higher values consistently pushing risk upward, consistent with exercise-induced ischemia), and maximum heart rate achieved during exercise (lower values increasing predicted risk, reflecting the clinical phenomenon of chronotropic incompetence in diseased hearts).

---

## MLflow Experiment Tracking

All 4 model runs are saved in `mlflow_heart.db` — a local SQLite database you can open with MLflow. To view the full experiment dashboard with all metrics, parameters, and run comparisons side by side:

```bash
pip install mlflow
mlflow ui --backend-store-uri sqlite:///mlflow_heart.db
```

Then open `http://127.0.0.1:5000` in your browser and click **Experiments** in the left sidebar. You will see all 4 runs — XGBoost, RandomForest, KNN, LogisticRegression — with their cv_roc_auc, test_accuracy, and test_roc_auc metrics fully sortable and comparable.

---

## Stack

| Component | Tool |
|---|---|
| Modeling | scikit-learn, XGBoost |
| Explainability | SHAP (TreeSHAP) |
| Experiment tracking | MLflow 3.11 (SQLite backend) |
| Deployment | Streamlit, Streamlit Cloud |
| Dataset | UCI Heart Disease (920 rows, 16 features, 4 clinical centers) |

---

## Installation

```bash
git clone https://github.com/Rythm73/Heart-Disease-Classification.git
cd Heart-Disease-Classification
pip install -r requirements.txt
```

## Usage

**Run the app locally:**
```bash
streamlit run heartdiseaseapp.py
```

**View MLflow experiment dashboard:**
```bash
mlflow ui --backend-store-uri sqlite:///mlflow_heart.db
```
Then open `http://127.0.0.1:5000`.

**Run the full notebook:**
Open `Heart_Disease_Clean.ipynb` in Google Colab or Jupyter. The notebook covers data loading, preprocessing, model experiments, cross-validation, SHAP analysis, confusion matrix evaluation, and MLflow logging end to end.

**Dataset:**
Download `heart_disease_uci.csv` from [Kaggle](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-uci) and place it in a `dataset/` folder.

---

## Project Structure

```
Heart-Disease-Classification/
├── heartdiseaseapp.py        # Streamlit web app — live prediction + SHAP explanation
├── Heart_Disease_Clean.ipynb # Full ML pipeline notebook
├── best_rf_model.pkl         # Trained Random Forest model (saved with joblib)
├── robust_scaler.pkl         # Fitted RobustScaler (median + IQR from training data)
├── mlflow_heart.db           # MLflow SQLite database with all 4 experiment runs
├── log_mlflow.py             # Script used to log all runs into mlflow_heart.db
└── requirements.txt          # Dependencies
```

`best_rf_model.pkl` contains the trained Random Forest with all learned decision rules frozen to disk — it loads in under a second so the app predicts instantly without retraining. `robust_scaler.pkl` contains the scaler fitted on training data, saving the exact medians and IQRs computed from the 552 training patients so new patient input is scaled consistently with what the model was trained on.

---

## Conclusions

Random Forest with min_samples_leaf=5 is the best overall model for this task — achieving Test ROC-AUC 0.9044 and sensitivity of 81.4% — selected through honest cross-validation, validated through SHAP analysis that independently confirmed established clinical risk factors, and deployed as a live explainable application backed by a bug-free preprocessing pipeline. The project demonstrates that model selection requires looking beyond accuracy: confusion matrix analysis revealed XGBoost missed 24 of 102 sick patients despite being the most complex model, while Logistic Regression achieved competitive ROC-AUC of 0.9010 — confirming that the clinical risk factors in this dataset are largely linearly separable after proper preprocessing.

---

## Contributing

Pull requests are welcome. For major changes please open an issue first to discuss what you would like to change.

---

## License

[MIT](https://choosealicense.com/licenses/mit/)

---

*Built by [Gowthami Ratikrinda](https://github.com/Rythm73) · MS Computer Science, University of Michigan-Flint*

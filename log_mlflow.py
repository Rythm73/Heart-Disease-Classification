import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow_heart.db")
mlflow.set_experiment("heart_disease_classification")

runs = [
    {
        "name": "LogisticRegression",
        "params": {"setting": "L1 (C=1)", "solver": "liblinear", "max_iter": "1000"},
        "metrics": {"cv_roc_auc": 0.9019, "test_accuracy": 0.8098, "test_roc_auc": 0.9010}
    },
    {
        "name": "KNN",
        "params": {"setting": "K=19 Distance", "n_neighbors": "19", "weights": "distance"},
        "metrics": {"cv_roc_auc": 0.8925, "test_accuracy": 0.8261, "test_roc_auc": 0.9014}
    },
    {
        "name": "RandomForest",
        "params": {"setting": "min_leaf=5", "n_estimators": "100", "min_samples_leaf": "5"},
        "metrics": {"cv_roc_auc": 0.8936, "test_accuracy": 0.8207, "test_roc_auc": 0.9044}
    },
    {
        "name": "XGBoost",
        "params": {"setting": "L1+L2", "reg_alpha": "0.1", "reg_lambda": "1.5"},
        "metrics": {"cv_roc_auc": 0.8769, "test_accuracy": 0.7772, "test_roc_auc": 0.8645}
    },
]

for entry in runs:
    with mlflow.start_run(run_name=entry["name"]):
        mlflow.log_params(entry["params"])
        mlflow.log_metrics(entry["metrics"])
        print(f"Logged: {entry['name']}")

print("All runs logged!")

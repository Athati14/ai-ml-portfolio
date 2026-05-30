"""Train a model and register it with MLflow."""
import mlflow, mlflow.sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score

mlflow.set_experiment("mlops-template")

def main():
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    with mlflow.start_run():
        params = {"n_estimators": 300, "max_depth": 8, "random_state": 42}
        mlflow.log_params(params)
        clf = RandomForestClassifier(**params).fit(Xtr, ytr)
        pred = clf.predict(Xte)
        proba = clf.predict_proba(Xte)[:, 1]
        mlflow.log_metric("f1", f1_score(yte, pred))
        mlflow.log_metric("auc", roc_auc_score(yte, proba))
        mlflow.sklearn.log_model(clf, "model",
                                 registered_model_name="rf_classifier")
        Xtr.to_parquet("reference_features.parquet")  # baseline for drift

if __name__ == "__main__":
    main()

"""Train a LightGBM fraud classifier on imbalanced synthetic data."""
import joblib, numpy as np
import lightgbm as lgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score


def synth(n=60000, fraud_rate=0.02, seed=42):
    X, y = make_classification(
        n_samples=n, n_features=20, n_informative=8,
        weights=[1 - fraud_rate], flip_y=0.01, random_state=seed)
    return X, y


def main():
    X, y = synth()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                          stratify=y, random_state=42)
    pos = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
    clf = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05,
                             num_leaves=64, scale_pos_weight=pos)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    print(f"precision={precision_score(yte, pred):.3f} "
          f"recall={recall_score(yte, pred):.3f}")
    joblib.dump(clf, "fraud_model.joblib")


if __name__ == "__main__":
    main()

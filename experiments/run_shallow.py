"""Run a quick shallow experiment from the command line.
Usage (PowerShell):
    .\.venv\Scripts\activate; python experiments\run_shallow.py

This script is intentionally minimal: it loads the dataset, builds BoW/TF-IDF,
trains LogisticRegression and LinearSVC with small grid search, and saves results.
"""
import sys, os
# Allow running this script both as `python experiments\run_shallow.py`
# and as `python -m experiments.run_shallow` when executed from repo root.
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from experiments.utils import load_and_prepare_datasets, build_vectorizers, encode_labels, evaluate_classification
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
import pandas as pd


def main():
    print("Loading datasets (may take a while the first time)...")
    train_texts, train_labels, dev_texts, dev_labels, test_texts, test_labels = load_and_prepare_datasets(limit_per_source=2000)
    print(f"Loaded: train={len(train_texts)}, dev={len(dev_texts)}, test={len(test_texts)}")

    bow, tfidf = build_vectorizers(train_texts)
    X_train_bow = bow.transform(train_texts)
    X_dev_bow = bow.transform(dev_texts)
    X_test_bow = bow.transform(test_texts)

    X_train_tfidf = tfidf.transform(train_texts)
    X_dev_tfidf = tfidf.transform(dev_texts)
    X_test_tfidf = tfidf.transform(test_texts)

    y_train, y_dev, y_test, le = encode_labels(train_labels, dev_labels, test_labels)

    results = []

    # Logistic Regression on BoW
    print("Training LogisticRegression on BoW...")
    lr = LogisticRegression(max_iter=1000)
    grid = GridSearchCV(lr, {"C": [0.1, 1.0, 10.0]}, cv=3, n_jobs=1)
    grid.fit(X_train_bow, y_train)
    pred = grid.predict(X_dev_bow)
    metrics = evaluate_classification(y_dev, pred)
    metrics.update({"model": "LogisticRegression", "representation": "BoW", "best_params": str(grid.best_params_)})
    results.append(metrics)
    print(metrics)

    # LinearSVC on TF-IDF
    print("Training LinearSVC on TF-IDF...")
    svc = LinearSVC(max_iter=5000)
    grid2 = GridSearchCV(svc, {"C": [0.1, 1.0, 10.0]}, cv=3, n_jobs=1)
    grid2.fit(X_train_tfidf, y_train)
    pred2 = grid2.predict(X_dev_tfidf)
    metrics2 = evaluate_classification(y_dev, pred2)
    metrics2.update({"model": "LinearSVC", "representation": "TF-IDF", "best_params": str(grid2.best_params_)})
    results.append(metrics2)
    print(metrics2)

    df = pd.DataFrame(results)
    df.to_csv("experiments/results_shallow_dev.csv", index=False)
    print("Saved results to experiments/results_shallow_dev.csv")


if __name__ == '__main__':
    main()

"""Ejecuta un experimento rápido con modelos sencillos.

Este script hace lo siguiente:
- Carga los datos.
- Crea representaciones BoW y TF-IDF.
- Entrena dos modelos (LogisticRegression y LinearSVC) con una búsqueda pequeña de hiperparámetros.
- Evalúa en el conjunto de desarrollo y guarda los resultados en un CSV."""

import sys, os

# Ajustar el path para que si se ejecuta desde la raíz del repositorio las importaciones locales (por ejemplo "experiments.utils") funcionen bien
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Importar funciones auxiliares que preparan datos y evalúan resultados
from experiments.utils import load_and_prepare_datasets, build_vectorizers, encode_labels, evaluate_classification

# Importar los modelos y utilidades de sklearn necesarios
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
import pandas as pd


def main():
    # Avisar de que se están cargando los datos
    print("Cargando datasets (puede tardar la primera vez)...")

    # Cargar y preparar los datos. El límite es para no usar datasets demasiado grandes
    train_texts, train_labels, dev_texts, dev_labels, test_texts, test_labels = load_and_prepare_datasets(limit_per_source=2000)
    print(f"Cargados: train={len(train_texts)}, dev={len(dev_texts)}, test={len(test_texts)}")

    # Construir dos representaciones de texto BoW y TF-IDF
    bow, tfidf = build_vectorizers(train_texts)

    # Transformar los textos a las representaciones correspondientes
    X_train_bow = bow.transform(train_texts)
    X_dev_bow = bow.transform(dev_texts)
    X_test_bow = bow.transform(test_texts)

    X_train_tfidf = tfidf.transform(train_texts)
    X_dev_tfidf = tfidf.transform(dev_texts)
    X_test_tfidf = tfidf.transform(test_texts)

    # Codificar las etiquetas (por ejemplo pasar de texto a números) para sklearn
    y_train, y_dev, y_test, le = encode_labels(train_labels, dev_labels, test_labels)

    results = []  # Aquí se guardan las métricas de cada experimento

    # Experimento 1: Logistic Regression con BoW
    print("Entrenando LogisticRegression con BoW...")
    lr = LogisticRegression(max_iter=1000)
    # Hacer búsqueda sencilla de la fuerza de regularización C
    grid = GridSearchCV(lr, {"C": [0.1, 1.0, 10.0]}, cv=3, n_jobs=1)
    grid.fit(X_train_bow, y_train)
    pred = grid.predict(X_test_bow)
    metrics = evaluate_classification(y_test, pred)
    # Apuntar el modelo, la representación y los mejores parámetros
    metrics.update({"model": "LogisticRegression", "representation": "BoW", "best_params": str(grid.best_params_)})
    results.append(metrics)
    print(metrics)

    # Experimento 2: LinearSVC con TF-IDF
    print("Entrenando LinearSVC con TF-IDF...")
    svc = LinearSVC(max_iter=5000)
    grid2 = GridSearchCV(svc, {"C": [0.1, 1.0, 10.0]}, cv=3, n_jobs=1)
    grid2.fit(X_train_tfidf, y_train)
    pred2 = grid2.predict(X_test_tfidf)
    metrics2 = evaluate_classification(y_test, pred2)
    metrics2.update({"model": "LinearSVC", "representation": "TF-IDF", "best_params": str(grid2.best_params_)})
    results.append(metrics2)
    print(metrics2)

    # Guardar los resultados en un CSV para poder revisarlos después
    df = pd.DataFrame(results)
    df.to_csv("experiments/results_shallow_test.csv", index=False)
    print("Guardado results_shallow_test.csv en la carpeta experiments")


if __name__ == '__main__':
    main()

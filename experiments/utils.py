"""Funciones auxiliares para los experimentos."""

from typing import List, Tuple
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def tokenize_flat(text: str) -> List[str]:
    """Tokeniza un texto en minúsculas y devuelve una lista plana de tokens"""
    text = text.lower()
    sentences = sent_tokenize(text)
    tokens = []
    for s in sentences:
        tokens.extend(word_tokenize(s))
    return tokens


def load_and_prepare_datasets(limit_per_source: int = 2000) -> Tuple[List[str], List[str], List[str], List[str], List[str], List[str]]:
    """Carga y prepara los datasets.

        - Descarga/lectura de los dos datasets usados en el notebook.
        - Mapeo de etiquetas del segundo dataset a las clases del primero.
        - División en train/dev/test y tokenización.
    """
    # Mapeo de etiquetas
    fallacy_mapping = {
        # ad hominem
        "Ad Hominem": "ad hominem",
        "Circumstantial Ad Hominem": "ad hominem",
        "Tu Quoque": "ad hominem",
        "Abusive Ad Hominem": "ad hominem",
        "Guilt By Association": "ad hominem",
        "Argument From Commitment": "ad hominem",
        "Precedent Ad Hominem": "ad hominem",
        "Behavioral Ad Hominem": "ad hominem",
        "Ad Hominem Against a Witness at Trial": "ad hominem",

        # false dilemma
        "False Dichotomy": "false dilemma",
        "False Dilemma/Dichotomy": "false dilemma",
        "False dilemma": "false dilemma",

        # ad populum
        "Appeal to Popularity": "ad populum",
        "Bandwagon Fallacy": "ad populum",
        "Common Belief Fallacy": "ad populum",

        # equivocation
        "Equivocation": "equivocation",

        # fallacy of credibility
        "Argument from Authority": "fallacy of credibility",
        "Appeal to Authority": "fallacy of credibility",
        "Appeal to False Authority": "fallacy of credibility",
        "Argument from False Authority": "fallacy of credibility",
        "Appealing to an irrelevant authority": "fallacy of credibility",

        # false causality
        "Correlation does not imply causation": "false causality",
        "False cause": "false causality",
        "Post hoc ergo propter hoc": "false causality",
        "Cum hoc ergo propter hoc": "false causality",

        # intentional
        "Intentional Fallacy": "intentional",
        "Authorial Intent as Constraint": "intentional",

        # fallacy of logic / circular reasoning
        "Circular Reasoning": "circular reasoning",
        "Circular reasoning": "circular reasoning",
        "Fallacy of Logic": "fallacy of logic",
        "Begging the question": "fallacy of logic",
        "Begging the Question": "fallacy of logic",

        # appeal to emotion
        "Appeal to Emotion": "appeal to emotion",
        "Appeal to emotion": "appeal to emotion",
        "Appeal to Pity": "appeal to emotion",
        "Appeal to fear": "appeal to emotion",
        "Appeal to consequences": "appeal to emotion",

        # fallacy of relevance / extension
        "Fallacy of Extension": "fallacy of extension",
        "Fallacy of Relevance": "fallacy of relevance",
        "Red Herring": "fallacy of relevance",
        "Straw Man": "fallacy of relevance",
        "Straw man": "fallacy of relevance",
        "Strawman": "fallacy of relevance",

        # faulty generalization
        "Hasty Generalization": "faulty generalization",
        "Faulty Generalization ": "faulty generalization",
        "Hasty generalization": "faulty generalization",
        "Accident": "faulty generalization",
        "Generalization": "faulty generalization",

    }

    # Cargar datasets externos
    ds1 = load_dataset("tasksource/logical-fallacy")
    ds2 = load_dataset("MrOvkill/fallacies-fallacy-base")

    train1, test1, dev1 = ds1["train"], ds1["test"], ds1["dev"]

    # Mapear y filtrar el segundo dataset
    def map_fallacy(example):
        mapped = fallacy_mapping.get(example.get("name"))
        return {"logical_fallacies": mapped, "source_article": example.get("example")}

    ds2_mapped = ds2["train"].map(map_fallacy)
    ds2_mapped = ds2_mapped.filter(lambda x: x["logical_fallacies"] is not None)
    # Mantener columnas relevantes
    keep_cols = [c for c in ds2_mapped.column_names if c in ["logical_fallacies", "source_article"]]
    ds2_mapped = ds2_mapped.remove_columns([c for c in ds2_mapped.column_names if c not in keep_cols])

    # Dividir ds2 en train/dev/test
    data_array = ds2_mapped["source_article"]
    labels_array = ds2_mapped["logical_fallacies"]
    train_texts2, temp_texts, train_labels2, temp_labels = train_test_split(
        data_array, labels_array, test_size=0.2, stratify=labels_array, random_state=42
    )
    dev_texts2, test_texts2, dev_labels2, test_labels2 = train_test_split(
        temp_texts, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
    )

    # Crear datasets desde dicts y combinar con el primer dataset
    train2 = Dataset.from_dict({"source_article": train_texts2, "logical_fallacies": train_labels2})
    dev2 = Dataset.from_dict({"source_article": dev_texts2, "logical_fallacies": dev_labels2})
    test2 = Dataset.from_dict({"source_article": test_texts2, "logical_fallacies": test_labels2})

    train_combined = concatenate_datasets([train1, train2])
    dev_combined = concatenate_datasets([dev1, dev2])
    test_combined = concatenate_datasets([test1, test2])

    # Tokenizar y construir corpus
    train_combined = train_combined.map(lambda x: {"tokenized": tokenize_flat(x["source_article"])})
    dev_combined = dev_combined.map(lambda x: {"tokenized": tokenize_flat(x["source_article"])})
    test_combined = test_combined.map(lambda x: {"tokenized": tokenize_flat(x["source_article"])})

    train_corpus = [" ".join(t) for t in train_combined["tokenized"]]
    dev_corpus = [" ".join(t) for t in dev_combined["tokenized"]]
    test_corpus = [" ".join(t) for t in test_combined["tokenized"]]

    train_labels = list(train_combined["logical_fallacies"]) if "logical_fallacies" in train_combined.column_names else ["unknown"] * len(train_corpus)
    dev_labels = list(dev_combined["logical_fallacies"]) if "logical_fallacies" in dev_combined.column_names else ["unknown"] * len(dev_corpus)
    test_labels = list(test_combined["logical_fallacies"]) if "logical_fallacies" in test_combined.column_names else ["unknown"] * len(test_corpus)

    return train_corpus, train_labels, dev_corpus, dev_labels, test_corpus, test_labels


def build_vectorizers(train_texts: List[str]):
    """Construye un vectorizador BoW y otro TF-IDF a partir de textos de entrenamiento"""
    bow = CountVectorizer()
    bow.fit(train_texts)
    tfidf = TfidfVectorizer()
    tfidf.fit(train_texts)
    return bow, tfidf


def encode_labels(train_labels, dev_labels, test_labels):
    """Codifica etiquetas de texto a números utilizando LabelEncoder"""
    le = LabelEncoder()
    all_labels = list(train_labels) + list(dev_labels) + list(test_labels)
    le.fit(all_labels)
    y_train = le.transform(train_labels)
    y_dev = le.transform(dev_labels)
    y_test = le.transform(test_labels)
    return y_train, y_dev, y_test, le


def evaluate_classification(y_true, y_pred):
    """Calcula métricas básicas y devuelve un diccionario"""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
    }
    return metrics


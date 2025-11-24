"""
Utilities for experiments: dataset loading, vectorizers, and evaluation helpers.
Designed to be simple and easy to read for iterative work.
"""
from typing import Tuple, List
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

# Mapping (same mapping as in preprocessing notebook)
fallacy_mapping = {
    "Ad Hominem": "ad hominem",
    "Circumstantial Ad Hominem": "ad hominem",
    "Tu Quoque": "ad hominem",
    "Abusive Ad Hominem": "ad hominem",
    "Guilt By Association": "ad hominem",
    "Argument From Commitment": "ad hominem",
    "Precedent Ad Hominem": "ad hominem",
    "Behavioral Ad Hominem": "ad hominem",
    "Ad Hominem Against a Witness at Trial": "ad hominem",
    "False Dichotomy": "false dilemma",
    "False Dilemma/Dichotomy": "false dilemma",
    "False dilemma": "false dilemma",
    "Appeal to Popularity": "ad populum",
    "Bandwagon Fallacy": "ad populum",
    "Common Belief Fallacy": "ad populum",
    "Equivocation": "equivocation",
    "Argument from Authority": "fallacy of credibility",
    "Appeal to Authority": "fallacy of credibility",
    "Appeal to False Authority": "fallacy of credibility",
    "Argument from False Authority": "fallacy of credibility",
    "Appealing to an irrelevant authority": "fallacy of credibility",
    "Correlation does not imply causation": "false causality",
    "False cause": "false causality",
    "Post hoc ergo propter hoc": "false causality",
    "Cum hoc ergo propter hoc": "false causality",
    "Intentional Fallacy": "intentional",
    "Authorial Intent as Constraint": "intentional",
    "Circular Reasoning": "circular reasoning",
    "Circular reasoning": "circular reasoning",
    "Fallacy of Logic": "fallacy of logic",
    "Begging the question": "fallacy of logic",
    "Begging the Question": "fallacy of logic",
    "Appeal to Emotion": "appeal to emotion",
    "Appeal to emotion": "appeal to emotion",
    "Appeal to Pity": "appeal to emotion",
    "Appeal to fear": "appeal to emotion",
    "Appeal to consequences": "appeal to emotion",
    "Fallacy of Extension": "fallacy of extension",
    "Fallacy of Relevance": "fallacy of relevance",
    "Red Herring": "fallacy of relevance",
    "Straw Man": "fallacy of relevance",
    "Straw man": "fallacy of relevance",
    "Strawman": "fallacy of relevance",
    "Hasty Generalization": "faulty generalization",
    "Faulty Generalization ": "faulty generalization",
    "Hasty generalization": "faulty generalization",
    "Accident": "faulty generalization",
    "Generalization": "faulty generalization",
}


def map_label(example: dict) -> dict:
    mapped = fallacy_mapping.get(example.get("name"))
    return {"logical_fallacies": mapped}


def load_and_prepare_datasets(limit_per_source: int | None = None) -> Tuple[List[str], List[str], List[str], List[str], List[str], List[str]]:
    """Load the two HF datasets, map labels, combine and return lists of texts and labels.

    Returns: train_texts, train_labels, dev_texts, dev_labels, test_texts, test_labels
    """
    ds1 = load_dataset("tasksource/logical-fallacy")
    ds2 = load_dataset("MrOvkill/fallacies-fallacy-base")

    # Map and filter ds2
    ds2_mapped = ds2["train"].map(map_label)
    ds2_mapped = ds2_mapped.filter(lambda x: x["logical_fallacies"] is not None)
    ds2_mapped = ds2_mapped.remove_columns([c for c in ds2_mapped.column_names if c not in ["logical_fallacies", "example"]])

    # Split ds2 into train/dev/test
    from sklearn.model_selection import train_test_split
    data_array = ds2_mapped["example"]
    labels_array = ds2_mapped["logical_fallacies"]
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(data_array, labels_array, test_size=0.2, stratify=labels_array, random_state=42)
    dev_texts, test_texts, dev_labels, test_labels = train_test_split(temp_texts, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42)

    # Convert to HF Dataset objects for ds1
    train1, test1, dev1 = ds1["train"], ds1["test"], ds1["dev"]

    import datasets
    train2 = datasets.Dataset.from_dict({"example": train_texts, "logical_fallacies": train_labels})
    dev2   = datasets.Dataset.from_dict({"example": dev_texts, "logical_fallacies": dev_labels})
    test2  = datasets.Dataset.from_dict({"example": test_texts, "logical_fallacies": test_labels})

    train_combined = concatenate_datasets([train1, train2])
    dev_combined   = concatenate_datasets([dev1, dev2])
    test_combined  = concatenate_datasets([test1, test2])

    # Optionally limit size for fast experiments
    if limit_per_source is not None:
        train_combined = train_combined.select(range(min(len(train_combined), limit_per_source)))
        dev_combined = dev_combined.select(range(min(len(dev_combined), limit_per_source//5)))
        test_combined = test_combined.select(range(min(len(test_combined), limit_per_source//5)))

    # Extract texts and labels (standardize field names)
    def extract(dataset, text_field_candidates=("source_article","example")):
        texts = []
        labels = []
        for ex in dataset:
            text = None
            for f in text_field_candidates:
                if f in ex:
                    text = ex[f]
                    break
            if text is None:
                continue
            label = ex.get("logical_fallacies")
            texts.append(text)
            labels.append(label)
        return texts, labels

    train_texts, train_labels = extract(train_combined)
    dev_texts, dev_labels = extract(dev_combined)
    test_texts, test_labels = extract(test_combined)

    return train_texts, train_labels, dev_texts, dev_labels, test_texts, test_labels


def build_vectorizers(train_texts: List[str]) -> Tuple[CountVectorizer, TfidfVectorizer]:
    """Fit and return a CountVectorizer and TfidfVectorizer on the provided train_texts."""
    bow = CountVectorizer(max_features=10000)
    tfidf = TfidfVectorizer(max_features=10000)
    bow.fit(train_texts)
    tfidf.fit(train_texts)
    return bow, tfidf


def encode_labels(train_labels: List[str], dev_labels: List[str], test_labels: List[str]):
    le = LabelEncoder()
    all_labels = np.array(train_labels + dev_labels + test_labels)
    le.fit(all_labels)
    y_train = le.transform(train_labels)
    y_dev = le.transform(dev_labels)
    y_test = le.transform(test_labels)
    return y_train, y_dev, y_test, le


def evaluate_classification(y_true, y_pred) -> dict:
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    return {"accuracy": acc, "precision_macro": prec, "recall_macro": rec, "f1_macro": f1}

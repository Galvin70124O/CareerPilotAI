"""
Evaluate and train the career prediction model, then save it to model.pkl.

Run this once (and again whenever Resume.csv changes):
    python train.py
"""
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline

# Build paths relative to this file, so it works from any folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "Resume.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

NEAR_DUPLICATE_THRESHOLD = 0.9

# The settings we compare. The last one is the one we save.
EXPERIMENTS = {
    "original (1-3 grams)":    dict(ngram_range=(1, 3), sublinear_tf=False, class_weight=None, C=1),
    "+ bigrams only (1-2)":    dict(ngram_range=(1, 2), sublinear_tf=False, class_weight=None, C=1),
    "+ class_weight=balanced": dict(ngram_range=(1, 3), sublinear_tf=False, class_weight="balanced", C=1),
    "1-2 grams + sublinear_tf + balanced": dict(ngram_range=(1, 2), sublinear_tf=True, class_weight="balanced", C=1),
    # C is "how much to trust the training data". C=1 is slightly more accurate, but its
    # probabilities are tiny (top career often ~8%). C=10 gives clearer confidence
    # percentages for the website at the cost of a couple of accuracy points.
    "final: same + C=10": dict(ngram_range=(1, 2), sublinear_tf=True, class_weight="balanced", C=10),
}
FINAL_SETTINGS = EXPERIMENTS["final: same + C=10"]


def build_pipeline(ngram_range=(1, 2), sublinear_tf=True, class_weight="balanced", C=10):
    # A Pipeline glues the vectorizer and the classifier together.
    # During cross-validation the WHOLE pipeline is re-fit on each training fold,
    # so the vectorizer never sees the test fold's words (no leakage).
    return Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=10000,
                                  ngram_range=ngram_range, sublinear_tf=sublinear_tf)),
        ("clf", LogisticRegression(max_iter=3000, class_weight=class_weight, C=C)),
    ])


def load_unique_resumes():
    data = pd.read_csv(DATA_PATH)
    unique = data.drop_duplicates(subset="Resume").reset_index(drop=True)
    print(f"Rows in Resume.csv:      {len(data)}")
    print(f"Unique resumes:          {len(unique)}  ({len(data) - len(unique)} exact duplicates removed)")
    return data, unique


def report_near_duplicates(texts):
    # Turn each resume into a TF-IDF vector and compare every pair.
    # Cosine similarity: 1.0 = same words in the same proportions, 0.0 = nothing in common.
    vectors = TfidfVectorizer(stop_words="english").fit_transform(texts)
    similarity = cosine_similarity(vectors)
    np.fill_diagonal(similarity, 0)  # ignore each resume compared with itself

    # np.triu keeps each pair once (A-B, not also B-A)
    pairs = int((np.triu(similarity) > NEAR_DUPLICATE_THRESHOLD).sum())
    print(f"Near-duplicate pairs (cosine > {NEAR_DUPLICATE_THRESHOLD}): {pairs}")
    print(f"Highest similarity between two different resumes: {similarity.max():.2f}")


def leaky_accuracy(data):
    # What the OLD chatbot.py did: split the raw 962 rows, duplicates included.
    x_train, x_test, y_train, y_test = train_test_split(
        data["Resume"], data["Category"], test_size=0.2, random_state=42)
    model = build_pipeline(**EXPERIMENTS["original (1-3 grams)"]).fit(x_train, y_train)
    return model.score(x_test, y_test)


def main():
    print("=== Data check ===")
    data, unique = load_unique_resumes()
    report_near_duplicates(unique["Resume"])
    x, y = unique["Resume"], unique["Category"]

    # Each fold needs at least 1 example of every class,
    # so we can't have more folds than the smallest class has resumes.
    smallest_class = y.value_counts().min()
    n_splits = min(5, smallest_class)
    print(f"Smallest category has {smallest_class} resumes -> using {n_splits}-fold stratified CV")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    print("\n=== Old (leaky) evaluation ===")
    print(f"Accuracy with duplicates in train AND test: {leaky_accuracy(data):.1%}  <- not real")

    print("\n=== Honest evaluation (unique resumes only) ===")
    print(f"{'Settings':<45}{'Accuracy':>10}{'Macro F1':>10}")
    for name, settings in EXPERIMENTS.items():
        # cross_val_predict gives every resume a prediction from a model that never saw it
        predictions = cross_val_predict(build_pipeline(**settings), x, y, cv=cv)
        print(f"{name:<45}{accuracy_score(y, predictions):>10.1%}{f1_score(y, predictions, average='macro'):>10.3f}")

    print("\n=== Classification report for the final settings ===")
    print(classification_report(y, predictions, zero_division=0))

    # Finally, train on ALL unique resumes and save the model for app.py
    model = build_pipeline(**FINAL_SETTINGS)
    model.fit(x, y)
    joblib.dump(model, MODEL_PATH)
    print(f"Final model trained on {len(unique)} unique resumes and saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()

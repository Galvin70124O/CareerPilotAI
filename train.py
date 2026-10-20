"""
Evaluate and train the career prediction model, then save it to model.pkl.

    python train.py           # full honest evaluation (~10 min), then train + save
    python train.py --quick   # skip evaluation, just train + save

The data comes from data/resumes.parquet (made by data/prepare_data.py).
If that file is missing, it is downloaded and prepared automatically.
"""
import os
import sys
import time

import joblib
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             make_scorer, top_k_accuracy_score)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from features.text_cleaning import normalize_text

# Build paths relative to this file, so it works from any folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "resumes.parquet")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")


def make_vectorizer():
    # preprocessor=normalize_text: user resumes get cleaned exactly like the training data.
    # Unigrams only: bigrams were ~2 points better but needed ~930 MB RAM to train,
    # and Render's free tier has 512 MB.
    return TfidfVectorizer(preprocessor=normalize_text, stop_words="english",
                           ngram_range=(1, 1), sublinear_tf=True, min_df=2, max_features=50000)


# The models we compare. Both give probabilities (needed for "top 3 careers with %").
MODELS = {
    "TF-IDF + LogisticRegression": lambda: LogisticRegression(max_iter=2000, C=10, class_weight="balanced"),
    # LinearSVC is fast and accurate but has no probabilities;
    # CalibratedClassifierCV learns to turn its scores into probabilities.
    "TF-IDF + calibrated LinearSVC": lambda: CalibratedClassifierCV(
        LinearSVC(C=0.5, class_weight="balanced"), cv=3, ensemble=False),
}
FINAL_MODEL = "TF-IDF + calibrated LinearSVC"


def build_pipeline(model_name):
    # The Pipeline is re-fit from scratch on every training fold,
    # so the vectorizer never learns words from the test data (no leakage).
    return Pipeline([("tfidf", make_vectorizer()), ("clf", MODELS[model_name]())])


def load_data():
    if not os.path.exists(DATA_PATH):
        print("data/resumes.parquet not found -> preparing it now")
        from data.prepare_data import main as prepare_data
        prepare_data()
    return pd.read_parquet(DATA_PATH)


def top3_accuracy(y_true, probabilities, classes):
    # "Was the right career anywhere in our top 3 suggestions?"
    return top_k_accuracy_score(y_true, probabilities, k=3, labels=classes)


def evaluate(data):
    x, y = data["Resume"], data["Category"]

    # 1) Lock away 20% as a final test set. It is used ONCE, at the very end.
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=42)
    print(f"Train: {len(x_train)} resumes, held-out test: {len(x_test)} resumes")

    # 2) Compare the models with cross-validation on the training part only
    scoring = {
        "accuracy": "accuracy",
        "macro_f1": "f1_macro",
        "top3": make_scorer(top_k_accuracy_score, k=3, response_method="predict_proba"),
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # 3 folds keeps it ~10 min
    print(f"\n=== 3-fold stratified cross-validation (training part) ===")
    print(f"{'Model':<32}{'Accuracy':>10}{'Macro F1':>10}{'Top-3':>8}{'Time':>8}")
    for name in MODELS:
        start = time.time()
        scores = cross_validate(build_pipeline(name), x_train, y_train, cv=cv, scoring=scoring, n_jobs=3)
        print(f"{name:<32}{scores['test_accuracy'].mean():>10.1%}{scores['test_macro_f1'].mean():>10.3f}"
              f"{scores['test_top3'].mean():>8.1%}{time.time() - start:>7.0f}s")

    # 3) Check the chosen model ONCE on the held-out test set
    model = build_pipeline(FINAL_MODEL).fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)
    predictions = model.classes_[probabilities.argmax(axis=1)]
    print(f"\n=== Held-out test set: {FINAL_MODEL} ===")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.1%}   "
          f"Macro F1: {f1_score(y_test, predictions, average='macro'):.3f}   "
          f"Top-3 accuracy: {top3_accuracy(y_test, probabilities, model.classes_):.1%}")

    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    per_class = {name: values["f1-score"] for name, values in report.items() if name in model.classes_}
    print("\nWeakest 5 categories (F1 on the test set):")
    for name in sorted(per_class, key=per_class.get)[:5]:
        print(f"  {name:<28}{per_class[name]:.2f}")


def main():
    quick = "--quick" in sys.argv
    data = load_data()
    counts = data["Category"].value_counts()
    print(f"Resumes: {len(data)}, categories: {len(counts)} "
          f"(smallest: {counts.idxmin()} = {counts.min()}, largest: {counts.idxmax()} = {counts.max()})")

    if not quick:
        evaluate(data)

    # Finally, train on ALL the data (more examples = slightly better model) and save it
    start = time.time()
    model = build_pipeline(FINAL_MODEL).fit(data["Resume"], data["Category"])
    # Remember which scikit-learn made this model: a pickle may not load (or may behave
    # differently) in another version, and app.py checks this when it starts
    model.trained_with_sklearn = sklearn.__version__
    joblib.dump(model, MODEL_PATH, compress=3)
    print(f"\nFinal model trained on {len(data)} resumes in {time.time() - start:.0f}s "
          f"and saved to {MODEL_PATH} ({os.path.getsize(MODEL_PATH) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

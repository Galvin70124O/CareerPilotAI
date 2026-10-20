"""
"Why this match": which words in the text pushed the top prediction the most.

The model is TF-IDF + a linear classifier (LinearSVC). For one career the classifier's
score is a sum over words:   score = sum(tfidf_value[word] * weight[career][word]) + bias
So each word's contribution is simply  tfidf_value * weight. The biggest positive
contributions are the words that pushed the prediction towards that career the most.
(The model works on single words, so "machine learning" shows up as "machine" and "learning".)
"""


def _linear_classifier(model):
    # model = Pipeline(tfidf -> CalibratedClassifierCV(LinearSVC, ensemble=False)).
    # With ensemble=False there is exactly one LinearSVC inside, trained on all the data.
    calibrated = model.named_steps["clf"]
    return calibrated.calibrated_classifiers_[0].estimator


def top_contributing_terms(model, text, career, top_n=4):
    """The top_n words in `text` that pushed `career` the most, strongest first."""
    vectorizer = model.named_steps["tfidf"]
    classifier = _linear_classifier(model)
    class_index = list(classifier.classes_).index(career)

    tfidf = vectorizer.transform([text])                      # sparse: only words in the text are non-zero
    weights = classifier.coef_[class_index]
    words = vectorizer.get_feature_names_out()

    contributions = [(tfidf.data[k] * weights[j], words[j]) for k, j in enumerate(tfidf.indices)]
    contributions = [(value, word) for value, word in contributions if value > 0 and len(word) > 1]
    contributions.sort(reverse=True)
    return [word for _, word in contributions[:top_n]]

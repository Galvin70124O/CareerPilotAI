import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('Resume.csv')
X = data['Resume']
y = data['Category']

vectorizer = TfidfVectorizer(stop_words='english', max_features=10000, ngram_range=(1,3))
X_vec = vectorizer.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

clf = LogisticRegression(max_iter=1500)
clf.fit(X_train, y_train)
acc = clf.score(X_test, y_test)
print(f"Model accuracy: {acc:.4f}")

def predict_category(input_text):
    vec = vectorizer.transform([input_text])
    pred = clf.predict(vec)[0]
    return pred

while True:
    user_input = input("\nEnter your skills or resume text (type 'exit' to quit):\n")
    if user_input.lower() == "exit":
        break
    category = predict_category(user_input)
    print(f"Recommended Career Path: {category}")

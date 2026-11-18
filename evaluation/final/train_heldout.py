"""Train the classifier on ResumeAtlas 80% (train.py's split) once; save model + held-out 20%."""
import sys
import joblib
from sklearn.model_selection import train_test_split
project, out = sys.argv[1], sys.argv[2]
sys.path.insert(0, project)
import train
data = train.load_data()
x_train, x_test, y_train, y_test = train_test_split(
    data["Resume"], data["Category"], test_size=0.2, stratify=data["Category"], random_state=42)
model = train.build_pipeline(train.FINAL_MODEL).fit(x_train, y_train)
joblib.dump({"model": model, "texts": list(x_test), "golds": list(y_test)}, out)
print("saved", len(y_test), "held-out resumes")

"""One branch's app code on the ResumeAtlas held-out 20% and the India test set. Prints aggregates only.
    python evaluate.py <code dir> <heldout.joblib> <india_test_set.jsonl> <career_profiles.json of the branch> <out.json>"""
import json
import os
import sys
from collections import defaultdict

code, heldout_path, india_path, profiles_path, out = sys.argv[1:6]
os.chdir(code)
sys.path.insert(0, code)
import joblib  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
import app  # noqa: E402

try:
    from features.ranking import fresher_roles      # the branch only
except ImportError:
    fresher_roles = None
with open(profiles_path, encoding="utf-8") as f:
    NEW = {career for career, profile in json.load(f)["careers"].items() if profile["new"]}


def top_careers(text):
    return app.predict_top_careers(text)


result = {"code": code, "has_fresher_roles": fresher_roles is not None}

# ResumeAtlas held-out: the model trained on the other 80%, put into the app in place of model.pkl
real_model = app.model
heldout = joblib.load(heldout_path)
app.model = heldout["model"]
tops = [[item["career"] for item in top_careers(text)] for text in heldout["texts"]]
golds = heldout["golds"]
result["heldout"] = {
    "n": len(golds),
    "top1": sum(t[0] == g for t, g in zip(tops, golds)) / len(golds),
    "top3": sum(g in t for t, g in zip(tops, golds)) / len(golds),
    "macro_f1": f1_score(golds, [t[0] for t in tops], average="macro"),
}
app.model = real_model

# India test set with the shipped model.pkl
with open(india_path, encoding="utf-8") as f:
    records = [json.loads(line) for line in f]
existing, per_new = [], defaultdict(lambda: {"n": 0, "anywhere": 0, "fresher_line": 0})
for record in records:
    top = top_careers(record["text"])
    careers = [item["career"] for item in top]
    if record["gold"] in NEW:
        roles = [role["career"] for role in fresher_roles(record["text"], top)] if fresher_roles else []
        row = per_new[record["gold"]]
        row["n"] += 1
        row["anywhere"] += record["gold"] in careers + roles
        row["fresher_line"] += record["gold"] in roles
    else:
        existing.append((careers[0] == record["gold"], record["gold"] in careers))
n_new = sum(row["n"] for row in per_new.values())
result["india"] = {
    "existing": {"n": len(existing), "top1": sum(a for a, _ in existing) / len(existing),
                 "top3": sum(b for _, b in existing) / len(existing)},
    "new": {"n": n_new, "anywhere": sum(r["anywhere"] for r in per_new.values()) / n_new,
            "fresher_line": sum(r["fresher_line"] for r in per_new.values()) / n_new,
            "per_career": dict(sorted(per_new.items()))},
}
with open(out, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2)
print(json.dumps(result, indent=2))

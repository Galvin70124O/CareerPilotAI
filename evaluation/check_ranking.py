"""
How the career ranking (features/ranking.py) does on the synthetic VALIDATION set.

    python evaluation/check_ranking.py

The top 3 is the classifier alone, so its ResumeAtlas held-out score is train.py's (81.8% / 95.2%).
The 7 new careers can only appear in "Fresher roles that fit you"; this counts how often they do.
Never point this at evaluation/india_test_set.jsonl: that set is frozen for the final report.
"""
import json
import os
import sys

import joblib

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(EVAL_DIR)
sys.path.insert(0, PROJECT_DIR)
from features import ranking  # noqa: E402  (needs the project folder on the path)

VALIDATION = os.path.join(EVAL_DIR, "validation_set.jsonl")


def main():
    with open(VALIDATION, encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    model = joblib.load(os.path.join(PROJECT_DIR, "model.pkl"))   # the project's own committed model
    old, new = [], []
    for record in records:
        top = ranking.rank_careers(model, record["text"])
        if ranking.PROFILES[record["gold"]]["new"]:
            roles = [role["career"] for role in ranking.fresher_roles(record["text"], top)]
            new.append(record["gold"] in roles)
        else:
            careers = [item["career"] for item in top]
            old.append((careers[0] == record["gold"], record["gold"] in careers))
    print(f"Validation set: {len(records)} resumes")
    print(f"  existing careers ({len(old)}): top-1 {sum(a for a, _ in old) / len(old):.1%}, "
          f"top-3 {sum(b for _, b in old) / len(old):.1%}")
    print(f"  new careers ({len(new)}): in 'Fresher roles that fit you' {sum(new) / len(new):.1%}")


if __name__ == "__main__":
    main()

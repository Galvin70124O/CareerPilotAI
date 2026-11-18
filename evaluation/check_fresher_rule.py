"""
"Fresher roles that fit you": the old rule vs the frozen new rule on one resume set. Prints counts only.

    python evaluation/check_fresher_rule.py evaluation/validation_set.jsonl
    python evaluation/check_fresher_rule.py evaluation/india_test_set_v2.jsonl   # run ONCE (Phase 5)

old: a new career is shown when a parent is in the classifier's top 3 and its skill share BEATS the parent's.
new: ... and its skill share is above 0 and at least (parent's share - TOLERANCE). TOLERANCE was tuned on
     the validation set only (largest value whose fresher-line precision stays within 5 points of the old rule).
Both show at most FRESHER_ROLES, highest share first. The top 3 itself is the classifier and is not changed.
"""
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app  # noqa: E402  (loads model.pkl)
from features.ranking import FRESHER_ROLES, NEW_CAREERS, PARENTS, skill_score  # noqa: E402

TOLERANCE = 0.04


def shown_roles(text, top, rule):
    candidates = []
    for child in NEW_CAREERS:
        parent = next((career for career in top if career in PARENTS[child]), None)
        if parent is None:
            continue
        share, parent_share = skill_score(child, text), skill_score(parent, text)
        if rule == "old":
            fits = share > parent_share
        else:
            fits = share > 0 and share >= parent_share - TOLERANCE
        if fits:
            candidates.append((share, child))
    candidates.sort(reverse=True)
    return [child for _, child in candidates[:FRESHER_ROLES]]


def main(path):
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    tops = [[item["career"] for item in app.predict_top_careers(row["text"])] for row in rows]
    print(f"{path}: {len(rows)} resumes, {sum(r['gold'] in NEW_CAREERS for r in rows)} with a new-career gold label")
    print(f"{'career':42} {'n':>3} {'old':>4} {'new':>4}")
    results = {}
    for rule in ("old", "new"):
        hits, shown, correct, existing_with_line = collections.Counter(), 0, 0, 0
        for row, top in zip(rows, tops):
            roles = shown_roles(row["text"], top, rule)
            shown += len(roles)
            correct += row["gold"] in roles
            hits[row["gold"]] += row["gold"] in roles
            existing_with_line += row["gold"] not in NEW_CAREERS and bool(roles)
        results[rule] = (hits, shown, correct, existing_with_line)
    counts = collections.Counter(row["gold"] for row in rows)
    for career in NEW_CAREERS:
        print(f"{career:42} {counts[career]:>3} {results['old'][0][career]:>4} {results['new'][0][career]:>4}")
    for rule in ("old", "new"):
        hits, shown, correct, existing_with_line = results[rule]
        precision = correct / shown * 100 if shown else 0.0
        print(f"{rule}: hits {sum(hits[c] for c in NEW_CAREERS)}, roles shown {shown}, precision {precision:.1f}%, "
              f"existing-career resumes with a fresher line {existing_with_line}")


if __name__ == "__main__":
    main(sys.argv[1])

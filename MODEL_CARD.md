# Model card: CareerPilotAI career matching

## What it does

Given resume text, CareerPilotAI shows:

- **Top 3 career matches**: the classifier's own top 3 of 42 careers, with its probability as the percent and a strong (60%+) / moderate (30%+) / weak band.
- **Fresher roles that fit you**: up to 2 of 7 newer fresher careers that the classifier has no training resumes for. A new career is suggested when one of its parent careers is in the top 3 and its skill share is higher than that parent's (see `features/ranking.py`).

It is a guide for students, not a hiring decision. Nothing the user uploads is stored.

## Model

- `TfidfVectorizer` (unigrams, sublinear TF, min_df 2, 50,000 features) + `CalibratedClassifierCV(LinearSVC(C=0.5, class_weight="balanced"), cv=3)`. Details in `train.py` and CLAUDE.md.
- Trained on ResumeAtlas (MIT): 10,337 resumes, 42 careers after cleaning and label fixes.
- Skill data for the 7 new careers: ESCO v1.2.1 (European Commission) and O*NET 31.0 (U.S. Department of Labor/ETA, CC BY 4.0), plus a short India list. See `career_profiles.json` for the source of every weight.

## Evaluation sets

| Set | Size | Written by | Used for |
|---|---|---|---|
| ResumeAtlas held-out 20% | 2,068 resumes | real resumes (dataset split, `random_state=42`) | the main list's accuracy |
| `evaluation/validation_set.jsonl` | 60 synthetic resumes | a separate agent, given only the label list | tuning during development |
| `evaluation/india_test_set.jsonl` | 60 synthetic resumes (18 existing careers, 6 for each of the 7 new careers) | a separate agent, given only the label list and variety targets | the final evaluation only, run once |
| `evaluation/india_test_set_v2.jsonl` | 64 synthetic resumes (8 for each of the 7 new careers, 1 each of 8 existing careers) | a new separate agent, given only the label list | the fresher-line rule check only (Phase 5), run once |

**The India test set is frozen.**
- **sha256:** `da9df68141d58cc6be67e5ff32919648812f11fef044a408feb7323d359ec3b3`, rechecked on 2026-11-18 before the run.
- **Committed first:** in `7a38852`, before any synonym, profile or ranking commit. It has not changed since.
- **Never used for tuning:** it was opened only for the single final run.

**The India test set v2 is a separate frozen set** for the "Fresher roles that fit you" rule. It does not replace the set above, and the v1 numbers below are unchanged.
- **sha256:** `28e47b63a22032a67ed77bebca197b36ae7062a4f0f8e69b15a1e0966a1e28e7` (of the copy git stores, LF line endings, like the v1 hash: `git show HEAD:evaluation/india_test_set_v2.jsonl | sha256sum`).
- **Committed first:** alone, before any change to the fresher-line rule.
- **Never used for tuning:** the rule was tuned on the validation set and frozen before v2 was run.

## Results (final run, 2026-11-18)

`main` = the app before this upgrade (`29a94bc`). Branch = `feature/career-data` at `65e3326`. Both use the same `model.pkl` and `train.py`. For the held-out row, one model trained on the other 80% was passed through each branch's own `predict_top_careers`.

| | main | branch |
|---|---|---|
| ResumeAtlas held-out top-1 | 81.8% | 81.8% |
| ResumeAtlas held-out top-3 | 95.2% | 95.2% |
| ResumeAtlas held-out macro-F1 | 0.810 | 0.810 |
| India set, existing careers (18): top-1 | 100% | 100% |
| India set, existing careers (18): top-3 | 100% | 100% |
| India set, new careers (42): right role shown anywhere | 0% (can't be shown) | 35.7% (15) |
| India set, new careers (42): right role in "Fresher roles" | 0% | 35.7% (15) |
| `import app` time (median of 5) | 2.71 s | 2.78 s |
| Memory after startup / after one prediction (median of 5, Windows working set) | 176.4 / 176.6 MB | 176.7 / 177.5 MB |

On the branch, new careers can appear only in the "Fresher roles" line. That is why the two new-career rows are equal.

**Per new career, India set (6 resumes each), branch:**

| New career | Right role shown |
|---|---|
| Data Analyst | 4 / 6 |
| Software Development Engineer (Fresher) | 3 / 6 |
| Technical Support Engineer | 3 / 6 |
| Graduate Engineer Trainee (Electrical) | 3 / 6 |
| Graduate Engineer Trainee (Civil) | 2 / 6 |
| Graduate Engineer Trainee (Mechanical) | 0 / 6 |
| QA/Test Engineer | 0 / 6 |

## Ranking designs that were tried and rejected

The rule was that the ResumeAtlas held-out top-1 and top-3 may each drop at most 2 points from 81.8% / 95.2%. Both alternatives failed, so the main list stays the classifier alone.

| Design | Held-out top-1 | Held-out top-3 | Why it lost accuracy |
|---|---|---|---|
| One 49-career list: 0.45 × classifier + 0.55 × skill share | 75.2% | 86.5% | the skill score pulled existing careers off the classifier's right answer, and new careers beat their parents |
| Shortlist: classifier top 3, a new career takes its parent's place if its skill share is higher (margin 0.0) | 78.9% | 91.4% | a new career came first in 80 results; in 65 its own parent was the right answer |

### Fresher-line rule: "within 0.04 of the parent" (Phase 5, rejected)

Old rule (kept): a new career is shown in "Fresher roles that fit you" when a parent is in the top 3 and its skill share **beats** the parent's. Tried: its share is above 0 and **at least the parent's share minus 0.04**. The main top 3 is not touched either way.

**Cause found on the validation set:** for every new-career validation resume the parent was already in the top 3; every miss was the child's share not beating the parent's. The parents Testing and Mechanical Engineer have short, top-heavy skill profiles (two skills carry about half the weight), so a few matches give them a high share that the longer child profiles can't beat.

**Tuning (validation set only):** 0.04 is the largest tolerance whose fresher-line precision stays within 5 points of the old rule. Hits 22 -> 26, precision 78.6% -> 74.3%. QA/Test Engineer would have needed about 0.18 (precision 64.2%).

**Pass criterion, written before v2 was run:** QA/Test Engineer and Graduate Engineer Trainee (Mechanical) both get more hits than the old rule; no new career gets fewer; precision drops at most 5 points; main top 3 unchanged.

**India test set v2, run once (2026-11-19).** A hit = the resume's own career is among the fresher roles shown.

| Career (8 resumes each) | Old rule | Within 0.04 |
|---|---|---|
| Software Development Engineer (Fresher) | 3 | 4 |
| Data Analyst | 5 | 5 |
| QA/Test Engineer | 1 | 2 |
| Technical Support Engineer | 4 | 5 |
| Graduate Engineer Trainee (Mechanical) | 2 | 2 |
| Graduate Engineer Trainee (Electrical) | 5 | 5 |
| Graduate Engineer Trainee (Civil) | 2 | 3 |
| **All 56** | **22** | **26** |
| Roles shown / precision | 29 / 75.9% | 35 / 74.3% |
| Existing-career resumes (8) with a fresher line | 2 | 2 |

**Result: not shipped.** Graduate Engineer Trainee (Mechanical) did not improve (2 -> 2), so the first condition fails. The other conditions held. These are v2 numbers; the v1 India-set numbers above are a different set and are unchanged.

## Limits

- **The India set is small and synthetic.** With 6 resumes per new career, one resume moves a per-career score by 16.7 points. The 100% on 18 existing-career resumes says little about real resumes.
- **QA/Test Engineer and Graduate Engineer Trainee (Mechanical) were never suggested** (0 of 6 each). On those resumes their skill share never beat their parent's (Testing, Mechanical Engineer), whose short profiles are top-heavy (see the fresher-line rule above). On v2 they were suggested for 1 and 2 of 8.
- **New careers can't reach the top 3.** The classifier has no training resumes for them.
- **The held-out numbers come from the ResumeAtlas text style**, which is pre-cleaned and has no punctuation. Real uploaded resumes may score differently.
- **Memory was measured on Windows** with Python 3.12 and scikit-learn 1.5.1. It is the process working set, not the Render instance.

## Reproduce

```
python evaluation/final/train_heldout.py . <out>/heldout.joblib                     # ~5 min, trains on 80%
git worktree add <main-wt> main
python -I evaluation/final/evaluate.py <code dir> <out>/heldout.joblib evaluation/india_test_set.jsonl career_profiles.json <out>/result.json
python -I evaluation/final/startup.py <code dir>                                    # run 5 times, take the median
python evaluation/check_fresher_rule.py evaluation/india_test_set_v2.jsonl          # fresher-line rule, old vs within 0.04
```

Run `evaluate.py` once with `<code dir>` = the `main` worktree and once with this repo. The India set is meant to be run once per release, not tuned on.

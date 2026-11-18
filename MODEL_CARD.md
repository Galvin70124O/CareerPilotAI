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

**The India test set is frozen.**
- **sha256:** `da9df68141d58cc6be67e5ff32919648812f11fef044a408feb7323d359ec3b3`, rechecked on 2026-11-18 before the run.
- **Committed first:** in `7a38852`, before any synonym, profile or ranking commit. It has not changed since.
- **Never used for tuning:** it was opened only for the single final run.

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

## Limits

- **The India set is small and synthetic.** With 6 resumes per new career, one resume moves a per-career score by 16.7 points. The 100% on 18 existing-career resumes says little about real resumes.
- **QA/Test Engineer and Graduate Engineer Trainee (Mechanical) were never suggested** (0 of 6 each). On those resumes their skill share never beat their parent's (Testing, Mechanical Engineer). Why has not been analysed yet.
- **New careers can't reach the top 3.** The classifier has no training resumes for them.
- **The held-out numbers come from the ResumeAtlas text style**, which is pre-cleaned and has no punctuation. Real uploaded resumes may score differently.
- **Memory was measured on Windows** with Python 3.12 and scikit-learn 1.5.1. It is the process working set, not the Render instance.

## Reproduce

```
python evaluation/final/train_heldout.py . <out>/heldout.joblib                     # ~5 min, trains on 80%
git worktree add <main-wt> main
python -I evaluation/final/evaluate.py <code dir> <out>/heldout.joblib evaluation/india_test_set.jsonl career_profiles.json <out>/result.json
python -I evaluation/final/startup.py <code dir>                                    # run 5 times, take the median
```

Run `evaluate.py` once with `<code dir>` = the `main` worktree and once with this repo. The India set is meant to be run once per release, not tuned on.

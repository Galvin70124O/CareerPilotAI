# CareerPilotAI

A Flask web app that suggests a career path from resume text. Built by a CSE student as a first project. **The goal is to learn while improving it, not just to end up with rewritten code.**

## What it does

1. The user uploads a resume (.txt / .pdf / .docx, max 2 MB) or types/dictates skills.
2. An uploaded resume is analyzed in one step: skills found, Resume Score (ATS-style /100), top-3 careers, skill gap, roadmap, free courses, What-If Simulator, remote jobs (Remotive API).
3. Typed text + **Send** gives the same (minus the Resume Score) for typed skills.

## Files

| File | Purpose |
|---|---|
| `app.py` | Flask app: loads `model.pkl`, reads uploads (pypdf, python-docx), top-3 prediction, Remotive jobs; routes `/` and `/api/simulate` |
| `features/` | One module per feature: `text_cleaning`, `skills`, `resume_score` (weights in `SCORE_WEIGHTS`), `roadmap`, `courses`, `simulator` |
| `roadmaps.json` | 3 stages x 3 steps per career; each step has a `skill` keyword used to tick it |
| `courses.json` | Free resources per skill; only URLs verified with HTTP 200 (2026-10-20). Never add unverified links |
| `templates/partials/` | One Jinja partial per feature card |
| `train.py` | Honest evaluation (held-out test + CV inside a Pipeline), trains final model, saves `model.pkl`. `--quick` skips evaluation |
| `data/prepare_data.py` | Downloads + cleans ResumeAtlas, saves `data/resumes.parquet` (committed, ~20 MB) |
| `skills.json` | 8 key skills per career (43 careers), chosen by how often that category's resumes mention them |
| `templates/index.html` | The only page. Bootstrap + custom dark CSS, browser speech-to-text |
| `tests/` | pytest (`pytest.ini` sets the path) |
| `chatbot.py` | CLI demo, loads `model.pkl` |
| `render.yaml` | Render blueprint: build runs `train.py --quick`, start runs gunicorn on `$PORT` |

## How to run

```
pip install -r requirements.txt
python train.py        # full evaluation (~5 min) + create model.pkl
python app.py          # http://localhost:10000
python -m pytest
```

## Model

- `TfidfVectorizer(preprocessor=normalize_text, stop_words='english', ngram_range=(1, 1), sublinear_tf=True, min_df=2, max_features=50000)`
- `CalibratedClassifierCV(LinearSVC(C=0.5, class_weight='balanced'), cv=3, ensemble=False)`
- Unigrams only because bigrams needed ~930 MB RAM to train (Render free tier: 512 MB). Deploy training peaks at ~434 MB.
- Held-out test: 80.2% accuracy, macro F1 0.80, top-3 93.6%.

## Dataset facts (checked)

- ResumeAtlas (MIT), 13,389 rows, 43 categories → 10,646 after cleaning (Blockchain 35 … Education 363).
- Text is pre-cleaned by the authors: lowercase, punctuation **deleted** ("asp.net" → "aspnet"). `features/text_cleaning.normalize_text` makes user text match.
- The name placeholder "claire" appeared in ~60% of resumes and sometimes replaced letters ("wiclaireh"); removed in `prepare_data.py`.
- Label noise: "Data Science" contains school science teachers, "Advocate" is mostly victim/customer advocates, "Testing" is software testing.

## Rules for working on this project

- **Explain first.** Before each change, say in 2–3 sentences what is changing and why.
- **Keep it simple.** Code should be easy for a beginner to read. No big refactors unless the user asks.
- **Feature logic goes in `features/`**, with pytest tests; keep all tests passing.
- Keep the current dark UI style; pages must work on mobile.
- **After each step/part:** say exactly how to test it, and commit (or suggest a commit message, if not asked to commit). Don't push unless asked.
- Teach as we go: when a change touches an ML or web concept, briefly explain the idea.

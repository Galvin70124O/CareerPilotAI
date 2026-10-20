# CareerPilotAI

A Flask web app that suggests a career path from resume text. Built by a CSE student as a first project. **The goal is to learn while improving it, not just to end up with rewritten code.**

## What it does

1. The user pastes or dictates skills/resume text, or uploads a resume (.txt / .pdf / .docx, max 2 MB).
2. **Extract Skills** reads the uploaded file and finds known skills from a fixed list (`KNOWN_SKILLS`).
3. **Send** predicts a career category from the text box using TF-IDF + LogisticRegression.
4. The predicted category is used to search remote jobs on the Remotive API (top 5).

## Files

| File | Purpose |
|---|---|
| `app.py` | The Flask app: loads `model.pkl` and `skills.json`, reads uploads, whole-word skill extraction, top-3 prediction, skill gap, Remotive job fetch, single `/` route |
| `train.py` | Dedupes data, near-duplicate check, stratified CV of several settings, trains final Pipeline, saves `model.pkl` |
| `skills.json` | Hand-written key skills per career (8 each), checked against `Resume.csv` |
| `templates/index.html` | The only page. Bootstrap + custom dark CSS, browser speech-to-text (Chrome `webkitSpeechRecognition`), mic loudness indicator |
| `tests/` | pytest: skill matching, skill gap, upload/text flows (`pytest.ini` sets the path) |
| `chatbot.py` | Older CLI demo: still trains its own (leaky) model, not yet updated |
| `Resume.csv` | Dataset: columns `Category`, `Resume`. 962 rows, 25 categories |
| `requirements.txt` | Pinned versions |
| `render.yaml` | Render blueprint: build runs `train.py`, start runs gunicorn on `$PORT` |

## How to run

```
pip install -r requirements.txt
python train.py        # evaluate + create model.pkl
python app.py          # serves on http://localhost:10000 (or $PORT)
python -m pytest       # run tests
```

The app is set up for Render (`host=0.0.0.0`, `PORT` env var, gunicorn in requirements).

## Model details

- `TfidfVectorizer(stop_words='english', max_features=10000, ngram_range=(1, 2), sublinear_tf=True)`
- `LogisticRegression(max_iter=3000, class_weight='balanced', C=10)` (C=10 chosen for usable confidence %, C=1 is ~3 points more accurate)
- Trained by `train.py` on the 166 unique resumes; `app.py` only loads `model.pkl`.

## Dataset facts (checked)

- 962 rows but only **166 unique resumes**. Many rows are exact duplicates.
- After removing duplicates, each category has only 3–13 examples.
- Some text has encoding damage (e.g. `NaÃ¯ve` instead of `Naïve`).

## Rules for working on this project

- **Small steps.** Make one change at a time, then stop so the user can run it and commit.
- **Explain first.** Before each change, say in 2–3 sentences what is changing and why.
- **Keep it simple.** Code should be easy for a beginner to read. No big refactors unless the user asks.
- **After each step:** say exactly how to test it, and suggest a git commit message.
- Teach as we go: when a change touches an ML or web concept (data leakage, model saving, word boundaries, etc.), briefly explain the idea.

## Status (2026-10-20)

Roadmap items 1–6 below are done, plus: top-3 predictions with confidence, a Skill Gap Analyzer (`skills.json`), pytest tests in `tests/`, pinned `requirements.txt`, `render.yaml` and `README.md`.
Honest accuracy: 86.1% / macro F1 0.81 (3-fold stratified CV, 166 unique resumes). Run `python train.py` to rebuild `model.pkl` and `python -m pytest` to test.

## Roadmap (known problems, in order)

1. **Safety basics.** `debug=True` is on in `app.py` (unsafe when deployed: it exposes an interactive debugger). There is no `.gitignore`.
2. **Train once, load many times.** The model retrains every time the app starts. Move training into a new `train.py` that saves the vectorizer and model to a file (e.g. with `joblib`), and have `app.py` just load them.
3. **Fix the fake 99% accuracy.** Because of the duplicate rows, the same resume ends up in both the train and test sets (data leakage). Remove duplicates before splitting, and report a realistic accuracy (e.g. with cross-validation, given the small dataset).
4. **Better skill matching.** `extract_skills_from_text` uses `skill in text`, so short skills like `"c"` match almost any text, and `"java"` matches inside `"javascript"`. Use whole-word matching instead (e.g. regex with word boundaries, careful with `c++`).
5. **One-click upload.** Uploading a file only extracts skills; the user then has to click **Send** to get a prediction. Uploading should also predict and show jobs.
6. **Delete `create_csv.py`.** It can overwrite the real `Resume.csv` with dummy data.

## Other things noticed (not on the roadmap yet)

- On upload, only the matched skill list goes into the text box, so predictions run on a short skill list while the model was trained on full resumes.
- `chatbot.py` duplicates the training code from `app.py` (worth sharing via `train.py` after step 2).
- `index.html` clears the results on page reload using `window.onload`.
- `static_folder='static'` is set but there is no `static/` folder (harmless).
- `requirements.txt` has no version numbers.

# CareerPilot AI

A Flask web app that suggests a career path from your resume or skills, shows which key skills you have and which you are missing for that career, and lists live remote jobs.

## Features

- **One-click resume analysis**: upload a `.txt`, `.pdf` or `.docx` resume (max 2 MB). The app extracts your skills and predicts your career in one step.
- **Type or speak your skills**: paste skills into the text box or use the 🎤 button (Chrome speech recognition).
- **Top 3 careers with confidence**: out of 25 career categories.
- **Skill Gap Analyzer**: for the top career, shows the key skills you already have and the ones to learn next (from `skills.json`).
- **Live remote jobs** for the predicted career from the [Remotive API](https://remotive.com/api/remote-jobs).
- **Whole-word skill matching**: `c` doesn't match "science", `git` doesn't match "GitHub", `c++` / `c#` are not counted as `c`.

## How it works

| Part | How |
|---|---|
| Model | TF-IDF (1–2 word phrases, `sublinear_tf`) + Logistic Regression (`class_weight="balanced"`, `C=10`) in a scikit-learn `Pipeline` |
| Training | `train.py` trains once and saves `model.pkl`. `app.py` only loads it |
| Data | `Resume.csv`: 962 rows, but only **166 unique resumes** across 25 categories |
| Skill extraction | Regex with boundaries against a fixed skill list |

## Model accuracy (honest numbers)

The original version reported **99.5%** accuracy. That number was not real: `Resume.csv` contains each resume about 6 times, so the same resume ended up in both the training and the test set (**data leakage**). The model was being tested on resumes it had already memorised.

`train.py` now removes the duplicates, checks for near-duplicates (none above 0.9 cosine similarity), and uses **3-fold stratified cross-validation** inside a `Pipeline` so the vectorizer only learns from the training folds.

| Settings | Accuracy | Macro F1 |
|---|---|---|
| Original settings | 45.8% | 0.31 |
| + `class_weight="balanced"` | 85.5% | 0.81 |
| + 1–2 grams + `sublinear_tf` (C=1) | 89.2% | 0.86 |
| **Final (same, C=10)** | **86.1%** | **0.81** |

The final model uses `C=10` because it gives meaningful confidence percentages (C=1 gave ~8% for the top career). With only 166 resumes, these numbers move by a few points depending on how the folds are split.

**Limitations:** only 3–13 resumes per category, so treat predictions as suggestions. Some categories (e.g. Web Designing) are often confused with others. Note: "Testing" in this dataset means *electrical equipment testing*, not software testing (that's "Automation Testing").

## Run it locally (Windows PowerShell)

```powershell
git clone <your-repo-url>
cd CareerPilotAI
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

python train.py        # prints the evaluation and creates model.pkl (~30 s)
python app.py          # open http://localhost:10000
```

Debug mode is off by default. To turn it on locally: `$env:FLASK_DEBUG="1"; python app.py`

Run the tests:

```powershell
python -m pytest
```

## Deploy to Render

The repo includes `render.yaml`. On Render: **New → Blueprint → pick this repo**. It will:

- build: `pip install -r requirements.txt && python train.py` (creates `model.pkl`)
- start: `gunicorn app:app --bind 0.0.0.0:$PORT`

## Project files

| File | Purpose |
|---|---|
| `app.py` | Flask app: file reading, skill extraction, prediction, skill gap, jobs |
| `train.py` | Evaluation + training, saves `model.pkl` |
| `skills.json` | Key skills for each career (used by the Skill Gap Analyzer) |
| `templates/index.html` | The web page |
| `tests/` | pytest tests for skill matching, skill gap and the upload/text flows |
| `chatbot.py` | Older command-line demo (still trains its own model) |
| `render.yaml` | Render deployment settings |

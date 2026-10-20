# CareerPilot AI

A Flask web app that reads your resume (or the skills you type), predicts the career it fits best, scores the resume like an ATS would, and shows what to learn next, with free resources and a "what if" simulator.

## Features

| Feature | What it does |
|---|---|
| **One-click resume analysis** | Upload a `.txt`, `.pdf` or `.docx` (max 2 MB). Skills, prediction, score and everything below appear right away. |
| **Top 3 careers** | 42 career categories, with confidence percentages. Works for uploaded resumes and for typed or spoken skills (🎤, Chrome). |
| **Resume Score** | ATS-style score out of 100 with a breakdown (sections, length & readability, action verbs, measurable impact, skill match, red flags), a ✓/✗ checklist and 3–5 tips that quote your resume. Weights live in `SCORE_WEIGHTS` in `features/resume_score.py`. |
| **Skill Gap Analyzer** | The 8 key skills of your top career (`skills.json`), split into "you have" and "to learn next". |
| **Career Roadmap** | Beginner → Intermediate → Job-ready steps for your top career (`roadmaps.json`). Steps you already cover are ticked; missing ones come first. |
| **Course Recommender** | 1–2 free resources per missing skill (official docs, Kaggle Learn, freeCodeCamp, Microsoft Learn, NPTEL, Khan Academy…). Every link in `courses.json` was checked to return HTTP 200 on 2026-10-20; skills without one get a YouTube search link. |
| **What-If Simulator** | Tick missing skills, click Simulate, and see before → after for the top 3 careers, confidence, skill gap and resume score. |
| **Live remote jobs** | Jobs for your top career from the [Remotive API](https://remotive.com/api/remote-jobs). |

## Dataset

**ResumeAtlas** by Heakl et al. ([Hugging Face](https://huggingface.co/datasets/ahmedheakl/resume-atlas), [paper](https://arxiv.org/abs/2406.18125), [code](https://github.com/noran-mohamed/Resume-Classification-Dataset)), licensed **MIT** according to its dataset card. All credit for the data goes to the authors.

`data/prepare_data.py` downloads a pinned revision and cleans it:

| Step | Resumes left |
|---|---|
| Downloaded | 13,389 (43 categories) |
| Drop resumes with >5% words broken by the "claire" name placeholder | 13,332 |
| Drop resumes under 50 words | 13,312 |
| Drop texts that appear with two different categories | 12,933 |
| Drop exact duplicates | 11,847 |
| Drop near-duplicates (cosine similarity > 0.9) | 10,646 |
| Label fixes: move 86 "Data Science" science-teacher resumes (no data tools) to Education; drop "Management" (309) | **10,337** (42 categories) |

Smallest category: Blockchain (35). Largest: Education (449). The result is saved as `data/resumes.parquet` (20 MB, not committed: `train.py` rebuilds it automatically when it's missing).

## Model and honest accuracy

TF-IDF on single words + a calibrated LinearSVC, in one scikit-learn `Pipeline`. The same `normalize_text()` cleans both the training data and your resume, so they look alike to the model.

Evaluation (`python train.py`): 20% of resumes are locked away as a test set. Models are compared with 3-fold stratified cross-validation on the other 80%, with the whole Pipeline re-fit inside every fold (no leakage).

| Model (after label fixes, 42 careers) | CV accuracy | CV macro F1 | CV top-3 accuracy |
|---|---|---|---|
| TF-IDF + LogisticRegression | 79.5% | 0.794 | 93.1% |
| **TF-IDF + calibrated LinearSVC** | **80.5%** | **0.805** | **94.2%** |

Chosen model on the held-out test set: **81.8% accuracy, macro F1 0.810, top-3 accuracy 95.2%**.

Weakest categories (test F1): Blockchain 0.67, React Developer 0.69, Consultant 0.69, BPO 0.70, Web Designing 0.72.

Before the label fixes (43 careers) the same model scored 80.2% / 0.799 / 93.6% on its test set, with Management (F1 0.35) as the weakest category. The two test sets are not identical (labels changed and Management was removed), so the comparison is indicative, not exact.

**Limitations**
- **Label noise we fixed** (rules in `data/prepare_data.py`, `fix_labels`):
  - "Data Science" contained many school science teachers. A Data Science resume that mentions teaching (teacher, classroom, grade, lesson, students) and **none** of the data tools (python, sql, statistics, R programming, tableau, power bi, data analysis, spark, pandas, machine learning, …) is moved to Education. 86 moved; a random sample of 10 was checked by hand and all were teachers/instructors.
  - "Management" was dropped: it's a job level found in every industry, not a career path, and the model only got 26% of them right.
  - "Advocate" mixes lawyers (~28%) and victim/patient/customer advocates (~46%). The labels are kept (the model gets 88% right), but the page shows it as **"Advocate (Legal & Social Services)"** (`features/careers.py`).
- Using word pairs (bigrams) was about 1–2 points more accurate, but training needed ~930 MB RAM, more than Render's free tier allows.
- The old version claimed 99.5% on a small dataset where every resume appeared ~6 times. That was data leakage.

## Run it locally (Windows PowerShell)

```powershell
git clone <your-repo-url>
cd CareerPilotAI
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# model.pkl is already in the repo, so you can skip training.
# To retrain (downloads the dataset the first time):
python train.py           # full evaluation (~4 min) + creates model.pkl
# or: python train.py --quick   (no evaluation, ~1 min)
python app.py             # open http://localhost:10000
python -m pytest          # run the tests
python chatbot.py         # command-line version
```

Debug mode is off by default. To turn it on locally: `$env:FLASK_DEBUG="1"; python app.py`

To rebuild the dataset from Hugging Face: `python data/prepare_data.py`

## Deploy to Render

`render.yaml` is a Render Blueprint. On Render: **New → Blueprint → pick this repo**.

- Build: `pip install -r requirements.txt` only. Render does **not** train; it uses the committed `model.pkl` (10 MB).
- Start: `gunicorn app:app --bind 0.0.0.0:$PORT`
- Python is pinned to 3.12.6 (`PYTHON_VERSION` in `render.yaml` and `.python-version`), and `requirements.txt` pins scikit-learn 1.5.1, the same versions that trained `model.pkl`. At startup the app logs `Model trained with scikit-learn X, installed scikit-learn Y` and warns if they differ.
- After retraining locally, commit the new `model.pkl`.

## Project files

| Path | Purpose |
|---|---|
| `app.py` | Flask routes: `/` (the page) and `/api/simulate` (What-If Simulator) |
| `features/` | One module per feature: `text_cleaning`, `skills`, `resume_score`, `roadmap`, `courses`, `simulator` |
| `train.py` | Evaluation + training, saves `model.pkl` |
| `data/prepare_data.py` | Downloads and cleans ResumeAtlas |
| `skills.json`, `roadmaps.json`, `courses.json` | Key skills, roadmaps and free courses per career |
| `templates/` | `index.html` + one partial per feature card |
| `tests/` | pytest tests |
| `chatbot.py` | Command-line version |
| `render.yaml` | Render deployment |

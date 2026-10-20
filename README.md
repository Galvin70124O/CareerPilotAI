# CareerPilot AI

A Flask web app that reads your resume (or the skills you type), predicts the career it fits best, scores the resume like an ATS would, and shows what to learn next, with free resources and a "what if" simulator.

## Features

| Feature | What it does |
|---|---|
| **One-click resume analysis** | Upload a `.txt`, `.pdf` or `.docx` (max 2 MB). Skills, prediction, score and everything below appear right away. |
| **Top 3 careers** | 43 career categories, with confidence percentages. Works for uploaded resumes and for typed or spoken skills (🎤, Chrome). |
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
| Drop near-duplicates (cosine similarity > 0.9) | **10,646** |

Smallest category: Blockchain (35). Largest: Education (363). The result is saved as `data/resumes.parquet` (20 MB, not committed: `train.py` rebuilds it automatically when it's missing).

## Model and honest accuracy

TF-IDF on single words + a calibrated LinearSVC, in one scikit-learn `Pipeline`. The same `normalize_text()` cleans both the training data and your resume, so they look alike to the model.

Evaluation (`python train.py`): 20% of resumes are locked away as a test set. Models are compared with 3-fold stratified cross-validation on the other 80%, with the whole Pipeline re-fit inside every fold (no leakage).

| Model | CV accuracy | CV macro F1 | CV top-3 accuracy |
|---|---|---|---|
| TF-IDF + LogisticRegression | 77.8% | 0.778 | 92.1% |
| **TF-IDF + calibrated LinearSVC** | **78.9%** | **0.789** | **92.8%** |

Chosen model on the held-out test set: **80.2% accuracy, macro F1 0.799, top-3 accuracy 93.6%**.

Weakest categories (test F1): Management 0.35, BPO 0.65, Operations Manager 0.65, Consultant 0.67, Database 0.68. These overlap heavily with other categories.

**Limitations**
- Some labels are noisy. "Data Science" includes school science teachers, and "Advocate" is mostly victim/customer advocates rather than lawyers.
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

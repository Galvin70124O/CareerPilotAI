# CareerPilotAI

A Flask web app that suggests a career path from resume text. Built by a CSE student as a first project. **The goal is to learn while improving it, not just to end up with rewritten code.**

## What it does

1. The user uploads a resume (.txt / .pdf / .docx, max 2 MB) or types/dictates skills.
2. An uploaded resume is analyzed in one step: skills found, Resume Score (ATS-style /100), top-3 careers, skill gap, roadmap, free courses, What-If Simulator, remote jobs (Remotive API).
3. Typed text + **Send** gives the same (minus the Resume Score) for typed skills.

## Files

| File | Purpose |
|---|---|
| `app.py` | Flask app: loads `model.pkl`, reads uploads (pypdf, python-docx), top-3 prediction; routes `/`, `/api/simulate`, `/health` |
| `features/` | One module per feature: `text_cleaning`, `skills`, `resume_score` (weights in `SCORE_WEIGHTS`), `roadmap`, `courses`, `simulator`, `jobs` (Remotive: one shared feed, 4 s timeout, 6-hour cache (Remotive terms: max ~4 calls/day), `None` = API failed; filters titles by `job_keywords`, fills with `related` jobs if < 3 match; page must credit Remotive), `careers` (display names) |
| `roadmaps.json` | 3 stages x 3 steps per career; each step has a `skill` keyword used to tick it |
| `courses.json` | Free resources per skill; only URLs verified with HTTP 200 (2026-10-20). Never add unverified links |
| `templates/partials/` | One Jinja partial per section/card; `macros.html` has `icon()` (Lucide SVGs) and `score_ring()` |
| `static/css/theme.css` | Design tokens ("printed career report": paper, ink, one blue accent): CSS variables for colors, type, spacing, radius, light + dark (`data-theme` + `prefers-color-scheme`). The only file with colour values |
| `static/css/app.css` | Component and layout styles; uses only the variables from `theme.css` |
| `DESIGN.md` | The design system in words: tokens, type and radius scales, the accent rule, components, do's and don'ts. Keep it in sync with `theme.css` |
| `static/img/` + `tools/make_brand_assets.py` | Logo, favicons, OG image (rebuild with the script, don't hand-edit the PNGs) |
| `features/report.py` | PDF report (fpdf2, core Helvetica: text goes through `latin1()`); route `POST /report` in app.py |
| `data/sample_resume.txt` | Fictional sample resume for "Try with a sample resume" (`sample=1` form value) |
| `static/js/app.js` | All page JS: theme toggle (localStorage), ARIA tabs, drop zone, loading skeleton, mic, smooth scroll, simulator fetch |
| `train.py` | Honest evaluation (held-out test + CV inside a Pipeline), trains final model, saves `model.pkl`. `--quick` skips evaluation |
| `data/prepare_data.py` | Downloads + cleans ResumeAtlas, saves `data/resumes.parquet` (~20 MB, git-ignored; `train.py` rebuilds it if missing) |
| `data/fetch_sources.py` | Downloads O*NET 31.0 + ESCO v1.2.1 (API, occupations pinned by URI) into `data/raw/` (git-ignored); caches each ESCO response, retries 2/4/8 s, writes `data/raw/esco/fetch_log.json` |
| `data/build_synonyms.py` + `skill_synonyms.json` | Synonyms per skill, kept by source (`esco`, `onet`, `india`). Rebuild with the script, never hand-edit the JSON. `has_skill` accepts them; R, Go and B.E./B.Tech use context rules in `SPECIAL_SKILL_PATTERNS` instead |
| `data/build_profiles.py` + `career_profiles.json` | Weighted key skills for all 49 careers, each with its evidence. New careers: ESCO essential 1.0 / optional 0.6 + O*NET hot 0.5 / listed 0.25 (else India-list 0.5; DSA for SDE (Fresher) 1.0, source `india-hiring`, approved by the owner), plus `parents` (the classifier careers they grow out of). Existing: share of TRAINING-split resumes + O*NET bonus. Rebuild with the script; it stops on a made-up ESCO citation |
| `features/ranking.py` | Career ranking. Top 3 = the classifier alone (its own probability as the percent, strong/moderate/weak band, "Matched because" = top weighted matched key skills). "Fresher roles that fit you" (`fresher_roles`, max 2, no band): a new career whose parent is in the top 3 and whose skill score (share of its key-skill weight matched) beats that parent's. Kept apart because mixing them cost 6.6 points of held-out top-1. `app.predict_top_careers` calls it |
| `evaluation/validation_set.jsonl` + `evaluation/check_ranking.py` | Synthetic 60-resume VALIDATION set (same career mix as the India set). `python evaluation/check_ranking.py`: top-1/top-3 for existing careers, and how often a new career shows in "Fresher roles that fit you" |
| `MODEL_CARD.md` + `evaluation/final/` | Final evaluation (run once per release): main vs branch on the ResumeAtlas held-out 20% and the India set, load time and memory, rejected ranking designs, limits. The scripts in `evaluation/final/` reproduce it |
| `evaluation/india_test_set.jsonl` | FROZEN 60-resume India test set (sha256 in MODEL_CARD.md). Never edit, never tune on it |
| `model.pkl` | Committed trained model (10 MB). Retrain locally and commit it; Render doesn't train |
| `skills.json` | Per career: `skills` (8 key skills; the 42 classifier careers: chosen by how often that category's resumes mention them; the 7 new careers: the top 8 by weight in `career_profiles.json`) and `job_keywords` (job-title words). Loaded only by `features/skills.py` (`CAREER_SKILLS`, `JOB_KEYWORDS`) |
| `templates/index.html` | Page layout (head, top bar, hero, results/preview, footer) |
| `tests/` | pytest (`pytest.ini` sets the path) |
| `chatbot.py` | CLI demo, loads `model.pkl` |
| `render.yaml` | Render blueprint: build is only `pip install`, start runs gunicorn on `$PORT`; Python pinned to 3.12.6 (also `.python-version`) to match the training machine |

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
- Unigrams only because bigrams needed ~930 MB RAM to train. Training peaks at ~434 MB (local only now).
- `train.py` stores `model.trained_with_sklearn`; `app.py` logs it vs the installed version at startup.
- Held-out test (42 careers, after label fixes): 81.8% accuracy, macro F1 0.81, top-3 95.2%.

## Dataset facts (checked)

- ResumeAtlas (MIT), 13,389 rows, 43 categories → 10,337 resumes, 42 categories after cleaning + label fixes (Blockchain 35 … Education 449).
- Text is pre-cleaned by the authors: lowercase, punctuation **deleted** ("asp.net" → "aspnet"). `features/text_cleaning.normalize_text` makes user text match.
- The name placeholder "claire" appeared in ~60% of resumes and sometimes replaced letters ("wiclaireh"); removed in `prepare_data.py`.
- Label fixes in `prepare_data.fix_labels`: Data Science science teachers without data tools -> Education (86); "Management" dropped.
- "Advocate" keeps its label but is shown as "Advocate (Legal & Social Services)" via `features/careers.display_name`. Use `display_name()` / the `display_name` Jinja filter for anything user-facing; keep the model label for lookups and job search.
- "Testing" is software testing.

## Rules for working on this project

- **Explain first.** Before each change, say in 2–3 sentences what is changing and why.
- **Keep it simple.** Code should be easy for a beginner to read. No big refactors unless the user asks.
- **Feature logic goes in `features/`**, with pytest tests; keep all tests passing.
- UI: use the tokens in `static/css/theme.css` (printed-report style: warm paper, black ink, Newsreader for headings and figures; the one Prussian-blue accent is ONLY for the score figure, the career-match bars and the confidence figure; buttons, links and focus are ink; green/amber/red only for score levels; radius scale 0/4/8/12, no pills; no shadows except the tooltip; no icon tiles or eyebrow labels). No inline styles, no inline JS handlers, no emoji icons (use `icon()`), no Bootstrap. Bars are `<progress>`; must work at 390 px and 1440 px, light and dark. `tests/test_ui.py` checks these rules.
- **After each step/part:** say exactly how to test it, and commit (or suggest a commit message, if not asked to commit). Don't push unless asked.
- Teach as we go: when a change touches an ML or web concept, briefly explain the idea.

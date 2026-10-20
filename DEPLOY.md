# Deploying CareerPilot AI on Render

The repo has a `render.yaml` Blueprint, so Render reads all settings from it:

| Setting | Value |
|---|---|
| Build command | `pip install -r requirements.txt` (no training: `model.pkl` is in the repo) |
| Start command | `gunicorn app:app --bind 0.0.0.0:$PORT --timeout 60` |
| Python | 3.12.6 (`PYTHON_VERSION` in `render.yaml`, also `.python-version`) |
| Health check | `GET /health` → `200 ok` |
| Plan | Free |

## Steps

1. Log in to https://dashboard.render.com (sign in with GitHub).
2. Click **New** → **Blueprint**.
3. Choose the **CareerPilotAI** repository (click *Configure account* first if it isn't listed, and give Render access to the repo).
4. Render shows the `careerpilotai` web service from `render.yaml`. Click **Apply** / **Deploy Blueprint**.
5. Open the service → **Logs** and wait for the build and start. Look for:
   - `Successfully installed ... scikit-learn-1.5.1 ...` (build)
   - `Listening at: http://0.0.0.0:10000` (gunicorn started)
   - `INFO careerpilot: Model trained with scikit-learn 1.5.1, installed scikit-learn 1.5.1` ✅

   If you instead see `WARNING careerpilot: scikit-learn version mismatch!`, check that `requirements.txt` still pins `scikit-learn==1.5.1`.
   If you see `Model file not found`, check that `model.pkl` is committed (`git ls-files model.pkl`).

## Test the live URL

Your URL is shown at the top of the service page, like `https://careerpilotai-xxxx.onrender.com`.

1. `https://<your-url>/health` → shows `ok`.
2. Open `https://<your-url>/` and upload `tests/fixtures/good_resume.txt`: you should see the Resume Score (about 92), top-3 careers, skill gap, roadmap, free courses and the What-If Simulator.
3. In the simulator, tick a skill and click **Simulate**: a before → after table appears.
4. Type `java spring hibernate` in the text box and click **Send**: Java Developer.

## Troubleshooting

**The first load is slow (30-60 s).** Normal on the free plan: the service sleeps after ~15 minutes without visits and has to wake up. Later requests are fast. (A paid instance doesn't sleep.)

**"Ran out of memory (used over 512MB)".** The running app uses about 170 MB, so this should not happen with the default settings. If it does:
- Make sure the build command is only `pip install -r requirements.txt` (training on Render needs ~450 MB).
- Keep gunicorn at 1 worker (the default; don't add `--workers 4`).
- Otherwise move to a paid instance with more RAM.

**"Live jobs are unavailable right now".** The jobs come from the Remotive API (`https://remotive.com/api/remote-jobs`). If it is down or takes longer than 4 seconds, the rest of the page still works and the next upload tries again. The Remotive feed is downloaded once and cached for 6 hours for all careers (at most 4 calls a day, as Remotive's terms advise), so new jobs can take up to 6 hours to appear. Remotive's listings are also delayed by 24 hours.

**Many jobs are marked "Related".** Remotive's free API returns the same ~10 jobs for every search, so for most careers few titles match the career's `job_keywords`.

## Updating the app

Push to `main` and Render redeploys automatically. After retraining the model (`python train.py`), commit the new `model.pkl` too, otherwise Render keeps serving the old model.

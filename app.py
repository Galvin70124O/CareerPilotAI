from flask import Flask, request, render_template
import joblib
import os
from werkzeug.utils import secure_filename
from docx import Document
import PyPDF2
import requests

from features.courses import recommend_courses
from features.resume_score import score_resume
from features.roadmap import build_roadmap
from features.skills import analyze_skill_gap, extract_skills_from_text

# -----------------------------
# CONFIGURATION
# -----------------------------
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'}
MAX_CONTENT_LENGTH = 2 * 1024 * 1024  # 2MB

# -----------------------------
# FLASK SETUP
# -----------------------------
app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# -----------------------------
# FILE HANDLING
# -----------------------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text_from_resume(file_storage):
    filename = secure_filename(file_storage.filename)
    ext = os.path.splitext(filename)[1].lower()
    text = ""
    if ext == ".txt":
        text = file_storage.read().decode('utf-8', errors='ignore')
    elif ext == ".pdf":
        pdf_reader = PyPDF2.PdfReader(file_storage)
        for page in pdf_reader.pages:
            if page.extract_text():
                text += page.extract_text() + "\n"
    elif ext == ".docx":
        file_storage.seek(0)
        document = Document(file_storage)
        for para in document.paragraphs:
            text += para.text + "\n"
    return text.strip()


def fetch_job_listings(job_title, limit=5):
    try:
        response = requests.get(
            "https://remotive.io/api/remote-jobs",
            params={"search": job_title, "limit": str(limit)},
            timeout=10
        )
        jobs = []
        if response.ok:
            job_data = response.json()
            for job in job_data.get("jobs", [])[:limit]:
                jobs.append({
                    "title": job["title"],
                    "company": job["company_name"],
                    "url": job["url"],
                    "location": job["candidate_required_location"]
                })
        return jobs
    except Exception:
        return []

# -----------------------------
# MODEL LOADING
# -----------------------------
# The model is trained by train.py, not here. The app only loads it.
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pkl")
MODEL_MISSING_MESSAGE = "Model file not found. Run 'python train.py' first, then restart the app."

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    print("⚠️", MODEL_MISSING_MESSAGE)
    model = None


def predict_top_careers(input_text, top_n=3):
    # predict_proba gives one probability per career, in the order of model.classes_
    probabilities = model.predict_proba([input_text])[0]
    ranked = sorted(zip(model.classes_, probabilities), key=lambda pair: pair[1], reverse=True)
    return [{"career": career, "percent": round(prob * 100)} for career, prob in ranked[:top_n]]

# -----------------------------
# ROUTES
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    input_text = ""      # what the user typed (shown again in the text box)
    found_skills = ""    # skills found in the uploaded resume
    top_careers = []     # [{"career": ..., "percent": ...}, ...]
    job_results = []
    skill_gap = None     # result of analyze_skill_gap for the top career
    resume_score = None  # result of score_resume (only for uploaded resumes)
    roadmap = None       # result of build_roadmap for the top career
    courses = []         # free resources for each missing skill
    error = ""
    source = ""          # "resume" or "text", so the page can say what we predicted from

    if request.method == "POST":
        file = request.files.get("resume_file")
        input_text = request.form.get("input_text", "").strip()
        text_to_predict = ""

        if model is None:
            error = MODEL_MISSING_MESSAGE

        # 1) A resume was uploaded: extract skills AND predict in one step
        elif file and file.filename:
            if not allowed_file(file.filename):
                error = "Only .txt, .pdf, and .docx files up to 2MB allowed."
            else:
                try:
                    resume_text = extract_text_from_resume(file)
                    if not resume_text:
                        error = "Could not read any text from this file (is it a scanned image?)."
                    else:
                        found_skills = extract_skills_from_text(resume_text)
                        # Predict from the FULL resume, the same kind of text the model was trained on
                        text_to_predict = resume_text
                        source = "resume"
                except Exception as e:
                    error = "Error reading file: " + str(e)

        # 2) No file: predict from the typed/spoken text
        elif input_text:
            text_to_predict = input_text
            source = "text"

        else:
            error = "Please upload a resume or type some skills first."

        if text_to_predict:
            top_careers = predict_top_careers(text_to_predict)
            skill_gap = analyze_skill_gap(top_careers[0]["career"], text_to_predict)
            roadmap = build_roadmap(top_careers[0]["career"], text_to_predict)
            courses = recommend_courses(skill_gap["missing"])
            job_results = fetch_job_listings(top_careers[0]["career"])
            if source == "resume":
                # The score needs the RAW resume (line breaks, bullets, emails), not cleaned text
                resume_score = score_resume(text_to_predict, top_careers[0]["career"])

    return render_template(
        "index.html",
        input_text=input_text,
        found_skills=found_skills,
        top_careers=top_careers,
        job_results=job_results,
        skill_gap=skill_gap,
        resume_score=resume_score,
        roadmap=roadmap,
        courses=courses,
        error=error,
        source=source
    )

# -----------------------------
# APP START
# -----------------------------
if __name__ == "__main__":
    # Use host=0.0.0.0 for Render or any cloud
    # Debug mode is OFF unless you set FLASK_DEBUG=1 (never turn it on in production)
    debug_mode = os.environ.get("FLASK_DEBUG") == "1"
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 10000)), debug=debug_mode)

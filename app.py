from flask import Flask, request, render_template
import joblib
import json
import os
import re
from werkzeug.utils import secure_filename
from docx import Document
import PyPDF2
import requests

# -----------------------------
# CONFIGURATION
# -----------------------------
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'}
MAX_CONTENT_LENGTH = 2 * 1024 * 1024  # 2MB

KNOWN_SKILLS = [
    "python", "java", "c", "c++", "c#", "html", "css", "javascript", "typescript", "sql", "mysql", "oracle", "mongodb",
    "data science", "machine learning", "deep learning", "artificial intelligence", "computer vision", "nlp",
    "cloud computing", "azure", "aws", "google cloud", "docker", "kubernetes", "git", "github", "embedded systems",
    "arduino", "raspberry pi", "networking", "linux", "windows", "cybersecurity", "blockchain", "hardware design",
    "soldering", "microcontroller programming", "excel", "powerpoint", "autocad", "matlab", "labview", "testing",
    "automation", "robotics", "communication", "teamwork", "leadership", "problem solving", "critical thinking",
    "creative thinking", "adaptability", "time management", "collaboration", "negotiation", "presentation",
    "active listening", "decision making", "conflict resolution", "public speaking", "research", "self-motivation",
    "organization", "project management", "interpersonal skills"
]

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


def skill_pattern(skill):
    # re.escape makes "+", "#" and "." literal characters instead of regex symbols.
    # Spaces become \s+ so "machine learning" also matches "machine\nlearning".
    escaped = re.escape(skill).replace(r"\ ", r"\s+")
    # (?<!...) = "not preceded by", (?!...) = "not followed by".
    # A skill must not touch a letter, digit, "_", "+" or "#", so:
    #   "c" does not match inside "c++", "c#" or "science"
    #   "git" does not match inside "github"
    return re.compile(r"(?<![\w+#])" + escaped + r"(?![\w+#])")


# Build every regex once when the app starts, not on every request
SKILL_PATTERNS = {skill: skill_pattern(skill) for skill in KNOWN_SKILLS}


def extract_skills_from_text(text):
    skills_found = []
    text_lower = text.lower()
    for skill, pattern in SKILL_PATTERNS.items():
        if pattern.search(text_lower):
            skills_found.append(skill)
    return ", ".join(skills_found)


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


# -----------------------------
# SKILL GAP ANALYZER
# -----------------------------
# skills.json lists the key skills for each career (hand-written, checked against Resume.csv)
SKILLS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skills.json")
with open(SKILLS_PATH, encoding="utf-8") as f:
    CAREER_SKILLS = json.load(f)


def analyze_skill_gap(career, text):
    key_skills = CAREER_SKILLS.get(career, [])
    text_lower = text.lower()
    have = [skill for skill in key_skills if skill_pattern(skill).search(text_lower)]
    missing = [skill for skill in key_skills if skill not in have]
    percent = round(len(have) / len(key_skills) * 100) if key_skills else 0
    return {"career": career, "have": have, "missing": missing, "total": len(key_skills), "percent": percent}


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
            job_results = fetch_job_listings(top_careers[0]["career"])

    return render_template(
        "index.html",
        input_text=input_text,
        found_skills=found_skills,
        top_careers=top_careers,
        job_results=job_results,
        skill_gap=skill_gap,
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

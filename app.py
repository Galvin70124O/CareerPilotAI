from flask import Flask, request, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import os
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
    "python", "java", "c", "c++", "html", "css", "javascript", "typescript", "sql", "mysql", "oracle", "mongodb",
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


def extract_skills_from_text(text):
    skills_found = []
    text_lower = text.lower()
    for skill in KNOWN_SKILLS:
        if skill in text_lower and skill not in skills_found:
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
# MODEL LOADING / TRAINING
# -----------------------------
try:
    data = pd.read_csv('Resume.csv')
    if 'Resume' in data.columns and 'Category' in data.columns:
        x = data['Resume']
        y = data['Category']

        vectorizer = TfidfVectorizer(stop_words='english', max_features=10000, ngram_range=(1, 3))
        x_vec = vectorizer.fit_transform(x)

        clf = LogisticRegression(max_iter=1500)
        clf.fit(x_vec, y)
    else:
        raise ValueError("Resume.csv must contain 'Resume' and 'Category' columns.")
except Exception as e:
    print("⚠️ Model training error:", e)
    clf, vectorizer = None, None


def predict_category(input_text):
    if not clf or not vectorizer:
        return "Model not available"
    vec = vectorizer.transform([input_text])
    pred = clf.predict(vec)[0]
    return pred

# -----------------------------
# ROUTES
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""
    skills_text = ""
    file_error = ""
    job_results = []

    if request.method == "POST":
        file = request.files.get("resume_file")
        text_area = request.form.get("input_text", "").strip()
        action = request.form.get("action", "extract")

        if file and file.filename and action == "extract":
            if not allowed_file(file.filename):
                file_error = "Only .txt, .pdf, and .docx files up to 2MB allowed."
            else:
                try:
                    resume_text = extract_text_from_resume(file)
                    skills_text = extract_skills_from_text(resume_text)
                    if not skills_text.strip():
                        file_error = "Could not find known skills in the file."
                except Exception as e:
                    file_error = "Error reading file: " + str(e)
        elif text_area and action == "predict":
            skills_text = text_area
            prediction = predict_category(skills_text)
            job_results = fetch_job_listings(prediction)

    return render_template(
        "index.html",
        prediction=prediction,
        skills_text=skills_text,
        file_error=file_error,
        job_results=job_results
    )

# -----------------------------
# APP START
# -----------------------------
if __name__ == "__main__":
    # Use host=0.0.0.0 for Render or any cloud
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 10000)), debug=True)

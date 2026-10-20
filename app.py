from flask import Flask, Response, jsonify, request, render_template
import joblib
import logging
import os
import sklearn
from werkzeug.utils import secure_filename
from docx import Document
from pypdf import PdfReader

from features.careers import display_name
from features.courses import recommend_courses
from features.explain import top_contributing_terms
from features.report import build_report_pdf
from features.jobs import fetch_job_listings
from features.resume_score import score_resume
from features.roadmap import build_roadmap
from features.simulator import simulate
from features.skills import CAREER_SKILLS, analyze_skill_gap, display_skill, extract_skills_from_text

# -----------------------------
# CONFIGURATION
# -----------------------------
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'}
MAX_CONTENT_LENGTH = 2 * 1024 * 1024  # 2MB
MAX_SIMULATE_TEXT = 100_000           # characters accepted by /api/simulate and /report
# "Try with a sample resume" analyzes this built-in, fictional resume
SAMPLE_RESUME_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sample_resume.txt")
SAMPLE_RESUME_NAME = "Sample resume (Alex Morgan)"

# -----------------------------
# FLASK SETUP
# -----------------------------
app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
# Lets templates write {{ career|display_name }} to show the user-facing career name
app.add_template_filter(display_name)
# {{ skill|display_skill }} shows "R" for "r", "Node.js" for "node.js", ...
app.add_template_filter(display_skill)

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
        pdf_reader = PdfReader(file_storage)
        for page in pdf_reader.pages:
            if page.extract_text():
                text += page.extract_text() + "\n"
    elif ext == ".docx":
        file_storage.seek(0)
        document = Document(file_storage)
        for para in document.paragraphs:
            text += para.text + "\n"
    return text.strip()


# -----------------------------
# MODEL LOADING
# -----------------------------
# The model is trained by train.py, not here. The app only loads it.
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pkl")
MODEL_MISSING_MESSAGE = "Model file not found. Run 'python train.py' first, then restart the app."

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("careerpilot")

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    # model.pkl is committed and loaded as-is on Render, so the scikit-learn versions must match
    trained_with = getattr(model, "trained_with_sklearn", "unknown")
    logger.info("Model trained with scikit-learn %s, installed scikit-learn %s", trained_with, sklearn.__version__)
    if trained_with != sklearn.__version__:
        logger.warning("scikit-learn version mismatch! Install scikit-learn==%s or retrain with: python train.py",
                       trained_with)
else:
    logger.warning(MODEL_MISSING_MESSAGE)
    model = None


def predict_top_careers(input_text, top_n=3):
    # predict_proba gives one probability per career, in the order of model.classes_
    probabilities = model.predict_proba([input_text])[0]
    ranked = sorted(zip(model.classes_, probabilities), key=lambda pair: pair[1], reverse=True)
    # "career" is the model's label (used for lookups and job search), "label" is what the user sees
    return [{"career": career, "label": display_name(career), "percent": round(prob * 100)}
            for career, prob in ranked[:top_n]]

# -----------------------------
# ROUTES
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    input_text = ""      # what the user typed (shown again in the text box)
    uploaded_name = ""   # file name of the uploaded resume, shown in the results bar
    found_skills = ""    # skills found in the uploaded resume
    top_careers = []     # [{"career": ..., "percent": ...}, ...]
    job_results = []     # live jobs; None means the jobs API failed
    skill_gap = None     # result of analyze_skill_gap for the top career
    resume_score = None  # result of score_resume (only for uploaded resumes)
    roadmap = None       # result of build_roadmap for the top career
    why_terms = []       # words in the text that pushed the top career the most
    courses = []         # free resources for each missing skill
    simulator_data = None  # what the What-If Simulator needs to send back to /api/simulate
    error = ""
    source = ""          # "resume" or "text", so the page can say what we predicted from

    if request.method == "POST":
        file = request.files.get("resume_file")
        input_text = request.form.get("input_text", "").strip()
        text_to_predict = ""

        if model is None:
            error = MODEL_MISSING_MESSAGE

        # 0) "Try with a sample resume": analyze the built-in sample like an uploaded resume
        elif request.form.get("sample") == "1":
            with open(SAMPLE_RESUME_PATH, encoding="utf-8") as f:
                text_to_predict = f.read()
            found_skills = extract_skills_from_text(text_to_predict)
            source = "resume"
            uploaded_name = SAMPLE_RESUME_NAME

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
                        uploaded_name = file.filename
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
            why_terms = top_contributing_terms(model, text_to_predict, top_careers[0]["career"])
            roadmap = build_roadmap(top_careers[0]["career"], text_to_predict)
            courses = recommend_courses(skill_gap["missing"])
            simulator_data = {"text": text_to_predict, "career": top_careers[0]["career"],
                              "is_resume": source == "resume"}
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
        jobs_unavailable=job_results is None,
        skill_gap=skill_gap,
        resume_score=resume_score,
        roadmap=roadmap,
        why_terms=why_terms,
        courses=courses,
        simulator_data=simulator_data,
        error=error,
        source=source,
        uploaded_name=uploaded_name
    )

@app.route("/health")
def health():
    # Render calls this to check the app is alive. Keep it tiny: no model, no API calls.
    return "ok", 200


def report_data(text, is_resume, source_label):
    """Everything the PDF report shows, computed with the same functions as the page."""
    top_careers = predict_top_careers(text)
    career = top_careers[0]["career"]
    skill_gap = analyze_skill_gap(career, text)
    return {
        "source_label": source_label,
        "top_careers": top_careers,
        "why_terms": top_contributing_terms(model, text, career),
        "resume_score": score_resume(text, career) if is_resume else None,
        "skill_gap": skill_gap,
        "roadmap": build_roadmap(career, text),
        "courses": recommend_courses(skill_gap["missing"]),
    }


@app.route("/report", methods=["POST"])
def download_report():
    """'Download report (PDF)': the results bar posts the analyzed text here."""
    if model is None:
        return MODEL_MISSING_MESSAGE, 503
    text = request.form.get("text", "")
    if not text.strip() or len(text) > MAX_SIMULATE_TEXT:
        return "Missing or too long text.", 400
    is_resume = request.form.get("is_resume") == "1"
    name = request.form.get("name", "").strip()[:120]
    source_label = f"Resume: {name}" if is_resume and name else ("Resume" if is_resume else "Typed skills")
    pdf = build_report_pdf(report_data(text, is_resume, source_label))
    return Response(pdf, mimetype="application/pdf",
                    headers={"Content-Disposition": 'attachment; filename="careerpilot-report.pdf"'})


@app.route("/api/simulate", methods=["POST"])
def api_simulate():
    """What-If Simulator. Input JSON: {"text", "career", "skills": [...], "is_resume"}."""
    if model is None:
        return jsonify({"error": MODEL_MISSING_MESSAGE}), 503
    data = request.get_json(silent=True) or {}
    text = data.get("text")
    career = data.get("career")
    skills = data.get("skills")

    # Check the input: only accept skills from this career's key skill list
    if not isinstance(text, str) or not text.strip() or len(text) > MAX_SIMULATE_TEXT:
        return jsonify({"error": "Missing or too long text."}), 400
    if career not in CAREER_SKILLS:
        return jsonify({"error": "Unknown career."}), 400
    if not isinstance(skills, list) or not skills or any(s not in CAREER_SKILLS[career] for s in skills):
        return jsonify({"error": "Pick at least one skill from the list."}), 400

    result = simulate(text, career, skills, bool(data.get("is_resume")), predict_top_careers)
    return jsonify(result)

# -----------------------------
# APP START
# -----------------------------
if __name__ == "__main__":
    # Use host=0.0.0.0 for Render or any cloud
    # Debug mode is OFF unless you set FLASK_DEBUG=1 (never turn it on in production)
    debug_mode = os.environ.get("FLASK_DEBUG") == "1"
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 10000)), debug=debug_mode)

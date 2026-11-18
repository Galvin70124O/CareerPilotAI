from flask import Flask, Response, jsonify, request, render_template
import joblib
import logging
import os
import sklearn
from werkzeug.utils import secure_filename
from docx import Document
from pypdf import PdfReader

from features import rate_limit
from features.careers import display_name
from features.compare import compare_versions, snapshot
from features.courses import recommend_courses
from features.ranking import PROFILES, fresher_roles, rank_careers
from features.readiness import RULE, resume_insights
from features.report import build_report_pdf
from features.resume_check import NOT_A_RESUME, looks_like_resume
from features.jobs import fetch_job_listings
from features.interview import interview_questions
from features.jd_match import MAX_JD_LENGTH, match_job
from features.resume_score import score_resume
from features.roadmap import build_roadmap
from features.simulator import simulate
from features.skills import CAREER_SKILLS, analyze_skill_gap, display_skill, extract_skills_from_text

# -----------------------------
# CONFIGURATION
# -----------------------------
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'}
MAX_FILE_BYTES = 2 * 1024 * 1024      # 2MB per uploaded resume, checked in read_upload
# A whole request may be a bit bigger: "Compare with another version" sends the analyzed text with the file
MAX_CONTENT_LENGTH = MAX_FILE_BYTES + 512 * 1024
UPLOAD_RULE = "Only .txt, .pdf, and .docx files up to 2MB allowed."
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
# The real number of careers we rank (42 classifier careers + the new ones), for page text
app.jinja_env.globals["career_count"] = len(PROFILES)
app.jinja_env.globals["jd_max"] = MAX_JD_LENGTH

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


def read_upload(file_storage):
    """(resume text, "") for a good upload, or ("", error message). The home page and
    'Compare with another version' both use it, so both files get the same limits and checks."""
    if not allowed_file(file_storage.filename):
        return "", UPLOAD_RULE
    file_storage.seek(0, os.SEEK_END)
    too_big = file_storage.tell() > MAX_FILE_BYTES
    file_storage.seek(0)
    if too_big:
        return "", UPLOAD_RULE
    try:
        text = extract_text_from_resume(file_storage)
    except Exception:   # a damaged or fake PDF/DOCX: log it, don't show the library's message
        logger.warning("Could not read an uploaded %s file", os.path.splitext(file_storage.filename)[1], exc_info=True)
        return "", "Could not open this file. Check it is a real PDF, DOCX or TXT file."
    if not text:
        return "", "Could not read any text from this file (is it a scanned image?)."
    if not looks_like_resume(text):     # an invoice or a letter: say so instead of a report built on nothing
        return "", NOT_A_RESUME
    return text, ""


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
    # The classifier's top 3 with bands and matched skills, see features/ranking.py
    return rank_careers(model, input_text, top_n)

# -----------------------------
# ROUTES
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    input_text = ""      # what the user typed (shown again in the text box)
    uploaded_name = ""   # file name of the uploaded resume, shown in the results bar
    found_skills = ""    # skills found in the uploaded resume
    top_careers = []     # [{"career": ..., "percent": ...}, ...]
    fresher = []         # "Fresher roles that fit you" (new careers, features/ranking.py)
    insights = None      # evidence tags, readiness, low-info warning (uploaded resumes only)
    job_results = []     # live jobs; None means the jobs API failed
    skill_gap = None     # result of analyze_skill_gap for the top career
    resume_score = None  # result of score_resume (only for uploaded resumes)
    roadmap = None       # result of build_roadmap for the top career
    why_terms = []       # words in the text that pushed the top career the most
    courses = []         # free resources for each missing skill
    simulator_data = None  # what the What-If Simulator needs to send back to /api/simulate
    interview = []       # interview practice questions for the top career
    fresher_interview = []  # ... and for the first "Fresher role that fits you", if there is one
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
            resume_text, error = read_upload(file)
            if resume_text:
                found_skills = extract_skills_from_text(resume_text)
                # Predict from the FULL resume, the same kind of text the model was trained on
                text_to_predict = resume_text
                source = "resume"
                uploaded_name = file.filename

        # 2) No file: predict from the typed/spoken text
        elif input_text:
            text_to_predict = input_text
            source = "text"

        else:
            error = "Please upload a resume or type some skills first."

        if text_to_predict:
            top_careers = predict_top_careers(text_to_predict)
            fresher = fresher_roles(text_to_predict, top_careers)
            skill_gap = analyze_skill_gap(top_careers[0]["career"], text_to_predict)
            why_terms = top_careers[0]["why_terms"]
            roadmap = build_roadmap(top_careers[0]["career"], text_to_predict)
            courses = recommend_courses(skill_gap["missing"])
            interview = interview_questions(top_careers[0]["career"])
            fresher_interview = interview_questions(fresher[0]["career"]) if fresher else []
            simulator_data = {"text": text_to_predict, "career": top_careers[0]["career"],
                              "is_resume": source == "resume"}
            job_results = fetch_job_listings(top_careers[0]["career"])
            if source == "resume":
                # The score needs the RAW resume (line breaks, bullets, emails), not cleaned text
                resume_score = score_resume(text_to_predict, top_careers[0]["career"])
                insights = resume_insights(text_to_predict, top_careers, skill_gap)

    return render_template(
        "index.html",
        input_text=input_text,
        found_skills=found_skills,
        top_careers=top_careers,
        fresher_roles=fresher,
        insights=insights,
        readiness_rule=RULE,
        job_results=job_results,
        jobs_unavailable=job_results is None,
        skill_gap=skill_gap,
        resume_score=resume_score,
        roadmap=roadmap,
        why_terms=why_terms,
        courses=courses,
        simulator_data=simulator_data,
        interview=interview,
        fresher_interview=fresher_interview,
        error=error,
        source=source,
        uploaded_name=uploaded_name
    )

@app.route("/health")
def health():
    # Render calls this to check the app is alive. Keep it tiny: no model, no API calls.
    return "ok", 200


def jd_problem(jd):
    """The error message for a bad job description, or "" if it is fine."""
    if not isinstance(jd, str) or not jd.strip():
        return "Paste a job description first."
    if len(jd) > MAX_JD_LENGTH:
        return f"That job description is over {MAX_JD_LENGTH:,} characters. Paste just the role and requirements."
    return ""


def jd_match(text, jd, is_resume):
    return match_job(model.named_steps["tfidf"], text, jd, is_resume)


def comparison(text, top_before, new_text):
    """'Compare with another version' for two texts, both against the first version's top career."""
    career = top_before[0]["career"]
    return compare_versions(snapshot(model, text, top_before, career),
                            snapshot(model, new_text, predict_top_careers(new_text), career))


def report_data(text, is_resume, source_label, jd="", compare=None):
    """Everything the PDF report shows, computed with the same functions as the page.
    compare: {"text", "name"} of a second version the user compared on the page, or None."""
    top_careers = predict_top_careers(text)
    career = top_careers[0]["career"]
    skill_gap = analyze_skill_gap(career, text)
    fresher = fresher_roles(text, top_careers)
    return {
        "source_label": source_label,
        "top_careers": top_careers,
        "fresher_roles": fresher,
        "why_terms": top_careers[0]["why_terms"],
        "resume_score": score_resume(text, career) if is_resume else None,
        "skill_gap": skill_gap,
        "insights": resume_insights(text, top_careers, skill_gap) if is_resume else None,
        "roadmap": build_roadmap(career, text),
        "courses": recommend_courses(skill_gap["missing"]),
        "interview": interview_questions(career),
        "fresher_interview": interview_questions(fresher[0]["career"]) if fresher else [],
        "jd_match": jd_match(text, jd, is_resume) if jd.strip() else None,
        "compare": ({**comparison(text, top_careers, compare["text"]), "after_name": compare["name"]}
                    if compare and is_resume else None),
    }


@app.route("/report", methods=["POST"])
def download_report():
    """'Download report (PDF)': the results bar posts the analyzed text here."""
    if too_many_requests():     # building a PDF is the heaviest request, so it shares the /api limit
        return TOO_MANY, 429
    if model is None:
        return MODEL_MISSING_MESSAGE, 503
    text = request.form.get("text", "")
    if not text.strip() or len(text) > MAX_SIMULATE_TEXT:
        return "Missing or too long text.", 400
    is_resume = request.form.get("is_resume") == "1"
    name = request.form.get("name", "").strip()[:120]
    # Browsers send form line breaks as \r\n, so count them as one character like the page did
    jd = request.form.get("jd", "").replace("\r\n", "\n")
    if jd.strip() and jd_problem(jd):
        return jd_problem(jd), 400
    source_label = f"Resume: {name}" if is_resume and name else ("Resume" if is_resume else "Typed skills")
    # Filled by app.js after "Compare with another version"; the comparison is worked out again here
    compare_text = request.form.get("compare_text", "")
    if len(compare_text) > MAX_SIMULATE_TEXT:
        return "Missing or too long text.", 400
    compare = ({"text": compare_text, "name": request.form.get("compare_name", "").strip()[:120] or "New version"}
               if compare_text.strip() else None)
    pdf = build_report_pdf(report_data(text, is_resume, source_label, jd, compare))
    return Response(pdf, mimetype="application/pdf",
                    headers={"Content-Disposition": 'attachment; filename="careerpilot-report.pdf"'})


def too_many_requests():
    """True if this visitor is over the /api rate limit (features/rate_limit.py)."""
    # On Render, Cloudflare puts the visitor's real IP in True-Client-IP; remote_addr would be the proxy
    ip = request.headers.get("True-Client-IP") or request.remote_addr
    return not rate_limit.allow(ip)


TOO_MANY = "Too many requests. Wait a minute and try again."


@app.route("/api/simulate", methods=["POST"])
def api_simulate():
    """What-If Simulator. Input JSON: {"text", "career", "skills": [...], "is_resume"}."""
    if too_many_requests():
        return jsonify({"error": TOO_MANY}), 429
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


@app.route("/api/jd-match", methods=["POST"])
def api_jd_match():
    """Job description match. Input JSON: {"text", "is_resume", "jd"}. Nothing is stored or logged."""
    if too_many_requests():
        return jsonify({"error": TOO_MANY}), 429
    if model is None:
        return jsonify({"error": MODEL_MISSING_MESSAGE}), 503
    data = request.get_json(silent=True) or {}
    text = data.get("text")
    jd = data.get("jd")
    if not isinstance(text, str) or not text.strip() or len(text) > MAX_SIMULATE_TEXT:
        return jsonify({"error": "Missing or too long text."}), 400
    if jd_problem(jd):
        return jsonify({"error": jd_problem(jd)}), 400
    return jsonify(jd_match(text, jd, bool(data.get("is_resume"))))


@app.route("/api/compare", methods=["POST"])
def api_compare():
    """'Compare with another version'. Form: "text" (the analyzed resume) + "resume_file" (the new version).
    Both go through the same analysis as the page, against the first version's top career. Nothing is stored."""
    if too_many_requests():
        return jsonify({"error": TOO_MANY}), 429
    if model is None:
        return jsonify({"error": MODEL_MISSING_MESSAGE}), 503
    text = request.form.get("text", "")
    if not text.strip() or len(text) > MAX_SIMULATE_TEXT:
        return jsonify({"error": "Missing or too long text."}), 400
    file = request.files.get("resume_file")
    if not file or not file.filename:
        return jsonify({"error": "Choose the other version of your resume first."}), 400
    new_text, error = read_upload(file)
    if error:
        return jsonify({"error": error}), 400
    # after_text goes into the report form so the PDF can include this comparison.
    # shortcut: a version over the /report text limit is silently left out of the PDF, add a note if people hit it
    after_text = new_text if len(new_text) <= MAX_SIMULATE_TEXT else ""
    return jsonify({**comparison(text, predict_top_careers(text), new_text), "after_text": after_text})


@app.errorhandler(413)
def too_large(error):
    """A request over MAX_CONTENT_LENGTH (the page's own 2 MB check normally stops this first)."""
    if request.path.startswith("/api/"):
        return jsonify({"error": UPLOAD_RULE}), 413
    return UPLOAD_RULE, 413

# -----------------------------
# APP START
# -----------------------------
if __name__ == "__main__":
    # Use host=0.0.0.0 for Render or any cloud
    # Debug mode is OFF unless you set FLASK_DEBUG=1 (never turn it on in production)
    debug_mode = os.environ.get("FLASK_DEBUG") == "1"
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 10000)), debug=debug_mode)

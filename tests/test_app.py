import io

from docx import Document

import app


def post(data, monkeypatch):
    # Don't call the real Remotive API during tests
    monkeypatch.setattr(app, "fetch_job_listings", lambda title, limit=5: [])
    client = app.app.test_client()
    response = client.post("/", data=data, content_type="multipart/form-data")
    return response.get_data(as_text=True)


RESUME = "Skills: Python, pandas, scikit-learn, machine learning, deep learning, SQL. Built NLP models."


def test_typed_skills_give_top_3(monkeypatch):
    html = post({"input_text": "python machine learning pandas"}, monkeypatch)
    assert "Recommended Career Path" in html
    assert html.count('class="cp-bar mb-2"') == 3
    assert "Skill gap for" in html


def test_txt_upload_extracts_and_predicts(monkeypatch):
    html = post({"resume_file": (io.BytesIO(RESUME.encode()), "cv.txt")}, monkeypatch)
    assert "Skills found in your resume" in html
    assert "machine learning" in html
    assert "Data Science" in html


def test_docx_upload_extracts_and_predicts(monkeypatch):
    doc = Document()
    doc.add_paragraph(RESUME)
    buffer = io.BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    html = post({"resume_file": (buffer, "cv.docx")}, monkeypatch)
    assert "Skills found in your resume" in html
    assert "Recommended Career Path" in html


def test_wrong_file_type_shows_error(monkeypatch):
    html = post({"resume_file": (io.BytesIO(b"hi"), "cv.exe")}, monkeypatch)
    assert "Only .txt, .pdf, and .docx" in html
    assert "Recommended Career Path" not in html


def test_empty_form_shows_error(monkeypatch):
    html = post({"input_text": ""}, monkeypatch)
    assert "Please upload a resume or type some skills" in html


def test_uploaded_resume_gets_a_score(monkeypatch):
    with open("tests/fixtures/good_resume.txt", "rb") as f:
        html = post({"resume_file": (io.BytesIO(f.read()), "cv.txt")}, monkeypatch)
    assert 'aria-label="Resume score' in html
    assert "Tips to improve" in html


def test_typed_skills_get_no_resume_score(monkeypatch):
    html = post({"input_text": "python sql"}, monkeypatch)
    assert 'aria-label="Resume score' not in html

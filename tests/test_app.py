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


def test_roadmap_shown_for_typed_skills(monkeypatch):
    html = post({"input_text": "java spring hibernate"}, monkeypatch)
    assert "Roadmap to Java Developer" in html


def test_courses_shown_for_missing_skills(monkeypatch):
    html = post({"input_text": "java"}, monkeypatch)
    assert "Free resources for your missing skills" in html
    assert "spring.io/guides" in html


def simulate_api(payload, monkeypatch):
    monkeypatch.setattr(app, "fetch_job_listings", lambda title, limit=5: [])
    return app.app.test_client().post("/api/simulate", json=payload)


def test_simulator_shown_with_missing_skills(monkeypatch):
    html = post({"input_text": "java"}, monkeypatch)
    assert "What if you learned" in html
    assert 'id="cp-sim-data"' in html


def test_simulate_api_before_after(monkeypatch):
    response = simulate_api({"text": "java", "career": "Java Developer",
                             "skills": ["spring", "hibernate"], "is_resume": False}, monkeypatch)
    assert response.status_code == 200
    result = response.get_json()
    assert result["after"]["skill_gap_percent"] > result["before"]["skill_gap_percent"]
    assert result["before"]["resume_score"] is None
    assert len(result["after"]["top_careers"]) == 3


def test_simulate_api_rejects_bad_input(monkeypatch):
    assert simulate_api({"text": "java", "career": "Astronaut", "skills": ["x"]}, monkeypatch).status_code == 400
    assert simulate_api({"text": "java", "career": "Java Developer", "skills": ["not-a-key-skill"]}, monkeypatch).status_code == 400
    assert simulate_api({"text": "", "career": "Java Developer", "skills": ["spring"]}, monkeypatch).status_code == 400
    assert simulate_api({"career": "Java Developer"}, monkeypatch).status_code == 400


def test_simulator_data_is_safely_escaped(monkeypatch):
    # A resume containing </script> must not break out of the JSON <script> block
    html = post({"resume_file": (io.BytesIO(b"Python dev </script><b>x</b> " * 5), "cv.txt")}, monkeypatch)
    data_block = html.split('id="cp-sim-data">')[1].split("</script>")[0]
    assert "<b>" not in data_block

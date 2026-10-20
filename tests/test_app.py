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


def test_pdf_upload_extracts_and_predicts(monkeypatch):
    with open("tests/fixtures/resume.pdf", "rb") as f:
        html = post({"resume_file": (io.BytesIO(f.read()), "cv.pdf")}, monkeypatch)
    assert "Skills found in your resume" in html
    assert "machine learning" in html
    assert 'aria-label="Resume score' in html


def test_model_records_its_scikit_learn_version():
    # model.pkl is committed and loaded on Render; it must say which scikit-learn trained it
    import sklearn
    assert app.model.trained_with_sklearn == sklearn.__version__


ADVOCATE_TEXT = "victim advocate, crisis intervention, court accompaniment, survivors of domestic violence"


def test_advocate_shown_with_display_name(monkeypatch):
    html = post({"input_text": ADVOCATE_TEXT}, monkeypatch)
    assert "Recommended Career Path:&nbsp;</b> Advocate (Legal &amp; Social Services)" in html
    assert "Skill gap for Advocate (Legal &amp; Social Services)" in html
    assert "Roadmap to Advocate (Legal &amp; Social Services)" in html


def test_jobs_are_searched_with_the_model_label(monkeypatch):
    searched = []
    monkeypatch.setattr(app, "fetch_job_listings", lambda title, limit=5: searched.append(title) or [])
    app.app.test_client().post("/", data={"input_text": ADVOCATE_TEXT})
    assert searched == ["Advocate"]


def test_simulate_api_returns_display_names(monkeypatch):
    result = simulate_api({"text": ADVOCATE_TEXT, "career": "Advocate", "skills": ["counseling"]}, monkeypatch).get_json()
    assert result["career_label"] == "Advocate (Legal & Social Services)"
    assert all("label" in c for c in result["after"]["top_careers"])


def test_health_route():
    response = app.app.test_client().get("/health")
    assert response.status_code == 200
    assert response.get_data(as_text=True) == "ok"


def test_render_yaml_uses_health_route_and_no_training():
    with open("render.yaml", encoding="utf-8") as f:
        render = f.read()
    assert "healthCheckPath: /health" in render
    assert "buildCommand: pip install -r requirements.txt\n" in render
    assert "train.py" not in render.split("buildCommand:")[1].split("\n")[0]
    assert "gunicorn app:app --bind 0.0.0.0:$PORT" in render
    assert "value: 3.12.6" in render


def page_with_jobs(monkeypatch, fake_result):
    monkeypatch.setattr(app, "fetch_job_listings", lambda title, limit=5: fake_result)
    return app.app.test_client().post("/", data={"input_text": "java spring"}).get_data(as_text=True)


def test_jobs_api_failure_shows_unavailable_message(monkeypatch):
    html = page_with_jobs(monkeypatch, None)
    assert "Live jobs are unavailable right now" in html
    assert "Recommended Career Path" in html          # the rest of the page still works


def test_jobs_are_listed_when_api_works(monkeypatch):
    html = page_with_jobs(monkeypatch, [{"title": "Java Engineer", "company": "Acme",
                                         "url": "https://example.com/j", "location": "Europe"}])
    assert "Java Engineer" in html and "unavailable" not in html


def test_no_jobs_found_message(monkeypatch):
    html = page_with_jobs(monkeypatch, [])
    assert "No remote jobs found for Java Developer right now" in html

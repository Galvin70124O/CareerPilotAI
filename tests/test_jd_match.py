"""features/jd_match.py, POST /api/jd-match, and the job description section of the PDF."""
import io

from pypdf import PdfReader

import app
from features.jd_match import MAX_JD_LENGTH, match_job

VECTORIZER = app.model.named_steps["tfidf"]
RESUME = ("Asha Rao\nSKILLS\nSQL, Excel\n\nPROJECTS\n"
          "- Built a library app in React with search queries and monthly reports\n"
          "- Cleaned sales data in Python and plotted weekly trends\n")


def match(resume, jd, is_resume=True):
    return match_job(VECTORIZER, resume, jd, is_resume)


def test_the_same_text_is_a_perfect_match():
    text = "Python developer. Builds REST APIs in Flask with SQL and Docker."
    result = match(text, text)
    assert (result["percent"], result["words_percent"], result["skills_percent"]) == (100, 100, 100)


def test_no_shared_words_or_skills_is_zero():
    result = match("Kiran\nSkills: Excel, accounting, Tally", "Kubernetes and Docker engineers wanted")
    assert (result["percent"], result["words_percent"], result["skills_percent"]) == (0, 0, 0)
    assert result["gaps"] == ["Kubernetes", "Docker"]     # in the order the ad names them


def test_reactjs_in_the_jd_matches_react_in_the_resume():
    result = match(RESUME, "Build screens in ReactJS.")
    assert "React" in result["found"]
    assert "React" not in result["gaps"]


def test_each_jd_skill_shows_once_in_the_order_the_ad_names_it():
    # RESUME lists SQL only under Skills (not proven) and uses React in a project (proven)
    result = match(RESUME, "Docker, Kubernetes and AWS on the job.\nSQL and React a plus. Docker again.")
    assert result["gaps"] == ["Docker", "Kubernetes", "AWS"]     # character order, not A-Z
    assert result["unproven"] == ["SQL"]
    assert result["found"] == ["React"]
    assert (result["skills_in_resume"], result["skills_total"]) == (2, 5)


def test_coursework_only_skills_are_not_proven_either():
    resume = "EDUCATION\nB.Tech CSE, coursework in Machine Learning\n\nPROJECTS\n- Built a quiz app in Python with a score board\n"
    result = match(resume, "Python and machine learning")
    assert result["unproven"] == ["Machine Learning"] and result["found"] == ["Python"]


def test_a_project_line_is_suggested_for_a_gap_and_named_honestly():
    result = match(RESUME, "Strong SQL for search queries and reports.")
    assert result["suggestions"] == [{"skill": "sql", "label": "SQL",
                                      "line": "Built a library app in React with search queries and monthly reports"}]


def test_typed_skills_get_no_evidence_tags_or_line_suggestions():
    result = match("python, sql", "Python, SQL and Docker", is_resume=False)
    assert result["gaps"] == ["Docker"] and result["found"] == ["Python", "SQL"]
    assert result["unproven"] == [] and result["suggestions"] == []


def test_a_jd_with_no_known_skills_uses_word_overlap_only():
    result = match(RESUME, "Nous recherchons un stagiaire motivé pour notre équipe à Paris.")
    assert result["skills_percent"] is None and result["percent"] == result["words_percent"]   # the page shows a note, not this
    assert result["gaps"] == []


def test_boilerplate_words_are_not_skills():
    result = match(RESUME, "Our organization offers benefits and state of the art training.")
    assert result["skills_total"] == 0


# --- the API and the PDF --------------------------------------------------------------

def post(jd, text=RESUME):
    return app.app.test_client().post("/api/jd-match", json={"text": text, "is_resume": True, "jd": jd})


def test_api_returns_the_match():
    response = post("Build screens in ReactJS and write SQL.")
    assert response.status_code == 200 and "React" in response.get_json()["found"]


def test_api_refuses_an_empty_or_oversized_jd():
    for jd in ["", "   \n ", None]:
        response = post(jd)
        assert response.status_code == 400 and "Paste a job description" in response.get_json()["error"]
    assert post("a" * (MAX_JD_LENGTH + 1)).status_code == 400
    assert post("a" * MAX_JD_LENGTH).status_code == 200


def test_api_refuses_missing_resume_text():
    assert post("SQL", text="").status_code == 400


def test_the_page_has_the_jd_card_for_resumes_and_typed_skills(monkeypatch):
    monkeypatch.setattr(app, "fetch_job_listings", lambda title, limit=5: [])
    client = app.app.test_client()
    typed = client.post("/", data={"input_text": "python, sql"}).get_data(as_text=True)
    resume = client.post("/", data={"resume_file": (io.BytesIO(RESUME.encode()), "cv.txt")},
                         content_type="multipart/form-data").get_data(as_text=True)
    for html in [typed, resume]:
        card = html.split('data-card="jd"')[1]
        assert f'data-max="{MAX_JD_LENGTH}"' in card and "0 / 8,000 characters" in card
        assert 'name="jd" value="" data-report-jd' in html     # the PDF form's empty JD field


def pdf_text(data):
    response = app.app.test_client().post("/report", data=data)
    assert response.status_code == 200
    return "\n".join(page.extract_text() for page in PdfReader(io.BytesIO(response.data)).pages)


def test_the_pdf_has_the_jd_section_only_when_a_jd_was_checked():
    base = {"text": RESUME, "is_resume": "1", "name": "cv.txt"}
    text = pdf_text({**base, "jd": "Strong SQL and Docker for search queries and reports."})
    assert "Job description match" in text and "Not yet proven in a project: SQL" in text
    assert "Job description match" not in pdf_text(base)


def test_the_pdf_counts_form_line_breaks_as_one_character():
    # browsers send a textarea's line breaks as \r\n: 8,000 characters on the page must still fit
    jd = ("SQL and Docker\n" * 600)[:MAX_JD_LENGTH].replace("\n", "\r\n")
    assert len(jd) > MAX_JD_LENGTH
    assert "Job description match" in pdf_text({"text": RESUME, "is_resume": "1", "jd": jd})

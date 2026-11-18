"""features/readiness.py: verdict, next step, low-information warning; and the page / PDF / top 3."""
import io

import numpy as np
from pypdf import PdfReader

import app
from features.evidence import CLAIMED, COURSEWORK, PROVEN
from features.readiness import CLOSE_GAP, NOT_READY, ONE_PROJECT, READY, low_information, readiness

# Data Science key skills: python, machine learning, sql, r, visualization, data analysis, statistics, tableau
DS = {"career": "Data Science", "label": "Data Science", "band": "moderate"}


def with_band(band):
    return {**DS, "band": band}


def test_ready_to_apply_needs_a_good_match_and_three_proven_skills():
    result = readiness(DS, {"python": PROVEN, "sql": PROVEN, "tableau": PROVEN, "statistics": CLAIMED})
    assert result["level"] == READY
    assert (result["proven"], result["have"], result["total"]) == (3, 4, 8)
    assert result["next_step"] is None


def test_one_project_away_names_the_skills_to_prove_coursework_first():
    result = readiness(with_band("strong"), {"python": PROVEN, "sql": CLAIMED, "statistics": COURSEWORK})
    assert result["level"] == ONE_PROJECT
    assert result["next_step"] == "Use Statistics (coursework) and SQL (claimed) in one project, so they are proven."


def test_one_project_away_with_one_skill_to_prove():
    result = readiness(DS, {"python": PROVEN, "sql": PROVEN, "tableau": CLAIMED})
    assert result["level"] == ONE_PROJECT
    assert result["next_step"] == "Use Tableau (claimed) in one project, so it is proven."


def test_not_ready_with_too_few_key_skills_says_what_to_learn():
    result = readiness(with_band("strong"), {"python": PROVEN, "sql": PROVEN})
    assert result["level"] == NOT_READY
    assert result["next_step"] == "Learn Machine Learning, so the resume shows 3 of the key skills for Data Science."


def test_not_ready_with_a_weak_match_even_with_proven_skills():
    result = readiness(with_band("weak"), {"python": PROVEN, "sql": PROVEN, "tableau": PROVEN, "r": PROVEN})
    assert result["level"] == NOT_READY
    assert result["next_step"] == ("Build one project that uses Python, SQL and R, then list it under Projects "
                                   "so the resume reads clearly as Data Science.")


def test_no_verdict_mentions_a_time_estimate():
    for band in ["strong", "weak"]:
        for skills in [{}, {"python": CLAIMED, "sql": CLAIMED, "r": CLAIMED}]:
            text = str(readiness(with_band(band), skills))
            assert "week" not in text and "month" not in text and "day" not in text


def careers(first, second):
    return [{"label": "Data Science", "percent": first}, {"label": "Python Developer", "percent": second},
            {"label": "Database", "percent": 3}]


def test_low_information_warning_when_the_top_two_are_close():
    warning = low_information(careers(34, 30), "SKILLS\nPython, SQL\n")
    assert warning["missing_sections"] == ["Projects", "Experience", "Education"]
    assert warning["message"] == ("Your top two matches are close (Data Science 34%, Python Developer 30%), so this "
                                  "resume doesn't separate them yet. Add Projects, Experience and Education sections "
                                  "with specific details.")


def test_low_information_warning_with_one_missing_section():
    text = "EDUCATION\nB.Sc\nSKILLS\nPython\nEXPERIENCE\nAnalyst\n"
    assert "Add a Projects section with specific details." in low_information(careers(40, 35), text)["message"]


def test_a_skills_label_line_counts_as_a_skills_section():
    text = "Kiran\nSkills: Excel, accounting, Tally, sales, customer service\n"
    assert low_information(careers(48, 46), text)["missing_sections"] == ["Projects", "Experience", "Education"]


def test_no_low_information_warning_when_the_top_two_are_far_apart():
    assert low_information(careers(60, 60 - CLOSE_GAP), "SKILLS\nPython\n") is None
    assert low_information(careers(70, 20), "SKILLS\nPython\n") is None


# --- the page, the PDF, and the top 3 ----------------------------------------------

FRESHER = open("tests/fixtures/good_resume.txt", encoding="utf-8").read()


def post(data, monkeypatch):
    monkeypatch.setattr(app, "fetch_job_listings", lambda title, limit=5: [])
    return app.app.test_client().post("/", data=data, content_type="multipart/form-data").get_data(as_text=True)


def test_an_uploaded_resume_shows_readiness_and_evidence_tags(monkeypatch):
    html = post({"resume_file": (io.BytesIO(FRESHER.encode()), "cv.txt")}, monkeypatch)
    gap = html.split('data-card="gap"')[1]
    assert "Readiness:" in gap and "How this is decided" in gap
    assert any(f'<span class="chip-evidence">{tag}</span>' in gap for tag in [PROVEN, COURSEWORK, CLAIMED])
    assert "week" not in gap.split("How this is decided")[0]


def test_typed_skills_show_no_readiness_tags_or_warning(monkeypatch):
    html = post({"input_text": "python, sql, statistics"}, monkeypatch)
    assert "Readiness:" not in html and "chip-evidence" not in html and 'class="note"' not in html


def test_the_top_3_is_still_the_classifier_alone():
    # evidence and readiness are display only: the top 3 is exactly the classifier's order
    probabilities = app.model.predict_proba([FRESHER])[0]
    classifier_top = [app.model.classes_[i] for i in np.argsort(probabilities)[::-1][:3]]
    assert [item["career"] for item in app.predict_top_careers(FRESHER)] == classifier_top


def test_the_pdf_has_readiness_and_tags_for_a_resume_only():
    client = app.app.test_client()
    resume = client.post("/report", data={"text": FRESHER, "is_resume": "1", "name": "cv.txt"})
    text = "\n".join(page.extract_text() for page in PdfReader(io.BytesIO(resume.data)).pages)
    assert "Readiness:" in text and any(f"({tag})" in text for tag in [PROVEN, COURSEWORK, CLAIMED])
    typed = client.post("/report", data={"text": "python, sql", "is_resume": "0"})
    assert "Readiness:" not in "\n".join(page.extract_text() for page in PdfReader(io.BytesIO(typed.data)).pages)

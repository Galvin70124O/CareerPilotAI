"""interview.json, features/interview.py, and the interview practice card and PDF section."""
import io
import re

from pypdf import PdfReader

import app
from features.interview import QUESTIONS, interview_questions
from features.ranking import PROFILES

PROFILE_SKILLS = {skill["name"] for profile in PROFILES.values() for skill in profile["key_skills"]}
# No made-up time estimates or statistics in the advice (the questions may hold numbers: aptitude)
TIME_OR_STAT = re.compile(r"\d+\s*(minute|hour|day|week|month|year)s?\b|%", re.I)


def test_every_career_has_ten_different_questions():
    assert set(QUESTIONS) == set(PROFILES)
    for career, questions in QUESTIONS.items():
        assert len(questions) == 10, career
        assert len({q["question"] for q in questions}) == 10, career


def test_every_tested_skill_exists_in_the_profiles():
    for career, questions in QUESTIONS.items():
        for q in questions:
            assert set(q) == {"question", "skill", "tip"}, career
            assert q["skill"] in PROFILE_SKILLS, (career, q["skill"])


def test_most_questions_test_the_careers_own_key_skills():
    for career, questions in QUESTIONS.items():
        own = {skill["name"] for skill in PROFILES[career]["key_skills"]}
        assert sum(q["skill"] in own for q in questions) >= 7, career


def test_tips_have_no_time_estimates_or_statistics():
    for career, questions in QUESTIONS.items():
        for q in questions:
            assert not TIME_OR_STAT.search(q["tip"]), (career, q["tip"])


def test_everything_prints_in_the_pdf_core_font():
    for questions in QUESTIONS.values():
        for q in questions:
            for text in q.values():
                text.encode("latin-1")     # fpdf2's core Helvetica is latin-1 only


def test_fresher_roles_get_dsa_and_aptitude():
    sde = interview_questions("Software Development Engineer (Fresher)")
    skills = [q["skill"] for q in sde]
    assert skills.count("data structures and algorithms") >= 3 and "problem solving" in skills
    assert interview_questions("Not a career") == []


def test_the_page_shows_the_top_careers_questions(monkeypatch):
    monkeypatch.setattr(app, "fetch_job_listings", lambda title, limit=5: [])
    with open("tests/fixtures/good_resume.txt", "rb") as f:
        html = app.app.test_client().post("/", data={"resume_file": (io.BytesIO(f.read()), "cv.txt")},
                                          content_type="multipart/form-data").get_data(as_text=True)
    card = html.split('data-card="interview"')[1].split("</article>")[0]
    assert "Interview practice for Data Science" in card
    assert card.count('class="interview-question"') == 10
    assert interview_questions("Data Science")[0]["question"] in card


def test_the_pdf_has_the_interview_section():
    response = app.app.test_client().post("/report", data={"text": "python, sql, pandas, machine learning", "is_resume": "0"})
    text = "\n".join(page.extract_text() for page in PdfReader(io.BytesIO(response.data)).pages)
    assert "Interview practice for" in text and "Skill:" in text


SDE_FRESHER = """Ravi Kumar
ravi.kumar@example.com | +91 90000 11111
Education
B.Tech Computer Science, 2026, 8.4 CGPA
Skills
Java, data structures and algorithms, problem solving, OOP, SQL, Git
Projects
Library management system in Java with SQL and Git. Solved 300 LeetCode problems on data structures and algorithms.
"""
SDE = "Software Development Engineer (Fresher)"


def test_the_first_fresher_role_gets_its_own_questions_on_the_page(monkeypatch):
    monkeypatch.setattr(app, "fetch_job_listings", lambda title, limit=5: [])
    html = app.app.test_client().post("/", data={"resume_file": (io.BytesIO(SDE_FRESHER.encode()), "cv.txt")},
                                      content_type="multipart/form-data").get_data(as_text=True)
    card = html.split('data-card="interview"')[1].split("</article>")[0]
    assert f"Interview practice for {SDE}" in card
    assert card.count('class="interview-question"') == 20      # top career + fresher role
    assert interview_questions(SDE)[0]["question"] in card


def test_no_fresher_heading_without_a_fresher_role(monkeypatch):
    monkeypatch.setattr(app, "fetch_job_listings", lambda title, limit=5: [])
    with open("tests/fixtures/good_resume.txt", "rb") as f:
        html = app.app.test_client().post("/", data={"resume_file": (io.BytesIO(f.read()), "cv.txt")},
                                          content_type="multipart/form-data").get_data(as_text=True)
    assert "interview-fresher" not in html


def test_the_pdf_has_the_fresher_roles_questions_too():
    response = app.app.test_client().post("/report", data={"text": SDE_FRESHER, "is_resume": "1"})
    text = "\n".join(page.extract_text() for page in PdfReader(io.BytesIO(response.data)).pages)
    assert f"Interview practice for {SDE}" in text
    assert interview_questions(SDE)[0]["question"][:40] in text.replace("\n", " ")

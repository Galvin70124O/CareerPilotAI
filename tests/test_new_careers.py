"""Every new career works end to end: ATS score, roadmap, What-If Simulator, PDF report, courses."""
import io
import json
import os

import pytest
from pypdf import PdfReader

import app
from features.careers import display_name
from features.courses import recommend_courses
from features.report import build_report_pdf
from features.resume_score import score_resume
from features.roadmap import build_roadmap
from features.simulator import simulate
from features.skills import CAREER_SKILLS, JOB_KEYWORDS, analyze_skill_gap

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(PROJECT_DIR, "career_profiles.json"), encoding="utf-8") as f:
    PROFILES = json.load(f)["careers"]
NEW_CAREERS = [career for career, profile in PROFILES.items() if profile["new"]]


def resume_for(career):
    """A short fictional resume that has this career's first two key skills and nothing else of note."""
    first, second = CAREER_SKILLS[career][:2]
    return ("Asha Verma\nasha.verma@example.com | +91 90000 00000\n\n"
            "Education\nBachelor's degree, 2025, 8.1 CGPA\n\n"
            f"Skills\n{first}, {second}\n\n"
            f"Projects\nFinal-year project where I used {first} and {second} every week.\n")


def test_there_are_seven_new_careers():
    assert len(NEW_CAREERS) == 7


@pytest.mark.parametrize("career", NEW_CAREERS)
def test_key_skills_are_the_top_8_of_the_profile_and_jobs_have_keywords(career):
    assert CAREER_SKILLS[career] == [skill["name"] for skill in PROFILES[career]["key_skills"][:8]]
    assert JOB_KEYWORDS[career]


@pytest.mark.parametrize("career", NEW_CAREERS)
def test_ats_score_works(career):
    result = score_resume(resume_for(career), career)
    assert 0 <= result["total"] <= 100
    assert any(display_name(career) in item["text"] for item in result["checklist"])


@pytest.mark.parametrize("career", NEW_CAREERS)
def test_roadmap_has_9_steps_and_ticks_what_the_resume_has(career):
    roadmap = build_roadmap(career, resume_for(career))
    assert roadmap["total"] == 9
    assert roadmap["done"] >= 2


@pytest.mark.parametrize("career", NEW_CAREERS)
def test_simulator_adding_missing_skills_raises_the_skill_gap(career):
    text = resume_for(career)
    missing = analyze_skill_gap(career, text)["missing"][:2]
    result = simulate(text, career, missing, True, app.predict_top_careers)
    assert result["after"]["skill_gap_percent"] > result["before"]["skill_gap_percent"]


@pytest.mark.parametrize("career", NEW_CAREERS)
def test_simulator_route_accepts_the_new_career(career):
    response = app.app.test_client().post("/api/simulate", json={
        "text": resume_for(career), "career": career, "skills": CAREER_SKILLS[career][2:4], "is_resume": True})
    assert response.status_code == 200
    assert response.get_json()["career_label"] == display_name(career)


@pytest.mark.parametrize("career", NEW_CAREERS)
def test_pdf_report_has_the_new_career(career):
    text = resume_for(career)
    skill_gap = analyze_skill_gap(career, text)
    report = {"source_label": "cv.txt",
              "top_careers": [{"career": career, "label": display_name(career), "percent": 60}],
              "why_terms": [], "resume_score": score_resume(text, career), "skill_gap": skill_gap,
              "roadmap": build_roadmap(career, text), "courses": recommend_courses(skill_gap["missing"])}
    pdf = bytes(build_report_pdf(report))
    assert pdf.startswith(b"%PDF")
    pdf_text = "\n".join(page.extract_text() for page in PdfReader(io.BytesIO(pdf)).pages)
    assert display_name(career) in " ".join(pdf_text.split())


@pytest.mark.parametrize("career", NEW_CAREERS)
def test_at_least_3_real_free_course_links(career):
    roadmap_skills = [step["skill"] for stage in build_roadmap(career, "")["stages"] for step in stage["steps"]]
    links = {resource["url"] for item in recommend_courses(roadmap_skills)
             for resource in item["resources"] if not resource["is_search"]}
    assert len(links) >= 3, links

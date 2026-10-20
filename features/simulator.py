"""
What-If Simulator: "what if I learned these skills?"

We add the chosen skills to the user's text, run everything again, and return
before/after numbers. The skill gap and resume score are always measured for the
SAME target career, so the before/after comparison is fair.
"""
from features.careers import display_name
from features.resume_score import score_resume
from features.skills import analyze_skill_gap, display_skill

# How a skill is written into the text. A bare "r" isn't counted as the R language
# (see has_skill in features/skills.py), so the simulator writes it out in full.
TEXT_FORMS = {"r": "R programming"}


def add_skills_to_text(text, skills):
    # A new "Skills:" line at the end, like a person would add to their resume
    written = [TEXT_FORMS.get(skill, skill) for skill in skills]
    return text.rstrip() + "\n\nSkills: " + ", ".join(written) + "\n"


def snapshot(text, career, is_resume, predict_top_careers):
    return {
        "top_careers": predict_top_careers(text),
        "skill_gap_percent": analyze_skill_gap(career, text)["percent"],
        "resume_score": score_resume(text, career)["total"] if is_resume else None,
    }


def simulate(text, career, added_skills, is_resume, predict_top_careers):
    """predict_top_careers is passed in (from app.py) so this module doesn't need the model."""
    new_text = add_skills_to_text(text, added_skills)
    return {
        "career": career,
        "career_label": display_name(career),
        "added_skills": added_skills,
        "added_skill_labels": [display_skill(skill) for skill in added_skills],
        "before": snapshot(text, career, is_resume, predict_top_careers),
        "after": snapshot(new_text, career, is_resume, predict_top_careers),
    }

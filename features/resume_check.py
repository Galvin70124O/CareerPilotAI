"""
"Does this upload look like a resume?" A quick check before the full analysis, so an invoice or a
letter gets a clear message instead of a career report built on nothing.

An upload is NOT a resume only when all three say so: fewer than MIN_SECTIONS resume sections
(Projects, Experience, Certifications, Skills, Education: the same heading test as the evidence tags),
fewer than MIN_SKILLS known skills, and fewer than MIN_WORDS words. So a CV with no headings still
passes on its skills or its length, and a short CV with two real sections passes too.
"""
from features.evidence import heading_of
from features.resume_score import lines_of
from features.skills import extract_skills_from_text

NOT_A_RESUME = "This doesn't look like a resume. Upload your CV as PDF, DOCX or TXT."
# Design choices, not measured from data
MIN_SECTIONS = 2
MIN_SKILLS = 2
MIN_WORDS = 80


def resume_sections(text):
    """The resume sections the text has headings (or labels like "Skills: ...") for."""
    # "Other" (Summary, Contact, Hobbies, ...) also appears in letters and forms, so it doesn't count
    return {heading_of(line) for line in lines_of(text)} - {None, "Other"}


def looks_like_resume(text):
    if len(resume_sections(text)) >= MIN_SECTIONS:
        return True
    found = extract_skills_from_text(text)          # a "python, sql" string, "" when none
    skill_count = len(found.split(", ")) if found else 0
    # Block only when skills AND length both say "not a resume"; either one alone is enough to pass
    return skill_count >= MIN_SKILLS or len(text.split()) >= MIN_WORDS

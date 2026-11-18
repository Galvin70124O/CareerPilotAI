"""
"Does this upload look like a resume?" A quick check before the full analysis, so an invoice or a
letter gets a clear message instead of a career report built on nothing.

An upload is NOT a resume when it has fewer than MIN_SECTIONS resume sections (Projects, Experience,
Certifications, Skills, Education: the same heading test as the evidence tags) AND either no known
skill or fewer than MIN_WORDS words. A short CV with two real sections still passes.
"""
from features.evidence import heading_of
from features.resume_score import lines_of
from features.skills import extract_skills_from_text

NOT_A_RESUME = "This doesn't look like a resume. Upload your CV as PDF, DOCX or TXT."
# Design choices, not measured from data
MIN_SECTIONS = 2
MIN_WORDS = 80


def resume_sections(text):
    """The resume sections the text has headings (or labels like "Skills: ...") for."""
    # "Other" (Summary, Contact, Hobbies, ...) also appears in letters and forms, so it doesn't count
    return {heading_of(line) for line in lines_of(text)} - {None, "Other"}


def looks_like_resume(text):
    if len(resume_sections(text)) >= MIN_SECTIONS:
        return True
    return bool(extract_skills_from_text(text)) and len(text.split()) >= MIN_WORDS

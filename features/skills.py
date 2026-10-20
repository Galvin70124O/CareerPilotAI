"""
Skill extraction and the Skill Gap Analyzer.

- extract_skills_from_text(): finds general skills (KNOWN_SKILLS) in a resume
- analyze_skill_gap():        compares a resume with the key skills of one career (skills.json)
"""
import json
import os
import re

from features.text_cleaning import normalize_text

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

KNOWN_SKILLS = [
    "python", "java", "c", "c++", "c#", "html", "css", "javascript", "typescript", "sql", "mysql", "oracle", "mongodb",
    "data science", "machine learning", "deep learning", "artificial intelligence", "computer vision", "nlp",
    "cloud computing", "azure", "aws", "google cloud", "docker", "kubernetes", "git", "github", "embedded systems",
    "arduino", "raspberry pi", "networking", "linux", "windows", "cybersecurity", "blockchain", "hardware design",
    "soldering", "microcontroller programming", "excel", "powerpoint", "autocad", "matlab", "labview", "testing",
    "automation", "robotics", "communication", "teamwork", "leadership", "problem solving", "critical thinking",
    "creative thinking", "adaptability", "time management", "collaboration", "negotiation", "presentation",
    "active listening", "decision making", "conflict resolution", "public speaking", "research", "self-motivation",
    "organization", "project management", "interpersonal skills"
]

# skills.json has, for each career:
#   "skills":       8 key skills, checked against what resumes in that category actually mention
#   "job_keywords": words a matching job TITLE contains (used to filter live jobs)
with open(os.path.join(PROJECT_DIR, "skills.json"), encoding="utf-8") as f:
    _careers = json.load(f)
CAREER_SKILLS = {career: info["skills"] for career, info in _careers.items()}
JOB_KEYWORDS = {career: info["job_keywords"] for career, info in _careers.items()}


def skill_pattern(skill):
    # re.escape makes "+", "#" and "." literal characters instead of regex symbols.
    # Spaces become \s+ so "machine learning" also matches "machine\nlearning".
    escaped = re.escape(skill).replace(r"\ ", r"\s+")
    # (?<!...) = "not preceded by", (?!...) = "not followed by".
    # A skill must not touch a letter, digit, "_", "+" or "#", so:
    #   "c" does not match inside "c++", "c#" or "science"
    #   "git" does not match inside "github"
    return re.compile(r"(?<![\w+#])" + escaped + r"(?![\w+#])")


# Build every regex once when the app starts, not on every request
SKILL_PATTERNS = {skill: skill_pattern(skill) for skill in KNOWN_SKILLS}


def extract_skills_from_text(text):
    skills_found = []
    text_lower = text.lower()
    for skill, pattern in SKILL_PATTERNS.items():
        if pattern.search(text_lower):
            skills_found.append(skill)
    return ", ".join(skills_found)


# Some skills are too short to match on their own. "r" alone would match "R&D", "Mr. R. Kumar"
# or "Grade: R", so we only count R when it clearly means the programming language.
DATA_TOOLS = r"(python|sql|sas|spss|stata|matlab|julia|scala|excel|tableau|power\s?bi|pandas|numpy|hadoop|spark)"
NOT_R_AND_D = r"(?!&|\w|\.\w)"   # stops "R&D", "Rated" and initials like "R.K."
SPECIAL_SKILL_PATTERNS = {
    "r": re.compile(
        r"\br\s*(programming|language|studio|shiny|markdown)\b"             # "R programming", "R Studio"
        r"|\b(rstudio|tidyverse|ggplot2?|dplyr|cran)\b"                     # R's own tools
        r"|\b" + DATA_TOOLS + r"\s*(,|/|&|\band\b)\s*r" + NOT_R_AND_D +    # "Python, R" or "SAS/R"
        r"|(?<![\w.&])r\s*(,|/|\band\b)\s*" + DATA_TOOLS + r"\b"            # "R, Python" or "R/SQL"
    ),
}


def has_skill(skill, text):
    """True if the skill is in the text, written either way: "node.js" or "nodejs", "ci/cd" or "cicd"."""
    if skill in SPECIAL_SKILL_PATTERNS:
        return SPECIAL_SKILL_PATTERNS[skill].search(text.lower()) is not None
    if skill_pattern(skill).search(text.lower()):
        return True
    # Also try the punctuation-free form (how the training data writes it).
    # Skip this for skills like "c#" whose clean form ("c") would mean something else.
    clean_skill = normalize_text(skill)
    if len(clean_skill) < 2:
        return False
    return skill_pattern(clean_skill).search(normalize_text(text)) is not None


def analyze_skill_gap(career, text):
    key_skills = CAREER_SKILLS.get(career, [])
    have = [skill for skill in key_skills if has_skill(skill, text)]
    missing = [skill for skill in key_skills if skill not in have]
    percent = round(len(have) / len(key_skills) * 100) if key_skills else 0
    return {"career": career, "have": have, "missing": missing, "total": len(key_skills), "percent": percent}

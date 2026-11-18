"""
Evidence strength: HOW a resume shows each skill, not just whether it mentions it.

    proven      the skill is in a Projects, Experience or Internship section,
                or on a real project / internship / GitHub line ("Project: Expense tracker in Flask")
    coursework  the skill is in an Education or Certifications section,
                or on a line that names a course (NPTEL, SWAYAM, Coursera, certification, coursework)
    claimed     anywhere else, e.g. only in the Skills list or the summary

A skill gets the strongest tag of all the lines it appears on (SQL in a project AND in the Skills
list is proven). Lines in the Skills section are always "claimed": the words "project" or
"internship" in a skill name ("Skills: Project Management, SQL") prove nothing.
Only for uploaded resumes: typed skills have no sections.
"""
import re

from features.resume_score import SECTION_HEADINGS, lines_of
from features.skills import has_skill

PROVEN, COURSEWORK, CLAIMED = "proven", "coursework", "claimed"
STRENGTH = {CLAIMED: 0, COURSEWORK: 1, PROVEN: 2}

# Checked in this order, so "Academic Projects" is Projects (not Education) and
# "Internships and Training" is Experience. "Other" headings end the section before them.
HEADINGS = [
    ("Projects", SECTION_HEADINGS["Projects"], PROVEN),
    ("Experience", SECTION_HEADINGS["Experience"], PROVEN),
    ("Certifications", r"certifications?|certificates?|courses?|coursework|trainings?|nptel|swayam", COURSEWORK),
    ("Skills", SECTION_HEADINGS["Skills"], CLAIMED),
    ("Education", SECTION_HEADINGS["Education"], COURSEWORK),
    ("Other", r"summary|objective|profile|about\s+me|achievements?|awards?|extra[\s-]?curricular|activities|"
              r"hobbies|interests|languages|personal|declaration|positions?\s+of\s+responsibility|"
              r"publications?|references?|contact|other", CLAIMED),
]

# A real project / internship line: starts with the label ("Final year project: ...", "Internship - ...")
# or links to GitHub. "Project Management" does not match: the label must be followed by : or -.
PROJECT_LINE = re.compile(r"^\W*((final[\s-]+year|mini|major|academic|personal|college)\s+)?"
                          r"(projects?|internships?)\s*[:\-–—]|github\.com", re.I)
COURSE_LINE = re.compile(r"\b(nptel|swayam|coursera|udemy|edx|certifi(ed|cate|cation)s?|coursework)\b", re.I)


HEADING_PREFIX = r"^\W*((key|core|relevant|other|additional|selected|major)\s+)?("
# After the heading word: nothing, a label ("Skills: Java, SQL"), or "and ..." ("Internships and Training").
# NOT another word, so a skill on its own line ("Project Management") is not a Projects heading.
HEADING_REST = r")\b(\s*$|\s*[:\-–—(&/|].*$|\s+and\s+.{0,30}$)"


def heading_of(line):
    """The section a heading or label line starts, or None if the line is neither."""
    for name, pattern, _ in HEADINGS:
        if re.match(HEADING_PREFIX + pattern + HEADING_REST, line, re.I):
            return name
    return None


def line_tags(text):
    """[(line, tag)] for every non-empty line of the resume."""
    section_tag = {name: tag for name, _, tag in HEADINGS}
    section = None
    tagged = []
    for line in lines_of(text):
        section = heading_of(line) or section
        if section == "Skills":
            tag = CLAIMED
        elif section_tag.get(section) == PROVEN or PROJECT_LINE.search(line):
            tag = PROVEN
        elif section_tag.get(section) == COURSEWORK or COURSE_LINE.search(line):
            tag = COURSEWORK
        else:
            tag = CLAIMED
        tagged.append((line, tag))
    return tagged


def evidence_tags(text, skills):
    """{skill: "proven" | "coursework" | "claimed"} for each of these skills found in the text."""
    tagged = line_tags(text)
    result = {}
    for skill in skills:
        if not has_skill(skill, text):
            continue
        tags = [tag for line, tag in tagged if has_skill(skill, line)]
        # a skill matched only across a line break (rare) still counts as found
        result[skill] = max(tags, key=STRENGTH.get) if tags else CLAIMED
    return result

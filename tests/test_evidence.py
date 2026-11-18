"""features/evidence.py: proven / coursework / claimed per skill, from the resume's sections."""
from features.evidence import CLAIMED, COURSEWORK, PROVEN, evidence_tags, heading_of


def tags(text, *skills):
    return evidence_tags(text, list(skills))


def test_skills_in_a_projects_section_are_proven():
    resume = "Asha Rao\n\nPROJECTS\nExpense tracker\n- Built with Python and Flask, data in SQL\n"
    assert tags(resume, "python", "sql") == {"python": PROVEN, "sql": PROVEN}


def test_skills_in_an_internship_or_experience_section_are_proven():
    resume = "INTERNSHIP\nIntern, Example Softech (2024)\n- Wrote unit tests in Java\n"
    assert tags(resume, "java") == {"java": PROVEN}
    resume = "Work Experience\nAnalyst, Example Bank\n- Built Tableau dashboards\n"
    assert tags(resume, "tableau") == {"tableau": PROVEN}


def test_a_real_project_or_github_line_is_proven_outside_a_projects_section():
    assert tags("EDUCATION\nB.Tech CSE\nFinal year project: crop prediction in Python\n", "python") == {"python": PROVEN}
    assert tags("Summary\nMy SQL work: github.com/asha-example/sql-queries\n", "sql") == {"sql": PROVEN}


def test_education_nptel_and_certifications_are_coursework():
    assert tags("EDUCATION\nB.Tech CSE\nCoursework: Data Structures, DBMS, SQL\n", "sql") == {"sql": COURSEWORK}
    assert tags("Summary\nCompleted NPTEL Python for Data Science\n", "python") == {"python": COURSEWORK}
    assert tags("CERTIFICATIONS\nGoogle Data Analytics - Tableau, SQL\n", "tableau") == {"tableau": COURSEWORK}


def test_a_skill_only_in_the_skills_list_is_claimed():
    assert tags("Ravi Kumar\n\nTECHNICAL SKILLS\nPython, SQL, Excel\n", "python", "excel") == {"python": CLAIMED, "excel": CLAIMED}


def test_project_management_in_a_skills_line_is_claimed_not_proven():
    # the words "project" / "internship" inside a skill name or a Skills line prove nothing
    assert tags("Skills: Project Management, SQL", "project management", "sql") == {
        "project management": CLAIMED, "sql": CLAIMED}


def test_a_skill_on_its_own_line_is_not_a_section_heading():
    assert heading_of("Project Management") is None
    assert heading_of("Internship Coordination") is None
    resume = "SKILLS\nProject Management\nSQL\n"
    assert tags(resume, "sql", "project management") == {"sql": CLAIMED, "project management": CLAIMED}


def test_headings_and_labels_start_sections():
    assert heading_of("PROJECTS") == "Projects"
    assert heading_of("Academic Projects") == "Projects"              # not Education
    assert heading_of("INTERNSHIPS AND TRAINING") == "Experience"
    assert heading_of("Key Skills") == "Skills"
    assert heading_of("Skills: Java, Python, C++, data structures and algorithms, OOP, SQL") == "Skills"
    assert heading_of("Achievements") == "Other"


def test_a_long_skills_label_line_after_projects_stays_claimed():
    resume = "PROJECTS\nChat app - Node.js\nSkills: Java, Python, C++, data structures and algorithms, OOP, SQL\n"
    assert tags(resume, "sql") == {"sql": CLAIMED}


def test_the_strongest_tag_wins():
    resume = ("SKILLS\nSQL, Python, Excel\n\nCERTIFICATIONS\nNPTEL: Python for Data Science\n\n"
              "PROJECTS\nSales dashboard - SQL queries feeding a report\n")
    assert tags(resume, "sql", "python", "excel") == {"sql": PROVEN, "python": COURSEWORK, "excel": CLAIMED}


def test_skills_not_in_the_resume_are_left_out():
    assert tags("SKILLS\nExcel\n", "excel", "python") == {"excel": CLAIMED}

import app
from features.skills import CAREER_SKILLS, analyze_skill_gap


def test_every_predictable_career_has_key_skills():
    # If the dataset gets a new category, skills.json must be updated too
    for career in app.model.classes_:
        assert career in CAREER_SKILLS, f"{career} is missing from skills.json"
        assert len(CAREER_SKILLS[career]) > 0


def test_have_and_missing_split():
    gap = analyze_skill_gap("Data Science", "I use Python, SQL and machine learning")
    assert gap["have"] == ["python", "machine learning", "sql"]
    assert "tableau" in gap["missing"]
    assert len(gap["have"]) + len(gap["missing"]) == gap["total"]


def test_percent():
    gap = analyze_skill_gap("DevOps", "docker kubernetes")
    assert gap["percent"] == round(2 / 8 * 100)


def test_unknown_career_is_safe():
    gap = analyze_skill_gap("Astronaut", "python")
    assert gap["total"] == 0 and gap["percent"] == 0


def test_skill_found_with_or_without_punctuation():
    # The training data writes "nodejs"; a user may write "Node.js" - both should count
    assert "node.js" in analyze_skill_gap("React Developer", "I know Node.js")["have"]
    assert "node.js" in analyze_skill_gap("React Developer", "I know nodejs")["have"]
    assert "ci/cd" in analyze_skill_gap("DevOps", "built CICD pipelines")["have"]


def test_c_sharp_not_found_from_plain_c():
    assert "c#" not in analyze_skill_gap("DotNet Developer", "I know C and C++")["have"]

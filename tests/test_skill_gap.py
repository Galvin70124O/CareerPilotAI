import app
from app import CAREER_SKILLS, analyze_skill_gap


def test_every_predictable_career_has_key_skills():
    # If a new category is added to Resume.csv, skills.json must be updated too
    for career in app.model.classes_:
        assert career in CAREER_SKILLS, f"{career} is missing from skills.json"
        assert len(CAREER_SKILLS[career]) > 0


def test_have_and_missing_split():
    gap = analyze_skill_gap("Data Science", "I use Python, SQL and machine learning")
    assert gap["have"] == ["python", "machine learning", "sql"]
    assert "deep learning" in gap["missing"]
    assert len(gap["have"]) + len(gap["missing"]) == gap["total"]


def test_percent():
    gap = analyze_skill_gap("Hadoop", "hadoop hive")
    assert gap["percent"] == round(2 / 8 * 100)


def test_unknown_career_is_safe():
    gap = analyze_skill_gap("Astronaut", "python")
    assert gap["total"] == 0 and gap["percent"] == 0

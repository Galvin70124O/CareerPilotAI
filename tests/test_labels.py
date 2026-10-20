import pandas as pd

from data.prepare_data import DROPPED_CATEGORIES, fix_labels
from features.careers import display_name
from features.skills import CAREER_SKILLS


def make_data(rows):
    return pd.DataFrame(rows, columns=["Category", "Resume"])


def test_science_teacher_without_data_tools_moves_to_education():
    data, moved = fix_labels(make_data([("Data Science", "middle school science teacher classroom lesson plans")]))
    assert data["Category"].tolist() == ["Education"]
    assert moved.tolist() == [True]


def test_real_data_people_are_never_moved():
    rows = [("Data Science", f"taught students then worked with {tool}") for tool in
            ["python", "sql", "statistics", "r programming", "tableau", "power bi", "data analysis", "spark"]]
    data, moved = fix_labels(make_data(rows))
    assert set(data["Category"]) == {"Data Science"}
    assert not moved.any()


def test_other_categories_are_not_relabelled():
    data, _ = fix_labels(make_data([("Arts", "art teacher classroom"), ("Education", "teacher students")]))
    assert data["Category"].tolist() == ["Arts", "Education"]


def test_management_is_dropped():
    data, moved = fix_labels(make_data([("Management", "general manager"), ("Sales", "sales targets")]))
    assert data["Category"].tolist() == ["Sales"]
    assert len(moved) == 1


def test_dropped_categories_have_no_skills_entry():
    for category in DROPPED_CATEGORIES:
        assert category not in CAREER_SKILLS


def test_display_name():
    assert display_name("Advocate") == "Advocate (Legal & Social Services)"
    assert display_name("Java Developer") == "Java Developer"

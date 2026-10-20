from features.text_cleaning import normalize_text


def test_lowercase_and_punctuation_deleted_like_dataset():
    assert normalize_text("ASP.NET, Node.js & CI/CD") == "aspnet nodejs cicd"


def test_commas_become_spaces():
    # "python,java" must stay two words, not "pythonjava"
    assert normalize_text("python,java;sql") == "python java sql"


def test_whitespace_and_bullets_collapsed():
    assert normalize_text("• Built\n\n  APIs\t") == "built apis"

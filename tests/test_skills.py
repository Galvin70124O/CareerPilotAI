from features.skills import extract_skills_from_text


def skills(text):
    # extract_skills_from_text returns "a, b, c"; turn it into a list for easy checks
    result = extract_skills_from_text(text)
    return result.split(", ") if result else []


def test_c_alone_is_found():
    assert "c" in skills("I know C and Python.")


def test_c_not_found_inside_other_words():
    assert "c" not in skills("Computer science and cloud computing")


def test_c_plus_plus_is_not_c():
    found = skills("Wrote games in C++")
    assert "c++" in found
    assert "c" not in found


def test_c_sharp_is_not_c():
    found = skills("Backend work in C#")
    assert "c#" in found
    assert "c" not in found


def test_c_and_c_plus_plus_together():
    found = skills("Languages: C/C++")
    assert "c" in found
    assert "c++" in found


def test_git_not_found_inside_github():
    found = skills("My projects are on GitHub")
    assert "github" in found
    assert "git" not in found


def test_git_alone_is_found():
    assert "git" in skills("Version control with Git.")


def test_testing_not_found_inside_pentesting():
    assert "testing" not in skills("Did pentesting for a bank")


def test_testing_alone_is_found():
    assert "testing" in skills("Manual testing, automation")


def test_java_not_found_inside_javascript():
    found = skills("Frontend in JavaScript")
    assert "javascript" in found
    assert "java" not in found


def test_multi_word_skill_across_line_break():
    assert "machine learning" in skills("Machine\nLearning projects")


def test_no_skills_returns_empty():
    assert skills("I like cooking.") == []

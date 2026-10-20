import os

from features.resume_score import SCORE_WEIGHTS, has_phone, score_resume

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def read(name):
    with open(os.path.join(FIXTURES, name), encoding="utf-8") as f:
        return f.read()


def checklist_ok(result, starts_with):
    return next(item["ok"] for item in result["checklist"] if item["text"].startswith(starts_with))


def test_weights_add_up_to_100():
    assert sum(SCORE_WEIGHTS.values()) == 100


def test_breakdown_adds_up_to_about_the_total():
    result = score_resume(read("good_resume.txt"), "Data Science")
    assert abs(sum(c["score"] for c in result["categories"]) - result["total"]) <= 3
    assert all(0 <= c["score"] <= c["max"] for c in result["categories"])


def test_good_resume_scores_high():
    assert score_resume(read("good_resume.txt"), "Data Science")["total"] >= 80


def test_weak_resume_scores_low():
    assert score_resume(read("weak_resume.txt"), "Web Designing")["total"] <= 30


def test_sections_and_links_found():
    result = score_resume(read("good_resume.txt"), "Data Science")
    for name in ["Education", "Skills", "Projects", "Experience", "Contact info", "GitHub/LinkedIn"]:
        assert checklist_ok(result, name), name


def test_missing_email_and_phone_are_red_flags():
    result = score_resume("Education\nB.Tech\nSkills\nPython", "Python Developer")
    assert not checklist_ok(result, "Email")
    assert not checklist_ok(result, "Phone")


def test_phone_needs_10_digits_not_a_date_range():
    assert has_phone("+91 98765 43210")
    assert not has_phone("2021 - 2025")


def test_tips_quote_the_resume():
    tips = " ".join(score_resume(read("weak_resume.txt"), "Web Designing")["tips"])
    assert "Responsible for maintaining" in tips   # weak bullet start is quoted


def test_three_to_five_tips():
    for name in ["good_resume.txt", "weak_resume.txt"]:
        assert 3 <= len(score_resume(read(name), "Data Science")["tips"]) <= 5


def test_too_much_i_is_flagged():
    result = score_resume(read("weak_resume.txt"), "Web Designing")
    assert not checklist_ok(result, 'Not overusing "I')

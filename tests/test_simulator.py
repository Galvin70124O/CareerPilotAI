from features.simulator import add_skills_to_text, simulate


def fake_predict(text):
    # A tiny stand-in for the real model: more "docker" -> more DevOps
    percent = 90 if "docker" in text.lower() else 40
    return [{"career": "DevOps", "percent": percent}, {"career": "Testing", "percent": 100 - percent}]


def test_skills_are_added_as_a_line():
    assert add_skills_to_text("I know Linux.", ["docker", "git"]).endswith("Skills: docker, git\n")


def test_before_and_after_change():
    result = simulate("I know linux and aws", "DevOps", ["docker", "kubernetes"], False, fake_predict)
    assert result["before"]["top_careers"][0]["percent"] == 40
    assert result["after"]["top_careers"][0]["percent"] == 90
    assert result["after"]["skill_gap_percent"] > result["before"]["skill_gap_percent"]
    assert result["before"]["resume_score"] is None     # typed text: no resume score


def test_resume_score_included_for_resumes():
    result = simulate("Education\nB.Tech\nSkills\nlinux", "DevOps", ["docker"], True, fake_predict)
    assert result["after"]["resume_score"] >= result["before"]["resume_score"]

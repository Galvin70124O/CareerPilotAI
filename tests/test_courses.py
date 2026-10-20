from urllib.parse import urlparse

from features.courses import COURSES, MAX_PER_SKILL, recommend_courses


def test_known_skill_gets_real_resources():
    result = recommend_courses(["python"])[0]
    assert result["skill"] == "python"
    assert 1 <= len(result["resources"]) <= MAX_PER_SKILL
    assert not result["resources"][0]["is_search"]
    assert result["resources"][0]["url"].startswith("https://")


def test_unknown_skill_gets_search_link():
    result = recommend_courses(["underwater basket weaving"])[0]
    assert result["resources"][0]["is_search"]
    assert "youtube.com/results?search_query=underwater+basket+weaving" in result["resources"][0]["url"].lower()


def test_order_is_kept():
    assert [r["skill"] for r in recommend_courses(["sql", "zzz", "git"])] == ["sql", "zzz", "git"]


def test_all_links_are_https_with_a_title():
    for skill, items in COURSES.items():
        assert items, skill
        for item in items:
            assert urlparse(item["url"]).scheme == "https", item
            assert item["title"].strip(), item


def test_every_course_skill_is_used_by_some_career():
    # A course for a skill no career uses could never be shown
    from features.roadmap import ROADMAPS
    from features.skills import CAREER_SKILLS
    used = {s for skills in CAREER_SKILLS.values() for s in skills}
    used |= {step["skill"] for roadmap in ROADMAPS.values() for stage in roadmap.values() for step in stage}
    assert set(COURSES) <= used, set(COURSES) - used

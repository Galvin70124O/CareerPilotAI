from features.roadmap import ROADMAPS, STAGES, build_roadmap
from features.skills import CAREER_SKILLS


def test_every_career_has_3_stages_with_steps():
    assert set(ROADMAPS) == set(CAREER_SKILLS)
    for career, roadmap in ROADMAPS.items():
        assert list(roadmap) == STAGES, career
        for stage in STAGES:
            assert len(roadmap[stage]) >= 3, (career, stage)


def test_roadmap_covers_all_key_skills():
    # Every skill in the skill gap should also appear somewhere in the roadmap
    for career, skills in CAREER_SKILLS.items():
        roadmap_skills = {step["skill"] for stage in ROADMAPS[career].values() for step in stage}
        assert set(skills) <= roadmap_skills, career


def test_steps_the_user_covers_are_marked_done():
    roadmap = build_roadmap("Python Developer", "I know Python and Git")
    beginner = {step["skill"]: step["done"] for step in roadmap["stages"][0]["steps"]}
    assert beginner == {"python": True, "git": True, "linux": False}


def test_missing_steps_come_first_in_each_stage():
    roadmap = build_roadmap("Python Developer", "I know Python and Git")
    beginner = roadmap["stages"][0]["steps"]
    assert beginner[0]["skill"] == "linux"      # not done -> first
    assert [s["done"] for s in beginner] == [False, True, True]


def test_unknown_career_returns_none():
    assert build_roadmap("Astronaut", "python") is None

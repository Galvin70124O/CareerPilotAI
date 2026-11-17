"""career_profiles.json: weighted skills for every career, built by data/build_profiles.py."""
import json
import os

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEW_CAREERS = [
    "Software Development Engineer (Fresher)", "Data Analyst", "QA/Test Engineer",
    "Technical Support Engineer", "Graduate Engineer Trainee (Mechanical)",
    "Graduate Engineer Trainee (Electrical)", "Graduate Engineer Trainee (Civil)",
]


def load(name):
    with open(os.path.join(PROJECT_DIR, name), encoding="utf-8") as f:
        return json.load(f)


PROFILES = load("career_profiles.json")["careers"]
OLD_SKILLS = load("skills.json")


def test_every_classifier_career_and_every_new_career_has_a_profile():
    old_careers = set(OLD_SKILLS) - set(NEW_CAREERS)
    assert set(PROFILES) == old_careers | set(NEW_CAREERS)
    assert len(old_careers) == 42


def test_new_careers_grow_out_of_existing_classifier_careers():
    for career in NEW_CAREERS:
        profile = PROFILES[career]
        assert profile["new"] is True
        assert profile["parents"], career
        for parent in profile["parents"]:
            assert parent in PROFILES and PROFILES[parent]["new"] is False


def test_weights_are_positive_sorted_and_explained():
    for career, profile in PROFILES.items():
        weights = [skill["weight"] for skill in profile["key_skills"]]
        assert all(weight > 0 for weight in weights), career
        assert weights == sorted(weights, reverse=True), career
        assert all(skill["evidence"] for skill in profile["key_skills"]), career


def test_existing_profiles_weight_the_same_key_skills_the_app_shows():
    for career, info in OLD_SKILLS.items():
        if career in NEW_CAREERS:
            continue
        assert {skill["name"] for skill in PROFILES[career]["key_skills"]} == set(info["skills"])


def test_new_career_evidence_comes_from_esco_onet_or_the_india_list():
    for career in NEW_CAREERS:
        for skill in PROFILES[career]["key_skills"]:
            assert set(skill["evidence"]) <= {"esco", "onet", "india"}, (career, skill["name"])

"""features/ranking.py: the classifier's top 3 with bands, and "Fresher roles that fit you"."""
import numpy as np

import app
from features import ranking

SDE = "Software Development Engineer (Fresher)"


class StubModel:
    """Looks like the classifier: 42 classes, Java Developer, Testing and Sales most likely."""
    classes_ = np.array(sorted(c for c, p in ranking.PROFILES.items() if not p["new"]))

    def predict_proba(self, texts):
        probabilities = np.full(len(self.classes_), 0.002)
        for career, probability in [("Java Developer", 0.55), ("Testing", 0.25), ("Sales", 0.12)]:
            probabilities[list(self.classes_).index(career)] = probability
        return np.array([probabilities])


def top(*careers):
    return [{"career": career} for career in careers]


def test_the_top_3_is_the_classifiers_own_order_and_percent(monkeypatch):
    monkeypatch.setattr(ranking, "top_contributing_terms", lambda model, text, career: [])
    result = ranking.rank_careers(StubModel(), "Skills: Java, Selenium, SDLC, DSA")
    assert [item["career"] for item in result] == ["Java Developer", "Testing", "Sales"]
    assert [item["percent"] for item in result] == [55, 25, 12]
    assert [item["band"] for item in result] == ["moderate", "weak", "weak"]
    assert not any(ranking.PROFILES[item["career"]]["new"] for item in result)


def test_skill_score_is_the_share_of_the_matched_weight():
    assert ranking.skill_score(SDE, "") == 0
    all_skills = ", ".join(skill["name"] for skill in ranking.PROFILES[SDE]["key_skills"])
    assert ranking.skill_score(SDE, "Skills: " + all_skills + ", internship, B.Tech") == 1
    assert 0 < ranking.skill_score(SDE, "Skills: Java, Git") < 1


def test_a_fresher_role_needs_its_parent_in_the_top_3():
    qa_text = "Skills: " + ", ".join(skill["name"] for skill in ranking.PROFILES["QA/Test Engineer"]["key_skills"])
    assert ranking.fresher_roles(qa_text, top("Sales", "Banking", "Finance")) == []
    roles = ranking.fresher_roles(qa_text, top("Sales", "Testing", "Banking"))
    assert [role["career"] for role in roles] == ["QA/Test Engineer"]
    assert set(roles[0]) == {"career", "label", "matched_because"}       # no percent, no band
    assert len(roles[0]["matched_because"]) == ranking.MATCHED_BECAUSE


def test_a_fresher_role_must_beat_its_parents_skill_score(monkeypatch):
    scores = {"QA/Test Engineer": 0.4, "Testing": 0.4}
    monkeypatch.setattr(ranking, "skill_score", lambda career, text: scores.get(career, 0.0))
    assert ranking.fresher_roles("", top("Testing", "Sales", "Banking")) == []      # a tie is not a win
    scores["QA/Test Engineer"] = 0.41
    assert [r["career"] for r in ranking.fresher_roles("", top("Testing", "Sales", "Banking"))] == ["QA/Test Engineer"]


def test_at_most_two_fresher_roles_highest_skill_score_first(monkeypatch):
    scores = {SDE: 0.5, "QA/Test Engineer": 0.7, "Data Analyst": 0.6}
    monkeypatch.setattr(ranking, "skill_score", lambda career, text: scores.get(career, 0.0))
    roles = ranking.fresher_roles("", top("Java Developer", "Testing", "Data Science"))
    assert [role["career"] for role in roles] == ["QA/Test Engineer", "Data Analyst"]


def test_bands():
    strong, moderate = ranking.BANDS[0][0], ranking.BANDS[1][0]
    assert ranking.band(100) == ranking.band(strong) == "strong"
    assert ranking.band(strong - 1) == ranking.band(moderate) == "moderate"
    assert ranking.band(moderate - 1) == ranking.band(0) == "weak"


def test_real_ranking_has_everything_the_page_needs():
    text = "B.Tech CSE fresher. Skills: Java, Python, C++, Git, DSA, debugging, SQL. Internship at Infosys."
    top_careers = app.predict_top_careers(text)
    assert len(top_careers) == 3
    for item in top_careers:
        assert set(item) == {"career", "label", "percent", "band", "matched_because", "why_terms"}
        assert item["band"] in {"strong", "moderate", "weak"}
        assert not ranking.PROFILES[item["career"]]["new"]
    for role in ranking.fresher_roles(text, top_careers):
        assert ranking.PROFILES[role["career"]]["new"]
        assert role["matched_because"]


def test_the_page_shows_fresher_roles_below_the_top_3():
    text = "B.Tech CSE fresher. Skills: Java, Python, C++, Git, DSA, debugging, data structures, OOP. Internship."
    html = app.app.test_client().post("/", data={"input_text": text}).get_data(as_text=True)
    assert html.index("career-list") < html.index("Fresher roles that fit you")
    fresher = html.split("Fresher roles that fit you")[1]
    assert SDE in fresher and "Matched because: Git, Java, Python" in fresher
    assert "career-band" not in fresher.split("</ul>")[0]       # bands only on the main list


def test_no_fresher_roles_line_when_none_fit():
    html = app.app.test_client().post("/", data={"input_text": "Sales executive, cold calling, CRM, targets"})
    assert "Fresher roles that fit you" not in html.get_data(as_text=True)

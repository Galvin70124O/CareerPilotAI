"""
Career ranking: the classifier's top 3, plus "Fresher roles that fit you" from the weighted skill profiles.

1. Top 3: the classifier's own top 3 of the 42 ResumeAtlas careers it was trained on, with its own
   probability as the percent (model.pkl is a CalibratedClassifierCV, so it is already calibrated),
   and a strong / moderate / weak band next to it.
2. Fresher roles: the 7 new careers have no training resumes, so the classifier can't rank them.
   A new career is suggested (at most FRESHER_ROLES) when one of its "parents" in career_profiles.json
   (SDE (Fresher) <- Java / Python / DotNet / React Developer) is in the top 3 and the new career's
   skill score beats that parent's. Skill score = the share of the career's total key-skill weight
   the text matches (a share, so a career with 8 key skills and one with 12 compare fairly).

Mixing the two into one list cost 6.6 points of top-1 on the ResumeAtlas held-out test, so they stay apart.
"""
import json
import os

from features.careers import display_name
from features.explain import top_contributing_terms
from features.skills import display_skill, has_skill

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(PROJECT_DIR, "career_profiles.json"), encoding="utf-8") as f:
    PROFILES = json.load(f)["careers"]
PARENTS = {career: profile["parents"] for career, profile in PROFILES.items()}
NEW_CAREERS = [career for career, profile in PROFILES.items() if profile["new"]]

BANDS = [(60, "strong"), (30, "moderate"), (0, "weak")]   # percent at or above -> band
MATCHED_BECAUSE = 3               # skills shown in "Matched because"
FRESHER_ROLES = 2                 # at most this many fresher roles


def matched_skills(career, text):
    """The career's key skills found in the text, highest weight first."""
    return [skill for skill in PROFILES[career]["key_skills"] if has_skill(skill["name"], text)]


def matched_because(career, text):
    return [display_skill(skill["name"]) for skill in matched_skills(career, text)[:MATCHED_BECAUSE]]


def skill_score(career, text):
    total = sum(skill["weight"] for skill in PROFILES[career]["key_skills"])
    return sum(skill["weight"] for skill in matched_skills(career, text)) / total


def band(percent):
    return next(name for threshold, name in BANDS if percent >= threshold)


def rank_careers(model, text, top_n=3):
    """Classifier top careers as [{"career", "label", "percent", "band", "matched_because", "why_terms"}]."""
    probabilities = model.predict_proba([text])[0]
    ranked = sorted(zip(model.classes_, probabilities), key=lambda pair: pair[1], reverse=True)
    results = []
    for career, probability in ranked[:top_n]:
        percent = round(probability * 100)
        results.append({
            "career": career,                      # model / profile label, used for lookups and job search
            "label": display_name(career),         # what the user sees
            "percent": percent,
            "band": band(percent),
            "matched_because": matched_because(career, text),
            "why_terms": top_contributing_terms(model, text, career),
        })
    return results


def fresher_roles(text, top_careers, limit=FRESHER_ROLES):
    """New careers whose parent is in top_careers and whose skill score beats that parent's,
    highest skill score first, as [{"career", "label", "matched_because"}]."""
    top = [item["career"] for item in top_careers]
    candidates = []
    for child in NEW_CAREERS:
        parent = next((career for career in top if career in PARENTS[child]), None)   # best-ranked parent
        score = skill_score(child, text)
        if parent and score > skill_score(parent, text):
            candidates.append((score, child))
    candidates.sort(key=lambda pair: pair[0], reverse=True)
    return [{"career": child, "label": display_name(child), "matched_because": matched_because(child, text)}
            for _, child in candidates[:limit]]

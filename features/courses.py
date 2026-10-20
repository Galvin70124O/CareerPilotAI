"""
Course Recommender: 1-2 FREE learning resources for each missing skill.

courses.json only contains links that were checked (HTTP 200). We never invent links:
if a skill has no entry, we show a YouTube search link instead.
"""
import json
import os
from urllib.parse import quote_plus

from features.skills import display_skill

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(PROJECT_DIR, "courses.json"), encoding="utf-8") as f:
    COURSES = json.load(f)["courses"]

MAX_PER_SKILL = 2


def search_link(skill):
    name = display_skill(skill)
    return {"title": f"Search free {name} courses on YouTube",
            "url": "https://www.youtube.com/results?search_query=" + quote_plus(f"{name} free course"),
            "is_search": True}


def recommend_courses(missing_skills):
    """Return [{"skill": ..., "resources": [{"title", "url", "is_search"}]}] in the same order as missing_skills."""
    recommendations = []
    for skill in missing_skills:
        resources = [dict(item, is_search=False) for item in COURSES.get(skill, [])[:MAX_PER_SKILL]]
        recommendations.append({"skill": skill, "resources": resources or [search_link(skill)]})
    return recommendations

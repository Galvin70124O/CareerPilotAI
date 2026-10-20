"""
Career Roadmap: a 3-stage path (Beginner -> Intermediate -> Job-ready) for one career.

roadmaps.json has 3 steps per stage. Each step names a "skill" keyword; if that skill is
in the user's text, the step is marked as done. Within each stage, steps still to do come first.
"""
import json
import os

from features.skills import has_skill

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(PROJECT_DIR, "roadmaps.json"), encoding="utf-8") as f:
    ROADMAPS = json.load(f)

STAGES = ["Beginner", "Intermediate", "Job-ready"]


def build_roadmap(career, text):
    """Return {"career", "stages": [{"name", "steps": [{"step", "skill", "done"}]}], "done", "total"} or None."""
    roadmap = ROADMAPS.get(career)
    if roadmap is None:
        return None
    stages = []
    for stage_name in STAGES:
        steps = [dict(step, done=has_skill(step["skill"], text)) for step in roadmap[stage_name]]
        # sorted() is "stable": False (not done) comes before True, and the original order stays otherwise
        steps = sorted(steps, key=lambda step: step["done"])
        stages.append({"name": stage_name, "steps": steps})
    all_steps = [step for stage in stages for step in stage["steps"]]
    return {"career": career, "stages": stages,
            "done": sum(step["done"] for step in all_steps), "total": len(all_steps)}
